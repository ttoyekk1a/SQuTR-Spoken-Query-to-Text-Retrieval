import time
import librosa
import numpy as np
from tqdm import tqdm
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

# Import FunASR
from funasr import AutoModel

# Import shared utilities
from asr_utils import (
    setup_logger, compute_metric, load_data, get_audio_paths, 
    get_normalizers, write_results, print_summary
)


def load_audio_file(args):
    """Loads audio file with a target sample rate (usually 16k)."""
    path, sr = args
    try:
        y, _ = librosa.load(path, sr=sr)
        return y
    except Exception:
        return None
class FunASRBatchInference:
    def __init__(self, model_dir, device="cuda:0"):
        print(f"Initializing FunASR model from: {model_dir} on {device}")
        
        # Initialize FunASR AutoModel
        # Use trust_remote_code and remote_code if local model files are required
        self.model = AutoModel(
            model=model_dir,
            trust_remote_code=False,
            remote_code="./model.py",
            device=device,
        )
        print("FunASR model loaded!")

    def process_batch(self, audio_arrays, language="中文"):
        """
        Processes a batch of audio arrays and returns a list of recognized texts.
        """
        # Filter out failed audio loads
        valid_indices = [i for i, audio in enumerate(audio_arrays) if audio is not None]
        valid_audios = [audio_arrays[i] for i in valid_indices]

        if not valid_audios:
            return [""] * len(audio_arrays)

        try:
            # FunASR Inference supports list of numpy arrays
            res = self.model.generate(
                input=valid_audios,
                batch_size=len(valid_audios),
                language=language,
                use_itn=False, # ITN is handled externally by compute_metric
            )
            
            # Map valid results back to original indices
            valid_texts = [item['text'] for item in res]
            
            final_texts = [""] * len(audio_arrays)
            for idx, text in zip(valid_indices, valid_texts):
                final_texts[idx] = text
            
            return final_texts

        except Exception as e:
            print(f"Error during FunASR inference: {e}")
            return [""] * len(audio_arrays)


# ================= Main Process =================
def main():
    parser = argparse.ArgumentParser(description="Run FunASR batch inference.")

    # Path arguments
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing jsonl and audios/")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save output jsonl file")
    parser.add_argument("--input_json_path", type=str, default="queries_with_audio.jsonl", help="Path to input jsonl file")
    parser.add_argument("--audio_base_path", type=str, default="audio", help="Base path for audio files")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, 
                        default="FunAudioLLM/Fun-ASR-Nano-2512",
                        help="Path to FunASR model ID or local path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cuda:0, cpu)")
    parser.add_argument("--language", type=str, default="中文", help="Target language for ASR")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=16, help="CPU threads for audio loading")
    parser.add_argument("--metric", type=str, default="cer", choices=["wer", "cer"], help="Metric to calculate")

    args = parser.parse_args()

    input_json_path = os.path.join(args.input_folder, args.input_json_path)
    audio_base_path = os.path.join(args.input_folder, args.audio_base_path)
    log_file = args.output_json.replace(".jsonl", ".log")
    
    logger = setup_logger(log_file)
    logger.info(f"Task Started. Input: {input_json_path}")
    print(f"Logging to: {log_file}")

    # 1. Read Data
    samples = load_data(input_json_path)
                    

    logger.info(f"Total samples loaded: {len(samples)}")

    # 2. Initialize FunASR Model
    asr_model = FunASRBatchInference(model_dir=args.model_path, device=args.device)

    # 3. Batch Processing and Inference
    total_metric_score = 0
    valid_count = 0
    global_start_time = time.time()

    out_dir = os.path.dirname(args.output_json)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    thread_pool = ThreadPoolExecutor(max_workers=args.num_workers)
    en_tn_model, zh_tn_model, inflect_parser = get_normalizers()

    with open(args.output_json, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(samples), args.batch_size), desc="Inferencing"):
            batch_samples = samples[i : i + args.batch_size]

            # --- Parallel Audio Loading ---
            audio_load_tasks = []
            for sample in batch_samples:
                try:
                    audio_filename = sample["audio"]
                except KeyError:
                    audio_filename = sample["tts"]
                full_audio_path = os.path.join(audio_base_path, audio_filename)
                # FunASR optimized for 16k
                audio_load_tasks.append((full_audio_path, 16000))

            audio_arrays = list(thread_pool.map(load_audio_file, audio_load_tasks))

            # --- Batch Inference ---
            try:
                batch_results = asr_model.process_batch(audio_arrays, language=args.language)
            except Exception as e:
                logger.error(f"Batch inference error at index {i}: {e}")
                batch_results = [""] * len(batch_samples)

            # --- Write results and accumulate metrics ---
            for j, sample in enumerate(batch_samples):
                asr_text = batch_results[j]
                ground_truth = sample.get("text", "")

                score = compute_metric(ground_truth, asr_text, args.metric, en_tn_model, zh_tn_model, inflect_parser)
                total_metric_score += score
                valid_count += 1

                write_results(f_out, sample, asr_text, score, args.metric)

            f_out.flush()

    thread_pool.shutdown()

    global_end_time = time.time()
    total_duration_ms = (global_end_time - global_start_time) * 1000

    if valid_count > 0:
        avg_metric_score = total_metric_score / valid_count
        summary_msg = print_summary(valid_count, total_duration_ms, avg_metric_score, args.metric, 
                                   args.output_json, "FunASR")
        logger.info(summary_msg)


if __name__ == "__main__":
    main()