import time
import librosa
import numpy as np
from tqdm import tqdm
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# ModelScope related imports
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Import shared utilities
from asr_utils import (
    setup_logger, compute_metric, load_data, get_audio_paths,
    get_normalizers, write_results, print_summary
)


def load_audio_file(args):
    """Load audio file using librosa."""
    path, sr = args
    try:
        y, _ = librosa.load(path, sr=sr)
        return y
    except Exception:
        return None


# ================= Core Class (Paraformer) =================
class ParaformerBatchASR:
    def __init__(self, model_path):
        print(f"Initializing Paraformer ASR model from: {model_path}")
        self.inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=model_path
        )
        print("Paraformer model loaded!")

    def process_batch(self, audio_arrays):
        """
        Process a batch of audio arrays and return a list of transcription results.
        Note: Paraformer pipeline handles resample and dtype internally.
        """
        results = []
        for audio in audio_arrays:
            if audio is None:
                results.append("")
            else:
                try:
                    rec_result = self.inference_pipeline(audio)[0]
                    text = rec_result['text']
                    results.append(text)
                except Exception as e:
                    print(f"Error during ASR inference: {e}")
                    results.append("")
        return results


# ================= Main Execution Flow =================
def main():
    parser = argparse.ArgumentParser(description="Run Paraformer ASR batch inference.")

    # Path arguments
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing queries_with_audio.jsonl and audios/")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save output jsonl file")
    parser.add_argument("--input_json_path", type=str, default="queries_with_audio.jsonl", help="Path to input jsonl file")
    parser.add_argument("--audio_base_path", type=str, default="audio", help="Base path for audio files")
    
    # Model path for Paraformer
    parser.add_argument("--model_path", type=str, 
                        default="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                        help="Path to Paraformer model")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for IO batching")
    parser.add_argument("--num_workers", type=int, default=32, help="CPU threads for audio loading")
    parser.add_argument("--metric", type=str, default="cer", choices=["wer", "cer"], help="Metric to calculate")

    args = parser.parse_args()

    input_json_path = os.path.join(args.input_folder, args.input_json_path)
    audio_base_path = os.path.join(args.input_folder, args.audio_base_path)
    log_file = args.output_json.replace(".jsonl", ".log")
    
    logger = setup_logger(log_file)
    logger.info(f"Task Started. Input: {input_json_path}")

    # 1. Data loading
    samples = load_data(input_json_path)

    logger.info(f"Total samples loaded: {len(samples)}")

    # 2. Initialize Paraformer Model
    asr_model = ParaformerBatchASR(model_path=args.model_path)

    # 3. Batch processing and inference
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

            # --- Concurrent audio loading ---
            audio_load_tasks = []
            for sample in batch_samples:
                audio_filename = sample.get("audio") or sample.get("tts")
                full_audio_path = os.path.join(audio_base_path, audio_filename)
                audio_load_tasks.append((full_audio_path, 16000)) # Paraformer expects 16k

            audio_arrays = list(thread_pool.map(load_audio_file, audio_load_tasks))

            # --- Batch inference ---
            try:
                batch_results = asr_model.process_batch(audio_arrays)
            except Exception as e:
                logger.error(f"Batch inference error at index {i}: {e}")
                batch_results = [""] * len(batch_samples)

            # --- Write results and calculate metrics ---
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
                                   args.output_json, "Paraformer")
        logger.info(summary_msg)


if __name__ == "__main__":
    main()