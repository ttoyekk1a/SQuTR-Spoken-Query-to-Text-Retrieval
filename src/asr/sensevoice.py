import time
import librosa
import numpy as np
from tqdm import tqdm
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

# ModelScope related imports
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Import shared utilities
from asr_utils import (
    setup_logger, compute_metric, load_data, get_audio_paths,
    get_normalizers, write_results, print_summary
)


def load_audio_file(args):
    path, sr = args
    try:
        # SenseVoice also supports 16k sampling rate; keeping 16k for compatibility
        y, _ = librosa.load(path, sr=sr)
        return y
    except Exception as e:
        # print(e)
        return None


# ================= Core Class (SenseVoice ASR) =================
class SenseVoiceBatchASR:
    def __init__(self, model_path, device="cuda:0"):
        print(f"Initializing SenseVoice ASR model from: {model_path} on {device}")
        self.inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=model_path,
            model_revision="master",
            device=device  # Pass device parameter to pipeline
        )
        print("SenseVoice model loaded!")

    def process_batch(self, audio_arrays, language="auto"):
        """
        Process a batch of audio arrays and return recognition results
        """
        results = []
        for audio in audio_arrays:
            if audio is None:
                results.append("")
            else:
                try:
                    # SenseVoice inference
                    # Input can be numpy array; pipeline handles internal processing
                    # language="auto" is default; can specify "zh" for Chinese-only content
                    rec_result = self.inference_pipeline(input=audio, language=language, use_itn=False)
                    
                    # ModelScope pipeline may return list or dict
                    if isinstance(rec_result, list):
                        rec_result = rec_result[0]
                    
                    # Extract text from SenseVoice result (contains 'text' field)
                    text = rec_result.get('text', '')
                    text = text.split(">")[-1]
                    results.append(text)
                except Exception as e:
                    print(f"Error during ASR inference: {e}")
                    results.append("")
        return results


# ================= Main Process =================
def main():
    parser = argparse.ArgumentParser(description="Run SenseVoice ASR batch inference.")

    # Path arguments
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing queries_with_audio.jsonl and audios/")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save output jsonl file")
    parser.add_argument("--input_json_path", type=str, default="queries_with_audio.jsonl", help="Path to input jsonl file")
    parser.add_argument("--audio_base_path", type=str, default="audio", help="Base path for audio files")
    
    # Model path set to SenseVoiceSmall default ID
    parser.add_argument("--model_path", type=str, 
                        default="iic/SenseVoiceSmall",
                        help="Path to ModelScope model ID or local path")
    
    # Device argument
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run model on (e.g., cuda:0, cpu)")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (for IO batching)")
    parser.add_argument("--num_workers", type=int, default=16, help="CPU threads for audio loading")
    parser.add_argument("--metric", type=str, default="cer", choices=["wer", "cer"], help="Metric to calculate")
    parser.add_argument("--language", type=str, default="auto", choices=["auto", "zn", "en"], help="Language for ASR")
    args = parser.parse_args()

    input_json_path = os.path.join(args.input_folder, args.input_json_path)
    audio_base_path = os.path.join(args.input_folder, args.audio_base_path)
    log_file = args.output_json.replace(".jsonl", ".log")
    
    logger = setup_logger(log_file)
    logger.info(f"Task Started. Input: {input_json_path}")

    # 1. Load data
    samples = load_data(input_json_path)
                    

    logger.info(f"Total samples loaded: {len(samples)}")

    # 2. Initialize SenseVoice model
    asr_model = SenseVoiceBatchASR(model_path=args.model_path, device=args.device)

    # 3. Batch inference
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
                try:
                    audio_filename = sample["audio"]
                except:
                    audio_filename = sample["tts"]
                full_audio_path = os.path.join(audio_base_path, audio_filename)
                # SenseVoice recommends 16k sampling; preprocessing at 16k is more reliable
                # although the model internally supports resampling
                audio_load_tasks.append((full_audio_path, 16000))

            audio_arrays = list(thread_pool.map(load_audio_file, audio_load_tasks))

            # --- Batch inference ---
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
                                   args.output_json, "SenseVoice")
        logger.info(summary_msg)


if __name__ == "__main__":
    main()