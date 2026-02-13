# Qwen ASR Batch Inference Script
import time
import json
import argparse
import os
import torch
import warnings
from tqdm import tqdm

# ================= Core Dependency: Qwen-ASR =================
try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    raise ImportError("Please ensure qwen_asr library is installed")

# Import shared utilities
from asr_utils import (
    setup_logger, compute_metric, load_data, get_audio_paths,
    get_normalizers, write_results, print_summary
)

# Ignore warnings
warnings.filterwarnings("ignore")


# ================= Core Class: QwenBatchASR Wrapper =================
class QwenBatchASR:
    def __init__(self, model_path, device="cuda", max_batch_size=32):
        print(f"Loading Qwen3ASR model from: {model_path}")
        
        try:
            # Initialize Qwen3ASR with bfloat16 for performance/precision balance
            self.model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map=device,
                max_inference_batch_size=max_batch_size,
                max_new_tokens=256,
            )
            print(f"Qwen3ASR model loaded successfully!")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load Qwen3ASR model. Details: {e}")
            raise e

    def process_batch(self, audio_paths, language=None):
        """Processes a batch of audio file paths."""
        results = [""] * len(audio_paths)
        valid_indices = []
        valid_paths = []

        # 1. Filter valid file paths
        for idx, path in enumerate(audio_paths):
            if path and os.path.exists(path):
                valid_indices.append(idx)
                valid_paths.append(path)
            else:
                if path:
                    print(f"Warning: Audio file not found: {path}")

        if not valid_paths:
            return results

        try:
            # 2. Construct language parameters
            # Use specific language if provided, otherwise allow model to auto-detect
            lang_param = None
            if language:
              lang_param = [language] * len(valid_paths)

            # 3. Inference (transcribe supports list input)
            outputs = self.model.transcribe(audio=valid_paths, language=lang_param)

            # 4. Map results back to original order
            for i, res_obj in enumerate(outputs):
                original_idx = valid_indices[i]
                results[original_idx] = res_obj.text.strip()

        except Exception as e:
            print(f"Batch inference error: {e}")
            import traceback
            traceback.print_exc()
            
        return results


# ================= Main Process =================
def main():
    parser = argparse.ArgumentParser(description="Run Qwen3ASR batch inference.")

    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing jsonl and audios")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save output jsonl file")
    parser.add_argument("--input_json_path", type=str, default="queries_with_audio.jsonl", help="Input jsonl filename")
    parser.add_argument("--audio_base_path", type=str, default="audio", help="Base path for audio files")
    
    parser.add_argument("--model_path", type=str, 
                        default="/path/to/Qwen3-ASR-1.7B",
                        help="Path to Qwen3ASR model")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing loop")
    parser.add_argument("--metric", type=str, default="wer", choices=["wer", "cer"], help="Metric for evaluation")
    
    parser.add_argument("--language", type=str, default=None, 
                        help="Force language (e.g., 'English', 'Chinese'). Default is auto-detect.")
    
    args = parser.parse_args()

    input_json_path = os.path.join(args.input_folder, args.input_json_path)
    audio_base_path = os.path.join(args.input_folder, args.audio_base_path)
    log_file = args.output_json.replace(".jsonl", ".log")
    
    logger = setup_logger(log_file)
    logger.info(f"Task Started. Model: {args.model_path}")

    # 1. Load Data
    samples = load_data(input_json_path)

    logger.info(f"Total samples loaded: {len(samples)}")

    # 2. Initialize Model
    try:
        asr_model = QwenBatchASR(model_path=args.model_path, device=args.device, max_batch_size=32)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return

    # 3. Batch Inference Loop
    total_metric_score = 0
    valid_count = 0
    global_start_time = time.time()

    en_tn_model, zh_tn_model, inflect_parser = get_normalizers()

    logger.info(f"Starting Inference: batch_size={args.batch_size}, language={args.language or 'Auto'}")

    with open(args.output_json, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(samples), args.batch_size), desc="Inferencing"):
            batch_samples = samples[i : i + args.batch_size]
            batch_audio_paths = []

            # Prepare audio file paths
            for sample in batch_samples:
                # Support both 'audio' or 'tts' field names
                audio_filename = sample.get("audio") or sample.get("tts")
                if audio_filename:
                    # Check if audio_filename is already an absolute path
                    if os.path.isabs(audio_filename):
                        full_path = audio_filename
                    else:
                        full_path = os.path.join(audio_base_path, audio_filename)
                    batch_audio_paths.append(full_path)
                else:
                    batch_audio_paths.append(None)

            # Perform batch inference
            batch_results = asr_model.process_batch(batch_audio_paths, language=args.language)

            # Calculate metrics and write results
            for j, sample in enumerate(batch_samples):
                asr_text = batch_results[j]
                ground_truth = sample.get("text", "") or sample.get("ref_text", "")

                score = compute_metric(ground_truth, asr_text, args.metric, en_tn_model, zh_tn_model, inflect_parser)
                total_metric_score += score
                valid_count += 1

                write_results(f_out, sample, asr_text, score, args.metric)

            f_out.flush()

    total_duration_ms = (time.time() - global_start_time) * 1000
    if valid_count > 0:
        avg_metric_score = total_metric_score / valid_count
        summary_msg = print_summary(valid_count, total_duration_ms, avg_metric_score, args.metric,
                                   args.output_json, "Qwen3ASR")
        logger.info(summary_msg)

if __name__ == "__main__":
    main()