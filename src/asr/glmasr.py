import time
import json
import argparse
import os
import torch
import warnings
from tqdm import tqdm

# Import transformers
from transformers import AutoModel, AutoProcessor
from modelscope import snapshot_download

# Import shared utilities
from asr_utils import (
    setup_logger, compute_metric, load_data, get_audio_paths,
    get_normalizers, write_results, print_summary
)

# Ignore warnings
warnings.filterwarnings("ignore")


# ================= Core class: GLM-ASR Wrapper (fixed version) =================
class GlmBatchASR:
    def __init__(self, model_id="ZhipuAI/GLM-ASR-Nano-2512", device="cuda:0"):
        print(f"Downloading/Loading GLM-ASR model: {model_id}")
        
        try:
            # 1. First download the model to local cache using modelscope
            # model_dir = snapshot_download(model_id)
            model_dir = model_id
            print(f"Model downloaded to: {model_dir}")

            # 2. Load using transformers with trust_remote_code=True
            self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_dir, 
                trust_remote_code=True,
                device_map=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            self.device = self.model.device
            self.dtype = self.model.dtype
            
            # Set to evaluation mode
            self.model.eval()
            print(f"GLM-ASR model loaded successfully on {self.device}!")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load GLM-ASR model. Details: {e}")
            raise e

    def process_batch(self, audio_paths):
        """
        Process a batch of audio file paths
        """
        results = [""] * len(audio_paths)
        valid_indices = []
        valid_paths = []

        # Filter invalid paths
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
            # Build input: GLM-ASR processor supports direct path list input
            # Note: apply_transcription_request is a GLM-ASR specific method existing in its remote code
            # If errors occur, check transformers version compatibility or model code updates
            inputs = self.processor.apply_transcription_request(valid_paths)
            
            # Move to GPU
            inputs = inputs.to(self.device, dtype=self.dtype)

            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=500)

            # Decode output tokens
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_length:]
            
            decoded_outputs = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)

            # Fill results back to original positions
            for i, text in enumerate(decoded_outputs):
                original_idx = valid_indices[i]
                results[original_idx] = text.strip()

        except Exception as e:
            print(f"Batch inference error: {e}")
            
        return results


# ================= Main process =================
def main():
    parser = argparse.ArgumentParser(description="Run GLM-ASR batch inference.")

    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing jsonl and audios")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save output jsonl file")
    parser.add_argument("--input_json_path", type=str, default="queries_with_audio.jsonl", help="Input jsonl filename")
    parser.add_argument("--audio_base_path", type=str, default="audio", help="Base path for audio files")
    
    # Default to ModelScope model ID
    parser.add_argument("--model_path", type=str, 
                        default="ZhipuAI/GLM-ASR-Nano-2512",
                        help="ModelScope model ID")
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (GLM-ASR requires significant VRAM, start small)")
    parser.add_argument("--metric", type=str, default="cer", choices=["wer", "cer"])
    
    args = parser.parse_args()

    input_json_path = os.path.join(args.input_folder, args.input_json_path)
    audio_base_path = os.path.join(args.input_folder, args.audio_base_path)
    log_file = args.output_json.replace(".jsonl", ".log")
    
    logger = setup_logger(log_file)
    logger.info(f"Task Started. Model: {args.model_path}")

    # 1. Load Data
    samples = load_data(input_json_path)

    logger.info(f"Total samples: {len(samples)}")

    # 2. Initialize model
    try:
        asr_model = GlmBatchASR(model_id=args.model_path, device=args.device)
    except Exception as e:
        logger.error(f"Model init failed: {e}")
        return

    # 3. Batch inference
    total_metric_score = 0
    valid_count = 0
    global_start_time = time.time()

    en_tn_model, zh_tn_model, inflect_parser = get_normalizers()

    with open(args.output_json, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(samples), args.batch_size), desc="Inferencing"):
            batch_samples = samples[i : i + args.batch_size]
            batch_audio_paths = []

            # Prepare audio file paths
            for sample in batch_samples:
                audio_filename = sample.get("audio") or sample.get("tts")
                if audio_filename:
                    full_path = os.path.join(audio_base_path, audio_filename)
                    batch_audio_paths.append(full_path)
                else:
                    batch_audio_paths.append(None)

            # Run inference
            batch_results = asr_model.process_batch(batch_audio_paths)

            # Calculate metrics and write results
            for j, sample in enumerate(batch_samples):
                asr_text = batch_results[j]
                ground_truth = sample.get("text", "")

                score = compute_metric(ground_truth, asr_text, args.metric, en_tn_model, zh_tn_model, inflect_parser)
                total_metric_score += score
                valid_count += 1

                write_results(f_out, sample, asr_text, score, args.metric)

            f_out.flush()

    total_duration_ms = time.time() - global_start_time
    if valid_count > 0:
        avg_metric_score = total_metric_score / valid_count
        summary_msg = print_summary(valid_count, total_duration_ms, avg_metric_score, args.metric,
                                   args.output_json, "GLM-ASR")
        logger.info(summary_msg)

if __name__ == "__main__":
    main()