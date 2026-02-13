import time
import librosa
import numpy as np
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

# Import shared utilities
from asr_utils import (
    setup_logger, compute_metric, load_data, get_audio_paths,
    get_normalizers, write_results, print_summary
)

def load_audio_file(args):
    """
    Function for multi-threaded audio loading.
    args: (full_path, sample_rate)
    """
    path, sr = args
    try:
        # Load audio using librosa (CPU and IO intensive)
        y, _ = librosa.load(path, sr=sr)
        return y
    except Exception as e:
        return None

# ================= Core Class =================
class WhisperBatchASR():
    def __init__(self, model_path, batch_size=32):
        print(f"Initializing vLLM with model: {model_path}...")
        
        self.llm = LLM(
            model=model_path,
            max_model_len=448,
            max_num_seqs=batch_size, # Maximum concurrency equals batch size
            limit_mm_per_prompt={"audio": 1},
            kv_cache_dtype="fp8",
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
        )
        print("vLLM Initialized!")

        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=256,
        )

    def process_batch(self, audio_arrays, language_token="<en>"):
        """
        Batch Inference.
        """
        prompts = []
        valid_indices = []
        
        # Construct Prompt
        for idx, audio in enumerate(audio_arrays):
            if audio is not None:
                prompts.append({
                    "encoder_prompt": {
                        "prompt": "", 
                        "multi_modal_data": {
                            "audio": (audio, 16000), 
                        },
                    },
                    "decoder_prompt": f"<|startoftranscript|>{language_token}<|transcribe|><|notimestamps|>",
                })
                valid_indices.append(idx)
        
        if not prompts:
            return [""] * len(audio_arrays)

        # Execute batch generation (vLLM handles concurrency internally)
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        
        # Reconstruct result order
        results = [""] * len(audio_arrays)
        for i, output in enumerate(outputs):
            original_idx = valid_indices[i]
            results[original_idx] = output.outputs[0].text
            
        return results

# ================= Main Process =================
def main():
    parser = argparse.ArgumentParser(description="Run vLLM Whisper ASR concurrently.")
    
    # Path Arguments
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing queries_with_audio.jsonl and audios/")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save output jsonl file")
    parser.add_argument("--input_json_path", type=str, default="queries_with_audio.jsonl", help="Path to input jsonl file")
    parser.add_argument("--audio_base_path", type=str, default="audios", help="Base path for audio files")
    
    # Model and Computation Arguments
    parser.add_argument("--model_path", type=str, default="openai/whisper-large-v3", help="Path to the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="CPU threads for audio loading")
    parser.add_argument("--language_token", type=str, default="<|en|>", help="Language token")
    
    # Evaluation Arguments
    parser.add_argument("--metric", type=str, default="cer", choices=["wer", "cer"], help="Metric to calculate")
    
    args = parser.parse_args()
    
    # Join paths
    input_json_path = os.path.join(args.input_folder, args.input_json_path)
    audio_base_path = os.path.join(args.input_folder, args.audio_base_path)
    log_file = os.path.join(args.output_json.replace("jsonl","log"))

    # Initialize Logger
    logger = setup_logger(log_file)
    logger.info(f"Task Started. Input: {input_json_path}")
    
    # 1. Load Data
    samples = load_data(input_json_path)
    
    logger.info(f"Total samples loaded: {len(samples)}")

    # 2. Initialize Model
    asr_model = WhisperBatchASR(
        model_path=args.model_path, 
        batch_size=args.batch_size,
    )

    # 3. Batch Processing and Inference
    total_metric_score = 0
    valid_count = 0
    
    # Timer for total elapsed time (includes IO and Inference)
    global_start_time = time.time()
    
    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_json)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Thread pool for parallel audio loading
    thread_pool = ThreadPoolExecutor(max_workers=args.num_workers)
    en_tn_model, zh_tn_model, inflect_parser = get_normalizers()

    with open(args.output_json, "w", encoding="utf-8") as f_out:
        # Iterate by batch_size
        for i in tqdm(range(0, len(samples), args.batch_size), desc="Inferencing"):
            batch_samples = samples[i : i + args.batch_size]
            
            # --- Concurrent Audio Loading ---
            audio_load_tasks = []
            for sample in batch_samples:
                if "audio" in sample:
                    audio_filename = sample["audio"]
                else:
                    audio_filename = sample["tts"][0] if isinstance(sample["tts"], list) else sample["tts"]
                
                full_audio_path = os.path.join(audio_base_path, audio_filename)
                audio_load_tasks.append((full_audio_path, 16000))
            
            audio_arrays = list(thread_pool.map(load_audio_file, audio_load_tasks))
            
            # --- Batch Inference ---
            try:
                batch_results = asr_model.process_batch(audio_arrays, language_token=args.language_token)
            except Exception as e:
                logger.error(f"Batch inference error at index {i}: {e}")
                batch_results = [""] * len(batch_samples)

            # --- Write results and accumulate metrics ---
            for j, sample in enumerate(batch_samples):
                asr_text = batch_results[j]
                ground_truth = sample.get("text", "") 
                
                # Compute CER/WER
                score = compute_metric(ground_truth, asr_text, args.metric, en_tn_model, zh_tn_model, inflect_parser)
                total_metric_score += score
                valid_count += 1
                
                # Write to output file
                write_results(f_out, sample, asr_text, score, args.metric)
            
            f_out.flush()

    thread_pool.shutdown()
    
    global_end_time = time.time()
    total_duration_ms = (global_end_time - global_start_time) * 1000

    # 4. Final Statistics
    if valid_count > 0:
        avg_metric_score = total_metric_score / valid_count
        
        summary_msg = print_summary(valid_count, total_duration_ms, avg_metric_score, args.metric,
                                   args.output_json, "Whisper")
        logger.info(summary_msg)

if __name__ == "__main__":
    main()