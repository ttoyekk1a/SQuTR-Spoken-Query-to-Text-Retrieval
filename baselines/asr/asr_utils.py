"""
Shared utilities for ASR batch inference scripts.
Consolidates common functionality across different ASR models.
"""

import re
import json
import logging
import os
import inflect
import jiwer
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer


# ================= Text Processing =================
def spell_out_number(text: str, inflect_parser):
    """Convert numerical digits into spoken words for better metric calculation."""
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation:
    1. Lowercase transformation.
    2. Keep only alphanumeric and Chinese characters.
    3. Remove spaces adjacent to Chinese characters (preserve EN-EN spaces only).
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9\u4e00-\u9fff\s]', '', text)
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+|\s+(?=[\u4e00-\u9fff])', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_metric(reference, hypothesis, metric_type="cer", 
                   en_tn_model=None, zh_tn_model=None, inflect_parser=None):
    """
    Calculate WER (Word Error Rate) or CER (Character Error Rate).
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text from ASR
        metric_type: "wer" or "cer"
        en_tn_model: English text normalizer
        zh_tn_model: Chinese text normalizer
        inflect_parser: Number-to-word converter
    
    Returns:
        Error rate (0.0 to 1.0)
    """
    if metric_type == "wer":
        reference = en_tn_model.normalize(reference)
        hypothesis = en_tn_model.normalize(hypothesis)
    else:
        reference = zh_tn_model.normalize(reference)
        hypothesis = zh_tn_model.normalize(hypothesis)

    reference = spell_out_number(reference, inflect_parser)
    hypothesis = spell_out_number(hypothesis, inflect_parser)

    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref:
        return 0.0 if not hyp else 1.0

    if metric_type == "wer":
        return jiwer.wer(ref, hyp)
    else:
        return jiwer.cer(ref, hyp)


# ================= Logging Configuration =================
def setup_logger(log_file_path: str):
    """Set up logger that outputs to both file and console."""
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("ASR_Batch_Inference")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ================= Data Loading =================
def load_data(json_file_path: str):
    """Load samples from JSONL file."""
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Input file not found: {json_file_path}")
    
    samples = []
    with open(json_file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def get_audio_paths(samples, audio_base_path: str, max_samples=None):
    """
    Extract audio paths from samples.
    
    Supports both 'audio' and 'tts' field names.
    """
    audio_paths = []
    for i, sample in enumerate(samples[:max_samples] if max_samples else samples):
        audio_filename = sample.get("audio") or sample.get("tts")
        
        if isinstance(audio_filename, list):
            audio_filename = audio_filename[0]
        
        if audio_filename:
            if os.path.isabs(audio_filename):
                audio_paths.append(audio_filename)
            else:
                audio_paths.append(os.path.join(audio_base_path, audio_filename))
        else:
            audio_paths.append(None)
    
    return audio_paths


# ================= Text Normalizer Initialization =================
def get_normalizers():
    """Initialize and return English and Chinese text normalizers."""
    return EnNormalizer(), ZhNormalizer(), inflect.engine()


# ================= Output Writing =================
def write_results(f_out, sample, asr_text: str, score: float, metric_type: str):
    """Write inference result to output file."""
    sample["asr_text"] = asr_text
    sample[metric_type] = score
    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_summary(valid_count, total_duration_ms, avg_metric_score, metric_type, output_file, model_name="ASR Model"):
    """Print and return inference summary statistics."""
    if valid_count == 0:
        return
    
    avg_time_per_sample = total_duration_ms / valid_count
    
    summary_msg = (
        f"\n{'='*50}\n"
        f"Inference Summary ({model_name})\n"
        f"{'='*50}\n"
        f"Total Processed:  {valid_count}\n"
        f"Total Duration:   {total_duration_ms/1000:.2f} s\n"
        f"Avg Latency:      {avg_time_per_sample:.2f} ms/sample\n"
        f"Average {metric_type.upper()}:      {avg_metric_score:.4f} ({avg_metric_score:.2%})\n"
        f"Output File:      {output_file}\n"
        f"{'='*50}"
    )
    print(summary_msg)
    return summary_msg
