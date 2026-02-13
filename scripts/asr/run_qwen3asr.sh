#!/usr/bin/env bash
set -e


# Runs Qwen3-ASR
script_dir="$(cd "$(dirname "$0")" && pwd)"
DATASET=${DATASET:-"/path/to/SQuTR/data/en/fiqa"}
AUDIO_DIR=${AUDIO_DIR:-"audio_clean"}
OUTPUT_JSON=${OUTPUT_JSON:-"$DATASET/$AUDIO_DIR/qwen3asr_result.jsonl"}

cd "$script_dir"
cd baselines/asr || { echo "Directory baselines/asr not found relative to $script_dir"; exit 1; }

python qwen3asr.py \
    --input_folder "$DATASET" \
    --output_json "$OUTPUT_JSON" \
    --input_json_path queries_with_audio.jsonl \
    --audio_base_path audios_clean \
    --model_path Qwen/Qwen3-ASR-1.7B \
    --batch_size 8 \
    --metric wer \
    --language English
