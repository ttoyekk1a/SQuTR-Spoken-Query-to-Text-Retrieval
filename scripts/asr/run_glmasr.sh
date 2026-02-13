#!/usr/bin/env bash
set -e


# Runs GLM-ASR
script_dir="$(cd "$(dirname "$0")" && pwd)"
DATASET=${DATASET:-"/path/to/SQuTR/data/en/fiqa"}
AUDIO_DIR=${AUDIO_DIR:-"audio_clean"}
OUTPUT_JSON=${OUTPUT_JSON:-"$DATASET/$AUDIO_DIR/glmasr_result.jsonl"}

cd "$script_dir"
cd baselines/asr || { echo "Directory baselines/asr not found relative to $script_dir"; exit 1; }

python glmasr.py \
    --input_folder "$DATASET" \
    --output_json "$OUTPUT_JSON" \
    --input_json_path queries_with_audio.jsonl \
    --audio_base_path audios_clean \
    --model_path zai-org/GLM-ASR-Nano-2512 \
    --batch_size 4 \
    --metric wer
