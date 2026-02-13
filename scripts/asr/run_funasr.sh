#!/usr/bin/env bash
set -e


# Runs FunASR
script_dir="$(cd "$(dirname "$0")" && pwd)"
DATASET=${DATASET:-"/path/to/SQuTR/data/en/fiqa"}
AUDIO_DIR=${AUDIO_DIR:-"audio_clean"}
OUTPUT_JSON=${OUTPUT_JSON:-"$DATASET/$AUDIO_DIR/funasr_result.jsonl"}

cd "$script_dir"
cd baselines/asr || { echo "Directory baselines/asr not found relative to $script_dir"; exit 1; }

python funasr.py \
    --input_folder "$DATASET" \
    --output_json "$OUTPUT_JSON" \
    --input_json_path queries_with_audio.jsonl \
    --audio_base_path audios_clean \
    --model_path FunAudioLLM/Fun-ASR-Nano-2512 \
    --language "英文" \
    --batch_size 16 \
    --num_workers 16 \
    --metric cer
