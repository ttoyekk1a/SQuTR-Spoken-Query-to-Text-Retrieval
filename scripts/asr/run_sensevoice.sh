#!/usr/bin/env bash
set -e


# Runs SenseVoice ASR
script_dir="$(cd "$(dirname "$0")" && pwd)"
DATASET=${DATASET:-"/path/to/SQuTR/data/en/fiqa"}
AUDIO_DIR=${AUDIO_DIR:-"audio_clean"}
OUTPUT_JSON=${OUTPUT_JSON:-"$DATASET/$AUDIO_DIR/sensevoice_result.jsonl"}

cd "$script_dir"
cd baselines/asr || { echo "Directory baselines/asr not found relative to $script_dir"; exit 1; }

python sensevoice.py \
    --input_folder "$DATASET" \
    --output_json "$OUTPUT_JSON" \
    --input_json_path queries_with_audio.jsonl \
    --audio_base_path audios_clean \
    --model_path iic/SenseVoiceSmall \
    --batch_size 16 \
    --num_workers 16 \
    --language auto \
    --metric cer
