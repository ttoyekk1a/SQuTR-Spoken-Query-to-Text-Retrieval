#!/usr/bin/env bash
set -e


# Runs Whisper Large V3 for ASR
script_dir="$(cd "$(dirname "$0")" && pwd)"
DATASET=${DATASET:-"/path/to/SQuTR/data/en/fiqa"}
AUDIO_DIR=${AUDIO_DIR:-"audio_clean"}
OUTPUT_JSON=${OUTPUT_JSON:-"$DATASET/$AUDIO_DIR/asr_result.jsonl"}

cd "$script_dir"
cd baselines/asr || { echo "Directory baselines/asr not found relative to $script_dir"; exit 1; }

python whisper.py \
    --input_folder "$DATASET" \
    --output_json "$OUTPUT_JSON" \
    --input_json_path queries_with_audio.jsonl \
    --audio_base_path audios_clean \
    --model_path openai/whisper-large-v3 \
    --batch_size 16 \
    --num_workers 16 \
    --language_token "<|en|>" \
    --metric wer
