#!/usr/bin/env bash

DATA_ROOT="./data/SQuTR"
DATASET="en/fiqa"
OUTPUT_PREFIX="bge-base-en-v1.5"
LOG_ROOT="./results"
MODEL_PATH="BAAI/bge-base-en-v1.5"
AUDIO_DIR="audio_clean"
QUERY_FILE="whisper-large-v3-result.jsonl"
QUERY_FIELD="asr_text"
BATCH_SIZE="32"

# Derived
DATA_DIR="${DATA_ROOT}/${DATASET}"
LOG_PATH="${LOG_ROOT}/${DATASET}/${OUTPUT_PREFIX}_${AUDIO_DIR}"
CORPUS_PATH="${DATA_DIR}/corpus.jsonl"
QUERY_PATH="${DATA_DIR}/${AUDIO_DIR}/${QUERY_FILE}"
QRELS_PATH="${DATA_DIR}/qrels/test.jsonl"

mkdir -p "$LOG_PATH"

python3 src/retrieval/mteb_use.py \
  --corpus_path "$CORPUS_PATH" \
  --query_path "$QUERY_PATH" \
  --qrels_path "$QRELS_PATH" \
  --model_path "$MODEL_PATH" \
  --log_path "$LOG_PATH" \
  --query_field "$QUERY_FIELD" \
  --batch_size "$BATCH_SIZE"

echo "Results written to ${LOG_PATH}"
