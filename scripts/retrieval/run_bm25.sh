DATA_ROOT="./data/SQuTR"
DATASET="en/fiqa"
OUTPUT_PREFIX="bm25"
AUDIO_DIR="audio_clean"
LOG_ROOT="./results"
LOG_PATH="${LOG_ROOT}/${DATASET}/${OUTPUT_PREFIX}_${AUDIO_DIR}/bm25_full_metrics.log"
AUDIO_PATH="${DATA_ROOT}/${DATASET}/${AUDIO_DIR}"
NDCG_K="10 20 100"
QUERY_FIELD="asr_text"
QUERY_FILE="whisper-large-v3-result.jsonl"

mkdir -p "$(dirname "$LOG_PATH")"

python src/retrieval/bm25_en.py \
	--data_dir "${DATA_ROOT}/${DATASET}" \
    --query_file "${QUERY_FILE}" \
	--log_path "$LOG_PATH" \
	--audio_path "$AUDIO_PATH" \
	--ndcg_k $NDCG_K \
	--query_field "$QUERY_FIELD"

echo "Results written to $LOG_PATH"
