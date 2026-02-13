DATA_ROOT="./data/SQuTR"
DATASET="en/fiqa"
OUTPUT_PREFIX="nv-omni-embed"
AUDIO_DIR="audio_clean"
LOG_ROOT="./results"
MODEL_PATH="nvidia/omni-embed-nemotron-3b"
QUERY_FILE="asr_result_cosy3.jsonl"
QUERY_FIELD="audio"
BATCH_SIZE="32"

# Derived
DATA_DIR="${DATA_ROOT}/${DATASET}"
LOG_PATH="${LOG_ROOT}/${DATASET}/${OUTPUT_PREFIX}_${AUDIO_DIR}"
AUDIO_PATH="${DATA_ROOT}/${DATASET}/${AUDIO_DIR}"

mkdir -p "$LOG_PATH"

python3 src/retrieval/omni_emb.py --data_dir ${DATA_DIR} \
	--log_path ${LOG_PATH} \
	--model_path ${MODEL_PATH} \
	--audio_path ${AUDIO_PATH} \
	--query_file ${QUERY_FILE} \
	--batch_size ${BATCH_SIZE} \
	--query_field ${QUERY_FIELD}

echo "Results written to ${LOG_PATH}"
