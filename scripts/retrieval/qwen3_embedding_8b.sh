vllm serve ./Qwen3-8B-Embedding \
    --port 8002 \
    --convert embed \
    --tensor-parallel-size 1