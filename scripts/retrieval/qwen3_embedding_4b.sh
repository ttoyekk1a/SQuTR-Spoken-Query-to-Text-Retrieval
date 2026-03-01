CUDA_VISIBLE_DEVICES=1 vllm serve ./Qwen3-Embedding-4B \
    --port 8001 \
    --convert embed  \
    --tensor-parallel-size 1