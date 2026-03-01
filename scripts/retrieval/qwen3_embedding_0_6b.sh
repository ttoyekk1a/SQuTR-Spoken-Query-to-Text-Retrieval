CUDA_VISIBLE_DEVICES=0 vllm serve ./Qwen3-Embedding-0.6B \
    --port 8000 \
    --convert embed