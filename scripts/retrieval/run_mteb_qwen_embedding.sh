python src/retrieval/qwen3_mteb_use.py \
    --model_size 4b \
    --data_dir_path "./Echo_Bench/en/fiqa/audio_noise_snr_10" \
    --query_field "asr_text" \
    --asr_result_file_name "asr_result" \
    --batch_size 256 \
    --log_path "./evaluation_results"