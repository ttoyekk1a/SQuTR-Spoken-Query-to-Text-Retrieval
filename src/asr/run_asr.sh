cd baselines/asr
DATASET="/path/to/SQuTR/data/en/fiqa" # Path to SQuTR data folder
OUTPUT_JSON=$DATASET/asr_result.jsonl # Path to output ASR result jsonl file

# ========== Run Whisper-Large-V3 ==========
python whisper.py \
    --input_folder $DATASET \
    --output_json $OUTPUT_JSON \
    --input_json_path queries_with_audio.jsonl \
    --audio_base_path audios_clean \
    --model_path openai/whisper-large-v3 \
    --batch_size 16 \
    --num_workers 16 \
    --language_token "<|en|>" \
    --metric wer

# ========== Run Qwen3-ASR (Commented) ==========
# python qwen3asr.py \
#     --input_folder $DATASET \
#     --output_json $DATASET/qwen3asr_result.jsonl \
#     --input_json_path queries_with_audio.jsonl \
#     --audio_base_path audios_clean \
#     --model_path Qwen/Qwen3-ASR-1.7B \
#     --batch_size 8 \
#     --metric wer \
#     --language English

# ========== Run Paraformer (Commented) ==========
# python paraformer.py \
#     --input_folder $DATASET \
#     --output_json $DATASET/paraformer_result.jsonl \
#     --input_json_path queries_with_audio.jsonl \
#     --audio_base_path audios_clean \
#     --model_path iic/speech_paraformer_asr-en-16k-vocab4199-pytorch \ # for ZH dataset, use iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
#     --batch_size 32 \
#     --num_workers 32 \
#     --metric wer

# ========== Run GLM-ASR (Commented) ==========
# python glmasr.py \
#     --input_folder $DATASET \
#     --output_json $DATASET/glmasr_result.jsonl \
#     --input_json_path queries_with_audio.jsonl \
#     --audio_base_path audios_clean \
#     --model_path zai-org/GLM-ASR-Nano-2512 \
#     --batch_size 4 \
#     --metric wer

# ========== Run FunASR (Commented) ==========
# python funasr.py \
#     --input_folder $DATASET \
#     --output_json $DATASET/funasr_result.jsonl \
#     --input_json_path queries_with_audio.jsonl \
#     --audio_base_path audios_clean \
#     --model_path FunAudioLLM/Fun-ASR-Nano-2512 \
#     --language 英文 \
#     --batch_size 16 \
#     --num_workers 16 \
#     --metric cer

# ========== Run SenseVoice (Commented) ==========
# python sensevoice.py \
#     --input_folder $DATASET \
#     --output_json $DATASET/sensevoice_result.jsonl \
#     --input_json_path queries_with_audio.jsonl \
#     --audio_base_path audios_clean \
#     --model_path iic/SenseVoiceSmall \
#     --batch_size 16 \
#     --num_workers 16 \
#     --language auto \
#     --metric cer


