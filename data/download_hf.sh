pip install -U huggingface_hub
huggingface-cli download --repo-type dataset SLLMCommunity/SQuTR --local-dir ./SQuTR
cd SQuTR
unzip source_data.zip