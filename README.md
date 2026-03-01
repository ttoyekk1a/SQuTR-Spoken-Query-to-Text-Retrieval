# SQuTR: A Robustness Benchmark for Spoken Query to Text Retrieval

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/SLLMCommunity/SQuTR)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2602.12783)
[![License: Code](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: Data](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

**SQuTR** (Spoken Query-to-Text Retrieval) is a large-scale bilingual benchmark designed to evaluate the robustness of information retrieval (IR) systems under realistic and complex acoustic perturbations.

While speech has become a primary interface for IR, performance often degrades significantly in noisy environments. SQuTR addresses this by extending 6 popular text retrieval datasets into the spoken domain, providing **37,317** complex queries across **6 domains**, synthesized with **200 real speakers**, and evaluated under **4 graded noise levels**.

---

## 🏆 Recognition
* **[2026-02]** SQuTR was featured as the **#1 Paper of the Day** on [Hugging Face Daily Papers](https://huggingface.co/papers/2602.12783)!

---

## 🌟 Key Features

* **Bilingual & Multi-Domain:** Includes 6 subsets from MTEB (English) and C-MTEB (Chinese) covering Wikipedia, Finance, Medical, and Encyclopedia domains.
* **High-Fidelity Synthesis:** Generated using **CosyVoice-3** with voice profiles from 200 real speakers (diverse genders, ages, and accents), totaling **190.4 hours** of audio.
* **Realistic Noise Modeling:** Features 17 categories of real-world environmental noise (e.g., transport, office, street) from DEMAND and NOISEX-92 datasets.
* **Robustness Evaluation:** Models four acoustic conditions: **Clean, Low Noise (20dB), Medium Noise (10dB), and High Noise (0dB)**.
* **Rigid Quality Control:** All samples undergo a three-stage verification process: automated filtering, ASR-based lexical consistency checks, and manual auditing by 10 bilingual annotators.

---

## 🛠 Dataset Generation Pipeline

SQuTR utilizes a sophisticated pipeline to ensure high data quality and acoustic diversity:

1.  **Text Processing:** Normalization of numbers, symbols, and abbreviations from the original MTEB/C-MTEB queries.
2.  **Voice Synthesis:** Each query is synthesized into three candidate versions using different speaker profiles; the one with the lowest WER/CER (via Whisper/Paraformer) is selected to minimize synthesis artifacts.
3.  **Acoustic Augmentation:** Noise is mixed based on RMS energy scaling to achieve precise Signal-to-Noise Ratio (SNR) levels.
4.  **Verification:** Human-in-the-loop validation for naturalness, semantic consistency, and noise level accuracy.

---

## 📊 Dataset Statistics

SQuTR maintains a balanced distribution of query complexities and domains. Statistics represent unique queries; each query is provided in all 4 acoustic conditions.

### Subset Breakdown
| Language | Subset | Source | Domain | Queries |
| :--- | :--- | :--- | :--- | :--- |
| **English** | **[NQ](https://huggingface.co/datasets/mteb/nq)** | MTEB | Wikipedia | 3,452 |
| | **[HotpotQA](https://huggingface.co/datasets/mteb/hotpotqa)** | MTEB | Wikipedia | 7,405 |
| | **[FiQA](https://huggingface.co/datasets/mteb/fiqa)** | MTEB | Finance | 648 |
| **Chinese** | **[MedicalRetrieval](https://huggingface.co/datasets/mteb/MedicalRetrieval)**| C-MTEB | Medical | 1,000 |
| | **[DuRetrieval](https://huggingface.co/datasets/mteb/DuRetrieval)** | C-MTEB | Encyclopedia | 2,000 |
| | **[T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval)** | C-MTEB | Encyclopedia | 22,812 |
| **Total** | | | | **37,317** |

### Overall Metrics
| Metric | English | Chinese | Total |
| :--- | :--- | :--- | :--- |
| **#Unique Queries** | 11,505 | 25,812 | 37,317 |
| **#Speakers** | 100 | 100 | 200 |
| **Total Speech Duration** | 76.4 h | 114.0 h | 190.4 h |
| **Average Speech Duration**| 5.98 s | 3.98 s | 4.59 s |
| **#Evaluation Instances** | 46,020 | 103,248 | 149,268 |

---

### Dataset Structure

```
SQuTR/
└── source_data/
    ├── en/ (English Datasets)
    │   ├── fiqa/ (Expanded in the screenshot)
    │   │   ├── audio_clean/              # Clean original audio files
    │   │   ├── audio_noise_snr_0/        # Audio with 0dB Signal-to-Noise Ratio
    │   │   ├── audio_noise_snr_10/       # Audio with 10dB Signal-to-Noise Ratio
    │   │   ├── audio_noise_snr_20/       # Audio with 20dB Signal-to-Noise Ratio
    │   │   ├── qrels/                    # Query relevance judgments
    │   │   ├── corpus.jsonl              # Text corpus documents
    │   │   ├── queries.jsonl             # Original text queries
    │   │   ├── queries_with_audio_clean.jsonl         # Metadata for clean audio queries
    │   │   ├── queries_with_audio_noise_snr_0.jsonl   # Metadata for 0dB noise queries
    │   │   ├── queries_with_audio_noise_snr_10.jsonl  # Metadata for 10dB noise queries
    │   │   └── queries_with_audio_noise_snr_20.jsonl  # Metadata for 20dB noise queries
    │   ├── hotpotqa/
    │   └── nq/
    └── zh/ (Chinese Datasets)
        ├── DuRetrieval/
        ├── MedicalRetrieval/
        └── T2Retrieval/
```

### Data Samples

**Chinese Dataset Example:**

Corpus (corpus.jsonl):
```json
{"_id":"30000001","text":"您好：脂肪瘤属良性肿瘤但术后容易复发，患者可以采用中草药消除，而且安全，不会对身体产生任何的伤害及毒副作用，治愈的希望也是比较大的。","title":""}
```

Clean Audio Query (queries_with_audio_clean.jsonl):
```json
{"_id": "1", "text": "多形型脂肪肉瘤（左阴囊内）", "audio": "1.wav"}
```

Noisy Audio Query (queries_with_audio_noise_snr_0.jsonl):
```json
{"_id": "1", "text": "多形型脂肪肉瘤（左阴囊内）", "audio": "noise_snr0_1.wav", "snr_db": 0, "noise_id": "demand_SCAFE_ch07"}
```

**English Dataset Example:**

Corpus (corpus.jsonl):
```json
{"_id": "3", "title": "", "text": "I'm not saying I don't like the idea of on-the-job training too, but you can't expect the company to do that. Training workers is not their job - they're building software. Perhaps educational systems in the U.S. (or their students) should worry a little about getting marketable skills in exchange for their massive investment in education, rather than getting out with thousands in student debt and then complaining that they aren't qualified to do anything."}
```

Clean Audio Query (queries_with_audio_clean.jsonl):
```json
{"_id": "4641", "text": "Where should I park my rainy-day / emergency fund?", "audio": "4641.wav"}
```

Noisy Audio Query (queries_with_audio_noise_snr_0.jsonl):
```json
{"_id": "4641", "text": "Where should I park my rainy-day / emergency fund?", "audio": "noise_snr0_4641.wav", "snr_db": 0, "noise_id": "demand_NRIVER_ch13"}
```

---

## 🚀 Retrieval Performance Benchmarks

We evaluate various retrieval models using a cascaded pipeline (ASR + Embedding) and end-to-end approaches. Results are reported as **nDCG@10 / MRR@10**.

### 1. Chinese Sub-dataset Retrieval Performance
| Model Configuration | Text (Ref) | Clean | Low (20dB) | Med (10dB) | High (0dB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lexical Baseline** | | | | | |
| BM25 | 0.4843 / 0.5756 | 0.4380 / 0.5246 | 0.4366 / 0.5229 | 0.4313 / 0.5177 | 0.4061 / 0.4895 |
| **Dense (BGE Series)** | | | | | |
| [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 0.6871 / 0.7446 | 0.6491 / 0.7064 | 0.6454 / 0.7025 | 0.6402 / 0.6978 | 0.6025 / 0.6593 |
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 0.7509 / 0.7890 | 0.7157 / 0.7557 | 0.7126 / 0.7523 | 0.7059 / 0.7457 | 0.6670 / 0.7075 |
| [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | 0.7662 / 0.8032 | 0.7306 / 0.7682 | 0.7274 / 0.7644 | 0.7212 / 0.7585 | 0.6801 / 0.7177 |
| [BGE-M3-dense](https://huggingface.co/BAAI/bge-m3) | 0.7320 / 0.7756 | 0.6937 / 0.7381 | 0.6914 / 0.7359 | 0.6864 / 0.7315 | 0.6460 / 0.6912 |
| **Dense (Other)** | | | | | |
| [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) | 0.6952 / 0.7446 | 0.6626 / 0.7122 | 0.6603 / 0.7105 | 0.6554 / 0.7047 | 0.6188 / 0.6681 |
| [Multilingual-E5-Large](https://huggingface.co/intfloat/multilingual-e5-large) | 0.7479 / 0.7900 | 0.7099 / 0.7528 | 0.7070 / 0.7503 | 0.7008 / 0.7447 | 0.6592 / 0.7019 |
| **Dense (Qwen3 Series)** | | | | | |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 0.7405 / 0.7840 | 0.7072 / 0.7512 | 0.7050 / 0.7492 | 0.6992 / 0.7438 | 0.6613 / 0.7063 |
| [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 0.7936 / 0.8237 | 0.7660 / 0.7978 | 0.7632 / 0.7958 | 0.7573 / 0.7899 | 0.7193 / 0.7528 |
| **[Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)** | **0.8033 / 0.8315** | **0.7760 / 0.8057** | **0.7741 / 0.8041** | **0.7686 / 0.7988** | **0.7302 / 0.7608** |
| **End-to-End Systems** | | | | | |
| [Omni-Embedding-Nemotron-3B](https://huggingface.co/nvidia/omni-embed-nemotron-3b)| - | 0.6648 / 0.7201 | 0.6614 / 0.7179 | 0.6507 / 0.7067 | 0.5742 / 0.6314 |

### 2. English Sub-dataset Retrieval Performance
| Model Configuration | Text (Ref) | Clean | Low (20dB) | Med (10dB) | High (0dB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lexical Baseline** | | | | | |
| BM25 | 0.3912 / 0.4547 | 0.3586 / 0.4197 | 0.3570 / 0.4177 | 0.3555 / 0.4157 | 0.3374 / 0.3956 |
| **Dense (BGE Series)** | | | | | |
| [bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 0.5345 / 0.5936 | 0.5070 / 0.5665 | 0.5035 / 0.5632 | 0.4992 / 0.5590 | 0.4756 / 0.5328 |
| [bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | 0.5578 / 0.6130 | 0.5260 / 0.5845 | 0.5220 / 0.5801 | 0.5194 / 0.5766 | 0.4962 / 0.5513 |
| [bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) | 0.5801 / 0.6299 | 0.5521 / 0.6058 | 0.5493 / 0.6040 | 0.5466 / 0.6015 | 0.5194 / 0.5721 |
| [BGE-M3-dense](https://huggingface.co/BAAI/bge-m3) | 0.5711 / 0.6368 | 0.5397 / 0.6035 | 0.5389 / 0.6034 | 0.5360 / 0.5996 | 0.5097 / 0.5686 |
| **Dense (Other)** | | | | | |
| [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) | 0.6029 / 0.6617 | 0.5797 / 0.6402 | 0.5775 / 0.6373 | 0.5747 / 0.6350 | 0.5497 / 0.6069 |
| [Stella-EN-400M-v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) | 0.6198 / 0.6786 | 0.6017 / 0.6599 | 0.5986 / 0.6573 | 0.5962 / 0.6546 | 0.5706 / 0.6255 |
| [All-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 0.4241 / 0.4862 | 0.3893 / 0.4433 | 0.3854 / 0.4387 | 0.3841 / 0.4368 | 0.3637 / 0.4136 |
| [Multilingual-E5-Large](https://huggingface.co/intfloat/multilingual-e5-large) | 0.5719 / 0.6323 | 0.5398 / 0.6006 | 0.5369 / 0.5967 | 0.5356 / 0.5960 | 0.5115 / 0.5698 |
| **Dense (Qwen3 Series)** | | | | | |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 0.5504 / 0.6234 | 0.5288 / 0.5988 | 0.5274 / 0.5978 | 0.5246 / 0.5961 | 0.5026 / 0.5697 |
| [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 0.6488 / 0.7110 | 0.6252 / 0.6886 | 0.6227 / 0.6860 | 0.6206 / 0.6839 | 0.5947 / 0.6565 |
| **[Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)** | **0.6686 / 0.7253** | **0.6450 / 0.7041** | **0.6424 / 0.7021** | **0.6405 / 0.7000** | **0.6120 / 0.6690** |
| **End-to-End Systems** | | | | | |
| [Omni-Embedding-Nemotron-3B](https://huggingface.co/nvidia/omni-embed-nemotron-3b)| - | 0.5712 / 0.5394 | 0.5680 / 0.5369 | 0.5605 / 0.5289 | 0.5236 / 0.4959 |

> **Note:** Cascaded systems use **Paraformer-Large** for Chinese and **Whisper-Large-v3** for English ASR. Models marked with "-" were not trained on that specific language's text.

---

## 📈 ASR Performance Benchmarks

We evaluate representative ASR models across all noise levels to provide a baseline for cascaded retrieval systems.

### 1. English Sub-dataset (Word Error Rate - WER %)
| Model | Clean | Low (20dB) | Medium (10dB) | High (0dB) |
| :--- | :---: | :---: | :---: | :---: |
| **[Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3)** | **3.33** | **4.10** | **4.48** | 7.75 |
| **[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)** | 4.49 | 4.90 | 5.15 | **6.98** |
| **[GLM-ASR-Nano](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)** | 6.58 | 6.37 | 6.25 | 9.13 |
| **[Fun-ASR-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)** | 6.47 | 6.74 | 6.93 | 10.03 |
| **[SenseVoice-Small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)** | 8.82 | 9.46 | 10.00 | 13.32 |

### 2. Chinese Sub-dataset (Character Error Rate - CER %)
| Model | Clean | Low (20dB) | Medium (10dB) | High (0dB) |
| :--- | :---: | :---: | :---: | :---: |
| **[Paraformer-Large](https://huggingface.co/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)** | **2.71** | **2.97** | **3.39** | 7.14 |
| **[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)** | 3.07 | 3.19 | 3.40 | **5.43** |
| **[Fun-ASR-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)** | 3.08 | 3.23 | 3.47 | 6.72 |
| **[GLM-ASR-Nano](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)** | 4.77 | 4.73 | 5.08 | 11.44 |
| **[SenseVoice-Small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)** | 5.32 | 6.22 | 6.44 | 11.99 |
|  **[Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3)** | 8.77 | 8.96 | 9.62 | 17.31 |

---

## 🛠️ Installation & Usage

### 1. Setup
```bash
git clone [https://github.com/ttoyekk1a/SQuTR-Spoken-Query-to-Text-Retrieval.git](https://github.com/ttoyekk1a/SQuTR-Spoken-Query-to-Text-Retrieval.git)
cd SQuTR-Spoken-Query-to-Text-Retrieval
pip install -r requirements.txt
```

## 📖 How to Use


### End-to-End: Omni-Embedding

This section demonstrates how to perform end-to-end audio retrieval using [NVIDIA Omni-Embedding (Nemotron-3B)](https://huggingface.co/nvidia/omni-embed-nemotron-3b).

#### 1. Run End-to-End Retrieval Script

You can run the following script (example: FiQA subset, audio_clean):

```bash
bash scripts/retrieval/run_omni_emb.sh
```

Or equivalently, run the following command directly:

```bash
python3 src/retrieval/omni_emb.py \
  --data_dir ./data/SQuTR/en/fiqa \
  --log_path ./results/en/fiqa/nv-omni-embed_audio_clean \
  --model_path nvidia/omni-embed-nemotron-3b \
  --audio_path ./data/SQuTR/en/fiqa/audio_clean \
  --query_file asr_result_cosy3.jsonl \
  --batch_size 32 \
  --query_field audio
```

The retrieval results will be saved in the `./results/en/fiqa/nv-omni-embed_audio_clean/` directory.

### Cascade: Dense Retrieval

#### Example: Whisper + BGE

Below is a typical pipeline for evaluating retrieval on the `FiQA` subset (audio_clean) using `Whisper-Large-V3` for ASR and `BAAI/bge-base-en-v1.5` for dense retrieval.

#### 1. Run Whisper ASR to transcribe audio queries

You can use the provided script:

```bash
bash scripts/asr/run_whisper_large_v3.sh
```

Or equivalently, run the following command directly:

```bash
python scripts/asr/baselines/asr/whisper.py \
    --input_folder ./data/SQuTR/en/fiqa \
    --output_json ./data/SQuTR/en/fiqa/audio_clean/asr_result.jsonl \
    --input_json_path queries_with_audio.jsonl \
    --audio_base_path audio_clean \
    --model_path openai/whisper-large-v3 \
    --batch_size 16 \
    --num_workers 16 \
    --language_token "<|en|>" \
    --metric wer
```

#### 2. Run dense retrieval evaluation with BGE

You can use the provided script:

```bash
bash scripts/retrieval/run_mteb_dense.sh
```

Or equivalently, run the following command directly:

```bash
python src/retrieval/mteb_use.py \
  --corpus_path ./data/SQuTR/en/fiqa/corpus.jsonl \
  --query_path ./data/SQuTR/en/fiqa/audio_clean/asr_result.jsonl \
  --qrels_path ./data/SQuTR/en/fiqa/qrels/test.jsonl \
  --model_path BAAI/bge-base-en-v1.5 \
  --log_path ./results/en/fiqa/bge-base-en-v1.5_audio_clean \
  --query_field asr_text \
  --batch_size 32
```

The final retrieval evaluation results will be saved in the `./results/en/fiqa/bge-base-en-v1.5_audio_clean/` directory.

#### 2. Run dense retrieval evaluation with Qwen3-Embedding

You can use the provided script:

```bash
bash scripts/retrieval/run_mteb_qwen_embedding.sh
```

Or equivalently, run the following command directly:

```bash
python src/retrieval/qwen3_mteb_use.py \
    --model_size 4b \
    --data_dir_path "./Echo_Bench/en/fiqa/audio_noise_snr_10" \
    --query_field "asr_text" \
    --asr_result_file_name "asr_result" \
    --batch_size 256 \
    --log_path "./evaluation_results"
```

### Cascade: Lexical Retrieval

#### Evaluation Example: BM25

This section demonstrates how to evaluate lexical retrieval (BM25) on the FiQA subset (audio_clean) using ASR outputs.

#### 1. Run BM25 retrieval evaluation

You can use the provided script:

```bash
bash scripts/retrieval/run_bm25.sh
```

Or equivalently, run the following command directly:

```bash
python src/retrieval/bm25_en.py \
  --data_dir ./data/SQuTR/en/fiqa \
  --query_file whisper-large-v3-result.jsonl \
  --log_path ./results/en/fiqa/bm25_audio_clean/bm25_full_metrics.log \
  --audio_path ./data/SQuTR/en/fiqa/audio_clean \
  --ndcg_k 10 20 100 \
  --query_field asr_text
```

The BM25 evaluation results will be saved in the `./results/en/fiqa/bm25_audio_clean/` directory.

## Citation
If you find the **SQuTR** dataset or code useful in your research, we would greatly appreciate it if you could cite our paper:

```
@misc{li2026squtrrobustnessbenchmarkspoken,
      title={SQuTR: A Robustness Benchmark for Spoken Query to Text Retrieval under Acoustic Noise}, 
      author={Yuejie Li and Ke Yang and Yueying Hua and Berlin Chen and Jianhao Nie and Yueping He and Caixin Kang},
      year={2026},
      eprint={2602.12783},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2602.12783}, 
}
```
