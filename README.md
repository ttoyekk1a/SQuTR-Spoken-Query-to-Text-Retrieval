# SQuTR — Spoken Query-to-Text Retrieval Benchmark

Overview
----
SQuTR is a bilingual (Chinese & English) spoken query → text retrieval benchmark. It collects spoken queries and retrieval corpora to evaluate retrieval systems that accept spoken queries (either end-to-end or via ASR).

Included datasets (known)
----
- Chinese: MedicalRetrieval, DuRetrieval, T2Retrieval
- English: FiQA, HotpotQA, NQ

Compatibility
----
- Corpus, qrels, and original queries are kept consistent with MTEB (Massive Text Embedding Benchmark) formatting and conventions.

What’s provided (placeholders)
----
- Data layout and file format (corpus / qrels / queries)
- Train / dev / test splits and item counts
- Audio details (format, sampling rate)
- Evaluation scripts and baseline instructions
- License and citation information

Data format (summary)
----
- Corpus: documents/utterances in MTEB-compatible format
- Qrels: relevance labels in MTEB-style qrel format
- Queries: original query text (and/or speech) aligned with MTEB query format

Evaluation (placeholder)
----
- Standard retrieval metrics (e.g., Recall@K, MAP, MRR, nDCG@K)
- Optional end-to-end evaluation: report ASR metrics such as WER/CER alongside retrieval metrics

How to contribute / next steps
----
- Fill in the placeholders above: counts, sampling rates, exact file paths, license, and any restricted-access instructions.
- Add evaluation scripts and baseline results under `scripts/` or `tools/` when available.

License & contact
----
- License: TBD — please add the dataset/license details.
- Contact: TBD — please add maintainer contact information.
