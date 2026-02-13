import argparse
import json
import os
import logging
import time
import torch
import numpy as np
from mteb import MTEB
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from sentence_transformers import SentenceTransformer

# ============================
# 1. Logging Configuration
# ============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================
# 2. Custom MTEB Retrieval Task
# ============================
class CustomRetrieval(AbsTaskRetrieval):
    """
    Custom retrieval task for evaluating local datasets.
    Metadata is required for MTEB registration.
    """
    metadata = TaskMetadata(
        name="CustomEval",
        description="Evaluation on custom local dataset",
        reference=None,
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-12-31"),
        domains=["Academic"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        dataset={"path": "", "revision": ""},
        eval_langs=["eng-Latn"],
    )

    def __init__(self, corpus_path, query_path, qrels_path, query_field, **kwargs):
        super().__init__(**kwargs)
        self.corpus_path = corpus_path
        self.query_path = query_path
        self.qrels_path = qrels_path
        self.query_field = query_field

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        split = "test"
        self.corpus = {split: {}}
        self.queries = {split: {}}
        self.relevant_docs = {split: {}}

        # 1. Load Corpus
        logger.info(f"Loading corpus from: {self.corpus_path}")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                doc_id = str(item.get("_id", item.get("id")))
                text = item.get("text", item.get("content", ""))
                title = item.get("title", "")
                self.corpus[split][doc_id] = {"text": text, "title": title}

        # 2. Load Queries
        logger.info(f"Loading queries from: {self.query_path}")
        with open(self.query_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                q_id = str(item.get("_id", item.get("id")))
                text = item.get(self.query_field, "")
                self.queries[split][q_id] = text

        # 3. Load Qrels
        logger.info(f"Loading qrels from: {self.qrels_path}")
        with open(self.qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                q_id = str(item.get("query-id", item.get("query_id")))
                c_id = str(item.get("corpus-id", item.get("corpus_id")))
                score = int(item.get("score", 1))
                
                if q_id not in self.relevant_docs[split]:
                    self.relevant_docs[split][q_id] = {}
                self.relevant_docs[split][q_id][c_id] = score

        self.data_loaded = True
        logger.info(f"Data loaded. Corpus: {len(self.corpus[split])}, Queries: {len(self.queries[split])}")

# ============================
# 3. Main Execution
# ============================
def main():
    parser = argparse.ArgumentParser(description="Embedding MTEB Evaluation")
    
    # Path Arguments
    parser.add_argument("--query_path", type=str, required=True, help="Path to queries.jsonl")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus.jsonl")
    parser.add_argument("--qrels_path", type=str, required=True, help="Path to qrels.jsonl")
    parser.add_argument("--model_path", type=str, default="infgrad/stella_en_1.5B_v5", help="Local path or HF ID")
    
    # Config Arguments
    parser.add_argument("--log_path", type=str, default="./results", help="Directory for MTEB results")
    parser.add_argument("--query_field", type=str, default="text", help="JSON key for query text")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    
    parser.add_argument("--instruction", type=str, 
                        default="",
                        help="Prompt prefix for queries. Set to empty string if not needed.")

    args = parser.parse_args()

    os.makedirs(args.log_path, exist_ok=True)
    
    task = CustomRetrieval(
        corpus_path=args.corpus_path,
        query_path=args.query_path,
        qrels_path=args.qrels_path,
        query_field=args.query_field
    )

    model_prompts = {
        "query": args.instruction,
        "document": "" 
    }
    
    logger.info(f"Loading model: {args.model_path}")
    logger.info(f"Batch size: {args.batch_size}, Max Length: {args.max_length}")
    logger.info(f"Using instruction: '{args.instruction}'")

    model_kwargs = {
        "device_map": "cuda", 
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "attn_implementation": "flash_attention_2"
    }
    if args.instruction:
        model = SentenceTransformer(
            args.model_path, 
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            prompts=model_prompts
        )
    else:
        model = SentenceTransformer(
            args.model_path, 
            trust_remote_code=True,
            model_kwargs=model_kwargs
        )

    # Sanity Check
    docs = ["Hello world", "Who are you?"]
    embeddings = model.encode(docs)
    logger.info(f"Sanity Check - Embedding Shape: {embeddings.shape}")
    
    time.sleep(5)

    # Run MTEB Evaluation
    evaluation = MTEB(tasks=[task])
    evaluation.run(
        model, 
        output_folder=args.log_path,
        overwrite_results=True
    )
    
    logger.info(f"Evaluation finished. Results saved to {args.log_path}")

if __name__ == "__main__":
    main()

