import json
import numpy as np
import bm25s
from nltk.stem import PorterStemmer
import argparse
import os
import logging
import sys

# --- Anserini / Lucene Default Stopwords ---
LUCENE_STOPWORDS = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", 
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", 
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
]

# Initialize Stemmer
stemmer = PorterStemmer()

def setup_logger(log_file):
    """Configure Logger to output to both file and console"""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def nltk_stemmer_batch(tokens_list):
    """Wrapper function to adapt NLTK stemmer to the bm25s interface."""
    return [stemmer.stem(token) for token in tokens_list]

def load_data(data_dir, audio_path, logger, query_field, query_file):
    """Load data with paths dynamically joined based on data_dir"""
    corpus_file = os.path.join(data_dir, 'corpus.jsonl')
    queries_file = os.path.join(data_dir, audio_path, query_file)
    qrels_file = os.path.join(data_dir, 'qrels', 'test.jsonl')

    logger.info(f"Loading corpus from {corpus_file}...")
    corpus_ids = []
    corpus_texts = []
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                corpus_ids.append(str(doc.get('_id')))
                corpus_texts.append((doc.get('title', "") + " " + doc.get("text", "")).strip())
    except FileNotFoundError:
        logger.error(f"Corpus file not found: {corpus_file}")
        sys.exit(1)

    logger.info(f"Loading queries from {queries_file}...")
    query_ids = []
    query_texts = []
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                q = json.loads(line)
                query_ids.append(str(q.get('_id')))
                query_texts.append(q.get(query_field, ""))
    except FileNotFoundError:
        logger.error(f"Queries file not found: {queries_file}")
        sys.exit(1)

    logger.info(f"Loading qrels from {qrels_file}...")
    qrels = {}
    try:
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                rel = json.loads(line)
                qid = str(rel.get('query-id') or rel.get('query_id') or rel.get('_id'))
                cid = str(rel.get('corpus-id') or rel.get('corpus_id') or rel.get('doc_id'))
                score = float(rel.get('score', 1))
                if qid not in qrels: qrels[qid] = {}
                qrels[qid][cid] = score
    except FileNotFoundError:
        logger.error(f"Qrels file not found: {qrels_file}")
        sys.exit(1)
            
    return corpus_ids, corpus_texts, query_ids, query_texts, qrels

def ndcg_at_k(r, k, ideal_r):
    """Calculate NDCG@K"""
    def dcg(scores):
        scores = np.asfarray(scores)[:k]
        if scores.size:
            return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
        return 0.0
    
    dcg_max = dcg(ideal_r)
    return dcg(r) / dcg_max if dcg_max else 0.0

def calc_mrr_at_k(retrieved_ids, relevant_dict, k=10):
    """Calculate MRR@K"""
    for rank, did in enumerate(retrieved_ids[:k]):
        # Consider relevant if score > 0
        if did in relevant_dict and relevant_dict[did] > 0:
            return 1.0 / (rank + 1)
    return 0.0

def calc_recall_at_k(retrieved_ids, relevant_dict, k=10):
    """Calculate Recall@K"""
    # Get all relevant document IDs (score > 0)
    total_relevant_ids = {did for did, score in relevant_dict.items() if score > 0}
    if not total_relevant_ids:
        return 0.0
    
    # Count relevant documents retrieved in the top K
    retrieved_set = set(retrieved_ids[:k])
    relevant_retrieved = len(total_relevant_ids.intersection(retrieved_set))
    
    return relevant_retrieved / len(total_relevant_ids)

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Run BM25s retrieval and evaluate NDCG, MRR, Recall.")
    
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to the dataset directory containing corpus.jsonl, queries.jsonl, etc.")
    parser.add_argument("--log_path", type=str, default="evaluation.log", 
                        help="Path to save the log file.")
    parser.add_argument("--ndcg_k", type=int, nargs="+", default=[10], 
                        help="List of K values for NDCG calculation (e.g., 10 20 100).")
    parser.add_argument("--audio_path", type=str, default="asr_result_cosy3", help="Sub-path for audio results")
    parser.add_argument("--query_field", type=str, default="text", 
                        help="The key name for the query text in the JSONL")
    parser.add_argument("--query_file", type=str, default="asr_result_cosy3.jsonl", 
                        help="The query filename")
    
    args = parser.parse_args()

    # 2. Setup Logger
    logger = setup_logger(args.log_path)
    logger.info(f"Arguments: {args}")

    # Determine max_k for retrieval
    # Must cover all NDCG K values, and at least 10 for Recall@10 and MRR@10
    max_k = max(max(args.ndcg_k), 10)
    logger.info(f"Retrieval will use top_k={max_k} to satisfy all metric requests.")

    # 3. Load Data
    corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_data(
        args.data_dir, args.audio_path, logger, args.query_field, args.query_file
    )
    
    # 4. Tokenize Corpus
    logger.info("Tokenizing corpus (Anserini settings, PorterStemmer)...")
    corpus_tokens = bm25s.tokenize(
        corpus_texts, 
        stemmer=nltk_stemmer_batch 
    )

    # 5. Build Index
    logger.info(f"Indexing with k1=0.9, b=0.4, method='lucene'...")
    retriever = bm25s.BM25(method='lucene', k1=0.9, b=0.4)
    retriever.index(corpus_tokens)

    # 6. Process Queries and Retrieval
    logger.info("Processing queries and Retrieving...")
    
    valid_indices = [i for i, qid in enumerate(query_ids) if qid in qrels]
    valid_query_ids = [query_ids[i] for i in valid_indices]
    valid_query_texts = [query_texts[i] for i in valid_indices]

    if not valid_query_ids:
        logger.error("No valid queries found with qrels!")
        return

    # Tokenize Queries
    query_tokens = bm25s.tokenize(
        valid_query_texts, 
        stopwords=LUCENE_STOPWORDS, 
        stemmer=nltk_stemmer_batch
    )

    # Retrieval
    results, scores = retriever.retrieve(query_tokens, k=max_k)
    
    # 7. Calculate Metrics
    logger.info("Calculating Metrics...")
    
    final_metrics = {}

    # --- Calculate NDCG (Supports multiple K values) ---
    for k_val in args.ndcg_k:
        ndcg_scores = []
        for i, qid in enumerate(valid_query_ids):
            doc_indices = results[i][:k_val]
            retrieved_ids = [corpus_ids[idx] for idx in doc_indices]
            
            retrieved_rel = [qrels[qid].get(did, 0) for did in retrieved_ids]
            ideal_rel = sorted(qrels[qid].values(), reverse=True)
            
            ndcg_scores.append(ndcg_at_k(retrieved_rel, k_val, ideal_rel))
        
        mean_ndcg = np.mean(ndcg_scores)
        final_metrics[f"NDCG@{k_val}"] = mean_ndcg
        logger.info(f"NDCG@{k_val}: {mean_ndcg:.4f}")

    # --- Calculate MRR@10 and Recall@10 ---
    mrr_scores = []
    recall_scores = []
    
    for i, qid in enumerate(valid_query_ids):
        # Get top 10 results for Recall/MRR calculation
        doc_indices_10 = results[i][:10]
        retrieved_ids_10 = [corpus_ids[idx] for idx in doc_indices_10]
        
        # MRR@10 for single query
        mrr_scores.append(calc_mrr_at_k(retrieved_ids_10, qrels[qid], k=10))
        
        # Recall@10 for single query
        recall_scores.append(calc_recall_at_k(retrieved_ids_10, qrels[qid], k=10))

    mean_mrr = np.mean(mrr_scores)
    mean_recall = np.mean(recall_scores)

    final_metrics["MRR@10"] = mean_mrr
    final_metrics["Recall@10"] = mean_recall

    logger.info(f"MRR@10: {mean_mrr:.4f}")
    logger.info(f"Recall@10: {mean_recall:.4f}")

    logger.info("="*40)
    logger.info("Final Results:")
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("="*40)

if __name__ == "__main__":
    main()