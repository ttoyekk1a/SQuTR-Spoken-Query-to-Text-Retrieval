import json
import numpy as np
import bm25s
import jieba  # Core change: Integration of Jieba for Chinese word segmentation
import argparse
import os
import logging
import sys

# --- Simplified Chinese Stopwords List ---
# Removing common particles like "de", "le", "shi" and punctuation 
# can significantly improve retrieval accuracy.
# CN_STOPWORDS = [
#     "的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "之", "用", "于", 
#     "把", "被", "在", "上", "下", "里", "个", "这", "那", "我", "你", "他", "她", "它", 
#     "，", "。", "？", "！", "、", "：", "；", "“", "”", "（", "）", "《", "》"
# ]

def setup_logger(log_file):
    """Configure the logger"""
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

# def chinese_splitter(text):
#     """
#     Custom tokenizer passed to bm25s.
#     Uses jieba.lcut for precise mode segmentation.
#     """
#     if not text:
#         return []
#     return jieba.lcut(text)

def load_data(data_dir, audio_path, logger, query_field, query_file):
    """Data loading logic"""
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
                doc_id = str(doc.get('id') or doc.get('_id'))
                corpus_ids.append(doc_id)
                # Concatenate title and text to enrich retrieval context
                corpus_texts.append((doc.get('title', "") + " " + doc.get("text", "")).strip())
    except FileNotFoundError:
        logger.error(f"Corpus file not found: {corpus_file}")
        sys.exit(1)

    logger.info(f"Loading queries from {queries_file}...")
    query_ids = []
    query_texts = []
    try:
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    q = json.loads(line)
                    query_ids.append(str(q.get('_id')))
                    query_texts.append(q.get(query_field, ""))
        except:
            with open(queries_file.replace(".jsonl", ".json"), 'r', encoding='utf-8') as f:
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
    # Get all relevant doc IDs (score > 0)
    total_relevant_ids = {did for did, score in relevant_dict.items() if score > 0}
    if not total_relevant_ids:
        return 0.0
    
    # Count relevant documents within top K results
    retrieved_set = set(retrieved_ids[:k])
    relevant_retrieved = len(total_relevant_ids.intersection(retrieved_set))
    
    return relevant_retrieved / len(total_relevant_ids)

def jieba_preprocess(texts, logger=None):
    """
    Preprocessing: Tokenize with Jieba and join tokens with spaces.
    Example: "NaturalLanguageProcessing" -> "Natural Language Processing"
    This allows bm25s's default split() to work correctly.
    """
    processed = []
    total = len(texts)
    for i, text in enumerate(texts):
        if not text:
            processed.append("")
            continue
        # Use Jieba precise mode
        seg_list = jieba.lcut(text) 
        # Join with spaces
        processed.append(" ".join(seg_list))
        
        if logger and i % 10000 == 0 and i > 0:
            logger.info(f"Pre-tokenized {i}/{total} docs...")
            
    return processed

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Run BM25s (Chinese) retrieval.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset.")
    parser.add_argument("--log_path", type=str, default="evaluation_cn.log", help="Log path.")
    parser.add_argument("--ndcg_k", type=int, nargs="+", default=[10], help="K values.")
    parser.add_argument("--audio_path", type=str, default="asr_result_cosy3", help="ASR result path")
    parser.add_argument("--query_field", type=str, default="text", help="Query field name")
    parser.add_argument("--query_file", type=str, default="asr_result_cosy3.jsonl", help="Query filename")
    args = parser.parse_args()

    # 2. Setup Logger
    logger = setup_logger(args.log_path)
    max_k = max(args.ndcg_k)
    
    # 3. Load Data
    corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_data(
        args.data_dir, args.audio_path, logger, args.query_field, args.query_file
    )
    
    # --- [Core Modification] 4. Tokenize Corpus ---
    logger.info("Pre-processing corpus with Jieba (step 1/2)...")
    # Manually tokenize and merge into space-separated strings
    corpus_texts_spaced = jieba_preprocess(corpus_texts, logger)
    
    logger.info("BM25s Tokenizing (step 2/2)...")
    # splitter parameter is not needed here as input is already space-delimited
    corpus_tokens = bm25s.tokenize(
        corpus_texts_spaced, 
        stemmer=None  # Explicitly disable stemming for Chinese
    )
    
    # 5. Build Index
    logger.info("Indexing...")
    retriever = bm25s.BM25(method='lucene', k1=0.9, b=0.4) 
    retriever.index(corpus_tokens)

    # 6. Process Queries
    logger.info("Processing queries...")
    valid_indices = [i for i, qid in enumerate(query_ids) if qid in qrels]
    valid_query_ids = [query_ids[i] for i in valid_indices]
    valid_query_texts = [query_texts[i] for i in valid_indices]

    if not valid_query_ids:
        logger.error("No valid queries!")
        return

    # Pre-process queries as well
    query_texts_spaced = jieba_preprocess(valid_query_texts)
    query_tokens = bm25s.tokenize(
        query_texts_spaced, 
        stemmer=None
    )

    # 7. Retrieval and Evaluation
    logger.info("Retrieving...")
    results, scores = retriever.retrieve(query_tokens, k=max_k)
    
    logger.info("Calculating Metrics...")
    final_metrics = {}
    for k_val in args.ndcg_k:
        ndcg_scores = []
        for i, qid in enumerate(valid_query_ids):
            doc_indices = results[i][:k_val]
            retrieved_ids = [corpus_ids[idx] for idx in doc_indices]
            
            # Calculate relevance scores
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
        # Use top 10 results for Recall/MRR
        doc_indices_10 = results[i][:10]
        retrieved_ids_10 = [corpus_ids[idx] for idx in doc_indices_10]
        
        mrr_scores.append(calc_mrr_at_k(retrieved_ids_10, qrels[qid], k=10))
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