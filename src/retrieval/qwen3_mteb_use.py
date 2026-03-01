import argparse
import json
import os
import logging
from typing import List, Optional, Union
import numpy as np
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI,OpenAI
from mteb import MTEB
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from transformers import AutoTokenizer
# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置区域 --- 

# 执行前需要运行
# sh scripts/retrieval/qwen3_embedding_0_6b.sh
# sh scripts/retrieval/qwen3_embedding_4b.sh
# sh scripts/retrieval/qwen3_embedding_8b.sh
MODEL_CONFIGS = {
    "0.6b": {
        "path": "./Qwen3-Embedding-0.6B",
        "base_url": "http://localhost:8000/v1"
    },
    "4b": {
        "path": "./Qwen3-Embedding-4B",
        "base_url": "http://localhost:8001/v1"
    },
    "8b": {
        "path": "./Qwen3-8B-Embedding",
        "base_url": "http://localhost:8002/v1"
    }
}

model_id = "./Qwen3-Embedding-0.6B"  # 或 HuggingFace ID，如 "qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

def truncate_prompt(prompt, max_length=32000):
    # 2. 将文本转为 Token ID
    if len(prompt) < max_length: # 粗略估计，英文通常 1 token ≈ 4 chars，中文 ≈ 1-2 chars
        return prompt
    tokens = tokenizer.encode(prompt)
    
    # 3. 检查并截断
    if len(tokens) > max_length:
        # 保留最后 max_length 个 token (通常保留结尾更重要)
        truncated_tokens = tokens[:max_length]
        # 解码回文本
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return prompt


class CustomModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        max_batch_size: int = 512,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_batch_size = max_batch_size
        
        logger.info(f"Initializing Client - URL: {self.base_url}, Model: {self.model_name}")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _embed(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        dimensionality: int | None = None,
    ) -> list[list[float]]:
        batches = [
            texts[i : i + self.max_batch_size]
            for i in range(0, len(texts), self.max_batch_size)
        ]

        all_embeddings = []
        iterator = tqdm(batches, desc="Embedding") if show_progress_bar else batches

        for batch in iterator:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error fetching embeddings: {e}")
                try:
                    logger.info("Retrying request...")
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                        encoding_format="float"
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                except Exception as e_retry: 
                    raise RuntimeError(f"Failed to get embeddings after retry: {e_retry}")

        embeddings_array = np.asarray(all_embeddings)
        if dimensionality and embeddings_array.shape[1] > dimensionality:
            embeddings_array = embeddings_array[:, :dimensionality]
        return embeddings_array

    def encode(
        self,
        inputs: Union[List[str], List[dict]],
        *,
        task_metadata: Optional[TaskMetadata] = None,
        prompt_type: Optional[str] = None,
        **kwargs,
    ):
        show_progress_bar = kwargs.pop("show_progress_bar", True)
        texts = []
        # if inputs and isinstance(inputs[0], dict):
        for batch in inputs:
            if "text" in batch:
                if isinstance(batch["text"], list):
                    texts.extend(batch["text"])
                else:
                    texts.append(batch["text"])
        texts = [truncate_prompt(t) for t in texts]
        return self._embed(texts, show_progress_bar=show_progress_bar)


class CustomRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DynamicPlaceHolder", # 这里的名字会在 __init__ 中被覆盖
        description="Evaluation on custom dataset",
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
        dataset = {"path": "","revision": ""}, 
        eval_langs=["eng-Latn"], 
    )

    def __init__(self, data_dir, dataset_name, args, query_field="text", **kwargs):
        super().__init__(**kwargs)
        
        # 动态修改任务名称，确保生成的 json 结果中 Task Name 正确
        self.metadata.name = dataset_name
        self.asr_result_file_name = args.asr_result_file_name
        # data_dir = Path(data_dir)
        parent = "/".join(data_dir.split("/")[:-1])
        # 自动构建路径 
        self.corpus_path = os.path.join(parent, "corpus.jsonl")

        self.query_path = os.path.join(data_dir, f"{self.asr_result_file_name}.jsonl")
        if not os.path.exists(self.query_path):
            self.query_path = os.path.join(data_dir, f"{self.asr_result_file_name}.json")

        self.qrels_path = os.path.join(parent, "qrels/test.jsonl")
        self.query_field = query_field
         
        # 检查文件是否存在 
        for p in [self.corpus_path, self.query_path, self.qrels_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Critical file not found: {p}")

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        split = "test"
        self.corpus = {split: {}}
        self.queries = {split: {}}
        self.relevant_docs = {split: {}}

        logger.info(f"Loading corpus from: {self.corpus_path}")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                doc_id = str(item.get("_id", item.get("id")))
                text = item.get("text", item.get("content", ""))
                title = item.get("title", "")
                self.corpus[split][doc_id] = {"text": text, "title": title}

        logger.info(f"Loading queries from: {self.query_path}")
        with open(self.query_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                q_id = str(item.get("_id", item.get("id")))
                query = item.get(self.query_field, "")
                
                # Qwen Prompt 模版
                task_description = 'Given a web search query, retrieve relevant passages that answer the query'
                prompt_template = f'Instruct: {task_description}\nQuery: {query}'
                
                self.queries[split][q_id] = prompt_template

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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Embedding MTEB Evaluation")
    
    # 核心参数
    parser.add_argument(
        "--data_dir_path", 
        type=str, 
        default="./Echo_Bench/en/fiqa/audio_noise_snr_0",
        help="Root directory of the dataset (containing corpus.jsonl, queries.jsonl, etc.)"
    )
    
    parser.add_argument(
        "--model_size", 
        type=str, 
        default="8b", 
        choices=["0.6b", "4b", "8b"], 
        help="Choose the model size (0.6b, 4b, 8b)"
    )
    
    # 可选配置 
    parser.add_argument("--log_path", type=str, default="./results_asr_ablation", help="Base directory to save MTEB results")
    parser.add_argument("--query_field", type=str, default="asr_text", help="Key name for query text in jsonl (default: text)")
    parser.add_argument("--batch_size", type=int, default=512, help="Inference batch size")
    parser.add_argument("--asr_result_file_name", type=str, default="asr_result_cosy3_paraformer", help="")

    args = parser.parse_args()

    # 1. 验证模型
    if args.model_size not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model size. Choices: {list(MODEL_CONFIGS.keys())}")
    
    selected_config = MODEL_CONFIGS[args.model_size]
    
    # 2. 自动推断 Dataset Name 

    norm_path = os.path.normpath(args.data_dir_path)
    dataset_name = norm_path.split("/")[-2] +"_"+ os.path.basename(norm_path)
    
    # 3. 构建结果路径: ./results/{dataset_name}/Qwen3-Embedding-{SIZE} 
    size_upper = args.model_size.upper() 
    formatted_model_name = f"Qwen3-Embedding-{size_upper}"
    final_output_folder = os.path.join(args.log_path, dataset_name, formatted_model_name, args.query_field, args.asr_result_file_name)
    
    os.makedirs(final_output_folder, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("=" * 10)
    logger.info(f"Dataset Name (Derived): {dataset_name}")
    logger.info(f"Data Directory: {args.data_dir_path}")
    logger.info(f"Model: {formatted_model_name} ({selected_config['base_url']})")
    logger.info(f"Output: {final_output_folder}")
    logger.info("=" * 10)
    logger.info("=" * 50)

    # 4. 初始化模型
    model = CustomModel(
        model_name=selected_config['path'],
        base_url=selected_config['base_url'],
        max_batch_size=args.batch_size
    )
    
    # 5. 初始化任务（传入文件夹路径即可）
    task = CustomRetrieval(
        data_dir=args.data_dir_path,
        dataset_name=dataset_name,
        args = args,
        query_field=args.query_field,
    )

    # 6. 运行
    evaluation = MTEB(tasks=[task])
    evaluation.run(
        model, 
        output_folder=final_output_folder,
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_results=True,
    )
    
    logger.info(f"Evaluation finished for {dataset_name}.")

if __name__ == "__main__":
    main()