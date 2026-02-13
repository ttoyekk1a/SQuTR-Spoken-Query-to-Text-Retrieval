import argparse
import json
import os
import logging
from typing import List, Optional, Union
import numpy as np
import torch
import soundfile as sf
import librosa
from tqdm import tqdm

from mteb import MTEB
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder

import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    raise ImportError("qwen_omni_utils not found")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioTextDualModel(AbsEncoder):
    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        device: str = None,
        text_prefix: str = "passage: ", # 代码二中默认使用了这个前缀
        dtype: str = "bf16", 
        max_length: int = 8192, # 根据显存调整
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        # Precision settings
        dtype_mapping = {
            "fp32": torch.float32, 
            "fp16": torch.float16, 
            "bf16": torch.bfloat16
        }
        self.torch_dtype = dtype_mapping.get(dtype, torch.bfloat16)
        
        logger.info(f"Initializing Omni-Embed Model from: {model_path}")
        logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")

        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            # load processor below
        
            
        except Exception as e:
            raise RuntimeError(f"Omni-Embed model loading failed: {e}")
            
        self.text_prefix = text_prefix

        # processor and default kwargs
        self.videos_kwargs = {
            "min_pixels": 32 * 14 * 14,
            "max_pixels": 64 * 28 * 28,
            "use_audio_in_video": False,
        }
        self.text_kwargs = {
            "truncation": True,
            "padding": True,
            "max_length": 1024, 
        }
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def _process_batch_omni(self, documents: List[List[dict]]) -> np.ndarray:
        documents_texts = self.processor.apply_chat_template(documents, add_generation_prompt=False, tokenize=False)
        audio, images, videos = process_mm_info(documents, use_audio_in_video=False)
        # print("="*100)
        # print(audio, images, videos, documents_texts)
        # print("="*100)
        videos_kwargs = {
            "min_pixels": 32*14*14,
            "max_pixels": 64*28*28,
            "use_audio_in_video": False,
        }
        text_kwargs = {
            "truncation": True,
            "padding": True,
            "max_length": 1024 * 3,
        }
        batch_dict = self.processor(
            text=documents_texts, 
            images=images, 
            videos=videos, 
            audio=audio,
            return_tensors="pt",
            text_kwargs=text_kwargs,
            videos_kwargs=videos_kwargs,
            audio_kwargs={"max_length": 2048000},
        )

        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            
            # Average Pooling
            attention_mask = batch_dict["attention_mask"]
            # expand mask dim: [B, L] -> [B, L, 1]
            last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            
            # Normalize
            embedding = F.normalize(embedding, dim=-1)
            
        return embedding.to(torch.float32).cpu().numpy()

    def _encode_audio(self, audio_paths: List[str]) -> np.ndarray:
        """Encode audio inputs (expects List[List[dict]] structure)."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(audio_paths), self.batch_size), desc="Encoding Audio"):
            batch_paths = audio_paths[i : i + self.batch_size]
            
            # build documents structure required by Omni
            batch_documents = []
            for path in batch_paths:
                # 注意：这里必须是 [[{...}]] 的结构
                batch_documents.append([
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio",
                                "audio": path
                            },
                        ]
                    }
                ])
            embeddings = self._process_batch_omni(batch_documents)
            all_embeddings.append(embeddings)
            
        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text inputs, applying optional text prefix."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding Text"):
            batch_texts = texts[i : i + self.batch_size]
            
            batch_documents = []
            for text in batch_texts:
                # apply prefix
                final_text = f"{self.text_prefix}{text}" if self.text_prefix else text
                
                batch_documents.append([
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": final_text
                            }
                        ]
                    }
                ])
            
            embeddings = self._process_batch_omni(batch_documents)
            all_embeddings.append(embeddings)
            
        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])

    def encode(
        self,
        inputs: Union[List[str], List[dict]],
        *,
        task_metadata: Optional[TaskMetadata] = None,
        **kwargs,
    ) -> np.ndarray:
        """Unified MTEB encode entry point."""
        if not inputs:
            return np.array([])
        
        # preprocess inputs and flatten batch
        input_data = []
        for batch in inputs:
            if isinstance(batch, dict) and "text" in batch:
                # MTEB 有时会传入 {"text": ["...", "..."]}
                if isinstance(batch["text"], list):
                    input_data.extend(batch["text"])
                else:
                    input_data.append(batch["text"])
            elif isinstance(batch, str):
                input_data.append(batch)
            else:
                input_data.append(str(batch))

        if not input_data:
            return np.array([])

        # simple check whether inputs are audio (by file suffix)
        first_item = input_data[0]
        is_audio_input = False
        
        # suffix check (can be improved)
        if isinstance(first_item, str) and len(first_item) < 1024: 
            if first_item.strip().lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                 is_audio_input = True
        
        if is_audio_input:
            logger.info(f"Detected Audio input mode (Batch size: {len(input_data)})")
            return self._encode_audio(input_data)
        else:
            # logger.info(f"Detected Text input mode (Batch size: {len(input_data)})")
            return self._encode_text(input_data)


# --- 任务类：MTEB 数据加载 (保持原样，逻辑正确) ---
class CustomAudioRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DynamicPlaceHolder", 
        description="End-to-End Speech to Text Retrieval Evaluation",
        reference=None,
        type="Retrieval",
        category="t2t", 
        modalities=["text", "text"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-12-31"),
        domains=["Spoken"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found", 
        bibtex_citation="",
        dataset = {"path": "","revision": ""}, 
        eval_langs=["eng-Latn"], 
    )

    def __init__(self, data_dir, dataset_name, query_field="wav_path", query_file="", audio_path="",  **kwargs):
        super().__init__(**kwargs)
        self.metadata.name = dataset_name
        self.audio_dir = audio_path
        self.corpus_path = os.path.join(data_dir, "corpus.jsonl")
        self.query_path = os.path.join(audio_path, query_file) 
        self.qrels_path = os.path.join(data_dir, "qrels/test.jsonl")
        self.query_field = query_field
        
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
                full_text = f"{title}\n{text}".strip() if title else text
                self.corpus[split][doc_id] = {"text": full_text}

        logger.info(f"Loading queries (Audio Paths) from: {self.query_path}")
        with open(self.query_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                q_id = str(item.get("_id", item.get("id")))
                wav_path = item.get(self.query_field, "")
                if not wav_path:
                    wav_path = item.get("audio", item.get("path", item.get("wav", "")))
                wav_path = os.path.join(self.audio_dir, wav_path)
                self.queries[split][q_id] = wav_path

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
    parser = argparse.ArgumentParser(description="Omni-Embed MTEB Evaluation")
    
    # path args
    parser.add_argument("--data_dir_path", type=str, required=True, help="Root directory of the dataset (containing corpus.jsonl, qrels/)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Omni-Embed model")
    
    # audio-specific args
    parser.add_argument("--audio_path", type=str, required=True, help="Base directory where audio files are stored")
    parser.add_argument("--query_file", type=str, default="queries.jsonl", help="Filename of the query jsonl (inside audio_path)")
    parser.add_argument("--query_field", type=str, default="wav_path", help="Key name for wav filename in query jsonl")
    
    # runtime args
    parser.add_argument("--log_path", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size (Suggest small for Audio)")
    parser.add_argument("--text_prefix", type=str, default="passage: ", help="Prefix for text documents (default: 'passage: ')")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Model precision")

    args = parser.parse_args()
    
    # prepare paths
    norm_path = os.path.normpath(args.data_dir_path)
    dataset_name = os.path.basename(norm_path)
    model_name_simple = os.path.basename(os.path.normpath(args.model_path))
    final_output_folder = os.path.join(args.log_path, dataset_name, model_name_simple)
    os.makedirs(final_output_folder, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info(f"Task: {dataset_name}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output: {final_output_folder}")
    logger.info("=" * 50)

    # initialize model
    model = AudioTextDualModel(
        model_path=args.model_path,
        batch_size=args.batch_size,
        dtype=args.dtype,
        text_prefix=args.text_prefix,
    )

    # quick sanity check
    try:
        logger.info("Running sanity check...")
        test_text = model.encode([{"text": "Hello World"}])
        logger.info(f"Sanity check passed. Text embedding shape: {test_text.shape}")
    except Exception as e:
        logger.error(f"Sanity check failed: {e}")
        return

    # initialize MTEB task
    task = CustomAudioRetrieval(
        data_dir=args.data_dir_path,
        dataset_name=dataset_name,
        query_field=args.query_field,
        audio_path=args.audio_path,
        query_file=args.query_file
    )

    # run evaluation
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
