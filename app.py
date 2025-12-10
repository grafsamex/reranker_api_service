import os
import logging
from pathlib import Path
from typing import List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Настройка
MODEL_NAME_OR_PATH = os.getenv("MODEL_PATH", "BAAI/bge-reranker-v2-m3")
USE_GPU = torch.cuda.is_available()

# Настройка кэша для HuggingFace
# Используем HF_HOME если установлен, иначе стандартный путь
HF_CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or os.path.expanduser("~/.cache/huggingface")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BGEReranker")

# Проверяем наличие модели в кэше
def check_model_in_cache(model_name: str, cache_dir: str) -> Tuple[bool, str]:
    """Проверяет, есть ли модель в кэше. Возвращает (найдено, путь)"""
    # Если передан прямой путь к модели
    if os.path.isdir(model_name) or os.path.isfile(model_name):
        logger.info(f"Model path is a local directory/file: {model_name}")
        if os.path.exists(model_name):
            return True, model_name
    
    # HuggingFace хранит модели в структуре: cache_dir/hub/models--ORG--MODEL_NAME/
    model_slug = model_name.replace("/", "--")
    hub_path = Path(cache_dir) / "hub" / f"models--{model_slug}"
    
    if hub_path.exists():
        logger.info(f"Found model cache at: {hub_path}")
        # Проверяем наличие файлов модели в snapshots
        snapshots = list(hub_path.glob("snapshots/*"))
        if snapshots:
            # Берем последний snapshot
            latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found {len(snapshots)} snapshot(s) in cache, using: {latest_snapshot}")
            # Проверяем наличие основных файлов модели
            model_files = list(latest_snapshot.glob("*.bin")) + list(latest_snapshot.glob("*.safetensors"))
            config_file = latest_snapshot / "config.json"
            if model_files and config_file.exists():
                logger.info(f"Model files found: {len(model_files)} model file(s), config.json exists")
                return True, str(latest_snapshot)
            else:
                logger.warning(f"Model cache incomplete: {len(model_files)} model files, config exists: {config_file.exists()}")
    
    # Также проверяем старую структуру кэша (transformers < 4.20)
    old_cache_path = Path(cache_dir) / model_name.replace("/", "--")
    if old_cache_path.exists():
        logger.info(f"Found model in old cache structure at: {old_cache_path}")
        return True, str(old_cache_path)
    
    # Проверяем все возможные пути в кэше
    cache_path = Path(cache_dir)
    if cache_path.exists():
        logger.info(f"Cache directory exists: {cache_path}")
        # Ищем любые упоминания модели
        for item in cache_path.rglob("*"):
            if model_slug.lower() in item.name.lower() or model_name.split("/")[-1].lower() in item.name.lower():
                if item.is_dir() and (item / "config.json").exists():
                    logger.info(f"Found potential model directory: {item}")
    
    return False, ""

# Загружаем модель один раз при старте
logger.info(f"Loading model from: {MODEL_NAME_OR_PATH}")
logger.info(f"GPU available: {USE_GPU}")
logger.info(f"Using cache directory: {HF_CACHE_DIR}")

# Проверяем, нужно ли принудительно использовать только локальные файлы
FORCE_LOCAL_FILES = os.getenv("FORCE_LOCAL_FILES", "false").lower() == "true"
if FORCE_LOCAL_FILES:
    logger.info("FORCE_LOCAL_FILES is set to true, will only use local files")

# Проверяем наличие кэша
cache_exists, model_path = check_model_in_cache(MODEL_NAME_OR_PATH, HF_CACHE_DIR)
if cache_exists and model_path:
    logger.info(f"Model found in cache at: {model_path}, loading from local files")
    # Используем найденный путь или оригинальное имя модели
    load_path = model_path if os.path.isdir(model_path) else MODEL_NAME_OR_PATH
    local_files_only = True
else:
    if FORCE_LOCAL_FILES:
        logger.error("FORCE_LOCAL_FILES is true but model not found in cache!")
        raise FileNotFoundError(
            f"Model not found in cache at {HF_CACHE_DIR}. "
            f"Please ensure model is cached or set FORCE_LOCAL_FILES=false"
        )
    logger.warning("Model not found in cache, will download if needed")
    load_path = MODEL_NAME_OR_PATH
    local_files_only = False

try:
    logger.info(f"Loading tokenizer from: {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        load_path, 
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        local_files_only=local_files_only
    )
    logger.info("Tokenizer loaded successfully")
    
    logger.info(f"Loading model from: {load_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path, 
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        local_files_only=local_files_only
    )
    logger.info("Model loaded successfully")
except Exception as e:
    if local_files_only:
        logger.warning(f"Failed to load from cache ({model_path}), trying to download: {e}")
        logger.info("Attempting to download model from HuggingFace Hub...")
        # Если не удалось загрузить из кэша, пробуем скачать
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME_OR_PATH, 
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            local_files_only=False
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME_OR_PATH, 
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            local_files_only=False
        )
        logger.info("Model downloaded and loaded successfully")
    else:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise
model.eval()

if USE_GPU:
    model = model.cuda()
    logger.info("Model moved to GPU")

app = FastAPI(title="BGE Reranker v2 m3 API", version="1.0")

class RerankRequest(BaseModel):
    query: str = Field(..., description="Search query")
    passages: List[str] = Field(default=None, description="List of documents to rerank (alternative to 'documents')")
    documents: List[str] = Field(default=None, description="List of documents to rerank (alternative to 'passages')")
    top_k: int = Field(default=5, ge=1, description="Number of top results to return")
    
    def get_documents(self) -> List[str]:
        """Get documents list from either 'passages' or 'documents' field"""
        if self.passages is not None:
            return self.passages
        elif self.documents is not None:
            return self.documents
        else:
            raise ValueError("Either 'passages' or 'documents' must be provided")

class RerankResult(BaseModel):
    index: int = Field(..., description="Original index of the document")
    document: str = Field(..., description="Document text")
    score: float = Field(..., description="Relevance score")

class RerankResponse(BaseModel):
    results: List[RerankResult] = Field(..., description="Reranked results sorted by relevance")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model name")
    device: str = Field(..., description="Device used (cuda or cpu)")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model=MODEL_NAME_OR_PATH,
        device="cuda" if USE_GPU else "cpu"
    )

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    try:
        # Get documents from either 'passages' or 'documents' field
        documents = request.get_documents()
        
        if not documents:
            raise HTTPException(status_code=400, detail="Passages or documents list is empty")
        
        top_k = min(request.top_k, len(documents))
        pairs = [[request.query, p] for p in documents]

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            if USE_GPU:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs, return_dict=True)
            scores = outputs.logits.view(-1).float()

        # Проверяем, что количество scores соответствует количеству documents
        if len(scores) != len(documents):
            logger.error(f"Score count mismatch: {len(scores)} scores for {len(documents)} documents")
            raise HTTPException(
                status_code=500,
                detail=f"Model returned {len(scores)} scores for {len(documents)} documents"
            )

        # Формируем результат
        scored = [
            RerankResult(
                index=i,
                document=documents[i],
                score=float(scores[i].item())
            )
            for i in range(len(documents))
        ]
        scored.sort(key=lambda x: x.score, reverse=True)

        return RerankResponse(results=scored[:top_k])
    
    except HTTPException:
        # Пробрасываем HTTPException без изменений
        raise
    except Exception as e:
        logger.error(f"Error during reranking: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")