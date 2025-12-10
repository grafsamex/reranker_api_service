import os
import logging
from typing import List
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

# Загружаем модель один раз при старте
logger.info(f"Loading model from: {MODEL_NAME_OR_PATH}")
logger.info(f"GPU available: {USE_GPU}")
logger.info(f"Using cache directory: {HF_CACHE_DIR}")

# Проверяем, нужно ли принудительно использовать только локальные файлы
FORCE_LOCAL_FILES = os.getenv("FORCE_LOCAL_FILES", "false").lower() == "true"
if FORCE_LOCAL_FILES:
    logger.info("FORCE_LOCAL_FILES is set to true, will only use local files")

# Используем стандартный механизм transformers - он сам найдет модель в кэше
# если переменные окружения установлены правильно
try:
    logger.info(f"Loading tokenizer from: {MODEL_NAME_OR_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, 
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        local_files_only=FORCE_LOCAL_FILES
    )
    logger.info("Tokenizer loaded successfully")
    
    logger.info(f"Loading model from: {MODEL_NAME_OR_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_OR_PATH, 
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        local_files_only=FORCE_LOCAL_FILES
    )
    logger.info("Model loaded successfully")
except Exception as e:
    if FORCE_LOCAL_FILES:
        logger.error(f"Failed to load model from cache: {e}", exc_info=True)
        raise FileNotFoundError(
            f"Model not found in cache at {HF_CACHE_DIR}. "
            f"Error: {str(e)}"
        )
    else:
        logger.warning(f"Failed to load from cache, will try to download: {e}")
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