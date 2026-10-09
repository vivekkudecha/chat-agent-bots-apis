from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------
    APP_NAME: str = "AI Bot Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    API_V1_PREFIX: str = "/api/v1"

    # ---------------------------------------------------------
    # PostgreSQL
    # ---------------------------------------------------------
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/ai_bot"
    )

    DB_ECHO: bool = False

    # ---------------------------------------------------------
    # Qdrant
    # ---------------------------------------------------------
    QDRANT_URL: str = "http://localhost:6333"

    QDRANT_API_KEY: str | None = None

    QDRANT_TIMEOUT_SECONDS: int = 30

    QDRANT_COLLECTION: str = "knowledge_chunks"

    # Must match the embedding model.
    EMBEDDING_DIMENSION: int = 768

    # ---------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------
    EMBEDDING_PROVIDER: str = "local"

    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    EMBEDDING_CONCURRENCY: int = 4

    # ---------------------------------------------------------
    # LLM & vLLM
    # ---------------------------------------------------------
    LLM_PROVIDER: str = "vllm"

    LLM_TIMEOUT_SECONDS: int = 120

    VLLM_BASE_URL: str = "http://localhost:8001/v1"

    VLLM_API_KEY: str = "local-vllm"

    VLLM_DEFAULT_MODEL: str = "gemma4:e4b"

    VLLM_TIMEOUT: int = 120

    LLM_VERIFY_SSL: bool = True

    LLM_TRUST_ENV: bool = True

    # ---------------------------------------------------------
    # Redis
    # ---------------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379/0"

    # ---------------------------------------------------------
    # Celery
    # ---------------------------------------------------------
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"

    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    CELERY_TASK_SOFT_TIME_LIMIT: int = 900

    CELERY_TASK_TIME_LIMIT: int = 1200

    # ---------------------------------------------------------
    # File Storage
    # ---------------------------------------------------------
    STORAGE_PROVIDER: str = "local"

    LOCAL_STORAGE_PATH: str = "./storage"

    MAX_UPLOAD_SIZE_MB: int = 200

    # ---------------------------------------------------------
    # OCR / PyMuPDF Settings
    # ---------------------------------------------------------
    TESSDATA_PREFIX: str = "./data/tessdata"

    OCR_LANGUAGE: str = "eng+hin+spa+fra+deu+guj+chi_sim+ara"

    OCR_DPI: int = 150

    # ---------------------------------------------------------
    # Language & Internationalization
    # ---------------------------------------------------------
    DEFAULT_LANGUAGE: str = "en"

    QDRANT_TEXT_TOKENIZER: str = "multilingual"

    RAG_STOP_WORDS: str | None = None

    # ---------------------------------------------------------
    # Authentication
    # ---------------------------------------------------------
    JWT_SECRET_KEY: str = "change-me-in-production"

    JWT_ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ---------------------------------------------------------
    # RAG defaults
    # ---------------------------------------------------------
    DEFAULT_CHUNK_SIZE: int = 800

    DEFAULT_CHUNK_OVERLAP: int = 100

    DEFAULT_TOP_K: int = 5

    DEFAULT_SCORE_THRESHOLD: float = 0.40

    RAG_MAX_CONTEXT_TOKENS: int = 1500

    RAG_HYBRID_SEARCH: bool = True

    # Hybrid (dense + BM25 sparse) index. Bumping the version moves new
    # writes and reads to "<QDRANT_COLLECTION>_v<N>"; reindex afterwards
    # with `python -m scripts.reindex_documents`.
    RAG_INDEX_VERSION: int = 2

    # int8 scalar quantization keeps ~4x more vectors in RAM (rescored
    # with full vectors), needed for 100k+ chunk collections.
    QDRANT_QUANTIZATION: bool = True

    QDRANT_ON_DISK_VECTORS: bool = False

    # Average chunk length (tokens) used for BM25 length normalisation.
    RAG_BM25_AVG_DOC_TOKENS: int = 160

    # Ingestion
    EMBEDDING_BATCH_SIZE: int = 32

    # Prefix applied to search queries (not documents). Empty string
    # disables it; unset picks a default for instruction-tuned models.
    EMBEDDING_QUERY_INSTRUCTION: str | None = None

    RAG_INGEST_PAGE_BATCH: int = 20

    RAG_UPSERT_BATCH_SIZE: int = 128

    DOCUMENT_TASK_SOFT_TIME_LIMIT: int = 3 * 3600

    DOCUMENT_TASK_TIME_LIMIT: int = 3 * 3600 + 300

    # Retrieval
    RAG_CANDIDATE_POOL: int = 40

    RAG_NEIGHBOR_WINDOW: int = 1

    RAG_MAX_PER_DOCUMENT: int = 3

    # Keep a below-threshold dense hit when this share of query terms
    # appears verbatim (codes, IDs, names).
    RAG_LEXICAL_MIN_COVERAGE: float = 0.6

    # Optional cross-encoder, e.g. "BAAI/bge-reranker-v2-m3".
    RAG_RERANKER_MODEL: str | None = None

    RAG_RERANK_THRESHOLD: float = 0.2

    # Agentic retrieval (grade evidence, refine query, retrieve again)
    RAG_AGENTIC: bool = True

    RAG_MAX_ROUNDS: int = 2

    RAG_GRADER_MODEL: str | None = None

    RAG_GRADER_PASSAGE_CHARS: int = 600

    # Sent with internal structured LLM calls (grading). "none" turns off
    # thinking on Ollama reasoning models such as gemma4/qwen3, which
    # otherwise spend the whole token budget reasoning. Empty disables.
    LLM_INTERNAL_REASONING_EFFORT: str | None = "none"

    # ---------------------------------------------------------
    # Guardrails
    # ---------------------------------------------------------
    ENABLE_INPUT_GUARDRAILS: bool = True

    ENABLE_OUTPUT_GUARDRAILS: bool = True

    # ---------------------------------------------------------
    # Pydantic Settings
    # ---------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()