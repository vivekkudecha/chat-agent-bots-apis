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

    # ---------------------------------------------------------
    # LLM & vLLM
    # ---------------------------------------------------------
    LLM_PROVIDER: str = "vllm"

    LLM_TIMEOUT_SECONDS: int = 120

    VLLM_BASE_URL: str = "http://localhost:8001/v1"

    VLLM_API_KEY: str = "local-vllm"

    VLLM_DEFAULT_MODEL: str = "google/gemma-3-27b-it"

    VLLM_TIMEOUT: int = 120

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

    OCR_LANGUAGE: str = "eng"

    OCR_DPI: int = 150

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