import uuid
from datetime import datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ---------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------

class ChunkingConfig(BaseModel):
    strategy: str = "recursive"

    chunk_size: int = Field(
        default=800,
        ge=100,
        le=10000,
    )

    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=2000,
    )

    separators: list[str] = Field(
        default_factory=lambda: [
            "\n\n",
            "\n",
            ". ",
            " ",
        ]
    )

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        return self


# ---------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------

class RetrievalConfig(BaseModel):

    search_type: str = Field(
        default="semantic",
        pattern="^(semantic|keyword|hybrid)$",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    score_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    enable_reranking: bool = False

    reranker_model: str | None = None

    hybrid_alpha: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )


# ---------------------------------------------------------
# Knowledge Base Create
# ---------------------------------------------------------

class KnowledgeBaseCreate(BaseModel):

    name: str = Field(
        ...,
        min_length=2,
        max_length=150,
    )

    description: str | None = None

    embedding_model: str | None = None

    chunking_config: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
    )

    retrieval_config: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
    )


# ---------------------------------------------------------
# Knowledge Base Update
# ---------------------------------------------------------

class KnowledgeBaseUpdate(BaseModel):

    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    description: str | None = None

    chunking_config: ChunkingConfig | None = None

    retrieval_config: RetrievalConfig | None = None


# ---------------------------------------------------------
# Knowledge Base Response
# ---------------------------------------------------------

class KnowledgeBaseResponse(BaseModel):

    id: uuid.UUID
    user_id: uuid.UUID

    name: str
    description: str | None

    embedding_model: str | None

    chunking_config: dict[str, Any]
    retrieval_config: dict[str, Any]

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


# ---------------------------------------------------------
# Knowledge Base List
# ---------------------------------------------------------

class KnowledgeBaseListResponse(BaseModel):

    total: int

    items: list[KnowledgeBaseResponse]

    page: int = 1
    page_size: int = 20


# ---------------------------------------------------------
# Attach Knowledge Base to Bot
# ---------------------------------------------------------

class BotKnowledgeBaseAttach(BaseModel):

    knowledge_base_id: uuid.UUID

    priority: int = Field(
        default=0,
        ge=0,
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
    )


# ---------------------------------------------------------
# Bot Knowledge Base Response
# ---------------------------------------------------------

class BotKnowledgeBaseResponse(BaseModel):

    bot_id: uuid.UUID

    knowledge_base_id: uuid.UUID

    priority: int

    config: dict[str, Any]

    model_config = ConfigDict(
        from_attributes=True,
    )