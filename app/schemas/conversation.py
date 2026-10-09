import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


def normalize_message_sources(raw_sources: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_sources, list):
        return []
    normalized: list[dict[str, Any]] = []
    for s in raw_sources:
        if not isinstance(s, dict):
            continue
        file_name = (
            s.get("file_name")
            or s.get("fileName")
            or s.get("title")
            or s.get("name")
            or "Document"
        )
        content_preview = (
            s.get("content_preview")
            or s.get("snippet")
            or s.get("excerpt")
            or ""
        )
        doc_id = s.get("document_id") or s.get("documentId") or s.get("id")
        kb_id = s.get("knowledge_base_id") or s.get("knowledgeBaseId")
        item = {
            "document_id": str(doc_id) if doc_id else None,
            "documentId": str(doc_id) if doc_id else None,
            "knowledge_base_id": str(kb_id) if kb_id else None,
            "knowledgeBaseId": str(kb_id) if kb_id else None,
            "file_name": file_name,
            "fileName": file_name,
            "title": file_name,
            "page": s.get("page"),
            "pages": s.get("pages") or ([] if s.get("page") is None else [s.get("page")]),
            "score": s.get("score") if s.get("score") is not None else 1.0,
            "chunk_count": s.get("chunk_count", 1),
            "content_preview": content_preview,
            "snippet": content_preview,
            "excerpt": content_preview,
            "url": s.get("url") or (s.get("metadata", {}) or {}).get("url"),
            "source_type": s.get("source_type") or (s.get("metadata", {}) or {}).get("source_type", "knowledge_base"),
            "metadata": s.get("metadata") if isinstance(s.get("metadata"), dict) else {},
        }
        normalized.append(item)
    return normalized


# ---------------------------------------------------------
# Types
# ---------------------------------------------------------

MessageRole = Literal[
    "system",
    "user",
    "assistant",
    "tool",
]


# ---------------------------------------------------------
# Create Conversation
# ---------------------------------------------------------

class ConversationCreate(BaseModel):

    bot_id: uuid.UUID

    title: str | None = Field(
        default=None,
        max_length=255,
    )

    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata_", "metadata"),
        serialization_alias="metadata",
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )


# ---------------------------------------------------------
# Update Conversation
# ---------------------------------------------------------

class ConversationUpdate(BaseModel):

    title: str | None = Field(
        default=None,
        max_length=255,
    )

    metadata_: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("metadata_", "metadata"),
        serialization_alias="metadata",
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )


# ---------------------------------------------------------
# Message Response
# ---------------------------------------------------------

class MessageResponse(BaseModel):

    id: uuid.UUID

    conversation_id: uuid.UUID

    role: MessageRole

    content: str

    model_id: uuid.UUID | None

    input_tokens: int | None
    output_tokens: int | None

    latency_ms: int | None

    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata_", "metadata"),
        serialization_alias="metadata",
    )

    sources: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    suggestions: list[str] = Field(
        default_factory=list,
    )

    needs_clarification: bool = False

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def sync_sources(self) -> "MessageResponse":
        existing_sources = self.sources or self.metadata_.get("sources", [])
        normalized = normalize_message_sources(existing_sources)
        if not normalized and self.metadata_.get("source_count", 0) > 0:
            count = self.metadata_.get("source_count", 1)
            normalized = [
                {
                    "document_id": None,
                    "documentId": None,
                    "knowledge_base_id": None,
                    "knowledgeBaseId": None,
                    "file_name": "Knowledge Document",
                    "fileName": "Knowledge Document",
                    "title": "Knowledge Document",
                    "page": None,
                    "pages": [],
                    "score": 1.0,
                    "chunk_count": 1,
                    "content_preview": "Source document cited during response generation.",
                    "snippet": "Source document cited during response generation.",
                    "excerpt": "Source document cited during response generation.",
                    "url": None,
                    "source_type": "knowledge_base",
                    "metadata": {},
                }
                for _ in range(count)
            ]
        self.sources = normalized
        if normalized or "sources" in self.metadata_:
            self.metadata_["sources"] = normalized
        stored = self.metadata_.get("suggestions")
        if not self.suggestions and isinstance(stored, list):
            self.suggestions = [str(item) for item in stored if item]
        self.needs_clarification = self.needs_clarification or bool(
            self.metadata_.get("needs_clarification")
        )
        return self


# ---------------------------------------------------------
# Conversation Response
# ---------------------------------------------------------

class ConversationResponse(BaseModel):

    id: uuid.UUID

    user_id: uuid.UUID
    bot_id: uuid.UUID

    title: str | None

    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata_", "metadata"),
        serialization_alias="metadata",
    )

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


# ---------------------------------------------------------
# Detailed Conversation
# ---------------------------------------------------------

class ConversationDetailResponse(ConversationResponse):

    messages: list[MessageResponse] = Field(
        default_factory=list,
    )


# ---------------------------------------------------------
# Conversation List
# ---------------------------------------------------------

class ConversationListResponse(BaseModel):

    total: int

    items: list[ConversationResponse]

    page: int = 1
    page_size: int = 20


# ---------------------------------------------------------
# Message List
# ---------------------------------------------------------

class MessageListResponse(BaseModel):

    total: int

    items: list[MessageResponse]

    page: int = 1
    page_size: int = 50