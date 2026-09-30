import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ---------------------------------------------------------
# Document Status
# ---------------------------------------------------------

DocumentStatus = Literal[
    "uploaded",
    "queued",
    "processing",
    "ready",
    "failed",
]


# ---------------------------------------------------------
# Document Response
# ---------------------------------------------------------

class DocumentResponse(BaseModel):

    id: uuid.UUID

    user_id: uuid.UUID
    knowledge_base_id: uuid.UUID

    original_name: str

    mime_type: str | None

    file_size: int | None

    storage_provider: str

    storage_key: str

    checksum: str | None

    status: DocumentStatus

    chunk_count: int

    error_message: str | None

    extraction_metadata: dict[str, Any]

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


# =========================================================
# DOCUMENT DETAIL RESPONSE
# Used for GET /documents/{document_id}
# =========================================================

class DocumentDetailResponse(BaseModel):

    model_config = ConfigDict(
        from_attributes=True
    )

    id: uuid.UUID

    knowledge_base_id: uuid.UUID

    user_id: uuid.UUID

    # -----------------------------------------------------
    # File information
    # -----------------------------------------------------

    file_name: str

    content_type: str | None = None

    file_size: int | None = None

    checksum: str | None = None

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    storage_key: str

    # -----------------------------------------------------
    # Processing
    # -----------------------------------------------------

    status: str

    error_message: str | None = None

    # -----------------------------------------------------
    # Extraction / indexing
    # -----------------------------------------------------

    chunk_count: int = 0

    page_count: int | None = None

    extraction_metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    # -----------------------------------------------------
    # Timestamps
    # -----------------------------------------------------

    created_at: datetime

    updated_at: datetime

    processed_at: datetime | None = None

# ---------------------------------------------------------
# Document Upload Response
# ---------------------------------------------------------

class DocumentUploadResponse(BaseModel):

    document_id: uuid.UUID

    knowledge_base_id: uuid.UUID

    filename: str

    status: DocumentStatus = "uploaded"

    message: str = "Document uploaded successfully"


# ---------------------------------------------------------
# Document Status Response
# ---------------------------------------------------------

class DocumentStatusResponse(BaseModel):

    document_id: uuid.UUID

    status: DocumentStatus

    chunk_count: int = 0

    error_message: str | None = None


# ---------------------------------------------------------
# Document List
# ---------------------------------------------------------

class DocumentListResponse(BaseModel):

    total: int

    items: list[DocumentResponse]

    page: int = 1
    page_size: int = 20


# ---------------------------------------------------------
# Document Reprocess Request
# ---------------------------------------------------------

class DocumentReprocessRequest(BaseModel):

    force: bool = False

    override_chunk_size: int | None = Field(
        default=None,
        ge=100,
        le=10000,
    )

    override_chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        le=2000,
    )


# ---------------------------------------------------------
# Document Processing Result
# ---------------------------------------------------------

class DocumentProcessingResult(BaseModel):

    document_id: uuid.UUID

    status: DocumentStatus

    total_chunks: int = 0

    qdrant_collection: str | None = None

    processing_time_ms: int | None = None

    error_message: str | None = None