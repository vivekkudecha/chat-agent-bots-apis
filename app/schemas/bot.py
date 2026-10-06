import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
)


# ---------------------------------------------------------
# Types
# ---------------------------------------------------------

BotStatus = Literal[
    "draft",
    "active",
    "disabled",
    "archived",
]

BotVisibility = Literal[
    "private",
    "organization",
    "public",
]


# ---------------------------------------------------------
# Bot Base
# ---------------------------------------------------------

class BotBase(BaseModel):

    name: str = Field(
        ...,
        min_length=2,
        max_length=150,
    )

    description: str | None = None

    visibility: BotVisibility = "private"

    avatar_url: str | None = None

    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata_", "metadata"),
        serialization_alias="metadata",
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )


# ---------------------------------------------------------
# Create Bot
# ---------------------------------------------------------

class BotCreateRequest(BotBase):

    slug: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="Optional unique slug. If omitted, it will be automatically system generated.",
    )

    system_instruction: str = Field(
        ...,
        min_length=1,
        max_length=50000,
    )

    welcome_message: str | None = None

    conversation_starters: list[str] = Field(
        default_factory=list,
        max_length=10,
    )


# ---------------------------------------------------------
# Update Bot
# ---------------------------------------------------------

class BotUpdateRequest(BaseModel):

    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    description: str | None = None

    visibility: BotVisibility | None = None

    avatar_url: str | None = None

    is_api_enabled: bool | None = None

    system_instruction: str | None = Field(
        default=None,
        min_length=1,
        max_length=50000,
    )

    welcome_message: str | None = None

    conversation_starters: list[str] | None = Field(
        default=None,
        max_length=10,
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
# Aliases
# ---------------------------------------------------------

BotCreate = BotCreateRequest
BotUpdate = BotUpdateRequest



# ---------------------------------------------------------
# Change Bot Status
# ---------------------------------------------------------

class BotStatusUpdate(BaseModel):

    status: BotStatus


# ---------------------------------------------------------
# Bot Version Create
# ---------------------------------------------------------

class BotVersionCreate(BaseModel):

    system_instruction: str = Field(
        ...,
        min_length=1,
        max_length=50000,
    )

    welcome_message: str | None = None

    conversation_starters: list[str] = Field(
        default_factory=list,
        max_length=10,
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
    )


# ---------------------------------------------------------
# Bot Version Response
# ---------------------------------------------------------

class BotVersionResponse(BaseModel):

    id: uuid.UUID
    bot_id: uuid.UUID

    version: int

    system_instruction: str

    welcome_message: str | None

    conversation_starters: list[str]

    config: dict[str, Any]

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


# ---------------------------------------------------------
# Bot Response
# ---------------------------------------------------------

class BotResponse(BaseModel):

    id: uuid.UUID
    user_id: uuid.UUID

    name: str
    slug: str

    description: str | None

    status: BotStatus
    visibility: BotVisibility

    avatar_url: str | None

    is_api_enabled: bool

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
# Detailed Bot Response
# ---------------------------------------------------------

class BotDetailResponse(BotResponse):

    versions: list[BotVersionResponse] = Field(
        default_factory=list,
    )


class BotEditableDocument(BaseModel):

    id: uuid.UUID
    original_name: str
    file_size: int | None
    status: str
    chunk_count: int

    model_config = ConfigDict(
        from_attributes=True,
    )


class BotEditResponse(BotDetailResponse):

    knowledge_base_id: uuid.UUID | None = None
    documents: list[BotEditableDocument] = Field(
        default_factory=list,
    )


# ---------------------------------------------------------
# Bot List Response
# ---------------------------------------------------------

class BotListResponse(BaseModel):

    total: int

    items: list[BotResponse]

    page: int = 1

    page_size: int = 20


# ---------------------------------------------------------
# Bot Creation With Documents
# ---------------------------------------------------------

class UploadedDocumentSummary(BaseModel):

    id: uuid.UUID
    original_name: str
    file_size: int
    status: str
    chunk_count: int = 0
    error_message: str | None = None
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


class BotCreationSummary(BaseModel):

    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int


class BotWithDocumentsResponse(BotDetailResponse):

    knowledge_base_id: uuid.UUID | None = None
    knowledge_base_name: str | None = None
    documents: list[UploadedDocumentSummary] = Field(
        default_factory=list,
    )
    summary: BotCreationSummary | None = None