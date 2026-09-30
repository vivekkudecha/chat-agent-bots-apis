import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


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
        alias="metadata",
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
        alias="metadata",
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
        alias="metadata",
        serialization_alias="metadata",
    )

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


# ---------------------------------------------------------
# Conversation Response
# ---------------------------------------------------------

class ConversationResponse(BaseModel):

    id: uuid.UUID

    user_id: uuid.UUID
    bot_id: uuid.UUID

    title: str | None

    metadata_: dict[str, Any] = Field(
        alias="metadata",
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