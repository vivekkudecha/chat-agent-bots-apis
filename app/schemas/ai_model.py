import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)


# ---------------------------------------------------------
# Types
# ---------------------------------------------------------

ModelType = Literal[
    "chat",
    "embedding",
    "reranker",
]


# ---------------------------------------------------------
# AI Model Base
# ---------------------------------------------------------

class AIModelBase(BaseModel):

    provider: str = Field(
        ...,
        min_length=2,
        max_length=50,
    )

    model_key: str = Field(
        ...,
        min_length=1,
        max_length=255,
    )

    display_name: str = Field(
        ...,
        min_length=2,
        max_length=255,
    )

    model_type: ModelType = "chat"

    endpoint_url: str | None = None

    context_window: int | None = Field(
        default=None,
        ge=1,
    )

    capabilities: dict[str, Any] = Field(
        default_factory=dict,
    )


# ---------------------------------------------------------
# Register AI Model
# ---------------------------------------------------------

class AIModelCreate(AIModelBase):
    pass


# ---------------------------------------------------------
# Update AI Model
# ---------------------------------------------------------

class AIModelUpdate(BaseModel):

    display_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=255,
    )

    endpoint_url: str | None = None

    context_window: int | None = Field(
        default=None,
        ge=1,
    )

    capabilities: dict[str, Any] | None = None

    is_active: bool | None = None


# ---------------------------------------------------------
# AI Model Response
# ---------------------------------------------------------

class AIModelResponse(AIModelBase):

    id: uuid.UUID

    is_active: bool

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


# ---------------------------------------------------------
# Bot Model Configuration
# ---------------------------------------------------------

class BotModelConfigCreate(BaseModel):

    model_id: uuid.UUID

    temperature: float = Field(
        default=0.7,
        ge=0,
        le=2,
    )

    top_p: float = Field(
        default=1.0,
        gt=0,
        le=1,
    )

    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=131072,
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
    )

    is_primary: bool = True


# ---------------------------------------------------------
# Update Bot Model Configuration
# ---------------------------------------------------------

class BotModelConfigUpdate(BaseModel):

    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
    )

    top_p: float | None = Field(
        default=None,
        gt=0,
        le=1,
    )

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=131072,
    )

    config: dict[str, Any] | None = None

    is_primary: bool | None = None


# ---------------------------------------------------------
# Bot Model Configuration Response
# ---------------------------------------------------------

class BotModelConfigResponse(BaseModel):

    id: uuid.UUID

    bot_id: uuid.UUID
    model_id: uuid.UUID

    temperature: float
    top_p: float
    max_tokens: int

    config: dict[str, Any]

    is_primary: bool

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )