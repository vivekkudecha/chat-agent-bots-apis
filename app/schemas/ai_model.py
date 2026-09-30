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

    name: str | None = None

    model_type: ModelType = "chat"

    endpoint_url: str | None = None

    context_window: int | None = Field(
        default=None,
        ge=1,
    )

    capabilities: dict[str, Any] = Field(
        default_factory=dict,
    )

    @model_validator(mode="before")
    @classmethod
    def populate_display_name(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "name" in data and "display_name" not in data:
                data["display_name"] = data["name"]
            elif "display_name" in data and "name" not in data:
                data["name"] = data["display_name"]
        return data


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

    name: str | None = Field(
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

    @model_validator(mode="before")
    @classmethod
    def sync_name(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "name" in data and "display_name" not in data:
                data["display_name"] = data["name"]
        return data


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



# =========================================================
# AI MODEL LIST RESPONSE
# =========================================================

class AIModelListResponse(BaseModel):

    items: list[AIModelResponse] = Field(
        default_factory=list
    )

    total: int

    page: int = 1

    page_size: int = 20


# =========================================================
# BOT MODEL CONFIG REQUEST
# =========================================================

class BotModelConfigRequest(BaseModel):

    model_id: uuid.UUID

    is_primary: bool = True

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )

    top_p: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
    )

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
    )

    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
    )

    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
    )

    stop: list[str] = Field(
        default_factory=list,
        max_length=10,
    )

    seed: int | None = None

    extra_config: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_stop_sequences(self):

        for value in self.stop:

            if not value:
                raise ValueError(
                    "Stop sequence cannot be empty."
                )

            if len(value) > 500:
                raise ValueError(
                    "Stop sequence cannot exceed 500 characters."
                )

        return self