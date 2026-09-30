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

BotStatus = Literal[
    "draft",
    "active",
    "disabled",
    "archived",
]

BotVisibility = Literal[
    "private",
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
        alias="metadata",
        serialization_alias="metadata",
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )


# ---------------------------------------------------------
# Create Bot
# ---------------------------------------------------------

class BotCreate(BotBase):

    slug: str = Field(
        ...,
        min_length=2,
        max_length=150,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
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

class BotUpdate(BaseModel):

    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    description: str | None = None

    visibility: BotVisibility | None = None

    avatar_url: str | None = None

    is_api_enabled: bool | None = None

    metadata_: dict[str, Any] | None = Field(
        default=None,
        alias="metadata",
        serialization_alias="metadata",
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )


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
# Detailed Bot Response
# ---------------------------------------------------------

class BotDetailResponse(BotResponse):

    versions: list[BotVersionResponse] = Field(
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