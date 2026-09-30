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

ToolType = Literal[
    "builtin",
    "http",
    "mcp",
]


# ---------------------------------------------------------
# Tool Base
# ---------------------------------------------------------

class ToolBase(BaseModel):

    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    display_name: str = Field(
        ...,
        min_length=2,
        max_length=150,
    )

    description: str | None = None

    tool_type: ToolType

    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
        }
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
    )

    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
    )


# ---------------------------------------------------------
# Create Tool
# ---------------------------------------------------------

class ToolCreate(ToolBase):
    pass


# ---------------------------------------------------------
# Update Tool
# ---------------------------------------------------------

class ToolUpdate(BaseModel):

    display_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    description: str | None = None

    input_schema: dict[str, Any] | None = None

    config: dict[str, Any] | None = None

    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        le=300,
    )

    is_active: bool | None = None


# ---------------------------------------------------------
# Tool Response
# ---------------------------------------------------------

class ToolResponse(ToolBase):

    id: uuid.UUID

    is_active: bool

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


# ---------------------------------------------------------
# Tool List
# ---------------------------------------------------------

class ToolListResponse(BaseModel):

    total: int

    items: list[ToolResponse]

    page: int = 1
    page_size: int = 20


# ---------------------------------------------------------
# Attach Tool to Bot
# ---------------------------------------------------------

class BotToolAttach(BaseModel):

    tool_id: uuid.UUID

    config: dict[str, Any] = Field(
        default_factory=dict,
    )

    requires_confirmation: bool = False

    is_enabled: bool = True


# ---------------------------------------------------------
# Update Bot Tool
# ---------------------------------------------------------

class BotToolUpdate(BaseModel):

    config: dict[str, Any] | None = None

    requires_confirmation: bool | None = None

    is_enabled: bool | None = None


# ---------------------------------------------------------
# Bot Tool Response
# ---------------------------------------------------------

class BotToolResponse(BaseModel):

    bot_id: uuid.UUID
    tool_id: uuid.UUID

    config: dict[str, Any]

    requires_confirmation: bool

    is_enabled: bool

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )