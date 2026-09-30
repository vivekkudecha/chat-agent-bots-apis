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

GuardrailType = Literal[
    "input",
    "retrieval",
    "tool",
    "output",
]

GuardrailAction = Literal[
    "block",
    "redact",
    "warn",
    "log",
]


# ---------------------------------------------------------
# Guardrail Base
# ---------------------------------------------------------

class GuardrailBase(BaseModel):

    code: str = Field(
        ...,
        min_length=2,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    name: str = Field(
        ...,
        min_length=2,
        max_length=150,
    )

    description: str | None = None

    guardrail_type: GuardrailType

    handler: str = Field(
        ...,
        min_length=2,
        max_length=255,
    )

    default_config: dict[str, Any] = Field(
        default_factory=dict,
    )


# ---------------------------------------------------------
# Create Guardrail
# ---------------------------------------------------------

class GuardrailCreate(GuardrailBase):
    pass


# ---------------------------------------------------------
# Update Guardrail
# ---------------------------------------------------------

class GuardrailUpdate(BaseModel):

    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    description: str | None = None

    handler: str | None = None

    default_config: dict[str, Any] | None = None

    is_active: bool | None = None


# ---------------------------------------------------------
# Guardrail Response
# ---------------------------------------------------------

class GuardrailResponse(GuardrailBase):

    id: uuid.UUID

    is_active: bool

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


# ---------------------------------------------------------
# Guardrail List
# ---------------------------------------------------------

class GuardrailListResponse(BaseModel):

    total: int

    items: list[GuardrailResponse]

    page: int = 1
    page_size: int = 20


# ---------------------------------------------------------
# Attach Guardrail to Bot
# ---------------------------------------------------------

class BotGuardrailAttach(BaseModel):

    guardrail_id: uuid.UUID

    config: dict[str, Any] = Field(
        default_factory=dict,
    )

    action: GuardrailAction = "block"

    priority: int = Field(
        default=100,
        ge=0,
        le=10000,
    )

    is_enabled: bool = True


# ---------------------------------------------------------
# Update Bot Guardrail
# ---------------------------------------------------------

class BotGuardrailUpdate(BaseModel):

    config: dict[str, Any] | None = None

    action: GuardrailAction | None = None

    priority: int | None = Field(
        default=None,
        ge=0,
        le=10000,
    )

    is_enabled: bool | None = None


# ---------------------------------------------------------
# Bot Guardrail Response
# ---------------------------------------------------------

class BotGuardrailResponse(BaseModel):

    bot_id: uuid.UUID
    guardrail_id: uuid.UUID

    config: dict[str, Any]

    action: GuardrailAction

    priority: int

    is_enabled: bool

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


class BotGuardrailCreate(BaseModel):

    guardrail_id: uuid.UUID

    action: GuardrailAction = "block"

    priority: int = Field(
        default=100,
        ge=0,
        le=10_000,
    )

    is_enabled: bool = True

    config: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


# =========================================================
# Guardrail Test
# =========================================================

class GuardrailExecutionResponse(BaseModel):
    code: str
    handler: str
    action: str
    passed: bool
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class GuardrailTestRequest(BaseModel):
    stage: str = Field(
        ...,
        pattern="^(input|retrieval|tool|output)$",
    )
    text: str = Field(
        ...,
        min_length=1,
    )


class GuardrailTestResponse(BaseModel):
    allowed: bool
    original_text: str
    final_text: str
    warnings: list[str] = Field(default_factory=list)
    executions: list[GuardrailExecutionResponse] = Field(default_factory=list)

