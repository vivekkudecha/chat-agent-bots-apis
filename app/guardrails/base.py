from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuardrailStage(str, Enum):
    INPUT = "input"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    OUTPUT = "output"


class GuardrailAction(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"
    ALLOW = "allow"


@dataclass
class GuardrailContext:
    """
    Runtime context passed to guardrails.

    Do not put secrets, API keys or raw credentials here.
    """

    user_id: str | None = None
    bot_id: str | None = None
    conversation_id: str | None = None

    stage: GuardrailStage = GuardrailStage.INPUT

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class GuardrailResult:
    """
    Result returned by an individual guardrail.
    """

    passed: bool

    guardrail_code: str

    message: str | None = None

    # Used when a guardrail transforms/redacts text.
    modified_text: str | None = None

    # Additional safe diagnostic information.
    details: dict[str, Any] = field(
        default_factory=dict
    )


class BaseGuardrail(ABC):

    code: str

    @abstractmethod
    async def evaluate(
        self,
        *,
        text: str,
        config: dict[str, Any],
        context: GuardrailContext,
    ) -> GuardrailResult:
        """
        Evaluate text against this guardrail.

        Implementations should avoid mutating external state.
        """
        raise NotImplementedError