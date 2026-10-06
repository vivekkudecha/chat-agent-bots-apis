from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all pluggable agent tools.
    Subclasses define their schema, execution handler, and metadata.
    """

    name: str
    display_name: str
    description: str
    tool_type: str = "custom"
    input_schema: dict[str, Any]
    timeout_seconds: int = 30
    is_active: bool = True

    @abstractmethod
    async def execute(
        self,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Executes the tool with parsed arguments and optional runtime context.
        """
        pass

    def to_openai_schema(self) -> dict[str, Any]:
        """
        Serializes the tool into standard OpenAI function calling JSON schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or self.display_name,
                "parameters": self.input_schema
                or {
                    "type": "object",
                    "properties": {},
                },
            },
        }
