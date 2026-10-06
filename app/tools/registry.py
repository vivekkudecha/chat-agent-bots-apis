from __future__ import annotations

import logging
from typing import Any, Callable
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool import Tool
from app.repositories.tool_repository import ToolRepository
from app.tools.base import BaseTool
from app.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central Tool Registry and Dispatcher.
    Maintains modular, pluggable tools for AI agents and executes them on demand.
    Supports both built-in Python tools and database-backed dynamic tools.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._handlers: dict[str, Callable] = {}

        # Register default built-in tools
        self.register(WebSearchTool())

    def register(self, tool: BaseTool) -> None:
        """
        Registers a pluggable tool into the registry.
        """
        self._tools[tool.name] = tool
        self._handlers[tool.name] = tool.execute
        logger.debug("Registered tool: %s (%s)", tool.name, tool.display_name)

    def register_handler(self, name: str, handler: Callable) -> None:
        """
        Registers a raw async handler function for a tool name.
        """
        self._handlers[name] = handler

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_schema(self, name: str) -> dict[str, Any] | None:
        tool = self.get_tool(name)
        if tool:
            return tool.to_openai_schema()
        return None

    def get_web_search_schema(self) -> dict[str, Any]:
        """
        Quick helper to return the schema for web search.
        """
        tool = self.get_tool("web_search")
        if tool:
            return tool.to_openai_schema()
        return WebSearchTool().to_openai_schema()

    @staticmethod
    def to_openai_schema(tool: Any) -> dict[str, Any]:
        """
        Converts Tool model or BaseTool to OpenAI function calling schema.
        """
        if hasattr(tool, "to_openai_schema"):
            return tool.to_openai_schema()

        # Fallback for SQLAlchemy Tool model
        return {
            "type": "function",
            "function": {
                "name": getattr(tool, "name", "unknown"),
                "description": getattr(tool, "description", None)
                or getattr(tool, "display_name", ""),
                "parameters": getattr(tool, "input_schema", None)
                or {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    async def get_active_bot_tools(
        self,
        db: AsyncSession | None,
        *,
        bot_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """
        Fetches enabled tools for a bot from the database and formats them as schemas.
        """
        if db is None:
            return []

        bot_tools = await ToolRepository.list_for_bot(
            db,
            bot_id=bot_id,
            enabled_only=True,
        )

        tools: list[dict[str, Any]] = []
        for bt in bot_tools:
            if bt.tool and bt.tool.is_active:
                tools.append(self.to_openai_schema(bt.tool))

        return tools

    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any] | str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Executes a registered tool or handler by name.
        """
        # 1. Check registered BaseTool
        tool = self._tools.get(name)
        if tool:
            try:
                return await tool.execute(arguments, context or {})
            except Exception as exc:
                logger.exception("Execution failed for tool '%s': %s", name, exc)
                return {"error": f"Tool '{name}' execution failed: {str(exc)}"}

        # 2. Check standalone handler
        handler = self._handlers.get(name)
        if handler:
            try:
                return await handler(arguments, context or {})
            except Exception as exc:
                logger.exception("Execution failed for handler '%s': %s", name, exc)
                return {"error": f"Tool execution failed: {str(exc)}"}

        logger.warning("No handler or tool registered for: %s", name)
        return {"error": f"Tool '{name}' is not registered or implemented."}


_registry_instance: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance
