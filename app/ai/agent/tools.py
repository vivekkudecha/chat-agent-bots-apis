import logging
from typing import Any, Callable
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool import Tool, BotTool
from app.repositories.tool_repository import ToolRepository

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Manages available tools, schema serialization, and tool execution.
    Designed for seamless future tool integrations (LangChain/LangGraph-compatible).
    """

    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register_handler(self, name: str, handler: Callable):
        self._handlers[name] = handler

    @staticmethod
    def to_openai_schema(tool: Tool) -> dict[str, Any]:
        """
        Converts Tool model to OpenAI function calling schema.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or tool.display_name,
                "parameters": tool.input_schema or {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    async def get_active_bot_tools(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """
        Fetches enabled tools for a bot and formats them as tool calling schemas.
        """
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
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Executes a registered tool handler.
        """
        handler = self._handlers.get(name)
        if not handler:
            logger.warning("No handler registered for tool: %s", name)
            return {"error": f"Tool '{name}' is not implemented yet."}

        try:
            return await handler(arguments, context or {})
        except Exception as exc:
            logger.exception("Execution failed for tool '%s': %s", name, exc)
            return {"error": f"Tool execution failed: {str(exc)}"}
