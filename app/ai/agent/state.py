import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RouteType(str, Enum):
    DIRECT = "direct"
    RAG = "rag"
    TOOL = "tool"


@dataclass
class AgentState:
    """
    State passed across agent graph nodes (Router -> Retriever/Tool -> Generator).
    """

    user_id: uuid.UUID
    bot_id: uuid.UUID
    conversation_id: uuid.UUID
    message: str

    route: RouteType = RouteType.DIRECT
    route_reason: str = ""

    has_kb: bool = False
    available_tools: list[dict[str, Any]] = field(default_factory=list)

    retrieval: Any | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    response_content: str = ""
