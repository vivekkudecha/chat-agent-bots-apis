# import uuid
# from datetime import datetime
# from typing import Any, Literal

# from pydantic import (
#     BaseModel,
#     Field,
# )


# # ---------------------------------------------------------
# # Chat Request
# # ---------------------------------------------------------

# class ChatRequest(BaseModel):

#     message: str = Field(
#         ...,
#         min_length=1,
#         max_length=50000,
#     )

#     conversation_id: uuid.UUID | None = None

#     stream: bool = False

#     use_knowledge_base: bool = True

#     enable_tools: bool = True

#     metadata: dict[str, Any] = Field(
#         default_factory=dict,
#     )


# # ---------------------------------------------------------
# # RAG Source
# # ---------------------------------------------------------

# class ChatSource(BaseModel):

#     document_id: uuid.UUID

#     knowledge_base_id: uuid.UUID

#     file_name: str | None = None

#     chunk_index: int | None = None

#     page: int | None = None

#     score: float | None = None

#     content_preview: str | None = None


# # ---------------------------------------------------------
# # Tool Call
# # ---------------------------------------------------------

# class ChatToolCall(BaseModel):

#     id: str

#     tool_name: str

#     arguments: dict[str, Any] = Field(
#         default_factory=dict,
#     )

#     status: Literal[
#         "pending",
#         "executing",
#         "completed",
#         "failed",
#         "rejected",
#     ] = "pending"

#     result: Any | None = None

#     execution_time_ms: int | None = None


# # ---------------------------------------------------------
# # Token Usage
# # ---------------------------------------------------------

# class ChatUsage(BaseModel):

#     input_tokens: int = 0

#     output_tokens: int = 0

#     total_tokens: int = 0

#     latency_ms: int | None = None

#     estimated_cost: float | None = None


# # ---------------------------------------------------------
# # Chat Response
# # ---------------------------------------------------------

# class ChatResponse(BaseModel):

#     conversation_id: uuid.UUID

#     message_id: uuid.UUID

#     bot_id: uuid.UUID

#     response: str

#     model: str

#     sources: list[ChatSource] = Field(
#         default_factory=list,
#     )

#     tool_calls: list[ChatToolCall] = Field(
#         default_factory=list,
#     )

#     usage: ChatUsage = Field(
#         default_factory=ChatUsage,
#     )

#     created_at: datetime


# # ---------------------------------------------------------
# # Streaming Events
# # ---------------------------------------------------------

# class ChatStreamEvent(BaseModel):

#     event: Literal[
#         "start",
#         "token",
#         "source",
#         "tool_call",
#         "tool_result",
#         "usage",
#         "done",
#         "error",
#     ]

#     conversation_id: uuid.UUID | None = None

#     data: dict[str, Any] = Field(
#         default_factory=dict,
#     )


# # ---------------------------------------------------------
# # Guardrail Rejection
# # ---------------------------------------------------------

# class ChatGuardrailResponse(BaseModel):

#     allowed: bool = False

#     guardrail_code: str

#     action: str

#     message: str


import uuid
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# =========================================================
# REQUEST
# =========================================================

class ChatRequest(BaseModel):

    bot_id: uuid.UUID

    message: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
    )

    conversation_id: (
        uuid.UUID | None
    ) = None

    enable_web_search: bool = Field(
        default=False,
        description="Enable live web search tool calling for this chat query",
    )

    web_search: bool | None = Field(
        default=None,
        description="Alias for enable_web_search",
    )

    @property
    def is_web_search_enabled(self) -> bool:
        if self.web_search is not None:
            return self.web_search
        return self.enable_web_search


# =========================================================
# SOURCE
# =========================================================

class ChatSourceResponse(BaseModel):

    document_id: uuid.UUID

    knowledge_base_id: uuid.UUID | None = None

    file_name: str | None = None

    page: int | None = None

    pages: list[int] = Field(
        default_factory=list
    )

    score: float

    chunk_count: int = 1

    content_preview: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# =========================================================
# TOKEN USAGE
# =========================================================

class ChatUsageResponse(BaseModel):

    input_tokens: int = 0

    output_tokens: int = 0

    total_tokens: int = 0

    prompt_tokens: int = 0

    completion_tokens: int = 0


# =========================================================
# RESPONSE
# =========================================================

class ChatResponse(BaseModel):

    model_config = ConfigDict(
        from_attributes=True
    )

    conversation_id: uuid.UUID

    message_id: uuid.UUID

    content: str

    model: str

    usage: ChatUsageResponse

    latency_ms: int

    sources: list[
        ChatSourceResponse
    ] = Field(
        default_factory=list
    )

    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Details of any tools executed during the turn",
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Follow-up questions the knowledge base can answer; render as "
            "clickable options that send the text as the next message"
        ),
    )

    needs_clarification: bool = Field(
        default=False,
        description="True when the reply is a clarifying question instead of an answer",
    )