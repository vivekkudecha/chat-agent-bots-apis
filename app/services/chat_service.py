import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ai.guardrails import (
    GuardrailStage,
)

from app.core.exceptions import (
    BotNotFoundException,
    ConversationNotFoundException,
    ModelExecutionException,
    ModelNotFoundException,
)

from app.ai.llm import (
    LLMProvider,
    get_llm_provider,
)

from app.repositories.bot_repository import (
    BotRepository,
)
from app.repositories.conversation_repository import (
    ConversationRepository,
)
from app.repositories.ai_model_repository import (
    AIModelRepository,
)
from app.repositories.usage_repository import (
    UsageRepository,
)
from app.repositories.audit_repository import (
    AuditRepository,
)
from app.repositories.knowledge_repository import (
    KnowledgeRepository,
)

from app.ai.agent import (
    AgentRouter,
    ChatAgentGraph,
    ChatAgentState,
    RouteType,
    ToolRegistry,
)
from app.ai.guardrails import (
    GuardrailService,
)
from app.ai.rag import (
    RetrievalResult,
    RetrievalService,
)
from app.ai.llm import (
    PromptBuilderService,
)
from app.ai.memory import (
    MemoryManager,
)


@dataclass
class ChatSource:
    document_id: uuid.UUID
    knowledge_base_id: uuid.UUID

    file_name: str | None
    page: int | None

    score: float

    pages: list[int] = field(default_factory=list)
    chunk_count: int = 1
    content_preview: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResult:
    conversation_id: uuid.UUID
    message_id: uuid.UUID

    content: str

    model: str

    input_tokens: int
    output_tokens: int
    total_tokens: int

    latency_ms: int

    sources: list[ChatSource]

    warnings: list[str]


class ChatService:

    def __init__(
        self,
        *,
        llm: LLMProvider | None = None,
    ):

        self.llm = (
            llm
            or get_llm_provider()
        )

        self.guardrails = (
            GuardrailService()
        )

        self.retrieval = (
            RetrievalService()
        )

        self.prompt_builder = (
            PromptBuilderService()
        )

        self.router = (
            AgentRouter(
                llm=self.llm
            )
        )

        self.tools = (
            ToolRegistry()
        )

        self.memory = (
            MemoryManager(
                llm=self.llm,
            )
        )

        self.graph = (
            ChatAgentGraph(
                router=self.router,
                retrieval_service=self.retrieval,
                prompt_builder=self.prompt_builder,
                llm_provider=self.llm,
                tool_registry=self.tools,
                guardrail_service=self.guardrails,
            )
        )

    # =====================================================
    # CHAT
    # =====================================================

    async def chat(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        bot_id: uuid.UUID,
        message: str,
        conversation_id: uuid.UUID | None = None,
    ) -> ChatResult:

        started_at = time.perf_counter()

        # ---------------------------------------------
        # Bot
        # ---------------------------------------------

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        # ---------------------------------------------
        # Active bot version
        # ---------------------------------------------

        bot_version = (
            await BotRepository
            .get_active_version(
                db,
                bot_id=bot.id,
            )
        )

        if not bot_version:
            raise BotNotFoundException(
                "Bot has no active version."
            )

        # ---------------------------------------------
        # Conversation
        # ---------------------------------------------

        conversation = (
            await self._resolve_conversation(
                db,
                user_id=user_id,
                bot_id=bot.id,
                conversation_id=(
                    conversation_id
                ),
            )
        )

        # ---------------------------------------------
        # INPUT GUARDRAILS
        # ---------------------------------------------

        input_evaluation = (
            await self.guardrails.evaluate(
                db,
                bot_id=bot.id,
                user_id=user_id,
                conversation_id=(
                    conversation.id
                ),
                stage=(
                    GuardrailStage.INPUT
                ),
                text=message,
            )
        )

        safe_user_message = (
            input_evaluation.final_text
        )

        # ---------------------------------------------
        # Persist user message
        #
        # We store the original message here.
        # ---------------------------------------------

        user_db_message = (
            await ConversationRepository
            .create_message(
                db,
                conversation_id=(
                    conversation.id
                ),
                role="user",
                content=message,
                metadata={
                    "guardrail_warnings": (
                        input_evaluation.warnings
                    ),
                    "sanitized_for_model": (
                        safe_user_message
                        != message
                    ),
                },
            )
        )

        await db.commit()

        # ---------------------------------------------
        # Chat Memory Context (Working memory, episodic summary, temporal)
        # ---------------------------------------------

        memory_context = await self.memory.build_context(
            db,
            conversation=conversation,
            current_message_id=user_db_message.id,
        )

        history = memory_context.working_history

        # ---------------------------------------------
        # Model configuration
        # ---------------------------------------------

        model_config = (
            await BotRepository
            .get_primary_model_config(
                db,
                bot_id=bot.id,
            )
        )

        model = None
        if model_config:
            model = (
                await AIModelRepository.get_by_id(
                    db,
                    model_config.model_id,
                )
            )

        if not model or not model.is_active:
            default_model_key = getattr(
                settings,
                "VLLM_DEFAULT_MODEL",
                "llama3.2:latest",
            )
            model = await AIModelRepository.get_by_key(
                db,
                provider=settings.LLM_PROVIDER,
                model_key=default_model_key,
            )
            if not model:
                model = await AIModelRepository.create(
                    db,
                    provider=settings.LLM_PROVIDER,
                    model_key=default_model_key,
                    display_name=default_model_key,
                    model_type="chat",
                )
                await db.commit()

        temperature = (
            model_config.temperature
            if model_config
            else 0.7
        )
        top_p = (
            model_config.top_p
            if model_config
            else 1.0
        )
        max_tokens = (
            model_config.max_tokens
            if model_config
            else 2048
        )

        # ---------------------------------------------
        # Tool & Knowledge Base Availability
        # ---------------------------------------------

        active_tools = (
            await self.tools.get_active_bot_tools(
                db,
                bot_id=bot.id,
            )
        )

        kb_links = (
            await KnowledgeRepository.list_for_bot(
                db,
                bot.id,
            )
        )
        has_kb = bool(kb_links)

        # ---------------------------------------------
        # LangGraph Workflow Execution
        # (Router -> RAG / Tools / Direct -> Generator)
        # ---------------------------------------------
        try:
            graph_output = await self.graph.run(
                {
                    "user_id": user_id,
                    "bot_id": bot.id,
                    "conversation_id": conversation.id,
                    "query": safe_user_message,
                    "bot_version": bot_version,
                    "history": history,
                    "memory_context": memory_context,
                    "has_kb": has_kb,
                    "available_tools": active_tools,
                    "model_key": model.model_key,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "db_session": db,
                }
            )

            response = graph_output["response"]
            safe_retrieval = graph_output.get("retrieval")
            route_val = graph_output.get("route", RouteType.DIRECT.value)
            route = RouteType(route_val)

        except Exception as exc:

            await self._record_failure(
                db,
                user_id=user_id,
                bot_id=bot.id,
                conversation_id=(
                    conversation.id
                ),
                model_id=model.id,
                model_name=model.model_key,
            )

            if isinstance(
                exc,
                ModelExecutionException,
            ):
                raise

            raise ModelExecutionException() from exc

        # ---------------------------------------------
        # OUTPUT GUARDRAILS
        # ---------------------------------------------

        output_evaluation = (
            await self.guardrails.evaluate(
                db,
                bot_id=bot.id,
                user_id=user_id,
                conversation_id=(
                    conversation.id
                ),
                stage=(
                    GuardrailStage.OUTPUT
                ),
                text=response.content,
            )
        )

        safe_output = (
            output_evaluation.final_text
        )

        # ---------------------------------------------
        # Latency
        # ---------------------------------------------

        latency_ms = int(
            (
                time.perf_counter()
                - started_at
            )
            * 1000
        )

        # ---------------------------------------------
        # Save assistant message
        # ---------------------------------------------

        assistant_message = (
            await ConversationRepository
            .create_message(
                db,
                conversation_id=(
                    conversation.id
                ),
                role="assistant",
                content=safe_output,
                model_id=model.id,
                input_tokens=(
                    response
                    .usage
                    .input_tokens
                ),
                output_tokens=(
                    response
                    .usage
                    .output_tokens
                ),
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": (
                        response.finish_reason
                    ),
                    "source_count": (
                        len(
                            safe_retrieval.chunks
                        )
                        if (
                            safe_retrieval
                            and safe_retrieval.chunks
                        )
                        else 0
                    ),
                    "guardrail_warnings": (
                        output_evaluation
                        .warnings
                    ),
                },
            )
        )

        # ---------------------------------------------
        # Usage
        # ---------------------------------------------

        await UsageRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            conversation_id=(
                conversation.id
            ),
            model_id=model.id,
            provider=model.provider,
            model_name=model.model_key,
            operation="chat",
            input_tokens=(
                response
                .usage
                .input_tokens
            ),
            output_tokens=(
                response
                .usage
                .output_tokens
            ),
            estimated_cost=Decimal("0"),
            latency_ms=latency_ms,
            status="success",
        )

        # ---------------------------------------------
        # Audit
        # ---------------------------------------------

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="conversation",
            resource_id=conversation.id,
            action="chat.completed",
            status="success",
            metadata={
                "model": model.model_key,
                "source_count": (
                    len(
                        safe_retrieval.chunks
                    )
                    if (
                        safe_retrieval
                        and safe_retrieval.chunks
                    )
                    else 0
                ),
                "input_guardrail_count": len(
                    input_evaluation.executions
                ),
                "output_guardrail_count": len(
                    output_evaluation.executions
                ),
            },
        )

        await db.commit()

        # ---------------------------------------------
        # Background Memory Compaction (Async)
        # ---------------------------------------------

        total_msgs = memory_context.total_conversation_messages + 2
        if self.memory.should_compact(conversation, total_msgs):
            try:
                from app.workers.memory_tasks import compact_conversation_task
                compact_conversation_task.delay(str(conversation.id), model=model.model_key)
            except Exception:
                # If Celery worker/broker is unreachable, fail silently
                pass

        # ---------------------------------------------
        # Sources (Distinct by Document / Reference File)
        # ---------------------------------------------

        sources = []
        if safe_retrieval and safe_retrieval.chunks:
            distinct_sources = (
                safe_retrieval.get_distinct_sources()
            )
            sources = [
                ChatSource(
                    document_id=src.document_id,
                    knowledge_base_id=(
                        src.knowledge_base_id
                    ),
                    file_name=src.file_name,
                    page=src.page,
                    score=src.score,
                    pages=src.pages,
                    chunk_count=src.chunk_count,
                    content_preview=(
                        src.content_preview
                    ),
                    metadata=src.metadata,
                )
                for src in distinct_sources
            ]

        warnings = [
            *input_evaluation.warnings,
            *output_evaluation.warnings,
        ]

        input_tokens = (
            response.usage.input_tokens
        )
        output_tokens = (
            response.usage.output_tokens
        )
        total_tokens = (
            response.usage.total_tokens
            if response.usage.total_tokens > 0
            else (input_tokens + output_tokens)
        )

        return ChatResult(
            conversation_id=(
                conversation.id
            ),
            message_id=(
                assistant_message.id
            ),
            content=safe_output,
            model=model.model_key,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            sources=sources,
            warnings=warnings,
        )

    # =====================================================
    # CONVERSATION
    # =====================================================

    async def _resolve_conversation(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        bot_id: uuid.UUID,
        conversation_id: uuid.UUID | None,
    ):

        if conversation_id:

            conversation = (
                await ConversationRepository
                .get_owned(
                    db,
                    conversation_id=(
                        conversation_id
                    ),
                    user_id=user_id,
                )
            )

            if not conversation:
                raise ConversationNotFoundException()

            # Prevent using a conversation created
            # for another bot.
            if conversation.bot_id != bot_id:
                raise ConversationNotFoundException()

            return conversation

        conversation = (
            await ConversationRepository.create(
                db,
                user_id=user_id,
                bot_id=bot_id,
                title=None,
            )
        )

        await db.commit()
        await db.refresh(conversation)

        return conversation

    # =====================================================
    # RETRIEVAL GUARDRAILS
    # =====================================================

    async def _guard_retrieval(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        retrieval: RetrievalResult,
    ) -> RetrievalResult:

        if not retrieval.chunks:
            return retrieval

        safe_chunks = []

        for chunk in retrieval.chunks:

            evaluation = (
                await self.guardrails.evaluate(
                    db,
                    bot_id=bot_id,
                    user_id=user_id,
                    conversation_id=(
                        conversation_id
                    ),
                    stage=(
                        GuardrailStage.RETRIEVAL
                    ),
                    text=chunk.text,
                    metadata={
                        "document_id": str(
                            chunk.document_id
                        ),
                        "knowledge_base_id": str(
                            chunk.knowledge_base_id
                        ),
                        "chunk_id": chunk.id,
                    },
                )
            )

            chunk.text = (
                evaluation.final_text
            )

            safe_chunks.append(chunk)

        return RetrievalResult(
            query=retrieval.query,
            chunks=safe_chunks,
            knowledge_base_ids=(
                retrieval
                .knowledge_base_ids
            ),
            total_results=len(
                safe_chunks
            ),
        )

    # =====================================================
    # FAILURE LOGGING
    # =====================================================

    async def _record_failure(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        bot_id: uuid.UUID,
        conversation_id: uuid.UUID,
        model_id: uuid.UUID,
        model_name: str,
    ) -> None:

        await db.rollback()

        try:

            await UsageRepository.create(
                db,
                user_id=user_id,
                bot_id=bot_id,
                conversation_id=(
                    conversation_id
                ),
                model_id=model_id,
                provider="vllm",
                model_name=model_name,
                operation="chat",
                status="failed",
            )

            await AuditRepository.create(
                db,
                user_id=user_id,
                bot_id=bot_id,
                resource_type=(
                    "conversation"
                ),
                resource_id=(
                    conversation_id
                ),
                action="chat.failed",
                status="failed",
            )

            await db.commit()

        except Exception:
            await db.rollback()