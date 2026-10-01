import time
import uuid
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.guardrails.base import (
    GuardrailStage,
)

from app.core.exceptions import (
    BotNotFoundException,
    ConversationNotFoundException,
    ModelExecutionException,
    ModelNotFoundException,
)

from app.integrations.llm import (
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

from app.services.guardrail_service import (
    GuardrailService,
)
from app.services.retrieval_service import (
    RetrievalResult,
    RetrievalService,
)
from app.services.prompt_builder_service import (
    PromptBuilderService,
)


@dataclass
class ChatSource:
    document_id: uuid.UUID
    knowledge_base_id: uuid.UUID

    file_name: str | None
    page: int | None

    score: float


@dataclass
class ChatResult:
    conversation_id: uuid.UUID
    message_id: uuid.UUID

    content: str

    model: str

    input_tokens: int
    output_tokens: int

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
        # Conversation history
        #
        # Important:
        # exclude the newly-created current user
        # message because PromptBuilder adds it
        # separately.
        # ---------------------------------------------

        history = (
            await ConversationRepository
            .get_recent_messages(
                db,
                conversation_id=(
                    conversation.id
                ),
                limit=21,
            )
        )

        history = [
            item
            for item in history
            if item.id != user_db_message.id
        ]

        history = history[-20:]

        # ---------------------------------------------
        # Retrieval
        # ---------------------------------------------

        retrieval = (
            await self.retrieval
            .retrieve_for_bot(
                db,
                user_id=user_id,
                bot_id=bot.id,
                query=safe_user_message,
                top_k=5,
            )
        )

        # ---------------------------------------------
        # RETRIEVAL GUARDRAILS
        #
        # Each retrieved chunk is untrusted.
        # ---------------------------------------------

        safe_retrieval = (
            await self._guard_retrieval(
                db,
                bot_id=bot.id,
                user_id=user_id,
                conversation_id=(
                    conversation.id
                ),
                retrieval=retrieval,
            )
        )

        # ---------------------------------------------
        # Prompt
        # ---------------------------------------------

        prompt = self.prompt_builder.build(
            bot_version=bot_version,
            user_message=(
                safe_user_message
            ),
            history=history,
            retrieval=safe_retrieval,
        )

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
        # LLM
        # ---------------------------------------------

        try:

            response = await self.llm.chat(
                messages=prompt.messages,
                model=model.model_key,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

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
                "source_count": len(
                    safe_retrieval.chunks
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
        # Sources
        # ---------------------------------------------

        sources = [
            ChatSource(
                document_id=chunk.document_id,
                knowledge_base_id=(
                    chunk.knowledge_base_id
                ),
                file_name=chunk.file_name,
                page=chunk.page,
                score=chunk.score,
            )
            for chunk
            in safe_retrieval.chunks
        ]

        warnings = [
            *input_evaluation.warnings,
            *output_evaluation.warnings,
        ]

        return ChatResult(
            conversation_id=(
                conversation.id
            ),
            message_id=(
                assistant_message.id
            ),
            content=safe_output,
            model=model.model_key,
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