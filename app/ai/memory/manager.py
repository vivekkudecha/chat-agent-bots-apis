from __future__ import annotations

import logging
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm.provider import LLMProvider, get_llm_provider
from app.ai.memory.schemas import ConversationSummary, MemoryContext, TemporalContext
from app.ai.memory.summarizer import ConversationSummarizer
from app.ai.memory.temporal import TemporalEngine
from app.models.conversation import Conversation, Message
from app.repositories.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Orchestrates short-term working memory, rolling episodic summary,
    and temporal awareness for a conversation.
    """

    WORKING_BUFFER_SIZE = 8       # Last 8 messages (4 turns) verbatim
    COMPACTION_THRESHOLD = 6      # Compact when >= 6 unsummarized messages exist outside buffer

    def __init__(
        self,
        *,
        llm: LLMProvider | None = None,
        working_buffer_size: int = WORKING_BUFFER_SIZE,
        compaction_threshold: int = COMPACTION_THRESHOLD,
    ):
        self.llm = llm or get_llm_provider()
        self.summarizer = ConversationSummarizer(llm=self.llm)
        self.working_buffer_size = working_buffer_size
        self.compaction_threshold = compaction_threshold

    async def build_context(
        self,
        db: AsyncSession,
        *,
        conversation: Conversation,
        current_message_id: uuid.UUID | None = None,
    ) -> MemoryContext:
        """
        Assembles working memory, conversation summary, and temporal context.
        """
        # 1. Load existing summary from conversation metadata
        meta = conversation.metadata_ or {}
        raw_summary = meta.get("summary")
        summary = ConversationSummary.from_dict(raw_summary) if raw_summary else None

        # 2. Fetch recent conversation messages
        # Fetch up to working_buffer_size + 2 to safely calculate temporal gap and history
        recent_messages = await ConversationRepository.get_recent_messages(
            db,
            conversation_id=conversation.id,
            limit=self.working_buffer_size + 5,
        )

        # Exclude the current newly-inserted user message if present
        if current_message_id:
            recent_messages = [
                m for m in recent_messages if m.id != current_message_id
            ]

        # 3. Calculate temporal context using the most recent previous message
        last_prev_message = recent_messages[-1] if recent_messages else None
        temporal = TemporalEngine.calculate(last_prev_message)

        # 4. Extract working history buffer
        working_history = recent_messages[-self.working_buffer_size :]

        return MemoryContext(
            conversation_id=conversation.id,
            temporal=temporal,
            summary=summary,
            working_history=working_history,
            total_conversation_messages=len(recent_messages),
        )

    def should_compact(
        self,
        conversation: Conversation,
        total_messages_count: int,
    ) -> bool:
        """
        Determines whether background compaction should be triggered.
        """
        meta = conversation.metadata_ or {}
        raw_summary = meta.get("summary")
        summarized_count = raw_summary.get("total_summarized_messages", 0) if raw_summary else 0

        # Uncompacted messages eligible for compaction (excluding active working buffer)
        eligible_count = total_messages_count - self.working_buffer_size
        uncompacted_count = eligible_count - summarized_count

        return uncompacted_count >= self.compaction_threshold

    async def compact_conversation(
        self,
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        model: str | None = None,
    ) -> ConversationSummary | None:
        """
        Executes rolling compaction of unsummarized messages older than the working buffer.
        """
        conversation = await ConversationRepository.get_by_id(db, conversation_id)
        if not conversation:
            return None

        # Load all messages for conversation in chronological order
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        all_messages = list(result.scalars().all())

        if len(all_messages) <= self.working_buffer_size:
            return None

        # Messages to keep in raw working buffer
        messages_to_keep = all_messages[-self.working_buffer_size :]
        older_messages = all_messages[: -self.working_buffer_size]

        meta = dict(conversation.metadata_ or {})
        existing_summary = ConversationSummary.from_dict(meta.get("summary"))

        # Find where to start compacting
        last_id = existing_summary.last_summarized_message_id
        start_idx = 0
        if last_id:
            for idx, msg in enumerate(older_messages):
                if str(msg.id) == str(last_id):
                    start_idx = idx + 1
                    break

        uncompacted_slice = older_messages[start_idx:]
        if not uncompacted_slice:
            return existing_summary

        # Generate updated summary
        new_summary = await self.summarizer.summarize_or_update(
            messages_to_compact=uncompacted_slice,
            existing_summary=existing_summary,
            model=model,
        )

        meta["summary"] = new_summary.to_dict()
        conversation.metadata_ = meta
        await db.commit()
        await db.refresh(conversation)

        logger.info(
            "Compacted %d messages for conversation %s. Total summarized: %d",
            len(uncompacted_slice),
            conversation_id,
            new_summary.total_summarized_messages,
        )
        return new_summary
