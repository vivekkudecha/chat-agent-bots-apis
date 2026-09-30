import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BotNotFoundException,
    ConversationNotFoundException,
)

from app.repositories.bot_repository import BotRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.audit_repository import AuditRepository


@dataclass
class ConversationListResult:
    items: list[Any]
    total: int


class ConversationService:

    # =====================================================
    # CREATE
    # =====================================================

    async def create(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        bot_id: uuid.UUID,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        # Verify bot ownership.
        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        conversation = await ConversationRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            title=title,
            metadata=metadata or {},
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="conversation",
            resource_id=conversation.id,
            action="conversation.created",
            status="success",
        )

        await db.commit()
        await db.refresh(conversation)

        return conversation

    # =====================================================
    # GET
    # =====================================================

    async def get(
        self,
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ):
        conversation = await ConversationRepository.get_owned(
            db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if not conversation:
            raise ConversationNotFoundException()

        return conversation

    # =====================================================
    # GET WITH MESSAGES
    # =====================================================

    async def get_detail(
        self,
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ):
        conversation = (
            await ConversationRepository.get_with_messages(
                db,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        )

        if not conversation:
            raise ConversationNotFoundException()

        return conversation

    # =====================================================
    # LIST
    # =====================================================

    async def list(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        bot_id: uuid.UUID | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> ConversationListResult:

        offset = (page - 1) * page_size

        items, total = await ConversationRepository.list_by_user(
            db,
            user_id=user_id,
            bot_id=bot_id,
            offset=offset,
            limit=page_size,
        )

        return ConversationListResult(
            items=items,
            total=total,
        )

    # =====================================================
    # UPDATE
    # =====================================================

    async def update(
        self,
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        **update_data,
    ):
        conversation = await self.get(
            db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        protected_fields = {
            "id",
            "user_id",
            "bot_id",
            "created_at",
            "updated_at",
        }

        update_data = {
            key: value
            for key, value in update_data.items()
            if key not in protected_fields
        }

        conversation = await ConversationRepository.update(
            db,
            conversation=conversation,
            **update_data,
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=conversation.bot_id,
            resource_type="conversation",
            resource_id=conversation.id,
            action="conversation.updated",
            status="success",
            metadata={
                "fields": list(update_data.keys())
            },
        )

        await db.commit()
        await db.refresh(conversation)

        return conversation

    # =====================================================
    # DELETE
    # =====================================================

    async def delete(
        self,
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:

        conversation = await self.get(
            db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        conversation_id_value = conversation.id
        bot_id = conversation.bot_id

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot_id,
            resource_type="conversation",
            resource_id=conversation_id_value,
            action="conversation.deleted",
            status="success",
        )

        await ConversationRepository.delete(
            db,
            conversation=conversation,
        )

        await db.commit()

    # =====================================================
    # GET MESSAGES
    # =====================================================

    async def get_messages(
        self,
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int = 100,
        before: uuid.UUID | None = None,
    ):
        conversation = await self.get(
            db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        return await ConversationRepository.get_recent_messages(
            db,
            conversation_id=conversation.id,
            limit=limit,
            before=before,
        )