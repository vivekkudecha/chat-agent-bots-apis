import re
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BotNotFoundException,
    KnowledgeBaseNotFoundException,
)
from app.models.bot import Bot, BotVersion
from app.repositories.audit_repository import AuditRepository
from app.repositories.bot_repository import BotRepository
from app.repositories.knowledge_repository import KnowledgeRepository


@dataclass
class BotListResult:
    items: list[Any]
    total: int


class BotService:

    # =====================================================
    # CREATE
    # =====================================================

    async def create(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        name: str,
        slug: str | None = None,
        description: str | None = None,
        system_instruction: str | None = None,
        welcome_message: str | None = None,
        conversation_starters: list[str] | None = None,
        visibility: str = "private",
        avatar_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Bot:

        bot_slug = (
            slug
            or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            or f"bot-{uuid.uuid4().hex[:8]}"
        )

        bot = await BotRepository.create(
            db,
            user_id=user_id,
            name=name.strip(),
            slug=bot_slug,
            description=description,
            visibility=visibility,
            avatar_url=avatar_url,
            metadata=metadata or {},
        )

        if system_instruction:
            await BotRepository.create_version(
                db,
                bot_id=bot.id,
                system_instruction=system_instruction,
                welcome_message=welcome_message,
                conversation_starters=conversation_starters or [],
            )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot",
            resource_id=bot.id,
            action="bot.created",
            status="success",
        )

        await db.commit()

        bot_with_versions = await BotRepository.get_with_versions(
            db,
            bot_id=bot.id,
        )
        return bot_with_versions or bot

    # =====================================================
    # GET
    # =====================================================

    async def get(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Bot:

        bot = await BotRepository.get_with_versions(
            db,
            bot_id=bot_id,
        )

        if not bot or bot.user_id != user_id:
            raise BotNotFoundException()

        return bot

    # =====================================================
    # LIST
    # =====================================================

    async def list(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> BotListResult:

        offset = (page - 1) * page_size

        items, total = await BotRepository.list_by_user(
            db,
            user_id=user_id,
            offset=offset,
            limit=page_size,
        )

        return BotListResult(
            items=items,
            total=total,
        )

    # =====================================================
    # UPDATE DRAFT
    # =====================================================

    async def update(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
        **update_data,
    ) -> Bot:

        bot = await self.get(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        protected_fields = {
            "id",
            "user_id",
            "created_at",
            "updated_at",
        }

        bot_updates = {
            key: value
            for key, value in update_data.items()
            if key not in protected_fields
            and key not in {"system_instruction", "welcome_message", "conversation_starters"}
            and value is not None
        }

        if bot_updates:
            bot = await BotRepository.update(
                db,
                bot=bot,
                **bot_updates,
            )

        instruction = update_data.get("system_instruction")
        welcome = update_data.get("welcome_message")
        starters = update_data.get("conversation_starters")

        if instruction or welcome or starters is not None:
            latest = await BotRepository.get_latest_version(
                db,
                bot_id=bot.id,
            )
            inst = instruction if instruction is not None else (latest.system_instruction if latest else "")
            welc = welcome if welcome is not None else (latest.welcome_message if latest else None)
            start = starters if starters is not None else (latest.conversation_starters if latest else [])

            if inst:
                await BotRepository.create_version(
                    db,
                    bot_id=bot.id,
                    system_instruction=inst,
                    welcome_message=welc,
                    conversation_starters=start,
                )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot",
            resource_id=bot.id,
            action="bot.updated",
            status="success",
            metadata={"fields": list(update_data.keys())},
        )

        await db.commit()

        bot_refreshed = await BotRepository.get_with_versions(
            db,
            bot_id=bot.id,
        )
        return bot_refreshed or bot

    # =====================================================
    # DELETE
    # =====================================================

    async def delete(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot",
            resource_id=bot.id,
            action="bot.deleted",
            status="success",
        )

        await BotRepository.delete(
            db,
            bot=bot,
        )

        await db.commit()

    # =====================================================
    # PUBLISH VERSION
    # =====================================================

    async def publish(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> BotVersion:

        bot = await self.get(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        latest_version = await BotRepository.get_latest_version(
            db,
            bot_id=bot.id,
        )

        instruction = latest_version.system_instruction if latest_version else "You are a helpful AI assistant."
        welcome = latest_version.welcome_message if latest_version else None
        starters = latest_version.conversation_starters if latest_version else []

        version = await BotRepository.create_version(
            db,
            bot_id=bot.id,
            system_instruction=instruction,
            welcome_message=welcome,
            conversation_starters=starters,
        )

        await BotRepository.update(
            db,
            bot=bot,
            status="published",
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot_version",
            resource_id=version.id,
            action="bot.published",
            status="success",
            metadata={"version": version.version},
        )

        await db.commit()
        await db.refresh(version)

        return version

    # =====================================================
    # LIST VERSIONS
    # =====================================================

    async def list_versions(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> list[BotVersion]:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        return await BotRepository.list_versions(
            db,
            bot_id=bot.id,
        )

    # =====================================================
    # ATTACH KNOWLEDGE BASE
    # =====================================================

    async def attach_knowledge_base(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        knowledge_base = await KnowledgeRepository.get_owned(
            db,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
        )

        if not knowledge_base:
            raise KnowledgeBaseNotFoundException()

        existing = await KnowledgeRepository.get_bot_knowledge_base(
            db,
            bot_id=bot.id,
            knowledge_base_id=knowledge_base.id,
        )

        if existing:
            return

        await KnowledgeRepository.attach_to_bot(
            db,
            bot_id=bot.id,
            knowledge_base_id=knowledge_base.id,
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="knowledge_base",
            resource_id=knowledge_base.id,
            action="bot.knowledge_base.attached",
            status="success",
        )

        await db.commit()

    # =====================================================
    # DETACH KNOWLEDGE BASE
    # =====================================================

    async def detach_knowledge_base(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        knowledge_base = await KnowledgeRepository.get_owned(
            db,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
        )

        if not knowledge_base:
            raise KnowledgeBaseNotFoundException()

        await KnowledgeRepository.detach_from_bot(
            db,
            bot_id=bot.id,
            knowledge_base_id=knowledge_base.id,
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="knowledge_base",
            resource_id=knowledge_base.id,
            action="bot.knowledge_base.detached",
            status="success",
        )

        await db.commit()