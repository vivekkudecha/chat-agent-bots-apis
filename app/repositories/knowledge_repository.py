import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.knowledge_base import (
    KnowledgeBase,
    BotKnowledgeBase,
)


class KnowledgeRepository:

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        name: str,
        description: str | None = None,
        embedding_model: str | None = None,
        chunking_config: dict | None = None,
        retrieval_config: dict | None = None,
    ) -> KnowledgeBase:

        kb = KnowledgeBase(
            user_id=user_id,
            name=name,
            description=description,
            embedding_model=embedding_model,
            chunking_config=chunking_config or {},
            retrieval_config=retrieval_config or {},
        )

        db.add(kb)

        await db.flush()
        await db.refresh(kb)

        return kb

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        knowledge_base_id: uuid.UUID,
    ) -> KnowledgeBase | None:

        result = await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == knowledge_base_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_owned(
        db: AsyncSession,
        *,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> KnowledgeBase | None:

        result = await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.user_id == user_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_with_documents(
        db: AsyncSession,
        *,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> KnowledgeBase | None:

        result = await db.execute(
            select(KnowledgeBase)
            .options(
                selectinload(
                    KnowledgeBase.documents
                )
            )
            .where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.user_id == user_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_user(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[KnowledgeBase], int]:

        count_result = await db.execute(
            select(
                func.count(KnowledgeBase.id)
            ).where(
                KnowledgeBase.user_id == user_id
            )
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(KnowledgeBase)
            .where(
                KnowledgeBase.user_id == user_id
            )
            .order_by(
                KnowledgeBase.updated_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        items = list(
            result.scalars().all()
        )

        return items, total

    @staticmethod
    async def update(
        db: AsyncSession,
        knowledge_base: KnowledgeBase,
        **values,
    ) -> KnowledgeBase:

        allowed_fields = {
            "name",
            "description",
            "embedding_model",
            "chunking_config",
            "retrieval_config",
        }

        for key, value in values.items():
            if key in allowed_fields:
                setattr(
                    knowledge_base,
                    key,
                    value,
                )

        await db.flush()
        await db.refresh(knowledge_base)

        return knowledge_base

    @staticmethod
    async def delete(
        db: AsyncSession,
        knowledge_base: KnowledgeBase,
    ) -> None:

        await db.delete(knowledge_base)
        await db.flush()

    # =====================================================
    # BOT ↔ KNOWLEDGE BASE
    # =====================================================

    @staticmethod
    async def attach_to_bot(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        priority: int = 0,
        config: dict | None = None,
    ) -> BotKnowledgeBase:

        link = BotKnowledgeBase(
            bot_id=bot_id,
            knowledge_base_id=knowledge_base_id,
            priority=priority,
            config=config or {},
        )

        db.add(link)

        await db.flush()
        await db.refresh(link)

        return link

    @staticmethod
    async def get_bot_link(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
    ) -> BotKnowledgeBase | None:

        result = await db.execute(
            select(BotKnowledgeBase).where(
                BotKnowledgeBase.bot_id == bot_id,
                BotKnowledgeBase.knowledge_base_id
                == knowledge_base_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_bot(
        db: AsyncSession,
        bot_id: uuid.UUID,
    ) -> list[BotKnowledgeBase]:

        result = await db.execute(
            select(BotKnowledgeBase)
            .options(
                selectinload(
                    BotKnowledgeBase.knowledge_base
                )
            )
            .where(
                BotKnowledgeBase.bot_id == bot_id
            )
            .order_by(
                BotKnowledgeBase.priority.desc()
            )
        )

        return list(
            result.scalars().all()
        )

    @staticmethod
    async def get_kb_ids_for_bot(
        db: AsyncSession,
        bot_id: uuid.UUID,
    ) -> list[uuid.UUID]:

        result = await db.execute(
            select(
                BotKnowledgeBase.knowledge_base_id
            ).where(
                BotKnowledgeBase.bot_id == bot_id
            )
        )

        return list(
            result.scalars().all()
        )

    @staticmethod
    async def update_bot_link(
        db: AsyncSession,
        link: BotKnowledgeBase,
        **values,
    ) -> BotKnowledgeBase:

        allowed_fields = {
            "priority",
            "config",
        }

        for key, value in values.items():
            if key in allowed_fields:
                setattr(link, key, value)

        await db.flush()
        await db.refresh(link)

        return link

    @staticmethod
    async def detach_from_bot(
        db: AsyncSession,
        link: BotKnowledgeBase | None = None,
        *,
        bot_id: uuid.UUID | None = None,
        knowledge_base_id: uuid.UUID | None = None,
    ) -> None:

        if link is None:
            if bot_id and knowledge_base_id:
                link = await KnowledgeRepository.get_bot_link(
                    db,
                    bot_id=bot_id,
                    knowledge_base_id=knowledge_base_id,
                )

        if link is not None:
            await db.delete(link)
            await db.flush()

    # Alias for compatibility
    get_bot_knowledge_base = get_bot_link