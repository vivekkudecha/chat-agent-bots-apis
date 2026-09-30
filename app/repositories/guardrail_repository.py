import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.guardrail import (
    Guardrail,
    BotGuardrail,
)


class GuardrailRepository:

    # =====================================================
    # GUARDRAIL REGISTRY
    # =====================================================

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        code: str,
        name: str,
        guardrail_type: str,
        handler: str,
        description: str | None = None,
        default_config: dict | None = None,
    ) -> Guardrail:

        guardrail = Guardrail(
            code=code,
            name=name,
            description=description,
            guardrail_type=guardrail_type,
            handler=handler,
            default_config=default_config or {},
        )

        db.add(guardrail)

        await db.flush()
        await db.refresh(guardrail)

        return guardrail

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        guardrail_id: uuid.UUID,
    ) -> Guardrail | None:

        result = await db.execute(
            select(Guardrail).where(
                Guardrail.id == guardrail_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_code(
        db: AsyncSession,
        code: str,
    ) -> Guardrail | None:

        result = await db.execute(
            select(Guardrail).where(
                Guardrail.code == code
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: int = 50,
        guardrail_type: str | None = None,
        active_only: bool = True,
    ) -> tuple[list[Guardrail], int]:

        conditions = []

        if guardrail_type:
            conditions.append(
                Guardrail.guardrail_type
                == guardrail_type
            )

        if active_only:
            conditions.append(
                Guardrail.is_active.is_(True)
            )

        count_result = await db.execute(
            select(
                func.count(Guardrail.id)
            ).where(*conditions)
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(Guardrail)
            .where(*conditions)
            .order_by(
                Guardrail.name.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return (
            list(result.scalars().all()),
            total,
        )

    @staticmethod
    async def update(
        db: AsyncSession,
        guardrail: Guardrail,
        **values,
    ) -> Guardrail:

        allowed_fields = {
            "name",
            "description",
            "handler",
            "default_config",
            "is_active",
        }

        for key, value in values.items():
            if key in allowed_fields:
                setattr(
                    guardrail,
                    key,
                    value,
                )

        await db.flush()
        await db.refresh(guardrail)

        return guardrail

    # =====================================================
    # BOT ↔ GUARDRAIL
    # =====================================================

    @staticmethod
    async def attach_to_bot(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        guardrail_id: uuid.UUID,
        config: dict | None = None,
        action: str = "block",
        priority: int = 100,
        is_enabled: bool = True,
    ) -> BotGuardrail:

        link = BotGuardrail(
            bot_id=bot_id,
            guardrail_id=guardrail_id,
            config=config or {},
            action=action,
            priority=priority,
            is_enabled=is_enabled,
        )

        db.add(link)

        await db.flush()
        await db.refresh(link)

        return link

    @staticmethod
    async def get_bot_guardrail(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        guardrail_id: uuid.UUID,
    ) -> BotGuardrail | None:

        result = await db.execute(
            select(BotGuardrail)
            .options(
                selectinload(
                    BotGuardrail.guardrail
                )
            )
            .where(
                BotGuardrail.bot_id == bot_id,
                BotGuardrail.guardrail_id
                == guardrail_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_bot(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        guardrail_type: str | None = None,
        enabled_only: bool = True,
    ) -> list[BotGuardrail]:

        conditions = [
            BotGuardrail.bot_id == bot_id
        ]

        if enabled_only:
            conditions.append(
                BotGuardrail.is_enabled.is_(True)
            )

        query = (
            select(BotGuardrail)
            .join(BotGuardrail.guardrail)
            .options(
                selectinload(
                    BotGuardrail.guardrail
                )
            )
            .where(*conditions)
        )

        if guardrail_type:
            query = query.where(
                Guardrail.guardrail_type
                == guardrail_type
            )

        query = query.order_by(
            BotGuardrail.priority.asc()
        )

        result = await db.execute(query)

        return list(
            result.scalars().all()
        )

    @staticmethod
    async def update_bot_guardrail(
        db: AsyncSession,
        bot_guardrail: BotGuardrail,
        **values,
    ) -> BotGuardrail:

        allowed_fields = {
            "config",
            "action",
            "priority",
            "is_enabled",
        }

        for key, value in values.items():
            if key in allowed_fields:
                setattr(
                    bot_guardrail,
                    key,
                    value,
                )

        await db.flush()
        await db.refresh(bot_guardrail)

        return bot_guardrail

    @staticmethod
    async def detach_from_bot(
        db: AsyncSession,
        bot_guardrail: BotGuardrail,
    ) -> None:

        await db.delete(bot_guardrail)
        await db.flush()