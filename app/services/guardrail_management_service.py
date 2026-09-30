import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BotNotFoundException,
    ConflictException,
    GuardrailNotFoundException,
)
from app.models.guardrail import BotGuardrail, Guardrail
from app.repositories.audit_repository import AuditRepository
from app.repositories.bot_repository import BotRepository
from app.repositories.guardrail_repository import GuardrailRepository


@dataclass
class GuardrailListResult:
    items: list[Any]
    total: int


class GuardrailManagementService:

    # =====================================================
    # CREATE DEFINITION
    # =====================================================

    async def create(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        code: str,
        name: str,
        handler: str,
        guardrail_type: str,
        description: str | None = None,
        default_config: dict[str, Any] | None = None,
    ) -> Guardrail:

        existing = await GuardrailRepository.get_by_code(
            db,
            code=code.strip().lower(),
        )

        if existing:
            raise ConflictException(
                message=f"Guardrail with code '{code}' already exists."
            )

        guardrail = await GuardrailRepository.create(
            db,
            code=code.strip().lower(),
            name=name.strip(),
            description=description,
            guardrail_type=guardrail_type,
            handler=handler.strip(),
            default_config=default_config or {},
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            resource_type="guardrail",
            resource_id=guardrail.id,
            action="guardrail.created",
            status="success",
        )

        await db.commit()
        await db.refresh(guardrail)

        return guardrail

    # =====================================================
    # GET DEFINITION
    # =====================================================

    async def get(
        self,
        db: AsyncSession,
        *,
        guardrail_id: uuid.UUID,
    ) -> Guardrail:

        guardrail = await GuardrailRepository.get_by_id(
            db,
            guardrail_id=guardrail_id,
        )

        if not guardrail:
            raise GuardrailNotFoundException()

        return guardrail

    # =====================================================
    # LIST DEFINITIONS
    # =====================================================

    async def list(
        self,
        db: AsyncSession,
        *,
        active_only: bool = True,
        page: int = 1,
        page_size: int = 50,
        guardrail_type: str | None = None,
    ) -> GuardrailListResult:

        offset = (page - 1) * page_size

        items, total = await GuardrailRepository.list(
            db,
            offset=offset,
            limit=page_size,
            guardrail_type=guardrail_type,
            active_only=active_only,
        )

        return GuardrailListResult(
            items=items,
            total=total,
        )

    # =====================================================
    # UPDATE DEFINITION
    # =====================================================

    async def update(
        self,
        db: AsyncSession,
        *,
        guardrail_id: uuid.UUID,
        **update_data,
    ) -> Guardrail:

        guardrail = await self.get(
            db,
            guardrail_id=guardrail_id,
        )

        protected_fields = {
            "id",
            "code",
            "guardrail_type",
            "created_at",
        }

        filtered = {
            k: v
            for k, v in update_data.items()
            if k not in protected_fields and v is not None
        }

        if filtered:
            guardrail = await GuardrailRepository.update(
                db,
                guardrail,
                **filtered,
            )
            await db.commit()
            await db.refresh(guardrail)

        return guardrail

    # =====================================================
    # ATTACH TO BOT
    # =====================================================

    async def attach_to_bot(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
        guardrail_id: uuid.UUID,
        action: str = "block",
        priority: int = 100,
        config: dict[str, Any] | None = None,
        enabled: bool | None = None,
        is_enabled: bool | None = None,
    ) -> BotGuardrail:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        guardrail = await GuardrailRepository.get_by_id(
            db,
            guardrail_id=guardrail_id,
        )

        if not guardrail:
            raise GuardrailNotFoundException()

        effective_enabled = True
        if is_enabled is not None:
            effective_enabled = is_enabled
        elif enabled is not None:
            effective_enabled = enabled

        existing = await GuardrailRepository.get_bot_guardrail(
            db,
            bot_id=bot.id,
            guardrail_id=guardrail.id,
        )

        if existing:
            existing = await GuardrailRepository.update_bot_guardrail(
                db,
                existing,
                action=action,
                priority=priority,
                config=config or existing.config,
                is_enabled=effective_enabled,
            )
            await db.commit()
            await db.refresh(existing)
            return existing

        link = await GuardrailRepository.attach_to_bot(
            db,
            bot_id=bot.id,
            guardrail_id=guardrail.id,
            config=config or {},
            action=action,
            priority=priority,
            is_enabled=effective_enabled,
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot_guardrail",
            resource_id=guardrail.id,
            action="bot.guardrail.attached",
            status="success",
        )

        await db.commit()
        await db.refresh(link)

        return link

    # =====================================================
    # LIST FOR BOT
    # =====================================================

    async def list_for_bot(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> list[BotGuardrail]:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        return await GuardrailRepository.list_for_bot(
            db,
            bot_id=bot.id,
            enabled_only=False,
        )

    # =====================================================
    # UPDATE BOT GUARDRAIL
    # =====================================================

    async def update_bot_guardrail(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        bot_guardrail_id: uuid.UUID,
        user_id: uuid.UUID,
        action: str | None = None,
        priority: int | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool | None = None,
        is_enabled: bool | None = None,
    ) -> BotGuardrail:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        link = await GuardrailRepository.get_bot_guardrail(
            db,
            bot_id=bot.id,
            guardrail_id=bot_guardrail_id,
        )

        if not link:
            raise GuardrailNotFoundException(
                message="Guardrail is not attached to this bot."
            )

        updates: dict[str, Any] = {}
        if action is not None:
            updates["action"] = action
        if priority is not None:
            updates["priority"] = priority
        if config is not None:
            updates["config"] = config

        effective_enabled = is_enabled if is_enabled is not None else enabled
        if effective_enabled is not None:
            updates["is_enabled"] = effective_enabled

        if updates:
            link = await GuardrailRepository.update_bot_guardrail(
                db,
                link,
                **updates,
            )
            await db.commit()
            await db.refresh(link)

        return link

    # =====================================================
    # REMOVE FROM BOT
    # =====================================================

    async def remove_from_bot(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        bot_guardrail_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        link = await GuardrailRepository.get_bot_guardrail(
            db,
            bot_id=bot.id,
            guardrail_id=bot_guardrail_id,
        )

        if not link:
            return

        await GuardrailRepository.detach_from_bot(
            db,
            link,
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot_guardrail",
            resource_id=bot_guardrail_id,
            action="bot.guardrail.removed",
            status="success",
        )

        await db.commit()
