import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BotNotFoundException,
    ModelNotFoundException,
)
from app.models.ai_model import AIModel, BotModelConfig
from app.repositories.ai_model_repository import AIModelRepository
from app.repositories.audit_repository import AuditRepository
from app.repositories.bot_repository import BotRepository


@dataclass
class AIModelListResult:
    items: list[Any]
    total: int


class AIModelService:

    # =====================================================
    # CREATE MODEL
    # =====================================================

    async def create(
        self,
        db: AsyncSession,
        *,
        provider: str,
        model_key: str,
        user_id: uuid.UUID | None = None,
        name: str | None = None,
        display_name: str | None = None,
        model_type: str = "chat",
        endpoint_url: str | None = None,
        context_window: int | None = None,
        capabilities: dict[str, Any] | None = None,
    ) -> AIModel:

        d_name = display_name or name or model_key

        model = await AIModelRepository.create(
            db,
            provider=provider.strip().lower(),
            model_key=model_key.strip(),
            display_name=d_name.strip(),
            model_type=model_type,
            endpoint_url=str(endpoint_url) if endpoint_url else None,
            context_window=context_window,
            capabilities=capabilities or {},
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            resource_type="ai_model",
            resource_id=model.id,
            action="ai_model.created",
            status="success",
        )

        await db.commit()
        await db.refresh(model)

        return model

    # =====================================================
    # LIST AVAILABLE MODELS
    # =====================================================

    async def list(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID | None = None,
        active_only: bool = True,
        page: int = 1,
        page_size: int = 20,
        model_type: str | None = None,
    ) -> AIModelListResult:

        offset = (page - 1) * page_size

        items, total = await AIModelRepository.list(
            db,
            offset=offset,
            limit=page_size,
            model_type=model_type,
            active_only=active_only,
        )

        return AIModelListResult(
            items=items,
            total=total,
        )

    # =====================================================
    # GET MODEL
    # =====================================================

    async def get(
        self,
        db: AsyncSession,
        *,
        model_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> AIModel:

        model = await AIModelRepository.get_by_id(
            db,
            model_id,
        )

        if not model:
            raise ModelNotFoundException()

        return model

    # =====================================================
    # GET MODEL BY KEY
    # =====================================================

    async def get_by_key(
        self,
        db: AsyncSession,
        *,
        model_key: str,
        provider: str | None = None,
    ) -> AIModel:

        if provider:
            model = await AIModelRepository.get_by_key(
                db,
                provider=provider,
                model_key=model_key,
            )
        else:
            items, _ = await AIModelRepository.list(
                db,
                limit=1,
                active_only=False,
            )
            matching = [m for m in items if m.model_key == model_key]
            model = matching[0] if matching else None

        if not model:
            raise ModelNotFoundException()

        return model

    # =====================================================
    # UPDATE MODEL
    # =====================================================

    async def update(
        self,
        db: AsyncSession,
        *,
        model_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        name: str | None = None,
        display_name: str | None = None,
        endpoint_url: str | None = None,
        context_window: int | None = None,
        capabilities: dict[str, Any] | None = None,
        is_active: bool | None = None,
    ) -> AIModel:

        model = await self.get(db, model_id=model_id)

        updates: dict[str, Any] = {}
        target_name = display_name or name
        if target_name is not None:
            updates["display_name"] = target_name.strip()
        if endpoint_url is not None:
            updates["endpoint_url"] = str(endpoint_url)
        if context_window is not None:
            updates["context_window"] = context_window
        if capabilities is not None:
            updates["capabilities"] = capabilities
        if is_active is not None:
            updates["is_active"] = is_active

        if updates:
            model = await AIModelRepository.update(
                db,
                model,
                **updates,
            )
            await db.commit()
            await db.refresh(model)

        return model

    # =====================================================
    # DELETE MODEL
    # =====================================================

    async def delete(
        self,
        db: AsyncSession,
        *,
        model_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> None:

        model = await self.get(db, model_id=model_id)
        await AIModelRepository.update(
            db,
            model,
            is_active=False,
        )
        await db.commit()

    # =====================================================
    # GET BOT MODEL CONFIG
    # =====================================================

    async def get_bot_model_config(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> BotModelConfig | None:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        return await AIModelRepository.get_primary_model(
            db,
            bot_id=bot.id,
        )

    # =====================================================
    # LIST BOT MODEL CONFIGS
    # =====================================================

    async def list_bot_models(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> list[BotModelConfig]:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        return await AIModelRepository.list_bot_configs(
            db,
            bot_id=bot.id,
        )

    # =====================================================
    # CONFIGURE BOT MODEL
    # =====================================================

    async def configure_bot_model(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
        model_id: uuid.UUID,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        is_primary: bool = True,
        **extra_config,
    ) -> BotModelConfig:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        model = await AIModelRepository.get_by_id(
            db,
            model_id,
        )

        if not model:
            raise ModelNotFoundException()

        if not model.is_active:
            raise ModelNotFoundException(
                "Selected model is inactive."
            )

        if (
            max_tokens is not None
            and model.context_window is not None
            and max_tokens > model.context_window
        ):
            raise ValueError(
                f"max_tokens cannot exceed context window ({model.context_window}) for this model."
            )

        if is_primary:
            await AIModelRepository.clear_primary(
                db,
                bot_id=bot.id,
            )

        existing_configs = await AIModelRepository.list_bot_configs(
            db,
            bot_id=bot.id,
        )
        matching = [c for c in existing_configs if c.model_id == model.id]

        if matching:
            config = await AIModelRepository.update_bot_config(
                db,
                matching[0],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or 2048,
                is_primary=is_primary,
                config=extra_config.get("config", matching[0].config),
            )
        else:
            config = await AIModelRepository.create_bot_config(
                db,
                bot_id=bot.id,
                model_id=model.id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or 2048,
                config=extra_config.get("config", {}),
                is_primary=is_primary,
            )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot_model_config",
            resource_id=config.id,
            action="bot.model.configured",
            status="success",
            metadata={
                "model_id": str(model.id),
                "model_key": model.model_key,
            },
        )

        await db.commit()
        await db.refresh(config)

        return config

    # =====================================================
    # REMOVE BOT MODEL CONFIG
    # =====================================================

    async def remove_bot_model(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
        model_id: uuid.UUID | None = None,
    ) -> None:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        configs = await AIModelRepository.list_bot_configs(
            db,
            bot_id=bot.id,
        )

        for cfg in configs:
            if model_id is None or cfg.model_id == model_id:
                await AIModelRepository.delete_bot_config(
                    db,
                    cfg,
                )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot_model_config",
            resource_id=model_id or bot.id,
            action="bot.model.removed",
            status="success",
        )

        await db.commit()