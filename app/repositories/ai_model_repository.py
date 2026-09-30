import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.ai_model import (
    AIModel,
    BotModelConfig,
)


class AIModelRepository:

    # =====================================================
    # MODEL REGISTRY
    # =====================================================

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        provider: str,
        model_key: str,
        display_name: str,
        model_type: str = "chat",
        endpoint_url: str | None = None,
        context_window: int | None = None,
        capabilities: dict | None = None,
    ) -> AIModel:

        model = AIModel(
            provider=provider,
            model_key=model_key,
            display_name=display_name,
            model_type=model_type,
            endpoint_url=endpoint_url,
            context_window=context_window,
            capabilities=capabilities or {},
        )

        db.add(model)

        await db.flush()
        await db.refresh(model)

        return model

    # -----------------------------------------------------
    # Get Model
    # -----------------------------------------------------

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        model_id: uuid.UUID,
    ) -> AIModel | None:

        result = await db.execute(
            select(AIModel).where(
                AIModel.id == model_id
            )
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # Get by Provider + Model Key
    # -----------------------------------------------------

    @staticmethod
    async def get_by_key(
        db: AsyncSession,
        *,
        provider: str,
        model_key: str,
    ) -> AIModel | None:

        result = await db.execute(
            select(AIModel).where(
                AIModel.provider == provider,
                AIModel.model_key == model_key,
            )
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # List Models
    # -----------------------------------------------------

    @staticmethod
    async def list(
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: int = 50,
        model_type: str | None = None,
        active_only: bool = True,
    ) -> tuple[list[AIModel], int]:

        conditions = []

        if model_type:
            conditions.append(
                AIModel.model_type == model_type
            )

        if active_only:
            conditions.append(
                AIModel.is_active.is_(True)
            )

        count_result = await db.execute(
            select(
                func.count(AIModel.id)
            ).where(*conditions)
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(AIModel)
            .where(*conditions)
            .order_by(
                AIModel.provider,
                AIModel.display_name,
            )
            .offset(offset)
            .limit(limit)
        )

        models = list(
            result.scalars().all()
        )

        return models, total

    # -----------------------------------------------------
    # Update Model
    # -----------------------------------------------------

    @staticmethod
    async def update(
        db: AsyncSession,
        model: AIModel,
        **values,
    ) -> AIModel:

        allowed_fields = {
            "display_name",
            "endpoint_url",
            "context_window",
            "capabilities",
            "is_active",
        }

        for key, value in values.items():

            if key in allowed_fields:
                setattr(model, key, value)

        await db.flush()
        await db.refresh(model)

        return model

    # =====================================================
    # BOT MODEL CONFIG
    # =====================================================

    @staticmethod
    async def create_bot_config(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        model_id: uuid.UUID,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        config: dict | None = None,
        is_primary: bool = True,
    ) -> BotModelConfig:

        bot_config = BotModelConfig(
            bot_id=bot_id,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            config=config or {},
            is_primary=is_primary,
        )

        db.add(bot_config)

        await db.flush()
        await db.refresh(bot_config)

        return bot_config

    # -----------------------------------------------------
    # Get Bot Config
    # -----------------------------------------------------

    @staticmethod
    async def get_bot_config(
        db: AsyncSession,
        config_id: uuid.UUID,
    ) -> BotModelConfig | None:

        result = await db.execute(
            select(BotModelConfig)
            .options(
                selectinload(
                    BotModelConfig.model
                )
            )
            .where(
                BotModelConfig.id == config_id
            )
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # List Models Attached to Bot
    # -----------------------------------------------------

    @staticmethod
    async def list_bot_configs(
        db: AsyncSession,
        bot_id: uuid.UUID,
    ) -> list[BotModelConfig]:

        result = await db.execute(
            select(BotModelConfig)
            .options(
                selectinload(
                    BotModelConfig.model
                )
            )
            .where(
                BotModelConfig.bot_id == bot_id
            )
            .order_by(
                BotModelConfig.is_primary.desc(),
                BotModelConfig.created_at.asc(),
            )
        )

        return list(
            result.scalars().all()
        )

    # -----------------------------------------------------
    # Primary Model
    # -----------------------------------------------------

    @staticmethod
    async def get_primary_model(
        db: AsyncSession,
        bot_id: uuid.UUID,
    ) -> BotModelConfig | None:

        result = await db.execute(
            select(BotModelConfig)
            .options(
                selectinload(
                    BotModelConfig.model
                )
            )
            .where(
                BotModelConfig.bot_id == bot_id,
                BotModelConfig.is_primary.is_(True),
            )
            .limit(1)
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # Clear Existing Primary
    # -----------------------------------------------------

    @staticmethod
    async def clear_primary(
        db: AsyncSession,
        bot_id: uuid.UUID,
    ) -> None:

        result = await db.execute(
            select(BotModelConfig).where(
                BotModelConfig.bot_id == bot_id,
                BotModelConfig.is_primary.is_(True),
            )
        )

        configs = result.scalars().all()

        for config in configs:
            config.is_primary = False

        await db.flush()

    # -----------------------------------------------------
    # Update Bot Model Config
    # -----------------------------------------------------

    @staticmethod
    async def update_bot_config(
        db: AsyncSession,
        bot_config: BotModelConfig,
        **values,
    ) -> BotModelConfig:

        allowed_fields = {
            "temperature",
            "top_p",
            "max_tokens",
            "config",
            "is_primary",
        }

        for key, value in values.items():

            if key in allowed_fields:
                setattr(
                    bot_config,
                    key,
                    value,
                )

        await db.flush()
        await db.refresh(bot_config)

        return bot_config

    # -----------------------------------------------------
    # Delete Bot Model Config
    # -----------------------------------------------------

    @staticmethod
    async def delete_bot_config(
        db: AsyncSession,
        bot_config: BotModelConfig,
    ) -> None:

        await db.delete(bot_config)

        await db.flush()