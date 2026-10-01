import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.bot import Bot, BotVersion


class BotRepository:

    # -----------------------------------------------------
    # Create Bot
    # -----------------------------------------------------

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        name: str,
        slug: str,
        description: str | None = None,
        visibility: str = "private",
        avatar_url: str | None = None,
        metadata: dict | None = None,
    ) -> Bot:

        bot = Bot(
            user_id=user_id,
            name=name,
            slug=slug,
            description=description,
            visibility=visibility,
            avatar_url=avatar_url,
            metadata_=metadata or {},
        )

        db.add(bot)

        await db.flush()
        await db.refresh(bot)

        return bot

    # -----------------------------------------------------
    # Get Bot
    # -----------------------------------------------------

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        bot_id: uuid.UUID,
    ) -> Bot | None:

        result = await db.execute(
            select(Bot).where(
                Bot.id == bot_id
            )
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # Get Bot with Versions
    # -----------------------------------------------------

    @staticmethod
    async def get_with_versions(
        db: AsyncSession,
        bot_id: uuid.UUID,
    ) -> Bot | None:

        result = await db.execute(
            select(Bot)
            .options(
                selectinload(Bot.versions)
            )
            .where(
                Bot.id == bot_id
            )
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # Get User-Owned Bot
    # -----------------------------------------------------

    @staticmethod
    async def get_owned_bot(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Bot | None:

        result = await db.execute(
            select(Bot).where(
                Bot.id == bot_id,
                Bot.user_id == user_id,
            )
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # Get by Slug
    # -----------------------------------------------------

    @staticmethod
    async def get_by_slug(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        slug: str,
    ) -> Bot | None:

        result = await db.execute(
            select(Bot).where(
                Bot.user_id == user_id,
                Bot.slug == slug,
            )
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # List User Bots
    # -----------------------------------------------------

    @staticmethod
    async def list_by_user(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        offset: int = 0,
        limit: int = 20,
        status: str | None = None,
    ) -> tuple[list[Bot], int]:

        conditions = [
            Bot.user_id == user_id
        ]

        if status:
            conditions.append(
                Bot.status == status
            )

        count_result = await db.execute(
            select(
                func.count(Bot.id)
            ).where(*conditions)
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(Bot)
            .where(*conditions)
            .order_by(
                Bot.updated_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        bots = list(
            result.scalars().all()
        )

        return bots, total

    # -----------------------------------------------------
    # Update Bot
    # -----------------------------------------------------

    @staticmethod
    async def update(
        db: AsyncSession,
        bot: Bot,
        **values,
    ) -> Bot:

        allowed_fields = {
            "name",
            "description",
            "visibility",
            "avatar_url",
            "status",
            "is_api_enabled",
            "metadata_",
        }

        for key, value in values.items():

            if key in allowed_fields:
                setattr(bot, key, value)

        await db.flush()
        await db.refresh(bot)

        return bot

    # -----------------------------------------------------
    # Delete Bot
    # -----------------------------------------------------

    @staticmethod
    async def delete(
        db: AsyncSession,
        bot: Bot,
    ) -> None:

        await db.delete(bot)

        await db.flush()

    # =====================================================
    # BOT VERSIONS
    # =====================================================

    @staticmethod
    async def create_version(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        system_instruction: str,
        welcome_message: str | None = None,
        conversation_starters: list | None = None,
        config: dict | None = None,
    ) -> BotVersion:

        latest_result = await db.execute(
            select(
                func.max(
                    BotVersion.version
                )
            ).where(
                BotVersion.bot_id == bot_id
            )
        )

        latest_version = (
            latest_result.scalar_one_or_none()
            or 0
        )

        version = BotVersion(
            bot_id=bot_id,
            version=latest_version + 1,
            system_instruction=system_instruction,
            welcome_message=welcome_message,
            conversation_starters=(
                conversation_starters or []
            ),
            config=config or {},
        )

        db.add(version)

        await db.flush()
        await db.refresh(version)

        return version

    # -----------------------------------------------------
    # Get Version
    # -----------------------------------------------------

    @staticmethod
    async def get_version(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        version: int,
    ) -> BotVersion | None:

        result = await db.execute(
            select(BotVersion).where(
                BotVersion.bot_id == bot_id,
                BotVersion.version == version,
            )
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # Latest Version
    # -----------------------------------------------------

    @staticmethod
    async def get_latest_version(
        db: AsyncSession,
        bot_id: uuid.UUID,
    ) -> BotVersion | None:

        result = await db.execute(
            select(BotVersion)
            .where(
                BotVersion.bot_id == bot_id
            )
            .order_by(
                BotVersion.version.desc()
            )
            .limit(1)
        )

        return result.scalar_one_or_none()

    # -----------------------------------------------------
    # Active Version (latest version)
    # -----------------------------------------------------

    @staticmethod
    async def get_active_version(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
    ) -> BotVersion | None:
        return await BotRepository.get_latest_version(db, bot_id=bot_id)

    # -----------------------------------------------------
    # Primary Model Config
    # -----------------------------------------------------

    @staticmethod
    async def get_primary_model_config(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
    ):
        from app.repositories.ai_model_repository import AIModelRepository

        return await AIModelRepository.get_primary_model(db, bot_id=bot_id)

    # -----------------------------------------------------
    # List Versions
    # -----------------------------------------------------

    @staticmethod
    async def list_versions(
        db: AsyncSession,
        bot_id: uuid.UUID,
    ) -> list[BotVersion]:

        result = await db.execute(
            select(BotVersion)
            .where(
                BotVersion.bot_id == bot_id
            )
            .order_by(
                BotVersion.version.desc()
            )
        )

        return list(
            result.scalars().all()
        )