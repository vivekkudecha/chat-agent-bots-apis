import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.tool import (
    Tool,
    BotTool,
)


class ToolRepository:

    # =====================================================
    # TOOL REGISTRY
    # =====================================================

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        name: str,
        display_name: str,
        tool_type: str,
        description: str | None = None,
        input_schema: dict | None = None,
        config: dict | None = None,
        timeout_seconds: int = 30,
    ) -> Tool:

        tool = Tool(
            name=name,
            display_name=display_name,
            description=description,
            tool_type=tool_type,
            input_schema=input_schema or {
                "type": "object",
                "properties": {},
            },
            config=config or {},
            timeout_seconds=timeout_seconds,
        )

        db.add(tool)

        await db.flush()
        await db.refresh(tool)

        return tool

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        tool_id: uuid.UUID,
    ) -> Tool | None:

        result = await db.execute(
            select(Tool).where(
                Tool.id == tool_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_name(
        db: AsyncSession,
        name: str,
    ) -> Tool | None:

        result = await db.execute(
            select(Tool).where(
                Tool.name == name
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: int = 50,
        tool_type: str | None = None,
        active_only: bool = True,
    ) -> tuple[list[Tool], int]:

        conditions = []

        if tool_type:
            conditions.append(
                Tool.tool_type == tool_type
            )

        if active_only:
            conditions.append(
                Tool.is_active.is_(True)
            )

        count_result = await db.execute(
            select(
                func.count(Tool.id)
            ).where(*conditions)
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(Tool)
            .where(*conditions)
            .order_by(
                Tool.display_name.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        tools = list(
            result.scalars().all()
        )

        return tools, total

    @staticmethod
    async def update(
        db: AsyncSession,
        tool: Tool,
        **values,
    ) -> Tool:

        allowed_fields = {
            "display_name",
            "description",
            "input_schema",
            "config",
            "timeout_seconds",
            "is_active",
        }

        for key, value in values.items():
            if key in allowed_fields:
                setattr(tool, key, value)

        await db.flush()
        await db.refresh(tool)

        return tool

    # =====================================================
    # BOT ↔ TOOL
    # =====================================================

    @staticmethod
    async def attach_to_bot(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        tool_id: uuid.UUID,
        config: dict | None = None,
        requires_confirmation: bool = False,
        is_enabled: bool = True,
    ) -> BotTool:

        bot_tool = BotTool(
            bot_id=bot_id,
            tool_id=tool_id,
            config=config or {},
            requires_confirmation=requires_confirmation,
            is_enabled=is_enabled,
        )

        db.add(bot_tool)

        await db.flush()
        await db.refresh(bot_tool)

        return bot_tool

    @staticmethod
    async def get_bot_tool(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        tool_id: uuid.UUID,
    ) -> BotTool | None:

        result = await db.execute(
            select(BotTool)
            .options(
                selectinload(
                    BotTool.tool
                )
            )
            .where(
                BotTool.bot_id == bot_id,
                BotTool.tool_id == tool_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_bot(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        enabled_only: bool = False,
    ) -> list[BotTool]:

        conditions = [
            BotTool.bot_id == bot_id
        ]

        if enabled_only:
            conditions.append(
                BotTool.is_enabled.is_(True)
            )

        result = await db.execute(
            select(BotTool)
            .options(
                selectinload(
                    BotTool.tool
                )
            )
            .where(*conditions)
        )

        return list(
            result.scalars().all()
        )

    @staticmethod
    async def update_bot_tool(
        db: AsyncSession,
        bot_tool: BotTool,
        **values,
    ) -> BotTool:

        allowed_fields = {
            "config",
            "requires_confirmation",
            "is_enabled",
        }

        for key, value in values.items():
            if key in allowed_fields:
                setattr(
                    bot_tool,
                    key,
                    value,
                )

        await db.flush()
        await db.refresh(bot_tool)

        return bot_tool

    @staticmethod
    async def detach_from_bot(
        db: AsyncSession,
        bot_tool: BotTool,
    ) -> None:

        await db.delete(bot_tool)
        await db.flush()