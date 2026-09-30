import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.usage import UsageLog


class UsageRepository:

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        user_id: uuid.UUID | None,
        provider: str,
        model_name: str,
        bot_id: uuid.UUID | None = None,
        conversation_id: uuid.UUID | None = None,
        model_id: uuid.UUID | None = None,
        operation: str = "chat",
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: Decimal | float = 0,
        currency: str = "USD",
        latency_ms: int | None = None,
        status: str = "success",
        metadata: dict | None = None,
    ) -> UsageLog:

        total_tokens = (
            input_tokens + output_tokens
        )

        usage = UsageLog(
            user_id=user_id,
            bot_id=bot_id,
            conversation_id=conversation_id,
            model_id=model_id,
            provider=provider,
            model_name=model_name,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=Decimal(
                str(estimated_cost)
            ),
            currency=currency,
            latency_ms=latency_ms,
            status=status,
            metadata_=metadata or {},
        )

        db.add(usage)

        await db.flush()
        await db.refresh(usage)

        return usage

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        usage_id: uuid.UUID,
    ) -> UsageLog | None:

        result = await db.execute(
            select(UsageLog).where(
                UsageLog.id == usage_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_user(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[UsageLog], int]:

        count_result = await db.execute(
            select(
                func.count(UsageLog.id)
            ).where(
                UsageLog.user_id == user_id
            )
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(UsageLog)
            .where(
                UsageLog.user_id == user_id
            )
            .order_by(
                UsageLog.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return (
            list(result.scalars().all()),
            total,
        )

    @staticmethod
    async def list_by_bot(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[UsageLog], int]:

        count_result = await db.execute(
            select(
                func.count(UsageLog.id)
            ).where(
                UsageLog.bot_id == bot_id
            )
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(UsageLog)
            .where(
                UsageLog.bot_id == bot_id
            )
            .order_by(
                UsageLog.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return (
            list(result.scalars().all()),
            total,
        )

    @staticmethod
    async def get_user_totals(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> dict:

        conditions = [
            UsageLog.user_id == user_id
        ]

        if start_at:
            conditions.append(
                UsageLog.created_at >= start_at
            )

        if end_at:
            conditions.append(
                UsageLog.created_at < end_at
            )

        result = await db.execute(
            select(
                func.coalesce(
                    func.sum(
                        UsageLog.input_tokens
                    ),
                    0,
                ).label("input_tokens"),

                func.coalesce(
                    func.sum(
                        UsageLog.output_tokens
                    ),
                    0,
                ).label("output_tokens"),

                func.coalesce(
                    func.sum(
                        UsageLog.total_tokens
                    ),
                    0,
                ).label("total_tokens"),

                func.coalesce(
                    func.sum(
                        UsageLog.estimated_cost
                    ),
                    0,
                ).label("estimated_cost"),

                func.count(
                    UsageLog.id
                ).label("requests"),
            )
            .where(*conditions)
        )

        row = result.one()

        return {
            "input_tokens": int(
                row.input_tokens or 0
            ),
            "output_tokens": int(
                row.output_tokens or 0
            ),
            "total_tokens": int(
                row.total_tokens or 0
            ),
            "estimated_cost": Decimal(
                str(
                    row.estimated_cost or 0
                )
            ),
            "requests": int(
                row.requests or 0
            ),
        }

    @staticmethod
    async def get_bot_totals(
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> dict:

        conditions = [
            UsageLog.bot_id == bot_id
        ]

        if start_at:
            conditions.append(
                UsageLog.created_at >= start_at
            )

        if end_at:
            conditions.append(
                UsageLog.created_at < end_at
            )

        result = await db.execute(
            select(
                func.coalesce(
                    func.sum(
                        UsageLog.input_tokens
                    ),
                    0,
                ).label("input_tokens"),

                func.coalesce(
                    func.sum(
                        UsageLog.output_tokens
                    ),
                    0,
                ).label("output_tokens"),

                func.coalesce(
                    func.sum(
                        UsageLog.total_tokens
                    ),
                    0,
                ).label("total_tokens"),

                func.coalesce(
                    func.sum(
                        UsageLog.estimated_cost
                    ),
                    0,
                ).label("estimated_cost"),

                func.count(
                    UsageLog.id
                ).label("requests"),
            )
            .where(*conditions)
        )

        row = result.one()

        return {
            "input_tokens": int(
                row.input_tokens or 0
            ),
            "output_tokens": int(
                row.output_tokens or 0
            ),
            "total_tokens": int(
                row.total_tokens or 0
            ),
            "estimated_cost": Decimal(
                str(
                    row.estimated_cost or 0
                )
            ),
            "requests": int(
                row.requests or 0
            ),
        }