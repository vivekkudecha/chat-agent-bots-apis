import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditLog


class AuditRepository:

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        action: str,
        resource_type: str,
        user_id: uuid.UUID | None = None,
        bot_id: uuid.UUID | None = None,
        resource_id: uuid.UUID | None = None,
        status: str = "success",
        ip_address: str | None = None,
        user_agent: str | None = None,
        description: str | None = None,
        old_values: dict[str, Any] | None = None,
        new_values: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:

        log = AuditLog(
            user_id=user_id,
            bot_id=bot_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            description=description,
            old_values=old_values,
            new_values=new_values,
            metadata_=metadata or {},
        )

        db.add(log)

        await db.flush()
        await db.refresh(log)

        return log

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        audit_id: uuid.UUID,
    ) -> AuditLog | None:

        result = await db.execute(
            select(AuditLog).where(
                AuditLog.id == audit_id
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
    ) -> tuple[list[AuditLog], int]:

        conditions = [
            AuditLog.user_id == user_id
        ]

        count_result = await db.execute(
            select(
                func.count(AuditLog.id)
            ).where(*conditions)
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(AuditLog)
            .where(*conditions)
            .order_by(
                AuditLog.created_at.desc()
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
    ) -> tuple[list[AuditLog], int]:

        conditions = [
            AuditLog.bot_id == bot_id
        ]

        count_result = await db.execute(
            select(
                func.count(AuditLog.id)
            ).where(*conditions)
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(AuditLog)
            .where(*conditions)
            .order_by(
                AuditLog.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return (
            list(result.scalars().all()),
            total,
        )

    @staticmethod
    async def list_by_resource(
        db: AsyncSession,
        *,
        resource_type: str,
        resource_id: uuid.UUID,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[AuditLog], int]:

        conditions = [
            AuditLog.resource_type
            == resource_type,

            AuditLog.resource_id
            == resource_id,
        ]

        count_result = await db.execute(
            select(
                func.count(AuditLog.id)
            ).where(*conditions)
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(AuditLog)
            .where(*conditions)
            .order_by(
                AuditLog.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return (
            list(result.scalars().all()),
            total,
        )