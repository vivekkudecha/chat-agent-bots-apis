import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User


class UserRepository:

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        name: str,
        email: str,
        password_hash: str,
        role: str = "user",
    ) -> User:

        user = User(
            name=name,
            email=email.lower().strip(),
            password_hash=password_hash,
            role=role,
        )

        db.add(user)

        await db.flush()
        await db.refresh(user)

        return user

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        user_id: uuid.UUID,
    ) -> User | None:

        result = await db.execute(
            select(User).where(
                User.id == user_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_email(
        db: AsyncSession,
        email: str,
    ) -> User | None:

        result = await db.execute(
            select(User).where(
                func.lower(User.email)
                == email.lower().strip()
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[User], int]:

        count_result = await db.execute(
            select(func.count(User.id))
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(User)
            .order_by(User.created_at.desc())
            .offset(offset)
            .limit(limit)
        )

        users = list(result.scalars().all())

        return users, total

    @staticmethod
    async def update(
        db: AsyncSession,
        user: User,
        **values,
    ) -> User:

        allowed_fields = {
            "name",
            "email",
            "password_hash",
            "role",
            "is_active",
            "is_superuser",
        }

        for key, value in values.items():
            if key in allowed_fields:
                setattr(user, key, value)

        await db.flush()
        await db.refresh(user)

        return user

    @staticmethod
    async def delete(
        db: AsyncSession,
        user: User,
    ) -> None:

        await db.delete(user)
        await db.flush()