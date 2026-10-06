import uuid

from pydantic import SecretStr
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BadRequestException,
    EmailAlreadyExistsException,
    InvalidCredentialsException,
    UserNotFoundException,
)
from app.core.security import (
    hash_password,
    verify_password,
)
from app.models import User
from app.repositories.user_repository import (
    UserRepository,
)


class UserService:

    # =====================================================
    # UPDATE PROFILE
    # =====================================================

    @staticmethod
    async def update_profile(
        db: AsyncSession,
        *,
        user: User,
        name: str | None = None,
        email: str | None = None,
    ) -> User:

        updates = {}

        if name is not None:
            cleaned_name = name.strip()
            if cleaned_name:
                updates["name"] = cleaned_name

        if email is not None:
            cleaned_email = (
                str(email)
                .lower()
                .strip()
            )
            if cleaned_email != user.email:
                existing = await UserRepository.get_by_email(
                    db,
                    cleaned_email,
                )
                if existing and existing.id != user.id:
                    raise EmailAlreadyExistsException()
                updates["email"] = cleaned_email

        if updates:
            try:
                user = await UserRepository.update(
                    db,
                    user,
                    **updates,
                )
                await db.commit()
                await db.refresh(user)
            except IntegrityError as exc:
                await db.rollback()
                raise EmailAlreadyExistsException() from exc
            except Exception:
                await db.rollback()
                raise

        return user

    # =====================================================
    # CHANGE PASSWORD
    # =====================================================

    @staticmethod
    async def change_password(
        db: AsyncSession,
        *,
        user: User,
        current_password: str | SecretStr,
        new_password: str | SecretStr,
    ) -> None:

        curr_pwd = (
            current_password.get_secret_value()
            if isinstance(current_password, SecretStr)
            else str(current_password)
        )
        new_pwd = (
            new_password.get_secret_value()
            if isinstance(new_password, SecretStr)
            else str(new_password)
        )

        if not verify_password(
            curr_pwd,
            user.password_hash,
        ):
            raise InvalidCredentialsException(
                message="Current password is incorrect."
            )

        if curr_pwd == new_pwd:
            raise BadRequestException(
                message="New password cannot be the same as current password."
            )

        new_hash = hash_password(new_pwd)

        try:
            await UserRepository.update(
                db,
                user,
                password_hash=new_hash,
            )
            await db.commit()
            await db.refresh(user)
        except Exception:
            await db.rollback()
            raise

    # =====================================================
    # LIST USERS (ADMIN)
    # =====================================================

    @staticmethod
    async def list_users(
        db: AsyncSession,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[User], int]:

        offset = (page - 1) * page_size
        return await UserRepository.list(
            db,
            offset=offset,
            limit=page_size,
        )

    # =====================================================
    # GET USER BY ID (ADMIN)
    # =====================================================

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        user_id: uuid.UUID,
    ) -> User:

        user = await UserRepository.get_by_id(
            db,
            user_id,
        )
        if not user:
            raise UserNotFoundException()
        return user

    # =====================================================
    # ADMIN UPDATE USER
    # =====================================================

    @staticmethod
    async def admin_update_user(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        role: str | None = None,
        is_active: bool | None = None,
        is_superuser: bool | None = None,
    ) -> User:

        user = await UserRepository.get_by_id(
            db,
            user_id,
        )
        if not user:
            raise UserNotFoundException()

        updates = {}
        if role is not None:
            updates["role"] = role
        if is_active is not None:
            updates["is_active"] = is_active
        if is_superuser is not None:
            updates["is_superuser"] = is_superuser

        if updates:
            user = await UserRepository.update(
                db,
                user,
                **updates,
            )
            await db.commit()
            await db.refresh(user)

        return user

    # =====================================================
    # DELETE USER (ADMIN)
    # =====================================================

    @staticmethod
    async def delete_user(
        db: AsyncSession,
        user_id: uuid.UUID,
    ) -> None:

        user = await UserRepository.get_by_id(
            db,
            user_id,
        )
        if not user:
            raise UserNotFoundException()

        await UserRepository.delete(
            db,
            user,
        )
        await db.commit()
