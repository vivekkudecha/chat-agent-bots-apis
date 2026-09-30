from pydantic import SecretStr
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BadRequestException,
    EmailAlreadyExistsException,
    InvalidCredentialsException,
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
