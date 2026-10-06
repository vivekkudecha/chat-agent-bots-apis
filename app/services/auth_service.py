from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User
from app.repositories.user_repository import (
    UserRepository,
)
from app.schemas import (
    UserCreate,
    UserLogin,
    TokenResponse,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_user_id_from_token,
    hash_password,
    verify_password,
)
from app.core.exceptions import (
    EmailAlreadyExistsException,
    InvalidCredentialsException,
    InvalidTokenException,
)


class AuthService:

    # =====================================================
    # REGISTER
    # =====================================================

    @staticmethod
    async def register(
        db: AsyncSession,
        data: UserCreate,
    ) -> User:

        email = (
            str(data.email)
            .lower()
            .strip()
        )

        existing_user = (
            await UserRepository.get_by_email(
                db,
                email,
            )
        )

        if existing_user:
            raise EmailAlreadyExistsException()

        password_hash = hash_password(
            data.password.get_secret_value()
        )

        role = getattr(data, "role", "user") or "user"

        try:

            user = await UserRepository.create(
                db,
                name=data.name.strip(),
                email=email,
                password_hash=password_hash,
                role=role,
            )

            await db.commit()

            await db.refresh(user)

            return user

        except IntegrityError as exc:

            await db.rollback()

            raise EmailAlreadyExistsException() from exc

        except Exception:

            await db.rollback()
            raise

    # =====================================================
    # LOGIN
    # =====================================================

    @staticmethod
    async def login(
        db: AsyncSession,
        data: UserLogin,
    ) -> TokenResponse:

        email = (
            str(data.email)
            .lower()
            .strip()
        )

        user = (
            await UserRepository.get_by_email(
                db,
                email,
            )
        )

        # Intentionally same response for:
        # - unknown email
        # - incorrect password

        if not user:
            raise InvalidCredentialsException()

        if not verify_password(
            data.password.get_secret_value(),
            user.password_hash,
        ):
            raise InvalidCredentialsException()

        if not user.is_active:
            raise InvalidCredentialsException()

        access_token = create_access_token(
            user_id=user.id,
        )

        refresh_token = create_refresh_token(
            user_id=user.id,
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
        )

    # =====================================================
    # REFRESH TOKEN
    # =====================================================

    @staticmethod
    async def refresh(
        db: AsyncSession,
        refresh_token: str,
    ) -> TokenResponse:

        try:

            payload = decode_refresh_token(
                refresh_token
            )

            user_id = get_user_id_from_token(
                payload
            )

        except ValueError as exc:

            raise InvalidTokenException() from exc

        user = await UserRepository.get_by_id(
            db,
            user_id,
        )

        if not user:
            raise InvalidTokenException()

        if not user.is_active:
            raise InvalidTokenException()

        new_access_token = (
            create_access_token(
                user_id=user.id,
            )
        )

        new_refresh_token = (
            create_refresh_token(
                user_id=user.id,
            )
        )

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
        )

    # =====================================================
    # CURRENT USER
    # =====================================================

    @staticmethod
    async def get_user(
        db: AsyncSession,
        user_id,
    ) -> User:

        user = await UserRepository.get_by_id(
            db,
            user_id,
        )

        if not user:
            raise InvalidTokenException()

        return user