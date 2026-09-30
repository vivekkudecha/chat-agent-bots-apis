import uuid
from typing import Annotated

from fastapi import Depends
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.repositories.user_repository import UserRepository
from app.core.security import (
    decode_access_token,
    get_user_id_from_token,
)
from app.core.exceptions import (
    AuthenticationException,
    InvalidTokenException,
    PermissionDeniedException,
)


# ---------------------------------------------------------
# Bearer Authentication
# ---------------------------------------------------------

bearer_scheme = HTTPBearer(
    auto_error=False,
)


# ---------------------------------------------------------
# Database Dependency Alias
# ---------------------------------------------------------

DBSession = Annotated[
    AsyncSession,
    Depends(get_db),
]


# ---------------------------------------------------------
# Extract Access Token
# ---------------------------------------------------------

async def get_access_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> str:

    if credentials is None:
        raise AuthenticationException()

    if credentials.scheme.lower() != "bearer":
        raise InvalidTokenException(
            "Bearer authentication is required."
        )

    return credentials.credentials


# ---------------------------------------------------------
# Current User
# ---------------------------------------------------------

async def get_current_user(
    db: DBSession,
    token: Annotated[
        str,
        Depends(get_access_token),
    ],
) -> User:

    try:
        payload = decode_access_token(token)

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
        raise AuthenticationException(
            "User account is inactive."
        )

    return user


# ---------------------------------------------------------
# Current User Alias
# ---------------------------------------------------------

CurrentUser = Annotated[
    User,
    Depends(get_current_user),
]


# ---------------------------------------------------------
# Admin User
# ---------------------------------------------------------

async def get_current_admin(
    current_user: CurrentUser,
) -> User:

    if (
        current_user.role != "admin"
        and not current_user.is_superuser
    ):
        raise PermissionDeniedException(
            "Administrator access required."
        )

    return current_user


AdminUser = Annotated[
    User,
    Depends(get_current_admin),
]


# ---------------------------------------------------------
# Superuser
# ---------------------------------------------------------

async def get_current_superuser(
    current_user: CurrentUser,
) -> User:

    if not current_user.is_superuser:
        raise PermissionDeniedException(
            "Superuser access required."
        )

    return current_user


SuperUser = Annotated[
    User,
    Depends(get_current_superuser),
]