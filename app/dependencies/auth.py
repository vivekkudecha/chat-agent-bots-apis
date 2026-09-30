import uuid

from fastapi import (
    Depends,
    HTTPException,
    status,
)

from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.repositories.user_repository import (
    UserRepository,
)

from app.core.security import (
    decode_access_token,
)


# =========================================================
# Bearer Scheme
# =========================================================

bearer_scheme = HTTPBearer(
    auto_error=False
)


# =========================================================
# 401 Helper
# =========================================================

def unauthorized(
    detail: str = "Authentication required.",
) -> HTTPException:

    return HTTPException(
        status_code=(
            status.HTTP_401_UNAUTHORIZED
        ),
        detail=detail,
        headers={
            "WWW-Authenticate": "Bearer"
        },
    )


# =========================================================
# CURRENT USER
# =========================================================

async def get_current_user(
    credentials: (
        HTTPAuthorizationCredentials | None
    ) = Depends(bearer_scheme),

    db: AsyncSession = Depends(get_db),

) -> User:

    # -----------------------------------------------------
    # Authorization header
    # -----------------------------------------------------

    if credentials is None:
        raise unauthorized()

    if (
        credentials.scheme.lower()
        != "bearer"
    ):
        raise unauthorized(
            "Invalid authentication scheme."
        )

    token = credentials.credentials

    if not token:
        raise unauthorized()

    # -----------------------------------------------------
    # Decode JWT
    # -----------------------------------------------------

    try:

        payload = decode_access_token(
            token
        )

    except Exception:

        raise unauthorized(
            "Invalid or expired access token."
        )

    if not payload:
        raise unauthorized(
            "Invalid or expired access token."
        )

    # -----------------------------------------------------
    # Token type
    #
    # Prevent refresh tokens from being used as
    # access tokens.
    # -----------------------------------------------------

    token_type = payload.get(
        "type"
    )

    if token_type != "access":

        raise unauthorized(
            "Invalid access token."
        )

    # -----------------------------------------------------
    # Subject
    # -----------------------------------------------------

    subject = payload.get(
        "sub"
    )

    if not subject:

        raise unauthorized(
            "Invalid token subject."
        )

    try:

        user_id = uuid.UUID(
            str(subject)
        )

    except (
        ValueError,
        TypeError,
    ):

        raise unauthorized(
            "Invalid token subject."
        )

    # -----------------------------------------------------
    # User
    # -----------------------------------------------------

    user = (
        await UserRepository.get_by_id(
            db,
            user_id,
        )
    )

    if not user:

        raise unauthorized(
            "User no longer exists."
        )

    # -----------------------------------------------------
    # Active check
    # -----------------------------------------------------

    if not user.is_active:

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "User account is inactive."
            ),
        )

    return user