from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import Any
import uuid

import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from app.config import settings


# ---------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------

password_hasher = PasswordHash.recommended()


def hash_password(
    password: str,
) -> str:
    """
    Hash plaintext password using the recommended
    pwdlib password hashing configuration.
    """

    return password_hasher.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    Verify plaintext password against stored hash.
    """

    return password_hasher.verify(
        plain_password,
        hashed_password,
    )


# ---------------------------------------------------------
# JWT
# ---------------------------------------------------------

def create_access_token(
    *,
    user_id: uuid.UUID,
    extra_claims: dict[str, Any] | None = None,
) -> str:

    now = datetime.now(timezone.utc)

    expires_at = now + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "type": "access",
        "iat": now,
        "exp": expires_at,
    }

    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def create_refresh_token(
    *,
    user_id: uuid.UUID,
) -> str:

    now = datetime.now(timezone.utc)

    expires_at = now + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )

    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": now,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


# ---------------------------------------------------------
# Decode Token
# ---------------------------------------------------------

def decode_token(
    token: str,
) -> dict[str, Any]:

    try:

        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[
                settings.JWT_ALGORITHM
            ],
        )

        return payload

    except InvalidTokenError as exc:

        raise ValueError(
            "Invalid or expired token"
        ) from exc


# ---------------------------------------------------------
# Access Token Validation
# ---------------------------------------------------------

def decode_access_token(
    token: str,
) -> dict[str, Any]:

    payload = decode_token(token)

    if payload.get("type") != "access":
        raise ValueError(
            "Invalid token type"
        )

    if not payload.get("sub"):
        raise ValueError(
            "Token subject is missing"
        )

    return payload


# ---------------------------------------------------------
# Refresh Token Validation
# ---------------------------------------------------------

def decode_refresh_token(
    token: str,
) -> dict[str, Any]:

    payload = decode_token(token)

    if payload.get("type") != "refresh":
        raise ValueError(
            "Invalid token type"
        )

    if not payload.get("sub"):
        raise ValueError(
            "Token subject is missing"
        )

    return payload


# ---------------------------------------------------------
# Extract User ID
# ---------------------------------------------------------

def get_user_id_from_token(
    payload: dict[str, Any],
) -> uuid.UUID:

    subject = payload.get("sub")

    if not subject:
        raise ValueError(
            "Token subject is missing"
        )

    try:
        return uuid.UUID(subject)

    except ValueError as exc:
        raise ValueError(
            "Invalid token subject"
        ) from exc