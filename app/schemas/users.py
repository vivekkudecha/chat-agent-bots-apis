import uuid
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
)


# ---------------------------------------------------------
# Base
# ---------------------------------------------------------

class UserBase(BaseModel):
    name: str = Field(
        ...,
        min_length=2,
        max_length=150,
    )

    email: EmailStr


# ---------------------------------------------------------
# Create User
# ---------------------------------------------------------

class UserCreate(UserBase):
    password: SecretStr = Field(
        ...,
        min_length=8,
        max_length=128,
    )
    role: str = Field(
        default="user",
        pattern="^(user|admin)$",
    )


# ---------------------------------------------------------
# Update User
# ---------------------------------------------------------

class UserUpdate(BaseModel):
    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    email: EmailStr | None = None


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------

class UserLogin(BaseModel):
    email: EmailStr
    password: SecretStr


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------
# User Response
# ---------------------------------------------------------

class UserResponse(UserBase):
    id: uuid.UUID

    role: str

    is_active: bool
    is_superuser: bool

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


# ---------------------------------------------------------
# Admin Update
# ---------------------------------------------------------

class UserAdminUpdate(BaseModel):
    role: str | None = Field(
        default=None,
        pattern="^(user|admin)$",
    )

    is_active: bool | None = None

    is_superuser: bool | None = None


# ---------------------------------------------------------
# User List Response
# ---------------------------------------------------------

class UserListResponse(BaseModel):
    items: list[UserResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------
# Password Change
# ---------------------------------------------------------

class PasswordChangeRequest(BaseModel):
    current_password: SecretStr = Field(
        ...,
        min_length=8,
        max_length=128,
    )
    new_password: SecretStr = Field(
        ...,
        min_length=8,
        max_length=128,
    )


# ---------------------------------------------------------
# Aliases
# ---------------------------------------------------------

RegisterRequest = UserCreate
RegisterResponse = UserResponse
LoginRequest = UserLogin
UserUpdateRequest = UserUpdate