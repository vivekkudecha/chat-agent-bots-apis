import uuid

from fastapi import (
    APIRouter,
    Depends,
    Query,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

from app.dependencies.auth import (
    get_current_user,
    require_admin,
)

from app.models import User

from app.schemas import (
    UserResponse,
    UserListResponse,
    UserUpdateRequest,
    UserAdminUpdate,
    PasswordChangeRequest,
)

from app.services import (
    UserService,
)


router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


# =========================================================
# CURRENT USER
# =========================================================

@router.get(
    "/me",
    response_model=UserResponse,
)
async def get_me(
    current_user: User = Depends(
        get_current_user
    ),
):

    return UserResponse.model_validate(
        current_user
    )


# =========================================================
# UPDATE PROFILE
# =========================================================

@router.patch(
    "/me",
    response_model=UserResponse,
)
async def update_me(
    payload: UserUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = UserService()

    user = await service.update_profile(
        db,
        user=current_user,
        name=payload.name,
        email=payload.email,
    )

    return UserResponse.model_validate(
        user
    )


# =========================================================
# CHANGE PASSWORD
# =========================================================

@router.post(
    "/me/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def change_password(
    payload: PasswordChangeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = UserService()

    await service.change_password(
        db,
        user=current_user,
        current_password=(
            payload.current_password
        ),
        new_password=(
            payload.new_password
        ),
    )

    return None


# =========================================================
# LIST USERS (ADMIN ONLY)
# =========================================================

@router.get(
    "",
    response_model=UserListResponse,
)
async def list_users(
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin),
):

    service = UserService()

    users, total = await service.list_users(
        db,
        page=page,
        page_size=page_size,
    )

    return UserListResponse(
        items=[
            UserResponse.model_validate(user)
            for user in users
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


# =========================================================
# GET USER BY ID (ADMIN ONLY)
# =========================================================

@router.get(
    "/{user_id}",
    response_model=UserResponse,
)
async def get_user_by_id(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin),
):

    service = UserService()

    user = await service.get_by_id(
        db,
        user_id=user_id,
    )

    return UserResponse.model_validate(user)


# =========================================================
# ADMIN UPDATE USER (ADMIN ONLY)
# =========================================================

@router.patch(
    "/{user_id}",
    response_model=UserResponse,
)
async def admin_update_user(
    user_id: uuid.UUID,
    payload: UserAdminUpdate,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin),
):

    service = UserService()

    user = await service.admin_update_user(
        db,
        user_id=user_id,
        role=payload.role,
        is_active=payload.is_active,
        is_superuser=payload.is_superuser,
    )

    return UserResponse.model_validate(user)


# =========================================================
# DELETE USER (ADMIN ONLY)
# =========================================================

@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin),
):

    service = UserService()

    await service.delete_user(
        db,
        user_id=user_id,
    )

    return None