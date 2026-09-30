from fastapi import (
    APIRouter,
    Depends,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

from app.dependencies.auth import (
    get_current_user,
)

from app.models import User

from app.schemas import (
    UserResponse,
    UserUpdateRequest,
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