import uuid

from fastapi import (
    APIRouter,
    Depends,
    Query,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models import User

from app.schemas.ai_model import (
    AIModelCreate,
    AIModelUpdate,
    AIModelResponse,
    AIModelListResponse,
    BotModelConfigRequest,
    BotModelConfigResponse,
)

from app.services import (
    AIModelService,
)


router = APIRouter(
    prefix="/models",
    tags=["Models"],
)


# =========================================================
# CREATE MODEL
# =========================================================

@router.post(
    "",
    response_model=AIModelResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_model(
    payload: AIModelCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AIModelService()

    model = await service.create(
        db,
        user_id=current_user.id,
        provider=payload.provider,
        name=payload.name,
        model_key=payload.model_key,
        endpoint_url=payload.endpoint_url,
        capabilities=payload.capabilities,
    )

    return AIModelResponse.model_validate(
        model
    )


# =========================================================
# LIST MODELS
# =========================================================

@router.get(
    "",
    response_model=AIModelListResponse,
)
async def list_models(
    active_only: bool = Query(
        default=True
    ),
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
    current_user: User = Depends(get_current_user),
):
    service = AIModelService()

    result = await service.list(
        db,
        user_id=current_user.id,
        active_only=active_only,
        page=page,
        page_size=page_size,
    )

    return AIModelListResponse(
        items=[
            AIModelResponse.model_validate(
                model
            )
            for model in result.items
        ],
        total=result.total,
        page=page,
        page_size=page_size,
    )


# =========================================================
# GET MODEL
# =========================================================

@router.get(
    "/{model_id}",
    response_model=AIModelResponse,
)
async def get_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AIModelService()

    model = await service.get(
        db,
        model_id=model_id,
        user_id=current_user.id,
    )

    return AIModelResponse.model_validate(
        model
    )


# =========================================================
# UPDATE MODEL
# =========================================================

@router.patch(
    "/{model_id}",
    response_model=AIModelResponse,
)
async def update_model(
    model_id: uuid.UUID,
    payload: AIModelUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AIModelService()

    model = await service.update(
        db,
        model_id=model_id,
        user_id=current_user.id,
        name=payload.name,
        endpoint_url=payload.endpoint_url,
        capabilities=payload.capabilities,
        is_active=payload.is_active,
    )

    return AIModelResponse.model_validate(
        model
    )


# =========================================================
# DELETE MODEL
# =========================================================

@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AIModelService()

    await service.delete(
        db,
        model_id=model_id,
        user_id=current_user.id,
    )

    return None


# =========================================================
# CONFIGURE MODEL FOR BOT
# =========================================================

@router.put(
    "/bots/{bot_id}",
    response_model=BotModelConfigResponse,
)
async def configure_bot_model(
    bot_id: uuid.UUID,
    payload: BotModelConfigRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AIModelService()

    config = await service.configure_bot_model(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        model_id=payload.model_id,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        is_primary=payload.is_primary,
    )

    return (
        BotModelConfigResponse
        .model_validate(config)
    )


# =========================================================
# GET BOT MODEL CONFIGS
# =========================================================

@router.get(
    "/bots/{bot_id}/configs",
    response_model=list[
        BotModelConfigResponse
    ],
)
async def get_bot_models(
    bot_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AIModelService()

    configs = (
        await service.list_bot_models(
            db,
            bot_id=bot_id,
            user_id=current_user.id,
        )
    )

    return [
        BotModelConfigResponse.model_validate(
            config
        )
        for config in configs
    ]


# =========================================================
# REMOVE MODEL FROM BOT
# =========================================================

@router.delete(
    "/bots/{bot_id}/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_bot_model(
    bot_id: uuid.UUID,
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AIModelService()

    await service.remove_bot_model(
        db,
        bot_id=bot_id,
        model_id=model_id,
        user_id=current_user.id,
    )

    return None