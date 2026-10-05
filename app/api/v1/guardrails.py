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

from app.schemas.guardrail import (
    GuardrailCreate,
    GuardrailUpdate,
    GuardrailResponse,
    GuardrailListResponse,
    BotGuardrailCreate,
    BotGuardrailUpdate,
    BotGuardrailResponse,
    GuardrailTestRequest,
    GuardrailTestResponse,
)

from app.services import (
    GuardrailManagementService,
)

from app.ai.guardrails import (
    GuardrailService,
    GuardrailStage,
)


router = APIRouter(
    prefix="/guardrails",
    tags=["Guardrails"],
)


# =========================================================
# CREATE DEFINITION
# =========================================================

@router.post(
    "",
    response_model=GuardrailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_guardrail(
    payload: GuardrailCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = GuardrailManagementService()

    guardrail = await service.create(
        db,
        user_id=current_user.id,
        code=payload.code,
        name=payload.name,
        description=payload.description,
        handler=payload.handler,
        guardrail_type=(
            payload.guardrail_type
        ),
        default_config=(
            payload.default_config
        ),
    )

    return GuardrailResponse.model_validate(
        guardrail
    )


# =========================================================
# LIST DEFINITIONS
# =========================================================

@router.get(
    "",
    response_model=GuardrailListResponse,
)
async def list_guardrails(
    active_only: bool = Query(
        default=True
    ),
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = GuardrailManagementService()

    result = await service.list(
        db,
        active_only=active_only,
        page=page,
        page_size=page_size,
    )

    return GuardrailListResponse(
        items=[
            GuardrailResponse.model_validate(
                item
            )
            for item in result.items
        ],
        total=result.total,
        page=page,
        page_size=page_size,
    )


# =========================================================
# GET DEFINITION
# =========================================================

@router.get(
    "/{guardrail_id}",
    response_model=GuardrailResponse,
)
async def get_guardrail(
    guardrail_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = GuardrailManagementService()

    guardrail = await service.get(
        db,
        guardrail_id=guardrail_id,
    )

    return GuardrailResponse.model_validate(
        guardrail
    )


# =========================================================
# UPDATE DEFINITION
# =========================================================

@router.patch(
    "/{guardrail_id}",
    response_model=GuardrailResponse,
)
async def update_guardrail(
    guardrail_id: uuid.UUID,
    payload: GuardrailUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = GuardrailManagementService()

    guardrail = await service.update(
        db,
        guardrail_id=guardrail_id,
        name=payload.name,
        description=payload.description,
        default_config=(
            payload.default_config
        ),
        is_active=payload.is_active,
    )

    return GuardrailResponse.model_validate(
        guardrail
    )


# =========================================================
# ATTACH GUARDRAIL TO BOT
# =========================================================

@router.post(
    "/bots/{bot_id}",
    response_model=BotGuardrailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def attach_guardrail(
    bot_id: uuid.UUID,
    payload: BotGuardrailCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = GuardrailManagementService()

    result = await service.attach_to_bot(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        guardrail_id=(
            payload.guardrail_id
        ),
        action=payload.action,
        priority=payload.priority,
        config=payload.config,
        is_enabled=payload.is_enabled,
    )

    return (
        BotGuardrailResponse
        .model_validate(result)
    )


# =========================================================
# LIST BOT GUARDRAILS
# =========================================================

@router.get(
    "/bots/{bot_id}/configs",
    response_model=list[
        BotGuardrailResponse
    ],
)
async def list_bot_guardrails(
    bot_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = GuardrailManagementService()

    items = await service.list_for_bot(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
    )

    return [
        BotGuardrailResponse.model_validate(
            item
        )
        for item in items
    ]


# =========================================================
# UPDATE BOT GUARDRAIL
# =========================================================

@router.patch(
    "/bots/{bot_id}/{bot_guardrail_id}",
    response_model=BotGuardrailResponse,
)
async def update_bot_guardrail(
    bot_id: uuid.UUID,
    bot_guardrail_id: uuid.UUID,
    payload: BotGuardrailUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = GuardrailManagementService()

    result = (
        await service.update_bot_guardrail(
            db,
            bot_id=bot_id,
            bot_guardrail_id=(
                bot_guardrail_id
            ),
            user_id=current_user.id,
            action=payload.action,
            priority=payload.priority,
            config=payload.config,
            is_enabled=payload.is_enabled,
        )
    )

    return (
        BotGuardrailResponse
        .model_validate(result)
    )


# =========================================================
# REMOVE FROM BOT
# =========================================================

@router.delete(
    "/bots/{bot_id}/{bot_guardrail_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_bot_guardrail(
    bot_id: uuid.UUID,
    bot_guardrail_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = GuardrailManagementService()

    await service.remove_from_bot(
        db,
        bot_id=bot_id,
        bot_guardrail_id=(
            bot_guardrail_id
        ),
        user_id=current_user.id,
    )

    return None


# =========================================================
# TEST BOT GUARDRAILS
# =========================================================

@router.post(
    "/bots/{bot_id}/test",
    response_model=GuardrailTestResponse,
)
async def test_guardrails(
    bot_id: uuid.UUID,
    payload: GuardrailTestRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = GuardrailService()

    stage = GuardrailStage(
        payload.stage
    )

    result = await service.evaluate(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        stage=stage,
        text=payload.text,
        metadata={
            "test_mode": True,
        },
    )

    return GuardrailTestResponse(
        allowed=result.allowed,
        original_text=(
            result.original_text
        ),
        final_text=result.final_text,
        warnings=result.warnings,
        executions=[
            GuardrailExecutionResponse(
                code=item.code,
                handler=item.handler,
                action=item.action.value if hasattr(item.action, "value") else str(item.action),
                passed=item.passed,
                message=item.message,
                details=item.details,
            )
            for item in result.executions
        ],
    )