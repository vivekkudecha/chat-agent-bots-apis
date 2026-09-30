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

from app.schemas.conversation import (
    ConversationCreate,
    ConversationResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationUpdate,
    MessageResponse,
)

from app.services import (
    ConversationService,
)


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


# =========================================================
# CREATE CONVERSATION
# =========================================================

@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ConversationService()

    conversation = await service.create(
        db,
        user_id=current_user.id,
        bot_id=payload.bot_id,
        title=payload.title,
        metadata=payload.metadata_,
    )

    return ConversationResponse.model_validate(conversation)


# =========================================================
# LIST CONVERSATIONS
# =========================================================

@router.get(
    "",
    response_model=ConversationListResponse,
)
async def list_conversations(
    bot_id: uuid.UUID | None = Query(
        default=None
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
    service = ConversationService()

    result = await service.list(
        db,
        user_id=current_user.id,
        bot_id=bot_id,
        page=page,
        page_size=page_size,
    )

    return ConversationListResponse(
        items=[
            ConversationResponse.model_validate(
                conversation
            )
            for conversation in result.items
        ],
        total=result.total,
        page=page,
        page_size=page_size,
    )


# =========================================================
# GET CONVERSATION
# =========================================================

@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
)
async def get_conversation(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ConversationService()

    conversation = await service.get_detail(
        db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    return (
        ConversationDetailResponse
        .model_validate(
            conversation
        )
    )


# =========================================================
# GET MESSAGES
# =========================================================

@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def get_messages(
    conversation_id: uuid.UUID,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    before: uuid.UUID | None = Query(
        default=None
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ConversationService()

    messages = await service.get_messages(
        db,
        conversation_id=conversation_id,
        user_id=current_user.id,
        limit=limit,
        before=before,
    )

    return [
        MessageResponse.model_validate(
            message
        )
        for message in messages
    ]


# =========================================================
# UPDATE CONVERSATION
# =========================================================

@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
async def update_conversation(
    conversation_id: uuid.UUID,
    payload: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ConversationService()

    update_dict = payload.model_dump(exclude_unset=True)

    conversation = await service.update(
        db,
        conversation_id=conversation_id,
        user_id=current_user.id,
        **update_dict,
    )

    return ConversationResponse.model_validate(
        conversation
    )


# =========================================================
# DELETE CONVERSATION
# =========================================================

@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ConversationService()

    await service.delete(
        db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    return None