import uuid

from fastapi import (
    APIRouter,
    Depends,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user

from app.models import User

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSourceResponse,
    ChatUsageResponse,
)

from app.services.chat_service import (
    ChatService,
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# =========================================================
# CHAT
# =========================================================

@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
) -> ChatResponse:

    service = ChatService()

    result = await service.chat(
        db,
        user_id=current_user.id,
        bot_id=payload.bot_id,
        message=payload.message,
        conversation_id=(
            payload.conversation_id
        ),
        enable_web_search=payload.is_web_search_enabled,
    )

    return ChatResponse(
        conversation_id=(
            result.conversation_id
        ),
        message_id=result.message_id,
        content=result.content,
        model=result.model,

        usage=ChatUsageResponse(
            input_tokens=(
                result.input_tokens
            ),
            output_tokens=(
                result.output_tokens
            ),
            total_tokens=(
                result.total_tokens
                or (
                    result.input_tokens
                    + result.output_tokens
                )
            ),
            prompt_tokens=(
                result.input_tokens
            ),
            completion_tokens=(
                result.output_tokens
            ),
        ),

        latency_ms=result.latency_ms,

        sources=[
            ChatSourceResponse(
                document_id=(
                    source.document_id
                ),
                knowledge_base_id=(
                    source
                    .knowledge_base_id
                ),
                file_name=(
                    source.file_name
                ),
                page=source.page,
                pages=source.pages,
                score=source.score,
                chunk_count=(
                    source.chunk_count
                ),
                content_preview=(
                    source.content_preview
                ),
                metadata=source.metadata,
            )
            for source in result.sources
        ],

        tool_calls=result.tool_calls,

        warnings=result.warnings,
    )