import json
from typing import Any
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    UploadFile,
    status,
)

from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from app.core.exceptions import (
    ValidationException,
)

from app.database import get_db

from app.dependencies.auth import (
    get_current_user,
)

from app.models import User

from app.schemas.bot import (
    BotCreateRequest,
    BotUpdateRequest,
    BotResponse,
    BotDetailResponse,
    BotListResponse,
    BotVersionResponse,
    BotWithDocumentsResponse,
)

from app.services import (
    BotService,
)


router = APIRouter(
    prefix="/bots",
    tags=["Bots"],
)


# =========================================================
# CREATE BOT
# =========================================================

@router.post(
    "",
    response_model=BotDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_bot(
    payload: BotCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = BotService()

    bot = await service.create(
        db,
        user_id=current_user.id,
        name=payload.name,
        slug=payload.slug,
        description=payload.description,
        system_instruction=payload.system_instruction,
        welcome_message=payload.welcome_message,
        conversation_starters=payload.conversation_starters,
        visibility=payload.visibility,
        avatar_url=payload.avatar_url,
        metadata=payload.metadata_,
    )

    return BotDetailResponse.model_validate(
        bot
    )


# =========================================================
# CREATE BOT WITH DOCUMENTS (CHUNKING & EMBEDDING)
# =========================================================

@router.post(
    "/with-documents",
    response_model=BotWithDocumentsResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_bot_with_documents(
    name: str = Form(..., description="Bot name"),
    slug: str | None = Form(default=None, description="Unique slug for the bot"),
    description: str | None = Form(default=None, description="Bot description"),
    system_instruction: str | None = Form(default=None, description="System instruction / prompt"),
    welcome_message: str | None = Form(default=None, description="Greeting message"),
    conversation_starters: str | None = Form(default=None, description="JSON array or newline/comma-separated conversation starters"),
    visibility: str = Form(default="private", description="Bot visibility: private, workspace, public"),
    avatar_url: str | None = Form(default=None, description="Avatar image URL"),
    metadata: str | None = Form(default=None, description="JSON string object for extra metadata"),
    knowledge_base_id: uuid.UUID | None = Form(default=None, description="Existing KB ID to attach and upload into"),
    knowledge_base_name: str | None = Form(default=None, description="Custom name if creating a new dedicated KB"),
    files: list[UploadFile] = File(default=[], description="Multiple files up to 200MB"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new bot and simultaneously upload, extract, chunk, embed,
    and index multiple documents (up to 200MB) in Qdrant vector database.
    """
    parsed_starters: list[str] | None = None
    if conversation_starters:
        trimmed = conversation_starters.strip()
        if trimmed.startswith("[") and trimmed.endswith("]"):
            try:
                loaded = json.loads(trimmed)
                if isinstance(loaded, list):
                    parsed_starters = [str(x) for x in loaded]
            except Exception:
                parsed_starters = [s.strip() for s in trimmed.strip("[]").split(",") if s.strip()]
        else:
            parsed_starters = [s.strip() for s in trimmed.splitlines() if s.strip()] or [trimmed]

    parsed_metadata: dict[str, Any] | None = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
            if not isinstance(parsed_metadata, dict):
                raise ValidationException("Form field 'metadata' must be a valid JSON object.")
        except json.JSONDecodeError as exc:
            raise ValidationException(f"Invalid JSON in 'metadata': {exc.msg}") from exc

    service = BotService()

    result = await service.create_with_documents(
        db,
        user_id=current_user.id,
        name=name,
        slug=slug,
        description=description,
        system_instruction=system_instruction,
        welcome_message=welcome_message,
        conversation_starters=parsed_starters,
        visibility=visibility,
        avatar_url=avatar_url,
        metadata=parsed_metadata,
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=knowledge_base_name,
        files=files,
    )

    return BotWithDocumentsResponse.model_validate(result)


# =========================================================
# LIST BOTS
# =========================================================

@router.get(
    "",
    response_model=BotListResponse,
)
async def list_bots(
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
    current_user: User = Depends(
        get_current_user
    ),
):

    service = BotService()

    result = await service.list(
        db,
        user_id=current_user.id,
        page=page,
        page_size=page_size,
    )

    return BotListResponse(
        items=[
            BotResponse.model_validate(
                bot
            )
            for bot in result.items
        ],
        total=result.total,
        page=page,
        page_size=page_size,
    )


# =========================================================
# GET BOT
# =========================================================

@router.get(
    "/{bot_id}",
    response_model=BotDetailResponse,
)
async def get_bot(
    bot_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = BotService()

    bot = await service.get(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
    )

    return BotDetailResponse.model_validate(
        bot
    )


# =========================================================
# UPDATE BOT
# =========================================================

@router.patch(
    "/{bot_id}",
    response_model=BotDetailResponse,
)
async def update_bot(
    bot_id: uuid.UUID,
    payload: BotUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = BotService()

    bot = await service.update(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        name=payload.name,
        description=payload.description,
        system_instruction=(
            payload.system_instruction
        ),
    )

    return BotDetailResponse.model_validate(
        bot
    )


# =========================================================
# DELETE BOT
# =========================================================

@router.delete(
    "/{bot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_bot(
    bot_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = BotService()

    await service.delete(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
    )

    return None


# =========================================================
# PUBLISH BOT VERSION
# =========================================================

@router.post(
    "/{bot_id}/publish",
    response_model=BotVersionResponse,
)
async def publish_bot(
    bot_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = BotService()

    version = await service.publish(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
    )

    return BotVersionResponse.model_validate(
        version
    )


# =========================================================
# LIST VERSIONS
# =========================================================

@router.get(
    "/{bot_id}/versions",
    response_model=list[
        BotVersionResponse
    ],
)
async def list_versions(
    bot_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = BotService()

    versions = await service.list_versions(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
    )

    return [
        BotVersionResponse.model_validate(
            version
        )
        for version in versions
    ]


# =========================================================
# ATTACH KNOWLEDGE BASE
# =========================================================

@router.post(
    "/{bot_id}/knowledge-bases/{knowledge_base_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def attach_knowledge_base(
    bot_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = BotService()

    await service.attach_knowledge_base(
        db,
        bot_id=bot_id,
        knowledge_base_id=(
            knowledge_base_id
        ),
        user_id=current_user.id,
    )

    return None


# =========================================================
# DETACH KNOWLEDGE BASE
# =========================================================

@router.delete(
    "/{bot_id}/knowledge-bases/{knowledge_base_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def detach_knowledge_base(
    bot_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    service = BotService()

    await service.detach_knowledge_base(
        db,
        bot_id=bot_id,
        knowledge_base_id=(
            knowledge_base_id
        ),
        user_id=current_user.id,
    )

    return None