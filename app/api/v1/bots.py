import json
from typing import Any
import logging
from pathlib import Path
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

from app.ai.rag import TextExtractionService
from app.config import settings
from app.core.exceptions import (
    ValidationException,
)

from app.database import get_db

from app.dependencies.auth import (
    get_current_user,
    require_admin,
)

from app.models import User

from app.schemas.bot import (
    BotCreateRequest,
    BotUpdateRequest,
    BotVisibility,
    BotResponse,
    BotDetailResponse,
    BotEditResponse,
    BotListResponse,
    BotVersionResponse,
    BotWithDocumentsResponse,
)

from app.services import (
    BotService,
)
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


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
        require_admin
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
    visibility: str = Form(default="public", description="Bot visibility: private, workspace, public"),
    avatar_url: str | None = Form(default=None, description="Avatar image URL"),
    metadata: str | None = Form(default=None, description="JSON string object for extra metadata"),
    knowledge_base_id: uuid.UUID | None = Form(default=None, description="Existing KB ID to attach and upload into"),
    knowledge_base_name: str | None = Form(default=None, description="Custom name if creating a new dedicated KB"),
    files: list[UploadFile] = File(default=[], description="Multiple files up to 200MB"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
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
    is_admin = current_user.role == "admin" or current_user.is_superuser

    result = await service.list(
        db,
        user_id=current_user.id,
        is_admin=is_admin,
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
    is_admin = current_user.role == "admin" or current_user.is_superuser

    bot = await service.get(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        is_admin=is_admin,
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
        require_admin
    ),
):

    service = BotService()

    bot = await service.update(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        is_admin=True,
        name=payload.name,
        description=payload.description,
        visibility=payload.visibility,
        system_instruction=(
            payload.system_instruction
        ),
        welcome_message=payload.welcome_message,
        conversation_starters=payload.conversation_starters,
    )

    return BotDetailResponse.model_validate(
        bot
    )


@router.get(
    "/{bot_id}/edit",
    response_model=BotEditResponse,
)
async def get_bot_edit_info(
    bot_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    service = BotService()
    bot = await service.get(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        is_admin=True,
    )
    links = await KnowledgeRepository.list_for_bot(db, bot_id=bot.id)
    knowledge_base_id = links[0].knowledge_base_id if links else None
    documents = []

    if knowledge_base_id:
        documents, _ = await DocumentRepository.list_by_knowledge_base(
            db,
            knowledge_base_id=knowledge_base_id,
            user_id=current_user.id,
            limit=10000,
        )

    return BotEditResponse(
        **BotDetailResponse.model_validate(bot).model_dump(),
        knowledge_base_id=knowledge_base_id,
        documents=documents,
    )


@router.patch(
    "/{bot_id}/edit",
    response_model=BotEditResponse,
)
async def update_bot_with_documents(
    bot_id: uuid.UUID,
    name: str = Form(..., min_length=2, max_length=150),
    system_instruction: str = Form(..., min_length=1, max_length=50000),
    description: str | None = Form(default=None),
    visibility: BotVisibility = Form(default="private"),
    welcome_message: str | None = Form(default=None),
    conversation_starters: str = Form(default="[]"),
    remove_document_ids: str = Form(default="[]"),
    files: list[UploadFile] = File(default=[]),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    if visibility not in {"private", "organization", "public"}:
        raise ValidationException("Bot visibility is invalid.")

    try:
        starters = json.loads(conversation_starters)
        remove_ids_raw = json.loads(remove_document_ids)
    except json.JSONDecodeError as exc:
        raise ValidationException(
            "Conversation starters and document removals must be valid JSON."
        ) from exc

    if not isinstance(starters, list) or not all(isinstance(item, str) for item in starters):
        raise ValidationException("Conversation starters must be a JSON array of strings.")
    if len(starters) > 10:
        raise ValidationException("A maximum of 10 conversation starters is allowed.")
    if not isinstance(remove_ids_raw, list):
        raise ValidationException("Document removals must be a JSON array of document IDs.")

    try:
        remove_ids = {uuid.UUID(item) for item in remove_ids_raw}
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValidationException("A document removal ID is invalid.") from exc

    bot_service = BotService()
    bot = await bot_service.get(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        is_admin=True,
    )
    links = await KnowledgeRepository.list_for_bot(db, bot_id=bot.id)
    if links:
        knowledge_base_id = links[0].knowledge_base_id
    else:
        knowledge_base_id = None

    old_documents = []
    if knowledge_base_id:
        old_documents, _ = await DocumentRepository.list_by_knowledge_base(
            db,
            knowledge_base_id=knowledge_base_id,
            user_id=current_user.id,
            limit=10000,
        )
    documents_by_id = {document.id: document for document in old_documents}
    unknown_remove_ids = remove_ids - documents_by_id.keys()
    if unknown_remove_ids:
        raise ValidationException(
            "One or more selected documents do not belong to this bot's knowledge base."
        )

    valid_files = [file for file in files if file.filename]
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    filenames = [Path(file.filename or "").name.casefold() for file in valid_files]
    if len(filenames) != len(set(filenames)):
        raise ValidationException(
            "Each uploaded document must have a unique file name."
        )

    for file in valid_files:
        extension = Path(file.filename or "").suffix.lower()
        if extension not in TextExtractionService.SUPPORTED_EXTENSIONS:
            raise ValidationException(
                f"Unsupported file format '{file.filename}'. Supported formats: "
                f"{', '.join(sorted(TextExtractionService.SUPPORTED_EXTENSIONS))}"
            )
        if file.size is not None and file.size > max_bytes:
            raise ValidationException(
                f"File '{file.filename}' exceeds the maximum allowed size of "
                f"{settings.MAX_UPLOAD_SIZE_MB}MB."
            )

    if knowledge_base_id and (valid_files or remove_ids):
        bot_link_count = await KnowledgeRepository.count_bot_links(
            db,
            knowledge_base_id=knowledge_base_id,
        )
        if bot_link_count > 1:
            raise ValidationException(
                "Documents in a shared knowledge base cannot be replaced or removed from one bot. "
                "Detach or copy the knowledge base before editing its documents."
            )

    if not knowledge_base_id and (valid_files or remove_ids):
        knowledge_base = await KnowledgeRepository.create(
            db,
            user_id=current_user.id,
            name=f"{bot.name} Knowledge Base",
            description=f"Auto-created knowledge base for bot {bot.name}",
        )
        await db.commit()
        await bot_service.attach_knowledge_base(
            db,
            bot_id=bot.id,
            knowledge_base_id=knowledge_base.id,
            user_id=current_user.id,
        )
        knowledge_base_id = knowledge_base.id

    document_service = DocumentService()
    new_documents = []
    replacement_ids = set(remove_ids)

    try:
        for file in valid_files:
            document = await document_service.upload(
                db,
                user_id=current_user.id,
                knowledge_base_id=knowledge_base_id,
                file=file,
            )
            await db.commit()
            await db.refresh(document)
            new_documents.append(document)

            processed_document = await document_service.process_document(
                db,
                document_id=document.id,
            )
            if processed_document.status != "ready":
                raise ValidationException(
                    f"Document '{file.filename}' did not finish processing."
                )

            uploaded_name = Path(file.filename or "").name.casefold()
            replacement_ids.update(
                old_document.id
                for old_document in old_documents
                if Path(old_document.original_name).name.casefold() == uploaded_name
            )

        documents_to_remove = [
            documents_by_id[document_id]
            for document_id in replacement_ids
            if document_id in documents_by_id
        ]
        documents_in_progress = [
            document.original_name
            for document in documents_to_remove
            if document.status in {"uploaded", "queued", "processing"}
        ]
        if documents_in_progress:
            raise ValidationException(
                "Wait for the existing document processing to finish before replacing or removing: "
                + ", ".join(documents_in_progress)
            )
    except Exception:
        for document in new_documents:
            try:
                await document_service.delete_document(
                    db,
                    user_id=current_user.id,
                    document_id=document.id,
                )
            except Exception:
                logger.exception(
                    "Failed to clean up newly uploaded document %s after edit failure",
                    document.id,
                )
        raise

    try:
        updated_bot = await bot_service.update(
            db,
            bot_id=bot.id,
            user_id=current_user.id,
            is_admin=True,
            name=name,
            description=description,
            visibility=visibility,
            system_instruction=system_instruction,
            welcome_message=welcome_message,
            conversation_starters=starters,
        )
    except Exception:
        for document in new_documents:
            try:
                await document_service.delete_document(
                    db,
                    user_id=current_user.id,
                    document_id=document.id,
                )
            except Exception:
                logger.exception(
                    "Failed to clean up newly uploaded document %s after bot update failure",
                    document.id,
                )
        raise

    for document_id in replacement_ids:
        await document_service.delete_document(
            db,
            user_id=current_user.id,
            document_id=document_id,
        )

    remaining_documents = []
    if knowledge_base_id:
        remaining_documents, _ = await DocumentRepository.list_by_knowledge_base(
            db,
            knowledge_base_id=knowledge_base_id,
            user_id=current_user.id,
            limit=10000,
        )
    return BotEditResponse(
        **BotDetailResponse.model_validate(updated_bot).model_dump(),
        knowledge_base_id=knowledge_base_id,
        documents=remaining_documents,
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
        require_admin
    ),
):

    service = BotService()

    await service.delete(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        is_admin=True,
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
        require_admin
    ),
):

    service = BotService()

    version = await service.publish(
        db,
        bot_id=bot_id,
        user_id=current_user.id,
        is_admin=True,
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
        require_admin
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
        is_admin=True,
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
        require_admin
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
        is_admin=True,
    )

    return None