import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models import User

from app.schemas.document import (
    DocumentResponse,
    DocumentDetailResponse,
    DocumentListResponse,
)

from app.services.document_service import (
    DocumentService,
)

from app.workers.document_tasks import (
    process_document_task,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# =========================================================
# UPLOAD
# =========================================================

@router.post(
    "/knowledge-bases/{knowledge_base_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    knowledge_base_id: uuid.UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = DocumentService()

    document = await service.upload(
        db,
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
        file=file,
    )

    # -----------------------------------------------------
    # Commit BEFORE queueing.
    #
    # Otherwise the Celery worker may execute before the
    # Document row becomes visible in PostgreSQL.
    # -----------------------------------------------------

    await db.commit()

    await db.refresh(
        document
    )

    # -----------------------------------------------------
    # Queue processing
    # -----------------------------------------------------

    process_document_task.delay(
        str(document.id)
    )

    return DocumentResponse.model_validate(
        document
    )


# =========================================================
# LIST DOCUMENTS FOR KB
# =========================================================

@router.get(
    "/knowledge-bases/{knowledge_base_id}",
    response_model=DocumentListResponse,
)
async def list_documents(
    knowledge_base_id: uuid.UUID,
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
    service = DocumentService()

    result = await service.list(
        db,
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
        page=page,
        page_size=page_size,
    )

    return DocumentListResponse(
        items=[
            DocumentResponse.model_validate(
                document
            )
            for document in result.items
        ],
        total=result.total,
        page=page,
        page_size=page_size,
    )


# =========================================================
# GET DOCUMENT
# =========================================================

@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = DocumentService()

    document = await service.get(
        db,
        document_id=document_id,
        user_id=current_user.id,
    )

    return DocumentDetailResponse.model_validate(
        document
    )


# =========================================================
# RETRY PROCESSING
# =========================================================

@router.post(
    "/{document_id}/retry",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = DocumentService()

    document = await service.prepare_retry(
        db,
        document_id=document_id,
        user_id=current_user.id,
    )

    await db.commit()

    await db.refresh(
        document
    )

    process_document_task.delay(
        str(document.id)
    )

    return DocumentResponse.model_validate(
        document
    )


# =========================================================
# DELETE DOCUMENT
# =========================================================

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = DocumentService()

    await service.delete_document(
        db,
        document_id=document_id,
        user_id=current_user.id,
    )

    await db.commit()

    return None