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

from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    KnowledgeBaseDetailResponse,
    KnowledgeBaseListResponse,
)

from app.services import (
    KnowledgeBaseService,
)


router = APIRouter(
    prefix="/knowledge-bases",
    tags=["Knowledge Bases"],
)


# =========================================================
# CREATE
# =========================================================

@router.post(
    "",
    response_model=KnowledgeBaseDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = KnowledgeBaseService()

    kb = await service.create(
        db,
        user_id=current_user.id,
        name=payload.name,
        description=payload.description,
        embedding_model=payload.embedding_model,
        chunking_config=payload.chunking_config.model_dump() if payload.chunking_config else None,
        retrieval_config=payload.retrieval_config.model_dump() if payload.retrieval_config else None,
    )

    detail = await service.get_detail(
        db,
        knowledge_base_id=kb.id,
        user_id=current_user.id,
    )

    return KnowledgeBaseDetailResponse.model_validate(
        detail
    )


# =========================================================
# LIST
# =========================================================

@router.get(
    "",
    response_model=KnowledgeBaseListResponse,
)
async def list_knowledge_bases(
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
    service = KnowledgeBaseService()

    result = await service.list(
        db,
        user_id=current_user.id,
        page=page,
        page_size=page_size,
    )

    return KnowledgeBaseListResponse(
        items=[
            KnowledgeBaseResponse.model_validate(
                kb
            )
            for kb in result.items
        ],
        total=result.total,
        page=page,
        page_size=page_size,
    )


# =========================================================
# GET
# =========================================================

@router.get(
    "/{knowledge_base_id}",
    response_model=KnowledgeBaseDetailResponse,
)
async def get_knowledge_base(
    knowledge_base_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = KnowledgeBaseService()

    detail = await service.get_detail(
        db,
        knowledge_base_id=knowledge_base_id,
        user_id=current_user.id,
    )

    return KnowledgeBaseDetailResponse.model_validate(
        detail
    )


# =========================================================
# UPDATE
# =========================================================

@router.patch(
    "/{knowledge_base_id}",
    response_model=KnowledgeBaseDetailResponse,
)
async def update_knowledge_base(
    knowledge_base_id: uuid.UUID,
    payload: KnowledgeBaseUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = KnowledgeBaseService()

    updates = {}
    if payload.name is not None:
        updates["name"] = payload.name
    if payload.description is not None:
        updates["description"] = payload.description
    if payload.chunking_config is not None:
        updates["chunking_config"] = payload.chunking_config.model_dump()
    if payload.retrieval_config is not None:
        updates["retrieval_config"] = payload.retrieval_config.model_dump()

    await service.update(
        db,
        knowledge_base_id=knowledge_base_id,
        user_id=current_user.id,
        **updates,
    )

    detail = await service.get_detail(
        db,
        knowledge_base_id=knowledge_base_id,
        user_id=current_user.id,
    )

    return KnowledgeBaseDetailResponse.model_validate(
        detail
    )


# =========================================================
# DELETE
# =========================================================

@router.delete(
    "/{knowledge_base_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_knowledge_base(
    knowledge_base_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = KnowledgeBaseService()

    await service.delete(
        db,
        knowledge_base_id=knowledge_base_id,
        user_id=current_user.id,
    )

    return None