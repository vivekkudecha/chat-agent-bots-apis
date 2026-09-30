import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    KnowledgeBaseNotFoundException,
)

from app.repositories.knowledge_repository import (
    KnowledgeRepository,
)
from app.repositories.document_repository import (
    DocumentRepository,
)
from app.repositories.audit_repository import (
    AuditRepository,
)

from app.services.vector_store_service import (
    VectorStoreService,
)

from app.integrations.storage import (
    get_storage_provider,
)


@dataclass
class KnowledgeBaseListResult:
    items: list[Any]
    total: int


@dataclass
class KnowledgeBaseDetailResult:
    id: uuid.UUID
    user_id: uuid.UUID

    name: str
    description: str | None

    chunking_config: dict[str, Any]

    document_count: int
    total_chunks: int

    documents: list[Any]

    is_active: bool

    created_at: Any
    updated_at: Any


class KnowledgeBaseService:

    def __init__(self):

        self.vector_store = (
            VectorStoreService()
        )

        self.storage = (
            get_storage_provider()
        )

    # =====================================================
    # CREATE
    # =====================================================

    async def create(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        name: str,
        description: str | None = None,
        embedding_model: str | None = None,
        chunking_config: dict[
            str,
            Any,
        ] | None = None,
        retrieval_config: dict[
            str,
            Any,
        ] | None = None,
    ):

        chunking_config = (
            chunking_config
            or {
                "chunk_size": 800,
                "chunk_overlap": 100,
            }
        )

        retrieval_config = (
            retrieval_config
            or {
                "search_type": "semantic",
                "top_k": 5,
            }
        )

        knowledge_base = (
            await KnowledgeRepository.create(
                db,
                user_id=user_id,
                name=name,
                description=description,
                embedding_model=embedding_model,
                chunking_config=(
                    chunking_config
                ),
                retrieval_config=(
                    retrieval_config
                ),
            )
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            resource_type="knowledge_base",
            resource_id=knowledge_base.id,
            action="knowledge_base.created",
            status="success",
        )

        await db.commit()
        await db.refresh(
            knowledge_base
        )

        return knowledge_base

    # =====================================================
    # GET
    # =====================================================

    async def get(
        self,
        db: AsyncSession,
        *,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
    ):

        knowledge_base = (
            await KnowledgeRepository.get_owned(
                db,
                knowledge_base_id=(
                    knowledge_base_id
                ),
                user_id=user_id,
            )
        )

        if not knowledge_base:

            raise (
                KnowledgeBaseNotFoundException()
            )

        return knowledge_base

    # =====================================================
    # GET DETAIL
    # =====================================================

    async def get_detail(
        self,
        db: AsyncSession,
        *,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> KnowledgeBaseDetailResult:

        knowledge_base = await self.get(
            db,
            knowledge_base_id=(
                knowledge_base_id
            ),
            user_id=user_id,
        )

        documents = (
            await DocumentRepository
            .list_by_knowledge_base(
                db,
                knowledge_base_id=(
                    knowledge_base.id
                ),
            )
        )

        total_chunks = sum(
            (
                getattr(
                    document,
                    "chunk_count",
                    0,
                )
                or 0
            )
            for document in documents
        )

        return KnowledgeBaseDetailResult(
            id=knowledge_base.id,
            user_id=knowledge_base.user_id,

            name=knowledge_base.name,
            description=(
                knowledge_base.description
            ),

            chunking_config=(
                knowledge_base.chunking_config
                or {}
            ),

            document_count=len(
                documents
            ),

            total_chunks=total_chunks,

            documents=documents,

            is_active=getattr(
                knowledge_base,
                "is_active",
                True,
            ),

            created_at=(
                knowledge_base.created_at
            ),

            updated_at=(
                knowledge_base.updated_at
            ),
        )

    # =====================================================
    # LIST
    # =====================================================

    async def list(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> KnowledgeBaseListResult:

        offset = (
            page - 1
        ) * page_size

        items, total = (
            await KnowledgeRepository
            .list_by_user(
                db,
                user_id=user_id,
                offset=offset,
                limit=page_size,
            )
        )

        return KnowledgeBaseListResult(
            items=items,
            total=total,
        )

    # =====================================================
    # UPDATE
    # =====================================================

    async def update(
        self,
        db: AsyncSession,
        *,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
        **update_data,
    ):

        knowledge_base = await self.get(
            db,
            knowledge_base_id=(
                knowledge_base_id
            ),
            user_id=user_id,
        )

        protected_fields = {
            "id",
            "user_id",
            "created_at",
            "updated_at",
        }

        update_data = {
            key: value
            for key, value
            in update_data.items()
            if key not in protected_fields
        }

        knowledge_base = (
            await KnowledgeRepository.update(
                db,
                knowledge_base=(
                    knowledge_base
                ),
                **update_data,
            )
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            resource_type="knowledge_base",
            resource_id=knowledge_base.id,
            action="knowledge_base.updated",
            status="success",
            metadata={
                "fields": list(
                    update_data.keys()
                ),
            },
        )

        await db.commit()

        await db.refresh(
            knowledge_base
        )

        return knowledge_base

    # =====================================================
    # DELETE
    # =====================================================

    async def delete(
        self,
        db: AsyncSession,
        *,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:

        knowledge_base = await self.get(
            db,
            knowledge_base_id=(
                knowledge_base_id
            ),
            user_id=user_id,
        )

        # -------------------------------------------------
        # Get documents before deleting DB records.
        # -------------------------------------------------

        documents = (
            await DocumentRepository
            .list_by_knowledge_base(
                db,
                knowledge_base_id=(
                    knowledge_base.id
                ),
            )
        )

        # -------------------------------------------------
        # Delete vectors from Qdrant
        #
        # Collection remains. Only points belonging
        # to this KB are removed.
        # -------------------------------------------------

        await self.vector_store.delete_knowledge_base(
            user_id=user_id,
            knowledge_base_id=(
                knowledge_base.id
            ),
        )

        # -------------------------------------------------
        # Delete physical files
        # -------------------------------------------------

        for document in documents:

            storage_key = getattr(
                document,
                "storage_key",
                None,
            )

            if not storage_key:
                continue

            try:

                await self.storage.delete(
                    storage_key
                )

            except FileNotFoundError:
                # DB cleanup should still proceed if
                # the local file is already missing.
                pass

        # -------------------------------------------------
        # Audit BEFORE deleting KB.
        #
        # Important if AuditLog.bot/resource relations
        # use foreign keys.
        # -------------------------------------------------

        await AuditRepository.create(
            db,
            user_id=user_id,
            resource_type="knowledge_base",
            resource_id=knowledge_base.id,
            action="knowledge_base.deleted",
            status="success",
            metadata={
                "document_count": len(
                    documents
                ),
            },
        )

        # -------------------------------------------------
        # Delete database entity.
        #
        # BotKnowledgeBase and Documents should ideally
        # use ON DELETE CASCADE.
        # -------------------------------------------------

        await KnowledgeRepository.delete(
            db,
            knowledge_base=(
                knowledge_base
            ),
        )

        await db.commit()