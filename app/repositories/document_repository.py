import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document


class DocumentRepository:

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        original_name: str,
        storage_key: str,
        mime_type: str | None = None,
        file_size: int | None = None,
        storage_provider: str = "local",
        checksum: str | None = None,
        extraction_metadata: dict | None = None,
    ) -> Document:

        document = Document(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            original_name=original_name,
            mime_type=mime_type,
            file_size=file_size,
            storage_provider=storage_provider,
            storage_key=storage_key,
            checksum=checksum,
            status="uploaded",
            chunk_count=0,
            extraction_metadata=(
                extraction_metadata or {}
            ),
        )

        db.add(document)

        await db.flush()
        await db.refresh(document)

        return document

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        document_id: uuid.UUID,
    ) -> Document | None:

        result = await db.execute(
            select(Document).where(
                Document.id == document_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_owned(
        db: AsyncSession,
        *,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Document | None:

        result = await db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_knowledge_base(
        db: AsyncSession,
        *,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
        offset: int = 0,
        limit: int = 20,
        status: str | None = None,
    ) -> tuple[list[Document], int]:

        conditions = [
            Document.knowledge_base_id
            == knowledge_base_id,

            Document.user_id
            == user_id,
        ]

        if status:
            conditions.append(
                Document.status == status
            )

        count_result = await db.execute(
            select(
                func.count(Document.id)
            ).where(*conditions)
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(Document)
            .where(*conditions)
            .order_by(
                Document.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        documents = list(
            result.scalars().all()
        )

        return documents, total

    @staticmethod
    async def find_by_checksum(
        db: AsyncSession,
        *,
        knowledge_base_id: uuid.UUID,
        checksum: str,
    ) -> Document | None:

        result = await db.execute(
            select(Document).where(
                Document.knowledge_base_id
                == knowledge_base_id,

                Document.checksum
                == checksum,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def update_status(
        db: AsyncSession,
        document: Document,
        *,
        status: str,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> Document:

        document.status = status
        document.error_message = error_message

        if chunk_count is not None:
            document.chunk_count = chunk_count

        await db.flush()
        await db.refresh(document)

        return document

    @staticmethod
    async def update_extraction_metadata(
        db: AsyncSession,
        document: Document,
        metadata: dict,
    ) -> Document:

        document.extraction_metadata = metadata

        await db.flush()
        await db.refresh(document)

        return document

    @staticmethod
    async def mark_queued(
        db: AsyncSession,
        document: Document,
    ) -> Document:

        return await DocumentRepository.update_status(
            db,
            document,
            status="queued",
        )

    @staticmethod
    async def mark_processing(
        db: AsyncSession,
        document: Document,
    ) -> Document:

        return await DocumentRepository.update_status(
            db,
            document,
            status="processing",
        )

    @staticmethod
    async def mark_ready(
        db: AsyncSession,
        document: Document,
        *,
        chunk_count: int,
    ) -> Document:

        return await DocumentRepository.update_status(
            db,
            document,
            status="ready",
            chunk_count=chunk_count,
            error_message=None,
        )

    @staticmethod
    async def mark_failed(
        db: AsyncSession,
        document: Document,
        *,
        error_message: str,
    ) -> Document:

        return await DocumentRepository.update_status(
            db,
            document,
            status="failed",
            error_message=error_message,
        )

    @staticmethod
    async def delete(
        db: AsyncSession,
        document: Document,
    ) -> None:

        await db.delete(document)
        await db.flush()