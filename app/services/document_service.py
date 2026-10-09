from app.integrations.storage import get_storage_provider
import asyncio
from dataclasses import dataclass
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document

from app.repositories.document_repository import (
    DocumentRepository,
)

from app.repositories.knowledge_repository import (
    KnowledgeRepository,
)

from app.config import settings

from app.ai.rag import (
    ChunkingService,
    TextExtractionService,
    VectorStoreService,
)
from app.ai.rag.text_extraction import ExtractionStats

from app.core.exceptions import (
    DuplicateDocumentException,
    KnowledgeBaseNotFoundException,
    DocumentNotFoundException,
    ValidationException,
)


logger = logging.getLogger(__name__)


@dataclass
class DocumentListResult:
    items: list[Any]
    total: int


class DocumentService:

    def __init__(self):

        self.extractor = (
            TextExtractionService()
        )

        self.vector_store = (
            VectorStoreService()
        )

        self.storage = (
            get_storage_provider()
        )

    # =====================================================
    # CHECKSUM
    # =====================================================

    @staticmethod
    def calculate_checksum(
        file_path: str | Path,
    ) -> str:

        sha256 = hashlib.sha256()

        with open(
            file_path,
            "rb",
        ) as file:

            while True:

                block = file.read(
                    1024 * 1024
                )

                if not block:
                    break

                sha256.update(block)

        return sha256.hexdigest()

    # =====================================================
    # UPLOAD
    # =====================================================

    async def upload(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        file: UploadFile,
    ) -> Document:

        knowledge_base = await KnowledgeRepository.get_owned(
            db,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
        )

        if not knowledge_base:
            raise KnowledgeBaseNotFoundException()

        storage_key = await self.storage.save_upload(
            file=file,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )

        local_path = await self.storage.get_local_path(storage_key)
        checksum = self.calculate_checksum(local_path)

        existing = await DocumentRepository.find_by_checksum(
            db,
            knowledge_base_id=knowledge_base_id,
            checksum=checksum,
        )

        if existing:
            await self.storage.delete(storage_key)
            raise DuplicateDocumentException()

        document = await DocumentRepository.create(
            db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            original_name=file.filename or "unknown",
            mime_type=file.content_type,
            file_size=local_path.stat().st_size,
            storage_provider="local",
            storage_key=storage_key,
            checksum=checksum,
        )

        return document

    # =====================================================
    # GET
    # =====================================================

    async def get(
        self,
        db: AsyncSession,
        *,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Document:

        document = await DocumentRepository.get_owned(
            db,
            document_id=document_id,
            user_id=user_id,
        )

        if not document:
            raise DocumentNotFoundException()

        return document

    # =====================================================
    # LIST
    # =====================================================

    async def list(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> DocumentListResult:

        knowledge_base = await KnowledgeRepository.get_owned(
            db,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
        )

        if not knowledge_base:
            raise KnowledgeBaseNotFoundException()

        offset = (page - 1) * page_size

        items, total = await DocumentRepository.list_by_knowledge_base(
            db,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            offset=offset,
            limit=page_size,
        )

        return DocumentListResult(
            items=items,
            total=total,
        )

    # =====================================================
    # PREPARE RETRY
    # =====================================================

    async def prepare_retry(
        self,
        db: AsyncSession,
        *,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Document:

        document = await self.get(
            db,
            document_id=document_id,
            user_id=user_id,
        )

        await DocumentRepository.mark_queued(
            db,
            document,
        )

        return document

    # =====================================================
    # REGISTER DOCUMENT
    # =====================================================

    async def register_document(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        file_path: str | Path,
        original_name: str,
        mime_type: str | None,
        storage_provider: str = "local",
        storage_key: str | None = None,
    ) -> Document:

        path = Path(file_path)

        if not path.exists():
            raise ValidationException(
                "Uploaded file does not exist."
            )

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
            raise KnowledgeBaseNotFoundException()

        checksum = self.calculate_checksum(
            path
        )

        existing = (
            await DocumentRepository
            .find_by_checksum(
                db,
                knowledge_base_id=(
                    knowledge_base_id
                ),
                checksum=checksum,
            )
        )

        if existing:
            raise DuplicateDocumentException()

        document = (
            await DocumentRepository.create(
                db,
                user_id=user_id,
                knowledge_base_id=(
                    knowledge_base_id
                ),
                original_name=original_name,
                mime_type=mime_type,
                file_size=path.stat().st_size,
                storage_provider=(
                    storage_provider
                ),
                storage_key=(
                    storage_key
                    or str(path)
                ),
                checksum=checksum,
            )
        )

        await db.commit()
        await db.refresh(document)

        return document

    # =====================================================
    # PROCESS DOCUMENT
    # =====================================================

    async def process_document(
        self,
        db: AsyncSession,
        *,
        document_id: uuid.UUID,
    ) -> Document:

        document = (
            await DocumentRepository.get_by_id(
                db,
                document_id,
            )
        )

        if not document:
            raise DocumentNotFoundException()

        knowledge_base = (
            await KnowledgeRepository.get_by_id(
                db,
                document.knowledge_base_id,
            )
        )

        if not knowledge_base:
            raise KnowledgeBaseNotFoundException()

        ingest_id = uuid.uuid4().hex

        try:

            # ---------------------------------------------
            # Processing
            # ---------------------------------------------

            await DocumentRepository.mark_processing(
                db,
                document,
            )

            await db.commit()

            local_path = (
                await self.storage.get_local_path(
                    document.storage_key
                )
            )

            chunk_config = (
                knowledge_base.chunking_config
                or {}
            )

            chunker = ChunkingService(
                chunk_size=chunk_config.get(
                    "chunk_size",
                    settings.DEFAULT_CHUNK_SIZE,
                ),
                chunk_overlap=chunk_config.get(
                    "chunk_overlap",
                    settings.DEFAULT_CHUNK_OVERLAP,
                ),
                separators=chunk_config.get(
                    "separators"
                ),
            )

            # ---------------------------------------------
            # Stream: extract -> chunk -> embed -> upsert
            #
            # Page batches flow through a bounded queue so
            # extraction of batch N+1 overlaps embedding of
            # batch N and memory stays flat for any page
            # count. Points are written under a new ingest
            # id; the previous version stays searchable
            # until this one completes.
            # ---------------------------------------------

            stats = ExtractionStats()

            chunk_count = await self._ingest(
                db,
                document=document,
                local_path=local_path,
                chunker=chunker,
                stats=stats,
                ingest_id=ingest_id,
            )

            if not chunk_count:
                raise ValidationException(
                    "No extractable text was "
                    "found in the document."
                )

            await self.vector_store.delete_stale_ingests(
                document.id,
                keep_ingest_id=ingest_id,
            )

            # ---------------------------------------------
            # Extraction Metadata
            # ---------------------------------------------

            await (
                DocumentRepository
                .update_extraction_metadata(
                    db,
                    document,
                    {
                        **stats.as_metadata(),
                        "chunk_count": chunk_count,
                        "ingest_id": ingest_id,
                        "index_version": settings.RAG_INDEX_VERSION,
                    },
                )
            )

            # ---------------------------------------------
            # Ready
            # ---------------------------------------------

            await DocumentRepository.mark_ready(
                db,
                document,
                chunk_count=chunk_count,
            )

            await db.commit()

            await db.refresh(document)

            return document

        except Exception as exc:

            await db.rollback()

            # Drop the partial ingest; the previous version
            # (if any) remains intact.
            try:
                await self.vector_store.delete_ingest(
                    document_id,
                    ingest_id,
                )
            except Exception:
                logger.exception(
                    "Failed to clean partial ingest %s of %s",
                    ingest_id,
                    document_id,
                )

            # Retrieve again because rollback can expire
            # or reset ORM state depending on session config.

            failed_document = (
                await DocumentRepository.get_by_id(
                    db,
                    document_id,
                )
            )

            if failed_document:

                await DocumentRepository.mark_failed(
                    db,
                    failed_document,
                    error_message=str(exc)[:2000],
                )

                await db.commit()

            raise

    async def _ingest(
        self,
        db: AsyncSession,
        *,
        document: Document,
        local_path: Any,
        chunker: ChunkingService,
        stats: ExtractionStats,
        ingest_id: str,
    ) -> int:

        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        file_name = document.original_name

        async def produce() -> None:
            try:
                async for sections in self.extractor.iter_batches(
                    local_path,
                    stats=stats,
                ):
                    chunks = chunker.feed(sections)
                    if chunks:
                        await queue.put(chunks)

                tail = chunker.flush()
                if tail:
                    await queue.put(tail)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        chunk_count = 0

        try:
            while (chunks := await queue.get()) is not None:

                chunk_count += await self.vector_store.upsert_chunks(
                    user_id=document.user_id,
                    knowledge_base_id=document.knowledge_base_id,
                    document_id=document.id,
                    ingest_id=ingest_id,
                    chunks=[
                        {
                            "index": chunk.index,
                            "text": chunk.text,
                            "page": chunk.page,
                            "metadata": {
                                **chunk.metadata,
                                "file_name": file_name,
                            },
                        }
                        for chunk in chunks
                    ],
                )

                # Progress for UIs polling the document.
                await DocumentRepository.update_extraction_metadata(
                    db,
                    document,
                    {
                        **(document.extraction_metadata or {}),
                        "progress": {
                            "pages_total": stats.page_count,
                            "pages_processed": stats.pages_with_text,
                            "chunks_indexed": chunk_count,
                        },
                    },
                )
                await db.commit()

            # Surface extraction errors.
            await producer

        finally:
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except (asyncio.CancelledError, Exception):
                    pass

        logger.info(
            "Indexed document %s: %s chunks from %s pages (ocr=%s)",
            document.id,
            chunk_count,
            stats.page_count,
            stats.ocr_pages,
        )

        return chunk_count

    # =====================================================
    # DELETE DOCUMENT
    # =====================================================

    async def delete_document(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:

        document = (
            await DocumentRepository.get_owned(
                db,
                document_id=document_id,
                user_id=user_id,
            )
        )

        if not document:
            raise DocumentNotFoundException()

        # ---------------------------------------------
        # Qdrant
        # ---------------------------------------------

        await self.vector_store.delete_document(
            document.id
        )

        # ---------------------------------------------
        # Original file
        # ---------------------------------------------

        # if document.storage_provider == "local":

        #     path = Path(
        #         document.storage_key
        #     )

        #     if path.exists():
        #         path.unlink()

        await self.storage.delete(document.storage_key)

        # ---------------------------------------------
        # PostgreSQL
        # ---------------------------------------------

        await DocumentRepository.delete(
            db,
            document,
        )

        await db.commit()