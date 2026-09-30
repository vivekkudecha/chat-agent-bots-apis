from app.integrations.storage import get_storage_provider
import hashlib
import uuid
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document

from app.repositories.document_repository import (
    DocumentRepository,
)

from app.repositories.knowledge_repository import (
    KnowledgeRepository,
)

from app.services.text_extraction_service import (
    TextExtractionService,
)

from app.services.chunking_service import (
    ChunkingService,
)

from app.services.vector_store_service import (
    VectorStoreService,
)

from app.core.exceptions import (
    DuplicateDocumentException,
    KnowledgeBaseNotFoundException,
    DocumentNotFoundException,
    ValidationException,
)


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

        try:

            # ---------------------------------------------
            # Processing
            # ---------------------------------------------

            await DocumentRepository.mark_processing(
                db,
                document,
            )

            await db.commit()

            # ---------------------------------------------
            # Extract
            # ---------------------------------------------

            local_path = (
                await self.storage.get_local_path(
                    document.storage_key
                )
            )

            extraction = (
                await self.extractor.extract(
                    local_path
                )
            )

            if not extraction.sections:

                raise ValidationException(
                    "No extractable text was "
                    "found in the document."
                )

            # ---------------------------------------------
            # Chunking configuration
            # ---------------------------------------------

            chunk_config = (
                knowledge_base.chunking_config
                or {}
            )

            chunker = ChunkingService(
                chunk_size=chunk_config.get(
                    "chunk_size",
                    800,
                ),
                chunk_overlap=chunk_config.get(
                    "chunk_overlap",
                    100,
                ),
                separators=chunk_config.get(
                    "separators"
                ),
            )

            chunks = (
                chunker.chunk_extraction(
                    extraction
                )
            )

            if not chunks:
                raise ValidationException(
                    "Document produced no chunks."
                )

            # ---------------------------------------------
            # Convert to vector-store format
            # ---------------------------------------------

            vector_chunks = []

            for chunk in chunks:

                vector_chunks.append(
                    {
                        "index": chunk.index,
                        "text": chunk.text,
                        "page": chunk.page,
                        "metadata": {
                            **chunk.metadata,
                            "file_name": (
                                document.original_name
                            ),
                        },
                    }
                )

            # ---------------------------------------------
            # Remove previous vectors
            #
            # Important for reprocessing.
            # ---------------------------------------------

            await self.vector_store.delete_document(
                document.id
            )

            # ---------------------------------------------
            # Embed + Qdrant
            # ---------------------------------------------

            chunk_count = (
                await self.vector_store
                .upsert_chunks(
                    user_id=document.user_id,
                    knowledge_base_id=(
                        document
                        .knowledge_base_id
                    ),
                    document_id=document.id,
                    chunks=vector_chunks,
                )
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
                        **extraction.metadata,
                        "chunk_count": (
                            chunk_count
                        ),
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