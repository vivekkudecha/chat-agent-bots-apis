import asyncio
import logging
import uuid

from celery.exceptions import (
    SoftTimeLimitExceeded,
)

from app.config import settings

from app.workers.celery_app import (
    celery_app,
)

from app.database import (
    AsyncSessionLocal,
)

from app.services.document_service import (
    DocumentService,
)


logger = logging.getLogger(__name__)


# =========================================================
# Async Runner
# =========================================================

def run_async(coro):
    """
    Run an async coroutine inside a Celery
    synchronous worker process.
    """

    return asyncio.run(coro)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

@celery_app.task(
    bind=True,
    name="documents.process",
    # Large scanned documents can take far longer than the global limit.
    soft_time_limit=settings.DOCUMENT_TASK_SOFT_TIME_LIMIT,
    time_limit=settings.DOCUMENT_TASK_TIME_LIMIT,
    autoretry_for=(
        ConnectionError,
        TimeoutError,
    ),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
)
def process_document_task(
    self,
    document_id: str,
) -> dict:

    logger.info(
        "Processing document %s",
        document_id,
    )

    try:

        parsed_id = uuid.UUID(
            document_id
        )

        return run_async(
            _process_document(
                parsed_id
            )
        )

    except SoftTimeLimitExceeded:

        logger.exception(
            "Document processing exceeded "
            "soft time limit: %s",
            document_id,
        )

        raise

    except Exception:

        logger.exception(
            "Document processing failed: %s",
            document_id,
        )

        raise


async def _process_document(
    document_id: uuid.UUID,
) -> dict:

    async with AsyncSessionLocal() as db:

        service = DocumentService()

        document = (
            await service.process_document(
                db,
                document_id=document_id,
            )
        )

        return {
            "document_id": str(
                document.id
            ),
            "status": document.status,
            "chunk_count": (
                document.chunk_count
            ),
        }


# =========================================================
# DELETE DOCUMENT VECTORS
# =========================================================

@celery_app.task(
    bind=True,
    name="documents.delete_vectors",
    autoretry_for=(
        ConnectionError,
        TimeoutError,
    ),
    retry_backoff=True,
    retry_backoff_max=60,
    max_retries=3,
)
def delete_document_vectors_task(
    self,
    document_id: str,
) -> dict:

    parsed_id = uuid.UUID(
        document_id
    )

    return run_async(
        _delete_vectors(
            parsed_id
        )
    )


async def _delete_vectors(
    document_id: uuid.UUID,
) -> dict:

    from app.ai.rag import (
        VectorStoreService,
    )

    vector_store = (
        VectorStoreService()
    )

    await vector_store.delete_document(
        document_id
    )

    return {
        "document_id": str(
            document_id
        ),
        "vectors_deleted": True,
    }