"""Re-ingest documents into the current hybrid index (RAG_INDEX_VERSION).

Usage:
    python -m scripts.reindex_documents                 # enqueue all to Celery
    python -m scripts.reindex_documents --inline        # process here, one by one
    python -m scripts.reindex_documents --kb <uuid> --status ready failed
    python -m scripts.reindex_documents --only-stale    # skip docs already on this index version

Documents stay searchable in the previous index version until each one is
re-ingested; the old points are removed per document on success.
"""

import argparse
import asyncio
import logging
import uuid

from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.document import Document


async def collect(args: argparse.Namespace) -> list[uuid.UUID]:
    async with AsyncSessionLocal() as db:
        query = select(Document.id, Document.extraction_metadata).where(
            Document.status.in_(args.status)
        )
        if args.kb:
            query = query.where(Document.knowledge_base_id == uuid.UUID(args.kb))
        rows = (await db.execute(query.order_by(Document.created_at))).all()

    ids = []
    for document_id, metadata in rows:
        version = (metadata or {}).get("index_version")
        if args.only_stale and version == settings.RAG_INDEX_VERSION:
            continue
        ids.append(document_id)
    return ids


async def process_inline(ids: list[uuid.UUID]) -> None:
    from app.services.document_service import DocumentService

    service = DocumentService()
    await service.vector_store.initialize()

    for number, document_id in enumerate(ids, start=1):
        async with AsyncSessionLocal() as db:
            try:
                document = await service.process_document(db, document_id=document_id)
                print(f"[{number}/{len(ids)}] {document.original_name}: {document.chunk_count} chunks")
            except Exception as exc:
                print(f"[{number}/{len(ids)}] {document_id} FAILED: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb", help="Only this knowledge base id")
    parser.add_argument("--status", nargs="+", default=["ready"], help="Document statuses to include")
    parser.add_argument("--only-stale", action="store_true", help="Skip documents already on this index version")
    parser.add_argument("--inline", action="store_true", help="Process in this process instead of Celery")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    # One event loop: the async DB engine's pool is bound to it.
    asyncio.run(run(args))


async def run(args: argparse.Namespace) -> None:
    ids = await collect(args)
    print(f"{len(ids)} document(s) -> {settings.QDRANT_COLLECTION}_v{settings.RAG_INDEX_VERSION}")

    if args.inline:
        await process_inline(ids)
        return

    from app.workers.document_tasks import process_document_task

    for document_id in ids:
        process_document_task.delay(str(document_id))
    print("Enqueued on the 'documents' queue (worker needs -Q documents).")

if __name__ == "__main__":
    main()
