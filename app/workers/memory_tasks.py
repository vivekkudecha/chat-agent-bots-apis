from __future__ import annotations

import asyncio
import logging
import uuid

from app.ai.memory.manager import MemoryManager
from app.database import AsyncSessionLocal
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def run_async(coro):
    """
    Run an async coroutine inside a Celery synchronous worker process.
    """
    return asyncio.run(coro)


@celery_app.task(
    bind=True,
    name="memory.compact_conversation",
    autoretry_for=(
        ConnectionError,
        TimeoutError,
    ),
    retry_backoff=True,
    retry_backoff_max=30,
    retry_jitter=True,
    max_retries=2,
)
def compact_conversation_task(
    self,
    conversation_id: str,
    model: str | None = None,
) -> dict:
    """
    Background worker task to compress older conversation messages
    into a structured rolling episodic summary.
    """
    logger.info("Starting background conversation compaction for %s", conversation_id)

    async def _execute():
        async with AsyncSessionLocal() as db:
            manager = MemoryManager()
            summary = await manager.compact_conversation(
                db,
                conversation_id=uuid.UUID(conversation_id),
                model=model,
            )
            return summary.to_dict() if summary else None

    try:
        result = run_async(_execute())
        logger.info("Completed conversation compaction for %s", conversation_id)
        return {"status": "success", "conversation_id": conversation_id, "summary": result}
    except Exception as exc:
        logger.exception("Failed to compact conversation %s: %s", conversation_id, exc)
        return {"status": "failed", "conversation_id": conversation_id, "error": str(exc)}
