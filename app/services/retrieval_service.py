import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.bot_repository import BotRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.vector_store_service import VectorStoreService

from app.core.exceptions import (
    BotNotFoundException,
    RetrievalException,
)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float

    document_id: uuid.UUID
    knowledge_base_id: uuid.UUID

    chunk_index: int | None = None
    page: int | None = None

    file_name: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class RetrievalResult:
    query: str

    chunks: list[RetrievedChunk]

    knowledge_base_ids: list[uuid.UUID]

    total_results: int = 0


class RetrievalService:

    def __init__(
        self,
        vector_store: VectorStoreService | None = None,
    ):
        self.vector_store = (
            vector_store
            or VectorStoreService()
        )

    # =====================================================
    # RETRIEVE FOR BOT
    # =====================================================

    async def retrieve_for_bot(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        bot_id: uuid.UUID,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> RetrievalResult:

        # ---------------------------------------------
        # Verify bot ownership
        # ---------------------------------------------

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        # ---------------------------------------------
        # Resolve KBs attached to bot
        # ---------------------------------------------

        kb_links = (
            await KnowledgeRepository.list_for_bot(
                db,
                bot_id,
            )
        )

        if not kb_links:
            return RetrievalResult(
                query=query,
                chunks=[],
                knowledge_base_ids=[],
                total_results=0,
            )

        # ---------------------------------------------
        # Security:
        # only use KBs owned by this user.
        # ---------------------------------------------

        knowledge_base_ids = []

        for link in kb_links:

            kb = link.knowledge_base

            if not kb:
                continue

            if kb.user_id != user_id:
                continue

            knowledge_base_ids.append(
                kb.id
            )

        if not knowledge_base_ids:
            return RetrievalResult(
                query=query,
                chunks=[],
                knowledge_base_ids=[],
                total_results=0,
            )

        # ---------------------------------------------
        # Search slightly more than required.
        #
        # This gives us room for deduplication.
        # ---------------------------------------------

        search_limit = max(
            top_k * 2,
            top_k,
        )

        try:

            results = (
                await self.vector_store.search(
                    query=query,
                    user_id=user_id,
                    knowledge_base_ids=(
                        knowledge_base_ids
                    ),
                    top_k=search_limit,
                    score_threshold=(
                        score_threshold
                    ),
                )
            )

        except Exception as exc:

            if isinstance(
                exc,
                RetrievalException,
            ):
                raise

            raise RetrievalException(
                "Knowledge retrieval failed."
            ) from exc

        # ---------------------------------------------
        # Deduplicate
        # ---------------------------------------------

        chunks = self._deduplicate(
            results,
            limit=top_k,
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            knowledge_base_ids=(
                knowledge_base_ids
            ),
            total_results=len(chunks),
        )

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    def _deduplicate(
        self,
        results: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:

        chunks: list[RetrievedChunk] = []

        seen_ids: set[str] = set()

        # Prevent near-identical chunk text from
        # consuming the whole context window.
        seen_text: set[str] = set()

        for result in results:

            point_id = str(
                result["id"]
            )

            if point_id in seen_ids:
                continue

            text = (
                result.get("text")
                or ""
            ).strip()

            if not text:
                continue

            normalized_text = (
                " ".join(
                    text.lower().split()
                )
            )

            if normalized_text in seen_text:
                continue

            try:

                document_id = uuid.UUID(
                    str(
                        result["document_id"]
                    )
                )

                knowledge_base_id = uuid.UUID(
                    str(
                        result[
                            "knowledge_base_id"
                        ]
                    )
                )

            except (
                ValueError,
                TypeError,
                KeyError,
            ):
                continue

            metadata = (
                result.get("metadata")
                or {}
            )

            chunks.append(
                RetrievedChunk(
                    id=point_id,
                    text=text,
                    score=float(
                        result.get(
                            "score",
                            0,
                        )
                    ),
                    document_id=document_id,
                    knowledge_base_id=(
                        knowledge_base_id
                    ),
                    chunk_index=result.get(
                        "chunk_index"
                    ),
                    page=result.get(
                        "page"
                    ),
                    file_name=metadata.get(
                        "file_name"
                    ),
                    metadata=metadata,
                )
            )

            seen_ids.add(point_id)
            seen_text.add(
                normalized_text
            )

            if len(chunks) >= limit:
                break

        return chunks