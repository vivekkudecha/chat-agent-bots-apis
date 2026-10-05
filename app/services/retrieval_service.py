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
class DistinctSource:
    document_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    file_name: str | None
    page: int | None
    pages: list[int] = field(default_factory=list)
    score: float = 0.0
    chunk_count: int = 1
    content_preview: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str

    chunks: list[RetrievedChunk]

    knowledge_base_ids: list[uuid.UUID]

    total_results: int = 0

    def get_distinct_sources(self) -> list[DistinctSource]:
        sources_map: dict[uuid.UUID, DistinctSource] = {}

        for chunk in self.chunks:
            doc_id = chunk.document_id
            if doc_id not in sources_map:
                pages = [chunk.page] if chunk.page is not None else []
                preview = chunk.text[:250].strip() if chunk.text else None
                if preview and len(chunk.text) > 250:
                    preview += "..."

                sources_map[doc_id] = DistinctSource(
                    document_id=doc_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    file_name=chunk.file_name,
                    page=chunk.page,
                    pages=pages,
                    score=round(chunk.score, 4),
                    chunk_count=1,
                    content_preview=preview,
                    metadata=dict(chunk.metadata),
                )
            else:
                src = sources_map[doc_id]
                src.chunk_count += 1
                if chunk.score > src.score:
                    src.score = round(chunk.score, 4)
                    if chunk.page is not None:
                        src.page = chunk.page
                    if chunk.text:
                        preview = chunk.text[:250].strip()
                        if len(chunk.text) > 250:
                            preview += "..."
                        src.content_preview = preview

                if chunk.page is not None and chunk.page not in src.pages:
                    src.pages.append(chunk.page)
                    src.pages.sort()

        return sorted(
            sources_map.values(),
            key=lambda s: s.score,
            reverse=True,
        )


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
        # Search a broader pool of candidates
        # to ensure distinct reference files are found.
        # ---------------------------------------------

        search_limit = max(
            top_k * 4,
            20,
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
        # Deduplicate & balance diversity
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

        candidates: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
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

            file_name = (
                result.get("file_name")
                or metadata.get("file_name")
            )

            candidates.append(
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
                    file_name=file_name,
                    metadata=metadata,
                )
            )

            seen_ids.add(point_id)
            seen_text.add(
                normalized_text
            )

        # Count distinct documents present
        doc_ids = {c.document_id for c in candidates}

        # If multiple distinct documents matched, ensure diversity across them
        # so one file does not consume all slots before other relevant files are considered
        if len(doc_ids) > 1:
            max_per_doc = max(1, limit // len(doc_ids) + 1)
            selected: list[RetrievedChunk] = []
            doc_counts: dict[uuid.UUID, int] = {}
            remaining: list[RetrievedChunk] = []

            for chunk in candidates:
                count = doc_counts.get(chunk.document_id, 0)
                if count < max_per_doc and len(selected) < limit:
                    selected.append(chunk)
                    doc_counts[chunk.document_id] = count + 1
                else:
                    remaining.append(chunk)

            # Fill any remaining slots up to limit with the next best scoring chunks
            while len(selected) < limit and remaining:
                selected.append(remaining.pop(0))

            return selected

        return candidates[:limit]