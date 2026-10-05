import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.repositories.bot_repository import BotRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.ai.rag.vector_store import VectorStoreService

from app.core.exceptions import (
    BotNotFoundException,
    RetrievalException,
)

logger = logging.getLogger(__name__)


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
        context_budget: int | None = None,
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
            use_hybrid = getattr(settings, "RAG_HYBRID_SEARCH", True)

            if use_hybrid:
                dense_task = self.vector_store.search(
                    query=query,
                    user_id=user_id,
                    knowledge_base_ids=knowledge_base_ids,
                    top_k=search_limit,
                    score_threshold=score_threshold,
                )
                lexical_task = self.vector_store.search_keyword(
                    query=query,
                    user_id=user_id,
                    knowledge_base_ids=knowledge_base_ids,
                    top_k=search_limit,
                )

                results_pair = await asyncio.gather(
                    dense_task,
                    lexical_task,
                    return_exceptions=True,
                )

                dense_res = (
                    results_pair[0]
                    if isinstance(results_pair[0], list)
                    else []
                )
                lexical_res = (
                    results_pair[1]
                    if isinstance(results_pair[1], list)
                    else []
                )

                if isinstance(results_pair[0], Exception):
                    logger.warning(
                        "Dense search error during hybrid retrieval: %s",
                        results_pair[0],
                    )
                if isinstance(results_pair[1], Exception):
                    logger.warning(
                        "Lexical search error during hybrid retrieval: %s",
                        results_pair[1],
                    )

                if dense_res and lexical_res:
                    results = self._reciprocal_rank_fusion(
                        dense_res,
                        lexical_res,
                    )
                elif dense_res:
                    results = dense_res
                else:
                    results = lexical_res
            else:
                results = await self.vector_store.search(
                    query=query,
                    user_id=user_id,
                    knowledge_base_ids=knowledge_base_ids,
                    top_k=search_limit,
                    score_threshold=score_threshold,
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

        # ---------------------------------------------
        # Small model optimization: Extract focused snippets
        # if token budget is constrained
        # ---------------------------------------------
        if context_budget and context_budget <= 1200 and chunks:
            max_chars_per_chunk = max(180, int((context_budget * 4) / max(1, len(chunks))))
            for chunk in chunks:
                chunk.text = self._extract_focused_snippet(
                    chunk.text,
                    query=query,
                    max_chars=max_chars_per_chunk,
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
    # FOCUSED SNIPPET EXTRACTION
    # =====================================================

    @staticmethod
    def _extract_focused_snippet(
        text: str,
        *,
        query: str,
        max_chars: int = 350,
    ) -> str:
        cleaned = text.strip()
        if len(cleaned) <= max_chars:
            return cleaned

        query_words = {
            w.lower()
            for w in re.findall(r"\w+", query)
            if len(w) > 2 and w.lower() not in {
                "the", "and", "for", "with", "this", "that", "what", "from",
                "have", "about", "your", "tell", "which",
            }
        }
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        if len(sentences) <= 1:
            return cleaned[:max_chars].strip() + "..."

        scored = []
        for idx, sentence in enumerate(sentences):
            sent_words = {w.lower() for w in re.findall(r"\w+", sentence)}
            overlap = len(query_words.intersection(sent_words)) if query_words else 1
            scored.append((overlap, idx, sentence))

        best = max(scored, key=lambda s: (s[0], -s[1]))
        best_idx = best[1]

        start_idx = max(0, best_idx - 1)
        end_idx = min(len(sentences), best_idx + 2)
        selected = sentences[start_idx:end_idx]

        snippet = " ".join(selected).strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].strip() + "..."
        return snippet

    # =====================================================
    # RECIPROCAL RANK FUSION (RRF)
    # =====================================================

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict[str, Any]],
        lexical_results: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:

        scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            item_id = str(item["id"])
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            if item_id not in doc_map:
                doc_map[item_id] = dict(item)

        for rank, item in enumerate(lexical_results):
            item_id = str(item["id"])
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            if item_id not in doc_map:
                doc_map[item_id] = dict(item)

        # Sort items by fused RRF score
        sorted_ids = sorted(
            scores.keys(),
            key=lambda i: scores[i],
            reverse=True,
        )
        fused_results: list[dict[str, Any]] = []

        # Normalization baseline: top ranking in both modalities
        max_possible_rrf = 2.0 / (k + 1)

        for item_id in sorted_ids:
            item = doc_map[item_id]
            original_score = item.get("score", 0.0)
            fused_score = round(
                min(1.0, scores[item_id] / max_possible_rrf),
                4,
            )
            item["rrf_score"] = scores[item_id]
            # Blend semantic confidence with lexical boost
            item["score"] = max(original_score, fused_score)
            fused_results.append(item)

        return fused_results

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

        # If multiple distinct documents matched, preserve primary relevance
        # while preventing any single document from starving all other references
        if len(doc_ids) > 1:
            max_per_doc = max(2, min(3, max(1, limit - 1)))
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