import asyncio
import logging
import math
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.repositories.bot_repository import BotRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.ai.rag.sparse import BM25SparseEncoder
from app.ai.rag.vector_store import VectorStoreService

from app.core.exceptions import (
    BotNotFoundException,
    RetrievalException,
)

logger = logging.getLogger(__name__)

# Document-control headings make poor follow-up suggestions.
_BOILERPLATE_SECTION = re.compile(
    r"\b(distribution list|version control|revision history|document control|"
    r"table of contents|approval|sign[- ]?off|change log|abbreviations)\b",
    re.IGNORECASE,
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

    # Queries run, candidate counts, gating decisions (observability).
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # Sections the search came close to (including near-misses that did
    # not pass the relevance gate); used to suggest clarifying questions.
    related_topics: list[dict[str, Any]] = field(default_factory=list)

    def get_distinct_sources(self) -> list[DistinctSource]:
        sources_map: dict[uuid.UUID, DistinctSource] = {}

        for chunk in self.chunks:
            doc_id = chunk.document_id
            chunk_pages = chunk.metadata.get("pages") or (
                [chunk.page] if chunk.page is not None else []
            )

            if doc_id not in sources_map:
                preview = chunk.text[:250].strip() if chunk.text else None
                if preview and len(chunk.text) > 250:
                    preview += "..."

                sources_map[doc_id] = DistinctSource(
                    document_id=doc_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    file_name=chunk.file_name,
                    page=chunk.page,
                    pages=sorted(set(chunk_pages)),
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

                src.pages = sorted(set(src.pages) | set(chunk_pages))

        return sorted(
            sources_map.values(),
            key=lambda s: s.score,
            reverse=True,
        )


@dataclass
class _Candidate:
    match: dict[str, Any]
    rrf: float = 0.0
    dense: float | None = None
    coverage: float = 0.0
    rerank: float | None = None
    terms: set[str] = field(default_factory=set)

    @property
    def id(self) -> str:
        return self.match["id"]


# ---------------------------------------------------------
# Optional cross-encoder
# ---------------------------------------------------------

_reranker: Any | None = None
_reranker_failed = False


def _get_reranker() -> Any | None:
    global _reranker, _reranker_failed

    model_name = settings.RAG_RERANKER_MODEL
    if not model_name or _reranker_failed:
        return None

    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(model_name, max_length=512)
        except Exception as exc:
            _reranker_failed = True
            logger.warning("Reranker %s unavailable: %s", model_name, exc)
            return None

    return _reranker


class RetrievalService:
    """
    Hybrid retrieval tuned for large corpora and small LLM context windows.

    1. Every query (original + agent refinements) runs a dense and a BM25
       search in one batched Qdrant request.
    2. Ranks are fused with reciprocal rank fusion across all lists.
    3. Candidates must be semantically close (dense cosine) or contain
       most query terms verbatim; nothing else reaches the LLM.
    4. Optional cross-encoder rerank, then near-duplicate removal and a
       per-document cap for diversity.
    5. Hits are expanded with neighbouring chunks into coherent passages
       ("small-to-big"), within the caller's context budget.
    """

    RRF_K = 60

    # Max characters of one expanded passage.
    MAX_PASSAGE_CHARS = 2400

    def __init__(
        self,
        vector_store: VectorStoreService | None = None,
    ):
        self.vector_store = (
            vector_store
            or VectorStoreService()
        )
        self.encoder = getattr(
            self.vector_store,
            "sparse_encoder",
            None,
        ) or BM25SparseEncoder()

    # =====================================================
    # RETRIEVE FOR BOT
    # =====================================================

    async def resolve_knowledge_base_ids(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        bot_id: uuid.UUID,
    ) -> list[uuid.UUID]:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        kb_links = await KnowledgeRepository.list_for_bot(
            db,
            bot_id,
        )

        # Security: only use KBs owned by this user.
        return [
            link.knowledge_base.id
            for link in kb_links or []
            if link.knowledge_base
            and link.knowledge_base.user_id == user_id
        ]

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
        queries: list[str] | None = None,
        document_ids: list[uuid.UUID] | None = None,
        exclude_ids: set[str] | None = None,
        knowledge_base_ids: list[uuid.UUID] | None = None,
    ) -> RetrievalResult:

        if knowledge_base_ids is None:
            knowledge_base_ids = await self.resolve_knowledge_base_ids(
                db,
                user_id=user_id,
                bot_id=bot_id,
            )

        if not knowledge_base_ids:
            return RetrievalResult(
                query=query,
                chunks=[],
                knowledge_base_ids=[],
                total_results=0,
            )

        return await self.retrieve(
            user_id=user_id,
            knowledge_base_ids=knowledge_base_ids,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            context_budget=context_budget,
            queries=queries,
            document_ids=document_ids,
            exclude_ids=exclude_ids,
        )

    async def retrieve(
        self,
        *,
        user_id: uuid.UUID,
        knowledge_base_ids: list[uuid.UUID],
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        context_budget: int | None = None,
        queries: list[str] | None = None,
        document_ids: list[uuid.UUID] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> RetrievalResult:

        threshold = (
            score_threshold
            if score_threshold is not None
            else settings.DEFAULT_SCORE_THRESHOLD
        )

        all_queries = self._unique_queries([query, *(queries or [])])
        pool = max(settings.RAG_CANDIDATE_POOL, top_k * 4)

        # ---------------------------------------------
        # 1. Search (dense + BM25 per query)
        # ---------------------------------------------

        try:
            results = await asyncio.gather(
                *(
                    self.vector_store.hybrid_search(
                        query=q,
                        user_id=user_id,
                        knowledge_base_ids=knowledge_base_ids,
                        limit=pool,
                        document_ids=document_ids,
                    )
                    for q in all_queries
                ),
                return_exceptions=True,
            )
        except Exception as exc:
            raise RetrievalException(
                "Knowledge retrieval failed."
            ) from exc

        failures = [r for r in results if isinstance(r, Exception)]
        if failures and len(failures) == len(results):
            raise RetrievalException(
                "Knowledge retrieval failed."
            ) from failures[0]
        for failure in failures:
            logger.warning("Hybrid search error: %s", failure)

        # ---------------------------------------------
        # 2. Fuse
        # ---------------------------------------------

        candidates: dict[str, _Candidate] = {}
        query_terms = [self.encoder.query_terms(q) for q in all_queries]

        for result in results:
            if isinstance(result, Exception):
                continue
            dense_hits, lexical_hits = result

            for rank, match in enumerate(dense_hits):
                cand = candidates.setdefault(match["id"], _Candidate(match))
                cand.rrf += 1.0 / (self.RRF_K + rank + 1)
                cand.dense = max(cand.dense or 0.0, float(match["score"]))

            for rank, match in enumerate(lexical_hits):
                cand = candidates.setdefault(match["id"], _Candidate(match))
                cand.rrf += 1.0 / (self.RRF_K + rank + 1)

        exclude_ids = exclude_ids or set()

        # ---------------------------------------------
        # 3. Relevance gate
        # ---------------------------------------------

        kept: list[_Candidate] = []
        for cand in candidates.values():
            if cand.id in exclude_ids or not (cand.match.get("text") or "").strip():
                continue

            cand.terms = self.encoder.query_terms(
                " ".join(
                    filter(
                        None,
                        [
                            cand.match.get("text"),
                            cand.match.get("section"),
                            cand.match.get("file_name"),
                        ],
                    )
                )
            )
            cand.coverage = max(
                (len(terms & cand.terms) / len(terms) for terms in query_terms if terms),
                default=0.0,
            )

            semantic = cand.dense is not None and cand.dense >= threshold
            lexical = cand.coverage >= settings.RAG_LEXICAL_MIN_COVERAGE
            if semantic or lexical:
                kept.append(cand)

        kept.sort(key=lambda c: c.rrf, reverse=True)

        # ---------------------------------------------
        # 4. Rerank (optional) + diversity
        # ---------------------------------------------

        reranked = await self._rerank(query, kept[:30])
        if reranked is not None:
            kept = reranked

        selected = self._diversify(kept, limit=top_k)

        # ---------------------------------------------
        # 5. Passages
        # ---------------------------------------------

        small_budget = bool(context_budget and context_budget <= 1200)

        if settings.RAG_NEIGHBOR_WINDOW > 0 and not small_budget and selected:
            chunks = await self._expand_neighbors(
                selected,
                user_id=user_id,
                threshold=threshold,
                char_budget=(context_budget * 4) if context_budget else None,
            )
        else:
            chunks = [self._to_chunk(c, threshold) for c in selected]

        if small_budget and chunks:
            max_chars = max(180, int((context_budget * 4) / max(1, len(chunks))))
            for chunk in chunks:
                chunk.text = self._extract_focused_snippet(
                    chunk.text,
                    query=query,
                    max_chars=max_chars,
                )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            knowledge_base_ids=knowledge_base_ids,
            total_results=len(chunks),
            related_topics=self._related_topics(
                candidates.values(),
                exclude_ids=exclude_ids,
            ),
            diagnostics={
                "queries": all_queries,
                "candidates": len(candidates),
                "passed_gate": len(kept),
                "selected": len(selected),
                "reranked": reranked is not None,
            },
        )

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _unique_queries(queries: list[str]) -> list[str]:
        seen: set[str] = set()
        unique = []
        for q in queries:
            q = (q or "").strip()
            key = " ".join(q.lower().split())
            if q and key not in seen:
                seen.add(key)
                unique.append(q)
        return unique[:4]

    async def _rerank(
        self,
        query: str,
        candidates: list[_Candidate],
    ) -> list[_Candidate] | None:

        model = _get_reranker()
        if model is None or not candidates:
            return None

        pairs = [(query, c.match["text"][:2000]) for c in candidates]
        try:
            scores = await asyncio.to_thread(model.predict, pairs)
        except Exception as exc:
            logger.warning("Rerank failed, keeping fused order: %s", exc)
            return None

        for cand, score in zip(candidates, scores):
            score = float(score)
            if score < 0 or score > 1:
                score = 1 / (1 + math.exp(-score))
            cand.rerank = score

        kept = [c for c in candidates if c.rerank >= settings.RAG_RERANK_THRESHOLD]
        return sorted(kept, key=lambda c: c.rerank, reverse=True)

    @staticmethod
    def _related_topics(
        candidates: Any,
        *,
        exclude_ids: set[str],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Distinct (document, section) pairs the query came close to."""

        floor = settings.RAG_SUGGESTION_MIN_SCORE
        ranked = sorted(
            (
                c for c in candidates
                if c.id not in exclude_ids
                and ((c.dense or 0.0) >= floor or c.coverage >= 0.5)
            ),
            key=lambda c: c.rrf,
            reverse=True,
        )

        topics: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()

        for cand in ranked:
            match = cand.match
            key = (match.get("file_name"), match.get("section"))
            leaf = (match.get("section") or "").split(">")[-1]
            if key in seen or _BOILERPLATE_SECTION.search(leaf):
                continue
            seen.add(key)
            topics.append(
                {
                    "document_id": str(match.get("document_id")),
                    "file_name": match.get("file_name"),
                    "section": match.get("section"),
                    "page": match.get("page"),
                    "preview": " ".join((match.get("text") or "").split())[:160],
                    "score": round(cand.dense or 0.0, 4),
                }
            )
            if len(topics) >= limit:
                break

        return topics

    @staticmethod
    def _diversify(
        candidates: list[_Candidate],
        *,
        limit: int,
    ) -> list[_Candidate]:
        """
        Drop near-duplicates (same passage from another ingest, repeated
        boilerplate) and cap chunks per document, back-filling with the
        best remaining candidates when slots are left.
        """

        max_per_doc = max(1, settings.RAG_MAX_PER_DOCUMENT)
        selected: list[_Candidate] = []
        overflow: list[_Candidate] = []
        per_doc: dict[str, int] = defaultdict(int)

        def is_duplicate(cand: _Candidate) -> bool:
            for other in selected:
                if not cand.terms or not other.terms:
                    continue
                overlap = len(cand.terms & other.terms) / len(cand.terms | other.terms)
                if overlap >= 0.85:
                    return True
            return False

        for cand in candidates:
            if len(selected) >= limit:
                break
            if is_duplicate(cand):
                continue
            doc_id = str(cand.match.get("document_id"))
            if per_doc[doc_id] >= max_per_doc:
                overflow.append(cand)
                continue
            selected.append(cand)
            per_doc[doc_id] += 1

        for cand in overflow:
            if len(selected) >= limit:
                break
            if not is_duplicate(cand):
                selected.append(cand)

        return selected

    @staticmethod
    def _relevance(cand: _Candidate, threshold: float) -> float:
        if cand.rerank is not None:
            return round(cand.rerank, 4)
        if cand.dense is not None:
            return round(cand.dense, 4)
        # Lexical-only hit: just at the semantic threshold, scaled by coverage.
        return round(threshold * (0.75 + 0.25 * cand.coverage), 4)

    def _to_chunk(
        self,
        cand: _Candidate,
        threshold: float,
    ) -> RetrievedChunk:
        match = cand.match
        page = match.get("page")
        page_end = match.get("page_end") or page
        pages = (
            list(range(page, page_end + 1))
            if page is not None and page_end is not None and page_end >= page
            else ([page] if page is not None else [])
        )

        return RetrievedChunk(
            id=match["id"],
            text=(match.get("text") or "").strip(),
            score=self._relevance(cand, threshold),
            document_id=uuid.UUID(str(match["document_id"])),
            knowledge_base_id=uuid.UUID(str(match["knowledge_base_id"])),
            chunk_index=match.get("chunk_index"),
            page=page,
            file_name=match.get("file_name"),
            metadata={
                **(match.get("metadata") or {}),
                "section": match.get("section"),
                "pages": pages,
                "dense_score": cand.dense,
                "lexical_coverage": round(cand.coverage, 3),
            },
        )

    async def _expand_neighbors(
        self,
        selected: list[_Candidate],
        *,
        user_id: uuid.UUID,
        threshold: float,
        char_budget: int | None,
    ) -> list[RetrievedChunk]:
        """
        Merge each hit with up to RAG_NEIGHBOR_WINDOW chunks on either side
        (same document and ingest) so the LLM sees complete explanations
        instead of fragments. Adjacent hits collapse into one passage.
        """

        window = settings.RAG_NEIGHBOR_WINDOW
        positions: dict[str, list[int]] = defaultdict(list)

        for cand in selected:
            index = cand.match.get("chunk_index")
            if index is None:
                continue
            doc_id = str(cand.match["document_id"])
            positions[doc_id].extend(
                i for i in range(index - window, index + window + 1) if i >= 0
            )

        try:
            fetched = await self.vector_store.fetch_chunks(
                user_id=user_id,
                positions=positions,
            )
        except Exception as exc:
            logger.warning("Neighbour expansion failed: %s", exc)
            return [self._to_chunk(c, threshold) for c in selected]

        by_position: dict[tuple[str, str | None, int], dict[str, Any]] = {
            (str(m["document_id"]), m.get("ingest_id"), m["chunk_index"]): m
            for m in fetched
            if m.get("chunk_index") is not None
        }

        passages: list[RetrievedChunk] = []
        covered: dict[tuple[str, str | None], set[int]] = defaultdict(set)
        remaining = char_budget

        for cand in selected:
            hit = self._to_chunk(cand, threshold)
            index = cand.match.get("chunk_index")
            key = (str(cand.match["document_id"]), cand.match.get("ingest_id"))

            if index is None:
                passages.append(hit)
                continue

            if index in covered[key]:
                # Already inside an earlier passage: lift its score.
                for passage in passages:
                    if (
                        str(passage.document_id) == key[0]
                        and index in passage.metadata.get("chunk_indexes", [])
                    ):
                        passage.score = max(passage.score, hit.score)
                continue

            # Grow outward from the hit while within the passage limit.
            run = [index]
            length = len(hit.text)
            limit = self.MAX_PASSAGE_CHARS
            if remaining is not None:
                limit = min(limit, max(len(hit.text), remaining // max(1, len(selected))))

            low = high = index
            for _ in range(window):
                for neighbor in (low - 1, high + 1):
                    match = by_position.get((*key, neighbor))
                    if match is None or neighbor in covered[key]:
                        continue
                    extra = len(match.get("text") or "")
                    if length + extra > limit:
                        continue
                    run.append(neighbor)
                    length += extra
                    low, high = min(low, neighbor), max(high, neighbor)

            run.sort()
            parts = []
            pages: set[int] = set(hit.metadata.get("pages") or [])

            for position in run:
                match = (
                    cand.match
                    if position == index
                    else by_position[(*key, position)]
                )
                text = (match.get("text") or "").strip()
                overlap = (match.get("metadata") or {}).get("overlap_chars", 0)
                # Skip the carried overlap when the previous chunk is present.
                if parts and overlap and (position - 1) in run:
                    text = text[overlap:].strip()
                parts.append(text)

                page = match.get("page")
                page_end = match.get("page_end") or page
                if page is not None:
                    pages.update(range(page, (page_end or page) + 1))

            hit.text = "\n\n".join(p for p in parts if p)
            hit.page = min(pages) if pages else hit.page
            hit.metadata["pages"] = sorted(pages)
            hit.metadata["chunk_indexes"] = run
            passages.append(hit)
            covered[key].update(run)

            if remaining is not None:
                remaining = max(0, remaining - len(hit.text))

        return passages

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

        # Multilingual sentence splitting: supports ., !, ?, Hindi ।, Chinese/Japanese 。, Arabic ؟, newlines
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?।。\n\r؟])\s*", cleaned)
            if s.strip()
        ]
        if len(sentences) <= 1:
            return cleaned[:max_chars].strip() + "..."

        # Language-agnostic Unicode token extraction
        query_words = {
            w.lower()
            for w in re.findall(r"\w+", query, re.UNICODE)
            if len(w) >= 2 or any(ord(c) > 127 for c in w)
        }

        scored = []
        for idx, sentence in enumerate(sentences):
            sent_words = {w.lower() for w in re.findall(r"\w+", sentence, re.UNICODE)}
            overlap = len(query_words.intersection(sent_words)) if query_words else 1
            # Substring matching for unsegmented or continuous scripts
            for qw in query_words:
                if qw in sentence.lower():
                    overlap += 1
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
