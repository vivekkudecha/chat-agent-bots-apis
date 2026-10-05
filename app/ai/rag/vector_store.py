import logging
import re
import uuid
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchText,
    MatchValue,
    PointStruct,
)

from app.config import settings

from app.ai.llm.embeddings import (
    EmbeddingProvider,
    get_embedding_provider,
)

from app.integrations.qdrant import (
    ensure_collection,
    get_qdrant_client,
)

from app.core.exceptions import (
    RetrievalException,
)

logger = logging.getLogger(__name__)


class VectorStoreService:

    def __init__(
        self,
        embedding_provider: (
            EmbeddingProvider | None
        ) = None,
    ):

        self.client = get_qdrant_client()

        self.embedding_provider = (
            embedding_provider
            or get_embedding_provider()
        )

        self.collection_name = (
            settings.QDRANT_COLLECTION
        )

    # =====================================================
    # INITIALIZE
    # =====================================================

    async def initialize(self) -> None:

        await ensure_collection(
            collection_name=self.collection_name,
            vector_size=(
                self.embedding_provider.dimension
            ),
        )

    # =====================================================
    # UPSERT CHUNKS
    # =====================================================

    async def upsert_chunks(
        self,
        *,
        user_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        document_id: uuid.UUID,
        chunks: list[dict[str, Any]],
    ) -> int:

        if not chunks:
            return 0

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        vectors = (
            await self.embedding_provider
            .embed_documents(texts)
        )

        points: list[PointStruct] = []

        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors)
        ):

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"{document_id}:"
                        f"{chunk.get('index', index)}"
                    ),
                )
            )

            payload = {
                "user_id": str(user_id),

                "knowledge_base_id": str(
                    knowledge_base_id
                ),

                "document_id": str(
                    document_id
                ),

                "chunk_index": chunk.get(
                    "index",
                    index,
                ),

                "text": chunk["text"],

                "page": chunk.get("page"),

                "file_name": chunk.get("metadata", {}).get(
                    "file_name"
                ),

                "metadata": chunk.get(
                    "metadata",
                    {},
                ),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert points in chunks of 250 to prevent connection timeouts
        batch_size = 250
        for i in range(0, len(points), batch_size):
            batch_points = points[i : i + batch_size]
            await self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
                wait=(i + batch_size >= len(points)),
            )

        return len(points)

    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    async def search(
        self,
        *,
        query: str,
        user_id: uuid.UUID,
        knowledge_base_ids: list[uuid.UUID],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:

        if not knowledge_base_ids:
            return []

        try:

            query_vector = (
                await self.embedding_provider
                .embed_query(query)
            )

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(
                            value=str(user_id)
                        ),
                    ),
                    FieldCondition(
                        key="knowledge_base_id",
                        match=MatchAny(
                            any=[
                                str(kb_id)
                                for kb_id
                                in knowledge_base_ids
                            ]
                        ),
                    ),
                ]
            )

            result = await self.client.query_points(
                collection_name=(
                    self.collection_name
                ),
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

            matches = []

            for point in result.points:

                payload = point.payload or {}

                matches.append(
                    {
                        "id": str(point.id),

                        "score": point.score,

                        "text": payload.get(
                            "text",
                            "",
                        ),

                        "document_id": (
                            payload.get(
                                "document_id"
                            )
                        ),

                        "knowledge_base_id": (
                            payload.get(
                                "knowledge_base_id"
                            )
                        ),

                        "chunk_index": (
                            payload.get(
                                "chunk_index"
                            )
                        ),

                        "page": payload.get(
                            "page"
                        ),

                        "file_name": (
                            payload.get("file_name")
                            or (
                                payload.get(
                                    "metadata",
                                    {},
                                ).get("file_name")
                            )
                        ),

                        "metadata": (
                            payload.get(
                                "metadata",
                                {},
                            )
                        ),
                    }
                )

            return matches

        except Exception as exc:

            raise RetrievalException(
                "Vector search failed."
            ) from exc

    # =====================================================
    # KEYWORD / LEXICAL SEARCH
    # =====================================================

    async def search_keyword(
        self,
        *,
        query: str,
        user_id: uuid.UUID,
        knowledge_base_ids: list[uuid.UUID],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if not knowledge_base_ids or not query.strip():
            return []

        try:
            # Language-agnostic Unicode token extraction (Latin, Indic, CJK, Arabic, Cyrillic, etc.)
            raw_tokens = re.findall(r"\w+", query, re.UNICODE)
            stop_words_setting = getattr(settings, "RAG_STOP_WORDS", None)
            custom_stops = (
                {w.strip().lower() for w in stop_words_setting.split(",") if w.strip()}
                if stop_words_setting
                else set()
            )

            tokens = [
                w.strip().lower()
                for w in raw_tokens
                if (len(w.strip()) >= 2 or any(ord(c) > 127 for c in w.strip()))
                and w.strip().lower() not in custom_stops
            ]

            must_conditions = [
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=str(user_id)),
                ),
                FieldCondition(
                    key="knowledge_base_id",
                    match=MatchAny(
                        any=[str(kb_id) for kb_id in knowledge_base_ids]
                    ),
                ),
            ]

            should_conditions = []
            clean_query = query.strip()
            if len(clean_query) >= 2 or any(ord(c) > 127 for c in clean_query):
                should_conditions.append(
                    FieldCondition(
                        key="text",
                        match=MatchText(text=clean_query),
                    )
                )

            for token in tokens[:8]:
                should_conditions.append(
                    FieldCondition(
                        key="text",
                        match=MatchText(text=token),
                    )
                )

            query_filter = Filter(
                must=must_conditions,
                should=should_conditions if should_conditions else None,
            )

            res, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=top_k * 3,
                with_payload=True,
            )

            if not res:
                return []

            matches = []
            query_lower = query.lower()
            query_tokens_set = set(tokens)

            for point in res:
                payload = point.payload or {}
                text = payload.get("text", "")
                text_lower = text.lower()

                token_hits = sum(1 for t in query_tokens_set if t in text_lower)
                overlap_ratio = (
                    token_hits / max(1, len(query_tokens_set))
                    if query_tokens_set
                    else 0.5
                )
                phrase_boost = 0.3 if query_lower in text_lower else 0.0
                lexical_score = min(1.0, 0.4 * overlap_ratio + phrase_boost + 0.3)

                matches.append(
                    {
                        "id": str(point.id),
                        "score": round(lexical_score, 4),
                        "text": text,
                        "document_id": payload.get("document_id"),
                        "knowledge_base_id": payload.get("knowledge_base_id"),
                        "chunk_index": payload.get("chunk_index"),
                        "page": payload.get("page"),
                        "file_name": (
                            payload.get("file_name")
                            or payload.get("metadata", {}).get("file_name")
                        ),
                        "metadata": payload.get("metadata", {}),
                    }
                )

            matches.sort(key=lambda m: m["score"], reverse=True)
            return matches[:top_k]

        except Exception as exc:
            logger.warning(
                "Keyword search failed, falling back to empty: %s",
                exc,
            )
            return []

    # =====================================================
    # SEARCH ONE KNOWLEDGE BASE
    # =====================================================

    async def search_knowledge_base(
        self,
        *,
        query: str,
        user_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:

        return await self.search(
            query=query,
            user_id=user_id,
            knowledge_base_ids=[
                knowledge_base_id
            ],
            top_k=top_k,
            score_threshold=score_threshold,
        )

    # =====================================================
    # DELETE DOCUMENT VECTORS
    # =====================================================

    async def delete_document(
        self,
        document_id: uuid.UUID,
    ) -> None:

        await self.client.delete(
            collection_name=(
                self.collection_name
            ),
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=str(
                                document_id
                            )
                        ),
                    )
                ]
            ),
            wait=True,
        )

    # =====================================================
    # DELETE KNOWLEDGE BASE VECTORS
    # =====================================================

    async def delete_knowledge_base(
        self,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> None:

        must_conditions = [
            FieldCondition(
                key="knowledge_base_id",
                match=MatchValue(
                    value=str(
                        knowledge_base_id
                    )
                ),
            )
        ]

        if user_id is not None:
            must_conditions.append(
                FieldCondition(
                    key="user_id",
                    match=MatchValue(
                        value=str(user_id)
                    ),
                )
            )

        await self.client.delete(
            collection_name=(
                self.collection_name
            ),
            points_selector=Filter(
                must=must_conditions
            ),
            wait=True,
        )