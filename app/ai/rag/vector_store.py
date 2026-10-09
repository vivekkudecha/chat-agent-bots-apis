import asyncio
import logging
import uuid
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    QuantizationSearchParams,
    QueryRequest,
    SearchParams,
    SparseVector,
)

from app.config import settings

from app.ai.llm.embeddings import (
    EmbeddingProvider,
    get_embedding_provider,
)

from app.ai.rag.chunking import ChunkingService
from app.ai.rag.sparse import BM25SparseEncoder

from app.integrations.qdrant import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    collection_exists,
    ensure_hybrid_collection,
    get_qdrant_client,
)

from app.core.exceptions import (
    RetrievalException,
)

logger = logging.getLogger(__name__)


def hybrid_collection_name() -> str:
    return f"{settings.QDRANT_COLLECTION}_v{settings.RAG_INDEX_VERSION}"


class VectorStoreService:
    """
    Qdrant access for knowledge chunks.

    Each point stores a named dense vector and a BM25 sparse vector built
    from the chunk's contextual embedding text (document + section header
    + body). Points carry an ``ingest_id`` so a reprocessed document is
    swapped atomically: new points are written first, then older ingests
    are deleted.
    """

    # Chunks indexed before the hybrid schema live in the base collection;
    # deletes still clean them up until they are reindexed.
    _legacy_exists: bool | None = None

    # Collections created/verified by this process (Celery workers do not
    # run the FastAPI lifespan, so the first write ensures the schema).
    _ready: set[str] = set()

    def __init__(
        self,
        embedding_provider: (
            EmbeddingProvider | None
        ) = None,
        sparse_encoder: BM25SparseEncoder | None = None,
    ):

        self.client = get_qdrant_client()

        self.embedding_provider = (
            embedding_provider
            or get_embedding_provider()
        )

        self.sparse_encoder = (
            sparse_encoder
            or BM25SparseEncoder()
        )

        self.collection_name = hybrid_collection_name()

        self.legacy_collection_name = (
            settings.QDRANT_COLLECTION
        )

    # =====================================================
    # INITIALIZE
    # =====================================================

    async def initialize(self) -> None:

        # Learn the real vector size before creating the collection. An
        # unreachable embedding server must not block API startup.
        try:
            await self.embedding_provider.probe()
        except Exception as exc:
            logger.warning(
                "Embedding probe failed (%s); using EMBEDDING_DIMENSION=%d",
                exc,
                self.embedding_provider.dimension,
            )

        await ensure_hybrid_collection(
            collection_name=self.collection_name,
            vector_size=(
                self.embedding_provider.dimension
            ),
        )
        VectorStoreService._ready.add(self.collection_name)

    async def ensure_ready(self) -> None:
        if self.collection_name not in VectorStoreService._ready:
            await self.initialize()

    # =====================================================
    # FILTERS
    # =====================================================

    @staticmethod
    def _scope_filter(
        *,
        user_id: uuid.UUID,
        knowledge_base_ids: list[uuid.UUID],
        document_ids: list[uuid.UUID] | None = None,
    ) -> Filter:

        must = [
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

        if document_ids:
            must.append(
                FieldCondition(
                    key="document_id",
                    match=MatchAny(
                        any=[str(doc_id) for doc_id in document_ids]
                    ),
                )
            )

        return Filter(must=must)

    @staticmethod
    def _search_params() -> SearchParams:
        return SearchParams(
            hnsw_ef=128,
            quantization=QuantizationSearchParams(
                rescore=True,
                oversampling=2.0,
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
        ingest_id: str | None = None,
    ) -> int:

        if not chunks:
            return 0

        await self.ensure_ready()

        embed_texts = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            embed_texts.append(
                chunk.get("embed_text")
                or ChunkingService.embedding_text(
                    chunk["text"],
                    file_name=metadata.get("file_name"),
                    section=metadata.get("section"),
                )
            )

        vectors = (
            await self.embedding_provider
            .embed_documents(embed_texts)
        )

        points: list[PointStruct] = []

        for position, (chunk, vector, embed_text) in enumerate(
            zip(chunks, vectors, embed_texts)
        ):
            index = chunk.get("index", position)
            metadata = chunk.get("metadata", {})

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{document_id}:{ingest_id}:{index}"
                    if ingest_id
                    else f"{document_id}:{index}",
                )
            )

            sparse = self.sparse_encoder.encode_document(embed_text)

            payload = {
                "user_id": str(user_id),
                "knowledge_base_id": str(knowledge_base_id),
                "document_id": str(document_id),
                "ingest_id": ingest_id,
                "chunk_index": index,
                "text": chunk["text"],
                "page": chunk.get("page"),
                "page_end": metadata.get("page_end", chunk.get("page")),
                "section": metadata.get("section"),
                "file_name": metadata.get("file_name"),
                "metadata": metadata,
            }

            vector_payload: dict[str, Any] = {DENSE_VECTOR: vector}
            if sparse.indices:
                vector_payload[SPARSE_VECTOR] = SparseVector(
                    indices=sparse.indices,
                    values=sparse.values,
                )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector_payload,
                    payload=payload,
                )
            )

        batch_size = max(1, settings.RAG_UPSERT_BATCH_SIZE)
        semaphore = asyncio.Semaphore(2)

        async def write(batch: list[PointStruct]) -> None:
            async with semaphore:
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )

        await asyncio.gather(
            *(
                write(points[i:i + batch_size])
                for i in range(0, len(points), batch_size)
            )
        )

        return len(points)

    # =====================================================
    # HYBRID SEARCH
    # =====================================================

    async def hybrid_search(
        self,
        *,
        query: str,
        user_id: uuid.UUID,
        knowledge_base_ids: list[uuid.UUID],
        limit: int,
        document_ids: list[uuid.UUID] | None = None,
        score_threshold: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Dense and BM25 candidate lists for one query, fetched in a single
        batched request. Raw scores are kept so the caller can gate on
        dense similarity and fuse ranks itself.
        """

        if not knowledge_base_ids or not query.strip():
            return [], []

        query_filter = self._scope_filter(
            user_id=user_id,
            knowledge_base_ids=knowledge_base_ids,
            document_ids=document_ids,
        )

        try:
            query_vector = (
                await self.embedding_provider
                .embed_query(query)
            )
            sparse = self.sparse_encoder.encode_query(query)

            requests = [
                QueryRequest(
                    query=query_vector,
                    using=DENSE_VECTOR,
                    filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    params=self._search_params(),
                    with_payload=True,
                )
            ]

            if sparse.indices:
                requests.append(
                    QueryRequest(
                        query=SparseVector(
                            indices=sparse.indices,
                            values=sparse.values,
                        ),
                        using=SPARSE_VECTOR,
                        filter=query_filter,
                        limit=limit,
                        with_payload=True,
                    )
                )

            responses = await self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=requests,
            )

        except Exception as exc:
            raise RetrievalException(
                "Vector search failed."
            ) from exc

        dense = [self._to_match(p) for p in responses[0].points]
        lexical = (
            [self._to_match(p) for p in responses[1].points]
            if len(responses) > 1
            else []
        )

        return dense, lexical

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

            result = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=DENSE_VECTOR,
                query_filter=self._scope_filter(
                    user_id=user_id,
                    knowledge_base_ids=knowledge_base_ids,
                ),
                limit=top_k,
                score_threshold=score_threshold,
                search_params=self._search_params(),
                with_payload=True,
            )

        except Exception as exc:
            raise RetrievalException(
                "Vector search failed."
            ) from exc

        return [self._to_match(p) for p in result.points]

    # =====================================================
    # KEYWORD / LEXICAL SEARCH (BM25)
    # =====================================================

    async def search_keyword(
        self,
        *,
        query: str,
        user_id: uuid.UUID,
        knowledge_base_ids: list[uuid.UUID],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        sparse = self.sparse_encoder.encode_query(query)

        if not knowledge_base_ids or not sparse.indices:
            return []

        try:
            result = await self.client.query_points(
                collection_name=self.collection_name,
                query=SparseVector(
                    indices=sparse.indices,
                    values=sparse.values,
                ),
                using=SPARSE_VECTOR,
                query_filter=self._scope_filter(
                    user_id=user_id,
                    knowledge_base_ids=knowledge_base_ids,
                ),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning(
                "Keyword search failed, falling back to empty: %s",
                exc,
            )
            return []

        return [self._to_match(p) for p in result.points]

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
    # FETCH CHUNKS BY POSITION (neighbour expansion)
    # =====================================================

    async def fetch_chunks(
        self,
        *,
        user_id: uuid.UUID,
        positions: dict[str, list[int]],
    ) -> list[dict[str, Any]]:
        """Chunks by (document_id -> chunk indexes), one request."""

        if not positions:
            return []

        per_document = [
            Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    ),
                    FieldCondition(
                        key="chunk_index",
                        match=MatchAny(any=sorted(set(indexes))),
                    ),
                ]
            )
            for document_id, indexes in positions.items()
            if indexes
        ]

        limit = sum(len(set(i)) for i in positions.values())

        points, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=str(user_id)),
                    )
                ],
                should=per_document,
            ),
            # Headroom for a document mid-reprocess (two ingests).
            limit=limit * 2,
            with_payload=True,
            with_vectors=False,
        )

        return [self._to_match(p) for p in points]

    # =====================================================
    # DELETE
    # =====================================================

    async def _legacy_available(self) -> bool:
        cls = type(self)
        if cls._legacy_exists is None:
            try:
                cls._legacy_exists = await collection_exists(
                    self.legacy_collection_name
                )
            except Exception:
                return False
        return cls._legacy_exists

    async def _delete(self, points_filter: Filter, *, include_legacy: bool = True) -> None:

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=points_filter,
            wait=True,
        )

        if include_legacy and await self._legacy_available():
            await self.client.delete(
                collection_name=self.legacy_collection_name,
                points_selector=points_filter,
                wait=True,
            )

    async def delete_document(
        self,
        document_id: uuid.UUID,
    ) -> None:

        await self._delete(
            Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id)),
                    )
                ]
            )
        )

    async def delete_stale_ingests(
        self,
        document_id: uuid.UUID,
        *,
        keep_ingest_id: str,
    ) -> None:
        """Drop every point of the document except the given ingest."""

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id)),
                    )
                ],
                must_not=[
                    FieldCondition(
                        key="ingest_id",
                        match=MatchValue(value=keep_ingest_id),
                    )
                ],
            ),
            wait=True,
        )

        if await self._legacy_available():
            await self.client.delete(
                collection_name=self.legacy_collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                ),
                wait=True,
            )

    async def delete_ingest(
        self,
        document_id: uuid.UUID,
        ingest_id: str,
    ) -> None:
        """Remove a partially written ingest after a failure."""

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id)),
                    ),
                    FieldCondition(
                        key="ingest_id",
                        match=MatchValue(value=ingest_id),
                    ),
                ]
            ),
            wait=True,
        )

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

        await self._delete(Filter(must=must_conditions))

    # =====================================================
    # RESULT MAPPING
    # =====================================================

    @staticmethod
    def _to_match(point: Any) -> dict[str, Any]:
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}

        return {
            "id": str(point.id),
            "score": getattr(point, "score", None) or 0.0,
            "text": payload.get("text", ""),
            "document_id": payload.get("document_id"),
            "knowledge_base_id": payload.get("knowledge_base_id"),
            "ingest_id": payload.get("ingest_id"),
            "chunk_index": payload.get("chunk_index"),
            "page": payload.get("page"),
            "page_end": payload.get("page_end"),
            "section": payload.get("section"),
            "file_name": (
                payload.get("file_name")
                or metadata.get("file_name")
            ),
            "metadata": metadata,
        }
