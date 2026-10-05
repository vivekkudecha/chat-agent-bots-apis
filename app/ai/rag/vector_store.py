import uuid
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
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

        await self.client.upsert(
            collection_name=(
                self.collection_name
            ),
            points=points,
            wait=True,
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