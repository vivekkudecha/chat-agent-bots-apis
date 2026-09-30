import asyncio
from abc import ABC, abstractmethod

from sentence_transformers import (
    SentenceTransformer,
)

from app.config import settings


# =========================================================
# Interface
# =========================================================

class EmbeddingProvider(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        pass

    @abstractmethod
    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        pass


# =========================================================
# Sentence Transformers
# =========================================================

class SentenceTransformerEmbeddingProvider(
    EmbeddingProvider
):

    def __init__(
        self,
        model_name: str,
    ):

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

        dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "Unable to determine embedding dimension."
            )

        self._dimension = dimension

    # -----------------------------------------------------
    # Dimension
    # -----------------------------------------------------

    @property
    def dimension(self) -> int:

        return self._dimension

    # -----------------------------------------------------
    # Documents
    # -----------------------------------------------------

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    # -----------------------------------------------------
    # Query
    # -----------------------------------------------------

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:

        embeddings = await asyncio.to_thread(
            self.model.encode,
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings[0].tolist()


# =========================================================
# Singleton
# =========================================================

_embedding_provider: (
    EmbeddingProvider | None
) = None


def get_embedding_provider() -> EmbeddingProvider:

    global _embedding_provider

    if _embedding_provider is None:

        _embedding_provider = (
            SentenceTransformerEmbeddingProvider(
                settings.EMBEDDING_MODEL
            )
        )

    return _embedding_provider