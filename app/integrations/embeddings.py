import asyncio
import logging
from abc import ABC, abstractmethod

import httpx

from app.config import settings
from app.core.exceptions import ModelExecutionException

logger = logging.getLogger(__name__)


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
# Ollama Embedding Provider
# =========================================================

class OllamaEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        dimension: int | None = None,
        timeout: float = 60.0,
    ):
        self.model_name = model_name

        raw_base_url = (
            base_url
            or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")

        for suffix in ("/api", "/v1"):
            if raw_base_url.endswith(suffix):
                raw_base_url = raw_base_url[:-len(suffix)]

        self.base_url = raw_base_url
        self._dimension = dimension or settings.EMBEDDING_DIMENSION
        self.timeout = timeout

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        batch_size = 32

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                try:
                    # 1. Try native Ollama /api/embed endpoint (supports batching)
                    response = await client.post(
                        f"{self.base_url}/api/embed",
                        json={
                            "model": self.model_name,
                            "input": batch,
                        },
                    )

                    if response.status_code == 200:
                        data = response.json()
                        embeddings = data.get("embeddings", [])
                        if embeddings:
                            if len(embeddings[0]) != self._dimension:
                                self._dimension = len(embeddings[0])
                            results.extend(embeddings)
                            continue

                    logger.warning(
                        "Ollama /api/embed returned status %d. Falling back to /api/embeddings.",
                        response.status_code,
                    )
                except httpx.HTTPError as exc:
                    logger.warning(
                        "Ollama /api/embed request failed: %s. Falling back to /api/embeddings.",
                        exc,
                    )

                # 2. Fallback to /api/embeddings per item
                for text in batch:
                    item_emb = await self._embed_single_fallback(client, text)
                    results.append(item_emb)

        return results

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        if not text:
            return [0.0] * self.dimension

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model_name,
                        "input": text,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings:
                        emb = embeddings[0]
                        if len(emb) != self._dimension:
                            self._dimension = len(emb)
                        return emb

                logger.warning(
                    "Ollama /api/embed returned status %d. Trying /api/embeddings.",
                    response.status_code,
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "Ollama /api/embed failed: %s. Trying /api/embeddings.",
                    exc,
                )

            return await self._embed_single_fallback(client, text)

    async def _embed_single_fallback(
        self,
        client: httpx.AsyncClient,
        text: str,
    ) -> list[float]:
        try:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])
            if not embedding:
                raise ModelExecutionException(
                    "Ollama returned empty embedding."
                )

            if len(embedding) != self._dimension:
                self._dimension = len(embedding)

            return embedding

        except Exception as exc:
            raise ModelExecutionException(
                f"Failed to generate embeddings from Ollama at {self.base_url} "
                f"for model '{self.model_name}': {exc}"
            ) from exc


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
        from sentence_transformers import SentenceTransformer

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
# Singleton Factory
# =========================================================

_embedding_provider: (
    EmbeddingProvider | None
) = None


def get_embedding_provider() -> EmbeddingProvider:
    global _embedding_provider

    if _embedding_provider is None:
        provider = (settings.EMBEDDING_PROVIDER or "").lower().strip()
        model_name = (settings.EMBEDDING_MODEL or "").strip()

        # Route to Ollama if explicitly specified or if using an Ollama model tag (e.g. contains colon or 'ollama')
        if (
            provider == "ollama"
            or (provider == "local" and (":" in model_name or "ollama" in model_name.lower() or "qwen" in model_name.lower()))
        ):
            _embedding_provider = OllamaEmbeddingProvider(
                model_name=model_name,
                base_url=getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"),
                dimension=settings.EMBEDDING_DIMENSION,
            )
        elif provider in (
            "sentence_transformers",
            "sentence-transformers",
            "local",
            "huggingface",
        ):
            _embedding_provider = SentenceTransformerEmbeddingProvider(
                model_name=model_name
            )
        else:
            raise RuntimeError(
                f"Unsupported embedding provider: {provider}"
            )

    return _embedding_provider