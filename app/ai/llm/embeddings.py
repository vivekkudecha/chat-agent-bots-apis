import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Any

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

    async def probe(self) -> int:
        """Confirm the real vector size (remote providers call the model)."""
        return self.dimension

    def describe(self) -> dict[str, Any]:
        return {"provider": type(self).__name__, "dimension": self.dimension}


# =========================================================
# Query Instructions
# =========================================================

# Asymmetric retrieval models expect queries (not documents) to carry an
# instruction; without it recall drops noticeably.
_QUERY_INSTRUCTIONS = (
    ("qwen3-embedding", "Instruct: Given a question, retrieve passages from documents that answer the question\nQuery: "),
    ("e5", "query: "),
    ("nomic-embed", "search_query: "),
    ("bge-", "Represent this sentence for searching relevant passages: "),
)


def query_instruction(model_name: str) -> str:
    configured = settings.EMBEDDING_QUERY_INSTRUCTION
    if configured is not None:
        return configured.replace("\\n", "\n")

    name = (model_name or "").lower()
    if "bge-m3" in name:
        return ""
    for marker, instruction in _QUERY_INSTRUCTIONS:
        if marker in name:
            return instruction
    return ""


# =========================================================
# HTTP Embedding Providers (shared transport)
# =========================================================

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class HTTPEmbeddingProvider(EmbeddingProvider):
    """
    Base for remote embedding servers.

    Mirrors the LLM client configuration: base URL, bearer API key, SSL
    verification, proxy env (trust_env), timeout and extra headers.
    Adds bounded concurrency, retries with exponential backoff, and a
    dimension check against the configured/probed vector size.

    One pooled ``httpx.AsyncClient`` is kept per event loop (Celery runs
    each task in a fresh loop via ``asyncio.run``).
    """

    provider_name = "http"

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        dimension: int | None = None,
        verify_ssl: bool = True,
        trust_env: bool = True,
        timeout: float = 60.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
        batch_size: int | None = None,
        concurrency: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.trust_env = trust_env
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.batch_size = max(1, batch_size or settings.EMBEDDING_BATCH_SIZE)
        self.concurrency = max(1, concurrency or settings.EMBEDDING_CONCURRENCY)
        self._transport = transport

        self._headers = {
            "Content-Type": "application/json",
            **(headers or {}),
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._dimension = dimension or settings.EMBEDDING_DIMENSION
        self._probed = False

        self._http: httpx.AsyncClient | None = None
        self._http_loop: asyncio.AbstractEventLoop | None = None

    # -----------------------------------------------------
    # Interface
    # -----------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        semaphore = asyncio.Semaphore(self.concurrency)
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        async def run(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._embed(batch)

        # gather preserves input order.
        results = await asyncio.gather(*(run(b) for b in batches))
        return [vector for batch in results for vector in batch]

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        if not text:
            return [0.0] * self.dimension

        [vector] = await self._embed([query_instruction(self.model_name) + text])
        return vector

    async def probe(self) -> int:
        """
        Embed a probe string to learn the model's real vector size; later
        responses of a different size are rejected instead of silently
        corrupting the index.
        """
        [vector] = await self._embed(["dimension probe"], strict=False)
        if len(vector) != settings.EMBEDDING_DIMENSION:
            logger.warning(
                "Embedding model %s returns %d dimensions but "
                "EMBEDDING_DIMENSION=%d; using %d.",
                self.model_name,
                len(vector),
                settings.EMBEDDING_DIMENSION,
                len(vector),
            )
        self._dimension = len(vector)
        self._probed = True
        return self._dimension

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "base_url": self.base_url,
            "verify_ssl": self.verify_ssl,
            "trust_env": self.trust_env,
            "auth": "Authorization" in self._headers,
            "dimension": self._dimension,
        }

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    # -----------------------------------------------------
    # Provider-specific request
    # -----------------------------------------------------

    @abstractmethod
    async def _request(self, texts: list[str]) -> list[list[float]]:
        pass

    # -----------------------------------------------------
    # Shared plumbing
    # -----------------------------------------------------

    async def _embed(
        self,
        texts: list[str],
        *,
        strict: bool = True,
    ) -> list[list[float]]:
        # Servers reject empty strings; a space embeds as "nothing".
        texts = [t if t and t.strip() else " " for t in texts]
        vectors = await self._request(texts)

        if len(vectors) != len(texts):
            raise ModelExecutionException(
                f"Embedding server returned {len(vectors)} vectors "
                f"for {len(texts)} inputs ({self.model_name})."
            )

        if strict:
            for vector in vectors:
                self._check_dimension(len(vector))

        return vectors

    def _check_dimension(self, size: int) -> None:
        if size == self._dimension:
            return
        if self._probed:
            raise ModelExecutionException(
                f"Embedding model {self.model_name} returned {size} dimensions, "
                f"expected {self._dimension}. The model changed since startup; "
                f"reindex with a new RAG_INDEX_VERSION."
            )
        logger.warning(
            "Embedding dimension %d differs from configured %d; adopting %d.",
            size,
            self._dimension,
            size,
        )
        self._dimension = size

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if (
            self._http is None
            or self._http.is_closed
            or self._http_loop is not loop
        ):
            self._http = httpx.AsyncClient(
                headers=self._headers,
                verify=self.verify_ssl,
                trust_env=self.trust_env,
                timeout=self.timeout,
                transport=self._transport,
                limits=httpx.Limits(
                    max_connections=self.concurrency * 2,
                    max_keepalive_connections=self.concurrency,
                ),
            )
            self._http_loop = loop
        return self._http

    async def _post(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        """POST with retries on transient failures; returns any final response."""

        client = self._client()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=payload)
                if response.status_code not in RETRYABLE_STATUS:
                    return response
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    request=response.request,
                    response=response,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                delay = min(8.0, 0.5 * 2 ** attempt) + random.uniform(0, 0.25)
                logger.warning(
                    "Embedding request to %s failed (%s); retry %d/%d in %.1fs",
                    url,
                    last_error,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

        raise ModelExecutionException(
            f"Embedding server {self.base_url} unreachable for model "
            f"'{self.model_name}' after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return

        detail = response.text[:300]
        if response.status_code in (401, 403):
            hint = "check EMBEDDING_API_KEY"
        elif response.status_code == 404 and "model" in detail.lower():
            hint = f"model '{self.model_name}' not available on the server"
        else:
            hint = "embedding request rejected"

        raise ModelExecutionException(
            f"Embedding server {self.base_url} returned HTTP "
            f"{response.status_code} ({hint}): {detail}"
        )


# =========================================================
# Ollama (native API)
# =========================================================

class OllamaEmbeddingProvider(HTTPEmbeddingProvider):
    """
    Native Ollama: batched POST /api/embed. Servers older than 0.3 only
    have /api/embeddings (one text per call); that is detected once and
    remembered.
    """

    provider_name = "ollama"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        dimension: int | None = None,
        timeout: float = 60.0,
        *,
        keep_alive: str | None = None,
        **kwargs: Any,
    ):
        root = (
            base_url
            or settings.OLLAMA_BASE_URL
            or "http://localhost:11434"
        ).rstrip("/")

        for suffix in ("/api", "/v1"):
            if root.endswith(suffix):
                root = root[:-len(suffix)]

        super().__init__(
            model_name=model_name,
            base_url=root,
            dimension=dimension,
            timeout=timeout,
            **kwargs,
        )
        self.keep_alive = keep_alive
        self._legacy_api = False

    async def _request(self, texts: list[str]) -> list[list[float]]:
        if not self._legacy_api:
            payload: dict[str, Any] = {
                "model": self.model_name,
                "input": texts,
                "truncate": True,
            }
            if self.keep_alive:
                payload["keep_alive"] = self.keep_alive

            response = await self._post(f"{self.base_url}/api/embed", payload)

            # 404 without a model error = endpoint missing (old server).
            if response.status_code == 404 and "model" not in response.text.lower():
                logger.warning(
                    "Ollama at %s has no /api/embed; using legacy /api/embeddings.",
                    self.base_url,
                )
                self._legacy_api = True
            else:
                self._raise_for_status(response)
                embeddings = response.json().get("embeddings") or []
                if not embeddings:
                    raise ModelExecutionException(
                        f"Ollama returned no embeddings for model '{self.model_name}'."
                    )
                return embeddings

        vectors = []
        for text in texts:
            response = await self._post(
                f"{self.base_url}/api/embeddings",
                {"model": self.model_name, "prompt": text},
            )
            self._raise_for_status(response)
            embedding = response.json().get("embedding") or []
            if not embedding:
                raise ModelExecutionException(
                    f"Ollama returned an empty embedding for model '{self.model_name}'."
                )
            vectors.append(embedding)
        return vectors


# =========================================================
# OpenAI-compatible (/v1/embeddings)
# =========================================================

class OpenAICompatibleEmbeddingProvider(HTTPEmbeddingProvider):
    """
    Any server implementing POST /v1/embeddings: vLLM, Hugging Face TEI,
    LiteLLM, OpenAI, Azure-style gateways, or Ollama's /v1 endpoint.
    """

    provider_name = "openai"

    def __init__(self, *, base_url: str, **kwargs: Any):
        url = base_url.rstrip("/")
        if url.endswith("/api"):
            url = url[:-4]
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        super().__init__(base_url=url, **kwargs)

    async def _request(self, texts: list[str]) -> list[list[float]]:
        response = await self._post(
            f"{self.base_url}/embeddings",
            {
                "model": self.model_name,
                "input": texts,
                "encoding_format": "float",
            },
        )
        self._raise_for_status(response)

        data = response.json().get("data") or []
        # The spec allows any order; "index" ties each vector to its input.
        data = sorted(data, key=lambda item: item.get("index", 0))
        return [item.get("embedding") or [] for item in data]


# =========================================================
# Sentence Transformers
# =========================================================

class SentenceTransformerEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        model_name: str,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Unable to determine embedding dimension.")

        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

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

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "local",
            "model": self.model_name,
            "dimension": self._dimension,
        }

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        embeddings = await asyncio.to_thread(
            self.model.encode,
            [query_instruction(self.model_name) + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings[0].tolist()


# =========================================================
# Singleton Factory
# =========================================================

_embedding_provider: EmbeddingProvider | None = None

OLLAMA_PROVIDERS = {"ollama"}
OPENAI_PROVIDERS = {"openai", "openai_compatible", "openai-compatible", "vllm", "tei", "litellm"}
LOCAL_PROVIDERS = {"local", "sentence_transformers", "sentence-transformers", "huggingface"}


def _setting(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def build_embedding_provider() -> EmbeddingProvider:
    provider = (settings.EMBEDDING_PROVIDER or "").lower().strip()
    model_name = (settings.EMBEDDING_MODEL or "").strip()

    # Legacy: "local" with an Ollama-style model tag meant Ollama.
    if provider == "local" and (
        ":" in model_name
        or "ollama" in model_name.lower()
        or "qwen" in model_name.lower()
    ):
        provider = "ollama"

    http_options: dict[str, Any] = {
        "model_name": model_name,
        "dimension": settings.EMBEDDING_DIMENSION,
        "verify_ssl": _setting(settings.EMBEDDING_VERIFY_SSL, settings.LLM_VERIFY_SSL),
        "trust_env": _setting(settings.EMBEDDING_TRUST_ENV, settings.LLM_TRUST_ENV),
        "timeout": settings.EMBEDDING_TIMEOUT_SECONDS,
        "max_retries": settings.EMBEDDING_MAX_RETRIES,
        "headers": settings.EMBEDDING_HEADERS,
    }

    if provider in OLLAMA_PROVIDERS:
        instance: EmbeddingProvider = OllamaEmbeddingProvider(
            base_url=settings.EMBEDDING_BASE_URL or settings.OLLAMA_BASE_URL,
            api_key=settings.EMBEDDING_API_KEY,
            keep_alive=settings.EMBEDDING_KEEP_ALIVE,
            **http_options,
        )
    elif provider in OPENAI_PROVIDERS:
        # Defaults to the LLM endpoint/key when embeddings share the server.
        instance = OpenAICompatibleEmbeddingProvider(
            base_url=settings.EMBEDDING_BASE_URL or settings.VLLM_BASE_URL,
            api_key=settings.EMBEDDING_API_KEY or settings.VLLM_API_KEY,
            **http_options,
        )
    elif provider in LOCAL_PROVIDERS:
        instance = SentenceTransformerEmbeddingProvider(model_name=model_name)
    else:
        raise RuntimeError(
            f"Unsupported EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
            f"Use one of: ollama, openai, local."
        )

    logger.info("Embedding provider: %s", instance.describe())
    return instance


def get_embedding_provider() -> EmbeddingProvider:
    global _embedding_provider

    if _embedding_provider is None:
        _embedding_provider = build_embedding_provider()

    return _embedding_provider
