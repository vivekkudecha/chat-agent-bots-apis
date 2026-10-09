import json
import unittest
from unittest.mock import patch

import httpx

# The application creates its Qdrant client at import time. Keep tests offline.
with patch("qdrant_client.AsyncQdrantClient"):
    from app.ai.llm import embeddings as module
    from app.ai.llm.embeddings import (
        OllamaEmbeddingProvider,
        OpenAICompatibleEmbeddingProvider,
        build_embedding_provider,
    )
    from app.core.exceptions import ModelExecutionException


class Server:
    """Scripted fake embedding server recording every request."""

    def __init__(self, handler):
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request, json.loads(request.content or b"{}"))

    @property
    def transport(self):
        return httpx.MockTransport(self)


def vector(text, dim=4):
    return [float(len(text) % 7), 1.0, 0.0, 0.5][:dim]


def ollama_ok(request, body):
    inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
    return httpx.Response(200, json={"embeddings": [vector(t) for t in inputs]})


def ollama(server, **kwargs):
    kwargs.setdefault("dimension", 4)
    return OllamaEmbeddingProvider(
        "qwen3-embedding:0.6b",
        base_url=kwargs.pop("base_url", "https://ollama.example.com/api/"),
        transport=server.transport,
        **kwargs,
    )


class OllamaProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_auth_headers_url_and_payload(self):
        server = Server(ollama_ok)
        provider = ollama(server, api_key="secret", headers={"X-Tenant": "acme"}, keep_alive="30m")

        await provider.embed_documents(["a", "bb"])

        [request] = server.requests
        self.assertEqual(str(request.url), "https://ollama.example.com/api/embed")
        self.assertEqual(request.headers["authorization"], "Bearer secret")
        self.assertEqual(request.headers["x-tenant"], "acme")
        body = json.loads(request.content)
        self.assertEqual(body, {"model": "qwen3-embedding:0.6b", "input": ["a", "bb"],
                                "truncate": True, "keep_alive": "30m"})

    async def test_ssl_and_proxy_options_reach_the_client(self):
        provider = OllamaEmbeddingProvider("m", base_url="https://x", verify_ssl=False, trust_env=False)
        with patch.object(module.httpx, "AsyncClient") as client:
            provider._client()
        options = client.call_args.kwargs
        self.assertFalse(options["verify"])
        self.assertFalse(options["trust_env"])

    async def test_batches_keep_input_order(self):
        server = Server(ollama_ok)
        provider = ollama(server, batch_size=2, concurrency=3)
        texts = ["a" * n for n in range(1, 8)]

        vectors = await provider.embed_documents(texts)

        self.assertEqual(len(server.requests), 4)
        self.assertEqual(vectors, [vector(t) for t in texts])

    async def test_query_gets_model_instruction_and_empty_text_is_safe(self):
        server = Server(ollama_ok)
        provider = ollama(server)

        await provider.embed_query("what is the leave policy")
        await provider.embed_documents([""])

        query = json.loads(server.requests[0].content)["input"][0]
        self.assertTrue(query.startswith("Instruct: "))
        self.assertTrue(query.endswith("Query: what is the leave policy"))
        self.assertEqual(json.loads(server.requests[1].content)["input"], [" "])

    async def test_retries_transient_errors(self):
        attempts = []

        def flaky(request, body):
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ConnectError("refused", request=request)
            if len(attempts) == 2:
                return httpx.Response(503, text="loading model")
            return ollama_ok(request, body)

        provider = ollama(Server(flaky), max_retries=2)
        with patch.object(module.asyncio, "sleep") as sleep:
            sleep.return_value = None
            vectors = await provider.embed_documents(["x"])

        self.assertEqual(len(attempts), 3)
        self.assertEqual(vectors, [vector("x")])

    async def test_gives_up_after_retries(self):
        provider = ollama(Server(lambda r, b: httpx.Response(502)), max_retries=1)
        with patch.object(module.asyncio, "sleep", return_value=None):
            with self.assertRaisesRegex(ModelExecutionException, "after 2 attempts"):
                await provider.embed_documents(["x"])

    async def test_auth_and_missing_model_errors_are_explicit(self):
        provider = ollama(Server(lambda r, b: httpx.Response(401, text="unauthorized")))
        with self.assertRaisesRegex(ModelExecutionException, "EMBEDDING_API_KEY"):
            await provider.embed_documents(["x"])

        provider = ollama(Server(lambda r, b: httpx.Response(404, json={"error": "model 'x' not found"})))
        with self.assertRaisesRegex(ModelExecutionException, "not available"):
            await provider.embed_documents(["x"])

    async def test_falls_back_to_legacy_endpoint_once(self):
        def old_server(request, body):
            if request.url.path == "/api/embed":
                return httpx.Response(404, text="404 page not found")
            return httpx.Response(200, json={"embedding": vector(body["prompt"])})

        server = Server(old_server)
        provider = ollama(server)

        await provider.embed_documents(["a", "bb"])
        await provider.embed_documents(["ccc"])

        paths = [r.url.path for r in server.requests]
        self.assertEqual(paths, ["/api/embed", "/api/embeddings", "/api/embeddings", "/api/embeddings"])

    async def test_probe_sets_dimension_then_rejects_changes(self):
        sizes = iter([3, 4])
        server = Server(lambda r, b: httpx.Response(200, json={"embeddings": [[0.1] * next(sizes)]}))
        provider = ollama(server, dimension=1024)

        self.assertEqual(await provider.probe(), 3)
        self.assertEqual(provider.dimension, 3)
        with self.assertRaisesRegex(ModelExecutionException, "returned 4 dimensions, expected 3"):
            await provider.embed_query("q")


class OpenAICompatibleProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_auth_and_index_order(self):
        def handler(request, body):
            data = [{"index": i, "embedding": vector(t)} for i, t in enumerate(body["input"])]
            return httpx.Response(200, json={"data": list(reversed(data))})

        server = Server(handler)
        provider = OpenAICompatibleEmbeddingProvider(
            model_name="bge-m3", base_url="https://gw.example.com", api_key="k",
            dimension=4, transport=server.transport,
        )

        vectors = await provider.embed_documents(["a", "bbb"])

        [request] = server.requests
        self.assertEqual(str(request.url), "https://gw.example.com/v1/embeddings")
        self.assertEqual(request.headers["authorization"], "Bearer k")
        self.assertEqual(json.loads(request.content)["encoding_format"], "float")
        self.assertEqual(vectors, [vector("a"), vector("bbb")])


class FactoryTests(unittest.TestCase):
    def build(self, **overrides):
        values = {
            "EMBEDDING_PROVIDER": "ollama", "EMBEDDING_MODEL": "qwen3-embedding:0.6b",
            "EMBEDDING_BASE_URL": None, "EMBEDDING_API_KEY": None,
            "EMBEDDING_VERIFY_SSL": None, "EMBEDDING_TRUST_ENV": None,
            "LLM_VERIFY_SSL": False, "LLM_TRUST_ENV": True,
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "VLLM_BASE_URL": "https://llm.example.com/v1", "VLLM_API_KEY": "llm-key",
            **overrides,
        }
        with patch.multiple(module.settings, **values):
            return build_embedding_provider()

    def test_ollama_inherits_llm_ssl_settings(self):
        provider = self.build(EMBEDDING_BASE_URL="https://remote-ollama:11434/v1", EMBEDDING_API_KEY="emb")
        self.assertIsInstance(provider, OllamaEmbeddingProvider)
        self.assertEqual(provider.base_url, "https://remote-ollama:11434")
        self.assertFalse(provider.verify_ssl)
        self.assertEqual(provider._headers["Authorization"], "Bearer emb")

    def test_embedding_settings_override_llm_settings(self):
        provider = self.build(EMBEDDING_VERIFY_SSL=True, EMBEDDING_TRUST_ENV=False)
        self.assertTrue(provider.verify_ssl)
        self.assertFalse(provider.trust_env)
        self.assertNotIn("Authorization", provider._headers)

    def test_openai_provider_defaults_to_llm_endpoint(self):
        provider = self.build(EMBEDDING_PROVIDER="openai")
        self.assertIsInstance(provider, OpenAICompatibleEmbeddingProvider)
        self.assertEqual(provider.base_url, "https://llm.example.com/v1")
        self.assertEqual(provider._headers["Authorization"], "Bearer llm-key")

    def test_unknown_provider(self):
        with self.assertRaisesRegex(RuntimeError, "ollama, openai, local"):
            self.build(EMBEDDING_PROVIDER="cohere")


if __name__ == "__main__":
    unittest.main()
