"""Check the configured embedding endpoint (URL, auth, SSL, model, dimension).

Usage:
    python -m scripts.check_embeddings
    EMBEDDING_PROVIDER=openai python -m scripts.check_embeddings

Exits non-zero when the endpoint fails or the dimension differs from
EMBEDDING_DIMENSION / the existing Qdrant collection.
"""

import asyncio
import sys
import time

from app.config import settings
from app.ai.llm.embeddings import build_embedding_provider


async def main() -> int:
    provider = build_embedding_provider()
    print("Config:", provider.describe())

    try:
        started = time.perf_counter()
        dimension = await provider.probe()
        probe_ms = (time.perf_counter() - started) * 1000

        texts = [f"Sample passage number {i} about leave policy." for i in range(16)]
        started = time.perf_counter()
        vectors = await provider.embed_documents(texts)
        batch_s = time.perf_counter() - started
        query = await provider.embed_query("What is the leave policy?")
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1
    finally:
        close = getattr(provider, "aclose", None)
        if close:
            await close()

    print(f"OK: dimension={dimension} probe={probe_ms:.0f}ms "
          f"batch={len(vectors)} texts in {batch_s:.2f}s query_dim={len(query)}")

    status = 0
    if dimension != settings.EMBEDDING_DIMENSION:
        print(f"WARNING: EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION} but model returns {dimension}")
        status = 2

    try:
        from app.ai.rag.vector_store import hybrid_collection_name
        from app.integrations.qdrant import DENSE_VECTOR, qdrant_client

        name = hybrid_collection_name()
        if await qdrant_client.collection_exists(name):
            info = await qdrant_client.get_collection(name)
            size = info.config.params.vectors[DENSE_VECTOR].size
            match = "matches" if size == dimension else "DOES NOT MATCH (reindex needed)"
            print(f"Qdrant {name}: {size}-dim vectors, {info.points_count} points - {match}")
            status = status or (0 if size == dimension else 3)
        else:
            print(f"Qdrant {name}: not created yet")
    except Exception as exc:
        print(f"Qdrant check skipped: {exc}")

    return status


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
