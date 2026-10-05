import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from app.config import settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Shared Qdrant Client
# ---------------------------------------------------------

qdrant_client = AsyncQdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY or None,
    timeout=settings.QDRANT_TIMEOUT_SECONDS,
)


# ---------------------------------------------------------
# Client Getter
# ---------------------------------------------------------

def get_qdrant_client() -> AsyncQdrantClient:
    return qdrant_client


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

async def check_qdrant_health() -> bool:

    try:
        await qdrant_client.get_collections()
        return True

    except Exception:
        logger.exception(
            "Qdrant health check failed"
        )
        return False


# ---------------------------------------------------------
# Collection Exists
# ---------------------------------------------------------

async def collection_exists(
    collection_name: str,
) -> bool:

    return await qdrant_client.collection_exists(
        collection_name=collection_name,
    )


# ---------------------------------------------------------
# Ensure Collection
# ---------------------------------------------------------

async def ensure_collection(
    *,
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE,
) -> None:

    exists = await collection_exists(
        collection_name
    )

    if not exists:
        logger.info(
            "Creating Qdrant collection: %s",
            collection_name,
        )

        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )

    # Ensure payload indexes for high-volume filtering & keyword search
    try:
        for field in ("user_id", "knowledge_base_id", "document_id"):
            await qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        # Dynamically resolve tokenizer type (default: MULTILINGUAL)
        tok_setting = getattr(settings, "QDRANT_TEXT_TOKENIZER", "multilingual").lower().strip()
        if tok_setting == "multilingual":
            tok_type = TokenizerType.MULTILINGUAL
        elif tok_setting == "whitespace":
            tok_type = TokenizerType.WHITESPACE
        elif tok_setting == "prefix":
            tok_type = TokenizerType.PREFIX
        else:
            tok_type = TokenizerType.WORD

        await qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=tok_type,
                lowercase=True,
            ),
        )
    except Exception as exc:
        logger.warning(
            "Failed or partial payload index creation on %s: %s",
            collection_name,
            exc,
        )


# ---------------------------------------------------------
# Delete Collection
# ---------------------------------------------------------

async def delete_collection(
    collection_name: str,
) -> None:

    exists = await collection_exists(
        collection_name
    )

    if not exists:
        return

    await qdrant_client.delete_collection(
        collection_name=collection_name,
    )


# ---------------------------------------------------------
# Close Client
# ---------------------------------------------------------

async def close_qdrant() -> None:

    await qdrant_client.close()