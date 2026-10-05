# Backward compatibility re-export from new app.ai layer
from app.ai.rag.vector_store import (
    VectorStoreService,
)

__all__ = [
    "VectorStoreService",
]