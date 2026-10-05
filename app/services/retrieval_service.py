# Backward compatibility re-export from new app.ai layer
from app.ai.rag.retrieval import (
    DistinctSource,
    RetrievalResult,
    RetrievalService,
    RetrievedChunk,
)

__all__ = [
    "RetrievedChunk",
    "DistinctSource",
    "RetrievalResult",
    "RetrievalService",
]