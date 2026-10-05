# Backward compatibility re-export from new app.ai layer
from app.ai.rag.chunking import (
    ChunkingService,
    TextChunk,
)

__all__ = [
    "ChunkingService",
    "TextChunk",
]