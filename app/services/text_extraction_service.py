# Backward compatibility re-export from new app.ai layer
from app.ai.rag.text_extraction import (
    ExtractionResult,
    TextExtractionService,
)

__all__ = [
    "ExtractionResult",
    "TextExtractionService",
]