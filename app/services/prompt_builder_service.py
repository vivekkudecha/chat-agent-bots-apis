# Backward compatibility re-export from new app.ai layer
from app.ai.llm.prompt_builder import (
    BuiltPrompt,
    PromptBuildResult,
    PromptBuilderService,
)

__all__ = [
    "PromptBuilderService",
    "PromptBuildResult",
    "BuiltPrompt",
]