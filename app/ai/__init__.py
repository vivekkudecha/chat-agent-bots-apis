"""
AI Layer package.
Contains all AI-specific engines, workflows, and subsystems:
- agent: LangGraph workflows, intent router, state, tools
- rag: Retrieval, vector stores, chunking, text extraction
- llm: LLM provider abstraction, prompt builders, embeddings
- guardrails: Input/output/retrieval safety guardrail evaluation
"""

from app.ai.agent import (
    AgentRouter,
    ChatAgentGraph,
    ChatAgentState,
    RouteType,
    ToolRegistry,
)
from app.ai.guardrails import (
    BaseGuardrail,
    GuardrailAction,
    GuardrailEvaluation,
    GuardrailService,
    GuardrailStage,
)
from app.ai.llm import (
    BuiltPrompt,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    PromptBuilderService,
    get_llm_provider,
)
from app.ai.rag import (
    ChunkingService,
    DistinctSource,
    RetrievalResult,
    RetrievalService,
    RetrievedChunk,
    TextChunk,
    TextExtractionService,
    VectorStoreService,
)

from app.ai.memory import (
    ConversationSummarizer,
    ConversationSummary,
    MemoryContext,
    MemoryManager,
    TemporalContext,
    TemporalEngine,
)

__all__ = [
    # Agent
    "ChatAgentGraph",
    "ChatAgentState",
    "AgentRouter",
    "RouteType",
    "ToolRegistry",
    # Memory
    "MemoryManager",
    "MemoryContext",
    "ConversationSummary",
    "TemporalContext",
    "ConversationSummarizer",
    "TemporalEngine",
    # RAG
    "RetrievalService",
    "RetrievalResult",
    "RetrievedChunk",
    "DistinctSource",
    "VectorStoreService",
    "ChunkingService",
    "TextChunk",
    "TextExtractionService",
    # LLM
    "LLMProvider",
    "get_llm_provider",
    "LLMResponse",
    "LLMUsage",
    "PromptBuilderService",
    "BuiltPrompt",
    # Guardrails
    "GuardrailService",
    "GuardrailStage",
    "GuardrailAction",
    "GuardrailEvaluation",
    "BaseGuardrail",
]
