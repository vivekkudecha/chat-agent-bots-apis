from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict, TYPE_CHECKING
import uuid

from langgraph.graph import END, START, StateGraph

from app.ai.agent.router import AgentRouter
from app.ai.agent.state import RouteType
from app.ai.agent.tools import ToolRegistry
from app.config import settings
from app.ai.llm.provider import LLMProvider, LLMResponse

if TYPE_CHECKING:
    from app.models.bot import BotVersion
    from app.models.conversation import Message
    from app.ai.llm.prompt_builder import PromptBuilderService
    from app.ai.rag.retrieval import RetrievalResult, RetrievalService

logger = logging.getLogger(__name__)


class ChatAgentState(TypedDict, total=False):
    """
    LangGraph typed state for the conversational agent workflow.
    Tracks query, routing decisions, retrieval context, tools, and LLM output.
    """

    # Session identifiers
    user_id: uuid.UUID
    bot_id: uuid.UUID
    conversation_id: uuid.UUID

    # Input payload & configuration
    query: str
    bot_version: Any
    history: list[Any]
    has_kb: bool
    available_tools: list[dict[str, Any]]
    model_key: str
    temperature: float
    top_p: float
    max_tokens: int | None
    context_window: int | None
    db_session: Any

    # Routing determination
    route: str  # "direct", "rag", "tool"
    route_reason: str

    # Memory Context (Working memory buffer, episodic summary, temporal awareness)
    memory_context: Any | None

    # Retrieval results
    retrieval: Any | None
    distinct_sources: list[dict[str, Any]]

    # Tool calls & results
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    enable_web_search: bool

    # Output & usage
    response: LLMResponse | None
    response_content: str
    usage: dict[str, int]


class ChatAgentGraph:
    """
    LangGraph compiled workflow for orchestrating chat queries:
    START -> router_node -> [DIRECT -> generate_node,
                             RAG -> retrieve_node -> generate_node,
                             TOOL -> tool_node -> generate_node] -> END
    """

    def __init__(
        self,
        router: AgentRouter,
        retrieval_service: Any,
        prompt_builder: Any,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        guardrail_service: Any = None,
    ):
        self.router = router
        self.retrieval_service = retrieval_service
        self.prompt_builder = prompt_builder
        self.llm = llm_provider
        self.tool_registry = tool_registry
        self.guardrails = guardrail_service
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ChatAgentState)

        # Register nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("tool_dispatch", self._tool_node)
        workflow.add_node("generate", self._generate_node)

        # Graph Entry Point
        workflow.add_edge(START, "router")

        # Conditional Edge after Router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "direct": "generate",
                "rag": "retrieve",
                "tool": "tool_dispatch",
            },
        )

        # Edge from Retriever to Generator
        workflow.add_edge("retrieve", "generate")

        # Edge from Tool Dispatch to Generator
        workflow.add_edge("tool_dispatch", "generate")

        # Completion edge
        workflow.add_edge("generate", END)

        return workflow.compile()

    # -------------------------------------------------------------
    # Node Callables
    # -------------------------------------------------------------

    async def _router_node(self, state: ChatAgentState) -> dict[str, Any]:
        """
        Evaluates intent: fast-path for small talk vs zero-shot LLM classification.
        """
        route = await self.router.route(
            message=state["query"],
            has_kb=state.get("has_kb", False),
            available_tools=state.get("available_tools", []),
            model=state.get("model_key"),
        )
        logger.info("LangGraph router decided: route=%s", route.value)
        return {
            "route": route.value,
            "route_reason": f"Routed to {route.value}",
        }

    def _route_decision(self, state: ChatAgentState) -> Literal["direct", "rag", "tool"]:
        """
        Conditional branch selector for LangGraph.
        """
        route_val = state.get("route", RouteType.DIRECT.value)
        if route_val == RouteType.RAG.value:
            return "rag"
        if route_val == RouteType.TOOL.value:
            return "tool"
        return "direct"

    async def _retrieve_node(self, state: ChatAgentState) -> dict[str, Any]:
        """
        Queries vector store with similarity score threshold filtering and distinct sources.
        """
        db = state.get("db_session")
        if not db or not state.get("has_kb"):
            return {"retrieval": None, "distinct_sources": []}

        try:
            context_window = state.get("context_window")
            max_gen_tokens = state.get("max_tokens") or 512
            rag_budget = None
            if context_window:
                rag_budget = max(250, min(1500, int((context_window - max_gen_tokens) * 0.40)))

            retrieval = await self.retrieval_service.retrieve_for_bot(
                db,
                user_id=state["user_id"],
                bot_id=state["bot_id"],
                query=state["query"],
                top_k=settings.DEFAULT_TOP_K,
                score_threshold=settings.DEFAULT_SCORE_THRESHOLD,
                context_budget=rag_budget,
            )

            safe_retrieval = retrieval
            if retrieval and retrieval.chunks and self.guardrails:
                from app.ai.guardrails.base import GuardrailStage

                safe_chunks = []
                for chunk in retrieval.chunks:
                    evaluation = await self.guardrails.evaluate(
                        db,
                        bot_id=state["bot_id"],
                        user_id=state["user_id"],
                        conversation_id=state["conversation_id"],
                        stage=GuardrailStage.RETRIEVAL,
                        text=chunk.text,
                    )
                    chunk.text = evaluation.final_text
                    safe_chunks.append(chunk)

                safe_retrieval.chunks = safe_chunks

            distinct_sources = (
                safe_retrieval.get_distinct_sources() if safe_retrieval else []
            )

            return {
                "retrieval": safe_retrieval,
                "distinct_sources": distinct_sources,
            }
        except Exception as exc:
            logger.exception("Error in LangGraph retrieve_node: %s", exc)
            return {"retrieval": None, "distinct_sources": []}

    async def _tool_node(self, state: ChatAgentState) -> dict[str, Any]:
        """
        Prepares tools or executes tool calls for tool calling flows.
        """
        tool_results = list(state.get("tool_results") or [])
        tool_calls = list(state.get("tool_calls") or [])

        available_tools = state.get("available_tools") or []
        has_web_search = any(
            t.get("function", {}).get("name") == "web_search"
            for t in available_tools
        )

        # If routed to TOOL and web search is enabled, execute web search for query
        if has_web_search and not tool_results:
            query = state.get("query", "")
            search_result = await self.tool_registry.execute_tool(
                "web_search",
                {"query": query},
                context={
                    "user_id": state.get("user_id"),
                    "bot_id": state.get("bot_id"),
                    "conversation_id": state.get("conversation_id"),
                },
            )
            tool_calls.append(
                {
                    "name": "web_search",
                    "arguments": {"query": query},
                }
            )
            tool_results.append(
                {
                    "name": "web_search",
                    "arguments": {"query": query},
                    "result": search_result,
                }
            )

        return {
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    async def _generate_node(self, state: ChatAgentState) -> dict[str, Any]:
        """
        Constructs context-aware prompt and calls LLM, executing any requested tool calls.
        """
        retrieval = state.get("retrieval")
        available_tools = state.get("available_tools") or []
        enable_web_search = state.get("enable_web_search", False)
        route = state.get("route", RouteType.DIRECT.value)

        # Pass tools to model if routed to TOOL, or web search is enabled, or tools exist
        llm_tools = None
        if route == RouteType.TOOL.value or enable_web_search or available_tools:
            llm_tools = available_tools or None

        tool_results = list(state.get("tool_results") or [])
        tool_calls = list(state.get("tool_calls") or [])

        prompt = await self.prompt_builder.build_async(
            bot_version=state["bot_version"],
            user_message=state["query"],
            history=state.get("history", []),
            retrieval=retrieval,
            memory_context=state.get("memory_context"),
            context_window=state.get("context_window"),
            max_generation_tokens=state.get("max_tokens"),
            tools=llm_tools,
            tool_results=tool_results or None,
        )

        response = await self.llm.chat(
            messages=prompt.messages,
            model=state["model_key"],
            temperature=state.get("temperature", 0.7),
            top_p=state.get("top_p", 1.0),
            max_tokens=state.get("max_tokens"),
            tools=llm_tools,
        )

        # If LLM requested dynamic function calling via tool_calls
        dynamic_calls = getattr(response, "tool_calls", None)
        if dynamic_calls:
            tool_calls.extend(dynamic_calls)
            for tc in dynamic_calls:
                fn = tc.get("function", {})
                t_name = fn.get("name", "")
                t_args = fn.get("arguments", {})
                exec_result = await self.tool_registry.execute_tool(
                    t_name,
                    t_args,
                    context={
                        "user_id": state.get("user_id"),
                        "bot_id": state.get("bot_id"),
                        "conversation_id": state.get("conversation_id"),
                    },
                )
                tool_results.append(
                    {
                        "name": t_name,
                        "arguments": t_args,
                        "result": exec_result,
                    }
                )

            # Follow-up generation with tool results included in prompt context
            followup_prompt = await self.prompt_builder.build_async(
                bot_version=state["bot_version"],
                user_message=state["query"],
                history=state.get("history", []),
                retrieval=retrieval,
                memory_context=state.get("memory_context"),
                context_window=state.get("context_window"),
                max_generation_tokens=state.get("max_tokens"),
                tools=None,
                tool_results=tool_results,
            )

            response = await self.llm.chat(
                messages=followup_prompt.messages,
                model=state["model_key"],
                temperature=state.get("temperature", 0.7),
                top_p=state.get("top_p", 1.0),
                max_tokens=state.get("max_tokens"),
            )

        # Merge any web search results into distinct_sources
        distinct_sources = list(state.get("distinct_sources") or [])
        for tr in tool_results:
            if tr.get("name") == "web_search" and isinstance(tr.get("result"), dict):
                for item in tr["result"].get("results", []):
                    distinct_sources.append(
                        {
                            "document_id": uuid.uuid4(),
                            "knowledge_base_id": uuid.UUID(int=0),
                            "file_name": item.get("title") or "Web Search",
                            "score": 1.0,
                            "content_preview": item.get("snippet"),
                            "metadata": {
                                "url": item.get("url"),
                                "source_type": "web_search",
                            },
                        }
                    )

        return {
            "response": response,
            "response_content": response.content,
            "usage": response.usage,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "distinct_sources": distinct_sources,
        }

    async def run(self, initial_state: ChatAgentState) -> ChatAgentState:
        """
        Executes the compiled LangGraph workflow.
        """
        result = await self.app.ainvoke(initial_state)
        return result
