from __future__ import annotations

import logging
import re
from typing import Any, Literal, TypedDict, TYPE_CHECKING
import uuid

from langgraph.graph import END, START, StateGraph

from app.ai.agent.router import AgentRouter
from app.ai.agent.state import RouteType
from app.ai.agent.tools import ToolRegistry
from app.config import settings
from app.ai.llm.provider import LLMProvider, LLMResponse, OpenAICompatibleProvider

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
    bot_instruction: str

    # Routing determination
    route: str  # "direct", "memory", "rag", "tool"
    route_reason: str
    resolved_query: str
    selected_tool: str | None
    tool_args: dict[str, Any]

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

        # Register nodes:
        # Context Builder -> Semantic Supervisor -> RAG / TOOL / DIRECT -> RESPONSE
        workflow.add_node("context_builder", self._context_builder_node)
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("tool_dispatch", self._tool_node)
        workflow.add_node("generate", self._generate_node)

        # Entry Point: User Query -> Context Builder -> Semantic Supervisor
        workflow.add_edge(START, "context_builder")
        workflow.add_edge("context_builder", "supervisor")

        # Semantic Supervisor Decision:
        workflow.add_conditional_edges(
            "supervisor",
            self._route_decision,
            {
                "direct": "generate",
                "rag": "retrieve",
                "tool": "tool_dispatch",
            },
        )

        # KB Relevance Gate (after RAG retrieval):
        # - Relevant data in KB -> generate response from RAG
        # - No relevant data in KB -> dynamic fallback to tool_dispatch (if tools enabled) or direct
        workflow.add_conditional_edges(
            "retrieve",
            self._kb_relevance_gate,
            {
                "generate": "generate",
                "tool": "tool_dispatch",
            },
        )

        # Tool Relevance & Execution -> generate response
        workflow.add_edge("tool_dispatch", "generate")

        # Completion edge -> RESPONSE
        workflow.add_edge("generate", END)

        return workflow.compile()

    # -------------------------------------------------------------
    # Node Callables
    # -------------------------------------------------------------

    async def _context_builder_node(self, state: ChatAgentState) -> dict[str, Any]:
        """
        Stage 1: Context Builder
        Prepares working context, extracts bot instruction, formats active tools,
        and resolves query parameters for the Semantic Supervisor.
        """
        bot_version = state.get("bot_version")
        bot_instruction = getattr(bot_version, "system_instruction", "") or ""
        return {
            "resolved_query": state.get("query", "").strip(),
            "bot_instruction": bot_instruction,
        }

    async def _supervisor_node(self, state: ChatAgentState) -> dict[str, Any]:
        """
        Stage 2: Semantic Supervisor
        Dynamically classifies query intent into DIRECT, RAG, or TOOL
        using semantic LLM reasoning without regexes or fixed patterns.
        """
        decision = await self.router.decide(
            message=state["query"],
            has_kb=state.get("has_kb", False),
            available_tools=state.get("available_tools", []),
            model=state.get("model_key", ""),
            history=state.get("history", []),
            bot_instruction=state.get("bot_instruction", ""),
            memory_context=state.get("memory_context"),
        )
        logger.info(
            "Semantic Supervisor decided: route=%s reason=%s query=%s tool=%s",
            decision.route.value,
            decision.reason,
            decision.query,
            decision.tool_name,
        )
        return {
            "route": decision.route.value,
            "route_reason": decision.reason,
            "resolved_query": decision.query,
            "selected_tool": decision.tool_name,
            "tool_args": decision.tool_args or {},
        }

    def _route_decision(self, state: ChatAgentState) -> Literal["direct", "rag", "tool"]:
        """
        Conditional branch selector after Semantic Supervisor.
        """
        route_val = state.get("route", RouteType.DIRECT.value)
        if route_val == RouteType.RAG.value:
            return "rag"
        if route_val == RouteType.TOOL.value:
            return "tool"
        return "direct"

    def _kb_relevance_gate(self, state: ChatAgentState) -> Literal["generate", "tool"]:
        """
        Stage 3a: KB Relevance Verification Gate
        - If RAG retrieved relevant data on the system, proceed immediately to generate (answering from RAG).
        - If RAG retrieved NO relevant data on the system, check if web search is enabled:
          if yes, dynamically fall back to TOOL; otherwise proceed to DIRECT response.
        """
        retrieval = state.get("retrieval")
        has_chunks = bool(
            retrieval and getattr(retrieval, "chunks", None) and len(retrieval.chunks) > 0
        )

        # If data is present on our system, ALWAYS return from RAG!
        if has_chunks:
            logger.info("KB relevance verified (data in system) -> answering from RAG")
            return "generate"

        # If RAG has NO data on our system, check if web search is enabled
        enable_web_search = state.get("enable_web_search", False)
        available_tools = state.get("available_tools") or []
        has_web_search = any(
            t.get("function", {}).get("name") == "web_search"
            for t in available_tools
        )

        if enable_web_search and has_web_search:
            logger.info("KB relevance check: no data in system -> dynamic fallback to web search")
            return "tool"

        logger.info("KB relevance check: no data in system and web search disabled -> direct response")
        return "generate"

    _post_retrieve_gate = _kb_relevance_gate

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

            retrieval_query = state.get("resolved_query") or state["query"]
            retrieval = await self.retrieval_service.retrieve_for_bot(
                db,
                user_id=state["user_id"],
                bot_id=state["bot_id"],
                query=retrieval_query,
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

            has_relevant = bool(
                safe_retrieval and safe_retrieval.chunks and len(safe_retrieval.chunks) > 0
            )
            route = RouteType.RAG.value if has_relevant else RouteType.DIRECT.value

            return {
                "route": route,
                "retrieval": safe_retrieval,
                "distinct_sources": distinct_sources,
            }
        except Exception as exc:
            logger.exception("Error in LangGraph retrieve_node: %s", exc)
            return {"route": RouteType.DIRECT.value, "retrieval": None, "distinct_sources": []}

    async def _tool_node(self, state: ChatAgentState) -> dict[str, Any]:
        """
        Executes web search or external tool when RAG did not have the data or tool was requested.
        """
        tool_results = list(state.get("tool_results") or [])
        tool_calls = list(state.get("tool_calls") or [])

        # If tools not yet executed, execute web search with query
        if not tool_results:
            selected_tool = state.get("selected_tool") or "web_search"
            tool_args = dict(state.get("tool_args") or {})
            if not tool_args.get("query"):
                tool_args["query"] = state.get("resolved_query") or state.get("query", "")
            exec_result = await self.tool_registry.execute_tool(
                selected_tool,
                tool_args,
                context={
                    "user_id": state.get("user_id"),
                    "bot_id": state.get("bot_id"),
                    "conversation_id": state.get("conversation_id"),
                },
            )
            tool_calls.append({"name": selected_tool, "arguments": tool_args})
            tool_results.append(
                {
                    "name": selected_tool,
                    "arguments": tool_args,
                    "result": exec_result,
                }
            )

        return {
            "route": RouteType.TOOL.value,
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

        tool_results = list(state.get("tool_results") or [])
        tool_calls = list(state.get("tool_calls") or [])

        # CRITICAL RULE: If RAG has data on our system (route == RAG), NEVER pass tools to LLM!
        # The LLM must answer strictly from RAG. External web search is completely disabled for this turn.
        if route == RouteType.RAG.value:
            tool_calls = []
            tool_results = []
            llm_tools = None
        elif not tool_results and route == RouteType.TOOL.value:
            llm_tools = available_tools or None
        else:
            llm_tools = None

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

            # Follow-up generation with tool results included in prompt context and tools=None
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
                tools=None,
            )

        # Safety net: If response.content still looks like raw tool JSON, recover
        if response.content:
            leaked_calls = OpenAICompatibleProvider._extract_tool_calls_from_content(
                response.content, available_tools
            )
            if leaked_calls:
                logger.warning("LLM response contained unparsed tool JSON; recovering...")
                for lc in leaked_calls:
                    fn = lc.get("function", {})
                    t_name = fn.get("name", "")
                    t_args = fn.get("arguments", {})
                    if not any(tr.get("name") == t_name for tr in tool_results):
                        exec_res = await self.tool_registry.execute_tool(
                            t_name,
                            t_args,
                            context={
                                "user_id": state.get("user_id"),
                                "bot_id": state.get("bot_id"),
                                "conversation_id": state.get("conversation_id"),
                            },
                        )
                        tool_results.append({"name": t_name, "arguments": t_args, "result": exec_res})

                recovery_prompt = await self.prompt_builder.build_async(
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
                    messages=recovery_prompt.messages,
                    model=state["model_key"],
                    temperature=0.3,
                    top_p=1.0,
                    max_tokens=state.get("max_tokens"),
                    tools=None,
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
