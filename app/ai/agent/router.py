from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any

from app.ai.agent.state import RouteType
from app.ai.llm.provider import LLMProvider, get_llm_provider
from app.ai.agent.conversation_context import ConversationContextBuilder

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """
    Semantic routing decision produced by the supervisor.

    The supervisor decides which capability should handle the request:
    DIRECT, RAG, or TOOL.
    """

    route: RouteType
    reason: str
    query: str

    confidence: float = 0.0

    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)

    # NONE    -> current request does not depend on prior conversation
    # REUSE   -> answer can be derived from prior conversation/tool evidence
    # REFRESH -> prior context helps, but fresh external information is required
    context_mode: str = "NONE"


class SemanticSupervisor:
    """
    Dynamic semantic supervisor for multi-agent routing.

    Routing is based on:
    - User intent
    - Conversation context
    - Bot instructions
    - Available private knowledge
    - Available tools/capabilities

    No domain-specific keyword or regex routing is used.
    """

    SUPERVISOR_SYSTEM_PROMPT = """
You are the Semantic Routing Supervisor of an enterprise AI assistant.

Your responsibility is NOT to answer the user's request.

Your responsibility is to determine which available execution capability
is required to answer the request correctly and reliably.

Route based on the semantic requirements of the request, conversation
context, available private knowledge, and available tools.

--------------------------------------------------
CORE PRINCIPLE
--------------------------------------------------

Choose the execution path based on WHERE the required information or
capability must come from.

DIRECT:
The base language model already has everything required.

RAG:
The answer depends on private/configured knowledge.

TOOL:
The answer depends on external information, external state, or an
external capability.

Do not route based on keywords.

--------------------------------------------------
AVAILABLE EXECUTION PATHS
--------------------------------------------------

DIRECT

Use DIRECT when the request can be answered reliably using:
- the base model's stable knowledge
- reasoning
- conversation context already available
- writing or transformation
- coding or explanation
- summarization of information already provided

DIRECT is appropriate only when retrieving additional information is
not necessary for correctness.

Examples:
- explaining a programming concept
- writing or rewriting text
- reasoning about provided information
- generating code
- explaining stable general knowledge
- summarizing conversation content

Do NOT use DIRECT when correctness depends on information that must be
retrieved from an external or private source.


RAG

Use RAG when the answer depends on information likely contained in the
configured private knowledge sources.

A knowledge base being available does NOT automatically justify RAG.

Use RAG when the user's intent is semantically related to organization,
documents, policies, uploaded files, private records, or other knowledge
represented by the configured knowledge source.

Examples:
- questions about uploaded documents
- company policies
- internal procedures
- private organizational information
- knowledge specific to the configured knowledge base

Do not use RAG for unrelated general questions.


TOOL

Use TOOL when fulfilling the request correctly requires an available
external capability.

This includes situations where the answer depends on:

- information whose current value or state may differ from the base
  model's stored knowledge
- information that must be retrieved from the internet
- external search
- external systems or databases
- external APIs
- performing an action
- obtaining information that is not already present in the conversation
  or private knowledge source

The important distinction is not whether the model knows something
about the subject.

The question is:

"Can the request be answered reliably without accessing the external
capability?"

If NO, select TOOL.

When an appropriate tool is available, do not substitute potentially
stale model knowledge for information that the tool can retrieve.

If TOOL is selected:
1. Select exactly one available tool.
2. Select the tool whose capability best matches the user's intent.
3. Generate arguments strictly according to that tool's schema.
4. Preserve the user's actual intent in the tool arguments.
5. Never invent a tool.
6. Never invent unsupported parameters.

--------------------------------------------------
CURRENT / EXTERNAL STATE
--------------------------------------------------

Some requests depend on information that changes over time or exists
outside the model.

Such requests require an external capability when a suitable tool is
available.

This determination must be semantic.

Do NOT maintain or rely on keyword lists such as:
"latest", "today", "news", "current", etc.

Instead determine whether answering the request correctly requires
observing external state at execution time.

For example, conceptually:

A request asking how a technology works may be answerable using DIRECT.

A request asking what version of that technology is currently released
requires external information if an appropriate tool exists.

A request asking about the historical role of a company may be DIRECT.

A request asking what is happening with that company right now may
require an external capability.

These are semantic distinctions, not keyword rules.

--------------------------------------------------
ROUTING PRIORITY
--------------------------------------------------

Determine the source required for a correct answer in this order:

1. Does the request require an external capability or external state?
   If yes and an appropriate tool exists:
   -> TOOL

2. Does the request depend on configured private knowledge?
   If yes:
   -> RAG

3. Can the request be reliably completed using the model and available
   conversation context alone?
   If yes:
   -> DIRECT

4. If the required capability does not exist:
   -> DIRECT

   The downstream assistant may then explain the limitation.

Do NOT choose DIRECT merely because the model might know something
about the topic.

Choose DIRECT only when external/private retrieval is unnecessary for
answering correctly.

--------------------------------------------------
CONVERSATION CONTEXT
--------------------------------------------------

Use conversation context to resolve references and follow-up requests.

Examples of references include:
- it
- that
- they
- he
- she
- this
- those
- the previous one
- the same company
- what about them

Rewrite contextual requests into a standalone query when necessary.

Example:

Conversation:
User: Tell me about the employee leave rules.
Assistant: ...

Current request:
What about fathers?

Standalone query:
What leave benefits are available to fathers?

Preserve the user's original meaning.

Do not introduce requirements the user did not express.

--------------------------------------------------
TOOL SELECTION
--------------------------------------------------

Available tools and their schemas will be provided separately.

Evaluate tools semantically by:
- their descriptions
- their supported operations
- their input schema

Never select a tool merely because it exists.

Never select a tool whose capability does not satisfy the request.

If multiple tools could theoretically help, select the single tool
that most directly satisfies the user's immediate request.

--------------------------------------------------
UNCERTAINTY
--------------------------------------------------

If uncertain between RAG and DIRECT:

Choose RAG only when the answer is reasonably expected to depend on
the configured private knowledge.

Otherwise choose DIRECT.

If uncertain between TOOL and DIRECT:

Ask whether the answer would remain reliable if no external information
were retrieved.

If reliability depends on external state and a suitable tool exists:
choose TOOL.

If external retrieval would merely provide optional supporting
information and is not necessary:
choose DIRECT.

This distinction is based on information dependency, not keywords.

--------------------------------------------------
CONFIDENCE
--------------------------------------------------

Return a confidence value between 0.0 and 1.0.

Confidence indicates certainty that the selected execution path is
appropriate.

Do not fabricate unnecessary precision.

Prefer values such as:
0.5
0.7
0.8
0.9
1.0

--------------------------------------------------
FOLLOW-UP / CONVERSATION CONTEXT POLICY
--------------------------------------------------

Previous turns are part of the current request's information context.

There are THREE context modes:

NONE
The current request is independent of previous turns.

REUSE
The current request is a follow-up, transformation, filtering, sorting,
comparison, clarification, or continuation that can be answered from
information already present in the previous conversation/tool evidence.

REFRESH
The previous conversation establishes the subject, but the current request
requires NEW external/current information.

Examples:

Previous user:
Tell me the movies that will be released this week in Mumbai.

Previous tool:
Returned movie-release evidence.

Current user:
Give me only Hindi movie names.

Correct:
- route = DIRECT
- context_mode = REUSE
- query = "From the previously retrieved Mumbai movie-release list for this
  week, return only the Hindi movie names."

Current user:
What about next week?

Correct:
- route = TOOL
- context_mode = REFRESH
- query = "Which movies will be released in Mumbai next week?"

Current user:
Sort those by release date.

Correct:
- route = DIRECT
- context_mode = REUSE

Current user:
Which of those are currently playing in PVR cinemas?

Correct:
- route = TOOL
- context_mode = REFRESH

CRITICAL:

If the current request can be answered by transforming/filtering/
summarizing previous retrieved evidence, DO NOT perform a new web search.

The previous assistant answer is NOT the authoritative source when it was
generated from a tool. Prefer the preserved PREVIOUS EXTERNAL TOOL EVIDENCE.

Do not invent missing facts from previous context. If previous evidence does
not contain enough information for the requested transformation, choose
REFRESH + TOOL when a suitable tool exists.

When context_mode = REUSE:
- route should normally be DIRECT
- the standalone query must explicitly preserve the previous subject,
  filters, location, date range, and other constraints
- do not discard constraints merely because the current message is short

When context_mode = REFRESH:
- preserve relevant context in the new tool query
- search only for the NEW information required by the current request

--------------------------------------------------
OUTPUT
--------------------------------------------------

Return ONLY valid JSON.

Do not include markdown.
Do not include explanations outside JSON.
Do not include code fences.

Required format:

{
    "route": "DIRECT" | "RAG" | "TOOL",
    "confidence": 0.0,
    "reason": "Short explanation of why this execution path is required",
    "query": "Standalone contextualized version of the user's request",
    "tool_name": null,
    "tool_args": {}
}

When route is DIRECT:
{
    "tool_name": null,
    "tool_args": {}
}

When route is RAG:
{
    "tool_name": null,
    "tool_args": {}
}

When route is TOOL:
{
    "tool_name": "<exact available tool name>",
    "tool_args": {
        "...": "arguments exactly matching the selected tool schema"
    }
}
"""


    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm or get_llm_provider()


    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    async def decide(
        self,
        *,
        message: str,
        has_kb: bool,
        available_tools: list[dict[str, Any]] | None = None,
        model: str = "",
        history: list[Any] | None = None,
        bot_instruction: str = "",
        memory_context: Any | None = None,
        conversation_context: dict[str, Any] | None = None,

        # Recommended:
        # Pass descriptions of attached KBs when available.
        knowledge_sources: list[dict[str, Any]] | None = None,
    ) -> RoutingDecision:

        cleaned = message.strip()

        if not cleaned:
            return RoutingDecision(
                route=RouteType.DIRECT,
                reason="Empty message",
                query="",
                confidence=1.0,
                context_mode="NONE",
            )

        tools = available_tools or []
        knowledge_sources = knowledge_sources or []

        # Nothing external/private is available.
        # No supervisor call is required.
        if not has_kb and not tools:
            return RoutingDecision(
                route=RouteType.DIRECT,
                reason="No private knowledge or external tools are configured",
                query=cleaned,
                confidence=1.0,
            )

        try:
            context = self._build_supervisor_context(
                message=cleaned,
                has_kb=has_kb,
                tools=tools,
                history=history,
                memory_context=memory_context,
                conversation_context=conversation_context,
                bot_instruction=bot_instruction,
                knowledge_sources=knowledge_sources,
            )

            target_model = self._resolve_model(model)

            response = await self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": self.SUPERVISOR_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": context,
                    },
                ],
                model=target_model,
                temperature=0.0,
                max_tokens=1024,
                tools=None,
            )

            raw_output = (response.content or "").strip()
            if not raw_output and getattr(response, "tool_calls", None):
                tc = response.tool_calls[0]
                raw_output = json.dumps(tc.get("function", {}).get("arguments", {}))

            parsed = self._parse_supervisor_response(
                raw_output
            )

            return self._build_decision(
                parsed=parsed,
                original_query=cleaned,
                has_kb=has_kb,
                tools=tools,
            )

        except Exception as exc:

            logger.warning(
                "Semantic supervisor failed: %s",
                exc,
                exc_info=True,
            )

            # IMPORTANT:
            # Never blindly fall back to RAG or TOOL.
            #
            # DIRECT is the safest fallback because it does not
            # accidentally query private data or execute external tools.

            return RoutingDecision(
                route=RouteType.DIRECT,
                reason="Supervisor unavailable; using safe direct fallback",
                query=cleaned,
                confidence=0.0,
                context_mode="NONE",
            )


    async def route(
        self,
        *,
        message: str,
        has_kb: bool,
        available_tools: list[dict[str, Any]] | None = None,
        model: str = "",
        history: list[Any] | None = None,
        bot_instruction: str = "",
        memory_context: Any | None = None,
        conversation_context: dict[str, Any] | None = None,
        knowledge_sources: list[dict[str, Any]] | None = None,
    ) -> RouteType:

        decision = await self.decide(
            message=message,
            has_kb=has_kb,
            available_tools=available_tools,
            model=model,
            history=history,
            bot_instruction=bot_instruction,
            memory_context=memory_context,
            conversation_context=conversation_context,
            knowledge_sources=knowledge_sources,
        )

        return decision.route


    # ---------------------------------------------------------
    # Context Builder
    # ---------------------------------------------------------

    def _build_supervisor_context(
        self,
        *,
        message: str,
        has_kb: bool,
        tools: list[dict[str, Any]],
        history: list[Any] | None,
        memory_context: Any | None,
        conversation_context: dict[str, Any] | None,
        bot_instruction: str,
        knowledge_sources: list[dict[str, Any]],
    ) -> str:

        sections: list[str] = []


        # -----------------------------------------------------
        # Bot scope
        # -----------------------------------------------------

        sections.append(
            "BOT SCOPE / INSTRUCTIONS:\n"
            + (
                bot_instruction[:4000]
                if bot_instruction
                else "General enterprise assistant"
            )
        )


        # -----------------------------------------------------
        # Knowledge sources
        # -----------------------------------------------------

        sections.append(
            self._format_knowledge_sources(
                has_kb=has_kb,
                knowledge_sources=knowledge_sources,
            )
        )


        # -----------------------------------------------------
        # Tools
        # -----------------------------------------------------

        sections.append(
            self._format_tools(tools)
        )


        # -----------------------------------------------------
        # Memory summary
        # -----------------------------------------------------

        memory_summary = self._extract_memory_summary(
            memory_context
        )

        if memory_summary:
            sections.append(
                "CONVERSATION MEMORY SUMMARY:\n"
                + memory_summary
            )


        # -----------------------------------------------------
        # Recent conversation + previous tool evidence
        # -----------------------------------------------------

        if conversation_context is None:
            conversation_context = ConversationContextBuilder().build(
                history
            )

        context_text = str(
            conversation_context.get("text") or ""
        ).strip()

        if context_text:
            sections.append(context_text)


        # -----------------------------------------------------
        # Current request
        # -----------------------------------------------------

        sections.append(
            "CURRENT USER REQUEST:\n"
            + message
        )


        sections.append(
            """
ROUTING TASK:

Determine which configured capability should handle the CURRENT USER REQUEST.

Consider:
- semantic intent
- bot scope
- private knowledge availability
- knowledge source descriptions
- conversation context
- available tools
- tool descriptions
- tool parameter schemas

Return only the required JSON object.
""".strip()
        )

        return "\n\n".join(sections)


    # ---------------------------------------------------------
    # Knowledge Sources
    # ---------------------------------------------------------

    def _format_knowledge_sources(
        self,
        *,
        has_kb: bool,
        knowledge_sources: list[dict[str, Any]],
    ) -> str:

        if not has_kb:
            return "PRIVATE KNOWLEDGE SOURCES:\nNone"

        if not knowledge_sources:
            return (
                "PRIVATE KNOWLEDGE SOURCES:\n"
                "A private knowledge base is attached, but detailed "
                "source descriptions are unavailable."
            )

        formatted: list[str] = []

        for index, source in enumerate(
            knowledge_sources,
            start=1,
        ):
            name = (
                source.get("name")
                or source.get("title")
                or source.get("id")
                or f"Knowledge Source {index}"
            )

            description = (
                source.get("description")
                or source.get("summary")
                or "No description provided"
            )

            formatted.append(
                f"{index}. {name}\n"
                f"   Description: {description}"
            )

        return (
            "PRIVATE KNOWLEDGE SOURCES:\n"
            + "\n".join(formatted)
        )


    # ---------------------------------------------------------
    # Tools
    # ---------------------------------------------------------

    def _format_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> str:

        if not tools:
            return "AVAILABLE TOOLS:\nNone"

        formatted: list[str] = []

        for index, tool in enumerate(tools, start=1):

            function = tool.get("function", {})

            name = function.get("name")

            if not name:
                continue

            description = (
                function.get("description")
                or "No description provided"
            )

            parameters = (
                function.get("parameters")
                or {}
            )

            try:
                schema = json.dumps(
                    parameters,
                    ensure_ascii=False,
                )
            except Exception:
                schema = "{}"

            formatted.append(
                f"{index}. Tool: {name}\n"
                f"   Description: {description}\n"
                f"   Parameters: {schema}"
            )

        if not formatted:
            return "AVAILABLE TOOLS:\nNone"

        return (
            "AVAILABLE TOOLS:\n"
            + "\n".join(formatted)
        )


    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------

    def _format_history(
        self,
        history: list[Any] | None,
        limit: int = 6,
    ) -> str:

        if not history:
            return ""

        messages: list[str] = []

        for msg in history[-limit:]:

            role = self._get_message_value(
                msg,
                "role",
            )

            content = self._get_message_value(
                msg,
                "content",
            )

            if not role or not content:
                continue

            content = str(content).strip()

            if not content:
                continue

            # Avoid sending excessive conversation content
            if len(content) > 500:
                content = content[:500] + "..."

            messages.append(
                f"{str(role).upper()}: {content}"
            )

        return "\n".join(messages)


    def _get_message_value(
        self,
        message: Any,
        key: str,
    ) -> Any:

        if isinstance(message, dict):
            return message.get(key)

        return getattr(
            message,
            key,
            None,
        )


    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------

    def _extract_memory_summary(
        self,
        memory_context: Any | None,
    ) -> str:

        if not memory_context:
            return ""

        if isinstance(memory_context, dict):
            summary = memory_context.get("summary")
        else:
            summary = getattr(
                memory_context,
                "summary",
                None,
            )

        if not summary:
            return ""

        summary = str(summary).strip()

        # Supervisor doesn't need huge memory payload
        if len(summary) > 1500:
            summary = summary[:1500] + "..."

        return summary


    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    def _resolve_model(
        self,
        model: str,
    ) -> str:

        from app.config import settings

        return (
            model
            or getattr(
                settings,
                "SUPERVISOR_MODEL",
                None,
            )
            or getattr(
                settings,
                "VLLM_DEFAULT_MODEL",
                "llama3.2:latest",
            )
        )


    # ---------------------------------------------------------
    # Supervisor Response Parser
    # ---------------------------------------------------------

    def _parse_supervisor_response(
        self,
        raw_text: str,
    ) -> dict[str, Any]:

        original_text = (raw_text or "").strip()
        if not original_text:
            raise ValueError("Supervisor returned empty response")

        # -----------------------------------------------------
        # 1. Handle Thinking Blocks (e.g. DeepSeek-R1 <think>...</think>)
        # -----------------------------------------------------
        think_content = ""
        if "</think>" in original_text:
            parts = original_text.split("</think>", 1)
            think_content = parts[0].replace("<think>", "").strip()
            working_text = parts[1].strip()
        elif "<think>" in original_text:
            think_content = original_text.replace("<think>", "").strip()
            working_text = ""
        else:
            working_text = original_text

        # If text outside think block is empty, inspect think_content
        if not working_text and think_content:
            working_text = think_content

        # -----------------------------------------------------
        # 2. Normalize Tokenizer Artifacts (Doubled Quotes)
        # -----------------------------------------------------
        working_text = re.sub(r'""+', '"', working_text)

        # -----------------------------------------------------
        # 3. Remove Markdown Code Blocks
        # -----------------------------------------------------
        if working_text.startswith("```"):
            working_text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", working_text)
            working_text = re.sub(r"\s*```$", "", working_text).strip()

        # -----------------------------------------------------
        # 4. Attempt Direct JSON Parse
        # -----------------------------------------------------
        try:
            parsed = json.loads(working_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # -----------------------------------------------------
        # 5. Extract Balanced JSON Object
        # -----------------------------------------------------
        json_candidate = self._extract_json_object(working_text)
        if json_candidate:
            try:
                parsed = json.loads(json_candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # -----------------------------------------------------
        # 6. Fallback: Robust Regex Field Extraction
        # Catches valid fields even if JSON is truncated mid-stream
        # -----------------------------------------------------
        route_match = re.search(r'["\']?route["\']?\s*[:=]\s*["\']?([A-Za-z]+)["\']?', working_text, re.IGNORECASE)
        if route_match:
            route_val = route_match.group(1).upper()
            conf_match = re.search(r'["\']?confidence["\']?\s*[:=]\s*([0-9.]+)', working_text)
            conf_val = float(conf_match.group(1)) if conf_match else 0.8
            reason_match = re.search(r'["\']?reason["\']?\s*[:=]\s*["\']([^"\']*)', working_text)
            reason_val = reason_match.group(1).strip() if reason_match else "Extracted from supervisor response"
            query_match = re.search(r'["\']?query["\']?\s*[:=]\s*["\']([^"\']*)', working_text)
            query_val = query_match.group(1).strip() if query_match else ""
            tool_match = re.search(r'["\']?tool_name["\']?\s*[:=]\s*["\']?([^"\'},\s]+)', working_text)
            tool_name = tool_match.group(1).strip() if tool_match and tool_match.group(1).strip() not in ("null", "None", "") else None

            return {
                "route": route_val,
                "confidence": conf_val,
                "reason": reason_val,
                "query": query_val,
                "tool_name": tool_name,
                "tool_args": {},
                "context_mode": "NONE",
            }

        # -----------------------------------------------------
        # 7. Semantic Heuristic Fallback from Reasoning Content
        # -----------------------------------------------------
        combined_text = (working_text + " " + think_content).upper()
        if "TOOL" in combined_text and any(w in combined_text for w in ["SEARCH", "WEB_SEARCH", "EXTERNAL"]):
            return {
                "route": "TOOL",
                "confidence": 0.8,
                "reason": "Deduced from supervisor reasoning context",
                "tool_name": "web_search",
                "tool_args": {},
            }
        elif "RAG" in combined_text or "KNOWLEDGE BASE" in combined_text:
            return {
                "route": "RAG",
                "confidence": 0.8,
                "reason": "Deduced from supervisor reasoning context",
            }
        elif "DIRECT" in combined_text:
            return {
                "route": "DIRECT",
                "confidence": 0.8,
                "reason": "Deduced from supervisor reasoning context",
            }

        raise ValueError(
            f"Unable to parse supervisor response: {original_text[:500]}"
        )


    def _extract_json_object(
        self,
        text: str,
    ) -> str | None:
        """
        Extract first balanced JSON object.

        Safer than regex {.*} when nested objects exist
        inside tool_args.
        """

        start = text.find("{")

        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(text)):

            char = text[index]

            if escape:
                escape = False
                continue

            if char == "\\":
                if in_string:
                    escape = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    return text[start:index + 1]

        return None


    # ---------------------------------------------------------
    # Decision Validation
    # ---------------------------------------------------------

    def _build_decision(
        self,
        *,
        parsed: dict[str, Any],
        original_query: str,
        has_kb: bool,
        tools: list[dict[str, Any]],
    ) -> RoutingDecision:

        route_str = str(
            parsed.get("route", "")
        ).strip().upper()

        reason = str(
            parsed.get("reason")
            or "Semantic supervisor decision"
        ).strip()

        query = str(
            parsed.get("query")
            or original_query
        ).strip()

        confidence = self._normalize_confidence(
            parsed.get("confidence")
        )

        tool_name = parsed.get("tool_name")

        tool_args = parsed.get("tool_args")

        if not isinstance(tool_args, dict):
            tool_args = {}

        context_mode = str(
            parsed.get("context_mode")
            or "NONE"
        ).strip().upper()

        if context_mode not in {
            "NONE",
            "REUSE",
            "REFRESH",
        }:
            context_mode = "NONE"


        # -----------------------------------------------------
        # DIRECT
        # -----------------------------------------------------

        if route_str == "DIRECT":

            if context_mode == "REFRESH":
                # A REFRESH request should not claim it can be answered
                # purely from existing context.
                context_mode = "NONE"


            return RoutingDecision(
                route=RouteType.DIRECT,
                reason=reason,
                query=query,
                confidence=confidence,
                context_mode=context_mode,
            )


        # -----------------------------------------------------
        # RAG
        # -----------------------------------------------------

        if route_str == "RAG":

            if not has_kb:

                logger.warning(
                    "Supervisor selected RAG but no KB is configured"
                )

                return RoutingDecision(
                    route=RouteType.DIRECT,
                    reason=(
                        "Private knowledge was requested but "
                        "no knowledge base is configured"
                    ),
                    query=query,
                    confidence=confidence,
                )

            return RoutingDecision(
                route=RouteType.RAG,
                reason=reason,
                query=query,
                confidence=confidence,
                context_mode=context_mode,
            )


        # -----------------------------------------------------
        # TOOL
        # -----------------------------------------------------

        if route_str == "TOOL":

            valid_tools = self._tool_map(tools)

            if not valid_tools:

                return RoutingDecision(
                    route=RouteType.DIRECT,
                    reason=(
                        "External capability was requested but "
                        "no tools are configured"
                    ),
                    query=query,
                    confidence=confidence,
                )


            # Never automatically choose tools[0].
            #
            # The supervisor must explicitly select a valid tool.

            if (
                not tool_name
                or tool_name not in valid_tools
            ):

                logger.warning(
                    "Supervisor selected invalid tool: %s",
                    tool_name,
                )

                return RoutingDecision(
                    route=RouteType.DIRECT,
                    reason=(
                        "Supervisor did not select a valid "
                        "configured tool"
                    ),
                    query=query,
                    confidence=confidence,
                )


            # Basic parameter validation.
            #
            # Full JSON Schema validation can be added here later.

            tool_args = self._sanitize_tool_args(
                tool=valid_tools[tool_name],
                tool_args=tool_args,
            )

            return RoutingDecision(
                route=RouteType.TOOL,
                reason=reason,
                query=query,
                confidence=confidence,
                tool_name=tool_name,
                tool_args=tool_args,
                context_mode=context_mode,
            )


        # -----------------------------------------------------
        # Unknown route
        # -----------------------------------------------------

        logger.warning(
            "Supervisor returned unknown route: %s",
            route_str,
        )

        return RoutingDecision(
            route=RouteType.DIRECT,
            reason="Unknown supervisor route; using safe direct fallback",
            query=original_query,
            confidence=0.0,
        )


    # ---------------------------------------------------------
    # Tool Validation
    # ---------------------------------------------------------

    def _tool_map(
        self,
        tools: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:

        result: dict[str, dict[str, Any]] = {}

        for tool in tools:

            function = tool.get(
                "function",
                {},
            )

            name = function.get("name")

            if name:
                result[name] = tool

        return result


    def _sanitize_tool_args(
        self,
        *,
        tool: dict[str, Any],
        tool_args: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Remove parameters invented by the supervisor.

        This performs lightweight validation.

        Recommended next step:
        validate with jsonschema before tool execution.
        """

        function = tool.get(
            "function",
            {},
        )

        schema = function.get(
            "parameters",
            {},
        )

        properties = schema.get(
            "properties",
            {},
        )

        # If schema doesn't define properties,
        # preserve arguments because some tool providers
        # may use custom schemas.
        if not properties:
            return tool_args

        allowed = set(properties.keys())

        sanitized = {
            key: value
            for key, value in tool_args.items()
            if key in allowed
        }

        return sanitized


    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    def _normalize_confidence(
        self,
        value: Any,
    ) -> float:

        try:
            confidence = float(value)

        except (TypeError, ValueError):
            return 0.5

        return max(
            0.0,
            min(1.0, confidence),
        )


# ---------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------

AgentRouter = SemanticSupervisor
AgentSupervisor = SemanticSupervisor