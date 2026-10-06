from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any

from app.ai.agent.state import RouteType
from app.ai.llm.provider import LLMProvider, get_llm_provider

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

Your responsibility is NOT to answer the user.

Your responsibility is to determine which available capability should
handle the user's request.

You must reason about the semantic intent of the request and the
capabilities available to you.

--------------------------------------------------
AVAILABLE EXECUTION PATHS
--------------------------------------------------

DIRECT

Use DIRECT when the base language model can answer the request without
retrieving private knowledge and without using an external capability.

Examples include:
- conversation
- reasoning
- writing
- explanation
- coding
- summarization of information already provided
- general knowledge
- requests where the required capability is unavailable


RAG

Use RAG when answering the request is likely to require information
contained inside the configured private knowledge sources.

Do NOT choose RAG merely because a knowledge base exists.

Choose RAG only when the user's request is semantically related to
information that could reasonably belong to the available knowledge
sources.


TOOL

Use TOOL when answering requires one of the available external
capabilities.

This may include:
- retrieving live/current information
- searching external systems
- executing an action
- calling an external API
- accessing information available through a configured tool

Never invent a tool.

If TOOL is selected:
1. Select exactly one available tool.
2. Generate tool arguments matching that tool's schema.
3. Never select a tool simply because it is available.

--------------------------------------------------
IMPORTANT ROUTING RULES
--------------------------------------------------

1. Route according to semantic intent, NOT keywords.

2. A configured knowledge base does NOT mean every question should use RAG.

3. An available tool does NOT mean every external-looking question
   should automatically use that tool.

4. Never invent capabilities.

5. Never invent tool names.

6. Never invent tool parameters that are not supported by the
   provided tool schema.

7. Use conversation context to resolve references such as:
   "it", "that", "they", "he", "she", "this", "those", etc.

8. Rewrite the request into a standalone query whenever context is
   required.

Example:

Conversation:
User: Tell me about the employee leave rules.
Assistant: ...

Current request:
What about fathers?

Standalone query:
What leave benefits are available to fathers?

9. Preserve the user's actual intent when rewriting the query.

10. If the required capability is unavailable, use DIRECT.
    The downstream assistant can explain the limitation.

11. If uncertain between RAG and DIRECT, prefer DIRECT unless the
    request clearly depends on private knowledge.

12. If uncertain between TOOL and DIRECT, prefer DIRECT unless an
    available tool is clearly necessary.

--------------------------------------------------
CONFIDENCE
--------------------------------------------------

Return a confidence value between 0.0 and 1.0.

Confidence indicates how certain you are that the selected execution
path is appropriate.

It is NOT a probability and should not be fabricated with unnecessary
precision.

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
    "reason": "Short explanation of why this capability is appropriate",
    "query": "Standalone contextualized version of the user's request",
    "tool_name": null,
    "tool_args": {}
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
                bot_instruction[:1500]
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
        # Recent conversation
        # -----------------------------------------------------

        history_text = self._format_history(history)

        if history_text:
            sections.append(
                "RECENT CONVERSATION:\n"
                + history_text
            )


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


        # -----------------------------------------------------
        # DIRECT
        # -----------------------------------------------------

        if route_str == "DIRECT":

            return RoutingDecision(
                route=RouteType.DIRECT,
                reason=reason,
                query=query,
                confidence=confidence,
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