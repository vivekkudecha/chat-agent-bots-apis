from __future__ import annotations

"""
Conversation context utilities.

The important distinction is:

1. Chat history      -> what the user and assistant said.
2. Tool evidence     -> what external systems actually returned.
3. Working context   -> a compact representation of the previous turns.

Do NOT rely on the assistant's previous answer as the source of truth when
that answer was produced from a tool. Preserve the original tool evidence.
"""

import json
from typing import Any


class ConversationContextBuilder:
    """
    Builds a compact, model-friendly working context from persisted turns.

    Expected history item shapes are intentionally flexible. The builder can
    read:

        {
            "role": "assistant",
            "content": "...",
            "metadata": {
                "agent": {
                    "route": "TOOL",
                    "resolved_query": "...",
                    "tool_calls": [...],
                    "tool_results": [...]
                }
            }
        }

    or ORM-like objects exposing the same attributes.

    It also supports `tool_results`, `tool_calls`, `route`, and
    `resolved_query` directly on the history item.
    """

    def __init__(
        self,
        *,
        max_turns: int = 8,
        max_message_chars: int = 1500,
        max_evidence_chars: int = 12000,
        max_total_chars: int = 24000,
    ):
        self.max_turns = max_turns
        self.max_message_chars = max_message_chars
        self.max_evidence_chars = max_evidence_chars
        self.max_total_chars = max_total_chars

    def build(
        self,
        history: list[Any] | None,
    ) -> dict[str, Any]:
        if not history:
            return {
                "text": "",
                "recent_turns": [],
                "previous_tool_evidence": [],
                "last_turn": None,
            }

        turns = []

        for item in history[-self.max_turns:]:
            turn = self._normalize_item(item)

            if (
                not turn["content"]
                and not turn["tool_results"]
            ):
                continue

            turns.append(turn)

        previous_tool_evidence = []

        for turn in turns:
            for tool_result in turn["tool_results"]:
                if isinstance(tool_result, dict):
                    previous_tool_evidence.append(
                        {
                            "turn_index": turn["index"],
                            "tool_name": (
                                tool_result.get("name")
                                or turn["tool_name"]
                                or "external_tool"
                            ),
                            "resolved_query": turn["resolved_query"],
                            "result": tool_result.get(
                                "result",
                                tool_result,
                            ),
                        }
                    )
                else:
                    previous_tool_evidence.append(
                        {
                            "turn_index": turn["index"],
                            "tool_name": (
                                turn["tool_name"]
                                or "external_tool"
                            ),
                            "resolved_query": turn["resolved_query"],
                            "result": tool_result,
                        }
                    )

        text = self._format_context(
            turns,
            previous_tool_evidence,
        )

        return {
            "text": text[: self.max_total_chars],
            "recent_turns": turns,
            "previous_tool_evidence": previous_tool_evidence,
            "last_turn": turns[-1] if turns else None,
        }

    def _normalize_item(
        self,
        item: Any,
    ) -> dict[str, Any]:
        role = self._get(item, "role") or ""
        content = self._get(item, "content") or ""

        metadata = self._get(item, "metadata")

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        if not isinstance(metadata, dict):
            metadata = {}

        # Support several common metadata names.
        agent = (
            metadata.get("agent")
            or metadata.get("agent_context")
            or metadata.get("context")
            or {}
        )

        if not isinstance(agent, dict):
            agent = {}

        route = (
            agent.get("route")
            or self._get(item, "route")
        )

        resolved_query = (
            agent.get("resolved_query")
            or self._get(item, "resolved_query")
        )

        tool_calls = (
            agent.get("tool_calls")
            or self._get(item, "tool_calls")
            or []
        )

        tool_results = (
            agent.get("tool_results")
            or self._get(item, "tool_results")
            or []
        )

        if not isinstance(tool_calls, list):
            tool_calls = []

        if not isinstance(tool_results, list):
            tool_results = []

        tool_name = None

        if tool_calls:
            first_call = tool_calls[-1]
            if isinstance(first_call, dict):
                tool_name = (
                    first_call.get("name")
                    or first_call.get("tool_name")
                )

        if not tool_name and tool_results:
            first_result = tool_results[-1]
            if isinstance(first_result, dict):
                tool_name = (
                    first_result.get("name")
                    or first_result.get("tool_name")
                )

        return {
            "index": self._get(item, "id") or id(item),
            "role": str(role),
            "content": str(content).strip()[
                : self.max_message_chars
            ],
            "route": route,
            "resolved_query": resolved_query,
            "tool_name": tool_name,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    def _format_context(
        self,
        turns: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> str:
        sections: list[str] = []

        if turns:
            lines = [
                "RECENT CONVERSATION:",
                "---------------------",
            ]

            for turn in turns:
                role = (
                    turn["role"].upper()
                    if turn["role"]
                    else "UNKNOWN"
                )

                content = turn["content"]

                if content:
                    lines.append(
                        f"{role}: {content}"
                    )

                if turn["resolved_query"]:
                    lines.append(
                        "  Resolved query: "
                        f"{turn['resolved_query']}"
                    )

                if turn["route"]:
                    lines.append(
                        "  Route: "
                        f"{turn['route']}"
                    )

            sections.append(
                "\n".join(lines)
            )

        if evidence:
            evidence_lines = [
                "",
                "PREVIOUS EXTERNAL TOOL EVIDENCE:",
                "--------------------------------",
                (
                    "This is source data retrieved in earlier turns. "
                    "Use it for follow-up filtering/summarization when "
                    "the current request refers to the previous result."
                ),
            ]

            used = 0

            for index, item in enumerate(
                evidence,
                start=1,
            ):
                block = self._format_evidence(
                    index,
                    item,
                )

                remaining = (
                    self.max_evidence_chars
                    - used
                )

                if remaining <= 300:
                    break

                block = block[:remaining]

                evidence_lines.append(
                    block
                )

                used += len(block)

            sections.append(
                "\n".join(evidence_lines)
            )

        return "\n\n".join(sections)

    @staticmethod
    def _format_evidence(
        index: int,
        item: dict[str, Any],
    ) -> str:
        tool_name = (
            item.get("tool_name")
            or "external_tool"
        )

        query = (
            item.get("resolved_query")
            or ""
        )

        result = item.get("result")

        if tool_name == "web_search":
            return (
                f"\n[PREVIOUS TOOL RESULT {index}]\n"
                f"Tool: {tool_name}\n"
                f"Query: {query}\n"
                "Evidence:\n"
                f"{ConversationContextBuilder._format_web_result(result)}"
            )

        return (
            f"\n[PREVIOUS TOOL RESULT {index}]\n"
            f"Tool: {tool_name}\n"
            f"Query: {query}\n"
            "Evidence:\n"
            f"{ConversationContextBuilder._compact_json(result)}"
        )

    @staticmethod
    def _format_web_result(
        result: Any,
    ) -> str:
        if not isinstance(result, dict):
            return ConversationContextBuilder._compact_json(
                result
            )

        rows = result.get("results")

        if not isinstance(rows, list):
            return ConversationContextBuilder._compact_json(
                result
            )

        lines = []

        if result.get("query"):
            lines.append(
                f"Search query: {result['query']}"
            )

        for index, row in enumerate(
            rows,
            start=1,
        ):
            if not isinstance(row, dict):
                continue

            title = str(
                row.get("title") or ""
            ).strip()

            url = str(
                row.get("url") or ""
            ).strip()

            snippet = str(
                row.get("snippet") or ""
            ).strip()

            lines.extend(
                [
                    f"[{index}] {title}",
                    f"URL: {url}",
                    f"Evidence: {snippet}",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _compact_json(
        value: Any,
        max_chars: int = 6000,
    ) -> str:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            text = str(value)

        return text[:max_chars]

    @staticmethod
    def _get(
        item: Any,
        key: str,
    ) -> Any:
        if isinstance(item, dict):
            return item.get(key)

        return getattr(
            item,
            key,
            None,
        )
