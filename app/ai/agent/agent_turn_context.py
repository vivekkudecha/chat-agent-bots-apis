from __future__ import annotations

"""
Helpers for persisting agent-specific turn context.

Store this alongside the assistant message (preferably in a JSON/JSONB
metadata column). This is what makes tool evidence available on the next
chat turn instead of losing it after the LangGraph invocation ends.
"""

import json
from typing import Any


def build_agent_turn_metadata(
    state: dict[str, Any],
    *,
    max_tool_result_chars: int = 12000,
) -> dict[str, Any]:
    tool_results = list(
        state.get("tool_results") or []
    )

    compact_results = []

    for item in tool_results:
        if not isinstance(item, dict):
            continue

        result = item.get("result")

        compact_results.append(
            {
                "name": item.get("name"),
                "arguments": item.get("arguments") or {},
                "result": _compact(
                    result,
                    max_tool_result_chars,
                ),
            }
        )

    return {
        "agent": {
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "resolved_query": state.get("resolved_query"),
            "context_mode": state.get(
                "context_mode",
                "NONE",
            ),
            "tool_calls": list(
                state.get("tool_calls") or []
            ),
            "tool_results": compact_results,
        }
    }


def _compact(
    value: Any,
    max_chars: int,
) -> Any:
    """
    Preserve structured data where possible.

    For web_search, retain the useful fields and truncate snippets rather
    than serializing an enormous page extraction into conversation metadata.
    """
    if not isinstance(value, dict):
        return value

    result = dict(value)

    rows = result.get("results")

    if isinstance(rows, list):
        compact_rows = []

        for row in rows[:10]:
            if not isinstance(row, dict):
                continue

            compact_rows.append(
                {
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "snippet": str(
                        row.get("snippet") or ""
                    )[:3000],
                }
            )

        result["results"] = compact_rows

    try:
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        encoded = str(result)

    if len(encoded) <= max_chars:
        return result

    # Last-resort truncation. Keep it explicit so the model knows evidence
    # was truncated instead of silently receiving malformed JSON.
    return {
        "truncated": True,
        "content": encoded[:max_chars],
    }
