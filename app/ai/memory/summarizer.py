from __future__ import annotations

import json
import logging
from typing import Any

from app.ai.llm.provider import LLMProvider, get_llm_provider
from app.ai.memory.schemas import ConversationSummary
from app.models.conversation import Message

logger = logging.getLogger(__name__)


SUMMARIZE_SYSTEM_PROMPT = """
You are an expert conversation analyst. Your task is to extract and update an episodic memory summary of a conversation between a User and an AI Assistant.

Maintain maximum information density while keeping the summary concise.
Focus specifically on:
1. The overarching topic and user goal.
2. Concrete decisions made, technical choices, or agreed solutions.
3. User preferences, declared facts, constraints, credentials/identifiers (non-sensitive), and names.
4. Any pending questions, tasks, or follow-ups.

Do NOT include greetings, conversational filler, or transient acknowledgments.

Output ONLY valid JSON matching this schema:
{
  "topic": "string",
  "user_intent": "string",
  "key_decisions": ["string"],
  "entities_and_facts": {"key": "value"},
  "pending_actions": ["string"]
}
""".strip()


class ConversationSummarizer:
    """
    Summarizes older conversation turns into a structured, rolling episodic memory.
    """

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm or get_llm_provider()

    async def summarize_or_update(
        self,
        *,
        messages_to_compact: list[Message],
        existing_summary: ConversationSummary | None = None,
        model: str | None = None,
    ) -> ConversationSummary:
        if not messages_to_compact:
            return existing_summary or ConversationSummary()

        formatted_dialogue = self._format_messages(messages_to_compact)

        prompt_content_parts = []
        if existing_summary and not existing_summary.is_empty():
            prompt_content_parts.append(
                "CURRENT CONVERSATION SUMMARY TO UPDATE:\n"
                f"{json.dumps(existing_summary.to_dict(), indent=2)}\n"
            )

        prompt_content_parts.append(
            "NEW MESSAGES TO INCORPORATE INTO SUMMARY:\n"
            f"{formatted_dialogue}\n\n"
            "Produce the updated JSON summary combining previous summary and new messages:"
        )

        user_prompt = "\n".join(prompt_content_parts)

        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=model or getattr(self.llm, "default_model", "llama3.2:latest"),
                temperature=0.2,
                max_tokens=1000,
            )

            raw_text = response.content.strip()
            summary_data = self._clean_and_parse_json(raw_text)

            new_summary = ConversationSummary.from_dict(summary_data)
            new_summary.last_summarized_message_id = str(messages_to_compact[-1].id)
            new_summary.total_summarized_messages = (
                (existing_summary.total_summarized_messages if existing_summary else 0)
                + len(messages_to_compact)
            )
            return new_summary

        except Exception as exc:
            logger.warning(
                "Failed to generate structured conversation summary: %s. Preserving existing summary.",
                exc,
            )
            if existing_summary:
                return existing_summary
            # Fallback simple summary
            return ConversationSummary(
                topic="Ongoing conversation",
                user_intent="General inquiry",
                last_summarized_message_id=str(messages_to_compact[-1].id),
                total_summarized_messages=len(messages_to_compact),
            )

    @staticmethod
    def _format_messages(messages: list[Message]) -> str:
        lines: list[str] = []
        for m in messages:
            role = m.role.capitalize()
            content = (m.content or "").strip()
            if role.lower() == "assistant" and content.startswith("{"):
                from app.ai.llm.provider import OpenAICompatibleProvider
                extracted = OpenAICompatibleProvider._extract_tool_calls_from_content(content)
                if extracted:
                    fn = extracted[0].get("function", {})
                    fn_name = fn.get("name", "tool")
                    args = fn.get("arguments", {})
                    q = args.get("query") or args.get("q") or ""
                    content = f"[Searched web for: '{q}']" if q else f"[Used tool: {fn_name}]"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _clean_and_parse_json(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_slice = text[start : end + 1]
            return json.loads(json_slice)

        return json.loads(text)
