"""Clarifying questions and follow-up suggestions grounded in the KB.

When a question is too vague or the knowledge base has no evidence for
it, the assistant asks one clarifying question and offers specific
questions it *can* answer, built from the sections retrieval came close
to. After a partial answer it suggests related follow-ups the same way.

Small-model friendly: one short JSON call, thinking disabled, strict
validation, and a deterministic fallback when the call fails.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

from app.ai.llm.provider import LLMProvider, LLMUsage
from app.ai.rag.agentic import parse_json_object
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Clarification:
    question: str
    suggestions: list[str] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)

    @property
    def text(self) -> str:
        return FollowUpSuggester.format_reply(self.question, self.suggestions)


def document_title(file_name: str | None) -> str:
    if not file_name:
        return ""
    stem = PurePath(file_name).stem
    return re.sub(r"[_\-]+", " ", stem).strip()


def section_leaf(section: str | None, parts: int = 2) -> str:
    if not section:
        return ""
    return " > ".join(p.strip() for p in section.split(">")[-parts:])


class FollowUpSuggester:

    SYSTEM_PROMPT = (
        "You help users of a document assistant phrase questions the "
        "knowledge base can answer. You never answer the question yourself. "
        "Reply with one JSON object only."
    )

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    # -----------------------------------------------------
    # Clarify (ambiguous / not found)
    # -----------------------------------------------------

    async def clarify(
        self,
        *,
        question: str,
        topics: list[dict[str, Any]],
        model: str,
        document_titles: list[str] | None = None,
        found_nothing: bool = False,
        topic: bool = False,
        bot_instruction: str = "",
    ) -> Clarification:

        if not topics and document_titles:
            topics = [{"file_name": title} for title in document_titles]

        if not topics:
            return Clarification(
                question=(
                    "I couldn't find that in the knowledge base. Could you add a "
                    f"bit more detail about what you'd like to know about \"{question}\"?"
                ),
            )

        if found_nothing:
            situation = "The knowledge base has no passage that answers the message."
        elif topic:
            situation = (
                "The message only names a topic the knowledge base covers. Do not answer; "
                "ask which aspect of it the user wants to know about."
            )
        else:
            situation = "The message is too short or unclear to know exactly what the user wants."

        prompt = (
            self._bot_rules(bot_instruction)
            + f'USER MESSAGE: "{question}"\n'
            f"SITUATION: {situation}\n\n"
            "KNOWLEDGE BASE TOPICS:\n"
            + self._format_topics(topics)
            + "\n\nReturn this JSON:\n"
            '{"question": "one short, friendly clarifying question to the user", '
            '"suggestions": ["up to 4 specific questions the user could ask instead"]}\n\n'
            "Rules:\n"
            "- Every suggestion must be answerable from the topics listed above; never invent topics.\n"
            "- Prefer topics related to the user message; if none relate, say briefly what the "
            "knowledge base covers and suggest questions about those topics.\n"
            "- Write each suggestion as a complete question the user would type, under 15 words.\n"
            "- Use the language of the user message unless the BOT INSTRUCTIONS require another.\n"
            "- Follow the BOT INSTRUCTIONS (tone, scope, language); only suggest questions "
            "within the scope they allow."
        )

        data, usage = await self._ask(prompt, model)
        suggestions = self.clean_suggestions(
            (data or {}).get("suggestions"),
            question=question,
        )
        clarifying = self._text((data or {}).get("question"))

        if not suggestions:
            return self._fallback(question, topics, found_nothing=found_nothing, usage=usage)

        return Clarification(
            question=clarifying or self._fallback_question(question, found_nothing),
            suggestions=suggestions,
            usage=usage,
        )

    # -----------------------------------------------------
    # Related follow-ups (partial answers)
    # -----------------------------------------------------

    async def related(
        self,
        *,
        question: str,
        answer: str,
        topics: list[dict[str, Any]],
        model: str,
        bot_instruction: str = "",
    ) -> tuple[list[str], LLMUsage]:

        if not topics:
            return [], LLMUsage()

        prompt = (
            self._bot_rules(bot_instruction)
            + f'QUESTION: "{question}"\n\n'
            f"ANSWER GIVEN (partial):\n{answer[:1200]}\n\n"
            "KNOWLEDGE BASE TOPICS:\n"
            + self._format_topics(topics)
            + "\n\nReturn this JSON:\n"
            '{"suggestions": ["up to 3 follow-up questions the user may want to ask next"]}\n\n'
            "Rules:\n"
            "- Each follow-up must be answerable from the topics listed above.\n"
            "- Do not repeat what the answer already covers.\n"
            "- Complete questions under 15 words, in the language of the question unless the "
            "BOT INSTRUCTIONS require another.\n"
            "- Stay within the scope and tone of the BOT INSTRUCTIONS."
        )

        data, usage = await self._ask(prompt, model)
        return (
            self.clean_suggestions(
                (data or {}).get("suggestions"),
                question=question,
                limit=min(3, settings.RAG_MAX_SUGGESTIONS),
            ),
            usage,
        )

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    @staticmethod
    def _bot_rules(bot_instruction: str) -> str:
        """Owner's bot instructions govern user-visible wording and scope."""
        text = (bot_instruction or "").strip()
        if not text:
            return ""
        return (
            "BOT INSTRUCTIONS (mandatory, written by the bot owner):\n"
            f"<bot_instructions>\n{text[:2000]}\n</bot_instructions>\n\n"
        )

    async def _ask(self, prompt: str, model: str) -> tuple[dict[str, Any] | None, LLMUsage]:
        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2,
                top_p=1.0,
                max_tokens=260,
                response_format={"type": "json_object"},
                reasoning_effort=settings.LLM_INTERNAL_REASONING_EFFORT or None,
            )
        except Exception as exc:
            logger.warning("Suggestion generation failed: %s", exc)
            return None, LLMUsage()

        data = parse_json_object(response.content or "")
        if data is None:
            logger.warning("Unparseable suggestions: %r", (response.content or "")[:300])
        return data, response.usage

    @staticmethod
    def _format_topics(topics: list[dict[str, Any]]) -> str:
        lines = []
        for number, topic in enumerate(topics[:8], start=1):
            label = " › ".join(
                part
                for part in (document_title(topic.get("file_name")), section_leaf(topic.get("section")))
                if part
            ) or "Untitled"
            preview = (topic.get("preview") or "")[:120]
            lines.append(f"{number}. {label}" + (f": {preview}" if preview else ""))
        return "\n".join(lines)

    @staticmethod
    def _text(value: Any) -> str:
        return " ".join(value.split()) if isinstance(value, str) else ""

    @classmethod
    def clean_suggestions(
        cls,
        items: Any,
        *,
        question: str,
        limit: int | None = None,
    ) -> list[str]:

        limit = limit or settings.RAG_MAX_SUGGESTIONS
        if not isinstance(items, list):
            return []

        asked = " ".join(question.lower().split())
        seen: set[str] = set()
        cleaned: list[str] = []

        for item in items:
            text = cls._text(item)
            text = re.sub(r"^(\d+[.)]|[-*•])\s*", "", text).strip(" \"'")
            key = text.lower()
            if not 8 <= len(text) <= 160 or key in seen or key == asked:
                continue
            seen.add(key)
            cleaned.append(text)
            if len(cleaned) >= limit:
                break

        return cleaned

    @staticmethod
    def _fallback_question(question: str, found_nothing: bool) -> str:
        if found_nothing:
            return f"I couldn't find a direct answer for \"{question}\". Did you mean one of these?"
        return f"Could you tell me a bit more about what you need regarding \"{question}\"? For example:"

    @classmethod
    def _fallback(
        cls,
        question: str,
        topics: list[dict[str, Any]],
        *,
        found_nothing: bool,
        usage: LLMUsage,
    ) -> Clarification:
        suggestions = []
        for topic in topics:
            document = document_title(topic.get("file_name"))
            leaf = section_leaf(topic.get("section"), parts=1)
            if document and leaf:
                suggestions.append(f"What does {document} say about {leaf}?")
            elif document:
                suggestions.append(f"What does {document} cover?")
        return Clarification(
            question=cls._fallback_question(question, found_nothing),
            suggestions=cls.clean_suggestions(suggestions, question=question),
            usage=usage,
        )

    @staticmethod
    def format_reply(question: str, suggestions: list[str]) -> str:
        if not suggestions:
            return question
        return question + "\n\n" + "\n".join(f"- {s}" for s in suggestions)

    @staticmethod
    def format_followups(suggestions: list[str]) -> str:
        if not suggestions:
            return ""
        return "\n\n**You might also ask:**\n" + "\n".join(f"- {s}" for s in suggestions)
