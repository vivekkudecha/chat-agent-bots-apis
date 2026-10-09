"""Evidence grading for agentic (corrective) retrieval.

After each retrieval round a single, compact LLM call decides which
passages actually help, whether they answer the question, and — when
they do not — what to search for next. Designed for small local models:
short prompt, truncated passages, strict JSON, thinking disabled, and a
fail-open fallback (keep everything) when the output cannot be parsed.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.ai.llm.provider import LLMProvider
from app.ai.rag.retrieval import RetrievalService
from app.config import settings

logger = logging.getLogger(__name__)


ANSWERABLE = ("yes", "partial", "no")


def parse_json_object(content: str) -> dict[str, Any] | None:
    """First JSON object in a small-model reply (think tags/fences tolerated)."""

    text = (content or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


@dataclass
class EvidenceGrade:
    # 0-based indexes into the graded passage list.
    relevant: list[int] = field(default_factory=list)
    answerable: str = "yes"
    missing: str = ""
    next_query: str = ""
    # The question is too short/unclear to know what is being asked.
    ambiguous: bool = False
    parsed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "relevant": self.relevant,
            "answerable": self.answerable,
            "missing": self.missing,
            "next_query": self.next_query,
            "ambiguous": self.ambiguous,
            "parsed": self.parsed,
        }


class EvidenceGrader:

    SYSTEM_PROMPT = (
        "You check whether search results answer a question. "
        "You never answer the question yourself. "
        "Reply with one JSON object only."
    )

    def __init__(
        self,
        llm: LLMProvider,
        *,
        passage_chars: int | None = None,
    ):
        self.llm = llm
        self.passage_chars = passage_chars or settings.RAG_GRADER_PASSAGE_CHARS

    def build_prompt(
        self,
        *,
        question: str,
        passages: list[Any],
        previous_queries: list[str],
    ) -> str:

        blocks = []
        for number, passage in enumerate(passages, start=1):
            label = []
            if getattr(passage, "file_name", None):
                label.append(passage.file_name)
            pages = (passage.metadata or {}).get("pages") or []
            if pages:
                label.append(
                    f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"
                )
            section = (passage.metadata or {}).get("section")
            if section:
                label.append(section[-80:])

            # Long (expanded) passages: show the window that best matches
            # the question rather than the leading neighbour chunk.
            text = " ".join((passage.text or "").split())
            if len(text) > self.passage_chars:
                text = RetrievalService._extract_focused_snippet(
                    passage.text,
                    query=question,
                    max_chars=self.passage_chars,
                )
                text = " ".join(text.split())

            header = f"[{number}]" + (f" ({' · '.join(label)})" if label else "")
            blocks.append(f"{header}\n{text}")

        tried = "; ".join(f'"{q}"' for q in previous_queries) or "none"

        return (
            f"QUESTION: {question}\n\n"
            "PASSAGES:\n" + "\n\n".join(blocks) + "\n\n"
            "Return this JSON:\n"
            '{"relevant": [numbers of passages containing information needed to answer], '
            '"answerable": "yes" if the relevant passages fully answer the question, '
            '"partial" if part of the needed information is missing, "no" if no passage helps, '
            '"missing": "what information is still missing, or empty", '
            '"next_query": "a short search query for the missing information, or empty", '
            '"ambiguous": true if the question is too short or unclear to know what the user '
            'wants (for example a single word that several passages answer differently), else false}\n\n'
            "Rules:\n"
            "- Judge only from the passage text, not from your own knowledge.\n"
            "- A passage on the same topic that does not contain the needed facts is not relevant.\n"
            f"- next_query must be different from the searches already tried: {tried}.\n"
            "- If the question is only a keyword or topic (e.g. \"LSA\", \"relocation\"), treat it as "
            "\"What does the knowledge base say about it?\": passages describing it are relevant and "
            "answerable is \"yes\" when they explain it; set ambiguous to true when they describe "
            "several different aspects.\n"
            "- Write next_query in the language of the question; if the question is only one or two "
            "words, use the language of the passages."
        )

    async def grade(
        self,
        *,
        question: str,
        passages: list[Any],
        model: str,
        previous_queries: list[str] | None = None,
    ) -> EvidenceGrade:

        if not passages:
            return EvidenceGrade(answerable="no", parsed=True)

        prompt = self.build_prompt(
            question=question,
            passages=passages,
            previous_queries=previous_queries or [],
        )

        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=settings.RAG_GRADER_MODEL or model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=220,
                response_format={"type": "json_object"},
                reasoning_effort=settings.LLM_INTERNAL_REASONING_EFFORT or None,
            )
        except Exception as exc:
            logger.warning("Evidence grading failed, keeping all passages: %s", exc)
            return self._fail_open(passages)

        grade = self.parse(response.content or "", passage_count=len(passages))
        if grade is None:
            logger.warning(
                "Unparseable evidence grade, keeping all passages: %r",
                (response.content or "")[:300],
            )
            return self._fail_open(passages)

        return grade

    @staticmethod
    def _fail_open(passages: list[Any]) -> EvidenceGrade:
        return EvidenceGrade(
            relevant=list(range(len(passages))),
            answerable="yes",
            parsed=False,
        )

    @staticmethod
    def parse(content: str, *, passage_count: int) -> EvidenceGrade | None:

        data = parse_json_object(content)
        if data is None:
            return None

        raw_relevant = data.get("relevant", [])
        if isinstance(raw_relevant, (int, str)):
            raw_relevant = [raw_relevant]

        relevant: list[int] = []
        for item in raw_relevant if isinstance(raw_relevant, list) else []:
            match = re.search(r"\d+", str(item))
            if match:
                number = int(match.group())
                if 1 <= number <= passage_count and number - 1 not in relevant:
                    relevant.append(number - 1)

        answerable = str(data.get("answerable", "")).strip().lower()
        if answerable not in ANSWERABLE:
            answerable = "partial" if relevant else "no"

        def text_field(name: str) -> str:
            value = data.get(name)
            if not isinstance(value, str):
                return ""
            value = value.strip()
            empty = {"", "no", "none", "nothing", "null", "n/a", "empty"}
            return "" if value.lower().strip(".") in empty else value

        ambiguous = data.get("ambiguous", False)
        if isinstance(ambiguous, str):
            ambiguous = ambiguous.strip().lower() in {"true", "yes", "1"}

        return EvidenceGrade(
            relevant=sorted(relevant),
            answerable=answerable,
            missing=text_field("missing"),
            next_query=text_field("next_query")[:200],
            ambiguous=bool(ambiguous),
            parsed=True,
        )
