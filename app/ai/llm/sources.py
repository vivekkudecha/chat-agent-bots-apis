"""Track references supplied to generation and select citations for the response."""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from markdown_it import MarkdownIt


@dataclass
class SourceCandidate:
    citation_id: str
    document_id: uuid.UUID | None
    knowledge_base_id: uuid.UUID | None
    file_name: str | None
    text: str
    score: float
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase for language-agnostic text comparison."""
    return " ".join(text.lower().split())


def _find_common_spans(s1: str, s2: str, min_len: int = 14) -> list[str]:
    """Dynamically find continuous shared substrings between two texts without fixed word lists."""
    s1_l = _normalize_text(s1)
    s2_l = _normalize_text(s2)
    if not s1_l or not s2_l:
        return []

    spans = []
    i = 0
    while i < len(s1_l):
        best_span = ""
        for j in range(i + min_len, len(s1_l) + 1):
            sub = s1_l[i:j]
            if sub in s2_l:
                if len(sub) > len(best_span):
                    best_span = sub
            else:
                break
        if best_span:
            spans.append(best_span)
            i += len(best_span)
        else:
            i += 1
    return spans


def _check_dynamic_grounding(answer: str, candidate_text: str) -> bool:
    """Language-agnostic dynamic grounding check without any hardcoded word or pattern lists.

    Verifies whether factual content, distinctive phrases, or entity codes from the
    candidate text were actually utilized to compose the answer.
    """
    if not answer.strip() or not candidate_text.strip():
        return False

    # 1. Substantive shared spans (continuous phrases >= 18 characters or cumulative >= 25 characters)
    spans = _find_common_spans(answer, candidate_text, min_len=14)
    if any(len(s.strip()) >= 18 for s in spans) or sum(len(s.strip()) for s in spans) >= 25:
        return True

    # 2. Distinctive numerical / alphanumeric entity code match (e.g. SB-553, 402-B, specific figures)
    cand_tokens = set(re.findall(r"\w+", candidate_text.lower()))
    ans_tokens = set(re.findall(r"\w+", answer.lower()))
    shared_tokens = cand_tokens & ans_tokens

    # Distinctive tokens containing digits or length >= 7
    distinctive_shared = {
        t for t in shared_tokens if (any(c.isdigit() for c in t) and len(t) >= 2) or len(t) >= 7
    }
    if len(distinctive_shared) >= 2 and len(shared_tokens) >= 3:
        return True

    return False


def select_cited_sources(
    answer: str, candidates: list[SourceCandidate],
) -> list[dict[str, Any]]:
    """Return only current-prompt references cited in or dynamically grounding the final answer.

    Unknown IDs, uncited candidates, and candidates whose content was not actually
    utilized in the answer are deliberately excluded. Works dynamically across any language
    without fixed word dictionaries.
    """
    if not candidates or not answer.strip():
        return []

    # Parse rendered prose, excluding code blocks/spans and link labels. Protect
    # escaped brackets before Markdown unescapes them into ordinary text tokens.
    prose_parts: list[str] = []
    cited_urls: set[str] = set()
    for block in MarkdownIt("commonmark").parse(answer.replace(r"\[", "［")):
        in_link = False
        for token in block.children or []:
            if token.type == "link_open":
                in_link = True
                cited_urls.add(token.attrGet("href") or "")
            elif token.type == "link_close":
                in_link = False
            elif token.type == "text" and not in_link:
                prose_parts.append(token.content)
    prose = "\n".join(prose_parts)

    cited_ids: set[str] = set()
    for group in re.findall(
        r"\[((?:KB|WEB)[1-9]\d*(?:\s*[,;]\s*(?:KB|WEB)[1-9]\d*)*)\]", prose,
    ):
        cited_ids.update(re.findall(r"(?:KB|WEB)[1-9]\d*", group))

    for url in re.findall(r'https?://[^\s<>\[\]"`]+', prose):
        url = url.rstrip(".,;!?")
        # Preserve balanced URL parentheses (e.g. Wikipedia article names).
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        cited_urls.add(url)

    sources: dict[tuple[str, str], dict[str, Any]] = {}
    seen_ids: set[str] = set()

    for candidate in candidates:
        url = candidate.metadata.get("url")
        is_web = candidate.metadata.get("source_type") == "web_search"

        # Explicit citation check (marker, URL, or distinct file name)
        is_explicitly_cited = (
            candidate.citation_id in cited_ids
            or (is_web and url and url in cited_urls)
        )
        if not is_explicitly_cited and not is_web and candidate.file_name:
            fn_stem = candidate.file_name.rsplit(".", 1)[0].strip().lower()
            if len(fn_stem) >= 5 and re.search(r"\b" + re.escape(fn_stem) + r"\b", prose.lower()):
                is_explicitly_cited = True

        # Dynamic grounding check (language-agnostic without fixed word lists)
        is_grounded = False
        if not is_explicitly_cited:
            is_grounded = _check_dynamic_grounding(prose, candidate.text)

        if not is_explicitly_cited and not is_grounded:
            continue

        if candidate.citation_id in seen_ids:
            continue
        seen_ids.add(candidate.citation_id)

        key = ("web", url) if is_web and url else ("document", str(candidate.document_id))
        preview = candidate.text[:250].strip()
        if len(candidate.text) > 250:
            preview += "..."

        if key not in sources:
            sources[key] = {
                "document_id": candidate.document_id,
                "knowledge_base_id": candidate.knowledge_base_id,
                "file_name": candidate.file_name,
                "page": candidate.page,
                "pages": [candidate.page] if candidate.page is not None else [],
                "score": round(candidate.score, 4),
                "chunk_count": 1,
                "content_preview": preview or None,
                "metadata": {**candidate.metadata, "citation_ids": [candidate.citation_id]},
            }
            continue

        source = sources[key]
        source["metadata"]["citation_ids"].append(candidate.citation_id)
        if not is_web:
            source["chunk_count"] += 1
            if candidate.page is not None and candidate.page not in source["pages"]:
                source["pages"].append(candidate.page)
                source["pages"].sort()
        if candidate.score > source["score"]:
            source.update(
                score=round(candidate.score, 4),
                page=candidate.page,
                content_preview=preview or None,
            )

    return list(sources.values())
