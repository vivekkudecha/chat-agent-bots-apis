from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ddgs import DDGS

from app.config import settings
from app.tools.base import BaseTool

logger = logging.getLogger(__name__)

_TIME_LIMITS = {
    "any": None,
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
}


class WebSearchTool(BaseTool):
    """
    DuckDuckGo-only web search.

    Design:
    - No keyword-based routing.
    - The caller/LLM can semantically choose freshness and region.
    - Web + news can be searched together.
    - Top result pages are extracted so snippets contain real evidence.
    - External response format remains unchanged:
        {
            "query": "...",
            "count": N,
            "results": [
                {"title": "...", "url": "...", "snippet": "..."}
            ]
        }
    """

    name: str = "web_search"
    display_name: str = "Live Web Search"
    description: str = (
        "Search the live web with DuckDuckGo. Use for current/external "
        "information. For time-sensitive requests, set time_range semantically. "
        "For geographically relevant searches, set the appropriate region."
    )
    tool_type: str = "search"
    timeout_seconds: int = 15

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Standalone search query.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
            "search_type": {
                "type": "string",
                "enum": ["auto", "web", "news"],
                "default": "auto",
                "description": (
                    "auto searches web and news together. "
                    "Use web or news only when semantically appropriate."
                ),
            },
            "time_range": {
                "type": "string",
                "enum": ["any", "day", "week", "month", "year"],
                "default": "any",
                "description": (
                    "Freshness constraint. Example: a request about this week "
                    "should use week; a request about today should use day."
                ),
            },
            "region": {
                "type": "string",
                "description": (
                    "DuckDuckGo region such as in-en, us-en, uk-en, wt-wt."
                ),
            },
        },
        "required": ["query"],
    }

    async def execute(
        self,
        arguments: dict[str, Any] | str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del context

        args = self._parse_arguments(arguments)

        query = str(
            args.get("query")
            or args.get("search_query")
            or args.get("q")
            or args.get("input")
            or ""
        ).strip().strip("\"'")

        if not query:
            return {
                "query": "",
                "count": 0,
                "results": [],
                "error": "Empty search query provided.",
            }

        max_results = self._int(
            args.get("max_results"),
            default=getattr(settings, "WEB_SEARCH_MAX_RESULTS", 5) or 5,
            minimum=1,
            maximum=10,
        )

        search_type = str(
            args.get("search_type") or "auto"
        ).strip().lower()

        if search_type not in {"auto", "web", "news"}:
            search_type = "auto"

        time_range = str(
            args.get("time_range") or "any"
        ).strip().lower()

        timelimit = _TIME_LIMITS.get(time_range)

        region = str(
            args.get("region")
            or getattr(settings, "WEB_SEARCH_REGION", "wt-wt")
            or "wt-wt"
        ).strip()

        try:
            results = await asyncio.wait_for(
                self._search(
                    query=query,
                    max_results=max_results,
                    search_type=search_type,
                    timelimit=timelimit,
                    region=region,
                ),
                timeout=float(
                    getattr(settings, "WEB_SEARCH_TOTAL_TIMEOUT", 25) or 25
                ),
            )

            if results:
                return {
                    "query": query,
                    "count": len(results),
                    "results": results,
                }

        except Exception as exc:
            logger.exception(
                "DuckDuckGo search failed query=%r: %s",
                query,
                exc,
            )

        return {
            "query": query,
            "count": 0,
            "results": [],
            "error": "No usable search results returned from DuckDuckGo.",
        }

    async def _search(
        self,
        *,
        query: str,
        max_results: int,
        search_type: str,
        timelimit: str | None,
        region: str,
    ) -> list[dict[str, str]]:
        candidate_count = min(
            20,
            max(max_results * 2, 8),
        )

        branches = (
            ["web", "news"]
            if search_type == "auto"
            else [search_type]
        )

        tasks = [
            asyncio.to_thread(
                self._ddg_search,
                branch,
                query,
                candidate_count,
                timelimit,
                region,
            )
            for branch in branches
        ]

        outputs = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        raw: list[dict[str, Any]] = []

        for branch, output in zip(
            branches,
            outputs,
            strict=True,
        ):
            if isinstance(output, Exception):
                logger.warning(
                    "DuckDuckGo %s branch failed: %s",
                    branch,
                    output,
                )
                continue

            raw.extend(output)

        if not raw:
            return []

        ranked = self._rank_and_dedupe(
            query=query,
            items=raw,
            max_results=max_results,
        )

        # Search snippets are often too shallow for list/research questions.
        # Enrich only the strongest results, while keeping the same response
        # schema by placing extracted evidence into `snippet`.
        enrich_count = min(
            len(ranked),
            int(
                getattr(
                    settings,
                    "WEB_SEARCH_EXTRACT_RESULTS",
                    3,
                )
                or 3
            ),
        )

        if enrich_count > 0:
            await self._enrich(
                ranked[:enrich_count]
            )

        return ranked

    def _ddg_search(
        self,
        search_type: str,
        query: str,
        max_results: int,
        timelimit: str | None,
        region: str,
    ) -> list[dict[str, Any]]:
        safesearch = (
            getattr(
                settings,
                "WEB_SEARCH_SAFESEARCH",
                "moderate",
            )
            or "moderate"
        )

        with DDGS(
            timeout=self.timeout_seconds
        ) as ddgs:
            kwargs = {
                "region": region,
                "safesearch": safesearch,
                "timelimit": timelimit,
                "max_results": max_results,
                "backend": "duckduckgo",
            }

            rows = (
                ddgs.news(query, **kwargs)
                if search_type == "news"
                else ddgs.text(query, **kwargs)
            )

            results = []

            for rank, item in enumerate(
                rows or [],
                start=1,
            ):
                title = str(
                    item.get("title") or ""
                ).strip()

                url = str(
                    item.get("url")
                    or item.get("href")
                    or item.get("link")
                    or ""
                ).strip()

                snippet = str(
                    item.get("body")
                    or item.get("snippet")
                    or item.get("description")
                    or ""
                ).strip()

                date = str(
                    item.get("date") or ""
                ).strip()

                source = str(
                    item.get("source") or ""
                ).strip()

                if date:
                    snippet = (
                        f"Published: {date}\n{snippet}"
                        if snippet
                        else f"Published: {date}"
                    )

                if source:
                    snippet = (
                        f"Source: {source}\n{snippet}"
                        if snippet
                        else f"Source: {source}"
                    )

                if title and url:
                    results.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "_rank": rank,
                        }
                    )

            return results

    async def _enrich(
        self,
        results: list[dict[str, str]],
    ) -> None:
        semaphore = asyncio.Semaphore(
            int(
                getattr(
                    settings,
                    "WEB_SEARCH_EXTRACT_CONCURRENCY",
                    3,
                )
                or 3
            )
        )

        max_chars = int(
            getattr(
                settings,
                "WEB_SEARCH_EXTRACT_MAX_CHARS",
                5000,
            )
            or 5000
        )

        async def enrich_one(
            item: dict[str, str],
        ) -> None:
            async with semaphore:
                try:
                    content = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._extract_page,
                            item["url"],
                        ),
                        timeout=float(self.timeout_seconds),
                    )
                except Exception as exc:
                    logger.debug(
                        "Page extraction failed url=%s: %s",
                        item.get("url"),
                        exc,
                    )
                    return

                content = self._clean_content(
                    content
                )

                if not content:
                    return

                original = item.get(
                    "snippet",
                    "",
                ).strip()

                extracted = content[:max_chars]

                item["snippet"] = (
                    f"{original}\n\nPage content:\n{extracted}"
                    if original
                    else extracted
                )

        await asyncio.gather(
            *(enrich_one(item) for item in results),
            return_exceptions=True,
        )

    def _extract_page(
        self,
        url: str,
    ) -> str:
        with DDGS(
            timeout=self.timeout_seconds
        ) as ddgs:
            result = ddgs.extract(
                url,
                fmt="text_markdown",
            )

        content = result.get(
            "content",
            "",
        )

        if isinstance(content, bytes):
            return content.decode(
                "utf-8",
                errors="replace",
            )

        return str(content or "")

    @classmethod
    def _rank_and_dedupe(
        cls,
        *,
        query: str,
        items: list[dict[str, Any]],
        max_results: int,
    ) -> list[dict[str, str]]:
        query_terms = cls._tokens(query)

        seen: set[str] = set()
        scored: list[
            tuple[
                float,
                dict[str, str],
            ]
        ] = []

        for item in items:
            title = str(
                item.get("title") or ""
            ).strip()

            url = cls._canonical_url(
                str(item.get("url") or "")
            )

            snippet = str(
                item.get("snippet") or ""
            ).strip()

            if (
                not title
                or not url
                or url in seen
            ):
                continue

            seen.add(url)

            all_terms = cls._tokens(
                f"{title} {snippet}"
            )

            title_terms = cls._tokens(
                title
            )

            if query_terms:
                coverage = (
                    len(
                        query_terms
                        & all_terms
                    )
                    / len(query_terms)
                )

                title_coverage = (
                    len(
                        query_terms
                        & title_terms
                    )
                    / len(query_terms)
                )
            else:
                coverage = 0.0
                title_coverage = 0.0

            rank = int(
                item.get("_rank") or 100
            )

            score = (
                coverage * 5.0
                + title_coverage * 3.0
                + 1.0 / max(rank, 1)
            )

            scored.append(
                (
                    score,
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    },
                )
            )

        scored.sort(
            key=lambda row: row[0],
            reverse=True,
        )

        return [
            result
            for _, result
            in scored[:max_results]
        ]

    @staticmethod
    def _parse_arguments(
        arguments: dict[str, Any] | str,
    ) -> dict[str, Any]:
        if isinstance(arguments, dict):
            parsed = dict(arguments)

        elif isinstance(arguments, str):
            raw = arguments.strip()
            raw = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                raw,
                flags=re.IGNORECASE,
            )

            try:
                loaded = json.loads(raw)

                parsed = (
                    loaded
                    if isinstance(
                        loaded,
                        dict,
                    )
                    else {"query": raw}
                )

            except json.JSONDecodeError:
                parsed = {
                    "query": raw
                }

        else:
            return {}

        for key in (
            "parameters",
            "arguments",
            "args",
        ):
            nested = parsed.get(key)

            if isinstance(
                nested,
                dict,
            ):
                return nested

        return parsed

    @staticmethod
    def _tokens(
        text: str,
    ) -> set[str]:
        return {
            token
            for token in re.findall(
                r"[a-z0-9][a-z0-9._+-]*",
                text.lower(),
            )
            if len(token) > 1
        }

    @staticmethod
    def _clean_content(
        text: str,
    ) -> str:
        text = str(
            text or ""
        ).replace(
            "\x00",
            "",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _canonical_url(
        url: str,
    ) -> str:
        try:
            parts = urlsplit(
                url.strip()
            )

            if (
                parts.scheme
                not in {"http", "https"}
                or not parts.netloc
            ):
                return ""

            query = [
                (key, value)
                for key, value
                in parse_qsl(
                    parts.query,
                    keep_blank_values=True,
                )
                if not key.lower().startswith(
                    "utm_"
                )
                and key.lower()
                not in {
                    "gclid",
                    "fbclid",
                    "msclkid",
                }
            ]

            return urlunsplit(
                (
                    parts.scheme.lower(),
                    parts.netloc.lower(),
                    parts.path or "/",
                    urlencode(query),
                    "",
                )
            )

        except Exception:
            return ""

    @staticmethod
    def _int(
        value: Any,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            value = int(value)
        except (
            TypeError,
            ValueError,
        ):
            value = default

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )
