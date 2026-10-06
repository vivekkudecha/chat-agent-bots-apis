from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Any

import httpx
from lxml import html

from app.config import settings
from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Live web search tool.
    Provides internet browsing capabilities to bots using Tavily, SerpAPI,
    or fast zero-dependency DuckDuckGo Lite search as default.
    """

    name: str = "web_search"
    display_name: str = "Live Web Search"
    description: str = (
        "Search the live web for recent news, up-to-date facts, current events, and external references."
    )
    tool_type: str = "search"
    timeout_seconds: int = 15

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(
        self,
        arguments: dict[str, Any] | str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Executes live web search using available provider backends.
        """
        # Handle cases where arguments are passed as a JSON string
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments)
            except Exception:
                parsed_args = {"query": arguments}
        elif isinstance(arguments, dict):
            parsed_args = arguments
        else:
            parsed_args = {}

        query = (parsed_args.get("query") or parsed_args.get("q") or "").strip()
        if not query:
            return {
                "query": "",
                "count": 0,
                "results": [],
                "error": "Empty search query provided.",
            }

        max_results = int(
            parsed_args.get("max_results")
            or getattr(settings, "WEB_SEARCH_MAX_RESULTS", 5)
            or 5
        )
        max_results = max(1, min(10, max_results))

        logger.info("Executing web search: query='%s', max_results=%d", query, max_results)

        # 1. Try Tavily if configured
        tavily_key = getattr(settings, "TAVILY_API_KEY", None)
        if tavily_key:
            try:
                results = await self._search_tavily(query, tavily_key, max_results)
                if results:
                    return {"query": query, "count": len(results), "results": results}
            except Exception as exc:
                logger.warning("Tavily search failed, falling back: %s", exc)

        # 2. Try SerpAPI if configured
        serpapi_key = getattr(settings, "SERPAPI_API_KEY", None)
        if serpapi_key:
            try:
                results = await self._search_serpapi(query, serpapi_key, max_results)
                if results:
                    return {"query": query, "count": len(results), "results": results}
            except Exception as exc:
                logger.warning("SerpAPI search failed, falling back: %s", exc)

        # 3. Default: High-reliability DuckDuckGo search
        try:
            results = await self._search_duckduckgo(query, max_results)
            return {"query": query, "count": len(results), "results": results}
        except Exception as exc:
            logger.exception("DuckDuckGo web search failed: %s", exc)
            return {
                "query": query,
                "count": 0,
                "results": [],
                "error": f"Search failed: {str(exc)}",
            }

    async def _search_tavily(
        self,
        query: str,
        api_key: str,
        max_results: int,
    ) -> list[dict[str, str]]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    }
                )
            return results

    async def _search_serpapi(
        self,
        query: str,
        api_key: str,
        max_results: int,
    ) -> list[dict[str, str]]:
        url = "https://serpapi.com/search"
        params = {
            "api_key": api_key,
            "q": query,
            "engine": "google",
            "num": max_results,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("organic_results", [])[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    }
                )
            return results

    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, str]]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        # Try DuckDuckGo Lite first (cleanest and fastest without JS)
        results: list[dict[str, str]] = []
        try:
            async with httpx.AsyncClient(
                headers=headers,
                timeout=self.timeout_seconds,
                follow_redirects=True,
            ) as client:
                resp = await client.post(
                    "https://lite.duckduckgo.com/lite/",
                    data={"q": query},
                )
                if resp.status_code == 200:
                    tree = html.fromstring(resp.content)
                    rows = tree.xpath("//table//tr")
                    for i in range(len(rows)):
                        link_nodes = rows[i].xpath('.//a[contains(@class, "result-link")]')
                        if link_nodes:
                            title = link_nodes[0].text_content().strip()
                            raw_url = link_nodes[0].get("href", "")
                            url = self._unwrap_ddg_url(raw_url)
                            snippet = ""
                            if i + 1 < len(rows):
                                snip_node = rows[i + 1].xpath(
                                    './/td[contains(@class, "result-snippet")]'
                                )
                                if snip_node:
                                    snippet = snip_node[0].text_content().strip()
                            if title and url:
                                results.append(
                                    {
                                        "title": title,
                                        "url": url,
                                        "snippet": snippet,
                                    }
                                )
                                if len(results) >= max_results:
                                    break
        except Exception as exc:
            logger.warning("DuckDuckGo Lite search attempt failed: %s", exc)

        if results:
            return results

        # Fallback to html.duckduckgo.com if Lite had no results
        try:
            async with httpx.AsyncClient(
                headers=headers,
                timeout=self.timeout_seconds,
                follow_redirects=True,
            ) as client:
                resp = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                )
                if resp.status_code == 200:
                    tree = html.fromstring(resp.content)
                    for node in tree.xpath(
                        '//div[contains(@class, "web-result") or contains(@class, "result__body")]'
                    )[:max_results]:
                        title_nodes = node.xpath(".//h2//text()")
                        snippet_nodes = node.xpath(
                            './/a[contains(@class, "result__snippet")]//text()'
                        )
                        link_nodes = node.xpath(
                            './/a[contains(@class, "result__url")]/@href'
                        )
                        title = "".join(title_nodes).strip()
                        snippet = "".join(snippet_nodes).strip()
                        raw_url = link_nodes[0].strip() if link_nodes else ""
                        url = self._unwrap_ddg_url(raw_url)
                        if title and url:
                            results.append(
                                {
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet,
                                }
                            )
        except Exception as exc:
            logger.warning("DuckDuckGo HTML search attempt failed: %s", exc)

        return results

    @staticmethod
    def _unwrap_ddg_url(raw_url: str) -> str:
        """
        Unwraps redirected DuckDuckGo links (/l/?uddg=...).
        """
        if not raw_url:
            return ""
        if "uddg=" in raw_url:
            try:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                if "uddg" in parsed:
                    return parsed["uddg"][0]
            except Exception:
                pass
        return raw_url
