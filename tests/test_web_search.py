import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.tools.web_search import WebSearchTool


class WebSearchToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tool = WebSearchTool()

    @patch("app.tools.web_search.WebSearchTool._search_duckduckgo_pip")
    async def test_primary_duckduckgo_pip_search_used(self, mock_ddg_pip):
        mock_ddg_pip.return_value = [
            {
                "title": "Python 3.14 Release",
                "url": "https://python.org/release",
                "snippet": "Python 3.14 is now available.",
            }
        ]

        result = await self.tool.execute({"query": "Python 3.14", "max_results": 3})

        mock_ddg_pip.assert_awaited_once_with("Python 3.14", 3)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["title"], "Python 3.14 Release")
        self.assertEqual(result["results"][0]["url"], "https://python.org/release")
        self.assertEqual(result["results"][0]["snippet"], "Python 3.14 is now available.")

    @patch("app.tools.web_search.WebSearchTool._search_duckduckgo")
    @patch("app.tools.web_search.WebSearchTool._search_duckduckgo_pip")
    async def test_fallback_when_primary_fails(self, mock_ddg_pip, mock_ddg_http):
        mock_ddg_pip.side_effect = Exception("Rate limit reached")
        mock_ddg_http.return_value = [
            {
                "title": "Fallback Result",
                "url": "https://example.com/fallback",
                "snippet": "Fallback content",
            }
        ]

        result = await self.tool.execute({"query": "Python news", "max_results": 5})

        mock_ddg_pip.assert_awaited_once()
        mock_ddg_http.assert_awaited_once_with("Python news", 5)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["title"], "Fallback Result")

    @patch("app.tools.web_search.WebSearchTool._search_duckduckgo")
    @patch("app.tools.web_search.WebSearchTool._search_duckduckgo_pip")
    async def test_fallback_when_primary_returns_empty(self, mock_ddg_pip, mock_ddg_http):
        mock_ddg_pip.return_value = []
        mock_ddg_http.return_value = [
            {
                "title": "HTTP Fallback Result",
                "url": "https://example.com/http",
                "snippet": "HTTP content",
            }
        ]

        result = await self.tool.execute({"query": "Rare topic", "max_results": 2})

        mock_ddg_pip.assert_awaited_once_with("Rare topic", 2)
        mock_ddg_http.assert_awaited_once_with("Rare topic", 2)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["title"], "HTTP Fallback Result")

    async def test_empty_query_handling(self):
        result = await self.tool.execute({"query": "   "})
        self.assertEqual(result["count"], 0)
        self.assertIn("error", result)

    @patch("app.tools.web_search.WebSearchTool._search_duckduckgo_pip")
    async def test_string_and_wrapped_json_arguments(self, mock_ddg_pip):
        mock_ddg_pip.return_value = [{"title": "OK", "url": "https://ok.com", "snippet": "OK"}]

        # JSON string in markdown
        arg_str = '```json\n{"query": "AI News", "max_results": 2}\n```'
        result = await self.tool.execute(arg_str)

        mock_ddg_pip.assert_awaited_with("AI News", 2)
        self.assertEqual(result["count"], 1)


if __name__ == "__main__":
    unittest.main()
