import unittest
import uuid
from datetime import datetime
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

# The application creates its Qdrant client at import time. Keep tests offline.
with patch("qdrant_client.AsyncQdrantClient"):
    from app.ai.agent.graph import ChatAgentGraph
    from app.ai.llm.prompt_builder import PromptBuilderService
    from app.ai.llm.provider import LLMResponse
    from app.ai.llm.sources import SourceCandidate, select_cited_sources
    from app.ai.rag.retrieval import RetrievedChunk, RetrievalResult
    from app.services import chat_service as service_module


def chunk(text, *, document_id=None, page=1):
    return RetrievedChunk(
        id=str(uuid.uuid4()), text=text, score=0.8,
        document_id=document_id or uuid.uuid4(), knowledge_base_id=uuid.UUID(int=1),
        file_name="policy.pdf", page=page,
    )


class ChatSourcesTests(unittest.IsolatedAsyncioTestCase):
    async def run_chat(self, answer, chunks=(), web_results=(), *, max_context=8192,
                       guarded_answer=None, responses=None):
        """Exercise real prompt building, LangGraph generation, and ChatService.

        Replace only external retrieval, inference, guardrails and persistence.
        """
        user_id, bot_id, conversation_id = [uuid.uuid4() for _ in range(3)]
        version = SimpleNamespace(system_instruction="Answer questions.", metadata_={})
        model = SimpleNamespace(id=uuid.uuid4(), model_key="test", provider="test",
                                is_active=True, context_window=max_context)
        retrieval = RetrievalResult("leave policy", list(chunks), [uuid.UUID(int=1)])
        route = "rag" if chunks else ("tool" if web_results else "direct")
        llm = SimpleNamespace(chat=AsyncMock(side_effect=responses or [
            LLMResponse(content=answer, model="test")
        ]))
        guard = SimpleNamespace(final_text=guarded_answer if guarded_answer is not None else answer,
                                warnings=[], executions=[])
        memory = SimpleNamespace(working_history=[], total_conversation_messages=0)
        tools = SimpleNamespace(
            get_active_bot_tools=AsyncMock(return_value=[]),
            execute_tool=AsyncMock(return_value={"results": list(web_results)}),
        )
        graph = ChatAgentGraph(
            router=None, retrieval_service=None, prompt_builder=PromptBuilderService(),
            llm_provider=llm, tool_registry=tools,
        )
        # Run generation through a real compiled graph to retain state fields.
        graph._context_builder_node = AsyncMock(return_value={})
        graph._supervisor_node = AsyncMock(return_value={"route": route})
        graph._retrieve_node = AsyncMock(return_value={"retrieval": retrieval})
        graph.app = graph._build_graph()
        db = SimpleNamespace(commit=AsyncMock())
        saved_messages = []

        async def save_message(db, **values):
            saved_messages.append(values)
            return SimpleNamespace(id=uuid.uuid4())

        async def evaluate(db, **values):
            if values["stage"].value == "input":
                return SimpleNamespace(final_text=values["text"], warnings=[], executions=[])
            return guard

        with ExitStack() as stack:
            # Avoid instantiating embedding models; generation itself stays real.
            stack.enter_context(patch.object(service_module, "RetrievalService"))
            service = service_module.ChatService(llm=llm)
            service.graph = graph
            service.tools = tools
            service.memory = SimpleNamespace(build_context=AsyncMock(return_value=memory),
                                             should_compact=lambda *args: False)
            service.guardrails = SimpleNamespace(evaluate=evaluate)
            service._resolve_conversation = AsyncMock(return_value=SimpleNamespace(id=conversation_id))
            for owner, name, value in [
                (service_module.BotRepository, "get_available_bot", SimpleNamespace(id=bot_id)),
                (service_module.BotRepository, "get_active_version", version),
                (service_module.BotRepository, "get_primary_model_config", None),
                (service_module.AIModelRepository, "get_by_key", model),
                (service_module.KnowledgeRepository, "list_for_bot", [object()] if chunks else []),
            ]:
                stack.enter_context(patch.object(owner, name, AsyncMock(return_value=value)))
            stack.enter_context(patch.object(service_module.ConversationRepository, "create_message", save_message))
            stack.enter_context(patch.object(service_module.UsageRepository, "create", AsyncMock()))
            stack.enter_context(patch.object(service_module.AuditRepository, "create", AsyncMock()))
            result = await service.chat(db, user_id=user_id, bot_id=bot_id, message="What is the leave policy?")
        return result, saved_messages[-1]

    async def test_returns_only_cited_document_and_persists_same_sources(self):
        relevant = chunk("Employees receive 20 days of leave.")
        irrelevant = chunk("The cafeteria opens at nine.")
        result, saved = await self.run_chat("You receive 20 days of leave. [KB1]", [relevant, irrelevant])
        self.assertEqual([s.document_id for s in result.sources], [relevant.document_id])
        self.assertEqual(saved["metadata"]["source_count"], 1)
        self.assertEqual(len(saved["metadata"]["sources"]), 1)

    async def test_uncited_answer_has_no_sources(self):
        result, saved = await self.run_chat("I could not find that information.", [chunk("Cafeteria hours.")])
        self.assertEqual(result.sources, [])
        self.assertEqual(saved["metadata"]["sources"], [])

    async def test_cannot_cite_document_excluded_by_prompt_budget(self):
        result, _ = await self.run_chat("A claim. [KB2]", [chunk("Long text " * 2000), chunk("Unseen text")], max_context=2500)
        self.assertEqual(result.sources, [])

    async def test_only_cited_pages_are_reported_for_same_document(self):
        doc_id = uuid.uuid4()
        result, _ = await self.run_chat("20 days. [KB2]", [
            chunk("Cafeteria hours.", document_id=doc_id, page=1),
            chunk("20 days of leave.", document_id=doc_id, page=8),
        ])
        self.assertEqual(result.sources[0].pages, [8])
        self.assertEqual(result.sources[0].chunk_count, 1)
        self.assertIn("20 days", result.sources[0].content_preview)

    async def test_returns_only_cited_web_result(self):
        result, saved = await self.run_chat("Policy is unchanged. [WEB2]", web_results=[
            {"title": "Sports", "url": "https://example.com/sports", "snippet": "Match results"},
            {"title": "Policy", "url": "https://example.com/policy", "snippet": "Policy is unchanged"},
        ])
        self.assertEqual([s.metadata["url"] for s in result.sources], ["https://example.com/policy"])
        self.assertEqual(saved["metadata"]["sources"][0]["url"], "https://example.com/policy")

    async def test_guardrail_replacement_cannot_retain_original_citations(self):
        result, _ = await self.run_chat("20 days. [KB1]", [chunk("20 days of leave.")], guarded_answer="Response removed.")
        self.assertEqual(result.sources, [])

    async def test_direct_answer_has_no_sources(self):
        result, _ = await self.run_chat("Hello!")
        self.assertEqual(result.sources, [])

    async def test_followup_generation_uses_its_own_source_catalog(self):
        answer = "20 days of leave. [WEB4]"
        result, _ = await self.run_chat(answer, web_results=[
            {"title": "Other", "url": "https://example.com/other", "snippet": "Other information"},
            {"title": "Policy", "url": "https://example.com/policy", "snippet": "20 days of leave"},
        ], responses=[
            LLMResponse(content="", model="test", tool_calls=[{
                "function": {"name": "web_search", "arguments": {"query": "leave policy"}},
            }]),
            LLMResponse(content=answer, model="test"),
        ])
        self.assertEqual([s.metadata["url"] for s in result.sources], ["https://example.com/policy"])


class CitationSelectionTests(unittest.TestCase):
    def test_markdown_code_and_escaped_markers_are_not_citations(self):
        candidate = SourceCandidate("KB1", uuid.uuid4(), uuid.UUID(int=1), "policy.pdf", "Leave", 0.8)
        for answer in ["~~~text\n[KB1]\n~~~", "Example:\n\n    [KB1]", "``[KB1]``", r"\[KB1]", "```text\n[KB1]"]:
            with self.subTest(answer=answer):
                self.assertEqual(select_cited_sources(answer, [candidate]), [])

    def test_grouped_citations_merge_only_cited_pages(self):
        doc_id = uuid.uuid4()
        candidates = [
            SourceCandidate(f"KB{i}", doc_id, uuid.UUID(int=1), "policy.pdf", "Leave policy", 0.8, page=i)
            for i in range(1, 4)
        ]
        sources = select_cited_sources("Leave details. [KB1, KB3] [KB1] [KB99]", candidates)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["pages"], [1, 3])
        self.assertEqual(sources[0]["chunk_count"], 2)

    def test_web_links_match_whole_urls_and_deduplicate(self):
        url = "https://example.com/policy"
        candidates = [
            SourceCandidate(f"WEB{i}", uuid.uuid4(), None, "Policy", "Leave details", 1.0,
                            metadata={"source_type": "web_search", "url": url})
            for i in range(1, 3)
        ]
        self.assertEqual(select_cited_sources("https://example.com/policy-unrelated", candidates), [])
        sources = select_cited_sources(f"See [policy]({url}). [WEB1] [WEB2]", candidates)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["chunk_count"], 1)

    def test_web_url_with_parentheses_remains_citable(self):
        url = "https://example.com/Leave_(policy)"
        candidate = SourceCandidate("WEB1", uuid.uuid4(), None, "Policy", "Leave details", 1.0,
                                    metadata={"source_type": "web_search", "url": url})
        sources = select_cited_sources(f"See [policy]({url}).", [candidate])
        self.assertEqual(len(sources), 1)

    def test_code_examples_and_unknown_ids_do_not_create_sources(self):
        candidate = SourceCandidate("KB1", uuid.uuid4(), uuid.UUID(int=1), "policy.pdf", "Leave", 0.8)
        for answer in ["`[KB1]`", "```text\n[KB1]\n```", "[KB99]", "[1]", "[KB1](https://unrelated.example)"]:
            with self.subTest(answer=answer):
                self.assertEqual(select_cited_sources(answer, [candidate]), [])

    def test_prompt_catalog_matches_only_visible_context(self):
        relevant = chunk("Leave details " * 500)
        unseen = chunk("Unseen source")
        prompt = PromptBuilderService().build(
            bot_version=SimpleNamespace(system_instruction="Help", metadata_={}),
            user_message="Leave policy?", context_window=1000, max_generation_tokens=500,
            retrieval=RetrievalResult("Leave", [relevant, unseen], [uuid.UUID(int=1)]),
        )
        self.assertEqual([c.document_id for c in prompt.source_candidates], [relevant.document_id])
        self.assertIn(prompt.source_candidates[0].text, prompt.context_text)
        self.assertLess(len(prompt.source_candidates[0].text), len(relevant.text))

    def test_dynamic_document_grounding_without_citation_marker(self):
        candidate = SourceCandidate(
            "KB1", uuid.uuid4(), uuid.UUID(int=1), "policy.pdf",
            "The Leave Policy outlines the company approach to employee leave, including procedures for requesting, approving, and managing leave. Employees receive 20 days of paid annual leave.",
            0.9,
            metadata={"source_type": "knowledge_base"},
        )
        answer = "The Leave Policy outlines the company approach to employee leave, including procedures for requesting, approving, and managing leave."
        sources = select_cited_sources(answer, [candidate])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["document_id"], candidate.document_id)

    def test_dynamic_grounding_returns_both_rag_and_web_sources(self):
        doc_cand = SourceCandidate(
            "KB1", uuid.uuid4(), uuid.UUID(int=1), "policy.pdf",
            "The Leave Policy outlines the company approach to employee leave. Employees receive 20 days of paid annual leave.",
            0.9,
            metadata={"source_type": "knowledge_base"},
        )
        web_cand = SourceCandidate(
            "WEB1", uuid.uuid4(), None, "California Labor Updates",
            "California SB-553 mandates workplace violence prevention plans starting July 2024.",
            1.0,
            metadata={"source_type": "web_search", "url": "https://example.com/sb553"},
        )
        answer = "Under our Leave Policy, employees receive 20 days of paid annual leave. Also, California SB-553 mandates workplace violence prevention plans starting July 2024."
        sources = select_cited_sources(answer, [doc_cand, web_cand])
        self.assertEqual(len(sources), 2)
        source_types = {s["metadata"]["source_type"] for s in sources}
        self.assertEqual(source_types, {"knowledge_base", "web_search"})

    def test_dynamic_grounding_returns_only_web_source_when_doc_not_used(self):
        doc_cand = SourceCandidate(
            "KB1", uuid.uuid4(), uuid.UUID(int=1), "policy.pdf",
            "The Leave Policy outlines the company approach to employee leave. Employees receive 20 days of paid annual leave.",
            0.9,
            metadata={"source_type": "knowledge_base"},
        )
        web_cand = SourceCandidate(
            "WEB1", uuid.uuid4(), None, "California Labor Updates",
            "California SB-553 mandates workplace violence prevention plans starting July 2024.",
            1.0,
            metadata={"source_type": "web_search", "url": "https://example.com/sb553"},
        )
        answer = "According to recent updates, California SB-553 mandates workplace violence prevention plans starting July 2024."
        sources = select_cited_sources(answer, [doc_cand, web_cand])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["metadata"]["source_type"], "web_search")

    def test_dynamic_grounding_returns_only_doc_source_when_web_not_used(self):
        doc_cand = SourceCandidate(
            "KB1", uuid.uuid4(), uuid.UUID(int=1), "policy.pdf",
            "The Leave Policy outlines the company approach to employee leave. Employees receive 20 days of paid annual leave.",
            0.9,
            metadata={"source_type": "knowledge_base"},
        )
        web_cand = SourceCandidate(
            "WEB1", uuid.uuid4(), None, "California Labor Updates",
            "California SB-553 mandates workplace violence prevention plans starting July 2024.",
            1.0,
            metadata={"source_type": "web_search", "url": "https://example.com/sb553"},
        )
        answer = "The Leave Policy outlines the company approach to employee leave. Employees receive 20 days of paid annual leave."
        sources = select_cited_sources(answer, [doc_cand, web_cand])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["metadata"]["source_type"], "knowledge_base")

    def test_dynamic_grounding_returns_no_sources_for_chitchat_or_refusal(self):
        doc_cand = SourceCandidate(
            "KB1", uuid.uuid4(), uuid.UUID(int=1), "policy.pdf",
            "The Leave Policy outlines the company approach to employee leave. Employees receive 20 days of paid annual leave.",
            0.9,
            metadata={"source_type": "knowledge_base"},
        )
        web_cand = SourceCandidate(
            "WEB1", uuid.uuid4(), None, "California Labor Updates",
            "California SB-553 mandates workplace violence prevention plans starting July 2024.",
            1.0,
            metadata={"source_type": "web_search", "url": "https://example.com/sb553"},
        )
        self.assertEqual(select_cited_sources("Hello! How can I help you today?", [doc_cand, web_cand]), [])
        self.assertEqual(select_cited_sources("I am sorry, but the provided documents do not contain that information.", [doc_cand, web_cand]), [])

    def test_dynamic_grounding_multilingual(self):
        doc_cand = SourceCandidate(
            "KB1", uuid.uuid4(), uuid.UUID(int=1), "policy.pdf",
            "कर्मचारियों को प्रति वर्ष 20 दिनों का सवेतन वार्षिक अवकाश मिलता है।",
            0.9,
            metadata={"source_type": "knowledge_base"},
        )
        answer = "कंपनी की नीति के अनुसार, कर्मचारियों को प्रति वर्ष 20 दिनों का सवेतन वार्षिक अवकाश मिलता है।"
        sources = select_cited_sources(answer, [doc_cand])
        self.assertEqual(len(sources), 1)
        self.assertEqual(select_cited_sources("नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ?", [doc_cand]), [])


    def test_message_response_retains_and_normalizes_sources(self):
        from app.schemas.conversation import MessageResponse

        doc_id = uuid.uuid4()
        msg = MessageResponse(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            role="assistant",
            content="Answer text",
            model_id=None,
            input_tokens=10,
            output_tokens=20,
            latency_ms=100,
            created_at=datetime.utcnow(),
            metadata_={
                "sources": [
                    {
                        "document_id": str(doc_id),
                        "file_name": "company_policy.pdf",
                        "content_preview": "20 days annual leave",
                        "page": 3,
                        "score": 0.95,
                    }
                ]
            },
        )
        # Verify top-level sources
        self.assertEqual(len(msg.sources), 1)
        self.assertEqual(msg.sources[0]["document_id"], str(doc_id))
        self.assertEqual(msg.sources[0]["file_name"], "company_policy.pdf")
        self.assertEqual(msg.sources[0]["fileName"], "company_policy.pdf")
        self.assertEqual(msg.sources[0]["excerpt"], "20 days annual leave")
        self.assertEqual(msg.sources[0]["page"], 3)
        self.assertEqual(msg.sources[0]["score"], 0.95)

        # Verify serialized dict has both top-level sources and metadata.sources
        dumped = msg.model_dump(by_alias=True)
        self.assertIn("sources", dumped)
        self.assertEqual(len(dumped["sources"]), 1)
        self.assertEqual(len(dumped["metadata"]["sources"]), 1)

    def test_message_response_normalizes_legacy_sources(self):
        from app.schemas.conversation import MessageResponse

        # Legacy format had "title" and "snippet" but no "file_name", "document_id", or "content_preview"
        msg = MessageResponse(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            role="assistant",
            content="Answer text",
            model_id=None,
            input_tokens=10,
            output_tokens=20,
            latency_ms=100,
            created_at=datetime.utcnow(),
            metadata_={
                "sources": [
                    {
                        "title": "rewards_policy.pdf",
                        "snippet": "Approved by CHRO",
                        "source_type": "knowledge_base",
                    }
                ]
            },
        )
        self.assertEqual(len(msg.sources), 1)
        self.assertEqual(msg.sources[0]["file_name"], "rewards_policy.pdf")
        self.assertEqual(msg.sources[0]["fileName"], "rewards_policy.pdf")
        self.assertEqual(msg.sources[0]["title"], "rewards_policy.pdf")
        self.assertEqual(msg.sources[0]["content_preview"], "Approved by CHRO")
        self.assertEqual(msg.sources[0]["excerpt"], "Approved by CHRO")
        self.assertEqual(msg.sources[0]["snippet"], "Approved by CHRO")
        self.assertEqual(msg.sources[0]["score"], 1.0)

    def test_message_response_synthesizes_fallback_when_source_count_without_sources(self):
        from app.schemas.conversation import MessageResponse

        msg = MessageResponse(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            role="assistant",
            content="Answer text",
            model_id=None,
            input_tokens=10,
            output_tokens=20,
            latency_ms=100,
            created_at=datetime.utcnow(),
            metadata_={
                "source_count": 2,
            },
        )
        self.assertEqual(len(msg.sources), 2)
        self.assertEqual(msg.sources[0]["file_name"], "Knowledge Document")
        self.assertIsNotNone(msg.sources[0]["excerpt"])


if __name__ == "__main__":
    unittest.main()

