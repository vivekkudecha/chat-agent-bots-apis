import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

# The application creates its Qdrant client at import time. Keep tests offline.
with patch("qdrant_client.AsyncQdrantClient"):
    from app.ai.agent.graph import ChatAgentGraph
    from app.ai.agent.router import RoutingDecision
    from app.ai.agent.state import RouteType
    from app.ai.llm.provider import LLMResponse, LLMUsage
    from app.ai.rag.agentic import EvidenceGrader
    from app.ai.rag.clarify import FollowUpSuggester
    from app.ai.rag.retrieval import RetrievalResult, RetrievalService, RetrievedChunk
    from app.ai.rag.sparse import BM25SparseEncoder


USER = uuid.UUID(int=7)
KB = uuid.UUID(int=1)
TOPICS = [
    {"file_name": "Long_Service_Awards.pdf", "section": "LSA Policy > 5. LSA Leave Guidelines", "preview": "3 extra days"},
    {"file_name": "Long_Service_Awards.pdf", "section": "LSA Policy > 4. Recognition Architecture", "preview": "INR 15,000"},
]


def llm(*contents):
    return SimpleNamespace(chat=AsyncMock(side_effect=[
        LLMResponse(content=c, model="m", usage=LLMUsage(10, 5, 15)) for c in contents
    ]))


class SuggesterTests(unittest.IsolatedAsyncioTestCase):
    def test_clean_suggestions(self):
        cleaned = FollowUpSuggester.clean_suggestions(
            ["1. How many LSA leave days?", "- how many lsa leave days?", "LSA", "x" * 200, 7, "lsa", "What gift at 10 years?"],
            question="LSA",
        )
        self.assertEqual(cleaned, ["How many LSA leave days?", "What gift at 10 years?"])

    async def test_clarify_uses_llm_json_and_lists_topics(self):
        model = llm('{"question": "What about LSA?", "suggestions": ["How many LSA leave days do I get?", "What is the LSA gift?"]}')
        result = await FollowUpSuggester(model).clarify(question="LSA", topics=TOPICS, model="m")

        self.assertEqual(result.text, "What about LSA?\n\n- How many LSA leave days do I get?\n- What is the LSA gift?")
        prompt = model.chat.await_args.kwargs["messages"][1]["content"]
        self.assertIn("Long Service Awards › LSA Policy > 5. LSA Leave Guidelines: 3 extra days", prompt)
        self.assertEqual(model.chat.await_args.kwargs["reasoning_effort"], "none")
        self.assertEqual(result.usage.total_tokens, 15)

    async def test_clarify_falls_back_to_topic_templates(self):
        result = await FollowUpSuggester(llm("not json")).clarify(
            question="LSA", topics=TOPICS, model="m", found_nothing=True,
        )
        self.assertIn("couldn't find a direct answer", result.question)
        self.assertEqual(result.suggestions, [
            "What does Long Service Awards say about 5. LSA Leave Guidelines?",
            "What does Long Service Awards say about 4. Recognition Architecture?",
        ])

    async def test_clarify_without_topics_uses_titles_or_plain_question(self):
        model = llm('{"question": "What do you need?", "suggestions": ["What does the travel policy cover?"]}')
        result = await FollowUpSuggester(model).clarify(
            question="bonus", topics=[], model="m", document_titles=["Travel_Policy.pdf"],
        )
        self.assertEqual(result.suggestions, ["What does the travel policy cover?"])
        self.assertIn("1. Travel Policy", model.chat.await_args.kwargs["messages"][1]["content"])

        silent = llm()
        result = await FollowUpSuggester(silent).clarify(question="bonus", topics=[], model="m")
        silent.chat.assert_not_awaited()
        self.assertEqual(result.suggestions, [])
        self.assertIn('"bonus"', result.text)

    async def test_related_limits_to_three(self):
        model = llm('{"suggestions": ["Q one here?", "Q two here?", "Q three here?", "Q four here?"]}')
        suggestions, _ = await FollowUpSuggester(model).related(question="q", answer="a", topics=TOPICS, model="m")
        self.assertEqual(len(suggestions), 3)


class GraderAmbiguityTests(unittest.TestCase):
    def test_parses_ambiguous_flag(self):
        grade = EvidenceGrader.parse('{"relevant": [1], "answerable": "partial", "ambiguous": "true"}', passage_count=1)
        self.assertTrue(grade.ambiguous)
        grade = EvidenceGrader.parse('{"relevant": [1], "answerable": "yes"}', passage_count=1)
        self.assertFalse(grade.ambiguous)


def match(index, text, score, *, section, doc=uuid.UUID(int=3)):
    return {"id": f"{doc}-{index}", "score": score, "text": text, "document_id": str(doc),
            "knowledge_base_id": str(KB), "chunk_index": index, "page": 1, "page_end": 1,
            "section": section, "file_name": "awards.pdf", "metadata": {}}


class RelatedTopicTests(unittest.IsolatedAsyncioTestCase):
    async def test_near_misses_become_topics(self):
        store = SimpleNamespace(
            sparse_encoder=BM25SparseEncoder(),
            hybrid_search=AsyncMock(return_value=([
                match(1, "Leave days for long service.", 0.36, section="Awards > Leave"),
                match(2, "Gift box details.", 0.31, section="Awards > Gifts"),
                match(3, "More leave detail.", 0.30, section="Awards > Leave"),
                match(4, "Cafeteria opens at nine.", 0.12, section="Facilities"),
                match(5, "Sl. No. Name", 0.35, section="Awards > 8. Document Distribution List"),
            ], [])),
        )
        result = await RetrievalService(vector_store=store).retrieve(
            user_id=USER, knowledge_base_ids=[KB], query="LSA", score_threshold=0.4,
        )
        self.assertEqual(result.chunks, [])
        self.assertEqual([t["section"] for t in result.related_topics], ["Awards > Leave", "Awards > Gifts"])


def chunk(i, score=0.6):
    return RetrievedChunk(id=f"c{i}", text=f"text {i}", score=score, document_id=uuid.UUID(int=i),
                          knowledge_base_id=KB, chunk_index=0, page=1, file_name="lsa.pdf",
                          metadata={"pages": [1]})


class ClarifyGraphTests(unittest.IsolatedAsyncioTestCase):
    async def run_graph(self, *, retrievals, llm_replies, supervisor_route="rag", query="LSA",
                        web=False, probe=None, settings_patch=None):
        retrieval = SimpleNamespace(
            retrieve_for_bot=AsyncMock(side_effect=[*(probe or []), *retrievals]),
            encoder=BM25SparseEncoder(),
            vector_store=SimpleNamespace(list_document_titles=AsyncMock(return_value=["Travel.pdf"])),
        )
        model = llm(*llm_replies, "final answer")
        router = SimpleNamespace(decide=AsyncMock(return_value=RoutingDecision(
            route=RouteType(supervisor_route), reason="r", query=query)))
        tools = SimpleNamespace(execute_tool=AsyncMock(return_value={"results": []}))
        graph = ChatAgentGraph(router=router, retrieval_service=retrieval, prompt_builder=None,
                               llm_provider=model, tool_registry=tools)
        graph._context_builder_node = AsyncMock(return_value={})
        generated = {}

        async def generate(state):
            generated.update(state)
            return {"response": LLMResponse(content="Partial answer.", model="m")}

        graph._generate_node = generate
        graph.app = graph._build_graph()
        state = {"user_id": USER, "bot_id": uuid.uuid4(), "conversation_id": uuid.uuid4(), "query": query,
                 "has_kb": True, "db_session": object(), "model_key": "m", "enable_web_search": web,
                 "available_tools": [{"function": {"name": "web_search"}}] if web else []}
        with patch.multiple("app.ai.agent.graph.settings", RAG_MAX_ROUNDS=1, **(settings_patch or {})):
            out = await graph.run(state)
        return out, generated, retrieval, model, tools

    def result(self, chunks, topics=TOPICS):
        return RetrievalResult("LSA", chunks, [KB], diagnostics={"queries": ["LSA"]}, related_topics=topics)

    async def test_ambiguous_question_gets_clarification(self):
        out, generated, _, _, _ = await self.run_graph(
            retrievals=[self.result([chunk(1), chunk(2)])],
            llm_replies=['{"relevant": [1, 2], "answerable": "no", "ambiguous": true}',
                         '{"question": "Which LSA detail?", "suggestions": ["How many LSA leave days do I get?"]}'],
        )
        self.assertEqual(generated, {})
        self.assertTrue(out["needs_clarification"])
        self.assertEqual(out["suggestions"], ["How many LSA leave days do I get?"])
        self.assertIsNone(out["retrieval"])
        self.assertEqual(out["response"].finish_reason, "clarification")

    async def test_nothing_found_without_web_clarifies_with_near_miss_topics(self):
        out, _, _, model, _ = await self.run_graph(
            retrievals=[self.result([])],
            llm_replies=['{"question": "Did you mean one of these?", "suggestions": ["What is the LSA gift?"]}'],
        )
        self.assertTrue(out["needs_clarification"])
        prompt = model.chat.await_args_list[0].kwargs["messages"][1]["content"]
        self.assertIn("no passage that answers", prompt)
        self.assertIn("5. LSA Leave Guidelines", prompt)

    async def test_nothing_found_and_no_topics_uses_document_catalog(self):
        out, _, retrieval, model, _ = await self.run_graph(
            retrievals=[self.result([], topics=[])],
            llm_replies=['{"question": "I have travel info.", "suggestions": ["What does the travel policy cover?"]}'],
        )
        retrieval.vector_store.list_document_titles.assert_awaited_once()
        self.assertEqual(out["suggestions"], ["What does the travel policy cover?"])

    async def test_nothing_found_with_web_still_uses_web(self):
        out, _, _, _, tools = await self.run_graph(retrievals=[self.result([])], llm_replies=[], web=True)
        tools.execute_tool.assert_awaited_once()
        self.assertFalse(out.get("needs_clarification"))

    async def test_clarify_disabled_keeps_old_behaviour(self):
        out, generated, _, _, _ = await self.run_graph(
            retrievals=[self.result([])], llm_replies=[], settings_patch={"RAG_CLARIFY": False},
        )
        self.assertIn("query", generated)
        self.assertFalse(out.get("needs_clarification"))

    async def test_partial_answer_gets_followups(self):
        out, generated, _, _, _ = await self.run_graph(
            retrievals=[self.result([chunk(1)])],
            llm_replies=['{"relevant": [1], "answerable": "partial", "next_query": ""}',
                         '{"suggestions": ["What gift is given at 10 years?"]}'],
        )
        self.assertTrue(generated["suggest_followups"])
        self.assertEqual(out["suggestions"], ["What gift is given at 10 years?"])
        self.assertEqual(out["response"].content,
                         "Partial answer.\n\n**You might also ask:**\n- What gift is given at 10 years?")
        self.assertFalse(out.get("needs_clarification"))

    async def test_ambiguous_but_answered_gets_answer_and_followups(self):
        out, generated, _, _, _ = await self.run_graph(
            retrievals=[self.result([chunk(1)])],
            llm_replies=['{"relevant": [1], "answerable": "yes", "ambiguous": true}',
                         '{"suggestions": ["How many LSA leave days do I get?"]}'],
            query="leave",
        )
        self.assertIn("query", generated)
        self.assertFalse(out.get("needs_clarification"))
        self.assertEqual(out["suggestions"], ["How many LSA leave days do I get?"])

    async def test_keyword_query_answered_fully_still_gets_followups(self):
        out, _, _, _, _ = await self.run_graph(
            retrievals=[self.result([chunk(1)])],
            llm_replies=['{"relevant": [1], "answerable": "yes"}', '{"suggestions": ["What is the LSA gift?"]}'],
        )
        self.assertEqual(out["suggestions"], ["What is the LSA gift?"])

    async def test_full_answer_has_no_followups(self):
        out, _, _, model, _ = await self.run_graph(
            query="What gift does an employee get after 10 years?",
            retrievals=[self.result([chunk(1)])],
            llm_replies=['{"relevant": [1], "answerable": "yes"}'],
        )
        self.assertEqual(out["response"].content, "Partial answer.")
        self.assertEqual(model.chat.await_count, 1)

    async def test_short_direct_query_is_probed_and_rerouted(self):
        out, _, retrieval, _, _ = await self.run_graph(
            supervisor_route="direct",
            probe=[self.result([chunk(1)])],
            retrievals=[self.result([chunk(1)])],
            llm_replies=['{"relevant": [1], "answerable": "yes"}'],
        )
        self.assertEqual(out["route"], "rag")
        self.assertEqual(retrieval.retrieve_for_bot.await_args_list[0].kwargs["top_k"], 3)

    async def test_greeting_stays_direct_when_probe_finds_nothing(self):
        out, _, retrieval, _, _ = await self.run_graph(
            supervisor_route="direct", query="hello", probe=[self.result([])], retrievals=[], llm_replies=[],
        )
        self.assertEqual(out["route"], "direct")
        self.assertEqual(retrieval.retrieve_for_bot.await_count, 1)

    async def test_long_direct_query_is_not_probed(self):
        out, _, retrieval, _, _ = await self.run_graph(
            supervisor_route="direct", query="explain how photosynthesis converts light into chemical energy",
            retrievals=[], llm_replies=[],
        )
        self.assertEqual(out["route"], "direct")
        retrieval.retrieve_for_bot.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
