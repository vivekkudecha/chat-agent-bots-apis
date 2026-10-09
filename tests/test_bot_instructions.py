import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

# The application creates its Qdrant client at import time. Keep tests offline.
with patch("qdrant_client.AsyncQdrantClient"):
    from app.ai.agent.graph import ChatAgentGraph
    from app.ai.llm.prompt_builder import PromptBuilderService
    from app.ai.llm.provider import LLMResponse
    from app.ai.rag.clarify import FollowUpSuggester
    from app.ai.rag.retrieval import RetrievalResult, RetrievedChunk


INSTRUCTION = "Always reply in Hindi. Only discuss HR policies. Keep answers under 80 words."


def build(instruction=INSTRUCTION, chunks=()):
    retrieval = RetrievalResult("q", list(chunks), [uuid.UUID(int=1)]) if chunks else None
    return PromptBuilderService().build(
        bot_version=SimpleNamespace(system_instruction=instruction, metadata_={}),
        user_message="What is the leave policy?",
        history=[],
        retrieval=retrieval,
        memory_context=None,
        context_window=8192,
        max_generation_tokens=512,
        tools=None,
        tool_results=None,
    )


def chunk():
    return RetrievedChunk(id="c1", text="Employees get 20 days of leave. " * 20, score=0.8,
                          document_id=uuid.uuid4(), knowledge_base_id=uuid.UUID(int=1),
                          page=1, file_name="leave.pdf")


def system_texts(messages):
    return [m["content"] for m in messages if m["role"] == "system"]


class PromptPriorityTests(unittest.TestCase):
    def test_instructions_are_mandatory_and_restated_last(self):
        messages = build(chunks=[chunk()]).messages
        systems = system_texts(messages)

        self.assertTrue(any(t.startswith("BOT INSTRUCTIONS (MANDATORY)") and INSTRUCTION in t for t in systems))
        # Reminder is the final system message, after the knowledge context.
        self.assertTrue(systems[-1].startswith(PromptBuilderService.BOT_REMINDER_PREFIX))
        self.assertIn(INSTRUCTION, systems[-1])
        knowledge = next(i for i, t in enumerate(systems) if t.startswith("KNOWLEDGE CONTEXT"))
        self.assertLess(knowledge, len(systems) - 1)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is the leave policy?"})

    def test_no_forced_english_over_bot_language(self):
        prompt = "\n".join(system_texts(build().messages))
        self.assertNotIn("respond in English only", prompt)
        self.assertNotIn("ALWAYS communicate and respond in English", prompt)
        self.assertIn("unless the BOT INSTRUCTIONS", prompt)

    def test_long_instructions_are_referenced_not_repeated(self):
        long_text = "Rule. " * 300
        reminder = system_texts(build(long_text).messages)[-1]
        self.assertTrue(reminder.startswith(PromptBuilderService.BOT_REMINDER_PREFIX))
        self.assertNotIn(long_text.strip(), reminder)

    def test_no_reminder_without_instructions(self):
        systems = system_texts(build("").messages)
        self.assertFalse(any(t.startswith(PromptBuilderService.BOT_REMINDER_PREFIX) for t in systems))

    def test_tool_grounding_keeps_reminder_last(self):
        messages = ChatAgentGraph._inject_grounding_messages(
            build().messages, system_instruction="TOOL RULES", grounding_content="RESULTS",
        )
        systems = system_texts(messages)
        self.assertTrue(systems[-1].startswith(PromptBuilderService.BOT_REMINDER_PREFIX))
        self.assertIn("TOOL RULES", systems)


class SuggesterInstructionTests(unittest.IsolatedAsyncioTestCase):
    async def test_clarify_and_related_prompts_carry_bot_instructions(self):
        llm = SimpleNamespace(chat=AsyncMock(side_effect=[
            LLMResponse(content='{"question": "किस बारे में?", "suggestions": ["छुट्टी नीति क्या है?"]}', model="m"),
            LLMResponse(content='{"suggestions": []}', model="m"),
        ]))
        suggester = FollowUpSuggester(llm)
        topics = [{"file_name": "leave.pdf", "section": "Leave"}]

        await suggester.clarify(question="leave", topics=topics, model="m", bot_instruction=INSTRUCTION)
        await suggester.related(question="q", answer="a", topics=topics, model="m", bot_instruction=INSTRUCTION)

        for call in llm.chat.await_args_list:
            prompt = call.kwargs["messages"][1]["content"]
            self.assertTrue(prompt.startswith("BOT INSTRUCTIONS (mandatory"))
            self.assertIn(INSTRUCTION, prompt)


if __name__ == "__main__":
    unittest.main()
