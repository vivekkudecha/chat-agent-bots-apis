import logging
import re
from typing import Any

from app.ai.agent.state import RouteType
from app.ai.llm.provider import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Intelligent query router inspired by LangGraph Adaptive RAG.
    Classifies the user query to choose:
    - RouteType.DIRECT: Conversational replies, greetings, small talk (skips RAG & tools).
    - RouteType.RAG: Queries knowledge base documents when specific facts or policies are needed.
    - RouteType.TOOL: Invokes tool calling when an action tool is requested.
    """

    # Fast path for obvious conversational pleasantries across languages (0ms latency, 0 token waste)
    CONVERSATIONAL_FAST_PATH = re.compile(
        r"^(?:"
        # English
        r"hi|hello|hey|heya|hiya|howdy|good\s+(morning|afternoon|evening|day|night)|"
        r"thanks|thank\s+you|thx|appreciate\s+it|many\s+thanks|"
        r"ok|okay|k|got\s+it|understood|cool|great|awesome|perfect|sure|yes|no|yep|nope|alright|fine|"
        r"bye|goodbye|cya|see\s+ya|see\s+you|take\s+care|who\s+are\s+you|how\s+are\s+you|"
        r"what\s+is\s+your\s+name|what\s+can\s+you\s+do|"
        # Hindi / Devanagari
        r"नमस्ते|नमस्कार|धन्यवाद|अलविदा|शुक्रिया|हाँ|नहीं|ठीक\s+है|आप\s+कैसे\s+हैं|"
        # Gujarati
        r"નમસ્તે|નમસ્કાર|આભાર|આવજો|હા|ના|કેમ\s+છો|"
        # Spanish
        r"hola|gracias|adiós|buenos\s+días|buenas\s+tardes|buenas\s+noches|sí|no|de\s+nada|"
        # French
        r"bonjour|merci|au\s+revoir|salut|oui|non|bonne\s+journée|"
        # German
        r"hallo|danke|guten\s+tag|tschuess|tschüss|ja|nein|"
        # Chinese
        r"你好|谢谢|再见|好的|是|不是|您好|"
        # Arabic
        r"مرحبا|شكرا|مع\s+السلامة|نعم|لا|أهلا"
        r")[\s!.,?।。\n\r؟]*$",
        re.IGNORECASE | re.UNICODE,
    )

    ROUTER_SYSTEM_PROMPT = """You are an intent routing supervisor for a multilingual AI agent.
Analyze the user's latest message in ANY language (English, Hindi, Spanish, French, German, Gujarati, Chinese, Arabic, Japanese, etc.) and classify what is needed:

Available choices:
- "DIRECT": Greetings, pleasantries, small talk, general conversation, or meta-questions not requiring external documents or tools.
- "RAG": When the user asks for specific company policies, guidelines, documents, factual information, or domain rules that should be searched in the knowledge base.
- "TOOL": When the user specifically requests an action, computation, or tool execution.

Respond with ONLY one word: DIRECT, RAG, or TOOL."""

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm or get_llm_provider()

    async def route(
        self,
        *,
        message: str,
        has_kb: bool,
        available_tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> RouteType:
        """
        Determines the routing decision for the current turn.
        """
        cleaned = message.strip()
        if not cleaned:
            return RouteType.DIRECT

        # If bot has no KBs and no tools, it can only respond directly
        if not has_kb and not available_tools:
            return RouteType.DIRECT

        # Fast path check for greetings and small-talk
        if self.CONVERSATIONAL_FAST_PATH.match(cleaned):
            logger.info("AgentRouter: Fast-path matched conversational greeting '%s' -> DIRECT", cleaned)
            return RouteType.DIRECT

        # If tools or KBs are present, use the LLM intent router for adaptive routing
        if model:
            try:
                router_messages = [
                    {"role": "system", "content": self.ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"User query: {cleaned}"},
                ]

                decision_response = await self.llm.chat(
                    messages=router_messages,
                    model=model,
                    temperature=0.0,
                    max_tokens=10,
                )

                decision = decision_response.content.strip().upper()

                if "RAG" in decision and has_kb:
                    logger.info("AgentRouter: LLM routed '%s' -> RAG", cleaned[:40])
                    return RouteType.RAG

                if "TOOL" in decision and available_tools:
                    logger.info("AgentRouter: LLM routed '%s' -> TOOL", cleaned[:40])
                    return RouteType.TOOL

                if "DIRECT" in decision:
                    logger.info("AgentRouter: LLM routed '%s' -> DIRECT", cleaned[:40])
                    return RouteType.DIRECT

            except Exception as exc:
                logger.warning("AgentRouter: LLM routing failed, falling back to default heuristic: %s", exc)

        # Fallback heuristic: If bot has KB, route informational queries to RAG
        if has_kb:
            return RouteType.RAG

        return RouteType.DIRECT
