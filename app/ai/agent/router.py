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
        r"what(?:\'s|s|\s+is)\s+today(?:'s)?(?:\s+date)?|today(?:'s)?\s+date|what\s+date\s+is\s+today|what\s+is\s+the\s+date|what\s+day\s+is\s+it|what\s+time\s+is\s+it|current\s+(?:date|time|year)|what\s+year\s+is\s+(?:this|it)|"
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

    ROUTER_SYSTEM_PROMPT = """You are an intent routing supervisor for an AI agent.
Analyze the user's latest message and recent conversation context to classify what is needed:

Available choices:
- "DIRECT": Greetings, general conversation, date/time inquiries, or follow-up questions/clarifications that can be answered directly using the existing conversation context without fetching external data.
- "RAG": When the user asks for specific company policies, internal documents, or domain rules that should be searched in the knowledge base.
- "TOOL": When the user specifically requests an action, web search, current news, live facts, or external data lookup not already covered in the conversation.

Respond with ONLY one word: DIRECT, RAG, or TOOL."""

    # Pattern for quick web search intent detection
    SEARCH_INTENT_PATTERN = re.compile(
        r"(?:\b(?:search\s+(?:the\s+web|online|for|google)|look\s+up|latest\s+(?:movies|films|news|songs|releases)|current\s+events|who\s+won|stock\s+price|live\s+score|weather\s+in|what\s+(?:happened|happned)\s+(?:in|to|at|today|recently)|what(?:\'s|\s+is)\s+happening|breaking\s+news|today(?:'s)?\s+news|news\s+(?:today|about|in|on)|movies\s+(?:of\s+this\s+year|released\s+this\s+year|in\s+\d{4})|(?:new|top|best)\s+movies\s+of\s+this\s+year)\b)",
        re.IGNORECASE,
    )

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm or get_llm_provider()

    async def route(
        self,
        *,
        message: str,
        has_kb: bool,
        available_tools: list[dict[str, Any]] | None = None,
        model: str = "",
        history: list[Any] | None = None,
    ) -> RouteType:
        """
        Determines the routing decision for the current turn with conversation context awareness.
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

        # Fast path for explicit search intent when search tool is available
        has_web_search = any(
            t.get("function", {}).get("name") == "web_search"
            for t in (available_tools or [])
        )
        if has_web_search and self.SEARCH_INTENT_PATTERN.search(cleaned):
            logger.info("AgentRouter: Fast-path matched search intent '%s' -> TOOL", cleaned[:40])
            return RouteType.TOOL

        # If tools or KBs are present, use the LLM intent router for adaptive routing
        if model:
            try:
                user_msg_parts: list[str] = []
                if history:
                    recent_hist: list[str] = []
                    for msg in history[-3:]:
                        r = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
                        c = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                        if c and not c.strip().startswith("{"):
                            recent_hist.append(f"{r.capitalize()}: {c[:140]}")
                    if recent_hist:
                        user_msg_parts.append("Recent conversation context:\n" + "\n".join(recent_hist))

                user_msg_parts.append(f"User query: {cleaned}")
                user_prompt_text = "\n\n".join(user_msg_parts)

                router_messages = [
                    {"role": "system", "content": self.ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt_text},
                ]

                decision_response = await self.llm.chat(
                    messages=router_messages,
                    model=model,
                    temperature=0.0,
                    max_tokens=10,
                )

                decision = decision_response.content.strip().upper()

                if "DIRECT" in decision:
                    logger.info("AgentRouter: LLM routed '%s' -> DIRECT", cleaned[:40])
                    return RouteType.DIRECT

                if "RAG" in decision and has_kb:
                    logger.info("AgentRouter: LLM routed '%s' -> RAG", cleaned[:40])
                    return RouteType.RAG

                if "TOOL" in decision and available_tools:
                    logger.info("AgentRouter: LLM routed '%s' -> TOOL", cleaned[:40])
                    return RouteType.TOOL

            except Exception as exc:
                logger.warning("AgentRouter: LLM routing failed, falling back to default heuristic: %s", exc)

        # Fallback heuristic: If bot has KB, route informational queries to RAG
        if has_kb:
            return RouteType.RAG

        if available_tools:
            return RouteType.TOOL

        return RouteType.DIRECT
