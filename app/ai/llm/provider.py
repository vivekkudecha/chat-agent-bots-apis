import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI

from app.config import settings
from app.core.exceptions import ModelExecutionException

logger = logging.getLogger(__name__)


# =========================================================
# RESPONSE
# =========================================================

@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    model: str
    finish_reason: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: Any | None = None


# =========================================================
# PROVIDER INTERFACE
# =========================================================

class LLMProvider(ABC):

    @abstractmethod
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def stream_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        pass


# =========================================================
# OPENAI-COMPATIBLE PROVIDER
# =========================================================

class OpenAICompatibleProvider(LLMProvider):

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        verify_ssl: bool | None = None,
        trust_env: bool | None = None,
        timeout: float | None = None,
    ):
        base_url_normalized = base_url.rstrip("/")
        if base_url_normalized.endswith("/api"):
            base_url_normalized = base_url_normalized[:-4] + "/v1"

        elif not base_url_normalized.endswith("/v1"):
            base_url_normalized = f"{base_url_normalized}/v1"

        self.verify_ssl = (
            settings.LLM_VERIFY_SSL if verify_ssl is None else verify_ssl
        )
        self.trust_env = (
            settings.LLM_TRUST_ENV if trust_env is None else trust_env
        )
        self.timeout = (
            settings.LLM_TIMEOUT_SECONDS if timeout is None else timeout
        )

        http_client = httpx.AsyncClient(
            verify=self.verify_ssl,
            trust_env=self.trust_env,
            timeout=self.timeout,
        )

        self.client = AsyncOpenAI(
            base_url=base_url_normalized,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not messages:
            return []

        system_parts: list[str] = []
        conversation: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")
            content = (message.get("content") or "").strip()

            if not content:
                continue

            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                # Merge consecutive identical roles
                if conversation and conversation[-1]["role"] == role:
                    prev = conversation[-1]["content"]
                    conversation[-1]["content"] = f"{prev}\n\n{content}".strip()
                else:
                    conversation.append({"role": role, "content": content})
            else:
                conversation.append({"role": role, "content": content})

        # Chat models (like Gemma, Llama, etc.) require:
        # 1. The first conversational message after system MUST be 'user'.
        # If conversation starts with 'assistant' (e.g. from history budget cutoff),
        # drop leading assistant messages until the first user message.
        while conversation and conversation[0]["role"] == "assistant":
            conversation.pop(0)

        # Re-merge any consecutive identical roles that may have resulted
        merged_convo: list[dict[str, Any]] = []
        for msg in conversation:
            if merged_convo and merged_convo[-1]["role"] == msg["role"]:
                prev = merged_convo[-1]["content"]
                merged_convo[-1]["content"] = f"{prev}\n\n{msg['content']}".strip()
            else:
                merged_convo.append(msg)

        normalized: list[dict[str, Any]] = []
        if system_parts:
            normalized.append({
                "role": "system",
                "content": "\n\n".join(system_parts),
            })

        normalized.extend(merged_convo)
        return normalized


    # =====================================================
    # NORMAL CHAT
    # =====================================================

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        try:

            normalized_messages = self._normalize_messages(messages)

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": normalized_messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**kwargs)

            if not response.choices:
                raise ModelExecutionException("Model returned no choices.")

            choice = response.choices[0]
            content = choice.message.content or ""
            usage = LLMUsage()

            raw_usage = getattr(response, "usage", None)
            if raw_usage:
                if isinstance(raw_usage, dict):
                    prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
                    completion_tokens = int(raw_usage.get("completion_tokens") or 0)
                    total_tokens = int(raw_usage.get("total_tokens") or 0)
                else:
                    prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)
                    total_tokens = int(getattr(raw_usage, "total_tokens", 0) or 0)

                if total_tokens == 0 and (prompt_tokens + completion_tokens) > 0:
                    total_tokens = prompt_tokens + completion_tokens

                usage = LLMUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            return LLMResponse(
                content=content,
                model=response.model,
                finish_reason=choice.finish_reason,
                usage=usage,
                raw=response,
            )

        except ModelExecutionException:
            raise

        except Exception as exc:
            logger.exception("LLM request failed: %s", exc)
            err_msg = str(exc)
            if hasattr(exc, "message") and exc.message:
                err_msg = exc.message
            raise ModelExecutionException(f"LLM request failed: {err_msg}") from exc

    # =====================================================
    # STREAMING
    # =====================================================

    async def stream_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        try:

            normalized_messages = self._normalize_messages(messages)

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": normalized_messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            stream = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content = delta.content or ""
                if content:
                    yield content

        except Exception as exc:
            logger.exception("LLM streaming request failed: %s", exc)
            err_msg = str(exc)
            if hasattr(exc, "message") and exc.message:
                err_msg = exc.message
            raise ModelExecutionException(f"LLM streaming request failed: {err_msg}") from exc


# =========================================================
# PROVIDER FACTORY
# =========================================================

_llm_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _llm_provider

    if _llm_provider is None:
        provider = settings.LLM_PROVIDER.lower().strip()

        if provider in ("vllm", "local", "ollama"):
            _llm_provider = OpenAICompatibleProvider(
                base_url=settings.VLLM_BASE_URL,
                api_key=settings.VLLM_API_KEY,
                # verify_ssl=settings.LLM_VERIFY_SSL,
                # trust_env=settings.LLM_TRUST_ENV,
                # timeout=settings.LLM_TIMEOUT_SECONDS,
            )
        else:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")

    return _llm_provider
