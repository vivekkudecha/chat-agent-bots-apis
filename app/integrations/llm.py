import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI

from app.config import settings

from app.core.exceptions import (
    ModelExecutionException,
)

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

    usage: LLMUsage = field(
        default_factory=LLMUsage
    )

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

class OpenAICompatibleProvider(
    LLMProvider
):

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
    ):

        base_url_normalized = base_url.rstrip("/")
        if base_url_normalized.endswith("/api"):
            base_url_normalized = base_url_normalized[:-4] + "/v1"
        elif not base_url_normalized.endswith("/v1"):
            base_url_normalized = f"{base_url_normalized}/v1"

        logger.info(f"Using OpenAI-compatible LLM provider at: {base_url_normalized}")

        http_client = httpx.AsyncClient(
            verify=settings.LLM_VERIFY_SSL,
            trust_env=settings.LLM_TRUST_ENV,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

        self.client = AsyncOpenAI(
            base_url=base_url_normalized,
            api_key=api_key,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=0,
            http_client=http_client,
        )

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages:
            return []

        system_parts: list[str] = []
        convo_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            if role == "system":
                system_parts.append(content)
            else:
                convo_messages.append({"role": role, "content": content})

        normalized: list[dict[str, Any]] = []

        if system_parts:
            normalized.append({
                "role": "system",
                "content": "\n\n".join(system_parts),
            })

        for msg in convo_messages:
            role = msg.get("role")
            content = msg.get("content")
            if normalized and normalized[-1]["role"] == role:
                prev = normalized[-1]["content"]
                normalized[-1]["content"] = f"{prev}\n\n{content}".strip()
            else:
                normalized.append({"role": role, "content": content})

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

            logger.debug("LLM chat request kwargs: %s", kwargs)

            if tools:
                kwargs["tools"] = tools

                kwargs["tool_choice"] = "auto"

            response = (
                await self.client
                .chat.completions.create(
                    **kwargs
                )
            )

            logger.debug("LLM chat response: %s", response)

            if not response.choices:
                raise ModelExecutionException(
                    "Model returned no choices."
                )

            choice = response.choices[0]

            content = (
                choice.message.content
                or ""
            )

            usage = LLMUsage()

            if response.usage:

                usage = LLMUsage(
                    input_tokens=(
                        response.usage
                        .prompt_tokens
                        or 0
                    ),
                    output_tokens=(
                        response.usage
                        .completion_tokens
                        or 0
                    ),
                    total_tokens=(
                        response.usage
                        .total_tokens
                        or 0
                    ),
                )

            return LLMResponse(
                content=content,
                model=response.model,
                finish_reason=(
                    choice.finish_reason
                ),
                usage=usage,
                raw=response,
            )

        except ModelExecutionException:
            raise

        except Exception as exc:
            logger.error("LLM request failed: %s", exc, exc_info=True)
            raise ModelExecutionException(
                f"LLM request failed: {exc}"
            ) from exc

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

            stream = (
                await self.client
                .chat.completions.create(
                    **kwargs
                )
            )

            async for chunk in stream:

                if not chunk.choices:
                    continue

                delta = (
                    chunk
                    .choices[0]
                    .delta
                )

                content = (
                    delta.content
                    or ""
                )

                if content:
                    yield content

        except Exception as exc:
            logger.error("LLM streaming request failed: %s", exc, exc_info=True)
            raise ModelExecutionException(
                f"LLM streaming request failed: {exc}"
            ) from exc


# =========================================================
# PROVIDER FACTORY
# =========================================================

_llm_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:

    global _llm_provider

    if _llm_provider is None:

        provider = (
            settings.LLM_PROVIDER
            .lower()
            .strip()
        )

        if provider in ("vllm", "local", "ollama"):

            _llm_provider = (
                OpenAICompatibleProvider(
                    base_url=(
                        settings.VLLM_BASE_URL
                    ),
                    api_key=(
                        settings.VLLM_API_KEY
                    ),
                )
            )

        else:

            raise RuntimeError(
                "Unsupported LLM provider: "
                f"{provider}"
            )

    return _llm_provider