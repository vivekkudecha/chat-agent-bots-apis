import json
import logging
import re
import uuid
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
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


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
            tool_calls = message.get("tool_calls")

            if not content and not tool_calls and role != "tool":
                continue

            if role == "system":
                system_parts.append(content)
            elif role == "tool":
                conversation.append(message)
            elif role in ("user", "assistant"):
                if tool_calls:
                    conversation.append(message)
                elif (
                    conversation
                    and conversation[-1]["role"] == role
                    and not conversation[-1].get("tool_calls")
                    and conversation[-1].get("role") != "tool"
                ):
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
            if (
                merged_convo
                and merged_convo[-1]["role"] == msg["role"]
                and msg["role"] in ("user", "assistant")
                and not merged_convo[-1].get("tool_calls")
                and not msg.get("tool_calls")
            ):
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

    @staticmethod
    def _extract_tool_calls_from_content(
        content: str, tools: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Fallback parser for open-source models (Llama 3.2, Gemma, Qwen) running
        on Ollama/vLLM that output function calls as JSON in the message content
        instead of populating tool_calls in the response choice.
        Handles doubled quotation marks (tokenizer artifacts like ""parameters""),
        markdown code fences, and text-embedded tool call JSON objects.
        """
        if not content:
            return []

        text = content.strip()

        # Strip markdown code blocks if wrapped
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        # Strip <tool_call> tags if wrapped
        text = re.sub(r"</?tool_call>", "", text).strip()

        # Normalize doubled quotes (e.g. Ollama/Llama 3.2 tokenizer artifact: ""parameters"")
        normalized = re.sub(r'""+', '"', text)

        # Build set of expected tool names if available
        known_tool_names = set()
        if tools:
            for t in tools:
                fn = t.get("function", {})
                if fn.get("name"):
                    known_tool_names.add(fn["name"])
        if not known_tool_names:
            known_tool_names.add("web_search")

        def try_parse_candidate(obj_str: str) -> dict | list | None:
            try:
                val = json.loads(obj_str)
                if isinstance(val, dict):
                    return val
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    return val
            except Exception:
                pass
            return None

        # 1. Try parsing the whole normalized string
        parsed = try_parse_candidate(normalized)

        # 2. If direct parse fails, try extracting a JSON block or object
        if not parsed:
            match = re.search(r"(\[\s*\{.*?\}\s*\]|\{.*?\})", normalized, re.DOTALL)
            if match:
                parsed = try_parse_candidate(match.group(1))

        if not parsed:
            return []

        def to_tool_call(item: dict) -> dict[str, Any] | None:
            name = item.get("name") or item.get("tool") or item.get("function")
            if not name or not isinstance(name, str):
                return None
            if known_tool_names and name not in known_tool_names:
                return None
            args = item.get("parameters") or item.get("arguments") or item.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            elif not isinstance(args, dict):
                args = {}

            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args,
                },
            }

        results: list[dict[str, Any]] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    tc = to_tool_call(item)
                    if tc:
                        results.append(tc)
        elif isinstance(parsed, dict):
            tc = to_tool_call(parsed)
            if tc:
                results.append(tc)

        return results

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

            # Extract tool calls if returned by model
            raw_tool_calls = getattr(choice.message, "tool_calls", None)
            parsed_tool_calls: list[dict[str, Any]] = []
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                    fn = getattr(tc, "function", None)
                    fn_name = getattr(fn, "name", "") if fn else ""
                    fn_args_raw = getattr(fn, "arguments", "{}") if fn else "{}"
                    if isinstance(fn_args_raw, str):
                        try:
                            fn_args = json.loads(fn_args_raw)
                        except Exception:
                            fn_args = {"raw": fn_args_raw}
                    elif isinstance(fn_args_raw, dict):
                        fn_args = fn_args_raw
                    else:
                        fn_args = {}

                    parsed_tool_calls.append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": fn_args,
                            },
                        }
                    )
            elif content:
                # Fallback: Extract tool calls from text if model outputs JSON in content
                extracted_calls = self._extract_tool_calls_from_content(content, tools)
                if extracted_calls:
                    parsed_tool_calls = extracted_calls
                    content = ""  # Clean raw JSON so it is not shown to user

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
                tool_calls=parsed_tool_calls,
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
