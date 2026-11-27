"""Data-Aware Router and LLM upstream provider adapters (OpenAI, Anthropic, GigaChat, YandexGPT)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import httpx

from pokrov.core.config import RouterConfig, UpstreamConfig
from pokrov.core.contracts import (
    Action,
    PolicyDecision,
    ProviderType,
    StreamFrame,
    UpstreamAdapter,
    UpstreamRequest,
    UpstreamResponse,
)
from pokrov.proxy.mock_llm import MockLLMAdapter, extract_block_text

logger = logging.getLogger("pokrov.proxy.router")

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

DEFAULT_BASE_URLS: dict[ProviderType, str] = {
    ProviderType.OPENAI: "https://api.openai.com/v1",
    ProviderType.OLLAMA: "http://127.0.0.1:11434/v1",
    ProviderType.VLLM: "http://127.0.0.1:8000/v1",
    ProviderType.ANTHROPIC: "https://api.anthropic.com",
    ProviderType.GIGACHAT: "https://gigachat.devices.sberbank.ru/api/v1",
    ProviderType.YANDEXGPT: "https://llm.api.cloud.yandex.net/foundationModels/v1",
}

_OPENAI_ALLOWED_EXTRA_KEYS = frozenset(
    {
        "top_p",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "n",
        "response_format",
        "seed",
        "tool_choice",
        "parallel_tool_calls",
        "stream_options",
        "reasoning_effort",
        "user",
        "metadata",
    }
)

_ANTHROPIC_ALLOWED_EXTRA_KEYS = frozenset(
    {
        "top_p",
        "top_k",
        "stop_sequences",
        "metadata",
        "tool_choice",
        "thinking",
    }
)

_GIGACHAT_ALLOWED_EXTRA_KEYS = frozenset(
    {
        "top_p",
        "n",
        "repetition_penalty",
        "update_interval",
        "function_call",
        "functions",
    }
)


def _as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` as a typed dict if it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def _build_ssl_verify(config: UpstreamConfig) -> ssl.SSLContext | bool:
    """Build TLS verification setting from ``UpstreamConfig`` gracefully."""
    if not config.tls_verify:
        return False
    if config.ca_bundle_path:
        bundle_path = Path(config.ca_bundle_path)
        if bundle_path.is_dir():
            return ssl.create_default_context(capath=str(bundle_path))
        if bundle_path.is_file():
            return ssl.create_default_context(cafile=str(bundle_path))
        logger.warning(
            "ca_bundle_path=%r not found for upstream %r; falling back to system CA store",
            config.ca_bundle_path,
            config.name,
        )
        return ssl.create_default_context()
    return True


def _create_async_client(config: UpstreamConfig) -> httpx.AsyncClient:
    """Create an ``httpx.AsyncClient`` configured for an upstream endpoint."""
    verify = _build_ssl_verify(config)
    return httpx.AsyncClient(
        verify=verify,
        timeout=httpx.Timeout(config.timeout_seconds),
    )


def _normalize_openai_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI or Anthropic tool definitions into OpenAI function tool schema."""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(dict(tool))
        elif "name" in tool:
            fn_obj: dict[str, Any] = {"name": tool["name"]}
            if "description" in tool:
                fn_obj["description"] = tool["description"]
            params = tool.get("input_schema") or tool.get("parameters")
            if isinstance(params, dict):
                fn_obj["parameters"] = params
            normalized.append({"type": "function", "function": fn_obj})
        else:
            normalized.append(dict(tool))
    return normalized


def _normalize_anthropic_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic or OpenAI tool definitions into Anthropic Messages API schema."""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if "input_schema" in tool and "name" in tool:
            normalized.append(dict(tool))
        elif tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            entry: dict[str, Any] = {
                "name": str(fn.get("name") or "tool"),
                "input_schema": (
                    fn["parameters"]
                    if isinstance(fn.get("parameters"), dict)
                    else {"type": "object", "properties": {}}
                ),
            }
            if isinstance(fn.get("description"), str):
                entry["description"] = fn["description"]
            normalized.append(entry)
        else:
            normalized.append(dict(tool))
    return normalized


def _normalize_gigachat_functions(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI or Anthropic tools into Sber GigaChat ``functions`` list."""
    functions: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            entry: dict[str, Any] = {
                "name": str(fn.get("name") or "function"),
                "parameters": (
                    fn["parameters"]
                    if isinstance(fn.get("parameters"), dict)
                    else {"type": "object", "properties": {}}
                ),
            }
            if isinstance(fn.get("description"), str):
                entry["description"] = fn["description"]
            functions.append(entry)
        elif "name" in tool:
            params = tool.get("parameters") or tool.get("input_schema")
            entry = {
                "name": str(tool["name"]),
                "parameters": (
                    params if isinstance(params, dict) else {"type": "object", "properties": {}}
                ),
            }
            if isinstance(tool.get("description"), str):
                entry["description"] = tool["description"]
            functions.append(entry)
    return functions


def _normalize_yandex_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI or Anthropic tools into YandexGPT ``tools`` list."""
    out: list[dict[str, Any]] = []
    for fn in _normalize_gigachat_functions(tools):
        out.append({"function": fn})
    return out


def _to_openai_messages(request: UpstreamRequest) -> list[dict[str, Any]]:
    """Convert ``UpstreamRequest`` messages into OpenAI Chat Completions format."""
    messages: list[dict[str, Any]] = []
    has_system = any(msg.get("role") in {"system", "developer"} for msg in request.messages)
    sys_text = request.system or (
        extract_block_text(request.system_blocks) if request.system_blocks else ""
    )
    if sys_text and not has_system:
        messages.append({"role": "system", "content": sys_text})

    for msg in request.messages:
        raw_role = str(msg.get("role", "user"))
        role = "system" if raw_role == "developer" else raw_role
        content = msg.get("content", "")
        if request.api_format == "anthropic" and isinstance(content, list):
            has_tool_blocks = any(
                isinstance(blk, dict) and blk.get("type") in {"tool_result", "tool_use", "thinking"}
                for blk in content
            )
            if has_tool_blocks:
                converted_content: Any = extract_block_text(content)
            else:
                converted_content = content
            messages.append({"role": role, "content": converted_content})
        else:
            copied = dict(msg)
            copied["role"] = role
            messages.append(copied)

    return messages


def _to_gigachat_messages(request: UpstreamRequest) -> list[dict[str, Any]]:
    """Convert ``UpstreamRequest`` messages into Sber GigaChat compatible format."""
    messages: list[dict[str, Any]] = []
    has_system = any(msg.get("role") in {"system", "developer"} for msg in request.messages)
    sys_text = request.system or (
        extract_block_text(request.system_blocks) if request.system_blocks else ""
    )
    if sys_text and not has_system:
        messages.append({"role": "system", "content": sys_text})

    for msg in request.messages:
        raw_role = str(msg.get("role", "user")).lower()
        if raw_role == "developer":
            role = "system"
        elif raw_role == "tool":
            role = "function"
        elif raw_role in {"system", "user", "assistant", "function"}:
            role = raw_role
        else:
            role = "user"

        raw_content = msg.get("content")
        flat_content = extract_block_text(raw_content) if raw_content is not None else ""
        entry: dict[str, Any] = {"role": role, "content": flat_content}
        if role == "function":
            fn_name = msg.get("name") or msg.get("tool_call_id") or "tool"
            entry["name"] = str(fn_name)
        messages.append(entry)

    return messages


def _to_anthropic_payload_messages(
    request: UpstreamRequest,
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Extract top-level ``system`` (preserving ``cache_control`` blocks) and valid messages."""
    extra_sys_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    for msg in request.messages:
        raw_role = str(msg.get("role", "user")).lower()
        content = msg.get("content")
        if raw_role in {"system", "developer"}:
            sys_text = extract_block_text(content)
            if sys_text:
                extra_sys_parts.append(sys_text)
            continue

        if raw_role == "tool":
            tool_use_id = str(msg.get("tool_call_id") or "tool")
            tr_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": extract_block_text(content),
            }
            if (
                messages
                and messages[-1].get("role") == "user"
                and isinstance(messages[-1].get("content"), list)
                and all(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in messages[-1]["content"]
                )
            ):
                messages[-1]["content"].append(tr_block)
            else:
                messages.append({"role": "user", "content": [tr_block]})
            continue

        role = "assistant" if raw_role == "assistant" else "user"
        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                blocks.extend(b for b in content if isinstance(b, dict))
            for tc in msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = _as_dict(tc.get("function"))
                raw_args = fn.get("arguments", "{}")
                parsed_input: dict[str, Any] = {}
                if isinstance(raw_args, dict):
                    parsed_input = raw_args
                elif isinstance(raw_args, str) and raw_args.strip():
                    try:
                        loaded = json.loads(raw_args)
                        parsed_input = loaded if isinstance(loaded, dict) else {"value": loaded}
                    except json.JSONDecodeError:
                        parsed_input = {"raw": raw_args}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tc.get("id") or "toolu_pokrov"),
                        "name": str(fn.get("name") or tc.get("name") or "tool"),
                        "input": parsed_input,
                    }
                )
            messages.append(
                {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}
            )
            continue

        safe_content: Any = content if content is not None else ""
        messages.append({"role": role, "content": safe_content})

    system_out: str | list[dict[str, Any]] | None
    if request.system_blocks:
        sys_blocks = [dict(b) for b in request.system_blocks]
        for extra in extra_sys_parts:
            sys_blocks.append({"type": "text", "text": extra})
        system_out = sys_blocks
    else:
        parts: list[str] = []
        if request.system:
            parts.append(request.system)
        parts.extend(extra_sys_parts)
        system_out = "\n\n".join(parts) if parts else None

    return system_out, messages


def _to_yandex_messages(request: UpstreamRequest) -> list[dict[str, str]]:
    """Convert ``UpstreamRequest`` messages into YandexGPT ``{role, text}`` format."""
    messages: list[dict[str, str]] = []
    has_system = any(msg.get("role") in {"system", "developer"} for msg in request.messages)
    sys_text = request.system or (
        extract_block_text(request.system_blocks) if request.system_blocks else ""
    )
    if sys_text and not has_system:
        messages.append({"role": "system", "text": sys_text})

    for msg in request.messages:
        raw_role = str(msg.get("role", "user")).lower()
        if raw_role == "developer":
            role = "system"
        elif raw_role == "tool":
            role = "user"
        elif raw_role in {"system", "user", "assistant"}:
            role = raw_role
        else:
            role = "user"
        text = extract_block_text(msg.get("content", ""))
        messages.append({"role": role, "text": text})

    return messages


def _extract_openai_text_from_payload(data: dict[str, Any]) -> str:
    """Extract assistant reply text from an OpenAI-compatible response payload."""
    choices = data.get("choices") or []
    if not choices or not isinstance(choices, list):
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return extract_block_text(content)
    text_val = first.get("text")
    return text_val if isinstance(text_val, str) else ""


def _extract_openai_sse_frames(payload: dict[str, Any]) -> list[StreamFrame]:
    """Extract structured ``StreamFrame`` items from a parsed OpenAI-compatible SSE chunk."""
    frames: list[StreamFrame] = []
    usage = payload.get("usage")
    if isinstance(usage, dict):
        p_tok = usage.get("prompt_tokens")
        c_tok = usage.get("completion_tokens")
        if p_tok is not None or c_tok is not None:
            frames.append(
                StreamFrame(
                    kind="usage",
                    prompt_tokens=int(p_tok) if p_tok is not None else None,
                    completion_tokens=int(c_tok) if c_tok is not None else None,
                )
            )

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return frames
    first = choices[0]
    if not isinstance(first, dict):
        return frames

    delta = first.get("delta")
    if isinstance(delta, dict):
        reasoning = delta.get("reasoning_content") or delta.get("thinking")
        if isinstance(reasoning, str) and reasoning:
            frames.append(StreamFrame(kind="thinking", text=reasoning, index=0))

        content = delta.get("content")
        if isinstance(content, str) and content:
            frames.append(StreamFrame(kind="text", text=content, index=0))

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_idx = int(tc.get("index", 0) or 0)
                tc_id = str(tc.get("id") or "")
                fn = _as_dict(tc.get("function"))
                fn_name = str(fn.get("name") or "")
                fn_args = str(fn.get("arguments") or "")
                frames.append(
                    StreamFrame(
                        kind="tool_call",
                        index=tc_idx,
                        tool_call_id=tc_id,
                        tool_name=fn_name,
                        arguments_delta=fn_args,
                    )
                )

        fn_call = delta.get("function_call")
        if isinstance(fn_call, dict):
            frames.append(
                StreamFrame(
                    kind="tool_call",
                    index=0,
                    tool_call_id="call_0",
                    tool_name=str(fn_call.get("name") or ""),
                    arguments_delta=str(fn_call.get("arguments") or ""),
                )
            )
    else:
        text_val = first.get("text")
        if isinstance(text_val, str) and text_val:
            frames.append(StreamFrame(kind="text", text=text_val, index=0))

    finish_reason = first.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        frames.append(StreamFrame(kind="finish", finish_reason=finish_reason))

    return frames


async def _iter_openai_sse_frames(response: httpx.Response) -> AsyncIterator[StreamFrame]:
    """Yield structured ``StreamFrame`` items from an OpenAI-compatible SSE ``httpx.Response``."""
    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str:
            continue
        if data_str == "[DONE]":
            break
        chunk = json.loads(data_str)
        if isinstance(chunk, dict):
            for frame in _extract_openai_sse_frames(chunk):
                yield frame


def _parse_openai_response(
    *,
    provider: ProviderType,
    upstream_name: str,
    fallback_model: str,
    response: httpx.Response,
) -> UpstreamResponse:
    """Parse a JSON OpenAI-compatible ``/chat/completions`` response."""
    data: dict[str, Any] = response.json()
    usage = _as_dict(data.get("usage"))
    return UpstreamResponse(
        provider=provider,
        model=str(data.get("model") or fallback_model),
        upstream_name=upstream_name,
        content=_extract_openai_text_from_payload(data),
        raw_payload=data,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        status_code=response.status_code,
    )


class _BaseHttpAdapter:
    """Shared lifecycle and configuration helpers for ``httpx``-backed adapters."""

    def __init__(
        self,
        config: UpstreamConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client if client is not None else _create_async_client(config)

    @property
    def name(self) -> str:
        """Configured upstream identifier."""
        return self._config.name

    @property
    def provider(self) -> ProviderType:
        """Upstream provider type."""
        return self._config.provider

    @property
    def is_local(self) -> bool:
        """Whether this upstream is inside the local/trusted contour."""
        return self._config.is_local

    @property
    def config(self) -> UpstreamConfig:
        """Return the immutable upstream configuration."""
        return self._config

    def _base_url(self) -> str:
        """Return normalized base URL without trailing slash."""
        raw = self._config.base_url or DEFAULT_BASE_URLS.get(self._config.provider, "")
        return raw.rstrip("/")

    def _resolve_env(self, env_name: str) -> str:
        """Read and trim an environment variable when configured."""
        if not env_name:
            return ""
        return os.environ.get(env_name, "").strip()

    def _resolve_model(self, request: UpstreamRequest, default_fallback: str = "") -> str:
        """Resolve outgoing provider model, honoring ``force_adapter_model`` and omitted default."""
        if request.force_adapter_model or request.model in {"", "pokrov-default"}:
            return self._config.model or default_fallback or request.model
        return request.model or self._config.model or default_fallback

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` if owned by this adapter."""
        if self._owns_client:
            await self._client.aclose()


class OpenAICompatAdapter(_BaseHttpAdapter):
    """Adapter for OpenAI-compatible endpoints (OpenAI, Ollama ``/v1``, vLLM ``/v1``)."""

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = self._resolve_env(self._config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(self, request: UpstreamRequest, *, stream: bool) -> dict[str, Any]:
        model = self._resolve_model(request)
        payload: dict[str, Any] = {
            k: v for k, v in request.extra_body.items() if k in _OPENAI_ALLOWED_EXTRA_KEYS
        }
        if "stop_sequences" in request.extra_body and "stop" not in payload:
            payload["stop"] = request.extra_body["stop_sequences"]
        payload["model"] = model
        payload["messages"] = _to_openai_messages(request)
        payload["stream"] = stream
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools is not None:
            payload["tools"] = _normalize_openai_tools(request.tools)
        return payload

    async def complete(self, request: UpstreamRequest) -> UpstreamResponse:
        """Send a non-streaming ``POST {base_url}/chat/completions`` request."""
        url = f"{self._base_url()}/chat/completions"
        payload = self._build_payload(request, stream=False)
        response = await self._client.post(url, json=payload, headers=self._headers())
        response.raise_for_status()
        return _parse_openai_response(
            provider=self._config.provider,
            upstream_name=self._config.name,
            fallback_model=str(payload["model"]),
            response=response,
        )

    async def stream_frames(self, request: UpstreamRequest) -> AsyncIterator[StreamFrame]:
        """Stream structured ``StreamFrame`` events from OpenAI-compatible SSE."""
        url = f"{self._base_url()}/chat/completions"
        payload = self._build_payload(request, stream=True)
        async with self._client.stream(
            "POST",
            url,
            json=payload,
            headers=self._headers(),
        ) as response:
            response.raise_for_status()
            async for frame in _iter_openai_sse_frames(response):
                yield frame

    async def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]:
        """Stream assistant text tokens via OpenAI SSE ``data: ...`` frames."""
        async for frame in self.stream_frames(request):
            if frame.kind == "text" and frame.text:
                yield frame.text


class AnthropicAdapter(_BaseHttpAdapter):
    """Adapter for Anthropic Messages API (``POST {base_url}/v1/messages``)."""

    def _endpoint(self) -> str:
        base = self._base_url()
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
        }
        api_key = self._resolve_env(self._config.api_key_env)
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def _build_payload(self, request: UpstreamRequest, *, stream: bool) -> dict[str, Any]:
        model = self._resolve_model(request, "claude-3-5-sonnet-latest")
        system_value, messages = _to_anthropic_payload_messages(request)
        payload: dict[str, Any] = {
            k: v
            for k, v in request.extra_body.items()
            if k in _ANTHROPIC_ALLOWED_EXTRA_KEYS and k != "mcp_servers"
        }
        if "stop" in request.extra_body and "stop_sequences" not in payload:
            raw_stop = request.extra_body["stop"]
            if isinstance(raw_stop, str):
                payload["stop_sequences"] = [raw_stop]
            elif isinstance(raw_stop, list):
                payload["stop_sequences"] = [str(s) for s in raw_stop]
        payload["model"] = model
        payload["messages"] = messages
        payload["max_tokens"] = request.max_tokens if request.max_tokens is not None else 1024
        payload["stream"] = stream
        if system_value is not None:
            payload["system"] = system_value
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools is not None:
            payload["tools"] = _normalize_anthropic_tools(request.tools)
        return payload

    async def complete(self, request: UpstreamRequest) -> UpstreamResponse:
        """Send a non-streaming ``POST {base_url}/v1/messages`` request."""
        url = self._endpoint()
        payload = self._build_payload(request, stream=False)
        response = await self._client.post(url, json=payload, headers=self._headers())
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        raw_content = data.get("content")
        if isinstance(raw_content, str):
            text = raw_content
        elif isinstance(raw_content, list):
            parts = [
                str(block.get("text", ""))
                for block in raw_content
                if isinstance(block, dict)
                and block.get("type", "text") == "text"
                and isinstance(block.get("text"), str)
            ]
            text = "".join(parts)
        else:
            text = ""

        usage = _as_dict(data.get("usage"))
        return UpstreamResponse(
            provider=ProviderType.ANTHROPIC,
            model=str(data.get("model") or payload["model"]),
            upstream_name=self._config.name,
            content=text,
            raw_payload=data,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            status_code=response.status_code,
        )

    async def stream_frames(self, request: UpstreamRequest) -> AsyncIterator[StreamFrame]:
        """Stream structured ``StreamFrame`` events from Anthropic Messages SSE."""
        url = self._endpoint()
        payload = self._build_payload(request, stream=True)
        block_meta: dict[int, tuple[str, str, str]] = {}

        async with self._client.stream(
            "POST",
            url,
            json=payload,
            headers=self._headers(),
        ) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                if data_str == "[DONE]":
                    break
                event_data = json.loads(data_str)
                if not isinstance(event_data, dict):
                    continue
                event_type = event_data.get("type")
                if event_type == "message_stop":
                    break
                if event_type == "message_start":
                    msg = _as_dict(event_data.get("message"))
                    usage = _as_dict(msg.get("usage"))
                    if "input_tokens" in usage or "output_tokens" in usage:
                        yield StreamFrame(
                            kind="usage",
                            prompt_tokens=(
                                int(usage["input_tokens"])
                                if usage.get("input_tokens") is not None
                                else None
                            ),
                            completion_tokens=(
                                int(usage["output_tokens"])
                                if usage.get("output_tokens") is not None
                                else None
                            ),
                        )
                elif event_type == "content_block_start":
                    idx = int(event_data.get("index", 0) or 0)
                    cb = _as_dict(event_data.get("content_block"))
                    cb_type = str(cb.get("type") or "text")
                    if cb_type == "tool_use":
                        tc_id = str(cb.get("id") or "")
                        tc_name = str(cb.get("name") or "")
                        block_meta[idx] = ("tool_use", tc_id, tc_name)
                        init_input = cb.get("input")
                        init_delta = (
                            json.dumps(init_input, ensure_ascii=False)
                            if isinstance(init_input, dict) and init_input
                            else ""
                        )
                        yield StreamFrame(
                            kind="tool_call",
                            index=idx,
                            tool_call_id=tc_id,
                            tool_name=tc_name,
                            arguments_delta=init_delta,
                        )
                    elif cb_type == "thinking":
                        block_meta[idx] = ("thinking", "", "")
                        init_think = cb.get("thinking")
                        if isinstance(init_think, str) and init_think:
                            yield StreamFrame(kind="thinking", index=idx, text=init_think)
                    else:
                        block_meta[idx] = ("text", "", "")
                        init_text = cb.get("text")
                        if isinstance(init_text, str) and init_text:
                            yield StreamFrame(kind="text", index=idx, text=init_text)
                elif event_type == "content_block_delta":
                    idx = int(event_data.get("index", 0) or 0)
                    delta = _as_dict(event_data.get("delta"))
                    delta_type = delta.get("type")
                    if delta_type in (None, "text_delta"):
                        text_val = delta.get("text")
                        if isinstance(text_val, str) and text_val:
                            yield StreamFrame(kind="text", index=idx, text=text_val)
                    elif delta_type == "input_json_delta":
                        pjson = delta.get("partial_json")
                        if isinstance(pjson, str):
                            _, tc_id, tc_name = block_meta.get(idx, ("tool_use", "", ""))
                            yield StreamFrame(
                                kind="tool_call",
                                index=idx,
                                tool_call_id=tc_id,
                                tool_name=tc_name,
                                arguments_delta=pjson,
                            )
                    elif delta_type == "thinking_delta":
                        think_val = delta.get("thinking")
                        if isinstance(think_val, str) and think_val:
                            yield StreamFrame(kind="thinking", index=idx, text=think_val)
                elif event_type == "message_delta":
                    delta = _as_dict(event_data.get("delta"))
                    usage = _as_dict(event_data.get("usage"))
                    if "output_tokens" in usage or "input_tokens" in usage:
                        yield StreamFrame(
                            kind="usage",
                            prompt_tokens=(
                                int(usage["input_tokens"])
                                if usage.get("input_tokens") is not None
                                else None
                            ),
                            completion_tokens=(
                                int(usage["output_tokens"])
                                if usage.get("output_tokens") is not None
                                else None
                            ),
                        )
                    stop_reason = delta.get("stop_reason")
                    if isinstance(stop_reason, str) and stop_reason:
                        yield StreamFrame(kind="finish", finish_reason=stop_reason)

    async def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]:
        """Stream assistant text from Anthropic ``content_block_delta`` SSE events."""
        async for frame in self.stream_frames(request):
            if frame.kind == "text" and frame.text:
                yield frame.text


class GigaChatAdapter(_BaseHttpAdapter):
    """Adapter for Sber GigaChat API with stampede-safe OAuth caching and custom CA support."""

    def __init__(
        self,
        config: UpstreamConfig,
        client: httpx.AsyncClient | None = None,
        *,
        auth_url: str | None = None,
        scope: str | None = None,
    ) -> None:
        super().__init__(config, client=client)
        self._auth_url = (
            auth_url or os.environ.get("GIGACHAT_AUTH_URL", "").strip() or DEFAULT_GIGACHAT_AUTH_URL
        )
        self._scope = (
            scope or os.environ.get("GIGACHAT_SCOPE", "").strip() or DEFAULT_GIGACHAT_SCOPE
        )
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    async def _ensure_access_token(
        self,
        *,
        force_refresh: bool = False,
        failed_token: str | None = None,
    ) -> str:
        """Acquire or return a cached GigaChat OAuth bearer token without refresh stampedes."""
        async with self._token_lock:
            now = time.time()
            if self._access_token is not None and now < self._token_expires_at - 30.0:
                if not force_refresh:
                    return self._access_token
                if failed_token is not None and self._access_token != failed_token:
                    return self._access_token

            credentials = self._resolve_env(self._config.api_key_env)
            headers: dict[str, str] = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
            }
            if credentials:
                if credentials.startswith(("Basic ", "Bearer ")):
                    headers["Authorization"] = credentials
                else:
                    headers["Authorization"] = f"Basic {credentials}"

            response = await self._client.post(
                self._auth_url,
                headers=headers,
                data={"scope": self._scope},
            )
            response.raise_for_status()
            token_data: dict[str, Any] = response.json()
            access_token = str(token_data["access_token"])

            if "expires_at" in token_data and token_data["expires_at"] is not None:
                raw_exp = float(token_data["expires_at"])
                expires_at = raw_exp / 1000.0 if raw_exp > 1e12 else raw_exp
            elif "expires_in" in token_data and token_data["expires_in"] is not None:
                expires_at = time.time() + float(token_data["expires_in"])
            else:
                expires_at = time.time() + 1800.0

            self._access_token = access_token
            self._token_expires_at = expires_at
            return access_token

    def _build_payload(self, request: UpstreamRequest, *, stream: bool) -> dict[str, Any]:
        model = self._resolve_model(request, "GigaChat")
        payload: dict[str, Any] = {
            k: v for k, v in request.extra_body.items() if k in _GIGACHAT_ALLOWED_EXTRA_KEYS
        }
        payload["model"] = model
        payload["messages"] = _to_gigachat_messages(request)
        payload["stream"] = stream
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools and "functions" not in payload:
            funcs = _normalize_gigachat_functions(request.tools)
            if funcs:
                payload["functions"] = funcs
        return payload

    def _chat_headers(self, access_token: str, *, accept: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": accept,
            "Authorization": f"Bearer {access_token}",
            "X-Request-ID": str(uuid.uuid4()),
        }

    async def complete(self, request: UpstreamRequest) -> UpstreamResponse:
        """Send a non-streaming ``POST {base_url}/chat/completions`` request to GigaChat."""
        url = f"{self._base_url()}/chat/completions"
        payload = self._build_payload(request, stream=False)

        token = await self._ensure_access_token()
        response = await self._client.post(
            url,
            json=payload,
            headers=self._chat_headers(token, accept="application/json"),
        )
        if response.status_code == 401:
            token = await self._ensure_access_token(force_refresh=True, failed_token=token)
            response = await self._client.post(
                url,
                json=payload,
                headers=self._chat_headers(token, accept="application/json"),
            )
        response.raise_for_status()
        return _parse_openai_response(
            provider=ProviderType.GIGACHAT,
            upstream_name=self._config.name,
            fallback_model=str(payload["model"]),
            response=response,
        )

    async def stream_frames(self, request: UpstreamRequest) -> AsyncIterator[StreamFrame]:
        """Stream structured ``StreamFrame`` events from GigaChat SSE."""
        url = f"{self._base_url()}/chat/completions"
        payload = self._build_payload(request, stream=True)
        prev_token: str | None = None

        for attempt in range(2):
            token = await self._ensure_access_token(
                force_refresh=(attempt == 1),
                failed_token=prev_token,
            )
            prev_token = token
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers=self._chat_headers(token, accept="text/event-stream"),
            ) as response:
                if response.status_code == 401 and attempt == 0:
                    continue
                response.raise_for_status()
                async for frame in _iter_openai_sse_frames(response):
                    yield frame
                return

    async def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]:
        """Stream assistant text from GigaChat ``/chat/completions`` SSE frames."""
        async for frame in self.stream_frames(request):
            if frame.kind == "text" and frame.text:
                yield frame.text


class YandexGPTAdapter(_BaseHttpAdapter):
    """Adapter for Yandex Cloud Foundation Models (``POST {base_url}/completion``)."""

    def _resolve_folder_id(self) -> str:
        if self._config.folder_id_env:
            return self._resolve_env(self._config.folder_id_env)
        return os.environ.get("YANDEX_FOLDER_ID", "").strip()

    def _resolve_auth_header(self) -> str:
        secret = self._resolve_env(self._config.api_key_env)
        is_iam_env = "iam" in self._config.api_key_env.lower()
        if not secret and not self._config.api_key_env:
            secret = os.environ.get("YANDEX_API_KEY", "").strip()
            if not secret:
                secret = os.environ.get("YANDEX_IAM_TOKEN", "").strip()
                is_iam_env = bool(secret)

        if not secret:
            return ""
        if secret.startswith(("Api-Key ", "Bearer ")):
            return secret
        if secret.startswith("t1.") or is_iam_env:
            return f"Bearer {secret}"
        return f"Api-Key {secret}"

    def _headers(self, folder_id: str) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        auth_header = self._resolve_auth_header()
        if auth_header:
            headers["Authorization"] = auth_header
        if folder_id:
            headers["x-folder-id"] = folder_id
        return headers

    def _resolve_model_uri(self, request: UpstreamRequest, folder_id: str) -> str:
        raw_model = self._resolve_model(request, "yandexgpt/latest")
        if raw_model.startswith(("gpt://", "ds://")):
            return raw_model
        if folder_id:
            return f"gpt://{folder_id}/{raw_model}"
        return raw_model

    def _build_payload(
        self,
        request: UpstreamRequest,
        *,
        folder_id: str,
        stream: bool,
    ) -> dict[str, Any]:
        completion_options: dict[str, Any] = {"stream": stream}
        if request.temperature is not None:
            completion_options["temperature"] = request.temperature
        if request.max_tokens is not None:
            completion_options["maxTokens"] = request.max_tokens

        raw_top_p = request.extra_body.get("top_p")
        if raw_top_p is not None:
            completion_options["topP"] = raw_top_p

        raw_stop = request.extra_body.get("stop") or request.extra_body.get("stop_sequences")
        if raw_stop is not None:
            if isinstance(raw_stop, str):
                completion_options["stopSequences"] = [raw_stop]
            elif isinstance(raw_stop, list):
                completion_options["stopSequences"] = [str(s) for s in raw_stop]

        payload: dict[str, Any] = {
            "modelUri": self._resolve_model_uri(request, folder_id),
            "completionOptions": completion_options,
            "messages": _to_yandex_messages(request),
        }
        if request.tools:
            y_tools = _normalize_yandex_tools(request.tools)
            if y_tools:
                payload["tools"] = y_tools
        return payload

    async def complete(self, request: UpstreamRequest) -> UpstreamResponse:
        """Send a non-streaming ``POST {base_url}/completion`` request to YandexGPT."""
        folder_id = self._resolve_folder_id()
        url = f"{self._base_url()}/completion"
        payload = self._build_payload(request, folder_id=folder_id, stream=False)
        response = await self._client.post(
            url,
            json=payload,
            headers=self._headers(folder_id),
        )
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        raw_result = data.get("result")
        result = raw_result if isinstance(raw_result, dict) else data
        alternatives = result.get("alternatives") or []
        text = ""
        if isinstance(alternatives, list) and alternatives:
            first_alt = alternatives[0]
            if isinstance(first_alt, dict):
                msg = first_alt.get("message")
                if isinstance(msg, dict):
                    text = str(msg.get("text") or msg.get("content") or "")

        usage = _as_dict(result.get("usage"))
        prompt_tokens = int(usage.get("inputTextTokens") or usage.get("prompt_tokens") or 0)
        completion_tokens = int(
            usage.get("completionTokens") or usage.get("completion_tokens") or 0
        )

        effective_model = self._resolve_model(request, str(payload["modelUri"]))
        return UpstreamResponse(
            provider=ProviderType.YANDEXGPT,
            model=effective_model,
            upstream_name=self._config.name,
            content=text,
            raw_payload=data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status_code=response.status_code,
        )

    async def stream_frames(self, request: UpstreamRequest) -> AsyncIterator[StreamFrame]:
        """Stream structured ``StreamFrame`` events from YandexGPT NDJSON/SSE chunks."""
        folder_id = self._resolve_folder_id()
        url = f"{self._base_url()}/completion"
        payload = self._build_payload(request, folder_id=folder_id, stream=True)
        emitted = ""
        finish_reason = "stop"

        async with self._client.stream(
            "POST",
            url,
            json=payload,
            headers=self._headers(folder_id),
        ) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    break

                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    continue
                raw_chunk_result = chunk.get("result")
                result = raw_chunk_result if isinstance(raw_chunk_result, dict) else chunk

                usage = result.get("usage")
                if isinstance(usage, dict):
                    p_tok = usage.get("inputTextTokens") or usage.get("prompt_tokens")
                    c_tok = usage.get("completionTokens") or usage.get("completion_tokens")
                    if p_tok is not None or c_tok is not None:
                        yield StreamFrame(
                            kind="usage",
                            prompt_tokens=int(p_tok) if p_tok is not None else None,
                            completion_tokens=int(c_tok) if c_tok is not None else None,
                        )

                alternatives = result.get("alternatives")
                if not isinstance(alternatives, list) or not alternatives:
                    continue
                first_alt = alternatives[0]
                if not isinstance(first_alt, dict):
                    continue
                status_val = first_alt.get("status")
                if isinstance(status_val, str) and "TRUNCATED" in status_val:
                    finish_reason = "length"
                msg = first_alt.get("message")
                if not isinstance(msg, dict):
                    continue

                tool_call_list = msg.get("toolCallList")
                if isinstance(tool_call_list, dict) and isinstance(
                    tool_call_list.get("toolCalls"), list
                ):
                    for tc_idx, tc in enumerate(tool_call_list["toolCalls"]):
                        if isinstance(tc, dict) and isinstance(tc.get("functionCall"), dict):
                            fn_c = tc["functionCall"]
                            args_obj = fn_c.get("arguments", {})
                            yield StreamFrame(
                                kind="tool_call",
                                index=tc_idx,
                                tool_call_id=f"ycall_{tc_idx}",
                                tool_name=str(fn_c.get("name") or ""),
                                arguments_delta=(
                                    json.dumps(args_obj, ensure_ascii=False)
                                    if isinstance(args_obj, dict)
                                    else str(args_obj)
                                ),
                            )
                            finish_reason = "tool_calls"

                text = msg.get("text")
                if not isinstance(text, str):
                    continue

                has_status = isinstance(status_val, str)
                if has_status:
                    delta = text[len(emitted) :] if text.startswith(emitted) else text
                    emitted = text
                elif emitted and len(text) > len(emitted) and text.startswith(emitted):
                    delta = text[len(emitted) :]
                    emitted = text
                else:
                    delta = text
                    emitted += delta

                if delta:
                    yield StreamFrame(kind="text", text=delta, index=0)

            yield StreamFrame(kind="finish", finish_reason=finish_reason)

    async def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]:
        """Stream assistant text from YandexGPT NDJSON or SSE ``completion`` chunks."""
        async for frame in self.stream_frames(request):
            if frame.kind == "text" and frame.text:
                yield frame.text


def _instantiate_adapter(
    ucfg: UpstreamConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> UpstreamAdapter:
    """Instantiate the concrete ``UpstreamAdapter`` for an ``UpstreamConfig``."""
    match ucfg.provider:
        case ProviderType.MOCK:
            return MockLLMAdapter(
                name=ucfg.name,
                model=ucfg.model or "pokrov-mock-llm",
            )
        case ProviderType.OPENAI | ProviderType.OLLAMA | ProviderType.VLLM:
            return OpenAICompatAdapter(ucfg, client=client)
        case ProviderType.ANTHROPIC:
            return AnthropicAdapter(ucfg, client=client)
        case ProviderType.GIGACHAT:
            return GigaChatAdapter(ucfg, client=client)
        case ProviderType.YANDEXGPT:
            return YandexGPTAdapter(ucfg, client=client)


class DataAwareRouter:
    """Routes LLM requests between cloud and local/Russian upstreams based on policy."""

    def __init__(
        self,
        config: RouterConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or RouterConfig()
        self._upstream_configs: dict[str, UpstreamConfig] = dict(self._config.upstreams)
        self._adapters: dict[str, UpstreamAdapter] = {}
        self._model_to_upstream: dict[str, str] = {}

        for name, ucfg in self._upstream_configs.items():
            self._adapters[name] = _instantiate_adapter(ucfg, client=client)
            if ucfg.model and ucfg.model not in self._model_to_upstream:
                self._model_to_upstream[ucfg.model] = name

        if "mock-local" not in self._adapters:
            mock_cfg = UpstreamConfig(
                name="mock-local",
                provider=ProviderType.MOCK,
                model="pokrov-mock-llm",
                is_local=True,
            )
            self._upstream_configs["mock-local"] = mock_cfg
            self._adapters["mock-local"] = MockLLMAdapter(
                name="mock-local",
                model="pokrov-mock-llm",
            )
            self._model_to_upstream.setdefault("pokrov-mock-llm", "mock-local")

    @property
    def config(self) -> RouterConfig:
        """Return router configuration."""
        return self._config

    def get_adapter(self, name: str) -> UpstreamAdapter:
        """Return the adapter registered under ``name`` or raise ``KeyError``."""
        if name not in self._adapters:
            raise KeyError(f"Unknown upstream adapter: {name!r}")
        return self._adapters[name]

    def _resolve_named_adapter(self, candidate: str) -> UpstreamAdapter | None:
        """Resolve an upstream by configured name or configured model name."""
        if candidate in self._adapters:
            return self._adapters[candidate]
        mapped_name = self._model_to_upstream.get(candidate)
        if mapped_name is not None:
            return self._adapters.get(mapped_name)
        return None

    def _select_local_adapter(self) -> UpstreamAdapter:
        """Pick the configured local upstream, falling back to any local or ``mock-local``."""
        configured = self._adapters.get(self._config.local_upstream)
        if configured is not None and configured.is_local:
            return configured
        for adapter in self._adapters.values():
            if adapter.is_local:
                return adapter
        return self._adapters["mock-local"]

    def select_upstream(
        self,
        decision: PolicyDecision,
        *,
        requested_upstream: str | None = None,
        allow_cloud: bool = True,
    ) -> UpstreamAdapter:
        """Select the target upstream according to ``PolicyDecision``, residency, and preference."""
        if decision.action == Action.ROUTE_LOCAL or decision.target_upstream is not None:
            if decision.target_upstream is not None:
                target = self._resolve_named_adapter(decision.target_upstream)
                if target is not None and (
                    target.is_local or (allow_cloud and decision.action != Action.ROUTE_LOCAL)
                ):
                    return target
            return self._select_local_adapter()

        if not allow_cloud:
            if requested_upstream is not None:
                req_local = self._resolve_named_adapter(requested_upstream)
                if req_local is not None and req_local.is_local:
                    return req_local
            default_local = self._adapters.get(self._config.default_upstream)
            if default_local is not None and default_local.is_local:
                return default_local
            return self._select_local_adapter()

        if requested_upstream is not None:
            requested = self._resolve_named_adapter(requested_upstream)
            if requested is not None:
                is_inferred_model = requested_upstream not in self._adapters
                if (
                    is_inferred_model
                    and self._config.default_upstream == "mock-local"
                    and isinstance(self._adapters.get("mock-local"), MockLLMAdapter)
                ):
                    ucfg = self._upstream_configs.get(requested.name)
                    if (
                        ucfg is not None
                        and ucfg.api_key_env
                        and not os.environ.get(ucfg.api_key_env, "").strip()
                    ):
                        return self._adapters["mock-local"]
                return requested

        default_adapter = self._adapters.get(self._config.default_upstream)
        if default_adapter is not None:
            return default_adapter
        return self._adapters["mock-local"]

    def list_models(self) -> list[dict[str, Any]]:
        """Return configured models in OpenAI ``/v1/models`` list item format."""
        models: list[dict[str, Any]] = []
        for name, adapter in self._adapters.items():
            ucfg = self._upstream_configs[name]
            model_id = ucfg.model or name
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": adapter.provider.value,
                    "upstream": name,
                    "is_local": adapter.is_local,
                }
            )
        return models

    async def aclose(self) -> None:
        """Close all owned HTTP adapter clients."""
        for adapter in self._adapters.values():
            close_fn = getattr(adapter, "aclose", None)
            if callable(close_fn):
                await close_fn()


__all__ = [
    "AnthropicAdapter",
    "DataAwareRouter",
    "GigaChatAdapter",
    "OpenAICompatAdapter",
    "YandexGPTAdapter",
]
