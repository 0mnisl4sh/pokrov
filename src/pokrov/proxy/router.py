"""Data-Aware Router and LLM upstream provider adapters (OpenAI, Anthropic, GigaChat, YandexGPT)."""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from pokrov.core.config import RouterConfig, UpstreamConfig
from pokrov.core.contracts import (
    Action,
    PolicyDecision,
    ProviderType,
    UpstreamAdapter,
    UpstreamRequest,
    UpstreamResponse,
)
from pokrov.proxy.mock_llm import MockLLMAdapter, extract_block_text

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


def _build_ssl_verify(config: UpstreamConfig) -> ssl.SSLContext | bool:
    """Build TLS verification setting from ``UpstreamConfig`` without deprecation warnings."""
    if not config.tls_verify:
        return False
    if config.ca_bundle_path:
        bundle_path = Path(config.ca_bundle_path)
        if bundle_path.is_dir():
            return ssl.create_default_context(capath=str(bundle_path))
        return ssl.create_default_context(cafile=str(bundle_path))
    return True


def _create_async_client(config: UpstreamConfig) -> httpx.AsyncClient:
    """Create an ``httpx.AsyncClient`` configured for an upstream endpoint."""
    verify = _build_ssl_verify(config)
    return httpx.AsyncClient(
        verify=verify,
        timeout=httpx.Timeout(config.timeout_seconds),
    )


def _to_openai_messages(request: UpstreamRequest) -> list[dict[str, Any]]:
    """Convert ``UpstreamRequest`` messages into OpenAI Chat Completions format."""
    messages: list[dict[str, Any]] = []
    has_system = any(msg.get("role") == "system" for msg in request.messages)
    if request.system and not has_system:
        messages.append({"role": "system", "content": request.system})

    for msg in request.messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if request.api_format == "anthropic" and isinstance(content, list):
            has_tool_blocks = any(
                isinstance(blk, dict) and blk.get("type") in {"tool_result", "tool_use"}
                for blk in content
            )
            if has_tool_blocks:
                converted_content: Any = extract_block_text(content)
            else:
                converted_content = content
            messages.append({"role": role, "content": converted_content})
        else:
            messages.append(dict(msg))

    return messages


def _to_anthropic_payload_messages(
    request: UpstreamRequest,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract top-level ``system`` prompt and Anthropic-compatible messages."""
    system_parts: list[str] = []
    if request.system:
        system_parts.append(request.system)

    messages: list[dict[str, Any]] = []
    for msg in request.messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if role == "system":
            sys_text = extract_block_text(content)
            if sys_text:
                system_parts.append(sys_text)
        elif role == "tool":
            tool_use_id = str(msg.get("tool_call_id") or "tool")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": extract_block_text(content),
                        }
                    ],
                }
            )
        else:
            messages.append({"role": role, "content": content})

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, messages


def _to_yandex_messages(request: UpstreamRequest) -> list[dict[str, str]]:
    """Convert ``UpstreamRequest`` messages into YandexGPT ``{role, text}`` format."""
    messages: list[dict[str, str]] = []
    has_system = any(msg.get("role") == "system" for msg in request.messages)
    if request.system and not has_system:
        messages.append({"role": "system", "text": request.system})

    for msg in request.messages:
        raw_role = str(msg.get("role", "user"))
        role = "user" if raw_role == "tool" else raw_role
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


def _extract_openai_sse_delta(payload: dict[str, Any]) -> str:
    """Extract delta text from a parsed OpenAI-compatible SSE chunk."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    text_val = first.get("text")
    return text_val if isinstance(text_val, str) else ""


async def _iter_openai_sse_deltas(response: httpx.Response) -> AsyncIterator[str]:
    """Yield text deltas from an OpenAI-compatible SSE ``httpx.Response``."""
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
            delta_text = _extract_openai_sse_delta(chunk)
            if delta_text:
                yield delta_text


def _as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` as a typed dict if it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


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
        model = request.model or self._config.model
        payload: dict[str, Any] = dict(request.extra_body)
        payload["model"] = model
        payload["messages"] = _to_openai_messages(request)
        payload["stream"] = stream
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools is not None:
            payload["tools"] = list(request.tools)
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

    async def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]:
        """Stream assistant text tokens via OpenAI SSE ``data: ...`` frames."""
        url = f"{self._base_url()}/chat/completions"
        payload = self._build_payload(request, stream=True)
        async with self._client.stream(
            "POST",
            url,
            json=payload,
            headers=self._headers(),
        ) as response:
            response.raise_for_status()
            async for delta_text in _iter_openai_sse_deltas(response):
                yield delta_text


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
        model = request.model or self._config.model
        system_prompt, messages = _to_anthropic_payload_messages(request)
        payload: dict[str, Any] = dict(request.extra_body)
        payload["model"] = model
        payload["messages"] = messages
        payload["max_tokens"] = request.max_tokens if request.max_tokens is not None else 1024
        payload["stream"] = stream
        if system_prompt is not None:
            payload["system"] = system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools is not None:
            payload["tools"] = list(request.tools)
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

    async def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]:
        """Stream assistant text from Anthropic ``content_block_delta`` SSE events."""
        url = self._endpoint()
        payload = self._build_payload(request, stream=True)
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
                if event_type == "content_block_delta":
                    delta = event_data.get("delta")
                    if isinstance(delta, dict):
                        delta_type = delta.get("type")
                        text_val = delta.get("text")
                        if (
                            delta_type in (None, "text_delta")
                            and isinstance(text_val, str)
                            and text_val
                        ):
                            yield text_val


class GigaChatAdapter(_BaseHttpAdapter):
    """Adapter for Sber GigaChat API with OAuth token caching and custom CA support."""

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

    async def _ensure_access_token(self, *, force_refresh: bool = False) -> str:
        """Acquire or return a cached GigaChat OAuth bearer token."""
        async with self._token_lock:
            now = time.time()
            if (
                not force_refresh
                and self._access_token is not None
                and now < self._token_expires_at - 30.0
            ):
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
        model = request.model or self._config.model or "GigaChat"
        payload: dict[str, Any] = dict(request.extra_body)
        payload["model"] = model
        payload["messages"] = _to_openai_messages(request)
        payload["stream"] = stream
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
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
            token = await self._ensure_access_token(force_refresh=True)
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

    async def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]:
        """Stream assistant text from GigaChat ``/chat/completions`` SSE frames."""
        url = f"{self._base_url()}/chat/completions"
        payload = self._build_payload(request, stream=True)

        for attempt in range(2):
            token = await self._ensure_access_token(force_refresh=(attempt == 1))
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers=self._chat_headers(token, accept="text/event-stream"),
            ) as response:
                if response.status_code == 401 and attempt == 0:
                    continue
                response.raise_for_status()
                async for delta_text in _iter_openai_sse_deltas(response):
                    yield delta_text
                return


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
        raw_model = request.model or self._config.model or "yandexgpt/latest"
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

        payload: dict[str, Any] = dict(request.extra_body)
        payload["modelUri"] = self._resolve_model_uri(request, folder_id)
        payload["completionOptions"] = completion_options
        payload["messages"] = _to_yandex_messages(request)
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

        return UpstreamResponse(
            provider=ProviderType.YANDEXGPT,
            model=request.model or self._config.model or str(payload["modelUri"]),
            upstream_name=self._config.name,
            content=text,
            raw_payload=data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status_code=response.status_code,
        )

    async def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]:
        """Stream assistant text from YandexGPT NDJSON or SSE ``completion`` chunks."""
        folder_id = self._resolve_folder_id()
        url = f"{self._base_url()}/completion"
        payload = self._build_payload(request, folder_id=folder_id, stream=True)
        emitted = ""

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
                alternatives = result.get("alternatives")
                if not isinstance(alternatives, list) or not alternatives:
                    continue
                first_alt = alternatives[0]
                if not isinstance(first_alt, dict):
                    continue
                msg = first_alt.get("message")
                if not isinstance(msg, dict):
                    continue
                text = msg.get("text")
                if not isinstance(text, str):
                    continue

                has_status = isinstance(first_alt.get("status"), str)
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
                    yield delta


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
    ) -> UpstreamAdapter:
        """Select the target upstream according to ``PolicyDecision`` and request preference."""
        if decision.action == Action.ROUTE_LOCAL or decision.target_upstream is not None:
            if decision.target_upstream is not None:
                target = self._resolve_named_adapter(decision.target_upstream)
                if target is not None and target.is_local:
                    return target
            return self._select_local_adapter()

        if requested_upstream is not None:
            requested = self._resolve_named_adapter(requested_upstream)
            if requested is not None:
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
