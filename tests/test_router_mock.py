"""Tests for MockLLMAdapter, upstream HTTP adapters, SSE parsing, and DataAwareRouter."""

from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from pokrov.core.config import RouterConfig, UpstreamConfig, load_config
from pokrov.core.contracts import (
    Action,
    PolicyDecision,
    ProviderType,
    UpstreamAdapter,
    UpstreamRequest,
)
from pokrov.proxy.mock_llm import (
    ECHOLEAK_PAYLOAD,
    ECHOLEAK_TRIGGER,
    MockLLMAdapter,
)
from pokrov.proxy.router import (
    AnthropicAdapter,
    DataAwareRouter,
    GigaChatAdapter,
    OpenAICompatAdapter,
    YandexGPTAdapter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FragmentedByteStream(httpx.AsyncByteStream):
    """Async byte stream yielding fixed-size byte fragments to test SSE reassembly."""

    def __init__(self, payload: bytes, chunk_bytes: int = 3) -> None:
        self._payload = payload
        self._chunk_bytes = chunk_bytes

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for idx in range(0, len(self._payload), self._chunk_bytes):
            yield self._payload[idx : idx + self._chunk_bytes]


def _write_self_signed_ca(cert_path: Path) -> None:
    """Create a valid self-signed CA PEM certificate inside ``tmp_path``."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Pokrov Test CA")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(private_key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def test_all_adapters_implement_upstream_adapter_protocol() -> None:
    """All 5 adapter classes must satisfy the runtime_checkable UpstreamAdapter Protocol."""
    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, json={}))
    )

    adapters: list[UpstreamAdapter] = [
        MockLLMAdapter(),
        OpenAICompatAdapter(
            UpstreamConfig(
                name="openai-test",
                provider=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
            ),
            client=mock_client,
        ),
        AnthropicAdapter(
            UpstreamConfig(
                name="anthropic-test",
                provider=ProviderType.ANTHROPIC,
                base_url="https://api.anthropic.com",
                model="claude-3-5-sonnet-latest",
            ),
            client=mock_client,
        ),
        GigaChatAdapter(
            UpstreamConfig(
                name="gigachat-test",
                provider=ProviderType.GIGACHAT,
                base_url="https://gigachat.devices.sberbank.ru/api/v1",
                model="GigaChat",
                is_local=True,
            ),
            client=mock_client,
        ),
        YandexGPTAdapter(
            UpstreamConfig(
                name="yandex-test",
                provider=ProviderType.YANDEXGPT,
                base_url="https://llm.api.cloud.yandex.net/foundationModels/v1",
                model="yandexgpt/latest",
                is_local=True,
            ),
            client=mock_client,
        ),
    ]

    for adapter in adapters:
        assert isinstance(adapter, UpstreamAdapter)


@pytest.mark.asyncio
async def test_mock_llm_adapter_complete_placeholders_and_streaming() -> None:
    """MockLLMAdapter preserves <PKR:...> placeholders, extracts user/tool, and fragments SSE."""
    with pytest.raises(ValueError, match="chunk_size"):
        MockLLMAdapter(chunk_size=0)

    adapter = MockLLMAdapter(name="mock-local", model="pokrov-mock-llm", chunk_size=2)
    assert adapter.name == "mock-local"
    assert adapter.provider == ProviderType.MOCK
    assert adapter.is_local is True

    req = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="pokrov-mock-llm",
        messages=(
            {"role": "system", "content": "Be safe"},
            {"role": "user", "content": "Первое сообщение"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Паспорт <PKR:RU_PASSPORT:a1b2c3d4> и СНИЛС <PKR:RU_SNILS:deadbeef>",
            },
        ),
    )

    resp = await adapter.complete(req)
    assert resp.status_code == 200
    assert resp.provider == ProviderType.MOCK
    assert resp.upstream_name == "mock-local"
    assert resp.model == "pokrov-mock-llm"
    assert "<PKR:RU_PASSPORT:a1b2c3d4>" in resp.content
    assert "<PKR:RU_SNILS:deadbeef>" in resp.content
    assert resp.content.startswith("Принято к обработке: ")
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0

    for chunk_size in (1, 2, 3):
        frag_adapter = MockLLMAdapter(chunk_size=chunk_size)
        chunks = [chunk async for chunk in frag_adapter.stream_text(req)]
        assert all(1 <= len(chunk) <= chunk_size for chunk in chunks)
        assert "".join(chunks) == resp.content

    # Anthropic tool_result structured content blocks
    tool_result_req = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="",
        api_format="anthropic",
        messages=(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "Token <PKR:EMAIL:11223344>"}],
                    }
                ],
            },
        ),
    )
    tool_resp = await adapter.complete(tool_result_req)
    assert tool_resp.model == "pokrov-mock-llm"
    assert "<PKR:EMAIL:11223344>" in tool_resp.content

    # EchoLeak simulation trigger in complete & stream_text
    leak_req = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="pokrov-mock-llm",
        messages=({"role": "user", "content": f"Summarize {ECHOLEAK_TRIGGER}"},),
    )
    leak_resp = await adapter.complete(leak_req)
    assert leak_resp.content == ECHOLEAK_PAYLOAD
    streamed_leak = "".join([c async for c in adapter.stream_text(leak_req)])
    assert streamed_leak == ECHOLEAK_PAYLOAD
    await adapter.aclose()


@pytest.mark.asyncio
async def test_openai_compat_adapter_openai_ollama_vllm_json_and_sse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAICompatAdapter handles OpenAI/Ollama/vLLM JSON, headers, cross-format, and SSE."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-123")
    captured: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append((str(request.url), dict(request.headers), body))
        if body.get("stream"):
            sse = (
                b": keep-alive\n\n"
                b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
                b'data: {"choices": [{"delta": {"content": "Hello "}}]}\n\n'
                b'data: {"choices": [{"delta": {"content": "Pokrov!"}}]}\n\n'
                b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_FragmentedByteStream(sse, chunk_bytes=3),
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK reply"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        openai_cfg = UpstreamConfig(
            name="openai-cloud",
            provider=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            is_local=False,
        )
        openai_adapter = OpenAICompatAdapter(openai_cfg, client=client)

        req = UpstreamRequest(
            provider=ProviderType.OPENAI,
            model="gpt-4o",
            system="You are helpful.",
            temperature=0.2,
            max_tokens=150,
            messages=({"role": "user", "content": "Hi"},),
        )
        resp = await openai_adapter.complete(req)
        assert resp.content == "OK reply"
        assert resp.prompt_tokens == 12
        assert resp.completion_tokens == 5
        assert resp.model == "gpt-4o"

        url, headers, payload = captured[-1]
        assert url == "https://api.openai.com/v1/chat/completions"
        assert headers["authorization"] == "Bearer sk-test-openai-123"
        assert payload["stream"] is False
        assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}

        chunks = [c async for c in openai_adapter.stream_text(req)]
        assert "".join(chunks) == "Hello Pokrov!"
        assert captured[-1][2]["stream"] is True

        # Ollama local without API key + Anthropic tool_result cross-format routing
        ollama_cfg = UpstreamConfig(
            name="ollama-local",
            provider=ProviderType.OLLAMA,
            base_url="http://127.0.0.1:11434/v1/",
            model="qwen2.5:7b",
            is_local=True,
        )
        ollama_adapter = OpenAICompatAdapter(ollama_cfg, client=client)
        ollama_req = UpstreamRequest(
            provider=ProviderType.OLLAMA,
            model="",
            api_format="anthropic",
            messages=(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "doc text"}
                    ],
                },
            ),
        )
        ollama_resp = await ollama_adapter.complete(ollama_req)
        assert ollama_resp.provider == ProviderType.OLLAMA
        assert ollama_resp.is_local if hasattr(ollama_resp, "is_local") else True
        assert ollama_adapter.is_local is True
        url, headers, payload = captured[-1]
        assert url == "http://127.0.0.1:11434/v1/chat/completions"
        assert "authorization" not in headers
        assert payload["model"] == "qwen2.5:7b"
        assert payload["messages"][0]["content"] == "doc text"

        # vLLM local
        vllm_cfg = UpstreamConfig(
            name="vllm-local",
            provider=ProviderType.VLLM,
            base_url="http://127.0.0.1:8000/v1",
            model="Qwen/Qwen2.5-14B-Instruct",
            is_local=True,
        )
        vllm_adapter = OpenAICompatAdapter(vllm_cfg, client=client)
        vllm_resp = await vllm_adapter.complete(
            UpstreamRequest(
                provider=ProviderType.VLLM,
                model="",
                messages=({"role": "user", "content": "ping"},),
            )
        )
        assert vllm_resp.provider == ProviderType.VLLM
        assert captured[-1][0] == "http://127.0.0.1:8000/v1/chat/completions"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_adapter_json_and_sse_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnthropicAdapter sends auth headers and parses JSON + content_block_delta SSE."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-999")
    seen_requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_requests.append((str(request.url), dict(request.headers), body))
        if body.get("stream"):
            sse = (
                b"event: message_start\n"
                b'data: {"type": "message_start", "message": {"id": "msg_1"}}\n\n'
                b"event: ping\n"
                b'data: {"type": "ping"}\n\n'
                b"event: content_block_delta\n"
                b'data: {"type": "content_block_delta", "index": 0, '
                b'"delta": {"type": "text_delta", "text": "Anthropic "}}\n\n'
                b"event: content_block_delta\n"
                b'data: {"type": "content_block_delta", "index": 0, '
                b'"delta": {"type": "text_delta", "text": "Stream"}}\n\n'
                b"event: message_stop\n"
                b'data: {"type": "message_stop"}\n\n'
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_FragmentedByteStream(sse, chunk_bytes=5),
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "Claude response"}],
                "usage": {"input_tokens": 21, "output_tokens": 9},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        cfg = UpstreamConfig(
            name="anthropic-cloud",
            provider=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com",
            model="claude-3-5-sonnet-latest",
            api_key_env="ANTHROPIC_API_KEY",
            is_local=False,
        )
        adapter = AnthropicAdapter(cfg, client=client)

        req = UpstreamRequest(
            provider=ProviderType.ANTHROPIC,
            model="claude-3-5-sonnet-latest",
            api_format="anthropic",
            system="Top system",
            messages=(
                {"role": "system", "content": "Second system"},
                {"role": "user", "content": "Check tool"},
                {"role": "tool", "tool_call_id": "call_9", "content": "tool out"},
            ),
        )
        resp = await adapter.complete(req)
        assert resp.provider == ProviderType.ANTHROPIC
        assert resp.content == "Claude response"
        assert resp.prompt_tokens == 21
        assert resp.completion_tokens == 9

        url, headers, payload = seen_requests[-1]
        assert url == "https://api.anthropic.com/v1/messages"
        assert headers["x-api-key"] == "sk-ant-test-999"
        assert headers["anthropic-version"] == "2023-06-01"
        assert payload["system"] == "Top system\n\nSecond system"
        assert payload["max_tokens"] == 1024
        assert payload["messages"][1]["content"][0]["type"] == "tool_result"

        chunks = [c async for c in adapter.stream_text(req)]
        assert "".join(chunks) == "Anthropic Stream"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_gigachat_adapter_oauth_caching_refresh_sse_and_tls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GigaChatAdapter caches OAuth tokens, refreshes on 401, streams SSE, respects TLS."""
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "base64-creds-xyz")
    oauth_calls = 0
    chat_calls = 0
    force_401_once = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal oauth_calls, chat_calls, force_401_once
        if request.url.path.endswith("/oauth"):
            oauth_calls += 1
            assert request.headers["authorization"] == "Basic base64-creds-xyz"
            assert "rquid" in request.headers
            assert "GIGACHAT_API_PERS" in request.content.decode("utf-8")
            return httpx.Response(
                200,
                json={
                    "access_token": f"giga-tok-{oauth_calls}",
                    "expires_at": 4_102_444_800_000,  # ms far future
                },
            )

        chat_calls += 1
        assert request.url.path.endswith("/chat/completions")
        if force_401_once and request.headers.get("authorization") == "Bearer giga-tok-1":
            force_401_once = False
            return httpx.Response(401, json={"message": "Token expired"})

        body = json.loads(request.content.decode("utf-8"))
        if body.get("stream"):
            sse = (
                b'data: {"choices": [{"delta": {"content": "\xd0\x9f\xd1\x80\xd0\xb8"}}]}\n\n'
                b'data: {"choices": [{"delta": {"content": "\xd0\xb2\xd0\xb5\xd1\x82"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_FragmentedByteStream(sse, chunk_bytes=4),
            )

        return httpx.Response(
            200,
            json={
                "model": "GigaChat",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Ответ Сбера"}}
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        cfg = UpstreamConfig(
            name="gigachat-ru",
            provider=ProviderType.GIGACHAT,
            base_url="https://gigachat.devices.sberbank.ru/api/v1",
            model="GigaChat",
            api_key_env="GIGACHAT_CREDENTIALS",
            tls_verify=True,
            is_local=True,
        )
        adapter = GigaChatAdapter(cfg, client=client)
        req = UpstreamRequest(
            provider=ProviderType.GIGACHAT,
            model="GigaChat",
            messages=({"role": "user", "content": "Привет"},),
        )

        # First call gets tok-1, hits 401 once, refreshes to tok-2, succeeds
        resp1 = await adapter.complete(req)
        assert resp1.content == "Ответ Сбера"
        assert resp1.provider == ProviderType.GIGACHAT
        assert oauth_calls == 2
        assert chat_calls == 2

        # Second call reuses cached tok-2 without another OAuth call
        resp2 = await adapter.complete(req)
        assert resp2.content == "Ответ Сбера"
        assert oauth_calls == 2

        # Streaming reuses cached tok-2
        streamed = "".join([c async for c in adapter.stream_text(req)])
        assert streamed == "Привет"
        assert oauth_calls == 2
    finally:
        await client.aclose()

    # Verify tls_verify=False and ca_bundle_path configure underlying AsyncClient SSL context
    ca_file = tmp_path / "mincifry-ca.pem"
    _write_self_signed_ca(ca_file)

    insecure_adapter = GigaChatAdapter(
        UpstreamConfig(
            name="gigachat-insecure",
            provider=ProviderType.GIGACHAT,
            tls_verify=False,
            is_local=True,
        )
    )
    custom_ca_adapter = GigaChatAdapter(
        UpstreamConfig(
            name="gigachat-ca",
            provider=ProviderType.GIGACHAT,
            tls_verify=True,
            ca_bundle_path=str(ca_file),
            is_local=True,
        )
    )
    try:
        insecure_pool = insecure_adapter._client._transport._pool  # type: ignore[attr-defined]
        assert insecure_pool._ssl_context.verify_mode == ssl.CERT_NONE

        ca_pool = custom_ca_adapter._client._transport._pool  # type: ignore[attr-defined]
        assert ca_pool._ssl_context.verify_mode == ssl.CERT_REQUIRED
    finally:
        await insecure_adapter.aclose()
        await custom_ca_adapter.aclose()


@pytest.mark.asyncio
async def test_yandexgpt_adapter_apikey_iam_and_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YandexGPTAdapter supports Api-Key & IAM auth, folder_id modelUri, and cumulative streams."""
    monkeypatch.setenv("YANDEX_API_KEY", "AQVN-test-api-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "b1gfolder123")
    seen: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append((str(request.url), dict(request.headers), body))
        if body.get("completionOptions", {}).get("stream"):
            ndjson = (
                json.dumps(
                    {
                        "result": {
                            "alternatives": [
                                {
                                    "message": {"role": "assistant", "text": "Яндекс"},
                                    "status": "ALTERNATIVE_STATUS_PARTIAL",
                                }
                            ]
                        }
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
                + json.dumps(
                    {
                        "result": {
                            "alternatives": [
                                {
                                    "message": {"role": "assistant", "text": "Яндекс GPT"},
                                    "status": "ALTERNATIVE_STATUS_FINAL",
                                }
                            ]
                        }
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
                + json.dumps(
                    {
                        "result": {
                            "alternatives": [
                                {
                                    "message": {"role": "assistant", "text": "Яндекс GPT"},
                                    "status": "ALTERNATIVE_STATUS_FINAL",
                                }
                            ]
                        }
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
            )
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                stream=_FragmentedByteStream(ndjson, chunk_bytes=7),
            )
        return httpx.Response(
            200,
            json={
                "result": {
                    "alternatives": [
                        {
                            "message": {"role": "assistant", "text": "Ответ Яндекса"},
                            "status": "ALTERNATIVE_STATUS_FINAL",
                        }
                    ],
                    "usage": {"inputTextTokens": "15", "completionTokens": "7"},
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        cfg = UpstreamConfig(
            name="yandexgpt-ru",
            provider=ProviderType.YANDEXGPT,
            base_url="https://llm.api.cloud.yandex.net/foundationModels/v1",
            model="yandexgpt/latest",
            api_key_env="YANDEX_API_KEY",
            folder_id_env="YANDEX_FOLDER_ID",
            is_local=True,
        )
        adapter = YandexGPTAdapter(cfg, client=client)
        req = UpstreamRequest(
            provider=ProviderType.YANDEXGPT,
            model="yandexgpt/latest",
            system="System rule",
            temperature=0.3,
            max_tokens=200,
            messages=(
                {"role": "user", "content": "Запрос"},
                {"role": "tool", "tool_call_id": "t1", "content": "Инструмент"},
            ),
        )

        resp = await adapter.complete(req)
        assert resp.provider == ProviderType.YANDEXGPT
        assert resp.content == "Ответ Яндекса"
        assert resp.prompt_tokens == 15
        assert resp.completion_tokens == 7

        url, headers, payload = seen[-1]
        assert url == "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        assert headers["authorization"] == "Api-Key AQVN-test-api-key"
        assert headers["x-folder-id"] == "b1gfolder123"
        assert payload["modelUri"] == "gpt://b1gfolder123/yandexgpt/latest"
        assert payload["messages"] == [
            {"role": "system", "text": "System rule"},
            {"role": "user", "text": "Запрос"},
            {"role": "user", "text": "Инструмент"},
        ]

        # Streaming cumulative deduplication
        deltas = [d async for d in adapter.stream_text(req)]
        assert deltas == ["Яндекс", " GPT"]
        assert "".join(deltas) == "Яндекс GPT"

        # IAM token prefix uses Bearer automatically
        monkeypatch.setenv("YANDEX_API_KEY", "t1.9euel-iam-token")
        await adapter.complete(req)
        assert seen[-1][1]["authorization"] == "Bearer t1.9euel-iam-token"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_data_aware_router_selection_list_models_and_policy_example() -> None:
    """DataAwareRouter guarantees mock-local, routes ROUTE_LOCAL safely, and lists models."""
    # Config deliberately omitting mock-local and setting default_upstream=openai-cloud
    custom_cfg = RouterConfig(
        default_upstream="openai-cloud",
        local_upstream="ollama-local",
        cloud_upstream="openai-cloud",
        upstreams={
            "ollama-local": UpstreamConfig(
                name="ollama-local",
                provider=ProviderType.OLLAMA,
                base_url="http://127.0.0.1:11434/v1",
                model="qwen2.5:7b",
                is_local=True,
            ),
            "openai-cloud": UpstreamConfig(
                name="openai-cloud",
                provider=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
                api_key_env="OPENAI_API_KEY",
                is_local=False,
            ),
        },
    )
    router = DataAwareRouter(custom_cfg)
    try:
        mock_adapter = router.get_adapter("mock-local")
        assert isinstance(mock_adapter, MockLLMAdapter)
        assert mock_adapter.is_local is True

        with pytest.raises(KeyError, match="Unknown upstream"):
            router.get_adapter("does-not-exist")

        allow_decision = PolicyDecision(action=Action.ALLOW, reason="clean")
        assert router.select_upstream(allow_decision).name == "openai-cloud"
        assert (
            router.select_upstream(allow_decision, requested_upstream="ollama-local").name
            == "ollama-local"
        )
        assert (
            router.select_upstream(allow_decision, requested_upstream="qwen2.5:7b").name
            == "ollama-local"
        )
        assert (
            router.select_upstream(allow_decision, requested_upstream="invalid-name").name
            == "openai-cloud"
        )

        # ROUTE_LOCAL forces local adapter even if client requested openai-cloud
        route_local_decision = PolicyDecision(
            action=Action.ROUTE_LOCAL,
            reason="pii detected",
            target_upstream=None,
        )
        selected_local = router.select_upstream(
            route_local_decision,
            requested_upstream="openai-cloud",
        )
        assert selected_local.name == "ollama-local"
        assert selected_local.is_local is True

        # target_upstream overrides requested_upstream
        targeted_decision = PolicyDecision(
            action=Action.MASK_REVERSIBLE,
            reason="override to mock",
            target_upstream="mock-local",
        )
        assert (
            router.select_upstream(targeted_decision, requested_upstream="openai-cloud").name
            == "mock-local"
        )

        # Safety check: ROUTE_LOCAL with a cloud target_upstream still forces a local adapter
        unsafe_target = PolicyDecision(
            action=Action.ROUTE_LOCAL,
            reason="misconfigured target",
            target_upstream="openai-cloud",
        )
        assert router.select_upstream(unsafe_target).is_local is True

        models = router.list_models()
        assert all(m["object"] == "model" for m in models)
        model_ids = {m["id"] for m in models}
        assert {"qwen2.5:7b", "gpt-4o-mini", "pokrov-mock-llm"} <= model_ids
    finally:
        await router.aclose()

    # Load full policy.example.yaml router config
    example_cfg = load_config(REPO_ROOT / "policy.example.yaml")
    example_router = DataAwareRouter(example_cfg.router)
    try:
        for upstream_name in (
            "mock-local",
            "ollama-local",
            "vllm-local",
            "openai-cloud",
            "anthropic-cloud",
            "gigachat-ru",
            "yandexgpt-ru",
        ):
            assert isinstance(example_router.get_adapter(upstream_name), UpstreamAdapter)
    finally:
        await example_router.aclose()


@pytest.mark.asyncio
async def test_adapter_edge_cases_errors_and_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify HTTP error propagation, /v1 URL normalization, and GigaChat OAuth lock."""
    # MockLLMAdapter system-only fallback and system EchoLeak trigger
    mock = MockLLMAdapter()
    sys_only = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="pokrov-mock-llm",
        system="System-only fallback",
        messages=(),
    )
    assert (await mock.complete(sys_only)).content == "Принято к обработке: System-only fallback"

    sys_leak = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="pokrov-mock-llm",
        system=f"Hidden {ECHOLEAK_TRIGGER}",
        messages=({"role": "user", "content": "normal"},),
    )
    assert (await mock.complete(sys_leak)).content == ECHOLEAK_PAYLOAD

    # OpenAI HTTP 500 raises HTTPStatusError
    err_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(500, json={"error": "boom"}))
    )
    try:
        oa_err = OpenAICompatAdapter(
            UpstreamConfig(
                name="oa-err",
                provider=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
            ),
            client=err_client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await oa_err.complete(
                UpstreamRequest(
                    provider=ProviderType.OPENAI,
                    model="gpt-4o-mini",
                    messages=({"role": "user", "content": "hi"},),
                )
            )
    finally:
        await err_client.aclose()

    # Anthropic base_url ending with /v1 avoids /v1/v1/messages
    seen_anthropic_urls: list[str] = []

    def ant_handler(request: httpx.Request) -> httpx.Response:
        seen_anthropic_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "ok"}], "usage": {}},
        )

    ant_client = httpx.AsyncClient(transport=httpx.MockTransport(ant_handler))
    try:
        ant = AnthropicAdapter(
            UpstreamConfig(
                name="ant-v1",
                provider=ProviderType.ANTHROPIC,
                base_url="https://api.anthropic.com/v1",
                model="claude-3-5-sonnet-latest",
            ),
            client=ant_client,
        )
        await ant.complete(
            UpstreamRequest(
                provider=ProviderType.ANTHROPIC,
                model="claude-3-5-sonnet-latest",
                messages=({"role": "user", "content": "hi"},),
            )
        )
        assert seen_anthropic_urls == ["https://api.anthropic.com/v1/messages"]
    finally:
        await ant_client.aclose()

    # GigaChat concurrent requests acquire OAuth token only once via lock & retry 401 on stream
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "creds")
    oauth_count = 0
    stream_attempts = 0

    def giga_handler(request: httpx.Request) -> httpx.Response:
        nonlocal oauth_count, stream_attempts
        if request.url.path.endswith("/oauth"):
            oauth_count += 1
            return httpx.Response(
                200,
                json={"access_token": f"tok-{oauth_count}", "expires_in": 1800},
            )
        body = json.loads(request.content.decode("utf-8"))
        if body.get("stream"):
            stream_attempts += 1
            if stream_attempts == 1:
                return httpx.Response(401, json={"error": "expired"})
            return httpx.Response(
                200,
                stream=_FragmentedByteStream(
                    b'data: {"choices": [{"delta": {"content": "stream-ok"}}]}\n\ndata: [DONE]\n\n',
                    chunk_bytes=5,
                ),
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    giga_client = httpx.AsyncClient(transport=httpx.MockTransport(giga_handler))
    try:
        giga = GigaChatAdapter(
            UpstreamConfig(
                name="giga-conc",
                provider=ProviderType.GIGACHAT,
                base_url="https://gigachat.devices.sberbank.ru/api/v1",
                model="GigaChat",
                api_key_env="GIGACHAT_CREDENTIALS",
                is_local=True,
            ),
            client=giga_client,
        )
        giga_req = UpstreamRequest(
            provider=ProviderType.GIGACHAT,
            model="GigaChat",
            messages=({"role": "user", "content": "hi"},),
        )
        res_a, res_b = await asyncio.gather(giga.complete(giga_req), giga.complete(giga_req))
        assert res_a.content == "ok"
        assert res_b.content == "ok"
        assert oauth_count == 1

        # First stream attempt gets 401 -> refreshes token once -> succeeds
        stream_out = "".join([c async for c in giga.stream_text(giga_req)])
        assert stream_out == "stream-ok"
        assert oauth_count == 2
        assert stream_attempts == 2
    finally:
        await giga_client.aclose()

    # YandexGPT SSE-wrapped stream + explicit gpt:// model URI
    def yandex_sse_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["modelUri"] == "gpt://custom-folder/yandexgpt-pro/rc"
        sse = (
            b'data: {"result": {"alternatives": [{"message": {"text": "A"}}]}}\n\n'
            b'data: {"result": {"alternatives": [{"message": {"text": "AB"}}]}}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, stream=_FragmentedByteStream(sse, chunk_bytes=8))

    y_client = httpx.AsyncClient(transport=httpx.MockTransport(yandex_sse_handler))
    try:
        y_adapter = YandexGPTAdapter(
            UpstreamConfig(
                name="yandex-sse",
                provider=ProviderType.YANDEXGPT,
                base_url="https://llm.api.cloud.yandex.net/foundationModels/v1",
                model="gpt://custom-folder/yandexgpt-pro/rc",
                is_local=True,
            ),
            client=y_client,
        )
        y_req = UpstreamRequest(
            provider=ProviderType.YANDEXGPT,
            model="gpt://custom-folder/yandexgpt-pro/rc",
            messages=({"role": "user", "content": "hi"},),
        )
        assert "".join([c async for c in y_adapter.stream_text(y_req)]) == "AB"
    finally:
        await y_client.aclose()
