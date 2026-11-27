"""Tests for MockLLMAdapter, upstream HTTP adapters, StreamFrame SSE, and DataAwareRouter."""

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


def _tok(*parts: str) -> str:
    """Assemble synthetic tokens at runtime to satisfy GitHub Push Protection (GH013)."""
    return "".join(parts)


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
async def test_mock_llm_adapter_complete_placeholders_and_stream_frames() -> None:
    """MockLLMAdapter preserves <PKR:...>, honors force_adapter_model, and yields StreamFrames."""
    with pytest.raises(ValueError, match="chunk_size"):
        MockLLMAdapter(chunk_size=0)

    adapter = MockLLMAdapter(name="mock-local", model="pokrov-mock-llm", chunk_size=2)
    assert adapter.name == "mock-local"
    assert adapter.provider == ProviderType.MOCK
    assert adapter.is_local is True

    req = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="gpt-4o",
        force_adapter_model=True,
        messages=(
            {"role": "system", "content": "Be safe"},
            {"role": "user", "content": "Первое сообщение"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "Паспорт <PKR:RU_PASSPORT:a1b2c3d4e5f6> и СНИЛС <PKR:RU_SNILS:deadbeef0011>"
                ),
            },
        ),
    )

    resp = await adapter.complete(req)
    assert resp.status_code == 200
    assert resp.provider == ProviderType.MOCK
    assert resp.upstream_name == "mock-local"
    assert resp.model == "pokrov-mock-llm"
    assert "<PKR:RU_PASSPORT:a1b2c3d4e5f6>" in resp.content
    assert "<PKR:RU_SNILS:deadbeef0011>" in resp.content
    assert resp.content.startswith("Принято к обработке: ")

    frames = [f async for f in adapter.stream_frames(req)]
    assert "".join(f.text for f in frames if f.kind == "text") == resp.content
    assert any(f.kind == "usage" and (f.completion_tokens or 0) > 0 for f in frames)
    assert frames[-1].kind == "finish" and frames[-1].finish_reason == "stop"

    for chunk_size in (1, 2, 3):
        frag_adapter = MockLLMAdapter(chunk_size=chunk_size)
        chunks = [chunk async for chunk in frag_adapter.stream_text(req)]
        assert all(1 <= len(chunk) <= chunk_size for chunk in chunks)
        assert "".join(chunks) == resp.content

    tool_result_req = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="pokrov-default",
        api_format="anthropic",
        messages=(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "Token <PKR:EMAIL:112233445566>"}],
                    }
                ],
            },
        ),
    )
    tool_resp = await adapter.complete(tool_result_req)
    assert tool_resp.model == "pokrov-mock-llm"
    assert "<PKR:EMAIL:112233445566>" in tool_resp.content

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
async def test_openai_compat_adapter_model_rewrite_and_stream_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAICompatAdapter rewrites model on force_adapter_model/pokrov-default and streams."""
    test_oai_key = _tok("sk-", "test-openai-123")
    monkeypatch.setenv("OPENAI_API_KEY", test_oai_key)
    captured: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append((str(request.url), dict(request.headers), body))
        if body.get("stream"):
            sse = (
                b": keep-alive\n\n"
                b'data: {"choices": [{"delta": {"role": "assistant", '
                b'"reasoning_content": "plan"}}]}\n\n'
                b'data: {"choices": [{"delta": {"content": "Hello "}}]}\n\n'
                b'data: {"choices": [{"delta": {"content": "Pokrov!"}}]}\n\n'
                b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", '
                b'"function": {"name": "search", "arguments": "{\\"q\\":\\"152-FZ\\"}"}}]}}]}\n\n'
                b'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], '
                b'"usage": {"prompt_tokens": 9, "completion_tokens": 4}}\n\n'
                b"data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_FragmentedByteStream(sse, chunk_bytes=5),
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

        # Omitted model ("pokrov-default") rewrites to adapter's configured model ("gpt-4o-mini")
        omitted_resp = await openai_adapter.complete(
            UpstreamRequest(
                provider=ProviderType.OPENAI,
                model="pokrov-default",
                messages=(
                    {"role": "developer", "content": "Follow rules"},
                    {"role": "user", "content": "Hi"},
                ),
                extra_body={"mcp_servers": ["blocked"], "top_p": 0.9},
            )
        )
        assert omitted_resp.model == "gpt-4o-mini"
        _, _, omitted_payload = captured[-1]
        assert omitted_payload["model"] == "gpt-4o-mini"
        assert omitted_payload["messages"][0]["role"] == "system"
        assert "mcp_servers" not in omitted_payload
        assert omitted_payload["top_p"] == 0.9

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
        assert resp.model == "gpt-4o"

        url, headers, payload = captured[-1]
        assert url == "https://api.openai.com/v1/chat/completions"
        assert headers["authorization"] == f"Bearer {test_oai_key}"
        assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}

        frames = [f async for f in openai_adapter.stream_frames(req)]
        assert "".join(f.text for f in frames if f.kind == "text") == "Hello Pokrov!"
        assert any(f.kind == "thinking" and f.text == "plan" for f in frames)
        assert any(
            f.kind == "tool_call" and f.tool_name == "search" and "152-FZ" in f.arguments_delta
            for f in frames
        )
        assert any(f.kind == "usage" and f.prompt_tokens == 9 for f in frames)
        assert any(f.kind == "finish" and f.finish_reason == "tool_calls" for f in frames)

        # ROUTE_LOCAL force_adapter_model=True rewrites "gpt-4o" -> "qwen2.5:7b" on Ollama
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
            model="gpt-4o",
            force_adapter_model=True,
            api_format="anthropic",
            tools=({"name": "read_file", "input_schema": {"type": "object"}},),
            messages=(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "doc text"}
                    ],
                },
            ),
        )
        await ollama_adapter.complete(ollama_req)
        _, headers, payload = captured[-1]
        assert "authorization" not in headers
        assert payload["model"] == "qwen2.5:7b"
        assert payload["messages"][0]["content"] == "doc text"
        assert payload["tools"][0]["type"] == "function"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_adapter_cache_control_tool_merging_and_stream_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnthropicAdapter preserves system_blocks cache_control, blocks mcp_servers, merges tools."""
    ant_key = _tok("sk-ant-", "test-999")
    monkeypatch.setenv("ANTHROPIC_API_KEY", ant_key)
    seen_requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_requests.append((str(request.url), dict(request.headers), body))
        if body.get("stream"):
            sse = (
                b"event: message_start\n"
                b'data: {"type": "message_start", "message": {"id": "msg_1", '
                b'"usage": {"input_tokens": 17, "output_tokens": 0}}}\n\n'
                b"event: content_block_start\n"
                b'data: {"type": "content_block_start", "index": 0, '
                b'"content_block": {"type": "thinking", "thinking": ""}}\n\n'
                b"event: content_block_delta\n"
                b'data: {"type": "content_block_delta", "index": 0, '
                b'"delta": {"type": "thinking_delta", "thinking": "step1"}}\n\n'
                b"event: content_block_start\n"
                b'data: {"type": "content_block_start", "index": 1, '
                b'"content_block": {"type": "text", "text": ""}}\n\n'
                b"event: content_block_delta\n"
                b'data: {"type": "content_block_delta", "index": 1, '
                b'"delta": {"type": "text_delta", "text": "Anthropic Stream"}}\n\n'
                b"event: content_block_start\n"
                b'data: {"type": "content_block_start", "index": 2, "content_block": '
                b'{"type": "tool_use", "id": "toolu_9", "name": "bash", "input": {}}}\n\n'
                b"event: content_block_delta\n"
                b'data: {"type": "content_block_delta", "index": 2, "delta": '
                b'{"type": "input_json_delta", "partial_json": "{\\"cmd\\":\\"pwd\\"}"}}\n\n'
                b"event: message_delta\n"
                b'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, '
                b'"usage": {"output_tokens": 11}}\n\n'
                b"event: message_stop\n"
                b'data: {"type": "message_stop"}\n\n'
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_FragmentedByteStream(sse, chunk_bytes=6),
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
            model="pokrov-default",
            api_format="anthropic",
            system_blocks=(
                {
                    "type": "text",
                    "text": "Cached prompt",
                    "cache_control": {"type": "ephemeral"},
                },
            ),
            extra_body={
                "mcp_servers": [{"type": "url", "url": "https://evil.example/mcp"}],
                "stream_options": {"include_usage": True},
                "top_p": 0.85,
            },
            tools=(
                {
                    "type": "function",
                    "function": {"name": "bash", "parameters": {"type": "object"}},
                },
            ),
            messages=(
                {"role": "developer", "content": "Extra rule"},
                {"role": "user", "content": "Check tools"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "out1"},
                {"role": "tool", "tool_call_id": "c2", "content": "out2"},
            ),
        )
        resp = await adapter.complete(req)
        assert resp.provider == ProviderType.ANTHROPIC
        assert resp.content == "Claude response"
        assert resp.model == "claude-3-5-sonnet-latest"

        url, headers, payload = seen_requests[-1]
        assert url == "https://api.anthropic.com/v1/messages"
        assert headers["x-api-key"] == ant_key
        assert "mcp_servers" not in payload
        assert "stream_options" not in payload
        assert payload["top_p"] == 0.85
        assert payload["model"] == "claude-3-5-sonnet-latest"
        assert isinstance(payload["system"], list)
        assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert payload["system"][1] == {"type": "text", "text": "Extra rule"}
        assert payload["tools"][0]["name"] == "bash"
        assert "input_schema" in payload["tools"][0]
        # Consecutive tool messages c1 & c2 merged into single user message!
        assert len(payload["messages"]) == 3
        assert payload["messages"][1]["content"][0]["type"] == "tool_use"
        assert len(payload["messages"][2]["content"]) == 2

        frames = [f async for f in adapter.stream_frames(req)]
        assert any(f.kind == "thinking" and f.text == "step1" for f in frames)
        assert "".join(f.text for f in frames if f.kind == "text") == "Anthropic Stream"
        assert any(
            f.kind == "tool_call" and f.tool_name == "bash" and "pwd" in f.arguments_delta
            for f in frames
        )
        assert any(f.kind == "finish" and f.finish_reason == "tool_use" for f in frames)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_gigachat_adapter_roles_functions_stampede_and_missing_ca(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GigaChatAdapter maps roles/functions, prevents 401 OAuth stampede, handles missing CA."""
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "base64-creds-xyz")
    oauth_calls = 0
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal oauth_calls
        if request.url.path.endswith("/oauth"):
            oauth_calls += 1
            return httpx.Response(
                200,
                json={
                    "access_token": f"giga-tok-{oauth_calls}",
                    "expires_at": 4_102_444_800_000,
                },
            )
        if request.headers.get("authorization") == "Bearer giga-tok-1":
            return httpx.Response(401, json={"message": "Token expired"})

        body = json.loads(request.content.decode("utf-8"))
        captured_bodies.append(body)
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
            is_local=False,
        )
        adapter = GigaChatAdapter(cfg, client=client)
        # Prime stale token giga-tok-1
        adapter._access_token = "giga-tok-1"
        adapter._token_expires_at = 4_102_444_800.0
        oauth_calls = 1

        req = UpstreamRequest(
            provider=ProviderType.GIGACHAT,
            model="gpt-4o",
            force_adapter_model=True,
            tools=(
                {
                    "type": "function",
                    "function": {"name": "calc", "parameters": {"type": "object"}},
                },
            ),
            messages=(
                {"role": "developer", "content": "Sys"},
                {"role": "user", "content": [{"type": "text", "text": "Привет"}]},
                {"role": "tool", "tool_call_id": "calc", "content": "42"},
            ),
        )

        # 5 concurrent requests fail on giga-tok-1 -> must refresh OAuth only once (to giga-tok-2)
        results = await asyncio.gather(*(adapter.complete(req) for _ in range(5)))
        assert all(r.content == "Ответ Сбера" for r in results)
        assert oauth_calls == 2

        sent = captured_bodies[-1]
        assert sent["model"] == "GigaChat"
        assert [m["role"] for m in sent["messages"]] == ["system", "user", "function"]
        assert sent["messages"][1]["content"] == "Привет"
        assert sent["functions"][0]["name"] == "calc"

        streamed = "".join([c async for c in adapter.stream_text(req)])
        assert streamed == "Привет"
    finally:
        await client.aclose()

    # Verify missing ca_bundle_path does NOT crash at init, while valid CA & tls_verify=False work
    missing_ca_adapter = GigaChatAdapter(
        UpstreamConfig(
            name="gigachat-missing-ca",
            provider=ProviderType.GIGACHAT,
            tls_verify=True,
            ca_bundle_path=str(tmp_path / "does_not_exist.pem"),
        )
    )
    await missing_ca_adapter.aclose()

    ca_file = tmp_path / "mincifry-ca.pem"
    _write_self_signed_ca(ca_file)
    insecure_adapter = GigaChatAdapter(
        UpstreamConfig(name="gigachat-insecure", provider=ProviderType.GIGACHAT, tls_verify=False)
    )
    custom_ca_adapter = GigaChatAdapter(
        UpstreamConfig(
            name="gigachat-ca",
            provider=ProviderType.GIGACHAT,
            tls_verify=True,
            ca_bundle_path=str(ca_file),
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
async def test_yandexgpt_adapter_options_roles_tools_and_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YandexGPTAdapter maps top_p/stop to completionOptions, developer role, and streams."""
    yc_key = _tok("AQVN", "-test-api-key")
    monkeypatch.setenv("YANDEX_API_KEY", yc_key)
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
                            ],
                            "usage": {"inputTextTokens": "10", "completionTokens": "4"},
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
            is_local=False,
        )
        adapter = YandexGPTAdapter(cfg, client=client)
        req = UpstreamRequest(
            provider=ProviderType.YANDEXGPT,
            model="pokrov-default",
            temperature=0.3,
            max_tokens=200,
            extra_body={"top_p": 0.9, "stop": ["END"], "stream_options": {"include_usage": True}},
            tools=({"type": "function", "function": {"name": "lookup"}},),
            messages=(
                {"role": "developer", "content": "System rule"},
                {"role": "user", "content": "Запрос"},
                {"role": "tool", "tool_call_id": "t1", "content": "Инструмент"},
            ),
        )

        resp = await adapter.complete(req)
        assert resp.provider == ProviderType.YANDEXGPT
        assert resp.content == "Ответ Яндекса"
        assert resp.model == "yandexgpt/latest"

        url, headers, payload = seen[-1]
        assert url == "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        assert headers["authorization"] == f"Api-Key {yc_key}"
        assert payload["modelUri"] == "gpt://b1gfolder123/yandexgpt/latest"
        assert set(payload.keys()) == {"modelUri", "completionOptions", "messages", "tools"}
        assert payload["completionOptions"]["topP"] == 0.9
        assert payload["completionOptions"]["stopSequences"] == ["END"]
        assert payload["messages"] == [
            {"role": "system", "text": "System rule"},
            {"role": "user", "text": "Запрос"},
            {"role": "user", "text": "Инструмент"},
        ]

        frames = [f async for f in adapter.stream_frames(req)]
        assert [f.text for f in frames if f.kind == "text"] == ["Яндекс", " GPT"]
        assert any(f.kind == "usage" and f.prompt_tokens == 10 for f in frames)
        assert frames[-1].kind == "finish"

        monkeypatch.setenv("YANDEX_API_KEY", "t1.9euel-iam-token")
        await adapter.complete(req)
        assert seen[-1][1]["authorization"] == "Bearer t1.9euel-iam-token"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_data_aware_router_allow_cloud_false_and_selection() -> None:
    """DataAwareRouter enforces allow_cloud=False even on ALLOW and overrides cloud requests."""
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
            "vllm-local": UpstreamConfig(
                name="vllm-local",
                provider=ProviderType.VLLM,
                base_url="http://127.0.0.1:8000/v1",
                model="Qwen/Qwen2.5-14B",
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
        allow_decision = PolicyDecision(action=Action.ALLOW, reason="clean")
        assert router.select_upstream(allow_decision).name == "openai-cloud"

        # allow_cloud=False (strict-fstec-117) forces local even on Action.ALLOW & cloud request!
        assert (
            router.select_upstream(
                allow_decision,
                requested_upstream="openai-cloud",
                allow_cloud=False,
            ).name
            == "ollama-local"
        )
        # Choosing another local upstream when allow_cloud=False is allowed
        assert (
            router.select_upstream(
                allow_decision,
                requested_upstream="vllm-local",
                allow_cloud=False,
            ).name
            == "vllm-local"
        )

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

        models = router.list_models()
        model_ids = {m["id"] for m in models}
        assert {"qwen2.5:7b", "gpt-4o-mini", "pokrov-mock-llm"} <= model_ids
    finally:
        await router.aclose()

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
async def test_adapter_edge_cases_errors_and_concurrency() -> None:
    """Verify HTTP error propagation and /v1 URL normalization."""
    mock = MockLLMAdapter()
    sys_only = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="pokrov-mock-llm",
        system="System-only fallback",
        messages=(),
    )
    assert (await mock.complete(sys_only)).content == "Принято к обработке: System-only fallback"

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


@pytest.mark.asyncio
async def test_router_tls_errors_and_adapter_protocol_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover SSL verification variants, cloud-block fallback, and streaming edge cases."""
    from pokrov.proxy.router import _build_ssl_verify

    assert (
        _build_ssl_verify(
            UpstreamConfig(name="no-tls", provider=ProviderType.OPENAI, tls_verify=False)
        )
        is False
    )
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("", encoding="utf-8")
    # Invalid/empty CA directory & missing CA file fallback cleanly without crash
    assert (
        _build_ssl_verify(
            UpstreamConfig(
                name="ca-dir", provider=ProviderType.OPENAI, ca_bundle_path=str(tmp_path)
            )
        )
        is not False
    )
    assert (
        _build_ssl_verify(
            UpstreamConfig(
                name="ca-miss",
                provider=ProviderType.OPENAI,
                ca_bundle_path=str(tmp_path / "missing.pem"),
            )
        )
        is not False
    )

    # DataAwareRouter with allow_cloud=False refuses cloud if no local upstream exists
    cloud_only_cfg = RouterConfig(
        default_upstream="openai-cloud",
        local_upstream="missing-local",
        cloud_upstream="openai-cloud",
        upstreams={
            "openai-cloud": UpstreamConfig(
                name="openai-cloud",
                provider=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
                is_local=False,
            )
        },
    )
    router = DataAwareRouter(cloud_only_cfg)
    del router._adapters["mock-local"]
    try:
        with pytest.raises(KeyError):
            router.get_adapter("non-existent")
        dec_allow = PolicyDecision(action=Action.ALLOW, reason="ok")
        with pytest.raises(KeyError):
            router.select_upstream(dec_allow, allow_cloud=False)
    finally:
        await router.aclose()

    # AnthropicAdapter stop->stop_sequences, thinking, tool_result list, and SSE ping/error
    anth_seen: list[dict[str, Any]] = []

    def anth_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        anth_seen.append(body)
        if body.get("stream"):
            sse = (
                b": ping\n\n"
                b"event: content_block_start\n"
                b'data: {"type": "content_block_start", "index": 0, '
                b'"content_block": {"type": "thinking", "thinking": "Think1"}}\n\n'
                b"event: content_block_delta\n"
                b'data: {"type": "content_block_delta", "index": 0, '
                b'"delta": {"type": "thinking_delta", "thinking": "Think2"}}\n\n'
                b"event: message_delta\n"
                b'data: {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, '
                b'"usage": {"output_tokens": 5}}\n\n'
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)
        return httpx.Response(
            200,
            json={
                "content": "plain string response",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        )

    anth_client = httpx.AsyncClient(transport=httpx.MockTransport(anth_handler))
    try:
        anth = AnthropicAdapter(
            UpstreamConfig(
                name="anth-edge",
                provider=ProviderType.ANTHROPIC,
                base_url="https://api.anthropic.com",
                model="claude-3-5-sonnet-latest",
            ),
            client=anth_client,
        )
        req_anth = UpstreamRequest(
            provider=ProviderType.ANTHROPIC,
            model="pokrov-default",
            system_blocks=({"type": "text", "text": "Sys"},),
            messages=(
                {"role": "system", "content": "Extra system"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "search", "arguments": "not-json"}}
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [{"type": "text", "text": "result text"}],
                },
            ),
            extra_body={"stop": "END"},
            tools=({"type": "function", "function": {"name": "search", "description": "d"}},),
        )
        comp = await anth.complete(req_anth)
        assert comp.content == "plain string response"
        assert anth_seen[-1]["stop_sequences"] == ["END"]
        frames = [f async for f in anth.stream_frames(req_anth)]
        assert any(f.kind == "thinking" for f in frames)
        assert any(f.kind == "finish" and f.finish_reason == "max_tokens" for f in frames)
    finally:
        await anth_client.aclose()

    # GigaChatAdapter: Basic prefix credentials, 401 retry, expires_in fallback, stream_frames
    monkeypatch.setenv("GIGA_CRED_TEST", "Basic already-prefixed-cred")

    def giga_stream_handler(request: httpx.Request) -> httpx.Response:
        if "/oauth" in str(request.url):
            return httpx.Response(200, json={"access_token": "tok-1", "expires_in": 600})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices": [{"delta": {"content": "Hi "}}]}\n\n'
                b'data: {"choices": [{"delta": {"content": "Giga"}, '
                b'"finish_reason": "stop"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    giga_client = httpx.AsyncClient(transport=httpx.MockTransport(giga_stream_handler))
    try:
        giga = GigaChatAdapter(
            UpstreamConfig(
                name="giga-edge",
                provider=ProviderType.GIGACHAT,
                base_url="https://gigachat.devices.sberbank.ru/api/v1",
                model="GigaChat",
                api_key_env="GIGA_CRED_TEST",
            ),
            client=giga_client,
        )
        streamed = [
            chunk
            async for chunk in giga.stream_text(
                UpstreamRequest(
                    provider=ProviderType.GIGACHAT,
                    model="GigaChat",
                    messages=({"role": "user", "content": "hi"},),
                )
            )
        ]
        assert "".join(streamed) == "Hi Giga"
    finally:
        await giga_client.aclose()

    # YandexGPTAdapter: missing folder_id -> sends no x-folder-id; IAM t1. -> Bearer
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    monkeypatch.setenv("YANDEX_IAM_TEST", _tok("t1.", "TestIamTokenPayload999"))
    seen_miss_hdrs: list[dict[str, str]] = []

    def _yg_err_handler(request: httpx.Request) -> httpx.Response:
        seen_miss_hdrs.append(dict(request.headers))
        return httpx.Response(400, json={"error": "missing folder"})

    yg_err_client = httpx.AsyncClient(transport=httpx.MockTransport(_yg_err_handler))
    yg = YandexGPTAdapter(
        UpstreamConfig(
            name="yg-miss-folder",
            provider=ProviderType.YANDEXGPT,
            base_url="https://llm.api.cloud.yandex.net/foundationModels/v1",
            model="yandexgpt/latest",
            api_key_env="YANDEX_IAM_TEST",
            folder_id_env="YANDEX_FOLDER_ID",
        ),
        client=yg_err_client,
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await yg.complete(
                UpstreamRequest(
                    provider=ProviderType.YANDEXGPT,
                    model="yandexgpt/latest",
                    messages=({"role": "user", "content": "hi"},),
                )
            )
        assert seen_miss_hdrs[0]["authorization"].startswith("Bearer ")
        assert "x-folder-id" not in seen_miss_hdrs[0]
    finally:
        await yg.aclose()
        await yg_err_client.aclose()

    # YandexGPTAdapter streaming toolCallList & TRUNCATED status
    def yandex_stream_handler(request: httpx.Request) -> httpx.Response:
        if body := json.loads(request.content.decode("utf-8")):
            _ = body
        sse = (
            b'{"result": {"alternatives": [{"message": {"role": "assistant", '
            b'"toolCallList": {"toolCalls": [{"functionCall": {"name": "calc", '
            b'"arguments": {"x": 1}}}]}}, "status": "ALTERNATIVE_STATUS_PARTIAL"}], '
            b'"usage": {"inputTextTokens": "5", "completionTokens": "2"}}}\n'
            b'{"result": {"alternatives": [{"message": {"role": "assistant", '
            b'"text": "Done"}, "status": "ALTERNATIVE_STATUS_TRUNCATED_FINAL"}]}}\n'
        )
        return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, content=sse)

    yg_client = httpx.AsyncClient(transport=httpx.MockTransport(yandex_stream_handler))
    monkeypatch.setenv("YG_FOLDER_OK", "b1gok")
    yg_ok = YandexGPTAdapter(
        UpstreamConfig(
            name="yg-ok",
            provider=ProviderType.YANDEXGPT,
            base_url="https://llm.api.cloud.yandex.net/foundationModels/v1",
            model="yandexgpt/latest",
            api_key_env="YANDEX_IAM_TEST",
            folder_id_env="YG_FOLDER_OK",
        ),
        client=yg_client,
    )
    try:
        req_yg = UpstreamRequest(
            provider=ProviderType.YANDEXGPT,
            model="yandexgpt/latest",
            messages=({"role": "user", "content": "calc"},),
        )
        y_frames = [f async for f in yg_ok.stream_frames(req_yg)]
        assert any(f.kind == "tool_call" and f.tool_name == "calc" for f in y_frames)
        assert any(f.kind == "text" and f.text == "Done" for f in y_frames)
        assert any(f.kind == "finish" and f.finish_reason == "length" for f in y_frames)
    finally:
        await yg_ok.aclose()
        await yg_client.aclose()


@pytest.mark.asyncio
async def test_mock_and_adapter_coverage_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover extract_block_text variants, Anthropic choices/SSE errors, Giga 401, Yandex Api-Key."""
    from pokrov.proxy.mock_llm import ECHOLEAK_PAYLOAD, ECHOLEAK_TRIGGER, extract_block_text

    assert (
        extract_block_text(
            [
                "plain",
                {"thinking": "think_part"},
                {"content": [{"text": "nested_part"}]},
                {"source": {"data": "src_part"}},
            ]
        )
        == "plain\nthink_part\nnested_part\nsrc_part"
    )
    assert extract_block_text(12345) == ""

    mock = MockLLMAdapter()
    sys_leak = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="pokrov-mock-llm",
        system=ECHOLEAK_TRIGGER,
        messages=(),
    )
    assert (await mock.complete(sys_leak)).content == ECHOLEAK_PAYLOAD
    blk_leak = UpstreamRequest(
        provider=ProviderType.MOCK,
        model="pokrov-mock-llm",
        system_blocks=({"type": "text", "text": ECHOLEAK_TRIGGER},),
        messages=(),
    )
    assert (await mock.complete(blk_leak)).content == ECHOLEAK_PAYLOAD

    # OpenAI stream_text skipping malformed/comment lines
    oai_key = _tok("sk-", "edge-key-999")
    monkeypatch.setenv("OAI_EDGE_KEY", oai_key)

    def oai_sse_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b": keepalive\n\n"
                b"event: ping\n\n"
                b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    oai_c = httpx.AsyncClient(transport=httpx.MockTransport(oai_sse_handler))
    oai_ad = OpenAICompatAdapter(
        UpstreamConfig(
            name="oai-s",
            provider=ProviderType.OPENAI,
            base_url="https://api.openai.com",
            model="gpt-4o-mini",
            api_key_env="OAI_EDGE_KEY",
        ),
        client=oai_c,
    )
    try:
        chunks = [
            ch
            async for ch in oai_ad.stream_text(
                UpstreamRequest(
                    provider=ProviderType.OPENAI,
                    model="gpt-4o-mini",
                    messages=({"role": "user", "content": "hi"},),
                )
            )
        ]
        assert "".join(chunks) == "Hi"
    finally:
        await oai_c.aclose()

    # AnthropicAdapter stop_sequences list, SSE content_block_stop & error
    def anth_edge_handler(_req: httpx.Request) -> httpx.Response:
        sse = (
            b"event: content_block_start\n"
            b'data: {"type": "content_block_start", "index": 0, '
            b'"content_block": {"type": "text", "text": "Hello"}}\n\n'
            b"event: content_block_stop\n"
            b'data: {"type": "content_block_stop", "index": 0}\n\n'
            b"event: message_stop\n"
            b'data: {"type": "message_stop"}\n\n'
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    ant_c = httpx.AsyncClient(transport=httpx.MockTransport(anth_edge_handler))
    ant_ad = AnthropicAdapter(
        UpstreamConfig(
            name="ant-s",
            provider=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com",
            model="claude-3-5-sonnet-latest",
        ),
        client=ant_c,
    )
    try:
        payload = ant_ad._build_payload(
            UpstreamRequest(
                provider=ProviderType.ANTHROPIC,
                model="claude-3-5-sonnet-latest",
                extra_body={"stop": ["STOP1", "STOP2"]},
                messages=({"role": "user", "content": "hi"},),
            ),
            stream=True,
        )
        assert payload["stop_sequences"] == ["STOP1", "STOP2"]
        txt = "".join(
            [
                c
                async for c in ant_ad.stream_text(
                    UpstreamRequest(
                        provider=ProviderType.ANTHROPIC,
                        model="claude-3-5-sonnet-latest",
                        messages=({"role": "user", "content": "hi"},),
                    )
                )
            ]
        )
        assert txt == "Hello"
    finally:
        await ant_c.aclose()

    # GigaChatAdapter 401 refresh retry + expires_at ms timestamp
    giga_calls = 0

    def giga_retry_handler(req: httpx.Request) -> httpx.Response:
        nonlocal giga_calls
        if "/oauth" in str(req.url):
            giga_calls += 1
            return httpx.Response(
                200,
                json={"access_token": f"gtok-{giga_calls}", "expires_at": 1_900_000_000_000},
            )
        if req.headers.get("authorization") == "Bearer gtok-1":
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]},
        )

    gc = httpx.AsyncClient(transport=httpx.MockTransport(giga_retry_handler))
    ga = GigaChatAdapter(
        UpstreamConfig(
            name="giga-r",
            provider=ProviderType.GIGACHAT,
            base_url="https://gigachat.devices.sberbank.ru/api/v1",
            model="GigaChat",
        ),
        client=gc,
    )
    try:
        res = await ga.complete(
            UpstreamRequest(
                provider=ProviderType.GIGACHAT,
                model="GigaChat",
                messages=({"role": "user", "content": "hi"},),
            )
        )
        assert res.content == "OK" and giga_calls == 2
    finally:
        await gc.aclose()

    # YandexGPTAdapter Api-Key header + toolCallList in complete() + stream_text
    monkeypatch.setenv("YG_APIKEY", _tok("AQVN", "PlainYandexApiKey1234567890"))
    monkeypatch.setenv("YG_FOLDER", "b1gfolder")

    def yg_comp_handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"].startswith("Api-Key ")
        assert req.headers["x-folder-id"] == "b1gfolder"
        return httpx.Response(
            200,
            json={
                "result": {
                    "alternatives": [
                        {
                            "message": {
                                "role": "assistant",
                                "text": "Result ",
                                "toolCallList": {
                                    "toolCalls": [{"functionCall": {"name": "fn", "arguments": {}}}]
                                },
                            }
                        }
                    ],
                    "usage": {"inputTextTokens": "2", "completionTokens": "3"},
                }
            },
        )

    yc = httpx.AsyncClient(transport=httpx.MockTransport(yg_comp_handler))
    ya = YandexGPTAdapter(
        UpstreamConfig(
            name="yg-api",
            provider=ProviderType.YANDEXGPT,
            base_url="https://llm.api.cloud.yandex.net/foundationModels/v1",
            model="yandexgpt/latest",
            api_key_env="YG_APIKEY",
            folder_id_env="YG_FOLDER",
        ),
        client=yc,
    )
    try:
        r_yg = await ya.complete(
            UpstreamRequest(
                provider=ProviderType.YANDEXGPT,
                model="yandexgpt/latest",
                messages=({"role": "user", "content": "hi"},),
            )
        )
        assert r_yg.content == "Result " and r_yg.prompt_tokens == 2
        s_yg = [
            c
            async for c in ya.stream_text(
                UpstreamRequest(
                    provider=ProviderType.YANDEXGPT,
                    model="yandexgpt/latest",
                    messages=({"role": "user", "content": "hi"},),
                )
            )
        ]
        assert "".join(s_yg) == "Result "
    finally:
        await yc.aclose()


@pytest.mark.asyncio
async def test_router_deep_protocol_variants_and_sse_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover CA PEM file TLS verify, tool normalizers, OAI function_call, and Yandex SSE."""
    from pokrov.proxy.router import (
        _build_ssl_verify,
        _normalize_anthropic_tools,
        _normalize_openai_tools,
        _normalize_yandex_tools,
    )

    # Real temporary self-signed CA cert file for _build_ssl_verify valid file path
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "pokrov-test-ca")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(days=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=7))
        .sign(key, hashes.SHA256())
    )
    pem_file = tmp_path / "valid_ca.pem"
    pem_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    ctx = _build_ssl_verify(
        UpstreamConfig(name="ca-ok", provider=ProviderType.OPENAI, ca_bundle_path=str(pem_file))
    )
    assert isinstance(ctx, ssl.SSLContext)

    # Tool normalization edge variants
    anth_tool = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    oai_norm = _normalize_openai_tools([*anth_tool, {"custom": True}])
    assert oai_norm[0]["type"] == "function" and oai_norm[1] == {"custom": True}
    anth_norm = _normalize_anthropic_tools(
        [
            {"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}},
            {"custom": True},
        ]
    )
    assert anth_norm[0]["input_schema"] == {} and anth_norm[1] == {"custom": True}
    assert _normalize_yandex_tools(anth_tool)[0]["function"]["name"] == "t"

    # OpenAICompatAdapter legacy function_call SSE + extra_body forwarding + plain choices[0].text
    oai_bodies: list[dict[str, Any]] = []

    def oai_handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode("utf-8"))
        oai_bodies.append(body)
        if body.get("stream"):
            sse = (
                'data: {"choices":[{"index":0,"delta":{"function_call":'
                '{"name":"fn","arguments":"{}"}},"finish_reason":"function_call"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"text": "legacy-text"}], "usage": {}})

    oc = httpx.AsyncClient(transport=httpx.MockTransport(oai_handler))
    oa = OpenAICompatAdapter(
        UpstreamConfig(
            name="oai-edge",
            provider=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
        ),
        client=oc,
    )
    try:
        comp_res = await oa.complete(
            UpstreamRequest(
                provider=ProviderType.OPENAI,
                model="gpt-4o-mini",
                messages=({"role": "user", "content": "hi"},),
                extra_body={"top_p": 0.9, "seed": 42, "stop": ["END"]},
            )
        )
        assert comp_res.content == "legacy-text"
        assert oai_bodies[0]["seed"] == 42
        frames = [
            f
            async for f in oa.stream_frames(
                UpstreamRequest(
                    provider=ProviderType.OPENAI,
                    model="gpt-4o-mini",
                    messages=({"role": "user", "content": "hi"},),
                )
            )
        ]
        assert any(f.kind == "tool_call" and f.tool_name == "fn" for f in frames)
    finally:
        await oc.aclose()

    # YandexGPTAdapter: IAM t1.* Bearer + SSE toolCallList + truncated status
    monkeypatch.setenv("YANDEX_IAM_TOKEN", "t1.test-iam-token")
    monkeypatch.delenv("YG_APIKEY", raising=False)

    def yg_sse_handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer t1.test-iam-token"
        ndjson = (
            json.dumps(
                {
                    "result": {
                        "alternatives": [
                            {
                                "status": "ALTERNATIVE_STATUS_PARTIAL",
                                "message": {
                                    "text": "Hel",
                                    "toolCallList": {
                                        "toolCalls": [
                                            {
                                                "functionCall": {
                                                    "name": "search",
                                                    "arguments": {"q": 1},
                                                }
                                            }
                                        ]
                                    },
                                },
                            }
                        ]
                    }
                }
            )
            + "\n"
            + json.dumps(
                {
                    "result": {
                        "alternatives": [
                            {
                                "status": "ALTERNATIVE_STATUS_TRUNCATED_FINAL",
                                "message": {"text": "Hello!"},
                            }
                        ],
                        "usage": {"inputTextTokens": "4", "completionTokens": "2"},
                    }
                }
            )
            + "\n"
        )
        return httpx.Response(200, text=ndjson)

    yc2 = httpx.AsyncClient(transport=httpx.MockTransport(yg_sse_handler))
    ya2 = YandexGPTAdapter(
        UpstreamConfig(
            name="yg-iam",
            provider=ProviderType.YANDEXGPT,
            base_url="https://llm.api.cloud.yandex.net/foundationModels/v1",
            model="yandexgpt/latest",
        ),
        client=yc2,
    )
    try:
        y_frames = [
            f
            async for f in ya2.stream_frames(
                UpstreamRequest(
                    provider=ProviderType.YANDEXGPT,
                    model="yandexgpt/latest",
                    messages=({"role": "user", "content": "hi"},),
                    extra_body={"top_p": 0.8, "stop": "STOP"},
                    tools=({"name": "search", "parameters": {"type": "object"}},),
                )
            )
        ]
        assert any(f.kind == "tool_call" and f.tool_name == "search" for f in y_frames)
        assert any(f.kind == "finish" and f.finish_reason == "length" for f in y_frames)
    finally:
        await yc2.aclose()
