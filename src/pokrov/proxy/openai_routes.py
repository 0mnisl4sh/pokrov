"""OpenAI-compatible HTTP and SSE routes for Pokrov LLM Security Gateway."""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from pokrov.core.contracts import (
    Action,
    ControlPoint,
    UpstreamRequest,
)

if TYPE_CHECKING:
    from pokrov.proxy.app import GatewayRuntime

logger = logging.getLogger("pokrov.proxy.openai")

router = APIRouter(tags=["openai"])


class ChatCompletionRequest(BaseModel):
    """OpenAI ``/v1/chat/completions`` request model."""

    model_config = ConfigDict(extra="allow")

    model: str = "pokrov-default"
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None


_TextTarget = tuple[int, int | None, str, ControlPoint]


def _extract_openai_targets(messages: list[dict[str, Any]]) -> list[_TextTarget]:
    """Extract all inspectable text fields across user, system, assistant, and tool messages."""
    targets: list[_TextTarget] = []
    for msg_idx, msg in enumerate(messages):
        role = str(msg.get("role", "user")).lower()
        control_point = ControlPoint.TOOL_RESULT if role == "tool" else ControlPoint.USER_INPUT
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                targets.append((msg_idx, None, content, control_point))
        elif isinstance(content, list):
            for block_idx, block in enumerate(content):
                if isinstance(block, dict):
                    text_val = block.get("text")
                    if isinstance(text_val, str) and text_val:
                        targets.append((msg_idx, block_idx, text_val, control_point))
    return targets


def _apply_openai_masked_texts(
    messages: list[dict[str, Any]],
    targets: list[_TextTarget],
    masked_texts: list[str],
) -> list[dict[str, Any]]:
    """Return a copy of OpenAI messages with masked strings written back."""
    copied: list[dict[str, Any]] = copy.deepcopy(messages)
    for (msg_idx, block_idx, _orig, _cp), masked in zip(targets, masked_texts, strict=True):
        if block_idx is None:
            copied[msg_idx]["content"] = masked
        else:
            content_list = copied[msg_idx]["content"]
            if isinstance(content_list, list) and isinstance(content_list[block_idx], dict):
                content_list[block_idx]["text"] = masked
    return copied


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List configured upstream models in OpenAI ``/v1/models`` format."""
    gateway: GatewayRuntime = request.app.state.gateway
    return {
        "object": "list",
        "data": gateway.services.list_models(),
    }


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
) -> JSONResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint with input/output security guardrails."""
    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    gateway: GatewayRuntime = request.app.state.gateway

    principal = gateway.extract_principal(request.headers, body_user=payload.user)
    body_session = None
    if payload.model_extra and isinstance(payload.model_extra.get("session_id"), str):
        body_session = payload.model_extra["session_id"]
    session_id = gateway.extract_session_id(
        request.headers,
        fallback_id=request_id,
        body_session=body_session,
    )

    try:
        profile, prof_cfg = gateway.resolve_profile(request.headers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    targets = _extract_openai_targets(payload.messages)
    items = [(text, cp) for _m, _b, text, cp in targets]
    roles = [str(payload.messages[m].get("role", "user")).lower() for m, _b, _t, _cp in targets]
    prompt_chars = sum(len(text) for text, _ in items)

    quota_resp = await gateway.enforce_quota(
        principal=principal,
        prompt_chars=prompt_chars,
        request_id=request_id,
        session_id=session_id,
        started=started,
        api_format="openai",
        control_point=ControlPoint.USER_INPUT,
        profile=profile,
    )
    if quota_resp is not None:
        return quota_resp

    inspections, agg_inspection, decision, primary_cp = gateway.inspect_items(
        items=items,
        session_id=session_id,
        principal=principal,
        profile=profile,
        prof_cfg=prof_cfg,
        roles=roles,
    )
    detections_count = len(agg_inspection.detections)

    if decision.action in (Action.BLOCK, Action.REQUIRE_APPROVAL):
        await gateway.emit_decision_audit(
            request_id=request_id,
            session_id=session_id,
            principal=principal,
            control_point=primary_cp,
            decision=decision,
            inspection=agg_inspection,
            upstream="none",
            profile=profile,
            started=started,
        )
        return gateway.build_block_response(
            decision=decision,
            request_id=request_id,
            started=started,
            detections_count=detections_count,
            api_format="openai",
        )

    masked_texts, _ = gateway.apply_input_masking(
        items=items,
        inspections=inspections,
        decision=decision,
        session_id=session_id,
        agg_inspection=agg_inspection,
    )
    masked_messages = _apply_openai_masked_texts(payload.messages, targets, masked_texts)

    requested_upstream = gateway.extract_requested_upstream(request.headers)
    adapter = gateway.services.resolve_upstream(decision, requested_upstream)

    extra_body: dict[str, Any] = dict(payload.model_extra) if payload.model_extra else {}
    extra_body.pop("session_id", None)

    upstream_req = UpstreamRequest(
        provider=adapter.provider,
        model=payload.model,
        messages=tuple(masked_messages),
        stream=payload.stream,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        extra_body=extra_body,
        api_format="openai",
        tools=tuple(payload.tools) if payload.tools else None,
    )

    if payload.stream:
        await gateway.emit_decision_audit(
            request_id=request_id,
            session_id=session_id,
            principal=principal,
            control_point=primary_cp,
            decision=decision,
            inspection=agg_inspection,
            upstream=adapter.name,
            profile=profile,
            started=started,
        )
        headers = gateway.build_headers(
            action=decision.action,
            upstream=adapter.name,
            request_id=request_id,
            started=started,
            detections_count=detections_count,
        )
        headers["Cache-Control"] = "no-cache"
        headers["Connection"] = "keep-alive"
        headers["X-Accel-Buffering"] = "no"

        return StreamingResponse(
            _stream_openai_sse(
                gateway=gateway,
                adapter=adapter,
                upstream_req=upstream_req,
                request_id=request_id,
                session_id=session_id,
                principal=principal,
                profile=profile,
                started=started,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        upstream_resp = await adapter.complete(upstream_req)
    except Exception as exc:
        logger.warning(
            "Upstream completion failed request_id=%s upstream=%s error=%s",
            request_id,
            adapter.name,
            type(exc).__name__,
        )
        return gateway.build_upstream_error_response(
            request_id=request_id,
            upstream=adapter.name,
            started=started,
            detections_count=detections_count,
            api_format="openai",
        )

    (
        restored_text,
        out_blocked,
        out_reason,
        out_dets,
        out_action,
    ) = await gateway.guard_and_restore_output(
        content=upstream_resp.content,
        session_id=session_id,
        principal=principal,
        profile=profile,
        prof_cfg=prof_cfg,
    )

    if out_blocked:
        out_cats = await gateway.emit_output_block_audit(
            request_id=request_id,
            session_id=session_id,
            principal=principal,
            detections=out_dets,
            reason=out_reason,
            upstream=adapter.name,
            profile=profile,
            started=started,
            action=out_action,
        )
        return gateway.build_output_block_response(
            reason=out_reason,
            categories=out_cats,
            request_id=request_id,
            upstream=adapter.name,
            started=started,
            detections_count=detections_count + len(out_dets),
            api_format="openai",
        )

    await gateway.emit_decision_audit(
        request_id=request_id,
        session_id=session_id,
        principal=principal,
        control_point=primary_cp,
        decision=decision,
        inspection=agg_inspection,
        upstream=adapter.name,
        profile=profile,
        started=started,
    )

    headers = gateway.build_headers(
        action=decision.action,
        upstream=adapter.name,
        request_id=request_id,
        started=started,
        detections_count=detections_count,
    )

    message_obj: dict[str, Any] = {"role": "assistant", "content": restored_text}
    raw_choices = upstream_resp.raw_payload.get("choices")
    if isinstance(raw_choices, list) and raw_choices and isinstance(raw_choices[0], dict):
        raw_msg = raw_choices[0].get("message")
        if isinstance(raw_msg, dict) and "tool_calls" in raw_msg:
            message_obj["tool_calls"] = raw_msg["tool_calls"]

    response_body: dict[str, Any] = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": upstream_resp.model or payload.model,
        "choices": [
            {
                "index": 0,
                "message": message_obj,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": upstream_resp.prompt_tokens,
            "completion_tokens": upstream_resp.completion_tokens,
            "total_tokens": upstream_resp.prompt_tokens + upstream_resp.completion_tokens,
        },
    }
    return JSONResponse(status_code=200, content=response_body, headers=headers)


async def _stream_openai_sse(
    *,
    gateway: GatewayRuntime,
    adapter: Any,
    upstream_req: UpstreamRequest,
    request_id: str,
    session_id: str,
    principal: str,
    profile: str,
    started: float,
) -> AsyncIterator[str]:
    """Stream OpenAI SSE chunks through ``SlidingWindowStreamGuard``."""
    cmpl_id = f"chatcmpl-{request_id}"
    created_ts = int(time.time())
    model_name = upstream_req.model
    guard = gateway.make_stream_guard(
        session_id=session_id,
        principal=principal,
        profile=profile,
    )
    first_chunk = True

    def _chunk_line(
        delta: dict[str, Any],
        finish_reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if extra:
            body.update(extra)
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    try:
        async for raw_chunk in adapter.stream_text(upstream_req):
            chunk_decision = guard.feed(raw_chunk)
            if chunk_decision.emit_text:
                delta: dict[str, Any] = {"content": chunk_decision.emit_text}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False
                yield _chunk_line(delta)

            if chunk_decision.blocked:
                await gateway.emit_output_block_audit(
                    request_id=request_id,
                    session_id=session_id,
                    principal=principal,
                    detections=chunk_decision.detections,
                    reason=chunk_decision.block_reason,
                    upstream=adapter.name,
                    profile=profile,
                    started=started,
                )
                yield _chunk_line(
                    {},
                    finish_reason="content_filter",
                    extra={
                        "pokrov_blocked": True,
                        "pokrov_reason": chunk_decision.block_reason,
                    },
                )
                yield "data: [DONE]\n\n"
                return

        final_decision = guard.flush()
        if final_decision.emit_text:
            delta = {"content": final_decision.emit_text}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            yield _chunk_line(delta)

        if final_decision.blocked:
            await gateway.emit_output_block_audit(
                request_id=request_id,
                session_id=session_id,
                principal=principal,
                detections=final_decision.detections,
                reason=final_decision.block_reason,
                upstream=adapter.name,
                profile=profile,
                started=started,
            )
            yield _chunk_line(
                {},
                finish_reason="content_filter",
                extra={
                    "pokrov_blocked": True,
                    "pokrov_reason": final_decision.block_reason,
                },
            )
            yield "data: [DONE]\n\n"
            return

        yield _chunk_line({}, finish_reason="stop")
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.warning(
            "Upstream SSE stream failed request_id=%s upstream=%s error=%s",
            request_id,
            adapter.name,
            type(exc).__name__,
        )
        yield _chunk_line(
            {},
            finish_reason="error",
            extra={"error": {"message": "Upstream stream interrupted", "type": "upstream_error"}},
        )
        yield "data: [DONE]\n\n"


__all__ = ["ChatCompletionRequest", "router"]
