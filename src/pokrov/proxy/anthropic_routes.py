"""Anthropic-compatible ``/v1/messages`` HTTP and SSE routes (Claude Code compatible)."""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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

logger = logging.getLogger("pokrov.proxy.anthropic")

router = APIRouter(tags=["anthropic"])


class AnthropicMessagesRequest(BaseModel):
    """Anthropic ``/v1/messages`` request model."""

    model_config = ConfigDict(extra="allow")

    model: str = "claude-3-5-sonnet-latest"
    max_tokens: int = Field(default=1024, ge=1)
    messages: list[dict[str, Any]] = Field(min_length=1)
    system: str | list[dict[str, Any]] | None = None
    stream: bool = False
    temperature: float | None = None
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _AnthropicTarget:
    kind: Literal["system_str", "system_block", "msg_str", "msg_block", "tool_str", "tool_block"]
    msg_index: int
    block_index: int | None
    sub_index: int | None
    text: str
    control_point: ControlPoint


def _extract_anthropic_targets(
    system: str | list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
) -> list[_AnthropicTarget]:
    """Extract all inspectable text spans from Anthropic system and message blocks."""
    targets: list[_AnthropicTarget] = []

    if isinstance(system, str):
        if system:
            targets.append(
                _AnthropicTarget(
                    kind="system_str",
                    msg_index=-1,
                    block_index=None,
                    sub_index=None,
                    text=system,
                    control_point=ControlPoint.USER_INPUT,
                )
            )
    elif isinstance(system, list):
        for block_idx, block in enumerate(system):
            if isinstance(block, dict):
                text_val = block.get("text")
                if isinstance(text_val, str) and text_val:
                    targets.append(
                        _AnthropicTarget(
                            kind="system_block",
                            msg_index=-1,
                            block_index=block_idx,
                            sub_index=None,
                            text=text_val,
                            control_point=ControlPoint.USER_INPUT,
                        )
                    )

    for msg_idx, msg in enumerate(messages):
        role = str(msg.get("role", "user")).lower()
        default_cp = ControlPoint.TOOL_RESULT if role == "tool" else ControlPoint.USER_INPUT
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                targets.append(
                    _AnthropicTarget(
                        kind="msg_str",
                        msg_index=msg_idx,
                        block_index=None,
                        sub_index=None,
                        text=content,
                        control_point=default_cp,
                    )
                )
        elif isinstance(content, list):
            for block_idx, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                btype = str(block.get("type", "")).lower()
                if btype == "tool_result":
                    tr_content = block.get("content")
                    if isinstance(tr_content, str):
                        if tr_content:
                            targets.append(
                                _AnthropicTarget(
                                    kind="tool_str",
                                    msg_index=msg_idx,
                                    block_index=block_idx,
                                    sub_index=None,
                                    text=tr_content,
                                    control_point=ControlPoint.TOOL_RESULT,
                                )
                            )
                    elif isinstance(tr_content, list):
                        for sub_idx, sub_block in enumerate(tr_content):
                            if isinstance(sub_block, dict):
                                sub_text = sub_block.get("text")
                                if isinstance(sub_text, str) and sub_text:
                                    targets.append(
                                        _AnthropicTarget(
                                            kind="tool_block",
                                            msg_index=msg_idx,
                                            block_index=block_idx,
                                            sub_index=sub_idx,
                                            text=sub_text,
                                            control_point=ControlPoint.TOOL_RESULT,
                                        )
                                    )
                else:
                    text_val = block.get("text")
                    if isinstance(text_val, str) and text_val:
                        targets.append(
                            _AnthropicTarget(
                                kind="msg_block",
                                msg_index=msg_idx,
                                block_index=block_idx,
                                sub_index=None,
                                text=text_val,
                                control_point=default_cp,
                            )
                        )
    return targets


def _apply_anthropic_masked_texts(
    system: str | list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    targets: list[_AnthropicTarget],
    masked_texts: list[str],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Write masked texts back into Anthropic system prompt and message blocks."""
    copied_messages: list[dict[str, Any]] = copy.deepcopy(messages)
    copied_system: str | list[dict[str, Any]] | None = copy.deepcopy(system)

    for target, masked in zip(targets, masked_texts, strict=True):
        if target.kind == "system_str":
            copied_system = masked
        elif (
            target.kind == "system_block"
            and isinstance(copied_system, list)
            and target.block_index is not None
        ):
            copied_system[target.block_index]["text"] = masked
        elif target.kind == "msg_str":
            copied_messages[target.msg_index]["content"] = masked
        elif target.kind == "msg_block" and target.block_index is not None:
            copied_messages[target.msg_index]["content"][target.block_index]["text"] = masked
        elif target.kind == "tool_str" and target.block_index is not None:
            copied_messages[target.msg_index]["content"][target.block_index]["content"] = masked
        elif (
            target.kind == "tool_block"
            and target.block_index is not None
            and target.sub_index is not None
        ):
            copied_messages[target.msg_index]["content"][target.block_index]["content"][
                target.sub_index
            ]["text"] = masked

    if isinstance(copied_system, str):
        system_str: str | None = copied_system
    elif isinstance(copied_system, list):
        parts = [
            str(b.get("text", ""))
            for b in copied_system
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        ]
        system_str = "\n".join(parts) if parts else None
    else:
        system_str = None

    return system_str, copied_messages


@router.post("/v1/messages", response_model=None)
async def create_message(
    payload: AnthropicMessagesRequest,
    request: Request,
) -> JSONResponse | StreamingResponse:
    """Anthropic ``/v1/messages`` endpoint supporting Claude Code and ``tool_result`` blocks."""
    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    gateway: GatewayRuntime = request.app.state.gateway

    body_user: str | None = None
    body_session: str | None = None
    if isinstance(payload.metadata, dict):
        if isinstance(payload.metadata.get("user_id"), str):
            body_user = payload.metadata["user_id"]
        if isinstance(payload.metadata.get("session_id"), str):
            body_session = payload.metadata["session_id"]

    principal = gateway.extract_principal(request.headers, body_user=body_user)
    session_id = gateway.extract_session_id(
        request.headers,
        fallback_id=request_id,
        body_session=body_session,
    )

    try:
        profile, prof_cfg = gateway.resolve_profile(request.headers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    targets = _extract_anthropic_targets(payload.system, payload.messages)
    items = [(target.text, target.control_point) for target in targets]
    roles = [
        "system"
        if target.msg_index < 0
        else (
            "tool"
            if target.control_point == ControlPoint.TOOL_RESULT
            else str(payload.messages[target.msg_index].get("role", "user")).lower()
        )
        for target in targets
    ]
    prompt_chars = sum(len(text) for text, _ in items)

    quota_resp = await gateway.enforce_quota(
        principal=principal,
        prompt_chars=prompt_chars,
        request_id=request_id,
        session_id=session_id,
        started=started,
        api_format="anthropic",
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
            api_format="anthropic",
        )

    masked_texts, _ = gateway.apply_input_masking(
        items=items,
        inspections=inspections,
        decision=decision,
        session_id=session_id,
        agg_inspection=agg_inspection,
    )
    masked_system, masked_messages = _apply_anthropic_masked_texts(
        payload.system,
        payload.messages,
        targets,
        masked_texts,
    )

    requested_upstream = gateway.extract_requested_upstream(request.headers)
    adapter = gateway.services.resolve_upstream(decision, requested_upstream)

    extra_body: dict[str, Any] = dict(payload.model_extra) if payload.model_extra else {}

    upstream_req = UpstreamRequest(
        provider=adapter.provider,
        model=payload.model,
        messages=tuple(masked_messages),
        stream=payload.stream,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        extra_body=extra_body,
        api_format="anthropic",
        system=masked_system,
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
            _stream_anthropic_sse(
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
            "Upstream Anthropic completion failed request_id=%s upstream=%s error=%s",
            request_id,
            adapter.name,
            type(exc).__name__,
        )
        return gateway.build_upstream_error_response(
            request_id=request_id,
            upstream=adapter.name,
            started=started,
            detections_count=detections_count,
            api_format="anthropic",
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
            api_format="anthropic",
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

    response_body: dict[str, Any] = {
        "id": f"msg_{request_id}",
        "type": "message",
        "role": "assistant",
        "model": upstream_resp.model or payload.model,
        "content": [{"type": "text", "text": restored_text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": upstream_resp.prompt_tokens,
            "output_tokens": upstream_resp.completion_tokens,
        },
    }
    return JSONResponse(status_code=200, content=response_body, headers=headers)


def _format_sse_event(event_name: str, data: dict[str, Any]) -> str:
    """Serialize a single Anthropic SSE event frame."""
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_anthropic_sse(
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
    """Stream Anthropic SSE frames through ``SlidingWindowStreamGuard``."""
    msg_id = f"msg_{request_id}"
    model_name = upstream_req.model
    guard = gateway.make_stream_guard(
        session_id=session_id,
        principal=principal,
        profile=profile,
    )

    yield _format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield _format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )

    try:
        async for raw_chunk in adapter.stream_text(upstream_req):
            chunk_decision = guard.feed(raw_chunk)
            if chunk_decision.emit_text:
                yield _format_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": chunk_decision.emit_text},
                    },
                )

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
                yield _format_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": 0},
                )
                yield _format_sse_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "content_filter", "stop_sequence": None},
                        "usage": {"output_tokens": 0},
                        "pokrov_blocked": True,
                        "pokrov_reason": chunk_decision.block_reason,
                    },
                )
                yield _format_sse_event("message_stop", {"type": "message_stop"})
                return

        final_decision = guard.flush()
        if final_decision.emit_text:
            yield _format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": final_decision.emit_text},
                },
            )

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
            yield _format_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            )
            yield _format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "content_filter", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                    "pokrov_blocked": True,
                    "pokrov_reason": final_decision.block_reason,
                },
            )
            yield _format_sse_event("message_stop", {"type": "message_stop"})
            return

        yield _format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
        yield _format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        )
        yield _format_sse_event("message_stop", {"type": "message_stop"})
    except Exception as exc:
        logger.warning(
            "Upstream Anthropic SSE stream failed request_id=%s upstream=%s error=%s",
            request_id,
            adapter.name,
            type(exc).__name__,
        )
        yield _format_sse_event(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": "Upstream stream interrupted"},
            },
        )
        yield _format_sse_event("message_stop", {"type": "message_stop"})


__all__ = ["AnthropicMessagesRequest", "router"]
