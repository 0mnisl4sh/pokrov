"""OpenAI-compatible HTTP and SSE routes for Pokrov LLM Security Gateway."""

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
    Detection,
    StreamFrame,
    UpstreamRequest,
)

if TYPE_CHECKING:
    from pokrov.core.config import ProfileConfig
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


@dataclass(frozen=True)
class _OpenAITarget:
    kind: Literal[
        "msg_str",
        "msg_block",
        "tool_call_args",
        "tool_call_dict_leaf",
        "function_call_args",
        "tool_desc",
        "tool_param_leaf",
    ]
    msg_index: int
    sub_index: int | None
    json_path: tuple[str | int, ...]
    text: str
    control_point: ControlPoint
    role: str


def _collect_json_string_leaves(
    node: Any,
    prefix: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], str]]:
    leaves: list[tuple[tuple[str | int, ...], str]] = []
    if isinstance(node, str):
        if node:
            leaves.append((prefix, node))
    elif isinstance(node, dict):
        for k, v in node.items():
            leaves.extend(_collect_json_string_leaves(v, (*prefix, str(k))))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves.extend(_collect_json_string_leaves(v, (*prefix, i)))
    return leaves


def _write_json_leaf(root: Any, path: tuple[str | int, ...], new_val: str) -> Any:
    if not path:
        return new_val
    cur = root
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = new_val
    return root


def _extract_openai_targets(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[_OpenAITarget]:
    """Extract all inspectable text fields across messages, tool_calls, and tool schemas."""
    targets: list[_OpenAITarget] = []
    for msg_idx, msg in enumerate(messages):
        role = str(msg.get("role", "user")).lower()
        control_point = ControlPoint.TOOL_RESULT if role == "tool" else ControlPoint.USER_INPUT
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                targets.append(
                    _OpenAITarget(
                        kind="msg_str",
                        msg_index=msg_idx,
                        sub_index=None,
                        json_path=(),
                        text=content,
                        control_point=control_point,
                        role=role,
                    )
                )
        elif isinstance(content, list):
            for block_idx, block in enumerate(content):
                if isinstance(block, dict):
                    text_val = block.get("text")
                    if isinstance(text_val, str) and text_val:
                        targets.append(
                            _OpenAITarget(
                                kind="msg_block",
                                msg_index=msg_idx,
                                sub_index=block_idx,
                                json_path=(),
                                text=text_val,
                                control_point=control_point,
                                role=role,
                            )
                        )

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc_idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if isinstance(args, str) and args:
                    targets.append(
                        _OpenAITarget(
                            kind="tool_call_args",
                            msg_index=msg_idx,
                            sub_index=tc_idx,
                            json_path=(),
                            text=args,
                            control_point=ControlPoint.TOOL_CALL,
                            role="assistant",
                        )
                    )
                elif isinstance(args, dict):
                    for path, leaf_str in _collect_json_string_leaves(args):
                        targets.append(
                            _OpenAITarget(
                                kind="tool_call_dict_leaf",
                                msg_index=msg_idx,
                                sub_index=tc_idx,
                                json_path=path,
                                text=leaf_str,
                                control_point=ControlPoint.TOOL_CALL,
                                role="assistant",
                            )
                        )

        fn_call = msg.get("function_call")
        if isinstance(fn_call, dict):
            fc_args = fn_call.get("arguments")
            if isinstance(fc_args, str) and fc_args:
                targets.append(
                    _OpenAITarget(
                        kind="function_call_args",
                        msg_index=msg_idx,
                        sub_index=None,
                        json_path=(),
                        text=fc_args,
                        control_point=ControlPoint.TOOL_CALL,
                        role="assistant",
                    )
                )

    if tools:
        for t_idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                continue
            fn_obj = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if not isinstance(fn_obj, dict):
                continue
            desc = fn_obj.get("description")
            if isinstance(desc, str) and desc:
                targets.append(
                    _OpenAITarget(
                        kind="tool_desc",
                        msg_index=-1,
                        sub_index=t_idx,
                        json_path=(),
                        text=desc,
                        control_point=ControlPoint.TOOL_CALL,
                        role="system",
                    )
                )
            params = fn_obj.get("parameters")
            if isinstance(params, dict):
                for path, leaf_str in _collect_json_string_leaves(params):
                    targets.append(
                        _OpenAITarget(
                            kind="tool_param_leaf",
                            msg_index=-1,
                            sub_index=t_idx,
                            json_path=path,
                            text=leaf_str,
                            control_point=ControlPoint.TOOL_CALL,
                            role="system",
                        )
                    )

    return targets


def _apply_openai_masked_texts(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    targets: list[_OpenAITarget],
    masked_texts: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return copies of OpenAI ``messages`` and ``tools`` with masked strings written back."""
    copied_msgs: list[dict[str, Any]] = copy.deepcopy(messages)
    copied_tools: list[dict[str, Any]] | None = copy.deepcopy(tools) if tools is not None else None

    for target, masked in zip(targets, masked_texts, strict=True):
        if target.kind == "msg_str":
            copied_msgs[target.msg_index]["content"] = masked
        elif target.kind == "msg_block" and target.sub_index is not None:
            content_list = copied_msgs[target.msg_index].get("content")
            if isinstance(content_list, list) and isinstance(content_list[target.sub_index], dict):
                content_list[target.sub_index]["text"] = masked
        elif target.kind == "tool_call_args" and target.sub_index is not None:
            copied_msgs[target.msg_index]["tool_calls"][target.sub_index]["function"][
                "arguments"
            ] = masked
        elif target.kind == "tool_call_dict_leaf" and target.sub_index is not None:
            args_dict = copied_msgs[target.msg_index]["tool_calls"][target.sub_index]["function"][
                "arguments"
            ]
            _write_json_leaf(args_dict, target.json_path, masked)
        elif target.kind == "function_call_args":
            copied_msgs[target.msg_index]["function_call"]["arguments"] = masked
        elif (
            target.kind == "tool_desc" and copied_tools is not None and target.sub_index is not None
        ):
            t_obj = copied_tools[target.sub_index]
            if isinstance(t_obj.get("function"), dict):
                t_obj["function"]["description"] = masked
            else:
                t_obj["description"] = masked
        elif (
            target.kind == "tool_param_leaf"
            and copied_tools is not None
            and target.sub_index is not None
        ):
            t_obj = copied_tools[target.sub_index]
            p_dict = (
                t_obj["function"]["parameters"]
                if isinstance(t_obj.get("function"), dict)
                else t_obj["parameters"]
            )
            _write_json_leaf(p_dict, target.json_path, masked)

    return copied_msgs, copied_tools


async def _iter_adapter_frames(
    adapter: Any,
    upstream_req: UpstreamRequest,
) -> AsyncIterator[StreamFrame]:
    """Yield ``StreamFrame`` items, honoring instance-level ``stream_text`` monkeypatches."""
    inst_dict = getattr(adapter, "__dict__", {})
    stream_frames_fn = getattr(adapter, "stream_frames", None)
    if callable(stream_frames_fn) and "stream_text" not in inst_dict:
        async for frame in stream_frames_fn(upstream_req):
            yield frame
        return

    async for raw_chunk in adapter.stream_text(upstream_req):
        if raw_chunk:
            yield StreamFrame(kind="text", text=raw_chunk, index=0)
    yield StreamFrame(kind="finish", finish_reason="stop")


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List configured upstream models in OpenAI ``/v1/models`` format."""
    gateway: GatewayRuntime = request.app.state.gateway
    gateway.verify_gateway_auth(request.headers)
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

    profile, prof_cfg = gateway.resolve_profile(request.headers)

    max_msgs = gateway.services.config.server.max_messages
    if len(payload.messages) > max_msgs:
        raise HTTPException(
            status_code=400,
            detail=f"messages length ({len(payload.messages)}) exceeds max_messages ({max_msgs}).",
        )

    principal = gateway.extract_principal(request.headers, body_user=payload.user)
    body_session = None
    if payload.model_extra and isinstance(payload.model_extra.get("session_id"), str):
        body_session = payload.model_extra["session_id"]
    session_id = gateway.extract_session_id(
        request.headers,
        fallback_id=request_id,
        body_session=body_session,
    )

    targets = _extract_openai_targets(payload.messages, payload.tools)
    items = [(t.text, t.control_point) for t in targets]
    roles = [t.role for t in targets]
    raw_body_len = len(await request.body())
    prompt_chars = max(sum(len(text) for text, _ in items), raw_body_len)

    quota_resp = await gateway.enforce_quota(
        principal=principal,
        prompt_chars=prompt_chars,
        request_id=request_id,
        session_id=session_id,
        started=started,
        api_format="openai",
        control_point=ControlPoint.USER_INPUT,
        profile=profile,
        prof_cfg=prof_cfg,
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
        principal=principal,
        agg_inspection=agg_inspection,
        roles=roles,
        prof_cfg=prof_cfg,
    )
    masked_messages, masked_tools = _apply_openai_masked_texts(
        payload.messages,
        payload.tools,
        targets,
        masked_texts,
    )

    explicit_upstream = gateway.extract_requested_upstream(request.headers)
    model_candidate = payload.model if payload.model not in {"", "pokrov-default"} else None
    adapter = gateway.select_upstream(
        decision,
        explicit_upstream or model_candidate,
        allow_cloud=prof_cfg.allow_cloud_upstreams,
    )
    force_adapter_model = (
        decision.action == Action.ROUTE_LOCAL
        or (not prof_cfg.allow_cloud_upstreams and not adapter.is_local)
        or payload.model in {"", "pokrov-default"}
    )

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
        tools=tuple(masked_tools) if masked_tools else None,
        force_adapter_model=force_adapter_model,
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
                prof_cfg=prof_cfg,
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

    # Always emit input decision audit consistently in non-streaming mode too (P2-03)
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

    raw_choices = upstream_resp.raw_payload.get("choices")
    raw_first = (
        raw_choices[0]
        if isinstance(raw_choices, list) and raw_choices and isinstance(raw_choices[0], dict)
        else {}
    )
    raw_msg = raw_first.get("message") if isinstance(raw_first.get("message"), dict) else {}
    content_explicitly_null = (
        isinstance(raw_msg, dict)
        and "content" in raw_msg
        and raw_msg.get("content") is None
        and not upstream_resp.content
    )

    all_out_dets: list[Detection] = []
    if content_explicitly_null:
        restored_text: str | None = None
    else:
        (
            r_txt,
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
        all_out_dets.extend(out_dets)
        if out_blocked:
            out_cats = await gateway.emit_output_block_audit(
                request_id=request_id,
                session_id=session_id,
                principal=principal,
                detections=all_out_dets,
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
                detections_count=detections_count + len(all_out_dets),
                api_format="openai",
            )
        restored_text = r_txt

    message_obj: dict[str, Any] = {"role": "assistant", "content": restored_text}

    has_tool_calls = False
    if isinstance(raw_msg, dict) and isinstance(raw_msg.get("tool_calls"), list):
        copied_tcs = copy.deepcopy(raw_msg["tool_calls"])
        for tc in copied_tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                (
                    tc_args_restored,
                    tc_blocked,
                    tc_reason,
                    tc_dets,
                    tc_action,
                ) = await gateway.guard_and_restore_output(
                    content=fn["arguments"],
                    session_id=session_id,
                    principal=principal,
                    profile=profile,
                    prof_cfg=prof_cfg,
                )
                all_out_dets.extend(tc_dets)
                if tc_blocked:
                    out_cats = await gateway.emit_output_block_audit(
                        request_id=request_id,
                        session_id=session_id,
                        principal=principal,
                        detections=all_out_dets,
                        reason=tc_reason,
                        upstream=adapter.name,
                        profile=profile,
                        started=started,
                        action=tc_action,
                    )
                    return gateway.build_output_block_response(
                        reason=tc_reason,
                        categories=out_cats,
                        request_id=request_id,
                        upstream=adapter.name,
                        started=started,
                        detections_count=detections_count + len(all_out_dets),
                        api_format="openai",
                    )
                fn["arguments"] = tc_args_restored
        message_obj["tool_calls"] = copied_tcs
        has_tool_calls = bool(copied_tcs)

    if isinstance(raw_msg, dict) and isinstance(raw_msg.get("function_call"), dict):
        copied_fc = copy.deepcopy(raw_msg["function_call"])
        if isinstance(copied_fc.get("arguments"), str):
            (
                fc_args_restored,
                fc_blocked,
                fc_reason,
                fc_dets,
                fc_action,
            ) = await gateway.guard_and_restore_output(
                content=copied_fc["arguments"],
                session_id=session_id,
                principal=principal,
                profile=profile,
                prof_cfg=prof_cfg,
            )
            all_out_dets.extend(fc_dets)
            if fc_blocked:
                out_cats = await gateway.emit_output_block_audit(
                    request_id=request_id,
                    session_id=session_id,
                    principal=principal,
                    detections=all_out_dets,
                    reason=fc_reason,
                    upstream=adapter.name,
                    profile=profile,
                    started=started,
                    action=fc_action,
                )
                return gateway.build_output_block_response(
                    reason=fc_reason,
                    categories=out_cats,
                    request_id=request_id,
                    upstream=adapter.name,
                    started=started,
                    detections_count=detections_count + len(all_out_dets),
                    api_format="openai",
                )
            copied_fc["arguments"] = fc_args_restored
        message_obj["function_call"] = copied_fc

    raw_finish = raw_first.get("finish_reason") if isinstance(raw_first, dict) else None
    finish_reason = (
        str(raw_finish)
        if isinstance(raw_finish, str) and raw_finish
        else ("tool_calls" if has_tool_calls else "stop")
    )

    headers = gateway.build_headers(
        action=decision.action,
        upstream=adapter.name,
        request_id=request_id,
        started=started,
        detections_count=detections_count,
    )

    response_body: dict[str, Any] = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": upstream_resp.model or payload.model,
        "choices": [
            {
                "index": 0,
                "message": message_obj,
                "finish_reason": finish_reason,
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
    prof_cfg: ProfileConfig,
    started: float,
) -> AsyncIterator[str]:
    """Stream OpenAI SSE frames (text, thinking, tool_calls, usage, finish) through StreamGuard."""
    cmpl_id = f"chatcmpl-{request_id}"
    created_ts = int(time.time())
    model_name = upstream_req.model
    guard = gateway.make_stream_guard(
        session_id=session_id,
        principal=principal,
        profile=profile,
    )
    thinking_guard = gateway.make_stream_guard(
        session_id=session_id,
        principal=principal,
        profile=profile,
    )
    first_chunk = True
    accum_tool_calls: dict[int, dict[str, Any]] = {}
    seen_finish_reason: str | None = None
    last_usage: dict[str, int] | None = None

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
        async for frame in _iter_adapter_frames(adapter, upstream_req):
            if frame.kind == "text" and frame.text:
                chunk_decision = guard.feed(frame.text)
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

            elif frame.kind == "thinking" and frame.text:
                th_dec = thinking_guard.feed(frame.text)
                if th_dec.emit_text:
                    delta = {"reasoning_content": th_dec.emit_text}
                    if first_chunk:
                        delta["role"] = "assistant"
                        first_chunk = False
                    yield _chunk_line(delta)
                if th_dec.blocked:
                    await gateway.emit_output_block_audit(
                        request_id=request_id,
                        session_id=session_id,
                        principal=principal,
                        detections=th_dec.detections,
                        reason=th_dec.block_reason,
                        upstream=adapter.name,
                        profile=profile,
                        started=started,
                    )
                    yield _chunk_line(
                        {},
                        finish_reason="content_filter",
                        extra={"pokrov_blocked": True, "pokrov_reason": th_dec.block_reason},
                    )
                    yield "data: [DONE]\n\n"
                    return

            elif frame.kind == "tool_call":
                entry = accum_tool_calls.setdefault(
                    frame.index,
                    {"id": "", "name": "", "args": []},
                )
                if frame.tool_call_id:
                    entry["id"] = frame.tool_call_id
                if frame.tool_name:
                    entry["name"] = frame.tool_name
                if frame.arguments_delta:
                    entry["args"].append(frame.arguments_delta)

            elif frame.kind == "usage":
                p_tok = frame.prompt_tokens or 0
                c_tok = frame.completion_tokens or 0
                last_usage = {
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": p_tok + c_tok,
                }

            elif frame.kind == "finish" and frame.finish_reason:
                seen_finish_reason = frame.finish_reason

        th_final = thinking_guard.flush()
        if th_final.emit_text:
            delta = {"reasoning_content": th_final.emit_text}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            yield _chunk_line(delta)
        if th_final.blocked:
            await gateway.emit_output_block_audit(
                request_id=request_id,
                session_id=session_id,
                principal=principal,
                detections=th_final.detections,
                reason=th_final.block_reason,
                upstream=adapter.name,
                profile=profile,
                started=started,
            )
            yield _chunk_line(
                {},
                finish_reason="content_filter",
                extra={"pokrov_blocked": True, "pokrov_reason": th_final.block_reason},
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

        if accum_tool_calls:
            out_tcs: list[dict[str, Any]] = []
            for tc_idx in sorted(accum_tool_calls):
                tc_item = accum_tool_calls[tc_idx]
                raw_args = "".join(tc_item["args"])
                (
                    restored_args,
                    tc_blocked,
                    tc_reason,
                    tc_dets,
                    _tc_act,
                ) = await gateway.guard_and_restore_output(
                    content=raw_args,
                    session_id=session_id,
                    principal=principal,
                    profile=profile,
                    prof_cfg=prof_cfg,
                )
                if tc_blocked:
                    await gateway.emit_output_block_audit(
                        request_id=request_id,
                        session_id=session_id,
                        principal=principal,
                        detections=tc_dets,
                        reason=tc_reason,
                        upstream=adapter.name,
                        profile=profile,
                        started=started,
                    )
                    yield _chunk_line(
                        {},
                        finish_reason="content_filter",
                        extra={"pokrov_blocked": True, "pokrov_reason": tc_reason},
                    )
                    yield "data: [DONE]\n\n"
                    return
                out_tcs.append(
                    {
                        "index": tc_idx,
                        "id": tc_item["id"] or f"call_{tc_idx}",
                        "type": "function",
                        "function": {
                            "name": tc_item["name"],
                            "arguments": restored_args,
                        },
                    }
                )
            tc_delta: dict[str, Any] = {"tool_calls": out_tcs}
            if first_chunk:
                tc_delta["role"] = "assistant"
                first_chunk = False
            yield _chunk_line(tc_delta)

        final_fr = seen_finish_reason or ("tool_calls" if accum_tool_calls else "stop")
        extra_end: dict[str, Any] | None = None
        stream_opts = upstream_req.extra_body.get("stream_options")
        if (
            last_usage is not None
            and isinstance(stream_opts, dict)
            and stream_opts.get("include_usage")
        ):
            extra_end = {"usage": last_usage}

        yield _chunk_line({}, finish_reason=final_fr, extra=extra_end)
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
