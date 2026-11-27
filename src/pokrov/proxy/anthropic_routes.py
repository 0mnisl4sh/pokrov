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
    Detection,
    ProviderType,
    StreamFrame,
    UpstreamRequest,
)

if TYPE_CHECKING:
    from pokrov.core.config import ProfileConfig
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
    kind: Literal[
        "system_str",
        "system_block",
        "msg_str",
        "msg_block",
        "tool_str",
        "tool_block",
        "tool_use_leaf",
        "thinking_block",
        "doc_field",
        "doc_sub_block",
        "tool_desc",
        "tool_schema_leaf",
    ]
    msg_index: int
    block_index: int | None
    sub_index: int | None
    json_path: tuple[str | int, ...]
    text: str
    control_point: ControlPoint
    role: str


def _collect_json_leaves(
    node: Any,
    prefix: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], str]]:
    leaves: list[tuple[tuple[str | int, ...], str]] = []
    if isinstance(node, str):
        if node:
            leaves.append((prefix, node))
    elif isinstance(node, dict):
        for k, v in node.items():
            leaves.extend(_collect_json_leaves(v, (*prefix, str(k))))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves.extend(_collect_json_leaves(v, (*prefix, i)))
    return leaves


def _write_json_leaf(root: Any, path: tuple[str | int, ...], new_val: str) -> Any:
    if not path:
        return new_val
    cur = root
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = new_val
    return root


def _extract_anthropic_targets(
    system: str | list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[_AnthropicTarget]:
    """Extract text spans from system, messages, tool_use, thinking, document, and tools."""
    targets: list[_AnthropicTarget] = []

    if isinstance(system, str):
        if system:
            targets.append(
                _AnthropicTarget(
                    kind="system_str",
                    msg_index=-1,
                    block_index=None,
                    sub_index=None,
                    json_path=(),
                    text=system,
                    control_point=ControlPoint.USER_INPUT,
                    role="system",
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
                            json_path=(),
                            text=text_val,
                            control_point=ControlPoint.USER_INPUT,
                            role="system",
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
                        json_path=(),
                        text=content,
                        control_point=default_cp,
                        role=role,
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
                                    json_path=(),
                                    text=tr_content,
                                    control_point=ControlPoint.TOOL_RESULT,
                                    role="tool",
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
                                            json_path=(),
                                            text=sub_text,
                                            control_point=ControlPoint.TOOL_RESULT,
                                            role="tool",
                                        )
                                    )
                elif btype == "tool_use":
                    tu_input = block.get("input")
                    if isinstance(tu_input, dict | list | str):
                        for path, leaf_str in _collect_json_leaves(tu_input):
                            targets.append(
                                _AnthropicTarget(
                                    kind="tool_use_leaf",
                                    msg_index=msg_idx,
                                    block_index=block_idx,
                                    sub_index=None,
                                    json_path=path,
                                    text=leaf_str,
                                    control_point=ControlPoint.TOOL_CALL,
                                    role="assistant",
                                )
                            )
                elif btype == "thinking":
                    think_text = block.get("thinking")
                    if isinstance(think_text, str) and think_text:
                        targets.append(
                            _AnthropicTarget(
                                kind="thinking_block",
                                msg_index=msg_idx,
                                block_index=block_idx,
                                sub_index=None,
                                json_path=(),
                                text=think_text,
                                control_point=ControlPoint.USER_INPUT,
                                role=role,
                            )
                        )
                elif btype == "document":
                    for doc_field in ("title", "context"):
                        fval = block.get(doc_field)
                        if isinstance(fval, str) and fval:
                            targets.append(
                                _AnthropicTarget(
                                    kind="doc_field",
                                    msg_index=msg_idx,
                                    block_index=block_idx,
                                    sub_index=None,
                                    json_path=(doc_field,),
                                    text=fval,
                                    control_point=ControlPoint.RAG_CONTEXT,
                                    role=role,
                                )
                            )
                    source = block.get("source")
                    if isinstance(source, dict) and source.get("type") == "text":
                        sdata = source.get("data")
                        if isinstance(sdata, str) and sdata:
                            targets.append(
                                _AnthropicTarget(
                                    kind="doc_field",
                                    msg_index=msg_idx,
                                    block_index=block_idx,
                                    sub_index=None,
                                    json_path=("source", "data"),
                                    text=sdata,
                                    control_point=ControlPoint.RAG_CONTEXT,
                                    role=role,
                                )
                            )
                    doc_content = block.get("content")
                    if isinstance(doc_content, list):
                        for sub_idx, sub_b in enumerate(doc_content):
                            if (
                                isinstance(sub_b, dict)
                                and isinstance(sub_b.get("text"), str)
                                and sub_b["text"]
                            ):
                                targets.append(
                                    _AnthropicTarget(
                                        kind="doc_sub_block",
                                        msg_index=msg_idx,
                                        block_index=block_idx,
                                        sub_index=sub_idx,
                                        json_path=(),
                                        text=sub_b["text"],
                                        control_point=ControlPoint.RAG_CONTEXT,
                                        role=role,
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
                                json_path=(),
                                text=text_val,
                                control_point=default_cp,
                                role=role,
                            )
                        )

    if tools:
        for t_idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                continue
            desc = tool.get("description")
            if isinstance(desc, str) and desc:
                targets.append(
                    _AnthropicTarget(
                        kind="tool_desc",
                        msg_index=-2,
                        block_index=t_idx,
                        sub_index=None,
                        json_path=(),
                        text=desc,
                        control_point=ControlPoint.TOOL_CALL,
                        role="system",
                    )
                )
            schema = tool.get("input_schema")
            if isinstance(schema, dict):
                for path, leaf_str in _collect_json_leaves(schema):
                    targets.append(
                        _AnthropicTarget(
                            kind="tool_schema_leaf",
                            msg_index=-2,
                            block_index=t_idx,
                            sub_index=None,
                            json_path=path,
                            text=leaf_str,
                            control_point=ControlPoint.TOOL_CALL,
                            role="system",
                        )
                    )

    return targets


def _apply_anthropic_masked_texts(
    system: str | list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    targets: list[_AnthropicTarget],
    masked_texts: list[str],
) -> tuple[
    str | None,
    tuple[dict[str, Any], ...] | None,
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
]:
    """Write masked texts back while preserving ``system_blocks`` (including ``cache_control``)."""
    copied_messages: list[dict[str, Any]] = copy.deepcopy(messages)
    copied_system: str | list[dict[str, Any]] | None = copy.deepcopy(system)
    copied_tools: list[dict[str, Any]] | None = copy.deepcopy(tools) if tools is not None else None

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
        elif target.kind == "tool_use_leaf" and target.block_index is not None:
            blk = copied_messages[target.msg_index]["content"][target.block_index]
            blk["input"] = _write_json_leaf(blk["input"], target.json_path, masked)
        elif target.kind == "thinking_block" and target.block_index is not None:
            copied_messages[target.msg_index]["content"][target.block_index]["thinking"] = masked
        elif target.kind == "doc_field" and target.block_index is not None:
            blk = copied_messages[target.msg_index]["content"][target.block_index]
            _write_json_leaf(blk, target.json_path, masked)
        elif (
            target.kind == "doc_sub_block"
            and target.block_index is not None
            and target.sub_index is not None
        ):
            copied_messages[target.msg_index]["content"][target.block_index]["content"][
                target.sub_index
            ]["text"] = masked
        elif (
            target.kind == "tool_desc"
            and copied_tools is not None
            and target.block_index is not None
        ):
            copied_tools[target.block_index]["description"] = masked
        elif (
            target.kind == "tool_schema_leaf"
            and copied_tools is not None
            and target.block_index is not None
        ):
            _write_json_leaf(
                copied_tools[target.block_index]["input_schema"],
                target.json_path,
                masked,
            )

    if isinstance(copied_system, str):
        system_str: str | None = copied_system
        system_blocks: tuple[dict[str, Any], ...] | None = None
    elif isinstance(copied_system, list):
        valid_blocks = tuple(b for b in copied_system if isinstance(b, dict))
        system_blocks = valid_blocks if valid_blocks else None
        system_str = None
    else:
        system_str = None
        system_blocks = None

    return system_str, system_blocks, copied_messages, copied_tools


async def _guard_json_tree(
    node: Any,
    *,
    gateway: GatewayRuntime,
    session_id: str,
    principal: str,
    profile: str,
    prof_cfg: ProfileConfig,
) -> tuple[Any, bool, str, list[Detection], Action]:
    """Recursively run ``guard_and_restore_output`` across string leaves of ``tool_use.input``."""
    if isinstance(node, str):
        return await gateway.guard_and_restore_output(
            content=node,
            session_id=session_id,
            principal=principal,
            profile=profile,
            prof_cfg=prof_cfg,
        )
    if isinstance(node, dict):
        out_dict: dict[str, Any] = {}
        dets: list[Detection] = []
        last_act = Action.ALLOW
        for k, v in node.items():
            rv, blocked, reason, sub_dets, sub_act = await _guard_json_tree(
                v,
                gateway=gateway,
                session_id=session_id,
                principal=principal,
                profile=profile,
                prof_cfg=prof_cfg,
            )
            dets.extend(sub_dets)
            if blocked:
                return {}, True, reason, dets, sub_act
            out_dict[k] = rv
            last_act = sub_act
        return out_dict, False, "", dets, last_act
    if isinstance(node, list):
        out_list: list[Any] = []
        dets = []
        last_act = Action.ALLOW
        for item in node:
            rv, blocked, reason, sub_dets, sub_act = await _guard_json_tree(
                item,
                gateway=gateway,
                session_id=session_id,
                principal=principal,
                profile=profile,
                prof_cfg=prof_cfg,
            )
            dets.extend(sub_dets)
            if blocked:
                return [], True, reason, dets, sub_act
            out_list.append(rv)
            last_act = sub_act
        return out_list, False, "", dets, last_act
    return node, False, "", [], Action.ALLOW


def _map_anthropic_stop_reason(reason: str | None, *, has_tool_use: bool) -> str:
    if reason in {"end_turn", "max_tokens", "stop_sequence", "tool_use", "content_filter"}:
        return reason
    if reason == "tool_calls":
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    return "tool_use" if has_tool_use else "end_turn"


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
    yield StreamFrame(kind="finish", finish_reason="end_turn")


@router.post("/v1/messages", response_model=None)
async def create_message(
    payload: AnthropicMessagesRequest,
    request: Request,
) -> JSONResponse | StreamingResponse:
    """Anthropic ``/v1/messages`` endpoint for Claude Code, ``tool_use``, and ``thinking``."""
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

    targets = _extract_anthropic_targets(payload.system, payload.messages, payload.tools)
    items = [(target.text, target.control_point) for target in targets]
    roles = [target.role for target in targets]
    raw_body_len = len(await request.body())
    prompt_chars = max(sum(len(text) for text, _ in items), raw_body_len)

    quota_resp = await gateway.enforce_quota(
        principal=principal,
        prompt_chars=prompt_chars,
        request_id=request_id,
        session_id=session_id,
        started=started,
        api_format="anthropic",
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
            api_format="anthropic",
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
    masked_system, masked_sys_blocks, masked_messages, masked_tools = _apply_anthropic_masked_texts(
        payload.system,
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
        or (
            adapter.provider not in {ProviderType.ANTHROPIC, ProviderType.MOCK}
            and payload.model.startswith("claude")
        )
    )

    extra_body: dict[str, Any] = dict(payload.model_extra) if payload.model_extra else {}
    extra_body.pop("mcp_servers", None)
    if payload.metadata is not None:
        cleaned_meta = {k: v for k, v in payload.metadata.items() if k != "session_id"}
        if cleaned_meta:
            extra_body["metadata"] = cleaned_meta

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
        system_blocks=masked_sys_blocks,
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
            _stream_anthropic_sse(
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

    # Always emit input decision audit consistently (P2-03)
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

    raw_payload = upstream_resp.raw_payload
    out_blocks: list[dict[str, Any]] = []
    all_out_dets: list[Detection] = []
    has_tool_use = False

    if isinstance(raw_payload.get("content"), list):
        for raw_blk in raw_payload["content"]:
            if not isinstance(raw_blk, dict):
                continue
            btype = str(raw_blk.get("type", "text"))
            blk_copy = copy.deepcopy(raw_blk)
            if btype == "text" and isinstance(blk_copy.get("text"), str):
                r_txt, b_blk, b_reason, b_dets, b_act = await gateway.guard_and_restore_output(
                    content=blk_copy["text"],
                    session_id=session_id,
                    principal=principal,
                    profile=profile,
                    prof_cfg=prof_cfg,
                )
                all_out_dets.extend(b_dets)
                if b_blk:
                    out_cats = await gateway.emit_output_block_audit(
                        request_id=request_id,
                        session_id=session_id,
                        principal=principal,
                        detections=all_out_dets,
                        reason=b_reason,
                        upstream=adapter.name,
                        profile=profile,
                        started=started,
                        action=b_act,
                    )
                    return gateway.build_output_block_response(
                        reason=b_reason,
                        categories=out_cats,
                        request_id=request_id,
                        upstream=adapter.name,
                        started=started,
                        detections_count=detections_count + len(all_out_dets),
                        api_format="anthropic",
                    )
                blk_copy["text"] = r_txt
            elif btype == "thinking" and isinstance(blk_copy.get("thinking"), str):
                r_th, b_blk, b_reason, b_dets, b_act = await gateway.guard_and_restore_output(
                    content=blk_copy["thinking"],
                    session_id=session_id,
                    principal=principal,
                    profile=profile,
                    prof_cfg=prof_cfg,
                )
                all_out_dets.extend(b_dets)
                if b_blk:
                    out_cats = await gateway.emit_output_block_audit(
                        request_id=request_id,
                        session_id=session_id,
                        principal=principal,
                        detections=all_out_dets,
                        reason=b_reason,
                        upstream=adapter.name,
                        profile=profile,
                        started=started,
                        action=b_act,
                    )
                    return gateway.build_output_block_response(
                        reason=b_reason,
                        categories=out_cats,
                        request_id=request_id,
                        upstream=adapter.name,
                        started=started,
                        detections_count=detections_count + len(all_out_dets),
                        api_format="anthropic",
                    )
                blk_copy["thinking"] = r_th
            elif btype == "tool_use":
                has_tool_use = True
                r_in, b_blk, b_reason, b_dets, b_act = await _guard_json_tree(
                    blk_copy.get("input", {}),
                    gateway=gateway,
                    session_id=session_id,
                    principal=principal,
                    profile=profile,
                    prof_cfg=prof_cfg,
                )
                all_out_dets.extend(b_dets)
                if b_blk:
                    out_cats = await gateway.emit_output_block_audit(
                        request_id=request_id,
                        session_id=session_id,
                        principal=principal,
                        detections=all_out_dets,
                        reason=b_reason,
                        upstream=adapter.name,
                        profile=profile,
                        started=started,
                        action=b_act,
                    )
                    return gateway.build_output_block_response(
                        reason=b_reason,
                        categories=out_cats,
                        request_id=request_id,
                        upstream=adapter.name,
                        started=started,
                        detections_count=detections_count + len(all_out_dets),
                        api_format="anthropic",
                    )
                blk_copy["input"] = r_in
            out_blocks.append(blk_copy)
    else:
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
                api_format="anthropic",
            )

        raw_choices = raw_payload.get("choices")
        raw_first = (
            raw_choices[0]
            if isinstance(raw_choices, list) and raw_choices and isinstance(raw_choices[0], dict)
            else {}
        )
        raw_msg = raw_first.get("message") if isinstance(raw_first.get("message"), dict) else {}
        oai_tcs = raw_msg.get("tool_calls") if isinstance(raw_msg, dict) else None
        if restored_text or not oai_tcs:
            out_blocks.append({"type": "text", "text": restored_text})
        if isinstance(oai_tcs, list):
            for tc_i, tc in enumerate(oai_tcs):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                raw_args_str = str(fn.get("arguments") or "{}")
                r_args, tc_blk, tc_reason, tc_dets, tc_act = await gateway.guard_and_restore_output(
                    content=raw_args_str,
                    session_id=session_id,
                    principal=principal,
                    profile=profile,
                    prof_cfg=prof_cfg,
                )
                all_out_dets.extend(tc_dets)
                if tc_blk:
                    out_cats = await gateway.emit_output_block_audit(
                        request_id=request_id,
                        session_id=session_id,
                        principal=principal,
                        detections=all_out_dets,
                        reason=tc_reason,
                        upstream=adapter.name,
                        profile=profile,
                        started=started,
                        action=tc_act,
                    )
                    return gateway.build_output_block_response(
                        reason=tc_reason,
                        categories=out_cats,
                        request_id=request_id,
                        upstream=adapter.name,
                        started=started,
                        detections_count=detections_count + len(all_out_dets),
                        api_format="anthropic",
                    )
                try:
                    parsed_in = json.loads(r_args) if r_args.strip() else {}
                    if not isinstance(parsed_in, dict):
                        parsed_in = {"value": parsed_in}
                except json.JSONDecodeError:
                    parsed_in = {"raw": r_args}
                has_tool_use = True
                out_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tc.get("id") or f"toolu_{tc_i}"),
                        "name": str(fn.get("name") or "tool"),
                        "input": parsed_in,
                    }
                )

    raw_stop = raw_payload.get("stop_reason")
    if raw_stop is None and isinstance(raw_payload.get("choices"), list) and raw_payload["choices"]:
        c0 = raw_payload["choices"][0]
        if isinstance(c0, dict):
            raw_stop = c0.get("finish_reason")
    stop_reason = _map_anthropic_stop_reason(
        str(raw_stop) if isinstance(raw_stop, str) else None,
        has_tool_use=has_tool_use,
    )

    usage_out: dict[str, Any] = {
        "input_tokens": upstream_resp.prompt_tokens,
        "output_tokens": upstream_resp.completion_tokens,
    }
    raw_usage = raw_payload.get("usage")
    if isinstance(raw_usage, dict):
        for cache_key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            if cache_key in raw_usage:
                usage_out[cache_key] = raw_usage[cache_key]

    headers = gateway.build_headers(
        action=decision.action,
        upstream=adapter.name,
        request_id=request_id,
        started=started,
        detections_count=detections_count,
    )

    response_body: dict[str, Any] = {
        "id": str(raw_payload.get("id") or f"msg_{request_id}"),
        "type": "message",
        "role": "assistant",
        "model": upstream_resp.model or payload.model,
        "content": out_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": raw_payload.get("stop_sequence"),
        "usage": usage_out,
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
    prof_cfg: ProfileConfig,
    started: float,
) -> AsyncIterator[str]:
    """Stream Anthropic SSE frames (thinking, text, tool_use, usage, stop_reason) safely."""
    msg_id = f"msg_{request_id}"
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

    started_msg = False
    open_block_kind: str | None = None
    open_block_idx = -1
    next_block_idx = 0
    input_tokens = 0
    output_tokens = 0
    seen_stop_reason: str | None = None
    accum_tool_calls: dict[int, dict[str, Any]] = {}

    def _ensure_msg_start() -> list[str]:
        nonlocal started_msg
        if started_msg:
            return []
        started_msg = True
        return [
            _format_sse_event(
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
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                    },
                },
            )
        ]

    def _open_block(kind: Literal["text", "thinking"]) -> list[str]:
        nonlocal open_block_kind, open_block_idx, next_block_idx
        events = _ensure_msg_start()
        if open_block_kind == kind:
            return events
        if open_block_kind is not None:
            events.append(
                _format_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": open_block_idx},
                )
            )
        open_block_idx = next_block_idx
        next_block_idx += 1
        open_block_kind = kind
        cb_obj: dict[str, Any] = (
            {"type": "thinking", "thinking": ""}
            if kind == "thinking"
            else {"type": "text", "text": ""}
        )
        events.append(
            _format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": open_block_idx,
                    "content_block": cb_obj,
                },
            )
        )
        return events

    def _close_open_block() -> list[str]:
        nonlocal open_block_kind
        if open_block_kind is None:
            return []
        open_block_kind = None
        return [
            _format_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": open_block_idx},
            )
        ]

    try:
        async for frame in _iter_adapter_frames(adapter, upstream_req):
            if frame.kind == "usage":
                if frame.prompt_tokens is not None:
                    input_tokens = frame.prompt_tokens
                if frame.completion_tokens is not None:
                    output_tokens = frame.completion_tokens

            elif frame.kind == "thinking" and frame.text:
                th_dec = thinking_guard.feed(frame.text)
                if th_dec.emit_text:
                    for ev in _open_block("thinking"):
                        yield ev
                    yield _format_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": open_block_idx,
                            "delta": {"type": "thinking_delta", "thinking": th_dec.emit_text},
                        },
                    )
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
                    for ev in _ensure_msg_start():
                        yield ev
                    for ev in _close_open_block():
                        yield ev
                    yield _format_sse_event(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "content_filter", "stop_sequence": None},
                            "usage": {"output_tokens": output_tokens},
                            "pokrov_blocked": True,
                            "pokrov_reason": th_dec.block_reason,
                        },
                    )
                    yield _format_sse_event("message_stop", {"type": "message_stop"})
                    return

            elif frame.kind == "text" and frame.text:
                if open_block_kind == "thinking":
                    th_fin = thinking_guard.flush()
                    if th_fin.emit_text:
                        yield _format_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": open_block_idx,
                                "delta": {"type": "thinking_delta", "thinking": th_fin.emit_text},
                            },
                        )
                chunk_decision = guard.feed(frame.text)
                if chunk_decision.emit_text:
                    for ev in _open_block("text"):
                        yield ev
                    yield _format_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": open_block_idx,
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
                    for ev in _ensure_msg_start():
                        yield ev
                    for ev in _close_open_block():
                        yield ev
                    yield _format_sse_event(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "content_filter", "stop_sequence": None},
                            "usage": {"output_tokens": output_tokens},
                            "pokrov_blocked": True,
                            "pokrov_reason": chunk_decision.block_reason,
                        },
                    )
                    yield _format_sse_event("message_stop", {"type": "message_stop"})
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

            elif frame.kind == "finish" and frame.finish_reason:
                seen_stop_reason = frame.finish_reason

        th_final = thinking_guard.flush()
        if th_final.emit_text:
            for ev in _open_block("thinking"):
                yield ev
            yield _format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": open_block_idx,
                    "delta": {"type": "thinking_delta", "thinking": th_final.emit_text},
                },
            )
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
            for ev in _ensure_msg_start():
                yield ev
            for ev in _close_open_block():
                yield ev
            yield _format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "content_filter", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                    "pokrov_blocked": True,
                    "pokrov_reason": th_final.block_reason,
                },
            )
            yield _format_sse_event("message_stop", {"type": "message_stop"})
            return

        final_decision = guard.flush()
        if final_decision.emit_text:
            for ev in _open_block("text"):
                yield ev
            yield _format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": open_block_idx,
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
            for ev in _ensure_msg_start():
                yield ev
            for ev in _close_open_block():
                yield ev
            yield _format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "content_filter", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                    "pokrov_blocked": True,
                    "pokrov_reason": final_decision.block_reason,
                },
            )
            yield _format_sse_event("message_stop", {"type": "message_stop"})
            return

        for ev in _ensure_msg_start():
            yield ev

        if next_block_idx == 0 and not accum_tool_calls:
            for ev in _open_block("text"):
                yield ev

        for ev in _close_open_block():
            yield ev

        for tc_idx in sorted(accum_tool_calls):
            tc_item = accum_tool_calls[tc_idx]
            raw_json = "".join(tc_item["args"])
            (
                restored_json,
                tc_blocked,
                tc_reason,
                tc_dets,
                _tc_act,
            ) = await gateway.guard_and_restore_output(
                content=raw_json,
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
                yield _format_sse_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "content_filter", "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens},
                        "pokrov_blocked": True,
                        "pokrov_reason": tc_reason,
                    },
                )
                yield _format_sse_event("message_stop", {"type": "message_stop"})
                return

            cb_idx = next_block_idx
            next_block_idx += 1
            yield _format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": cb_idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc_item["id"] or f"toolu_{cb_idx}",
                        "name": tc_item["name"] or "tool",
                        "input": {},
                    },
                },
            )
            if restored_json:
                yield _format_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": cb_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": restored_json,
                        },
                    },
                )
            yield _format_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": cb_idx},
            )

        stop_reason = _map_anthropic_stop_reason(
            seen_stop_reason,
            has_tool_use=bool(accum_tool_calls),
        )
        yield _format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
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
        for ev in _ensure_msg_start():
            yield ev
        for ev in _close_open_block():
            yield ev
        yield _format_sse_event(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": "Upstream stream interrupted"},
            },
        )
        yield _format_sse_event("message_stop", {"type": "message_stop"})


__all__ = ["AnthropicMessagesRequest", "router"]
