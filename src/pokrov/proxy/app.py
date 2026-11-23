"""FastAPI application factory, GatewayServices, and Dual-Layer async audit dispatcher."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from pokrov.audit.siem import PokrovSiemExporter
from pokrov.audit.store import SqliteAuditStore
from pokrov.core.compliance_map import aggregate_compliance_refs, get_category_severity
from pokrov.core.config import PokrovConfig, ProfileConfig, load_config
from pokrov.core.contracts import (
    HEADER_ACTION,
    HEADER_DETECTIONS_COUNT,
    HEADER_LATENCY_MS,
    HEADER_REQUEST_ID,
    HEADER_UPSTREAM,
    Action,
    AuditEvent,
    AuditStore,
    ComplianceRef,
    ControlPoint,
    Detection,
    DetectionPipeline,
    EntityCategory,
    InspectionRequest,
    InspectionResult,
    MessageItem,
    PolicyDecision,
    PolicyEngine,
    QuotaManager,
    Severity,
    SiemExporter,
    SpanSource,
    UpstreamAdapter,
    VaultBackend,
    max_severity,
)
from pokrov.detectors.pipeline import PokrovDetectionPipeline
from pokrov.policy.engine import DeclarativePolicyEngine
from pokrov.policy.quotas import TokenBucketQuotaManager
from pokrov.policy.vault import InMemoryAesVault, LocalEncryptedVault
from pokrov.proxy.anthropic_routes import router as anthropic_router
from pokrov.proxy.openai_routes import router as openai_router
from pokrov.proxy.router import DataAwareRouter
from pokrov.proxy.stream_unmasker import (
    SlidingWindowStreamGuard,
    mask_text_spans,
    sanitize_echoleak,
)

logger = logging.getLogger("pokrov.proxy.app")
_CROSS_TURN_DIGITS_RE = re.compile(r"\d[\d\s-]*\d|\d+")

_SECRET_PREFIXES = (
    "sk-",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "github_pat_",
    "glpat-",
    "AKIA",
    "AQVN",
    "eyJ",
)


@dataclass
class GatewayServices:
    """Dependency bundle injected into the Pokrov HTTP gateway."""

    config: PokrovConfig
    pipeline: DetectionPipeline
    policy_engine: PolicyEngine
    vault: VaultBackend
    quotas: QuotaManager
    resolve_upstream: Callable[[PolicyDecision, str | None], UpstreamAdapter]
    list_models: Callable[[], list[dict[str, Any]]]
    audit_store: AuditStore | None = None
    siem_exporter: SiemExporter | None = None
    aclose_fn: Callable[[], Awaitable[None]] | None = None


class AuditDispatcher:
    """Dual-Layer non-blocking bounded async queue for Zero-Retention audit & SIEM."""

    def __init__(
        self,
        audit_store: AuditStore | None,
        siem_exporter: SiemExporter | None,
        *,
        max_queue_size: int = 1024,
    ) -> None:
        self._audit_store = audit_store
        self._siem_exporter = siem_exporter
        self._max_queue_size = max_queue_size
        self._queue: asyncio.Queue[AuditEvent | None] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._running = False
        self.dropped_count = 0

    @property
    def is_running(self) -> bool:
        """Return True while the lifespan background worker is active."""
        return self._running

    @property
    def qsize(self) -> int:
        """Current number of queued audit events."""
        if self._queue is None:
            return 0
        return self._queue.qsize()

    async def start(self) -> None:
        """Start the bounded async queue worker."""
        if self._running:
            return
        queue: asyncio.Queue[AuditEvent | None] = asyncio.Queue(maxsize=self._max_queue_size)
        self._queue = queue
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop(queue))

    async def stop(self) -> None:
        """Drain queued audit events and stop the background worker cleanly."""
        if not self._running:
            return
        self._running = False
        queue = self._queue
        task = self._worker_task
        self._worker_task = None
        if queue is not None and task is not None:
            try:
                await asyncio.wait_for(queue.put(None), timeout=2.0)
                await asyncio.wait_for(task, timeout=5.0)
            except TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._queue = None

    async def flush(self) -> None:
        """Wait until all currently queued events have been processed."""
        if self._running and self._queue is not None:
            await asyncio.wait_for(self._queue.join(), timeout=5.0)

    async def enqueue(self, event: AuditEvent) -> None:
        """Queue an ``AuditEvent`` without blocking; fall back when lifespan is inactive."""
        if self._running and self._queue is not None:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped_count += 1
                logger.warning("Audit queue full; dropped event_id=%s", event.event_id)
            return
        await self._persist_event(event)

    async def _worker_loop(self, queue: asyncio.Queue[AuditEvent | None]) -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    break
                await self._persist_event(item)
            finally:
                queue.task_done()

    async def _persist_event(self, event: AuditEvent) -> None:
        if self._audit_store is not None:
            try:
                self._audit_store.record(event)
            except Exception as exc:
                logger.warning(
                    "Audit store persistence failed event_id=%s error=%s",
                    event.event_id,
                    type(exc).__name__,
                )
        if self._siem_exporter is not None:
            try:
                await self._siem_exporter.send(event)
            except Exception as exc:
                logger.warning(
                    "SIEM export failed event_id=%s error=%s",
                    event.event_id,
                    type(exc).__name__,
                )


def _ascii_header_value(value: str) -> str:
    """Guarantee strict RFC 9110 ASCII header values."""
    if value.isascii():
        return value
    return value.encode("ascii", errors="replace").decode("ascii")


def _sanitize_principal_token(raw_token: str) -> str:
    """Avoid storing raw provider secret keys as principals."""
    if raw_token.startswith(_SECRET_PREFIXES) or len(raw_token) > 48:
        digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()[:12]
        return f"key:{digest}"
    return raw_token


def _compute_severity(
    inspection: InspectionResult | None,
    categories: Sequence[EntityCategory],
) -> Severity:
    """Compute highest severity from inspection result and categories."""
    if inspection is not None and inspection.highest_severity is not None:
        return inspection.highest_severity
    if categories:
        return max_severity([get_category_severity(cat) for cat in categories])
    return Severity.LOW


class GatewayRuntime:
    """Shared gateway orchestration over the frozen ``contracts.py`` Protocols."""

    def __init__(self, services: GatewayServices, dispatcher: AuditDispatcher) -> None:
        self.services = services
        self.dispatcher = dispatcher
        self._fallback_salt = os.urandom(32)

    def extract_principal(
        self,
        headers: Mapping[str, str],
        body_user: str | None = None,
    ) -> str:
        """Resolve caller principal without exposing raw API secrets."""
        explicit = headers.get("x-pokrov-principal", "").strip()
        if explicit:
            return explicit
        if body_user and body_user.strip():
            return body_user.strip()
        auth = headers.get("authorization", "").strip()
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if token:
                return _sanitize_principal_token(token)
        api_key = headers.get("x-api-key", "").strip()
        if api_key:
            return _sanitize_principal_token(api_key)
        return "anonymous"

    def extract_session_id(
        self,
        headers: Mapping[str, str],
        *,
        fallback_id: str,
        body_session: str | None = None,
    ) -> str:
        """Resolve session identifier for Vault isolation."""
        header_sid = headers.get("x-pokrov-session-id", "").strip()
        if header_sid:
            return header_sid
        if body_session and body_session.strip():
            return body_session.strip()
        return fallback_id

    def resolve_profile(
        self,
        headers: Mapping[str, str],
        requested_profile: str | None = None,
    ) -> tuple[str, ProfileConfig]:
        """Resolve active or header-selected policy profile."""
        header_profile = headers.get("x-pokrov-profile", "").strip()
        profile_name = requested_profile or header_profile or self.services.config.active_profile
        prof_cfg = self.services.config.get_profile(profile_name)
        return profile_name, prof_cfg

    def extract_requested_upstream(self, headers: Mapping[str, str]) -> str | None:
        """Extract optional upstream override from ``X-Pokrov-Upstream`` header."""
        val = headers.get("x-pokrov-upstream", "").strip()
        return val or None

    def build_headers(
        self,
        *,
        action: Action,
        upstream: str,
        request_id: str,
        started: float,
        detections_count: int,
    ) -> dict[str, str]:
        """Construct strict ASCII ``X-Pokrov-*`` HTTP response headers."""
        latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        return {
            HEADER_ACTION: _ascii_header_value(action.value),
            HEADER_UPSTREAM: _ascii_header_value(upstream or "none"),
            HEADER_REQUEST_ID: _ascii_header_value(request_id),
            HEADER_LATENCY_MS: f"{latency_ms:.2f}",
            HEADER_DETECTIONS_COUNT: str(max(0, detections_count)),
        }

    def compute_session_hmac(self, session_id: str) -> str:
        """Compute keyed HMAC-SHA256 digest of ``session_id`` for Zero-Retention audit."""
        if self.services.audit_store is not None:
            return self.services.audit_store.compute_hmac(session_id)
        return hmac.new(
            self._fallback_salt,
            session_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def emit_audit(
        self,
        *,
        request_id: str,
        session_id: str,
        principal: str,
        control_point: ControlPoint,
        action: Action,
        categories: Sequence[EntityCategory],
        detections: Sequence[Detection],
        evasion_detected: bool,
        upstream: str,
        profile: str,
        latency_ms: float,
        compliance_refs: Sequence[ComplianceRef],
        reason: str,
        tool_name: str = "",
    ) -> None:
        """Build and asynchronously dispatch a Zero-Retention ``AuditEvent``."""
        dedup_cats = tuple(dict.fromkeys(categories))
        value_hmacs = tuple(dict.fromkeys(d.value_hmac for d in detections if d.value_hmac))
        detector_names = tuple(dict.fromkeys(d.detector for d in detections if d.detector))
        refs = tuple(compliance_refs) if compliance_refs else aggregate_compliance_refs(dedup_cats)
        severity = _compute_severity(None, dedup_cats)

        event = AuditEvent(
            event_id=uuid.uuid4().hex,
            timestamp=datetime.now(UTC),
            request_id=request_id,
            session_id_hmac=self.compute_session_hmac(session_id),
            principal=principal,
            control_point=control_point,
            action=action,
            severity=severity,
            categories=dedup_cats,
            value_hmacs=value_hmacs,
            detector_names=detector_names,
            evasion_detected=evasion_detected,
            upstream=upstream,
            policy_profile=profile,
            latency_ms=round(max(0.0, latency_ms), 3),
            compliance_refs=refs,
            reason=reason,
            tool_name=tool_name,
        )
        await self.dispatcher.enqueue(event)

    async def emit_decision_audit(
        self,
        *,
        request_id: str,
        session_id: str,
        principal: str,
        control_point: ControlPoint,
        decision: PolicyDecision,
        inspection: InspectionResult,
        upstream: str,
        profile: str,
        started: float,
    ) -> None:
        """Emit a Zero-Retention ``AuditEvent`` from an inspection & policy decision."""
        await self.emit_audit(
            request_id=request_id,
            session_id=session_id,
            principal=principal,
            control_point=control_point,
            action=decision.action,
            categories=decision.categories,
            detections=inspection.detections,
            evasion_detected=inspection.evasion_detected,
            upstream=upstream,
            profile=profile,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            compliance_refs=decision.compliance_refs,
            reason=decision.reason,
        )

    async def emit_output_block_audit(
        self,
        *,
        request_id: str,
        session_id: str,
        principal: str,
        detections: Sequence[Detection],
        reason: str,
        upstream: str,
        profile: str,
        started: float,
        action: Action = Action.BLOCK,
    ) -> tuple[EntityCategory, ...]:
        """Emit a ``FINAL_OUTPUT`` block ``AuditEvent`` and return deduplicated categories."""
        cats = tuple(dict.fromkeys(d.category for d in detections)) or (
            EntityCategory.MARKDOWN_EXFIL,
        )
        await self.emit_audit(
            request_id=request_id,
            session_id=session_id,
            principal=principal,
            control_point=ControlPoint.FINAL_OUTPUT,
            action=action,
            categories=cats,
            detections=detections,
            evasion_detected=False,
            upstream=upstream,
            profile=profile,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            compliance_refs=tuple(ref for d in detections for ref in d.compliance_refs),
            reason=reason or "Blocked by output guardrail",
        )
        return cats

    async def enforce_quota(
        self,
        *,
        principal: str,
        prompt_chars: int,
        request_id: str,
        session_id: str,
        started: float,
        api_format: Literal["openai", "anthropic"],
        control_point: ControlPoint,
        profile: str,
    ) -> JSONResponse | None:
        """Check quotas and return HTTP 429 JSONResponse with ``Retry-After`` if exceeded."""
        estimated_tokens = max(1, (prompt_chars + 3) // 4) if prompt_chars > 0 else 0
        quota_res = self.services.quotas.check_and_consume(
            principal=principal,
            prompt_chars=prompt_chars,
            estimated_tokens=estimated_tokens,
        )
        if quota_res.allowed:
            return None

        retry_after = max(1, quota_res.retry_after_seconds)
        latency_ms = (time.perf_counter() - started) * 1000.0
        reason = quota_res.reason or "Quota exceeded"
        await self.emit_audit(
            request_id=request_id,
            session_id=session_id,
            principal=principal,
            control_point=control_point,
            action=Action.BLOCK,
            categories=(),
            detections=(),
            evasion_detected=False,
            upstream="none",
            profile=profile,
            latency_ms=latency_ms,
            compliance_refs=(),
            reason=reason,
        )

        headers = self.build_headers(
            action=Action.BLOCK,
            upstream="none",
            request_id=request_id,
            started=started,
            detections_count=0,
        )
        headers["Retry-After"] = str(retry_after)

        if api_format == "anthropic":
            body: dict[str, Any] = {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": reason},
                "pokrov": {"request_id": request_id, "retry_after_seconds": retry_after},
            }
        else:
            body = {
                "error": {
                    "message": reason,
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "request_id": request_id,
                }
            }
        return JSONResponse(status_code=429, content=body, headers=headers)

    def inspect_items(
        self,
        *,
        items: Sequence[tuple[str, ControlPoint]],
        session_id: str,
        principal: str,
        profile: str,
        prof_cfg: ProfileConfig,
        roles: Sequence[str] | None = None,
    ) -> tuple[list[InspectionResult], InspectionResult, PolicyDecision, ControlPoint]:
        """Inspect each text item, run multi-turn window, and evaluate ``PolicyDecision``."""
        per_item: list[InspectionResult] = []
        all_detections: list[Detection] = []
        all_refs: list[ComplianceRef] = []
        seen_refs: set[tuple[str, str]] = set()
        evasion_detected = False
        severities: list[Severity] = []
        primary_cp = items[0][1] if items else ControlPoint.USER_INPUT

        for text, cp in items:
            insp = self.services.pipeline.inspect(
                InspectionRequest(
                    text=text,
                    control_point=cp,
                    session_id=session_id,
                    principal=principal,
                    policy_profile=profile,
                    direction="input",
                    enable_ner=prof_cfg.enable_ner,
                    enable_ml=prof_cfg.enable_ml,
                )
            )
            per_item.append(insp)
            if insp.detections:
                if cp == ControlPoint.TOOL_RESULT or primary_cp != ControlPoint.TOOL_RESULT:
                    primary_cp = cp
                all_detections.extend(insp.detections)
            if insp.evasion_detected:
                evasion_detected = True
            if insp.highest_severity is not None:
                severities.append(insp.highest_severity)
            for ref in insp.compliance_refs:
                key = (ref.framework, ref.article)
                if key not in seen_refs:
                    seen_refs.add(key)
                    all_refs.append(ref)

        multi_turn_triggered = False
        if len(items) >= 2:
            msg_items: list[MessageItem] = []
            for idx, (text, cp) in enumerate(items):
                role = (
                    roles[idx]
                    if roles is not None and idx < len(roles)
                    else ("tool" if cp == ControlPoint.TOOL_RESULT else "user")
                )
                msg_items.append(
                    MessageItem(
                        role=role,
                        content=text,
                        control_point=cp,
                        message_index=idx,
                    )
                )
            mt_insp = self.services.pipeline.inspect(
                InspectionRequest(
                    messages=tuple(msg_items),
                    control_point=primary_cp,
                    session_id=session_id,
                    principal=principal,
                    policy_profile=profile,
                    direction="input",
                    enable_ner=prof_cfg.enable_ner,
                    enable_ml=prof_cfg.enable_ml,
                )
            )
            cross_turn_dets = [
                det for det in mt_insp.detections if det.span_source == SpanSource.CROSS_TURN
            ]
            if cross_turn_dets:
                multi_turn_triggered = True
                all_detections.extend(cross_turn_dets)
                if mt_insp.evasion_detected:
                    evasion_detected = True
                if mt_insp.highest_severity is not None:
                    severities.append(mt_insp.highest_severity)
                for ref in mt_insp.compliance_refs:
                    key = (ref.framework, ref.article)
                    if key not in seen_refs:
                        seen_refs.add(key)
                        all_refs.append(ref)

        agg_inspection = InspectionResult(
            detections=tuple(all_detections),
            evasion_detected=evasion_detected,
            highest_severity=max_severity(severities) if severities else None,
            multi_turn_triggered=multi_turn_triggered,
            control_point=primary_cp,
            compliance_refs=tuple(all_refs),
        )
        decision = self.services.policy_engine.evaluate(
            agg_inspection,
            direction="input",
            principal=principal,
            profile=profile,
        )
        return per_item, agg_inspection, decision, primary_cp

    def apply_input_masking(
        self,
        *,
        items: Sequence[tuple[str, ControlPoint]],
        inspections: Sequence[InspectionResult],
        decision: PolicyDecision,
        session_id: str,
        agg_inspection: InspectionResult | None = None,
    ) -> tuple[list[str], int]:
        """Apply Vault or static redaction across inspected items when policy demands masking."""
        if decision.action not in (Action.MASK, Action.MASK_REVERSIBLE):
            return [text for text, _ in items], 0

        allowed_cats = set(decision.categories) if decision.categories else None
        masked_list: list[str] = []
        total_replaced = 0
        for (text, _cp), insp in zip(items, inspections, strict=True):
            masked, replaced = mask_text_spans(
                text,
                insp.detections,
                action=decision.action,
                vault=self.services.vault,
                session_id=session_id,
                allowed_categories=allowed_cats,
            )
            masked_list.append(masked)
            total_replaced += replaced

        if agg_inspection is not None:
            for det in agg_inspection.detections:
                if det.span_source != SpanSource.CROSS_TURN:
                    continue
                if allowed_cats and det.category not in allowed_cats:
                    continue
                for idx in det.source_message_indices:
                    if not (0 <= idx < len(masked_list)):
                        continue
                    if any(d.category == det.category for d in inspections[idx].detections):
                        continue
                    turn_text = masked_list[idx]

                    def _replace_fragment(
                        match: re.Match[str],
                        *,
                        cat: EntityCategory = det.category,
                    ) -> str:
                        nonlocal total_replaced
                        total_replaced += 1
                        if decision.action == Action.MASK_REVERSIBLE:
                            return self.services.vault.store(session_id, cat, match.group(0))
                        return f"[REDACTED:{cat.value.upper()}]"

                    masked_list[idx] = _CROSS_TURN_DIGITS_RE.sub(_replace_fragment, turn_text)

        return masked_list, total_replaced

    def build_block_response(
        self,
        *,
        decision: PolicyDecision,
        request_id: str,
        started: float,
        detections_count: int,
        api_format: Literal["openai", "anthropic"],
    ) -> JSONResponse:
        """Build HTTP 403 JSON response for blocked input policy violations."""
        headers = self.build_headers(
            action=decision.action,
            upstream="none",
            request_id=request_id,
            started=started,
            detections_count=detections_count,
        )
        cats = [cat.value for cat in decision.categories]
        refs = [ref.model_dump(mode="json") for ref in decision.compliance_refs]

        if api_format == "anthropic":
            body: dict[str, Any] = {
                "type": "error",
                "error": {
                    "type": "permission_error",
                    "message": decision.reason,
                },
                "pokrov": {
                    "action": decision.action.value,
                    "categories": cats,
                    "compliance_refs": refs,
                    "request_id": request_id,
                },
            }
        else:
            body = {
                "error": {
                    "message": decision.reason,
                    "type": "pokrov_policy_blocked",
                    "code": "policy_blocked",
                    "action": decision.action.value,
                    "categories": cats,
                    "compliance_refs": refs,
                    "request_id": request_id,
                }
            }
        status_code = decision.http_status if 400 <= decision.http_status < 500 else 403
        return JSONResponse(status_code=status_code, content=body, headers=headers)

    def build_output_block_response(
        self,
        *,
        reason: str,
        categories: Sequence[EntityCategory],
        request_id: str,
        upstream: str,
        started: float,
        detections_count: int,
        api_format: Literal["openai", "anthropic"],
    ) -> JSONResponse:
        """Build HTTP 403 JSON response when output guardrails block an LLM response."""
        headers = self.build_headers(
            action=Action.BLOCK,
            upstream=upstream,
            request_id=request_id,
            started=started,
            detections_count=detections_count,
        )
        cats = [cat.value for cat in categories]
        refs = [ref.model_dump(mode="json") for ref in aggregate_compliance_refs(categories)]

        if api_format == "anthropic":
            body: dict[str, Any] = {
                "type": "error",
                "error": {"type": "permission_error", "message": reason},
                "pokrov": {
                    "action": Action.BLOCK.value,
                    "categories": cats,
                    "compliance_refs": refs,
                    "request_id": request_id,
                },
            }
        else:
            body = {
                "error": {
                    "message": reason,
                    "type": "pokrov_output_blocked",
                    "code": "output_blocked",
                    "action": Action.BLOCK.value,
                    "categories": cats,
                    "compliance_refs": refs,
                    "request_id": request_id,
                }
            }
        return JSONResponse(status_code=403, content=body, headers=headers)

    def build_upstream_error_response(
        self,
        *,
        request_id: str,
        upstream: str,
        started: float,
        detections_count: int,
        api_format: Literal["openai", "anthropic"],
    ) -> JSONResponse:
        """Build safe HTTP 502 error response without leaking prompt data."""
        headers = self.build_headers(
            action=Action.ALLOW,
            upstream=upstream,
            request_id=request_id,
            started=started,
            detections_count=detections_count,
        )
        if api_format == "anthropic":
            body: dict[str, Any] = {
                "type": "error",
                "error": {"type": "api_error", "message": "Upstream provider request failed"},
                "pokrov": {"request_id": request_id},
            }
        else:
            body = {
                "error": {
                    "message": "Upstream provider request failed",
                    "type": "upstream_error",
                    "code": "bad_gateway",
                    "request_id": request_id,
                }
            }
        return JSONResponse(status_code=502, content=body, headers=headers)

    async def guard_and_restore_output(
        self,
        *,
        content: str,
        session_id: str,
        principal: str,
        profile: str,
        prof_cfg: ProfileConfig,
    ) -> tuple[str, bool, str, list[Detection], Action]:
        """Run EchoLeak + output pipeline checks, then restore Vault placeholders."""
        cleaned, exfil_dets = sanitize_echoleak(content)
        if exfil_dets:
            return (
                "",
                True,
                "EchoLeak protection: external HTTP/HTTPS URL blocked in output",
                exfil_dets,
                Action.BLOCK,
            )

        out_insp = self.services.pipeline.inspect(
            InspectionRequest(
                text=cleaned,
                control_point=ControlPoint.FINAL_OUTPUT,
                session_id=session_id,
                principal=principal,
                policy_profile=profile,
                direction="output",
                enable_ner=prof_cfg.enable_ner,
                enable_ml=prof_cfg.enable_ml,
            )
        )
        out_dets = list(out_insp.detections)
        out_action = Action.ALLOW
        if out_dets:
            out_dec = self.services.policy_engine.evaluate(
                out_insp,
                direction="output",
                principal=principal,
                profile=profile,
            )
            out_action = out_dec.action
            if out_dec.action in (Action.BLOCK, Action.REQUIRE_APPROVAL):
                return "", True, out_dec.reason, out_dets, out_dec.action
            if out_dec.action in (Action.MASK, Action.MASK_REVERSIBLE):
                allowed_cats = set(out_dec.categories) if out_dec.categories else None
                cleaned, _ = mask_text_spans(
                    cleaned,
                    out_dets,
                    action=Action.MASK,
                    vault=self.services.vault,
                    session_id=session_id,
                    allowed_categories=allowed_cats,
                )

        restored_text, _ = self.services.vault.restore_text(session_id, cleaned)
        return restored_text, False, "", out_dets, out_action

    def make_stream_guard(
        self,
        *,
        session_id: str,
        principal: str,
        profile: str,
    ) -> SlidingWindowStreamGuard:
        """Construct a ``SlidingWindowStreamGuard`` for SSE responses."""
        return SlidingWindowStreamGuard(
            vault=self.services.vault,
            session_id=session_id,
            output_pipeline=self.services.pipeline,
            policy_engine=self.services.policy_engine,
            policy_profile=profile,
            principal=principal,
        )


class InspectContextRequest(BaseModel):
    """Request payload for ``POST /v1/inspect/context`` (RAG checkpoint)."""

    model_config = ConfigDict(extra="allow")

    text: str = ""
    chunks: list[str] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)
    session_id: str = ""
    principal: str = "anonymous"
    policy_profile: str | None = None
    source: str = "rag"


def create_app(services: GatewayServices) -> FastAPI:
    """Create the FastAPI application wired strictly to ``GatewayServices``."""
    dispatcher = AuditDispatcher(
        services.audit_store,
        services.siem_exporter,
    )
    runtime = GatewayRuntime(services, dispatcher)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await dispatcher.start()
        try:
            yield
        finally:
            await dispatcher.stop()
            if services.aclose_fn is not None:
                await services.aclose_fn()

    app = FastAPI(
        title="Pokrov (Покров) AI Security Gateway",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.services = services
    app.state.audit_dispatcher = dispatcher
    app.state.gateway = runtime

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness and configuration health endpoint."""
        return {
            "status": "ok",
            "service": "pokrov",
            "profile": services.config.active_profile,
            "queue_size": dispatcher.qsize,
        }

    @app.post("/v1/inspect/context")
    async def inspect_context(
        payload: InspectContextRequest,
        request: Request,
    ) -> JSONResponse:
        """RAG / Retrieval Context checkpoint (``ControlPoint.RAG_CONTEXT``)."""
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        principal = runtime.extract_principal(
            request.headers,
            body_user=payload.principal if payload.principal != "anonymous" else None,
        )
        session_id = runtime.extract_session_id(
            request.headers,
            fallback_id=request_id,
            body_session=payload.session_id or None,
        )
        try:
            profile, prof_cfg = runtime.resolve_profile(
                request.headers,
                requested_profile=payload.policy_profile,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        chunk_list = payload.chunks or payload.documents
        items: list[tuple[str, ControlPoint]] = []
        if payload.text:
            items.append((payload.text, ControlPoint.RAG_CONTEXT))
        for chunk in chunk_list:
            items.append((chunk, ControlPoint.RAG_CONTEXT))

        inspections, agg_inspection, decision, _ = runtime.inspect_items(
            items=items,
            session_id=session_id,
            principal=principal,
            profile=profile,
            prof_cfg=prof_cfg,
        )

        blocked = decision.action in (Action.BLOCK, Action.REQUIRE_APPROVAL)
        if blocked:
            sanitized_text = ""
            sanitized_chunks: list[str] = []
        else:
            masked_list, _ = runtime.apply_input_masking(
                items=items,
                inspections=inspections,
                decision=decision,
                session_id=session_id,
                agg_inspection=agg_inspection,
            )
            if payload.text:
                sanitized_text = masked_list[0]
                sanitized_chunks = masked_list[1:]
            else:
                sanitized_chunks = masked_list
                sanitized_text = "\n".join(sanitized_chunks)

        latency_ms = (time.perf_counter() - started) * 1000.0
        await runtime.emit_audit(
            request_id=request_id,
            session_id=session_id,
            principal=principal,
            control_point=ControlPoint.RAG_CONTEXT,
            action=decision.action,
            categories=decision.categories,
            detections=agg_inspection.detections,
            evasion_detected=agg_inspection.evasion_detected,
            upstream="pokrov-inspect",
            profile=profile,
            latency_ms=latency_ms,
            compliance_refs=decision.compliance_refs,
            reason=decision.reason,
        )

        headers = runtime.build_headers(
            action=decision.action,
            upstream="pokrov-inspect",
            request_id=request_id,
            started=started,
            detections_count=len(agg_inspection.detections),
        )
        response_body: dict[str, Any] = {
            "request_id": request_id,
            "control_point": ControlPoint.RAG_CONTEXT.value,
            "action": decision.action.value,
            "blocked": blocked,
            "allowed": not blocked,
            "reason": decision.reason,
            "detections_count": len(agg_inspection.detections),
            "categories": [cat.value for cat in decision.categories],
            "compliance_refs": [ref.model_dump(mode="json") for ref in decision.compliance_refs],
            "sanitized_text": sanitized_text,
            "sanitized_chunks": sanitized_chunks,
            "detections": [det.model_dump(mode="json") for det in agg_inspection.detections],
        }
        return JSONResponse(status_code=200, content=response_body, headers=headers)

    app.include_router(openai_router)
    app.include_router(anthropic_router)
    return app


def create_default_app(config: PokrovConfig | None = None) -> FastAPI:
    """Wire all production Pokrov subsystems into a ready-to-serve FastAPI app."""
    cfg = config or load_config()
    audit_store: SqliteAuditStore | None = (
        SqliteAuditStore(cfg.audit) if cfg.audit.enabled else None
    )
    hmac_fn = audit_store.compute_hmac if audit_store is not None else None
    pipeline = PokrovDetectionPipeline(cfg, hmac_fn=hmac_fn)
    policy_engine = DeclarativePolicyEngine(cfg)

    vault: VaultBackend
    if cfg.vault.mode == "local_encrypted":
        vault = LocalEncryptedVault(
            local_vault_path=cfg.vault.local_vault_path,
            ttl_seconds=cfg.vault.ttl_seconds,
            max_entries=cfg.vault.max_entries,
        )
    else:
        vault = InMemoryAesVault(
            ttl_seconds=cfg.vault.ttl_seconds,
            max_entries=cfg.vault.max_entries,
        )

    active_prof = cfg.get_profile()
    quotas = TokenBucketQuotaManager(active_prof.quotas)
    router = DataAwareRouter(cfg.router)
    siem_exporter = PokrovSiemExporter(cfg.siem) if cfg.siem.enabled else None

    async def _aclose() -> None:
        await router.aclose()
        if audit_store is not None:
            audit_store.close()

    services = GatewayServices(
        config=cfg,
        pipeline=pipeline,
        policy_engine=policy_engine,
        vault=vault,
        quotas=quotas,
        resolve_upstream=lambda decision, req_model: router.select_upstream(
            decision, requested_upstream=req_model
        ),
        list_models=router.list_models,
        audit_store=audit_store,
        siem_exporter=siem_exporter,
        aclose_fn=_aclose,
    )
    app = create_app(services)
    app.state.router = router
    app.state.audit_store = audit_store
    app.state.siem_exporter = siem_exporter
    return app


__all__ = [
    "AuditDispatcher",
    "GatewayRuntime",
    "GatewayServices",
    "InspectContextRequest",
    "create_app",
    "create_default_app",
]
