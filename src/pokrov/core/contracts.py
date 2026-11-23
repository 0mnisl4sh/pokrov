"""Frozen domain contracts, constants, enums, models, and protocols for Pokrov.

All cross-module communication happens strictly through the models and
``typing.Protocol`` interfaces defined here. Parallel worker agents MUST NOT
modify this file directly; schema changes go through the Coordinator only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Protocol & Header Constants (strictly ASCII per RFC 9110)
# ---------------------------------------------------------------------------

HEADER_ACTION = "X-Pokrov-Action"
HEADER_UPSTREAM = "X-Pokrov-Upstream"
HEADER_REQUEST_ID = "X-Pokrov-Request-Id"
HEADER_LATENCY_MS = "X-Pokrov-Latency-Ms"
HEADER_DETECTIONS_COUNT = "X-Pokrov-Detections"

PLACEHOLDER_PREFIX = "<PKR:"
PLACEHOLDER_SUFFIX = ">"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EntityCategory(StrEnum):
    """Supported PII, secret, injection, and security finding categories."""

    RU_PASSPORT = "ru_passport"
    RU_SNILS = "ru_snils"
    RU_INN = "ru_inn"
    RU_OGRN = "ru_ogrn"
    RU_BANK_CARD = "ru_bank_card"
    RU_BIK_ACCOUNT = "ru_bik_account"
    RU_OMS = "ru_oms"
    RU_PHONE = "ru_phone"
    RU_AUTO_PLATE = "ru_auto_plate"
    RU_DRIVER_LICENSE = "ru_driver_license"
    RU_PERSON_NAME = "ru_person_name"
    RU_ADDRESS = "ru_address"
    RU_ORG_NAME = "ru_org_name"
    EMAIL = "email"
    IP_ADDRESS = "ip_address"
    SECRET_KEY = "secret_key"  # noqa: S105
    CUSTOM_PATTERN = "custom_pattern"
    PROMPT_INJECTION = "prompt_injection"
    INDIRECT_INJECTION = "indirect_injection"
    MARKDOWN_EXFIL = "markdown_exfil"
    TOXIC_OR_LEAK = "toxic_or_leak"
    MCP_POLICY_VIOLATION = "mcp_policy_violation"
    INFRA_MISCONFIG = "infra_misconfig"


class Action(StrEnum):
    """Policy enforcement actions ordered from least to most restrictive."""

    ALLOW = "allow"
    FLAG = "flag"
    MASK = "mask"
    MASK_REVERSIBLE = "mask_reversible"
    ROUTE_LOCAL = "route_local"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


ACTION_PRIORITY: dict[Action, int] = {
    Action.ALLOW: 0,
    Action.FLAG: 1,
    Action.MASK: 2,
    Action.MASK_REVERSIBLE: 3,
    Action.ROUTE_LOCAL: 4,
    Action.REQUIRE_APPROVAL: 5,
    Action.BLOCK: 6,
}


def action_severity(action: Action) -> int:
    """Return numeric precedence for a policy action."""
    return ACTION_PRIORITY[action]


def max_action(actions: Sequence[Action]) -> Action:
    """Return the highest-precedence action from a sequence (defaults to ALLOW)."""
    if not actions:
        return Action.ALLOW
    return max(actions, key=action_severity)


class ControlPoint(StrEnum):
    """5+ runtime security checkpoints across LLM and agent pipelines."""

    USER_INPUT = "user_input"
    RAG_CONTEXT = "rag_context"
    MODEL_PROMPT = "model_prompt"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_OUTPUT = "final_output"
    IDE_FILE = "ide_file"


class SpanSource(StrEnum):
    """Origin of a detected span relative to text normalization."""

    DIRECT = "direct"
    DECODED_BLOB = "decoded_blob"
    CROSS_TURN = "cross_turn"


class Severity(StrEnum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def max_severity(severities: Sequence[Severity]) -> Severity:
    """Return the highest severity from a sequence (defaults to LOW)."""
    if not severities:
        return Severity.LOW
    return max(severities, key=lambda sev: SEVERITY_ORDER[sev])


class FileGuardMode(StrEnum):
    """Protection mode for local workspace files in IDE Guard."""

    ALLOW = "allow"
    DENY = "deny"
    SANITIZE_MIRROR = "sanitize_mirror"
    ROUTE_LOCAL = "route_local"


class ProviderType(StrEnum):
    """Supported LLM upstream adapter types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VLLM = "vllm"
    GIGACHAT = "gigachat"
    YANDEXGPT = "yandexgpt"
    MOCK = "mock"


class HitlStatus(StrEnum):
    """Resolution states for Human-in-the-Loop approvals."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMED_OUT = "timed_out"


# ---------------------------------------------------------------------------
# Frozen Domain Models (Zero-Retention compliant)
# ---------------------------------------------------------------------------


class ComplianceRef(BaseModel):
    """Reference to an applicable regulatory or standard requirement."""

    model_config = ConfigDict(frozen=True)

    framework: str
    article: str
    title: str
    risk_hint: str = ""


class NormalizedSegment(BaseModel):
    """Normalized text representation with back-mapping to raw text offsets."""

    model_config = ConfigDict(frozen=True)

    text: str
    span_source: SpanSource = SpanSource.DIRECT
    index_map: tuple[int, ...] = ()
    outer_start: int = 0
    outer_end: int = 0
    evasion_techniques: tuple[str, ...] = ()

    def map_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a span ``[start, end)`` in normalized text back to raw text offsets."""
        if self.span_source == SpanSource.DECODED_BLOB:
            return (self.outer_start, self.outer_end)

        if not self.index_map:
            return (start, end)

        map_len = len(self.index_map)
        if start >= map_len:
            tail = self.index_map[-1] + 1
            return (tail, tail)

        clamped_start = max(0, start)
        raw_start = self.index_map[clamped_start]

        if end <= clamped_start:
            return (raw_start, raw_start)

        last_norm_idx = min(end - 1, map_len - 1)
        raw_end = max(raw_start, self.index_map[last_norm_idx] + 1)
        return (raw_start, raw_end)


class NormalizationResult(BaseModel):
    """Output of the anti-evasion text normalizer."""

    model_config = ConfigDict(frozen=True)

    original_length: int = Field(ge=0)
    primary: NormalizedSegment
    decoded_segments: tuple[NormalizedSegment, ...] = ()
    evasion_detected: bool = False
    evasion_techniques: tuple[str, ...] = ()
    truncated: bool = False


class Detection(BaseModel):
    """Single detection finding.

    Zero-Retention invariant: MUST NEVER store raw PII, secret values, or raw text.
    """

    model_config = ConfigDict(frozen=True)

    category: EntityCategory
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    detector: str
    span_source: SpanSource = SpanSource.DIRECT
    evasion_detected: bool = False
    evasion_techniques: tuple[str, ...] = ()
    compliance_refs: tuple[ComplianceRef, ...] = ()
    masked_preview: str = ""
    value_hmac: str = ""
    source_message_indices: tuple[int, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_span_bounds(self) -> Detection:
        if self.end < self.start:
            raise ValueError(f"Detection end ({self.end}) must be >= start ({self.start})")
        return self


class MessageItem(BaseModel):
    """Single message in a multi-turn conversation or tool-result chain."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str
    control_point: ControlPoint = ControlPoint.USER_INPUT
    tool_call_id: str | None = None
    message_index: int = 0


class InspectionRequest(BaseModel):
    """Input payload for the detection pipeline."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    messages: tuple[MessageItem, ...] = ()
    control_point: ControlPoint = ControlPoint.USER_INPUT
    session_id: str = ""
    principal: str = "anonymous"
    policy_profile: str = "balanced-152fz"
    direction: Literal["input", "output"] = "input"
    enable_ner: bool = False
    enable_ml: bool = False
    time_budget_ms: float = 500.0


class InspectionResult(BaseModel):
    """Aggregated detections and execution telemetry from a pipeline run."""

    model_config = ConfigDict(frozen=True)

    detections: tuple[Detection, ...] = ()
    evasion_detected: bool = False
    highest_severity: Severity | None = None
    layers_executed: tuple[int, ...] = (1,)
    early_exited: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    multi_turn_triggered: bool = False
    latency_ms: float = 0.0
    control_point: ControlPoint = ControlPoint.USER_INPUT
    compliance_refs: tuple[ComplianceRef, ...] = ()


class PolicyDecision(BaseModel):
    """Enforcement decision produced by the policy engine."""

    model_config = ConfigDict(frozen=True)

    action: Action
    reason: str
    matched_rules: tuple[str, ...] = ()
    target_upstream: str | None = None
    categories: tuple[EntityCategory, ...] = ()
    compliance_refs: tuple[ComplianceRef, ...] = ()
    http_status: int = 200
    redaction_map_count: int = 0


class QuotaCheckResult(BaseModel):
    """Outcome of rate-limit and payload-size quota checks."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str = ""
    retry_after_seconds: int = 0
    remaining_rpm: int | None = None
    remaining_tpm: int | None = None
    remaining_daily_requests: int | None = None


class StreamChunkDecision(BaseModel):
    """Per-chunk verdict emitted by the streaming SSE unmasker & guard."""

    model_config = ConfigDict(frozen=True)

    emit_text: str
    blocked: bool = False
    block_reason: str = ""
    unmasked_count: int = 0
    detections: tuple[Detection, ...] = ()


class AuditEvent(BaseModel):
    """Zero-Retention audit log entry.

    Only HMAC digests, categories, actions, and metadata may be persisted.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    timestamp: datetime
    request_id: str
    session_id_hmac: str = ""
    principal: str = "anonymous"
    control_point: ControlPoint = ControlPoint.USER_INPUT
    action: Action = Action.ALLOW
    severity: Severity = Severity.LOW
    categories: tuple[EntityCategory, ...] = ()
    value_hmacs: tuple[str, ...] = ()
    detector_names: tuple[str, ...] = ()
    evasion_detected: bool = False
    upstream: str = ""
    policy_profile: str = ""
    latency_ms: float = 0.0
    compliance_refs: tuple[ComplianceRef, ...] = ()
    reason: str = ""
    tool_name: str = ""


class AuditStats(BaseModel):
    """Aggregated audit metrics for dashboards and compliance reports."""

    model_config = ConfigDict(frozen=True)

    total_requests: int = 0
    blocked_count: int = 0
    masked_count: int = 0
    routed_local_count: int = 0
    flagged_count: int = 0
    allowed_count: int = 0
    approval_required_count: int = 0
    by_category: dict[str, int] = Field(default_factory=dict)
    by_control_point: dict[str, int] = Field(default_factory=dict)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    period_start: datetime | None = None
    period_end: datetime | None = None


class UpstreamRequest(BaseModel):
    """Normalized request passed to an LLM upstream adapter."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderType
    model: str
    messages: tuple[dict[str, Any], ...]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    api_format: Literal["openai", "anthropic"] = "openai"
    system: str | None = None
    tools: tuple[dict[str, Any], ...] | None = None


class UpstreamResponse(BaseModel):
    """Normalized non-streaming response returned by an upstream adapter."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderType
    model: str
    upstream_name: str
    content: str
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status_code: int = 200


class MCPToolCallRequest(BaseModel):
    """Inbound MCP ``tools/call`` inspection request."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    server_name: str = "default"
    principal: str = "anonymous"
    session_id: str = ""


class MCPToolDecision(BaseModel):
    """Policy verdict for an MCP tool call."""

    model_config = ConfigDict(frozen=True)

    action: Action
    reason: str
    rule_id: str = ""
    sanitized_arguments: dict[str, Any] | None = None
    approval_id: str | None = None
    compliance_refs: tuple[ComplianceRef, ...] = ()


class HitlApprovalRequest(BaseModel):
    """Pending or resolved Human-in-the-Loop approval record."""

    model_config = ConfigDict(frozen=True)

    approval_id: str
    tool_name: str
    server_name: str
    principal: str
    reason: str
    arguments_summary: str
    created_at: datetime
    expires_at: datetime
    status: HitlStatus = HitlStatus.PENDING


class ParsedDocument(BaseModel):
    """Extracted plain text/tables from a local workspace document."""

    model_config = ConfigDict(frozen=True)

    path: str
    format: str
    text: str
    truncated: bool = False
    page_or_sheet_count: int = 1


class IdeHookDecision(BaseModel):
    """Verdict returned to IDE PreToolUse / beforeReadFile hooks."""

    model_config = ConfigDict(frozen=True)

    permission: Literal["allow", "deny", "ask"]
    reason: str = ""
    agent_message: str = ""
    sanitized_mirror_path: str | None = None
    detections_count: int = 0
    categories: tuple[EntityCategory, ...] = ()


class AuditFinding(BaseModel):
    """Infrastructure or model-file security finding from ``pokrov audit``."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    title: str
    severity: Severity
    file_path: str
    description: str
    remediation: str
    line: int = 1
    compliance_refs: tuple[ComplianceRef, ...] = ()
    snippet_redacted: str = ""


class InfraScanReport(BaseModel):
    """Summary report produced by the AI infrastructure auditor."""

    model_config = ConfigDict(frozen=True)

    target_path: str
    scanned_files: int = 0
    findings: tuple[AuditFinding, ...] = ()
    passed: bool = True
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Runtime-Checkable Protocol Interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class TextNormalizer(Protocol):
    """Anti-evasion normalizer producing mapped segments and decoded blobs."""

    def normalize(self, text: str) -> NormalizationResult: ...


@runtime_checkable
class BaseDetector(Protocol):
    """Individual detector operating on a normalized segment."""

    @property
    def name(self) -> str: ...

    @property
    def layer(self) -> int: ...

    def detect(self, segment: NormalizedSegment, raw_text: str) -> list[Detection]: ...


@runtime_checkable
class DetectionPipeline(Protocol):
    """Multi-layer detection pipeline with caching and multi-turn window."""

    def inspect(self, request: InspectionRequest) -> InspectionResult: ...


@runtime_checkable
class PolicyEngine(Protocol):
    """Evaluates inspection findings against declarative policy profiles."""

    def evaluate(
        self,
        inspection: InspectionResult,
        *,
        direction: Literal["input", "output"] = "input",
        principal: str = "anonymous",
        profile: str | None = None,
    ) -> PolicyDecision: ...


@runtime_checkable
class VaultBackend(Protocol):
    """Reversible encrypted storage mapping placeholders to sensitive values."""

    def store(self, session_id: str, category: EntityCategory, raw_value: str) -> str: ...

    def retrieve(self, session_id: str, placeholder: str) -> str | None: ...

    def restore_text(self, session_id: str, text: str) -> tuple[str, int]: ...

    def purge_session(self, session_id: str) -> int: ...

    def cleanup_expired(self) -> int: ...


@runtime_checkable
class QuotaManager(Protocol):
    """Rate-limit and payload-budget guard per principal/API key."""

    def check_and_consume(
        self,
        principal: str,
        prompt_chars: int,
        estimated_tokens: int,
    ) -> QuotaCheckResult: ...


@runtime_checkable
class UpstreamAdapter(Protocol):
    """Adapter for an LLM provider endpoint."""

    @property
    def name(self) -> str: ...

    @property
    def provider(self) -> ProviderType: ...

    @property
    def is_local(self) -> bool: ...

    async def complete(self, request: UpstreamRequest) -> UpstreamResponse: ...

    def stream_text(self, request: UpstreamRequest) -> AsyncIterator[str]: ...


@runtime_checkable
class StreamGuard(Protocol):
    """Sliding-window SSE stream unmasker and output guardrail."""

    def feed(self, chunk: str) -> StreamChunkDecision: ...

    def flush(self) -> StreamChunkDecision: ...


@runtime_checkable
class AuditStore(Protocol):
    """Zero-Retention audit persistence and query backend."""

    def record(self, event: AuditEvent) -> None: ...

    def query(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        action: Action | None = None,
        category: EntityCategory | None = None,
        since: datetime | None = None,
    ) -> list[AuditEvent]: ...

    def stats(self, *, since: datetime | None = None) -> AuditStats: ...

    def compute_hmac(self, raw_value: str) -> str: ...

    def close(self) -> None: ...


@runtime_checkable
class SiemExporter(Protocol):
    """SIEM formatter and asynchronous event shipper."""

    def format_cef(self, event: AuditEvent) -> str: ...

    def format_syslog(self, event: AuditEvent) -> str: ...

    def format_jsonl(self, event: AuditEvent) -> str: ...

    async def send(self, event: AuditEvent) -> bool: ...


@runtime_checkable
class ReportGenerator(Protocol):
    """Compliance report renderer (DOCX / HTML / passport)."""

    def render_bytes(
        self,
        events: Sequence[AuditEvent],
        stats: AuditStats,
        *,
        title: str = "Отчёт Покров (Pokrov)",
    ) -> bytes: ...


@runtime_checkable
class MCPToolPolicy(Protocol):
    """RBAC and argument validator for MCP tool calls."""

    def evaluate_call(self, request: MCPToolCallRequest) -> MCPToolDecision: ...


@runtime_checkable
class HitlGate(Protocol):
    """Human-in-the-Loop approval coordinator for risky tool invocations."""

    async def request_approval(
        self,
        req: MCPToolCallRequest,
        reason: str,
        arguments_summary: str,
    ) -> HitlStatus: ...

    def resolve(self, approval_id: str, approved: bool) -> HitlApprovalRequest | None: ...

    def list_pending(self) -> list[HitlApprovalRequest]: ...


@runtime_checkable
class DocumentParser(Protocol):
    """Local document text extractor for IDE Guard and safe_read_file."""

    def supports(self, path: str) -> bool: ...

    def parse(self, path: str, *, max_bytes: int = 10_000_000) -> ParsedDocument: ...


__all__ = [
    "ACTION_PRIORITY",
    "HEADER_ACTION",
    "HEADER_DETECTIONS_COUNT",
    "HEADER_LATENCY_MS",
    "HEADER_REQUEST_ID",
    "HEADER_UPSTREAM",
    "PLACEHOLDER_PREFIX",
    "PLACEHOLDER_SUFFIX",
    "SEVERITY_ORDER",
    "Action",
    "AuditEvent",
    "AuditFinding",
    "AuditStats",
    "AuditStore",
    "BaseDetector",
    "ComplianceRef",
    "ControlPoint",
    "Detection",
    "DetectionPipeline",
    "DocumentParser",
    "EntityCategory",
    "FileGuardMode",
    "HitlApprovalRequest",
    "HitlGate",
    "HitlStatus",
    "IdeHookDecision",
    "InfraScanReport",
    "InspectionRequest",
    "InspectionResult",
    "MCPToolCallRequest",
    "MCPToolDecision",
    "MCPToolPolicy",
    "MessageItem",
    "NormalizationResult",
    "NormalizedSegment",
    "PolicyDecision",
    "PolicyEngine",
    "ProviderType",
    "QuotaCheckResult",
    "QuotaManager",
    "ReportGenerator",
    "Severity",
    "SiemExporter",
    "SpanSource",
    "StreamChunkDecision",
    "StreamGuard",
    "TextNormalizer",
    "UpstreamAdapter",
    "UpstreamRequest",
    "UpstreamResponse",
    "VaultBackend",
    "action_severity",
    "max_action",
    "max_severity",
]
