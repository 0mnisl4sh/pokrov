"""Phase 0 contract, compliance map, configuration, and CLI skeleton tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from pokrov import APP_NAME, __version__, cli
from pokrov.core import (
    CATEGORY_DEFAULT_SEVERITY,
    COMPLIANCE_MAP,
    HEADER_ACTION,
    HEADER_DETECTIONS_COUNT,
    HEADER_LATENCY_MS,
    HEADER_REQUEST_ID,
    HEADER_UPSTREAM,
    LEGAL_DISCLAIMER,
    Action,
    AuditEvent,
    AuditStore,
    BaseDetector,
    ComplianceRef,
    ControlPoint,
    Detection,
    DetectionPipeline,
    DocumentParser,
    EntityCategory,
    HitlGate,
    MCPToolPolicy,
    NormalizedSegment,
    PolicyEngine,
    QuotaManager,
    ReportGenerator,
    Severity,
    SiemExporter,
    SpanSource,
    StreamGuard,
    TextNormalizer,
    UpstreamAdapter,
    VaultBackend,
    action_severity,
    aggregate_compliance_refs,
    get_category_severity,
    get_compliance_refs,
    load_config,
    max_action,
    max_severity,
    policy_hash,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_every_entity_category_has_compliance_and_severity() -> None:
    """100% of EntityCategory members must exist in COMPLIANCE_MAP and CATEGORY_DEFAULT_SEVERITY."""
    assert len(EntityCategory) >= 18
    for category in EntityCategory:
        refs = get_compliance_refs(category)
        assert refs, f"Missing compliance refs for {category}"
        assert category in COMPLIANCE_MAP
        assert category in CATEGORY_DEFAULT_SEVERITY
        sev = get_category_severity(category)
        assert isinstance(sev, Severity)

    aggregated = aggregate_compliance_refs(
        [EntityCategory.RU_PASSPORT, EntityCategory.RU_SNILS, EntityCategory.PROMPT_INJECTION]
    )
    assert len(aggregated) >= 3
    assert len({(ref.framework, ref.article) for ref in aggregated}) == len(aggregated)
    assert "СЗИ" in LEGAL_DISCLAIMER


def test_action_priority_and_max_helpers() -> None:
    """Action order: block > require_approval > route_local > mask_rev > mask > flag > allow."""
    ordered = [
        Action.ALLOW,
        Action.FLAG,
        Action.MASK,
        Action.MASK_REVERSIBLE,
        Action.ROUTE_LOCAL,
        Action.REQUIRE_APPROVAL,
        Action.BLOCK,
    ]
    priorities = [action_severity(a) for a in ordered]
    assert priorities == sorted(priorities)
    assert len(set(priorities)) == len(ordered)

    assert max_action([]) == Action.ALLOW
    assert max_action([Action.FLAG, Action.ROUTE_LOCAL, Action.MASK_REVERSIBLE]) == (
        Action.ROUTE_LOCAL
    )
    assert max_action([Action.MASK, Action.BLOCK, Action.REQUIRE_APPROVAL]) == Action.BLOCK

    assert max_severity([]) == Severity.LOW
    assert max_severity([Severity.LOW, Severity.HIGH, Severity.MEDIUM]) == Severity.HIGH
    assert max_severity([Severity.MEDIUM, Severity.CRITICAL]) == Severity.CRITICAL


def test_normalized_segment_map_span() -> None:
    """NormalizedSegment.map_span must accurately map DIRECT, DECODED_BLOB, and empty index maps."""
    # Example: raw text "4\u200b5 12" -> normalized "45 12"
    # Raw indices kept in normalized: 0 ('4'), 2 ('5'), 3 (' '), 4 ('1'), 5 ('2')
    seg_direct = NormalizedSegment(
        text="45 12",
        span_source=SpanSource.DIRECT,
        index_map=(0, 2, 3, 4, 5),
        evasion_techniques=("zero_width",),
    )
    assert seg_direct.map_span(0, 5) == (0, 6)
    assert seg_direct.map_span(0, 2) == (0, 3)
    assert seg_direct.map_span(3, 5) == (4, 6)
    assert seg_direct.map_span(2, 2) == (3, 3)
    assert seg_direct.map_span(10, 12) == (6, 6)

    seg_decoded = NormalizedSegment(
        text="112-233-445 95",
        span_source=SpanSource.DECODED_BLOB,
        outer_start=15,
        outer_end=43,
        evasion_techniques=("base64",),
    )
    assert seg_decoded.map_span(0, 14) == (15, 43)

    seg_empty_map = NormalizedSegment(text="plain", span_source=SpanSource.DIRECT)
    assert seg_empty_map.map_span(1, 4) == (1, 4)


def test_detection_and_audit_event_zero_retention_invariants() -> None:
    """Detection and AuditEvent must be immutable, validate spans, and omit raw values."""
    with pytest.raises(ValidationError):
        Detection(
            category=EntityCategory.RU_SNILS,
            start=10,
            end=5,
            confidence=0.95,
            detector="ru_pii",
        )

    det = Detection(
        category=EntityCategory.RU_PASSPORT,
        start=4,
        end=15,
        confidence=0.98,
        detector="ru_pii",
        masked_preview="45** ****21",
        value_hmac="a1b2c3d4e5f6",
        compliance_refs=(
            ComplianceRef(
                framework="152-ФЗ",
                article="ст. 19",
                title="Меры защиты",
            ),
        ),
    )
    assert "raw_value" not in Detection.model_fields
    assert "raw_text" not in Detection.model_fields
    assert "text" not in Detection.model_fields
    with pytest.raises(ValidationError):
        det.confidence = 0.1  # type: ignore[misc]

    raw_secret = "4512 654321"
    event = AuditEvent(
        event_id="evt-1",
        timestamp=datetime.now(UTC),
        request_id="req-1",
        session_id_hmac="sess_hmac_001",
        principal="dev-user",
        control_point=ControlPoint.USER_INPUT,
        action=Action.MASK_REVERSIBLE,
        severity=Severity.CRITICAL,
        categories=(EntityCategory.RU_PASSPORT,),
        value_hmacs=("hmac_only_digest",),
        detector_names=("ru_pii",),
        upstream="mock-local",
        policy_profile="balanced-152fz",
        latency_ms=2.4,
        reason="Detected Russian passport",
    )
    dumped = event.model_dump_json()
    assert raw_secret not in dumped
    assert "prompt" not in AuditEvent.model_fields
    assert "raw_value" not in AuditEvent.model_fields


def test_http_headers_are_strict_ascii() -> None:
    """All HTTP header names must be pure ASCII per RFC 9110."""
    headers = [
        HEADER_ACTION,
        HEADER_UPSTREAM,
        HEADER_REQUEST_ID,
        HEADER_LATENCY_MS,
        HEADER_DETECTIONS_COUNT,
    ]
    for header in headers:
        assert header.isascii(), f"Non-ASCII HTTP header: {header!r}"
        assert header.startswith("X-Pokrov-")


def test_all_protocols_are_runtime_checkable() -> None:
    """All 14 frozen Protocol interfaces must support isinstance/issubclass checks."""
    protocols = [
        TextNormalizer,
        BaseDetector,
        DetectionPipeline,
        PolicyEngine,
        VaultBackend,
        QuotaManager,
        UpstreamAdapter,
        StreamGuard,
        AuditStore,
        SiemExporter,
        ReportGenerator,
        MCPToolPolicy,
        HitlGate,
        DocumentParser,
    ]
    assert len(protocols) == 14
    for proto in protocols:
        assert getattr(proto, "_is_runtime_protocol", False), (
            f"{proto.__name__} not runtime_checkable"
        )

    class DummyNormalizer:
        def normalize(self, text: str) -> object:
            return text

    assert isinstance(DummyNormalizer(), TextNormalizer)


def test_load_config_example_defaults_and_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """policy.example.yaml, default config, env overrides, and policy_hash must work."""
    default_cfg = load_config(None)
    assert default_cfg.active_profile == "balanced-152fz"
    assert "dev-audit" in default_cfg.profiles
    assert "balanced-152fz" in default_cfg.profiles
    assert "strict-fstec-117" in default_cfg.profiles

    example_path = REPO_ROOT / "policy.example.yaml"
    cfg = load_config(example_path)
    assert cfg.get_profile("dev-audit").rules[EntityCategory.RU_PASSPORT].action == Action.FLAG
    assert (
        cfg.get_profile("balanced-152fz").rules[EntityCategory.RU_PASSPORT].action
        == Action.MASK_REVERSIBLE
    )
    assert (
        cfg.get_profile("strict-fstec-117").rules[EntityCategory.RU_PASSPORT].action
        == Action.ROUTE_LOCAL
    )

    h1 = policy_hash(cfg, "balanced-152fz")
    h2 = policy_hash(cfg, "balanced-152fz")
    h3 = policy_hash(cfg, "strict-fstec-117")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64

    with pytest.raises(ValueError, match="Unknown policy profile"):
        cfg.get_profile("non-existent-profile")

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    assert load_config(empty_yaml).active_profile == "balanced-152fz"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad_yaml)

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")

    monkeypatch.setenv("POKROV_POLICY_PATH", str(example_path))
    monkeypatch.setenv("POKROV_PROFILE", "strict-fstec-117")
    monkeypatch.setenv("POKROV_HOST", "127.0.0.2")
    monkeypatch.setenv("POKROV_PORT", "9090")
    monkeypatch.setenv("POKROV_DB_PATH", "custom_audit.db")
    monkeypatch.setenv("POKROV_HMAC_SALT", "super-salt")

    env_cfg = load_config()
    assert env_cfg.active_profile == "strict-fstec-117"
    assert env_cfg.server.host == "127.0.0.2"
    assert env_cfg.server.port == 9090
    assert env_cfg.audit.db_path == "custom_audit.db"
    assert env_cfg.audit.hmac_salt == "super-salt"

    monkeypatch.setenv("POKROV_PORT", "not-a-number")
    with pytest.raises(ValueError, match="Invalid POKROV_PORT"):
        load_config(None)


def test_cli_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI entrypoint must handle version, empty args, and registered stubs."""
    assert cli.main(["version"]) == 0
    captured = capsys.readouterr()
    assert f"{APP_NAME} {__version__}" in captured.out

    assert cli.main([]) == 0
    captured_help = capsys.readouterr()
    assert "serve" in captured_help.out
    assert "mcp" in captured_help.out
    assert "audit" in captured_help.out

    assert cli.main(["mcp"]) == 2
    captured_stub = capsys.readouterr()
    assert "mcp" in captured_stub.err
