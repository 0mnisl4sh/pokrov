"""Tests for Zero-Retention DOCX, offline HTML5, GOST cards, and CLI report/serve wiring."""

from __future__ import annotations

import hashlib
import hmac
import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from pokrov import cli
from pokrov.audit import report_docx, report_html
from pokrov.audit.report_docx import DocxComplianceReportGenerator
from pokrov.audit.report_html import (
    ACTION_REMEDIATION_RU,
    CATEGORY_LABELS_RU,
    REGULATORY_OVERVIEW_ITEMS,
    HtmlComplianceReportGenerator,
    _classify_incident,
    build_gost_56545_dict,
    render_gost_56545_passport,
)
from pokrov.audit.store import SqliteAuditStore
from pokrov.core.compliance_map import LEGAL_DISCLAIMER, get_compliance_refs
from pokrov.core.config import AuditConfig, PokrovConfig, ServerConfig, VaultConfig
from pokrov.core.contracts import (
    Action,
    AuditEvent,
    AuditStats,
    ComplianceRef,
    ControlPoint,
    EntityCategory,
    ReportGenerator,
    Severity,
)

_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _tok(*parts: str) -> str:
    """Assemble synthetic secret strings at runtime to avoid secret-scanner false positives."""
    return "".join(parts)


def _sample_events_and_stats() -> tuple[list[AuditEvent], AuditStats]:
    """Create deterministic Zero-Retention audit events and aggregated KPI stats."""
    base_time = datetime(2026, 11, 20, 10, 0, 0, tzinfo=UTC)
    events = [
        AuditEvent(
            event_id="evt-001",
            timestamp=base_time,
            request_id="req-001",
            session_id_hmac="sess_hmac_alpha_01",
            principal="analyst-ivan",
            control_point=ControlPoint.USER_INPUT,
            action=Action.MASK_REVERSIBLE,
            severity=Severity.CRITICAL,
            categories=(
                EntityCategory.RU_PASSPORT,
                EntityCategory.RU_SNILS,
                EntityCategory.RU_DATE_OF_BIRTH,
            ),
            value_hmacs=("hmac_passport_aaa111", "hmac_snils_bbb222", "hmac_dob_fff666"),
            detector_names=("ru_pii",),
            evasion_detected=False,
            upstream="openai-cloud",
            policy_profile="balanced-152fz",
            latency_ms=4.25,
            compliance_refs=get_compliance_refs(EntityCategory.RU_PASSPORT),
            reason="Detected Russian PII in user prompt",
        ),
        AuditEvent(
            event_id="evt-002",
            timestamp=base_time + timedelta(minutes=15),
            request_id="req-002",
            session_id_hmac="sess_hmac_beta_02",
            principal="coding-agent",
            control_point=ControlPoint.RAG_CONTEXT,
            action=Action.BLOCK,
            severity=Severity.CRITICAL,
            categories=(EntityCategory.INDIRECT_INJECTION, EntityCategory.MARKDOWN_EXFIL),
            value_hmacs=("hmac_inj_ccc333",),
            detector_names=("prompt_injection",),
            evasion_detected=True,
            upstream="mock-local",
            policy_profile="strict-fstec-117",
            latency_ms=7.80,
            compliance_refs=get_compliance_refs(EntityCategory.MARKDOWN_EXFIL),
            reason="Blocked EchoLeak indirect injection in RAG chunk",
        ),
        AuditEvent(
            event_id="evt-003",
            timestamp=base_time + timedelta(minutes=30),
            request_id="req-003",
            session_id_hmac="sess_hmac_gamma_03",
            principal="mcp-worker",
            control_point=ControlPoint.TOOL_CALL,
            action=Action.ROUTE_LOCAL,
            severity=Severity.HIGH,
            categories=(EntityCategory.RU_INN, EntityCategory.SECRET_KEY),
            value_hmacs=("hmac_inn_ddd444", "hmac_secret_eee555"),
            detector_names=("ru_pii", "secrets"),
            evasion_detected=False,
            upstream="ollama-local",
            policy_profile="strict-fstec-117",
            latency_ms=11.40,
            compliance_refs=get_compliance_refs(EntityCategory.SECRET_KEY),
            reason="Routed sensitive payload to local contour",
            tool_name="query_customer_db",
        ),
    ]
    stats = AuditStats(
        total_requests=150,
        blocked_count=12,
        masked_count=34,
        routed_local_count=9,
        flagged_count=5,
        allowed_count=88,
        approval_required_count=2,
        by_category={
            EntityCategory.RU_PASSPORT.value: 18,
            EntityCategory.RU_SNILS.value: 16,
            EntityCategory.RU_DATE_OF_BIRTH.value: 4,
            EntityCategory.INDIRECT_INJECTION.value: 7,
            EntityCategory.MARKDOWN_EXFIL.value: 5,
            EntityCategory.RU_INN.value: 6,
            EntityCategory.SECRET_KEY.value: 3,
        },
        by_control_point={
            ControlPoint.USER_INPUT.value: 110,
            ControlPoint.RAG_CONTEXT.value: 25,
            ControlPoint.TOOL_CALL.value: 15,
        },
        p50_latency_ms=3.50,
        p95_latency_ms=14.75,
        period_start=base_time,
        period_end=base_time + timedelta(hours=1),
    )
    return events, stats


def _extract_docx_text(document_xml_bytes: bytes) -> str:
    """Parse word/document.xml and concatenate all <w:t> text elements."""
    root = ET.fromstring(document_xml_bytes)  # noqa: S314
    texts = [node.text for node in root.iter(f"{_WORD_NS}t") if node.text]
    return "\n".join(texts)


def test_generators_implement_report_protocol() -> None:
    """Both DOCX and HTML generators must satisfy the runtime_checkable ReportGenerator Protocol."""
    docx_gen = DocxComplianceReportGenerator()
    html_gen = HtmlComplianceReportGenerator()
    assert isinstance(docx_gen, ReportGenerator)
    assert isinstance(html_gen, ReportGenerator)


def test_ru_date_of_birth_and_regulatory_distinctions() -> None:
    """Verify RU_DATE_OF_BIRTH coverage and exact 23-FZ, RKN 140, KoAP, and FSTEC distinctions."""
    assert set(CATEGORY_LABELS_RU) == set(EntityCategory)
    assert CATEGORY_LABELS_RU[EntityCategory.RU_DATE_OF_BIRTH] == "Дата рождения субъекта ПДн"
    assert (
        report_docx.CATEGORY_LABELS_RU[EntityCategory.RU_DATE_OF_BIRTH]
        == "Дата рождения субъекта ПДн"
    )
    assert EntityCategory.RU_DATE_OF_BIRTH in report_html._PII_CATEGORIES
    assert EntityCategory.RU_DATE_OF_BIRTH in report_docx._PII_CATEGORIES
    assert "152-ФЗ" in _classify_incident((EntityCategory.RU_DATE_OF_BIRTH,))

    # Pseudonymization vs route_local distinction in ACTION_REMEDIATION_RU
    assert "псевдонимизац" in ACTION_REMEDIATION_RU[Action.MASK_REVERSIBLE]
    assert "140" in ACTION_REMEDIATION_RU[Action.MASK_REVERSIBLE]
    assert "route_local" in ACTION_REMEDIATION_RU[Action.ROUTE_LOCAL]
    assert "ст. 12" in ACTION_REMEDIATION_RU[Action.ROUTE_LOCAL]
    assert "23-ФЗ" in ACTION_REMEDIATION_RU[Action.ROUTE_LOCAL]

    combined_reg = " | ".join(f"{name}: {desc}" for name, desc in REGULATORY_OVERVIEW_ITEMS)
    assert "23-ФЗ" in combined_reg
    assert "01.07.2025" in combined_reg
    assert "Приказ РКН № 140" in combined_reg
    assert "01.09.2025" in combined_reg
    assert "псевдонимизац" in combined_reg
    assert "ч. 1–2" in combined_reg
    assert "ч. 12–18" in combined_reg
    assert "1 000 субъектов" in combined_reg
    assert "Приказ ФСТЭК России № 117 (от 11.04.2025, с 01.03.2026)" in combined_reg
    assert "разд. 3.18 МД ФСТЭК России от 12.04.2026" in combined_reg


def test_docx_valid_zip_xml_structure_and_content(tmp_path: Path) -> None:
    """DocxComplianceReportGenerator must produce a valid OOXML archive with all sections."""
    events, stats = _sample_events_and_stats()
    fixed_now = datetime(2026, 11, 22, 12, 30, 0, tzinfo=UTC)
    gen = DocxComplianceReportGenerator(now_fn=lambda: fixed_now)

    docx_bytes = gen.render_bytes(events, stats, title="Отчёт аудита Покров — Тест")
    out_file = tmp_path / "compliance_report.docx"
    out_file.write_bytes(docx_bytes)

    assert out_file.stat().st_size > 1000
    assert zipfile.is_zipfile(io.BytesIO(docx_bytes))

    required_entries = {
        "[Content_Types].xml",
        "_rels/.rels",
        "word/_rels/document.xml.rels",
        "word/styles.xml",
        "word/document.xml",
    }
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zf:
        assert zf.testzip() is None
        names = set(zf.namelist())
        assert required_entries.issubset(names)

        # Every XML / .rels archive entry must parse cleanly via ET.fromstring
        for entry_name in names:
            raw_entry = zf.read(entry_name)
            parsed = ET.fromstring(raw_entry)  # noqa: S314
            assert parsed is not None

        doc_text = _extract_docx_text(zf.read("word/document.xml"))

    # Title, generation timestamp, period, disclaimer
    assert "Отчёт аудита Покров — Тест" in doc_text
    assert "2026-11-22 12:30:00 UTC" in doc_text
    assert "2026-11-20 10:00:00 UTC — 2026-11-20 11:00:00 UTC" in doc_text
    assert LEGAL_DISCLAIMER in doc_text

    # KPI values
    for expected in ("150", "12", "34", "9", "5", "88", "2", "3.50", "14.75"):
        assert expected in doc_text

    # Breakdown by EntityCategory and ControlPoint
    assert "ru_passport" in doc_text
    assert "ru_snils" in doc_text
    assert "ru_date_of_birth" in doc_text
    assert "Дата рождения субъекта ПДн" in doc_text
    assert "markdown_exfil" in doc_text
    assert "user_input" in doc_text
    assert "rag_context" in doc_text
    assert "tool_call" in doc_text

    # Regulatory coverage: 152-FZ / 23-FZ / RKN 140, KoAP 13.11 (420-FZ), FSTEC 117 & MD 3.18
    assert "152-ФЗ" in doc_text
    assert "23-ФЗ" in doc_text
    assert "140" in doc_text
    assert "13.11" in doc_text
    assert "420-ФЗ" in doc_text
    assert "117" in doc_text
    assert "3.18" in doc_text
    assert "РСБ" in doc_text

    # Zero-Retention incident summary table
    for ev in events:
        assert ev.event_id in doc_text
        for digest in ev.value_hmacs:
            assert digest in doc_text


def test_docx_incident_table_capped_at_200_rows() -> None:
    """DOCX incident table must cap rows at 200 and include a clear truncation summary line."""
    base_time = datetime(2026, 11, 22, 10, 0, 0, tzinfo=UTC)
    events = [
        AuditEvent(
            event_id=f"cap-{idx:03d}",
            timestamp=base_time + timedelta(seconds=idx),
            request_id=f"req-cap-{idx:03d}",
            control_point=ControlPoint.USER_INPUT,
            action=Action.MASK_REVERSIBLE,
            severity=Severity.MEDIUM,
            categories=(EntityCategory.RU_DATE_OF_BIRTH,),
            value_hmacs=(f"hmac_cap_{idx:03d}",),
        )
        for idx in range(250)
    ]
    docx_bytes = DocxComplianceReportGenerator().render_bytes(
        events,
        AuditStats(total_requests=250, masked_count=250),
    )
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zf:
        doc_text = _extract_docx_text(zf.read("word/document.xml"))

    assert "cap-000" in doc_text
    assert "cap-199" in doc_text
    assert "cap-200" not in doc_text
    assert "cap-249" not in doc_text
    assert "Показаны первые 200 из 250 событий" in doc_text
    assert "скрыто строк: 50" in doc_text


def test_docx_handles_empty_inputs_and_xml_special_control_chars() -> None:
    """DOCX renderer must handle empty reports and sanitize XML-illegal control chars."""
    gen = DocxComplianceReportGenerator()
    empty_bytes = gen.render_bytes([], AuditStats())
    with zipfile.ZipFile(io.BytesIO(empty_bytes), "r") as zf:
        for entry in zf.namelist():
            ET.fromstring(zf.read(entry))  # noqa: S314
        empty_text = _extract_docx_text(zf.read("word/document.xml"))
    assert LEGAL_DISCLAIMER in empty_text
    assert "152-ФЗ" in empty_text
    assert "3.18" in empty_text

    tricky_event = AuditEvent(
        event_id='evt-<&>"\x00\x07',
        timestamp=datetime(2026, 11, 21, 9, 0, 0),
        request_id="req-xml-1",
        principal='user&"<>\x0b',
        control_point=ControlPoint.FINAL_OUTPUT,
        action=Action.FLAG,
        severity=Severity.LOW,
        categories=(EntityCategory.CUSTOM_PATTERN,),
        value_hmacs=("hmac_xml_safe_999",),
        reason="Test <&>'\" \x01\x0c XML escaping\nsecond line",
    )
    tricky_bytes = gen.render_bytes(
        [tricky_event],
        AuditStats(total_requests=1, flagged_count=1),
        title='Title <&> "\x00safe"',
    )
    with zipfile.ZipFile(io.BytesIO(tricky_bytes), "r") as zf:
        for entry in zf.namelist():
            ET.fromstring(zf.read(entry))  # noqa: S314
        tricky_text = _extract_docx_text(zf.read("word/document.xml"))
    assert "hmac_xml_safe_999" in tricky_text
    assert "custom_pattern" in tricky_text
    assert "final_output" in tricky_text


def test_html_report_is_self_contained_offline_and_complete(tmp_path: Path) -> None:
    """HTML5 report must work completely offline with zero external URLs or CDN links."""
    events, stats = _sample_events_and_stats()
    fixed_now = datetime(2026, 11, 22, 14, 0, 0, tzinfo=UTC)
    gen = HtmlComplianceReportGenerator(now_fn=lambda: fixed_now)

    html_bytes = gen.render_bytes(events, stats, title="Покров — Оффлайн отчёт")
    out_file = tmp_path / "compliance_report.html"
    out_file.write_bytes(html_bytes)

    html_text = html_bytes.decode("utf-8")
    assert html_text.startswith("<!DOCTYPE html>")
    assert "<style>" in html_text

    # Strict offline / air-gapped invariant: zero external resources
    lowered = html_text.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "<script" not in lowered
    assert "<link" not in lowered
    assert "@import" not in lowered

    # Core report contents
    assert "Покров — Оффлайн отчёт" in html_text
    assert "2026-11-22 14:00:00 UTC" in html_text
    assert LEGAL_DISCLAIMER in html_text
    assert "152-ФЗ" in html_text
    assert "23-ФЗ" in html_text
    assert "Приказ РКН № 140" in html_text
    assert "13.11" in html_text
    assert "420-ФЗ" in html_text
    assert "ФСТЭК" in html_text
    assert "3.18" in html_text
    assert "РСБ" in html_text
    assert "ГОСТ Р 59709–59712" in html_text
    assert "ГОСТ Р 56545-2015" in html_text
    assert "Дата рождения субъекта ПДн" in html_text
    assert "hmac_passport_aaa111" in html_text
    assert "hmac_inj_ccc333" in html_text
    assert "14.75" in html_text


def test_gost_56545_vuln_vs_gost_59709_incident_cards() -> None:
    """INFRA_MISCONFIG uses GOST R 56545-2015; runtime incidents use GOST R 59709-59712."""
    t0 = datetime(2026, 11, 22, 15, 0, 0, tzinfo=UTC)

    infra_ev = AuditEvent(
        event_id="evt-vuln-01",
        timestamp=t0,
        request_id="req-vuln-01",
        control_point=ControlPoint.IDE_FILE,
        action=Action.FLAG,
        severity=Severity.HIGH,
        categories=(EntityCategory.INFRA_MISCONFIG,),
        reason="Unsafe pickle weight file detected",
    )
    vuln_dict = build_gost_56545_dict(infra_ev)
    assert vuln_dict["standard"] == "ГОСТ Р 56545-2015 (разд. 5, Прил. А)"
    assert vuln_dict["document_type"] == "Паспорт уязвимости"
    vuln_html = render_gost_56545_passport(infra_ev)
    assert "Паспорт уязвимости" in vuln_html
    assert "ГОСТ Р 56545-2015 (разд. 5, Прил. А)" in vuln_html

    for runtime_cat in (
        EntityCategory.SECRET_KEY,
        EntityCategory.PROMPT_INJECTION,
        EntityCategory.RU_PASSPORT,
        EntityCategory.RU_DATE_OF_BIRTH,
        EntityCategory.MCP_POLICY_VIOLATION,
    ):
        inc_ev = AuditEvent(
            event_id=f"evt-inc-{runtime_cat.value}",
            timestamp=t0,
            request_id=f"req-inc-{runtime_cat.value}",
            action=Action.BLOCK,
            severity=Severity.CRITICAL,
            categories=(runtime_cat,),
        )
        inc_dict = build_gost_56545_dict(inc_ev)
        assert inc_dict["standard"] == "ГОСТ Р 59709–59712 (с учётом ГОСТ Р 56545-2015)"
        assert inc_dict["document_type"] == "Карточка события / инцидента ИБ"
        inc_html = render_gost_56545_passport(inc_ev)
        assert "Карточка события / инцидента ИБ" in inc_html
        assert "ГОСТ Р 59709–59712 (с учётом ГОСТ Р 56545-2015)" in inc_html


def test_html_and_gost_passport_xss_escaping_and_structure() -> None:
    """HTML report and GOST incident card must escape all untrusted fields against XSS."""
    xss_payload = '<script>alert("xss")</script>'
    attr_breakout = '"><img src=x onerror=alert(1)>'
    single_quote_xss = "' onclick='alert(2)"

    malicious_event = AuditEvent(
        event_id=f"evt-{xss_payload}",
        timestamp=datetime(2026, 11, 22, 15, 0, 0, tzinfo=UTC),
        request_id=f"req-{attr_breakout}",
        session_id_hmac="sess_xss_hmac",
        principal=f"user-{single_quote_xss}",
        control_point=ControlPoint.TOOL_CALL,
        action=Action.BLOCK,
        severity=Severity.CRITICAL,
        categories=(EntityCategory.PROMPT_INJECTION, EntityCategory.MCP_POLICY_VIOLATION),
        value_hmacs=("hmac_xss_digest_123",),
        detector_names=("prompt_injection",),
        evasion_detected=True,
        upstream=f"up-{xss_payload}",
        policy_profile=f"prof-{attr_breakout}",
        latency_ms=5.12,
        reason=f"Blocked attack: {xss_payload} {attr_breakout}",
        tool_name=f"tool-{xss_payload}",
    )

    # 1. Structured GOST R 59709-59712 runtime incident card dict assertions
    gost_dict = build_gost_56545_dict(malicious_event)
    assert gost_dict["standard"] == "ГОСТ Р 59709–59712 (с учётом ГОСТ Р 56545-2015)"
    assert gost_dict["document_type"] == "Карточка события / инцидента ИБ"
    assert gost_dict["incident_id"] == malicious_event.event_id
    assert gost_dict["request_id"] == malicious_event.request_id
    assert gost_dict["control_point"] == ControlPoint.TOOL_CALL.value
    assert gost_dict["severity"] == Severity.CRITICAL.value
    assert gost_dict["action"] == Action.BLOCK.value
    assert gost_dict["evasion_detected"] == "Да"
    assert "hmac_xss_digest_123" in gost_dict["value_hmacs"]
    assert "prompt_injection" in gost_dict["categories"]
    assert "mcp_policy_violation" in gost_dict["categories"]
    assert "ФСТЭК" in gost_dict["regulatory_basis"]
    assert "ГОСТ Р 59709–59712" in gost_dict["regulatory_basis"]
    assert "Zero-Retention" in gost_dict["zero_retention_status"]
    assert all(isinstance(v, str) for v in gost_dict.values())

    # 2. Standalone GOST HTML card assertions
    passport_html = render_gost_56545_passport(malicious_event)
    assert passport_html.startswith("<!DOCTYPE html>")
    assert "ГОСТ Р 59709–59712 (с учётом ГОСТ Р 56545-2015)" in passport_html
    assert "Карточка события / инцидента ИБ" in passport_html
    assert "hmac_xss_digest_123" in passport_html
    assert "http://" not in passport_html.lower()
    assert "https://" not in passport_html.lower()

    # Raw XSS tags/handlers MUST NOT appear as DOM elements or attributes
    class _DomSafetyParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.tags: list[str] = []
            self.attrs: list[tuple[str, str | None]] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            self.tags.append(tag.lower())
            self.attrs.extend((name.lower(), val) for name, val in attrs)

    for doc in (passport_html,):
        parser = _DomSafetyParser()
        parser.feed(doc)
        assert "script" not in parser.tags
        assert "img" not in parser.tags
        assert not any(attr.startswith("on") for attr, _ in parser.attrs)

    assert "<script>" not in passport_html
    assert "<img " not in passport_html
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in passport_html
    assert "&quot;&gt;&lt;img src=x onerror=alert(1)&gt;" in passport_html
    assert "&#x27; onclick=&#x27;alert(2)" in passport_html

    # 3. Full HTML report XSS escaping
    report_html_str = (
        HtmlComplianceReportGenerator()
        .render_bytes(
            [malicious_event],
            AuditStats(total_requests=1, blocked_count=1),
            title=f"Report {xss_payload}",
        )
        .decode("utf-8")
    )
    report_parser = _DomSafetyParser()
    report_parser.feed(report_html_str)
    assert "script" not in report_parser.tags
    assert "img" not in report_parser.tags
    assert not any(attr.startswith("on") for attr, _ in report_parser.attrs)
    assert "<script>" not in report_html_str
    assert "<img " not in report_html_str
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in report_html_str
    assert "&quot;&gt;&lt;img src=x onerror=alert(1)&gt;" in report_html_str


def test_zero_retention_no_raw_pii_in_docx_html_or_passport(tmp_path: Path) -> None:
    """Raw PII and secrets must NEVER appear anywhere in generated DOCX XML, HTML, or passports."""
    salt = b"test-zero-retention-salt"
    synthetic_secrets = [
        ("4512 654321", EntityCategory.RU_PASSPORT),
        ("112-233-445 95", EntityCategory.RU_SNILS),
        ("7707083893", EntityCategory.RU_INN),
        ("2200700123456789", EntityCategory.RU_BANK_CARD),
        ("+7-916-555-35-35", EntityCategory.RU_PHONE),
        ("14.08.1987", EntityCategory.RU_DATE_OF_BIRTH),
        ("ivan.petrov@example.ru", EntityCategory.EMAIL),
        (_tok("sk-", "proj-", "verySecretOpenAiToken987654321"), EntityCategory.SECRET_KEY),
        (_tok("sk-", "ant-", "api03-secretAnthropicKey12345"), EntityCategory.SECRET_KEY),
        (_tok("AQ", "VN", "secretYandexCloudKey9876543210"), EntityCategory.SECRET_KEY),
        ("Иванов Иван Иванович", EntityCategory.RU_PERSON_NAME),
        ("г. Москва, ул. Тверская, д. 10, кв. 42", EntityCategory.RU_ADDRESS),
        (_tok("-----BEGIN ", "PRIVATE KEY-----", "MIIEvQIBADANBg"), EntityCategory.SECRET_KEY),
    ]

    events: list[AuditEvent] = []
    base_time = datetime(2026, 11, 22, 8, 0, 0, tzinfo=UTC)
    expected_digests: list[str] = []

    for idx, (raw_value, category) in enumerate(synthetic_secrets):
        digest = hmac.new(salt, raw_value.encode("utf-8"), hashlib.sha256).hexdigest()
        expected_digests.append(digest)
        events.append(
            AuditEvent(
                event_id=f"zr-evt-{idx:02d}",
                timestamp=base_time + timedelta(seconds=idx * 10),
                request_id=f"zr-req-{idx:02d}",
                session_id_hmac=f"zr-sess-{idx:02d}",
                principal="qa-operator",
                control_point=ControlPoint.USER_INPUT,
                action=Action.MASK_REVERSIBLE,
                severity=Severity.CRITICAL,
                categories=(category,),
                value_hmacs=(digest,),
                detector_names=("ru_pii",),
                upstream="mock-local",
                policy_profile="balanced-152fz",
                latency_ms=2.1,
                compliance_refs=get_compliance_refs(category),
                reason=f"Policy matched category {category.value}",
            )
        )

    stats = AuditStats(
        total_requests=len(events),
        masked_count=len(events),
        p50_latency_ms=2.1,
        p95_latency_ms=2.1,
    )

    docx_bytes = DocxComplianceReportGenerator().render_bytes(events, stats)
    html_bytes = HtmlComplianceReportGenerator().render_bytes(events, stats)

    docx_path = tmp_path / "zr.docx"
    html_path = tmp_path / "zr.html"
    docx_path.write_bytes(docx_bytes)
    html_path.write_bytes(html_bytes)

    html_text = html_path.read_text(encoding="utf-8")
    passport_texts = [render_gost_56545_passport(ev) for ev in events]
    passport_dicts = [build_gost_56545_dict(ev) for ev in events]

    # Collect all raw and XML bytes inside the .docx archive
    docx_xml_blobs: list[str] = []
    with zipfile.ZipFile(docx_path, "r") as zf:
        for name in zf.namelist():
            docx_xml_blobs.append(zf.read(name).decode("utf-8"))
    combined_docx_xml = "\n".join(docx_xml_blobs)

    for raw_secret, _ in synthetic_secrets:
        assert raw_secret.encode("utf-8") not in docx_bytes
        assert raw_secret not in combined_docx_xml
        assert raw_secret.encode("utf-8") not in html_bytes
        assert raw_secret not in html_text
        for p_html in passport_texts:
            assert raw_secret not in p_html
        for p_dict in passport_dicts:
            assert all(raw_secret not in val for val in p_dict.values())

    # Confirm all HMAC digests were faithfully rendered in both formats
    for digest in expected_digests:
        assert digest in combined_docx_xml
        assert digest in html_text


def test_empty_stats_fallback_aggregates_from_events_and_large_batch() -> None:
    """When stats dicts are empty, category and control point breakdowns fall back to events."""
    base_time = datetime(2026, 11, 22, 9, 0, 0, tzinfo=UTC)
    events = [
        AuditEvent(
            event_id=f"batch-{i:03d}",
            timestamp=base_time + timedelta(seconds=i),
            request_id=f"req-{i:03d}",
            control_point=ControlPoint.IDE_FILE if i % 2 == 0 else ControlPoint.FINAL_OUTPUT,
            action=Action.MASK,
            severity=Severity.HIGH,
            categories=(EntityCategory.RU_PHONE,),
            value_hmacs=(f"hmac_{i:03d}",),
        )
        for i in range(120)
    ]
    # Minimal AuditStats with empty by_category / by_control_point and no period bounds
    stats = AuditStats(total_requests=120, masked_count=120)

    docx_bytes = DocxComplianceReportGenerator().render_bytes(events, stats)
    html_text = HtmlComplianceReportGenerator().render_bytes(events, stats).decode("utf-8")

    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zf:
        doc_text = _extract_docx_text(zf.read("word/document.xml"))

    # Period derived from events, fallback breakdowns populated
    assert "2026-11-22 09:00:00 UTC" in doc_text
    assert "ru_phone" in doc_text
    assert "ide_file" in doc_text
    assert "final_output" in doc_text
    assert "120" in doc_text

    assert "ru_phone" in html_text
    assert "ide_file" in html_text
    assert "batch-119" in html_text
    assert re.search(r"Отображены первые 100 паспортов из 120 событий", html_text)


def test_html_empty_and_edge_variants() -> None:
    """Cover empty HTML reports, open-ended periods, custom stats keys, and GOST edge cases."""
    html_gen = HtmlComplianceReportGenerator()
    empty_html = html_gen.render_bytes([], AuditStats()).decode("utf-8")
    assert "Обнаружения по категориям за период отсутствуют." in empty_html
    assert "События по контрольным точкам отсутствуют." in empty_html
    assert "События инцидентов отсутствуют." in empty_html
    assert "Паспорта инцидентов не сформированы." in empty_html

    t0 = datetime(2026, 11, 22, 6, 0, 0, tzinfo=UTC)
    start_only_html = html_gen.render_bytes(
        [],
        AuditStats(
            period_start=t0,
            by_category={"custom_ext_cat": 3, EntityCategory.RU_PASSPORT.value: 2},
            by_control_point={"custom_cp": 4},
        ),
    ).decode("utf-8")
    assert "с 2026-11-22 06:00:00 UTC" in start_only_html
    assert "custom_ext_cat" in start_only_html
    assert "custom_cp" in start_only_html

    end_only_html = html_gen.render_bytes([], AuditStats(period_end=t0)).decode("utf-8")
    assert "по 2026-11-22 06:00:00 UTC" in end_only_html

    # Empty categories passport
    empty_cat_ev = AuditEvent(
        event_id="evt-empty-cat",
        timestamp=t0,
        request_id="req-empty-cat",
        action=Action.ALLOW,
    )
    empty_dict = build_gost_56545_dict(empty_cat_ev)
    assert empty_dict["categories"] == "none"
    assert empty_dict["standard"] == "ГОСТ Р 59709–59712 (с учётом ГОСТ Р 56545-2015)"
    assert "ФСТЭК России № 117" in empty_dict["regulatory_basis"]

    # Infra + toxic + route_local + custom compliance ref
    infra_ev = AuditEvent(
        event_id="evt-infra",
        timestamp=t0,
        request_id="req-infra",
        action=Action.ROUTE_LOCAL,
        severity=Severity.HIGH,
        categories=(EntityCategory.INFRA_MISCONFIG, EntityCategory.TOXIC_OR_LEAK),
        upstream="ollama-local",
        compliance_refs=(
            ComplianceRef(
                framework="Внутренний стандарт", article="п. 4.2", title="Контроль весов"
            ),
        ),
    )
    infra_dict = build_gost_56545_dict(infra_ev)
    assert infra_dict["standard"] == "ГОСТ Р 56545-2015 (разд. 5, Прил. А)"
    assert infra_dict["document_type"] == "Паспорт уязвимости"
    assert "ollama-local" in infra_dict["remediation"]
    assert "Конфигурационная уязвимость" in infra_dict["incident_class"]
    assert "Внутренний стандарт" in infra_dict["regulatory_basis"]

    # Event list with custom compliance ref + category present in events but not stats.by_category
    custom_ref_html = html_gen.render_bytes(
        [infra_ev],
        AuditStats(
            total_requests=1,
            routed_local_count=1,
            by_category={EntityCategory.RU_SNILS.value: 1},
        ),
    ).decode("utf-8")
    assert "Внутренний стандарт" in custom_ref_html
    assert "infra_misconfig" in custom_ref_html


def test_cli_serve_workers_timeout_report_and_planned_help(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI [planned] help markers, serve workers/timeout_keep_alive, and report command."""
    help_text = cli.build_parser().format_help()
    for planned_cmd in ("mcp", "audit", "bench", "protect", "watch", "restore", "hook"):
        assert re.search(rf"\b{planned_cmd}\b.*\[planned\]", help_text)

    # 1. RAM vault + server.workers > 1 -> clamps workers=1 and logs WARNING
    uvicorn_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *a, **kw: uvicorn_calls.append((a, kw)),
    )

    ram_yaml = tmp_path / "ram_workers.yaml"
    ram_yaml.write_text(
        "server:\n  workers: 4\n  request_timeout_seconds: 45.5\n"
        "vault:\n  mode: memory\n"
        f"audit:\n  db_path: {tmp_path / 'cli_ram.db'!s}\n  hmac_salt: 0123456789abcdef\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.WARNING, logger="pokrov.cli")
    caplog.clear()
    assert cli.main(["-c", str(ram_yaml), "serve", "--host", "127.0.0.1", "--port", "8181"]) == 0
    assert any("Forcing workers=1" in rec.message for rec in caplog.records)
    assert len(uvicorn_calls) == 1
    _, kw_ram = uvicorn_calls[0]
    assert kw_ram["workers"] == 1
    assert kw_ram["timeout_keep_alive"] == 46
    assert kw_ram["port"] == 8181

    # 2. local_encrypted vault + server.workers == 3 -> preserves workers=3 without warning
    cfg_enc = PokrovConfig(
        server=ServerConfig(workers=3, request_timeout_seconds=30.0),
        vault=VaultConfig(mode="local_encrypted", local_vault_path=str(tmp_path / "vault.enc")),
        audit=AuditConfig(db_path=str(tmp_path / "cli_enc.db"), hmac_salt="0123456789abcdef"),
    )
    caplog.clear()
    w_enc, keep_alive_enc = cli._resolve_serve_runtime_options(cfg_enc)
    assert w_enc == 3
    assert keep_alive_enc == 30
    assert not caplog.records

    # 3. CLI report command exports DOCX and JSONL from SQLite audit DB
    report_db = tmp_path / "report_cli.db"
    with SqliteAuditStore(AuditConfig(db_path=str(report_db), hmac_salt="0123456789abcdef")) as st:
        st.record(
            AuditEvent(
                event_id="cli-evt-1",
                timestamp=datetime.now(UTC),
                request_id="cli-req-1",
                action=Action.MASK_REVERSIBLE,
                severity=Severity.MEDIUM,
                categories=(EntityCategory.RU_DATE_OF_BIRTH,),
                value_hmacs=("c" * 64,),
            )
        )
    rep_yaml = tmp_path / "rep.yaml"
    rep_yaml.write_text(
        f"audit:\n  db_path: {report_db!s}\n  hmac_salt: 0123456789abcdef\n",
        encoding="utf-8",
    )
    out_docx = tmp_path / "out.docx"
    assert (
        cli.main(["-c", str(rep_yaml), "report", "--format", "docx", "--output", str(out_docx)])
        == 0
    )
    assert zipfile.is_zipfile(out_docx)
    capsys.readouterr()

    assert cli.main(["-c", str(rep_yaml), "report", "--format", "jsonl"]) == 0
    jsonl_stdout = capsys.readouterr().out.strip()
    assert "cli-evt-1" in jsonl_stdout
