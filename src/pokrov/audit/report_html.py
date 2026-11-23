"""Self-contained offline HTML5 compliance report and GOST R 56545-2015 passport generator."""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pokrov.core.compliance_map import (
    LEGAL_DISCLAIMER,
    aggregate_compliance_refs,
    get_compliance_refs,
)
from pokrov.core.contracts import (
    Action,
    AuditEvent,
    AuditStats,
    ComplianceRef,
    ControlPoint,
    EntityCategory,
    Severity,
)

CATEGORY_LABELS_RU: dict[EntityCategory, str] = {
    EntityCategory.RU_PASSPORT: "Паспорт гражданина РФ",
    EntityCategory.RU_SNILS: "СНИЛС",
    EntityCategory.RU_INN: "ИНН",
    EntityCategory.RU_OGRN: "ОГРН / ОГРНИП",
    EntityCategory.RU_BANK_CARD: "Банковская карта",
    EntityCategory.RU_BIK_ACCOUNT: "БИК и банковский счёт",
    EntityCategory.RU_OMS: "Полис ОМС",
    EntityCategory.RU_PHONE: "Телефонный номер РФ",
    EntityCategory.RU_AUTO_PLATE: "Госномер ТС РФ",
    EntityCategory.RU_DRIVER_LICENSE: "Водительское удостоверение",
    EntityCategory.RU_PERSON_NAME: "ФИО субъекта ПДн",
    EntityCategory.RU_ADDRESS: "Адрес проживания / регистрации",
    EntityCategory.RU_ORG_NAME: "Наименование организации",
    EntityCategory.EMAIL: "Электронная почта (Email)",
    EntityCategory.IP_ADDRESS: "IP-адрес",
    EntityCategory.SECRET_KEY: "Секрет / API-ключ / учётные данные",
    EntityCategory.CUSTOM_PATTERN: "Пользовательское правило",
    EntityCategory.PROMPT_INJECTION: "Прямая атака Prompt Injection",
    EntityCategory.INDIRECT_INJECTION: "Косвенная атака Indirect Injection",
    EntityCategory.MARKDOWN_EXFIL: "Эксфильтрация через Markdown / HTML",
    EntityCategory.TOXIC_OR_LEAK: "Нежелательный контент / утечка",
    EntityCategory.MCP_POLICY_VIOLATION: "Нарушение политики MCP-инструментов",
    EntityCategory.INFRA_MISCONFIG: "Уязвимость конфигурации ИИ-инфраструктуры",
}

CONTROL_POINT_LABELS_RU: dict[ControlPoint, str] = {
    ControlPoint.USER_INPUT: "Входной запрос пользователя",
    ControlPoint.RAG_CONTEXT: "Контекст RAG / базы знаний",
    ControlPoint.MODEL_PROMPT: "Итоговый промпт модели",
    ControlPoint.TOOL_CALL: "Вызов инструмента (MCP / tool_call)",
    ControlPoint.TOOL_RESULT: "Ответ инструмента (tool_result)",
    ControlPoint.FINAL_OUTPUT: "Исходящий ответ модели",
    ControlPoint.IDE_FILE: "Локальный файл рабочей среды IDE",
}

ACTION_LABELS_RU: dict[Action, str] = {
    Action.ALLOW: "Разрешено",
    Action.FLAG: "Зафиксировано (флаг)",
    Action.MASK: "Статическое маскирование",
    Action.MASK_REVERSIBLE: "Обратимое маскирование (Vault)",
    Action.ROUTE_LOCAL: "Перенаправление в локальный контур",
    Action.REQUIRE_APPROVAL: "Запрос HITL-одобрения",
    Action.BLOCK: "Заблокировано",
}

ACTION_REMEDIATION_RU: dict[Action, str] = {
    Action.ALLOW: "Передача разрешена без модификации; событие зарегистрировано в аудите.",
    Action.FLAG: "Трафик пропущен в режиме мониторинга; инцидент зафиксирован по правилам РСБ.",
    Action.MASK: "Критические фрагменты заменены необратимыми масками до отправки во внешнюю LLM.",
    Action.MASK_REVERSIBLE: (
        "Данные токенизированы через AES-256-GCM Vault с временным TTL; "
        "во внешний контур переданы только обезличенные плейсхолдеры."
    ),
    Action.ROUTE_LOCAL: (
        "Запрос автоматически перенаправлен в доверенный локальный / российский контур ИИ "
        "для исключения трансграничной передачи ПДн."
    ),
    Action.REQUIRE_APPROVAL: (
        "Исполнение операции приостановлено до явного подтверждения оператором (HITL)."
    ),
    Action.BLOCK: ("Запрос или ответ полностью заблокирован шлюзом Покров до передачи получателю."),
}

SEVERITY_LABELS_RU: dict[Severity, str] = {
    Severity.LOW: "Низкий",
    Severity.MEDIUM: "Средний",
    Severity.HIGH: "Высокий",
    Severity.CRITICAL: "Критический",
}

_PII_CATEGORIES: frozenset[EntityCategory] = frozenset(
    {
        EntityCategory.RU_PASSPORT,
        EntityCategory.RU_SNILS,
        EntityCategory.RU_INN,
        EntityCategory.RU_OGRN,
        EntityCategory.RU_BANK_CARD,
        EntityCategory.RU_BIK_ACCOUNT,
        EntityCategory.RU_OMS,
        EntityCategory.RU_PHONE,
        EntityCategory.RU_AUTO_PLATE,
        EntityCategory.RU_DRIVER_LICENSE,
        EntityCategory.RU_PERSON_NAME,
        EntityCategory.RU_ADDRESS,
        EntityCategory.RU_ORG_NAME,
        EntityCategory.EMAIL,
        EntityCategory.IP_ADDRESS,
    }
)

_INJECTION_CATEGORIES: frozenset[EntityCategory] = frozenset(
    {
        EntityCategory.PROMPT_INJECTION,
        EntityCategory.INDIRECT_INJECTION,
        EntityCategory.MARKDOWN_EXFIL,
    }
)

REGULATORY_OVERVIEW_ITEMS: tuple[tuple[str, str], ...] = (
    (
        "152-ФЗ (ст. 12, ч. 5 ст. 18, ст. 19)",
        "Локализация первичной обработки ПДн граждан РФ, контроль трансграничной передачи "
        "в зарубежные LLM, обезличивание и режим Zero-Retention без хранения сырых ПДн в логах.",
    ),
    (
        "ст. 13.11 КоАП РФ (в ред. 420-ФЗ)",
        "Снижение риска оборотных штрафов (ч. 12–16) за утечку ПДн и штрафов (ч. 8–9) "
        "за нарушение требований о локализации баз данных граждан РФ.",
    ),
    (
        "Приказ ФСТЭК России № 117 (п. 3.18, РСБ)",
        "Фильтрация входящих запросов и исходящих ответов моделей ИИ, контроль автономных "
        "агентов, регистрация событий безопасности (РСБ) и экспорт в корпоративные SIEM.",
    ),
    (
        "ГОСТ Р 56545-2015 (разд. 5)",
        "Стандартизованное представление паспорта инцидента информационной безопасности "
        "для расследований, аудита и отчётности.",
    ),
)


@dataclass(frozen=True)
class KpiRow:
    """Single KPI metric row in compliance summaries."""

    key: str
    label: str
    value: str


@dataclass(frozen=True)
class CategoryBreakdownRow:
    """Aggregated statistics row for an EntityCategory."""

    raw_key: str
    label_ru: str
    count: int
    frameworks: str


@dataclass(frozen=True)
class ControlPointBreakdownRow:
    """Aggregated statistics row for a ControlPoint."""

    raw_key: str
    label_ru: str
    count: int


@dataclass(frozen=True)
class RegulatoryCoverageRow:
    """Mapping row linking a regulatory requirement to triggered categories."""

    framework: str
    article: str
    title: str
    risk_hint: str
    categories: str
    hits: int


@dataclass(frozen=True)
class ReportContext:
    """Normalized, Zero-Retention report data consumed by HTML and DOCX renderers."""

    title: str
    generated_at_str: str
    period_str: str
    legal_disclaimer: str
    kpis: tuple[KpiRow, ...]
    category_rows: tuple[CategoryBreakdownRow, ...]
    control_point_rows: tuple[ControlPointBreakdownRow, ...]
    regulatory_rows: tuple[RegulatoryCoverageRow, ...]
    events_count: int


def format_timestamp(dt: datetime) -> str:
    """Format a datetime consistently in UTC (or naive ISO-like string)."""
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_period(stats: AuditStats, events: Sequence[AuditEvent]) -> str:
    """Build a human-readable reporting period from stats or event timestamps."""
    start = stats.period_start
    end = stats.period_end
    if start is None and end is None and events:
        timestamps = [event.timestamp for event in events]
        start = min(timestamps)
        end = max(timestamps)
    if start is not None and end is not None:
        return f"{format_timestamp(start)} — {format_timestamp(end)}"
    if start is not None:
        return f"с {format_timestamp(start)}"
    if end is not None:
        return f"по {format_timestamp(end)}"
    return "не задан (события за период отсутствуют)"


def format_category_label(raw_key: str) -> str:
    """Return Russian label plus enum key for an EntityCategory string."""
    try:
        cat = EntityCategory(raw_key)
    except ValueError:
        return raw_key
    return f"{CATEGORY_LABELS_RU.get(cat, cat.value)} ({cat.value})"


def format_control_point_label(raw_key: str) -> str:
    """Return Russian label plus enum key for a ControlPoint string."""
    try:
        cp = ControlPoint(raw_key)
    except ValueError:
        return raw_key
    return f"{CONTROL_POINT_LABELS_RU.get(cp, cp.value)} ({cp.value})"


def _classify_incident(categories: tuple[EntityCategory, ...]) -> str:
    """Map event categories to a GOST R 56545-2015 incident class label."""
    if not categories:
        return "Событие информационной безопасности ИИ"
    cat_set = set(categories)
    classes: list[str] = []
    if cat_set & _PII_CATEGORIES:
        classes.append("Угроза утечки / неправомерной передачи ПДн (152-ФЗ)")
    if EntityCategory.SECRET_KEY in cat_set:
        classes.append("Утечка секретов и аутентификационных данных")
    if cat_set & _INJECTION_CATEGORIES:
        classes.append("Атака на модель ИИ (Prompt Injection / эксфильтрация)")
    if EntityCategory.MCP_POLICY_VIOLATION in cat_set:
        classes.append("Нарушение политики безопасности ИИ-агента (MCP)")
    if EntityCategory.INFRA_MISCONFIG in cat_set:
        classes.append("Конфигурационная уязвимость ИИ-инфраструктуры")
    if EntityCategory.CUSTOM_PATTERN in cat_set or EntityCategory.TOXIC_OR_LEAK in cat_set:
        classes.append("Нарушение корпоративной политики контента")
    return "; ".join(classes) if classes else "Событие информационной безопасности ИИ"


def _collect_event_compliance_refs(event: AuditEvent) -> tuple[ComplianceRef, ...]:
    """Return deduplicated ComplianceRef tuple from event or category map."""
    if event.compliance_refs:
        seen: set[tuple[str, str]] = set()
        ordered: list[ComplianceRef] = []
        for ref in event.compliance_refs:
            key = (ref.framework, ref.article)
            if key not in seen:
                seen.add(key)
                ordered.append(ref)
        for ref in aggregate_compliance_refs(event.categories):
            key = (ref.framework, ref.article)
            if key not in seen:
                seen.add(key)
                ordered.append(ref)
        return tuple(ordered)
    return aggregate_compliance_refs(event.categories)


def build_report_context(
    events: Sequence[AuditEvent],
    stats: AuditStats,
    *,
    title: str,
    generated_at: datetime,
) -> ReportContext:
    """Aggregate AuditStats and AuditEvent sequences into Zero-Retention report tables."""
    kpis = (
        KpiRow("total_requests", "Всего проверено запросов", str(stats.total_requests)),
        KpiRow("blocked_count", "Заблокировано (block)", str(stats.blocked_count)),
        KpiRow("masked_count", "Замаскировано (mask / mask_reversible)", str(stats.masked_count)),
        KpiRow(
            "routed_local_count",
            "Перенаправлено в локальный/РФ контур (route_local)",
            str(stats.routed_local_count),
        ),
        KpiRow("flagged_count", "Зафиксировано флагом (flag)", str(stats.flagged_count)),
        KpiRow(
            "approval_required_count",
            "Требует HITL-подтверждения (require_approval)",
            str(stats.approval_required_count),
        ),
        KpiRow("allowed_count", "Разрешено без вмешательства (allow)", str(stats.allowed_count)),
        KpiRow("p50_latency_ms", "Задержка инспекции p50 (мс)", f"{stats.p50_latency_ms:.2f}"),
        KpiRow("p95_latency_ms", "Задержка инспекции p95 (мс)", f"{stats.p95_latency_ms:.2f}"),
    )

    cat_counts: dict[str, int]
    if stats.by_category:
        cat_counts = dict(stats.by_category)
    else:
        counter: Counter[str] = Counter()
        for ev in events:
            for cat in ev.categories:
                counter[cat.value] += 1
        cat_counts = dict(counter)

    category_rows: list[CategoryBreakdownRow] = []
    detected_enums: list[EntityCategory] = []
    seen_enum_set: set[EntityCategory] = set()

    for raw_key, count in sorted(cat_counts.items(), key=lambda item: (-item[1], item[0])):
        try:
            cat_enum = EntityCategory(raw_key)
        except ValueError:
            frameworks_str = "Пользовательская категория"
        else:
            if cat_enum not in seen_enum_set:
                seen_enum_set.add(cat_enum)
                detected_enums.append(cat_enum)
            refs = get_compliance_refs(cat_enum)
            frameworks_str = ", ".join(dict.fromkeys(f"{r.framework} {r.article}" for r in refs))
        category_rows.append(
            CategoryBreakdownRow(
                raw_key=raw_key,
                label_ru=format_category_label(raw_key),
                count=count,
                frameworks=frameworks_str,
            )
        )

    for ev in events:
        for cat_enum in ev.categories:
            if cat_enum not in seen_enum_set:
                seen_enum_set.add(cat_enum)
                detected_enums.append(cat_enum)

    cp_counts: dict[str, int]
    if stats.by_control_point:
        cp_counts = dict(stats.by_control_point)
    else:
        cp_counter: Counter[str] = Counter()
        for ev in events:
            cp_counter[ev.control_point.value] += 1
        cp_counts = dict(cp_counter)

    control_point_rows = tuple(
        ControlPointBreakdownRow(
            raw_key=raw_key,
            label_ru=format_control_point_label(raw_key),
            count=count,
        )
        for raw_key, count in sorted(cp_counts.items(), key=lambda item: (-item[1], item[0]))
    )

    # Map (framework, article) to ComplianceRef, triggered categories, and hit counts.
    ref_map: dict[tuple[str, str], ComplianceRef] = {}
    ref_cats: dict[tuple[str, str], list[str]] = {}
    ref_hits: Counter[tuple[str, str]] = Counter()

    for cat_enum in detected_enums:
        cat_hits = cat_counts.get(cat_enum.value, 0)
        if cat_hits == 0:
            cat_hits = sum(1 for ev in events if cat_enum in ev.categories)
        for ref in get_compliance_refs(cat_enum):
            key = (ref.framework, ref.article)
            ref_map.setdefault(key, ref)
            cat_list = ref_cats.setdefault(key, [])
            if cat_enum.value not in cat_list:
                cat_list.append(cat_enum.value)
            ref_hits[key] += max(cat_hits, 1)

    for ev in events:
        for ref in ev.compliance_refs:
            key = (ref.framework, ref.article)
            ref_map.setdefault(key, ref)
            cat_list = ref_cats.setdefault(key, [])
            for cat_enum in ev.categories:
                if cat_enum.value not in cat_list:
                    cat_list.append(cat_enum.value)
            if key not in ref_hits:
                ref_hits[key] += 1

    regulatory_rows = tuple(
        RegulatoryCoverageRow(
            framework=ref.framework,
            article=ref.article,
            title=ref.title,
            risk_hint=ref.risk_hint,
            categories=", ".join(ref_cats.get(key, [])) or "—",
            hits=ref_hits[key],
        )
        for key, ref in ref_map.items()
    )

    return ReportContext(
        title=title,
        generated_at_str=format_timestamp(generated_at),
        period_str=format_period(stats, events),
        legal_disclaimer=LEGAL_DISCLAIMER,
        kpis=kpis,
        category_rows=tuple(category_rows),
        control_point_rows=control_point_rows,
        regulatory_rows=regulatory_rows,
        events_count=len(events),
    )


def build_gost_56545_dict(event: AuditEvent) -> dict[str, str]:
    """Map a Zero-Retention AuditEvent to structured GOST R 56545-2015 incident passport fields."""
    refs = _collect_event_compliance_refs(event)
    regulatory_basis = (
        "; ".join(f"{ref.framework} ({ref.article}): {ref.title}" for ref in refs)
        if refs
        else "Приказ ФСТЭК России № 117 (РСБ / п. 3.18): Регистрация событий безопасности"
    )

    remediation = ACTION_REMEDIATION_RU.get(
        event.action,
        f"Применено действие политики: {event.action.value}.",
    )
    if event.action == Action.ROUTE_LOCAL and event.upstream:
        remediation = f"{remediation} Целевой локальный контур: {event.upstream}."

    categories_str = (
        ", ".join(cat.value for cat in event.categories) if event.categories else "none"
    )
    categories_ru = (
        ", ".join(CATEGORY_LABELS_RU.get(cat, cat.value) for cat in event.categories)
        if event.categories
        else "Категории угроз не заданы"
    )
    description = (
        event.reason.strip()
        if event.reason.strip()
        else (
            f"Зафиксировано событие на контрольной точке '{event.control_point.value}' "
            f"по категориям: {categories_str}."
        )
    )

    return {
        "standard": "ГОСТ Р 56545-2015 (разд. 5)",
        "incident_id": event.event_id,
        "request_id": event.request_id,
        "detected_at": format_timestamp(event.timestamp),
        "incident_class": _classify_incident(event.categories),
        "categories": categories_str,
        "categories_ru": categories_ru,
        "severity": event.severity.value,
        "severity_ru": SEVERITY_LABELS_RU.get(event.severity, event.severity.value),
        "control_point": event.control_point.value,
        "control_point_ru": CONTROL_POINT_LABELS_RU.get(
            event.control_point, event.control_point.value
        ),
        "principal": event.principal,
        "session_id_hmac": event.session_id_hmac or "—",
        "action": event.action.value,
        "action_ru": ACTION_LABELS_RU.get(event.action, event.action.value),
        "evasion_detected": "Да" if event.evasion_detected else "Нет",
        "detectors": ", ".join(event.detector_names) if event.detector_names else "—",
        "upstream": event.upstream or "—",
        "policy_profile": event.policy_profile or "—",
        "tool_name": event.tool_name or "—",
        "latency_ms": f"{event.latency_ms:.2f}",
        "value_hmacs": ", ".join(event.value_hmacs) if event.value_hmacs else "—",
        "description": description,
        "remediation": remediation,
        "regulatory_basis": regulatory_basis,
        "zero_retention_status": (
            "Zero-Retention: сырые ПДн, секреты и исходные промпты не сохраняются; "
            "в паспорте отражены только HMAC-дайджесты и метаданные."
        ),
    }


def _esc(value: str) -> str:
    """HTML-escape text including quotes for XSS-safe rendering."""
    return html.escape(value, quote=True)


_OFFLINE_CSS = """
:root {
  --bg: #f8fafc;
  --surface: #ffffff;
  --border: #cbd5e1;
  --text: #0f172a;
  --muted: #475569;
  --accent: #1e3a5f;
  --accent-soft: #e2e8f0;
  --warn-bg: #fffbeb;
  --warn-border: #d97706;
  --crit: #991b1b;
  --crit-bg: #fee2e2;
  --high: #9a3412;
  --high-bg: #ffedd5;
  --med: #854d0e;
  --med-bg: #fef9c3;
  --low: #166534;
  --low-bg: #dcfce7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  font-size: 14px;
}
.container { max-width: 1200px; margin: 0 auto; }
header.hero {
  background: var(--accent);
  color: #ffffff;
  padding: 24px 28px;
  border-radius: 10px;
  margin-bottom: 20px;
}
header.hero h1 { margin: 0 0 8px 0; font-size: 24px; font-weight: 700; }
.meta-row { display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; opacity: 0.92; }
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.16);
  color: #ffffff;
}
.disclaimer {
  background: var(--warn-bg);
  border-left: 4px solid var(--warn-border);
  border-radius: 6px;
  padding: 14px 18px;
  margin-bottom: 22px;
  font-size: 13px;
  color: #78350f;
}
section.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px 24px;
  margin-bottom: 20px;
}
section.panel h2 {
  margin: 0 0 14px 0;
  font-size: 18px;
  color: var(--accent);
  border-bottom: 1px solid var(--accent-soft);
  padding-bottom: 8px;
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.kpi-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
}
.kpi-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.kpi-value { font-size: 22px; font-weight: 700; color: var(--accent); }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th, td {
  border: 1px solid var(--border);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #f1f5f9;
  color: var(--accent);
  font-weight: 600;
}
tr:nth-child(even) td { background: #fcfdff; }
.mono {
  font-family: "Cascadia Code", Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  word-break: break-all;
}
.sev {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
  font-size: 12px;
}
.sev-critical { background: var(--crit-bg); color: var(--crit); }
.sev-high { background: var(--high-bg); color: var(--high); }
.sev-medium { background: var(--med-bg); color: var(--med); }
.sev-low { background: var(--low-bg); color: var(--low); }
.gost-passport {
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: 4px solid var(--accent);
  border-radius: 8px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
.gost-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--accent-soft);
}
.gost-title { font-size: 16px; font-weight: 700; color: var(--accent); }
.gost-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 10px 18px;
}
.gost-field { border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px; }
.gost-key { font-size: 11px; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.gost-val { font-size: 13px; color: var(--text); margin-top: 2px; word-break: break-word; }
.gost-footer {
  margin-top: 12px;
  font-size: 12px;
  color: var(--muted);
  background: #f8fafc;
  padding: 8px 10px;
  border-radius: 6px;
}
.empty-note { color: var(--muted); font-style: italic; margin: 6px 0; }
@media print {
  body { padding: 0; background: #ffffff; }
  section.panel, .gost-passport { break-inside: avoid; }
}
""".strip()


def _severity_css_class(severity: Severity) -> str:
    return f"sev-{severity.value}"


def _render_passport_article(event: AuditEvent) -> str:
    """Render an escaped HTML5 <article> block for one GOST R 56545-2015 passport."""
    fields = build_gost_56545_dict(event)
    sev_class = _severity_css_class(event.severity)

    passport_rows: tuple[tuple[str, str, bool], ...] = (
        ("Стандарт", fields["standard"], False),
        ("Идентификатор инцидента", fields["incident_id"], True),
        ("Идентификатор запроса", fields["request_id"], True),
        ("Дата и время обнаружения", fields["detected_at"], False),
        ("Класс инцидента", fields["incident_class"], False),
        ("Категории (коды)", fields["categories"], True),
        ("Категории (описание)", fields["categories_ru"], False),
        ("Уровень критичности", f"{fields['severity_ru']} ({fields['severity']})", False),
        ("Контрольная точка", f"{fields['control_point_ru']} ({fields['control_point']})", False),
        ("Субъект (инициатор)", fields["principal"], False),
        ("HMAC сессии", fields["session_id_hmac"], True),
        ("Применённое действие", f"{fields['action_ru']} ({fields['action']})", False),
        ("Признаки обхода (анти-обход)", fields["evasion_detected"], False),
        ("Детекторы", fields["detectors"], False),
        ("Профиль политик", fields["policy_profile"], False),
        ("Целевой upstream", fields["upstream"], False),
        ("Инструмент (MCP / tool)", fields["tool_name"], False),
        ("Задержка инспекции (мс)", fields["latency_ms"], False),
        ("HMAC-дайджесты (Zero-Retention)", fields["value_hmacs"], True),
        ("Описание инцидента", fields["description"], False),
        ("Меры реагирования", fields["remediation"], False),
        ("Нормативное основание", fields["regulatory_basis"], False),
    )

    grid_items: list[str] = []
    for label, value, is_mono in passport_rows:
        val_cls = "gost-val mono" if is_mono else "gost-val"
        grid_items.append(
            f'<div class="gost-field">'
            f'<div class="gost-key">{_esc(label)}</div>'
            f'<div class="{val_cls}">{_esc(value)}</div>'
            f"</div>"
        )

    return (
        '<article class="gost-passport">'
        '<div class="gost-header">'
        f'<div class="gost-title">Паспорт инцидента ИБ ({_esc(fields["standard"])}): '
        f'<span class="mono">{_esc(fields["incident_id"])}</span></div>'
        f'<span class="sev {_esc(sev_class)}">{_esc(fields["severity_ru"])} '
        f"({_esc(fields['severity'])})</span>"
        "</div>"
        f'<div class="gost-grid">{"".join(grid_items)}</div>'
        f'<div class="gost-footer">{_esc(fields["zero_retention_status"])}</div>'
        "</article>"
    )


def render_gost_56545_passport(event: AuditEvent) -> str:
    """Render a standalone offline HTML5 card for a GOST R 56545-2015 incident passport."""
    fields = build_gost_56545_dict(event)
    page_title = f"Паспорт инцидента ГОСТ Р 56545-2015 — {fields['incident_id']}"
    card_html = _render_passport_article(event)
    return (
        "<!DOCTYPE html>"
        '<html lang="ru">'
        "<head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f"<title>{_esc(page_title)}</title>"
        f"<style>{_OFFLINE_CSS}</style>"
        "</head>"
        "<body>"
        '<main class="container">'
        f"{card_html}"
        f'<div class="disclaimer">{_esc(LEGAL_DISCLAIMER)}</div>'
        "</main>"
        "</body>"
        "</html>"
    )


class HtmlComplianceReportGenerator:
    """Self-contained offline HTML5 compliance report generator (ReportGenerator Protocol)."""

    def __init__(self, *, now_fn: Callable[[], datetime] | None = None) -> None:
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def render_bytes(
        self,
        events: Sequence[AuditEvent],
        stats: AuditStats,
        *,
        title: str = "Отчёт Покров (Pokrov)",
    ) -> bytes:
        """Render a self-contained air-gapped HTML5 compliance report as UTF-8 bytes."""
        ctx = build_report_context(events, stats, title=title, generated_at=self._now_fn())

        kpi_cards = "".join(
            f'<div class="kpi-card"><div class="kpi-label">{_esc(k.label)}</div>'
            f'<div class="kpi-value">{_esc(k.value)}</div></div>'
            for k in ctx.kpis
        )
        kpi_table_rows = "".join(
            f'<tr><td>{_esc(k.label)}</td><td class="mono">{_esc(k.key)}</td>'
            f"<td><strong>{_esc(k.value)}</strong></td></tr>"
            for k in ctx.kpis
        )

        if ctx.category_rows:
            cat_table = (
                "<table><thead><tr><th>Категория</th><th>Количество</th>"
                "<th>Нормативные ссылки</th></tr></thead><tbody>"
                + "".join(
                    f"<tr><td>{_esc(row.label_ru)}</td><td>{row.count}</td>"
                    f"<td>{_esc(row.frameworks)}</td></tr>"
                    for row in ctx.category_rows
                )
                + "</tbody></table>"
            )
        else:
            cat_table = '<p class="empty-note">Обнаружения по категориям за период отсутствуют.</p>'

        if ctx.control_point_rows:
            cp_table = (
                "<table><thead><tr><th>Контрольная точка</th>"
                "<th>Количество событий</th></tr></thead><tbody>"
                + "".join(
                    f"<tr><td>{_esc(row.label_ru)}</td><td>{row.count}</td></tr>"
                    for row in ctx.control_point_rows
                )
                + "</tbody></table>"
            )
        else:
            cp_table = '<p class="empty-note">События по контрольным точкам отсутствуют.</p>'

        reg_overview_list = "".join(
            f"<li><strong>{_esc(name)}:</strong> {_esc(desc)}</li>"
            for name, desc in REGULATORY_OVERVIEW_ITEMS
        )
        if ctx.regulatory_rows:
            reg_table = (
                "<table><thead><tr><th>Документ / стандарт</th><th>Статья / пункт</th>"
                "<th>Требование</th><th>Категории срабатываний</th>"
                "<th>Событий</th><th>Риск и назначение меры</th></tr></thead><tbody>"
                + "".join(
                    f"<tr><td><strong>{_esc(r.framework)}</strong></td>"
                    f"<td>{_esc(r.article)}</td>"
                    f"<td>{_esc(r.title)}</td>"
                    f'<td class="mono">{_esc(r.categories)}</td>'
                    f"<td>{r.hits}</td>"
                    f"<td>{_esc(r.risk_hint)}</td></tr>"
                    for r in ctx.regulatory_rows
                )
                + "</tbody></table>"
            )
        else:
            reg_table = (
                '<p class="empty-note">Инциденты с нормативными срабатываниями '
                "за отчётный период не зафиксированы.</p>"
            )

        if events:
            inc_rows_html = []
            for ev in events:
                cats_str = ", ".join(c.value for c in ev.categories) or "—"
                hmacs_str = ", ".join(ev.value_hmacs) or "—"
                sev_cls = _severity_css_class(ev.severity)
                inc_rows_html.append(
                    "<tr>"
                    f'<td class="mono">{_esc(ev.event_id)}</td>'
                    f"<td>{_esc(format_timestamp(ev.timestamp))}</td>"
                    f"<td>{_esc(ev.control_point.value)}</td>"
                    f"<td>{_esc(ev.action.value)}</td>"
                    f'<td><span class="sev {_esc(sev_cls)}">{_esc(ev.severity.value)}</span></td>'
                    f'<td class="mono">{_esc(cats_str)}</td>'
                    f'<td class="mono">{_esc(hmacs_str)}</td>'
                    f"<td>{_esc(ev.reason or '—')}</td>"
                    "</tr>"
                )
            incidents_table = (
                "<table><thead><tr><th>ID события</th><th>Время</th>"
                "<th>Контрольная точка</th><th>Действие</th><th>Критичность</th>"
                "<th>Категории</th><th>HMAC-дайджесты (Zero-Retention)</th>"
                "<th>Основание</th></tr></thead><tbody>"
                + "".join(inc_rows_html)
                + "</tbody></table>"
            )
            passports_html = "".join(_render_passport_article(ev) for ev in events[:100])
            if len(events) > 100:
                passports_html += (
                    f'<p class="empty-note">Отображены первые 100 паспортов '
                    f"из {len(events)} событий.</p>"
                )
        else:
            incidents_table = '<p class="empty-note">События инцидентов отсутствуют.</p>'
            passports_html = '<p class="empty-note">Паспорта инцидентов не сформированы.</p>'

        document = (
            "<!DOCTYPE html>"
            '<html lang="ru">'
            "<head>"
            '<meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            f"<title>{_esc(ctx.title)}</title>"
            f"<style>{_OFFLINE_CSS}</style>"
            "</head>"
            "<body>"
            '<main class="container">'
            '<header class="hero">'
            f"<h1>{_esc(ctx.title)}</h1>"
            '<div class="meta-row">'
            f"<span>Сформирован: {_esc(ctx.generated_at_str)}</span>"
            f"<span>Период: {_esc(ctx.period_str)}</span>"
            f"<span>Событий в выборке: {ctx.events_count}</span>"
            '<span class="badge">Zero-Retention • Offline HTML5</span>'
            "</div>"
            "</header>"
            f'<div class="disclaimer">{_esc(ctx.legal_disclaimer)}</div>'
            '<section class="panel">'
            "<h2>1. Ключевые показатели (KPI)</h2>"
            f'<div class="kpi-grid">{kpi_cards}</div>'
            "<table><thead><tr><th>Метрика</th><th>Код</th><th>Значение</th></tr></thead>"
            f"<tbody>{kpi_table_rows}</tbody></table>"
            "</section>"
            '<section class="panel">'
            "<h2>2. Разбивка по категориям угроз (EntityCategory) и контрольным точкам</h2>"
            "<h3>Категории обнаружений</h3>"
            f"{cat_table}"
            "<h3>Контрольные точки (ControlPoint)</h3>"
            f"{cp_table}"
            "</section>"
            '<section class="panel">'
            "<h2>3. Нормативное покрытие (152-ФЗ, ст. 13.11 КоАП РФ / 420-ФЗ, ФСТЭК № 117)</h2>"
            f"<ul>{reg_overview_list}</ul>"
            f"{reg_table}"
            "</section>"
            '<section class="panel">'
            "<h2>4. Сводка инцидентов (Zero-Retention — только HMAC-дайджесты)</h2>"
            f"{incidents_table}"
            "</section>"
            '<section class="panel">'
            "<h2>5. Паспорта инцидентов по ГОСТ Р 56545-2015</h2>"
            f"{passports_html}"
            "</section>"
            "</main>"
            "</body>"
            "</html>"
        )
        return document.encode("utf-8")


__all__ = [
    "ACTION_LABELS_RU",
    "ACTION_REMEDIATION_RU",
    "CATEGORY_LABELS_RU",
    "CONTROL_POINT_LABELS_RU",
    "REGULATORY_OVERVIEW_ITEMS",
    "SEVERITY_LABELS_RU",
    "CategoryBreakdownRow",
    "ControlPointBreakdownRow",
    "HtmlComplianceReportGenerator",
    "KpiRow",
    "RegulatoryCoverageRow",
    "ReportContext",
    "build_gost_56545_dict",
    "build_report_context",
    "format_category_label",
    "format_control_point_label",
    "format_period",
    "format_timestamp",
    "render_gost_56545_passport",
]
