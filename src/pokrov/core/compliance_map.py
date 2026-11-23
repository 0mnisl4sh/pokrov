"""Regulatory and standards compliance mapping for Pokrov.

Maps every ``EntityCategory`` to applicable references in Russian Federal Law
152-FZ, Code of Administrative Offences Art. 13.11 (420-FZ), FSTEC Order
No. 117 / methodological guidance sec. 3.18 (12.04.2026), and GOST R 56545-2015.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pokrov.core.contracts import ComplianceRef, EntityCategory, Severity

LEGAL_DISCLAIMER = (
    "Покров (Pokrov) является инструментом технической поддержки и автоматизации мер "
    "защиты информации при использовании технологий ИИ. Продукт не является "
    "сертифицированным СЗИ и не заменяет аттестацию информационной системы; "
    "нормативные привязки носят справочный характер и подлежат верификации по модели "
    "угроз и политике безопасности организации."
)

# Reusable reference definitions
_REF_152FZ_LOCALIZATION = ComplianceRef(
    framework="152-ФЗ",
    article="ч. 5 ст. 18",
    title="Локализация первичной обработки и баз персональных данных граждан РФ",
    risk_hint="Запрет неправомерной передачи ПДн на зарубежные серверы LLM без локализации.",
)

_REF_152FZ_CROSS_BORDER = ComplianceRef(
    framework="152-ФЗ",
    article="ст. 12",
    title="Трансграничная передача персональных данных",
    risk_hint="Требуется уведомление РКН и оценка достаточности мер защиты принимающей стороны.",
)

_REF_152FZ_TECH_PROTECTION = ComplianceRef(
    framework="152-ФЗ",
    article="ст. 19",
    title="Меры по обеспечению безопасности ПДн при обработке и обезличивании",
    risk_hint="Обязательное маскирование, шифрование и исключение хранения сырых ПДн в логах.",
)

_REF_KOAP_13_11_PII = ComplianceRef(
    framework="КоАП РФ (420-ФЗ)",
    article="ст. 13.11 ч. 12–16",
    title="Оборотные штрафы за утечку ПДн и неправомерную обработку",
    risk_hint="Штрафы для юрлиц от 3–15 млн ₽ до 1–3% годовой выручки при повторных утечках.",
)

_REF_KOAP_13_11_LOCALIZATION = ComplianceRef(
    framework="КоАП РФ (420-ФЗ)",
    article="ст. 13.11 ч. 8–9",
    title="Нарушение требований о локализации баз данных граждан РФ",
    risk_hint="Штрафы для юрлиц от 1–6 млн ₽ (повторно до 18 млн ₽).",
)

_REF_FSTEC_117_IO_FILTER = ComplianceRef(
    framework="ФСТЭК № 117 (п. 3.18)",
    article="п. 3.18 (от 12.04.2026)",
    title="Фильтрация входных запросов и исходящих ответов моделей ИИ",
    risk_hint="Предотвращение утечек конфиденциальной информации и атак prompt injection.",
)

_REF_FSTEC_117_AGENTS = ComplianceRef(
    framework="ФСТЭК № 117 (п. 3.18)",
    article="п. 3.18 (агенты ИИ)",
    title="Контроль автономных и иных агентов ИИ и защита от несанкционированного воздействия",
    risk_hint="Ограничение полномочий MCP-инструментов, проверка аргументов и HITL-подтверждение.",
)

_REF_FSTEC_117_SERIALIZATION = ComplianceRef(
    framework="ФСТЭК № 117 (п. 3.18)",
    article="п. 3.18 (инфраструктура)",
    title="Изоляция ИИ-сегмента и отказ от небезопасных форматов сериализации моделей",
    risk_hint="Выявление pickle/.pt/.pkl моделей и открытых эндпоинтов Ollama/vLLM/RAG.",
)

_REF_FSTEC_117_RSB = ComplianceRef(
    framework="ФСТЭК № 117 (РСБ)",
    article="РСБ / п. 3.18",
    title="Регистрация событий безопасности и интеграция с SIEM",
    risk_hint="Zero-Retention журнал событий безопасности в форматах CEF/Syslog без сырых ПДн.",
)

_REF_GOST_56545 = ComplianceRef(
    framework="ГОСТ Р 56545-2015",
    article="разд. 5",
    title="Структура паспорта инцидента и конфигурационной уязвимости",
    risk_hint="Стандартизованная карточка инцидента ИБ для отчётности и расследований.",
)

_PII_COMMON_REFS: tuple[ComplianceRef, ...] = (
    _REF_152FZ_LOCALIZATION,
    _REF_152FZ_CROSS_BORDER,
    _REF_152FZ_TECH_PROTECTION,
    _REF_KOAP_13_11_PII,
    _REF_KOAP_13_11_LOCALIZATION,
    _REF_FSTEC_117_IO_FILTER,
)

_CONTACT_PII_REFS: tuple[ComplianceRef, ...] = (
    _REF_152FZ_CROSS_BORDER,
    _REF_152FZ_TECH_PROTECTION,
    _REF_KOAP_13_11_PII,
    _REF_FSTEC_117_IO_FILTER,
)

_SECRET_REFS: tuple[ComplianceRef, ...] = (
    _REF_152FZ_TECH_PROTECTION,
    _REF_FSTEC_117_IO_FILTER,
    _REF_FSTEC_117_RSB,
    _REF_GOST_56545,
)

_INJECTION_REFS: tuple[ComplianceRef, ...] = (
    _REF_FSTEC_117_IO_FILTER,
    _REF_FSTEC_117_AGENTS,
    _REF_FSTEC_117_RSB,
    _REF_GOST_56545,
)

COMPLIANCE_MAP: Mapping[EntityCategory, tuple[ComplianceRef, ...]] = {
    EntityCategory.RU_PASSPORT: _PII_COMMON_REFS,
    EntityCategory.RU_SNILS: _PII_COMMON_REFS,
    EntityCategory.RU_INN: _PII_COMMON_REFS,
    EntityCategory.RU_OGRN: (
        _REF_152FZ_TECH_PROTECTION,
        _REF_FSTEC_117_IO_FILTER,
    ),
    EntityCategory.RU_BANK_CARD: _PII_COMMON_REFS,
    EntityCategory.RU_BIK_ACCOUNT: _PII_COMMON_REFS,
    EntityCategory.RU_OMS: _PII_COMMON_REFS,
    EntityCategory.RU_PHONE: _PII_COMMON_REFS,
    EntityCategory.RU_AUTO_PLATE: _CONTACT_PII_REFS,
    EntityCategory.RU_DRIVER_LICENSE: _PII_COMMON_REFS,
    EntityCategory.RU_PERSON_NAME: _PII_COMMON_REFS,
    EntityCategory.RU_ADDRESS: _PII_COMMON_REFS,
    EntityCategory.RU_ORG_NAME: (
        _REF_152FZ_TECH_PROTECTION,
        _REF_FSTEC_117_IO_FILTER,
    ),
    EntityCategory.EMAIL: _CONTACT_PII_REFS,
    EntityCategory.IP_ADDRESS: (
        _REF_152FZ_TECH_PROTECTION,
        _REF_FSTEC_117_IO_FILTER,
        _REF_FSTEC_117_RSB,
    ),
    EntityCategory.SECRET_KEY: _SECRET_REFS,
    EntityCategory.CUSTOM_PATTERN: (
        _REF_152FZ_TECH_PROTECTION,
        _REF_FSTEC_117_IO_FILTER,
        _REF_FSTEC_117_RSB,
    ),
    EntityCategory.PROMPT_INJECTION: _INJECTION_REFS,
    EntityCategory.INDIRECT_INJECTION: _INJECTION_REFS,
    EntityCategory.MARKDOWN_EXFIL: _INJECTION_REFS,
    EntityCategory.TOXIC_OR_LEAK: (
        _REF_FSTEC_117_IO_FILTER,
        _REF_FSTEC_117_RSB,
    ),
    EntityCategory.MCP_POLICY_VIOLATION: (
        _REF_FSTEC_117_AGENTS,
        _REF_FSTEC_117_RSB,
        _REF_GOST_56545,
    ),
    EntityCategory.INFRA_MISCONFIG: (
        _REF_FSTEC_117_SERIALIZATION,
        _REF_FSTEC_117_RSB,
        _REF_GOST_56545,
    ),
}

CATEGORY_DEFAULT_SEVERITY: Mapping[EntityCategory, Severity] = {
    EntityCategory.RU_PASSPORT: Severity.CRITICAL,
    EntityCategory.RU_SNILS: Severity.CRITICAL,
    EntityCategory.RU_INN: Severity.HIGH,
    EntityCategory.RU_OGRN: Severity.LOW,
    EntityCategory.RU_BANK_CARD: Severity.CRITICAL,
    EntityCategory.RU_BIK_ACCOUNT: Severity.HIGH,
    EntityCategory.RU_OMS: Severity.HIGH,
    EntityCategory.RU_PHONE: Severity.MEDIUM,
    EntityCategory.RU_AUTO_PLATE: Severity.MEDIUM,
    EntityCategory.RU_DRIVER_LICENSE: Severity.HIGH,
    EntityCategory.RU_PERSON_NAME: Severity.MEDIUM,
    EntityCategory.RU_ADDRESS: Severity.MEDIUM,
    EntityCategory.RU_ORG_NAME: Severity.LOW,
    EntityCategory.EMAIL: Severity.MEDIUM,
    EntityCategory.IP_ADDRESS: Severity.LOW,
    EntityCategory.SECRET_KEY: Severity.CRITICAL,
    EntityCategory.CUSTOM_PATTERN: Severity.HIGH,
    EntityCategory.PROMPT_INJECTION: Severity.CRITICAL,
    EntityCategory.INDIRECT_INJECTION: Severity.CRITICAL,
    EntityCategory.MARKDOWN_EXFIL: Severity.CRITICAL,
    EntityCategory.TOXIC_OR_LEAK: Severity.MEDIUM,
    EntityCategory.MCP_POLICY_VIOLATION: Severity.CRITICAL,
    EntityCategory.INFRA_MISCONFIG: Severity.HIGH,
}


def get_compliance_refs(category: EntityCategory) -> tuple[ComplianceRef, ...]:
    """Return regulatory citations associated with a detection category."""
    return COMPLIANCE_MAP[category]


def aggregate_compliance_refs(
    categories: Iterable[EntityCategory],
) -> tuple[ComplianceRef, ...]:
    """Return deduplicated compliance references preserving first-seen order."""
    seen: set[tuple[str, str]] = set()
    ordered: list[ComplianceRef] = []
    for category in categories:
        for ref in COMPLIANCE_MAP[category]:
            key = (ref.framework, ref.article)
            if key not in seen:
                seen.add(key)
                ordered.append(ref)
    return tuple(ordered)


def get_category_severity(category: EntityCategory) -> Severity:
    """Return default risk severity for an entity category."""
    return CATEGORY_DEFAULT_SEVERITY[category]


__all__ = [
    "CATEGORY_DEFAULT_SEVERITY",
    "COMPLIANCE_MAP",
    "LEGAL_DISCLAIMER",
    "aggregate_compliance_refs",
    "get_category_severity",
    "get_compliance_refs",
]
