"""Regulatory and standards compliance mapping for Pokrov.

Maps every ``EntityCategory`` to applicable references in Russian Federal Law
152-FZ (including 23-FZ localization and RKN Order No. 140), Code of
Administrative Offences Art. 13.11 (420-FZ), FSTEC Order No. 117 (11.04.2025) /
Methodological Document sec. 3.18 (12.04.2026), GOST R 59709-59712, and
GOST R 56545-2015.
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
    framework="152-ФЗ (в ред. 23-ФЗ)",
    article="ч. 5 ст. 18 (с 01.07.2025)",
    title="Прямой запрет первичной записи и хранения ПДн граждан РФ в зарубежных БД",
    risk_hint=(
        "Первичная запись в зарубежные БД запрещена; полное исключение трансграничного "
        "контура обеспечивает Data-Aware маршрутизация (route_local) в РФ/on-prem LLM."
    ),
)

_REF_152FZ_CROSS_BORDER = ComplianceRef(
    framework="152-ФЗ",
    article="ст. 12",
    title="Трансграничная передача персональных данных",
    risk_hint=(
        "Обратимая токенизация (псевдонимизация) снижает риск в трафике, но при наличии "
        "косвенных идентификаторов не отменяет уведомление РКН; для исключения ст. 12 "
        "используйте route_local."
    ),
)

_REF_152FZ_TECH_PROTECTION = ComplianceRef(
    framework="152-ФЗ / Приказ РКН № 140",
    article="ст. 19; Приказ РКН № 140 (с 01.09.2025)",
    title="Меры защиты и обезличивания ПДн (метод введения идентификаторов, раздельное хранение)",
    risk_hint=(
        "Псевдонимизация через RAM AES-256-GCM Vault, учёт операций без хранения сырых ПДн "
        "в логах и соблюдение требований Приказа РКН № 140 от 19.06.2025."
    ),
)

_REF_KOAP_13_11_PII = ComplianceRef(
    framework="КоАП РФ (420-ФЗ с 30.05.2025)",
    article="ст. 13.11 ч. 1–2, ч. 11–18",
    title="Ответственность за неправомерную передачу и массовую утечку ПДн",
    risk_hint=(
        "Единичная передача: ч. 1–2 (150–700 тыс. ₽); неуведомление об инциденте: ч. 11 "
        "(1–3 млн ₽); утечки >= 1 000 субъектов / 10 000 ID (RAG/MCP): ч. 12–15 (3–15 млн ₽ "
        "и 1–3% выручки от 20 до 500 млн ₽ повторно); спецкатегории/биометрия: "
        "ч. 16–18 (10–20 млн ₽)."
    ),
)

_REF_KOAP_13_11_LOCALIZATION = ComplianceRef(
    framework="КоАП РФ (420-ФЗ)",
    article="ст. 13.11 ч. 8–9",
    title="Нарушение требований о локализации баз данных граждан РФ",
    risk_hint="Штрафы для юрлиц и ИП от 1–6 млн ₽ (повторно по ч. 9: 6–18 млн ₽).",
)

_REF_FSTEC_117_IO_FILTER = ComplianceRef(
    framework="Приказ ФСТЭК № 117 / МД ФСТЭК",
    article="разд. 3.18 МД от 12.04.2026 (к Приказу № 117 от 11.04.2025)",
    title="Контроль и фильтрация входных запросов и исходящих данных моделей ИИ",
    risk_hint=(
        "Предотвращение утечек конфиденциальной информации и атак prompt injection "
        "(в силе с 01.03.2026)."
    ),
)

_REF_FSTEC_117_AGENTS = ComplianceRef(
    framework="Приказ ФСТЭК № 117 / МД ФСТЭК",
    article="разд. 3.18 МД от 12.04.2026 (автономные агенты ИИ)",
    title="Контроль автономных агентов ИИ и защита от несанкционированного воздействия",
    risk_hint="Ограничение полномочий MCP-инструментов, проверка аргументов и HITL-подтверждение.",
)

_REF_FSTEC_117_SERIALIZATION = ComplianceRef(
    framework="Приказ ФСТЭК № 117 / МД ФСТЭК",
    article="разд. 3.18 МД от 12.04.2026 (отказ от pickle)",
    title=(
        "Изоляция ИИ-сегмента и переход на безопасные форматы моделей (ONNX, protobuf, safetensors)"
    ),
    risk_hint="Выявление pickle/.pt/.pkl моделей и открытых эндпоинтов Ollama/vLLM/RAG.",
)

_REF_FSTEC_117_RSB = ComplianceRef(
    framework="Приказ ФСТЭК № 117 (РСБ)",
    article="РСБ / разд. 3.18 МД от 12.04.2026",
    title="Регистрация событий безопасности и интеграция с сертифицированными SIEM",
    risk_hint="Zero-Retention журнал событий безопасности в форматах CEF/Syslog без сырых ПДн.",
)

_REF_GOST_59709 = ComplianceRef(
    framework="ГОСТ Р 59709–59712",
    article="управление инцидентами ИБ",
    title="Карточка события и компьютерного инцидента ИБ (рантайм-блокировка / РСБ)",
    risk_hint="Стандартизованная фиксация ранних признаков инцидентов и нарушений политик ИИ.",
)

_REF_GOST_56545 = ComplianceRef(
    framework="ГОСТ Р 56545-2015",
    article="разд. 5, Прил. А",
    title="Паспорт конфигурационной уязвимости ПО/ИИ-инфраструктуры (правила описания уязвимостей)",
    risk_hint=(
        "Применяется к статическим находкам уязвимостей конфигураций и форматов моделей "
        "(INFRA_MISCONFIG)."
    ),
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
    _REF_GOST_59709,
)

_INJECTION_REFS: tuple[ComplianceRef, ...] = (
    _REF_FSTEC_117_IO_FILTER,
    _REF_FSTEC_117_AGENTS,
    _REF_FSTEC_117_RSB,
    _REF_GOST_59709,
)

_REF_152FZ_INN_SCOPE = ComplianceRef(
    framework="152-ФЗ / ФНС ЕГРЮЛ",
    article="ст. 3, ст. 12, ст. 19",
    title="ИНН физического лица / ИП (12 цифр = ПДн) и ИНН организации (10 цифр = сведения ЕГРЮЛ)",
    risk_hint=(
        "12-значный ИНН физлица/ИП подлежит защите как ПДн; "
        "10-значный ИНН юрлица относится к реквизитам контрагента."
    ),
)

COMPLIANCE_MAP: Mapping[EntityCategory, tuple[ComplianceRef, ...]] = {
    EntityCategory.RU_PASSPORT: _PII_COMMON_REFS,
    EntityCategory.RU_SNILS: _PII_COMMON_REFS,
    EntityCategory.RU_INN: (
        _REF_152FZ_INN_SCOPE,
        _REF_152FZ_CROSS_BORDER,
        _REF_152FZ_TECH_PROTECTION,
        _REF_KOAP_13_11_PII,
        _REF_FSTEC_117_IO_FILTER,
    ),
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
    EntityCategory.RU_DATE_OF_BIRTH: _PII_COMMON_REFS,
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
        _REF_GOST_59709,
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
    EntityCategory.RU_DATE_OF_BIRTH: Severity.MEDIUM,
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
