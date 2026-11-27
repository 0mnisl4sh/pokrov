"""Comprehensive unit, privacy, accuracy, and performance tests for Russian PII and NER."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import os
import sys
import time
import types

import pytest

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.config import CustomPatternRule
from pokrov.core.contracts import (
    Action,
    BaseDetector,
    BaseModel,
    EntityCategory,
    NormalizedSegment,
    SpanSource,
)
from pokrov.detectors.ner import RuNerDetector
from pokrov.detectors.ru_pii import (
    RuPiiDetector,
    _is_unsafe_regex,
    is_valid_cbr_bik_account,
    is_valid_inn,
    is_valid_luhn,
    is_valid_ogrn,
    is_valid_snils,
)


def _test_hmac(value: str) -> str:
    return hmac.new(b"test-secret-key", value.encode("utf-8"), hashlib.sha256).hexdigest()


def _seg(text: str) -> NormalizedSegment:
    return NormalizedSegment(text=text, span_source=SpanSource.DIRECT)


def _timing_scale() -> float:
    """Allow headroom when coverage instrumentation or shared CI runners add overhead."""
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") or sys.gettrace() is not None:
        return 4.0
    mon = getattr(sys, "monitoring", None)
    if mon is not None and mon.get_tool(mon.COVERAGE_ID) is not None:
        return 4.0
    return 1.0


def test_detectors_satisfy_base_detector_protocol() -> None:
    """Both RuPiiDetector and RuNerDetector must satisfy BaseDetector protocol."""
    pii = RuPiiDetector(hmac_fn=_test_hmac)
    ner = RuNerDetector(hmac_fn=_test_hmac)

    assert isinstance(pii, BaseDetector)
    assert isinstance(ner, BaseDetector)
    assert pii.name == "ru_pii"
    assert pii.layer == 1
    assert ner.name == "ru_ner"
    assert ner.layer == 2
    assert pii.detect(_seg("   "), "   ") == []
    assert ner.detect(_seg(""), "") == []


@pytest.mark.parametrize(
    ("text", "expected_valid"),
    [
        ("СНИЛС: 112-233-445 95", True),
        ("112-233-445 95", True),
        ("087-654-303 00", True),
        ("112-233-458 00", True),  # sum == 100 -> 00
        ("112-233-459 00", True),  # sum == 101 -> 00
        ("11223344595", True),
        ("СНИЛС 112\u00a0233\u00a0445\u00a095", True),
        ("СНИЛС 112-233-445 96", False),  # invalid checksum
        ("СНИЛС 001-001-998 00", False),  # <= 001-001-998 boundary
        ("СНИЛС 000-000-000 00", False),
        ("11223344599", False),
    ],
)
def test_snils_valid_and_invalid_checksums(text: str, expected_valid: bool) -> None:
    """SNILS detection must enforce mod 101 and > 001-001-998 boundary."""
    detector = RuPiiDetector()
    dets = [d for d in detector.detect(_seg(text), text) if d.category == EntityCategory.RU_SNILS]
    assert bool(dets) is expected_valid


def test_unicode_non_ascii_digits_never_crash_validators() -> None:
    """P0-04: Superscripts, circled, fullwidth, and math-bold digits must never raise ValueError."""
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    circled = str.maketrans("0123456789", "⓪①②③④⑤⑥⑦⑧⑨")
    fullwidth = str.maketrans("0123456789", "０１２３４５６７８９")
    math_bold = str.maketrans("0123456789", "𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗")

    for trans in (superscripts, circled, fullwidth, math_bold):
        assert is_valid_snils("11223344595".translate(trans)) is False
        assert is_valid_inn("7707083893".translate(trans)) is False
        assert is_valid_inn("500100732259".translate(trans)) is False
        assert is_valid_ogrn("1027700132195".translate(trans)) is False
        assert is_valid_luhn("2200700712345676".translate(trans)) is False
        assert (
            is_valid_cbr_bik_account(
                "044525225".translate(trans),
                "40702810938000012345".translate(trans),
            )
            is False
        )


def test_snils_helper_and_phone_disambiguation() -> None:
    """11-digit mobile numbers starting with 89/79 must not false-positive as SNILS."""
    assert is_valid_snils("11223344595")
    assert is_valid_snils("08765430300")
    assert not is_valid_snils("00100199800")
    assert not is_valid_snils("11223344596")

    detector = RuPiiDetector()
    phone_text = "Телефон клиента: 89161234567"
    dets = detector.detect(_seg(phone_text), phone_text)
    cats = {d.category for d in dets}
    assert EntityCategory.RU_PHONE in cats
    assert EntityCategory.RU_SNILS not in cats


@pytest.mark.parametrize(
    ("text", "expected_valid", "expected_subtype"),
    [
        ("ИНН 7707083893", True, "org_10"),
        ("7707083893", True, "org_10"),
        ("ИНН 7707083894", False, ""),
        ("ИНН 0007083893", False, ""),
        ("ИНН 500100732259", True, "person_12"),
        ("500100732259", True, "person_12"),
        ("ИНН 500100732258", False, ""),
        ("ИНН 500100732249", False, ""),
    ],
)
def test_inn_10_and_12_checksums_and_subtypes(
    text: str,
    expected_valid: bool,
    expected_subtype: str,
) -> None:
    """10-digit and 12-digit INN must enforce FNS checksum weights and set subtype metadata."""
    assert is_valid_inn("7707083893")
    assert is_valid_inn("500100732259")
    assert not is_valid_inn("7707083894")
    assert not is_valid_inn("500100732258")

    detector = RuPiiDetector()
    dets = [d for d in detector.detect(_seg(text), text) if d.category == EntityCategory.RU_INN]
    assert bool(dets) is expected_valid
    if expected_valid:
        assert dets[0].metadata["subtype"] == expected_subtype


@pytest.mark.parametrize(
    ("text", "expected_valid", "expected_subtype"),
    [
        ("ОГРН 1027700132195", True, "ogrn"),
        ("ОГРН 1027700132240", True, "ogrn"),  # mod 11 == 10 -> 0
        ("1027700132195", True, "ogrn"),
        ("ОГРН 1027700132196", False, ""),  # bad mod 11 % 10
        ("ОГРНИП 304500116000157", True, "ogrnip"),
        ("304500116000157", True, "ogrnip"),
        ("ОГРНИП 304500116000158", False, ""),  # bad mod 13 % 10
        ("КПП 773601001", True, "kpp"),
        ("kpp: 7707AZ001", True, "kpp"),
    ],
)
def test_ogrn_ogrnip_and_kpp(
    text: str,
    expected_valid: bool,
    expected_subtype: str,
) -> None:
    """OGRN (13), OGRNIP (15), and contextual KPP (9) must detect with exact subtypes."""
    assert is_valid_ogrn("1027700132195")
    assert is_valid_ogrn("304500116000157")
    assert not is_valid_ogrn("1027700132196")
    assert not is_valid_ogrn("304500116000158")

    detector = RuPiiDetector()
    dets = [d for d in detector.detect(_seg(text), text) if d.category == EntityCategory.RU_OGRN]
    assert bool(dets) is expected_valid
    if expected_valid:
        assert dets[0].metadata["subtype"] == expected_subtype


def test_bank_cards_brands_amex_collision_sku_filter_and_cvv_expiry() -> None:
    """Bank cards enforce Luhn, detect UnionPay/Amex/Maestro, detect CVV/expiry, and skip SKUs."""
    assert is_valid_luhn("2200700712345676")
    assert is_valid_luhn("4276123456789014")
    assert is_valid_luhn("5469123456789018")
    assert is_valid_luhn("4111111111119")
    assert is_valid_luhn("378282246310005")
    assert not is_valid_luhn("2200700712345675")
    assert not is_valid_luhn("0000000000000000")

    detector = RuPiiDetector()

    # Invalid Luhn rejected (and CVV/expiry must NOT fire when PAN fails Luhn!)
    bad_text = "Карта Мир 2200 7007 1234 5675, срок 12/28, cvv 123"
    assert not [
        d
        for d in detector.detect(_seg(bad_text), bad_text)
        if d.category == EntityCategory.RU_BANK_CARD
    ]

    generic_luhn = "6011000990139424"
    generic_det = [
        d
        for d in detector.detect(_seg(generic_luhn), generic_luhn)
        if d.category == EntityCategory.RU_BANK_CARD
    ]
    assert len(generic_det) == 1

    mir_plain = "2200700712345676"
    mir_plain_det = [
        d
        for d in detector.detect(_seg(mir_plain), mir_plain)
        if d.category == EntityCategory.RU_BANK_CARD
    ]
    assert len(mir_plain_det) == 1
    assert mir_plain_det[0].metadata["brand"] == "mir"
    assert mir_plain_det[0].confidence > generic_det[0].confidence

    # Card with nearby expiry & CVV emits PAN + 2 card_cvv_expiry detections
    mir_ctx = "Карта Мир 2200 7007 1234 5676, срок 08/29, cvv 999"
    mir_ctx_det = [
        d
        for d in detector.detect(_seg(mir_ctx), mir_ctx)
        if d.category == EntityCategory.RU_BANK_CARD
    ]
    pan_dets = [d for d in mir_ctx_det if d.metadata.get("subtype") != "card_cvv_expiry"]
    aux_dets = [d for d in mir_ctx_det if d.metadata.get("subtype") == "card_cvv_expiry"]
    assert len(pan_dets) == 1
    assert pan_dets[0].confidence > mir_plain_det[0].confidence
    assert len(aux_dets) == 2

    # Brands: Visa, MasterCard, UnionPay, Amex, Maestro
    for sample, expected_brand in (
        ("Visa 4276-1234-5678-9014", "visa"),
        ("MasterCard 5469 1234 5678 9018", "mastercard"),
        ("UnionPay 6234567890123451", "unionpay"),
        ("Amex 3782 822463 10005", "amex"),
        ("Maestro 676196000012345675", "maestro"),
    ):
        b_dets = [
            d
            for d in detector.detect(_seg(sample), sample)
            if d.category == EntityCategory.RU_BANK_CARD
        ]
        assert len(b_dets) == 1
        assert b_dets[0].metadata["brand"] == expected_brand

    # P0-08: Unspaced 15-digit Amex PAN that also passes OGRNIP mod-13 MUST still emit RU_BANK_CARD
    amex_ogrnip_collision = '{"field": "378210000000009"}'
    assert is_valid_luhn("378210000000009") and is_valid_ogrn("378210000000009")
    amex_dets = [
        d
        for d in detector.detect(_seg(amex_ogrnip_collision), amex_ogrnip_collision)
        if d.category == EntityCategory.RU_BANK_CARD
    ]
    assert len(amex_dets) == 1
    assert amex_dets[0].metadata["brand"] == "amex"
    assert amex_dets[0].confidence > 0.85

    # SKU / marketplace article negative filter suppresses bare Luhn lookalike
    sku_text = "Артикул товара на маркетплейсе: 2202-2006-1234-5676? Нет, SKU-90817263."
    assert not [
        d
        for d in detector.detect(_seg(sku_text), sku_text)
        if d.category == EntityCategory.RU_BANK_CARD
    ]


def test_bik_account_standalone_cbr_and_ru_iban() -> None:
    """Test BIK+account (emitting BIK span too), standalone CBR accounts, standalone BIK, IBAN."""
    bik = "044525225"
    rs_valid = "40702810938000012345"
    ks_valid = "30101810400000000225"
    rs_invalid = "40702810938000012346"

    assert is_valid_cbr_bik_account(bik, rs_valid)
    assert is_valid_cbr_bik_account(bik, ks_valid)
    assert not is_valid_cbr_bik_account(bik, rs_invalid)
    assert not is_valid_cbr_bik_account("054525225", rs_valid)

    detector = RuPiiDetector()

    # Valid BIK + both checking and correspondent accounts -> 2 accounts + 1 BIK span
    good_text = f"Реквизиты: БИК {bik}, р/с {rs_valid}, к/с {ks_valid}"
    dets = [
        d
        for d in detector.detect(_seg(good_text), good_text)
        if d.category == EntityCategory.RU_BIK_ACCOUNT
    ]
    assert len(dets) == 3
    matched_slices = {good_text[d.start : d.end] for d in dets}
    assert matched_slices == {bik, rs_valid, ks_valid}

    # Account appearing before BIK -> both account and BIK detected
    reversed_text = f"Расчётный счёт {rs_valid} открыт в банке с БИК {bik}"
    rev_dets = [
        d
        for d in detector.detect(_seg(reversed_text), reversed_text)
        if d.category == EntityCategory.RU_BIK_ACCOUNT
    ]
    assert {reversed_text[d.start : d.end] for d in rev_dets} == {rs_valid, bik}

    # Invalid account checksum next to explicit BIK -> account rejected, standalone BIK still caught
    bad_text = f"БИК {bik}, р/с {rs_invalid}"
    bad_dets = [
        d
        for d in detector.detect(_seg(bad_text), bad_text)
        if d.category == EntityCategory.RU_BIK_ACCOUNT
    ]
    assert [bad_text[d.start : d.end] for d in bad_dets] == [bik]

    # Standalone valid CBR 20-digit account with banking context + standalone BIK + RU IBAN
    standalone_rs = f"Расчётный счёт {rs_valid}"
    assert len(detector.detect(_seg(standalone_rs), standalone_rs)) == 1

    standalone_bik = f"Банковский БИК {bik}"
    assert len(detector.detect(_seg(standalone_bik), standalone_bik)) == 1

    iban_text = "Счёт получателя RU45FOMI8350305641395 в реестре"
    iban_dets = [
        d
        for d in detector.detect(_seg(iban_text), iban_text)
        if d.category == EntityCategory.RU_BIK_ACCOUNT
    ]
    assert len(iban_dets) == 1
    assert iban_dets[0].metadata["subtype"] == "iban"

    # Bare 20-digit non-context number does not false-positive
    bare_noise = "40702810100000000001"
    assert detector.detect(_seg(bare_noise), bare_noise) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Паспорт РФ: 4512 654321", True),
        ("Данные: 45 12 654321, выдан ОВД", True),
        ("паспорт 4512654321", True),
        ("серия 45 12 номер 654321", True),
        ("Паспорт серия 4512\nномер 654321", True),
        ("Паспорт 4512/654321", True),
        ("Паспорт 4512№654321", True),
        ("4512–654321", True),
        ('{"passport": "4512654321"}', True),
        ('{"doc_num": "4512654321"}', True),
        (
            "Паспорт гражданина Российской Федерации, выданный отделом УФМС России "
            "по гор. Москве и Московской области в 2020 году: 4512654321",
            True,
        ),
        ("Загранпаспорт РФ 75 № 1234567", True),
        ("4512 654321", True),
        ("Заказ № 1234567890", False),
        ("Заказ № 4512 654321", False),
        ("Номер партии оборудования: 45 12 654321 (складской учёт заказа).", False),
        ("Order 4512654321", False),
        ("Дата документа 01.02.2025 и 2025-11-23", False),
        ("Паспорт 0012 654321", False),
        ("Паспорт 4555 123456", False),
    ],
)
def test_passport_detection_and_false_positive_suppression(text: str, expected: bool) -> None:
    """Passport detector catches internal/foreign formats and suppresses order/batch IDs."""
    detector = RuPiiDetector()
    dets = [
        d for d in detector.detect(_seg(text), text) if d.category == EntityCategory.RU_PASSPORT
    ]
    assert bool(dets) is expected


def test_phones_emails_dob_ipv4_ipv6_hex_boundaries_and_custom_redos() -> None:
    """Test expanded phones, trailing-dot/IDN emails, DOB, IPv6, IPv4 FP filters, hex bounds."""
    custom_rules = [
        CustomPatternRule(
            name="internal_contract",
            pattern=r"ДОГ-\d{4}-\d{5}",
            category=EntityCategory.CUSTOM_PATTERN,
            action=Action.MASK,
            confidence=0.93,
        ),
        CustomPatternRule(name="broken_regex", pattern=r"[unclosed("),
        CustomPatternRule(name="redos_nested", pattern=r"(a+)+b"),
        CustomPatternRule(name="redos_alt", pattern=r"(?:a|a)+$"),
        CustomPatternRule(name="redos_adj", pattern=r"^\d+\d+\d+\d+X$"),
    ]
    detector = RuPiiDetector(custom_patterns=custom_rules, hmac_fn=_test_hmac)

    # Verify _is_unsafe_regex catches all exponential/polynomial ReDoS forms
    assert _is_unsafe_regex(r"(a+)+b") is True
    assert _is_unsafe_regex(r"(?:a|a)+$") is True
    assert _is_unsafe_regex(r"(?:a|aa)+$") is True
    assert _is_unsafe_regex(r"^(?:a|aa){22}$") is True
    assert _is_unsafe_regex(r"^\d+\d+\d+\d+X$") is True
    assert _is_unsafe_regex(r".*.*") is True
    assert _is_unsafe_regex(r"ДОГ-\d{4}-\d{5}") is False

    # Expanded Phones (bare 7, KZ +7 7xx, dots, CIS +375/+998, 10-digit)
    for phone_sample in (
        "+7 (916) 123-45-67",
        "8 (495) 123-45-67",
        "79991234567",
        "7 (999) 123-45-67",
        "8.999.123.45.67",
        "+7 (701) 123-45-67",
        "+375 (29) 123-45-67",
        "+998 90 123-45-67",
        "(999) 123-45-67",
        "Звоните 9991234567",
    ):
        p_dets = [
            d
            for d in detector.detect(_seg(phone_sample), phone_sample)
            if d.category == EntityCategory.RU_PHONE
        ]
        assert len(p_dets) == 1, f"Failed phone: {phone_sample}"

    # P0-02: Sentence-ending period emails & Cyrillic .рф / IDN emails
    email_cases = [
        ("Отправьте отчёт на roman31@example.com.", "roman31@example.com"),
        ("Пишите на ivan.petrov+work@company.ru.", "ivan.petrov+work@company.ru"),
        ("Адрес: иван.петров@почта.рф.", "иван.петров@почта.рф"),
        ("Портал info@госуслуги.рф работает.", "info@госуслуги.рф"),
        ("Punycode ivan@xn--80aswg.xn--p1ai!", "ivan@xn--80aswg.xn--p1ai"),
    ]
    for sample, expected_email in email_cases:
        e_dets = [
            d for d in detector.detect(_seg(sample), sample) if d.category == EntityCategory.EMAIL
        ]
        assert len(e_dets) == 1, f"Failed email in {sample!r}"
        assert sample[e_dets[0].start : e_dets[0].end] == expected_email

    # Date of birth
    for dob_sample in (
        "Дата рождения: 15.04.1988",
        "Клиент (д.р. 19/03/1976) обратился в офис",
        '{"dob": "1992-11-30"}',
    ):
        dob_dets = [
            d
            for d in detector.detect(_seg(dob_sample), dob_sample)
            if d.category == EntityCategory.RU_DATE_OF_BIRTH
        ]
        assert len(dob_dets) == 1

    # IPv4 & IPv6 valid vs SemVer / loopback / RFC 5737 false positives
    ip_good = "Хост 192.168.10.25 и IPv6 2a02:6b8:a::a, fe80::1ff:fe23:4567:890a"
    ip_dets = [
        d
        for d in detector.detect(_seg(ip_good), ip_good)
        if d.category == EntityCategory.IP_ADDRESS
    ]
    assert {ip_good[d.start : d.end] for d in ip_dets} == {
        "192.168.10.25",
        "2a02:6b8:a::a",
        "fe80::1ff:fe23:4567:890a",
    }

    for benign_ip_like in (
        "Обновите клиент до версии 1.2.3.4 перед запуском миграции.",
        "Используется драйвер PostgreSQL ODBC 13.2.0.1 в контейнере.",
        "Пакет openssl 3.0.13.1 исправляет ошибку.",
        "Версия прошивки маршрутизатора v2.14.8.12.",
        "Требуется .NET Runtime 8.0.10.2 или выше.",
        "Пункт 4.2.1.3 регламента утверждён 15.03.2026.",
        "Запустите сервер на 127.0.0.1:8080 или host = 0.0.0.0, маска 255.255.255.0.",
        "Пример документации RFC 5737: используйте 192.0.2.1 или 198.51.100.1.",
    ):
        assert not [
            d
            for d in detector.detect(_seg(benign_ip_like), benign_ip_like)
            if d.category == EntityCategory.IP_ADDRESS
        ], f"False positive IP on {benign_ip_like!r}"

    # P1-05: Hex / container / Sentry / Span IDs must NEVER partially match numeric PII
    for hex_text in (
        "Container ID: 89991234567a9f01b2c3d4e5.",
        "Sentry event_id: 7707083893abcdef1234567890abcdef.",
        "Span ID: 2202200612345676ff.",
        "Request ID: 550e8400-e29b-41d4-a716-446655440000.",
    ):
        assert detector.detect(_seg(hex_text), hex_text) == [], f"Corrupted hex ID: {hex_text!r}"

    # Custom pattern carries rule_name and rule_action metadata
    custom_text = "Подписан договор ДОГ-2026-00142"
    cust_dets = [
        d
        for d in detector.detect(_seg(custom_text), custom_text)
        if d.category == EntityCategory.CUSTOM_PATTERN
    ]
    assert len(cust_dets) == 1
    assert cust_dets[0].metadata == {"rule_name": "internal_contract", "rule_action": "mask"}


def test_ner_fallback_names_oblique_ethnic_caps_lowercase_ip_and_addresses() -> None:
    """RuNerDetector fallback extracts Slavic/ethnic/oblique/CAPS FIO, ИП FIO, and addresses."""
    detector = RuNerDetector(hmac_fn=_test_hmac, use_natasha=False)
    text = (
        "Клиент Иванов Иван Иванович, представитель И. О. Сидоров и Петрова А. С., "
        "а также Мария Соколова проживают по адресу г. Москва, ул. Тверская, д. 12, кв. 34 "
        "и 191186, г. Санкт-Петербург, Невский пр., д. 28. "
        'Контрагенты: ООО «Ромашка», АО "Техно-Плюс" и ИП Ковалёв Д. Н. '
        "Ложные цели: Московский Государственный Университет, Новый Год, Нижний Новгород, "
        "постановка оперы Борис Годунов, ул. А. С. Пушкина, д. 10."
    )
    dets = detector.detect(_seg(text), text)

    names = [text[d.start : d.end] for d in dets if d.category == EntityCategory.RU_PERSON_NAME]
    addrs = [text[d.start : d.end] for d in dets if d.category == EntityCategory.RU_ADDRESS]
    orgs = [text[d.start : d.end] for d in dets if d.category == EntityCategory.RU_ORG_NAME]

    assert "Иванов Иван Иванович" in names
    assert "И. О. Сидоров" in names
    assert "Петрова А. С." in names
    assert "Мария Соколова" in names
    assert "Ковалёв Д. Н." in names  # P0-08: citizen FIO extracted from ИП!
    assert "Московский Государственный Университет" not in names
    assert "Новый Год" not in names
    assert "Нижний Новгород" not in names
    assert "Борис Годунов" not in names
    assert "А. С. Пушкина" not in names

    assert "г. Москва, ул. Тверская, д. 12, кв. 34" in addrs
    assert "191186, г. Санкт-Петербург, Невский пр., д. 28" in addrs
    assert "ул. А. С. Пушкина, д. 10" in addrs

    assert "ООО «Ромашка»" in orgs
    assert 'АО "Техно-Плюс"' in orgs
    assert "ИП Ковалёв Д. Н." in orgs

    # Oblique 2-word stems, non-Slavic names, initials, ALL CAPS, contextual lowercase
    for fio_sample in (
        "Заявление Сергея Смирнова",
        "Выдать Сергею Смирнову",
        "Подписано Сергеем Смирновым",
        "Согласовано Алексеем Поповым",
        "Проверено Дмитрием Соколовым",
        "Заявление Павла Морозова",
        "Книга Льва Толстого",
        "Подписано Любовью Орловой",
        "Клиент Рустам Хабибуллин",
        "Клиент Гульназ Хабибуллина",
        "Клиент Магомед Гаджиев",
        "Клиент Мамедов Рашид Оглы",
        "Клиент Абдуллаев Тимур Рустамович угли",
        "Клиент Ким Александр",
        "Клиент Берия Лаврентий",
        "Подписал И.И.Иванов",
        "Подписал И. Иванов",
        "Клиент ИВАНОВ ИВАН ИВАНОВИЧ",
        "клиент: иванов иван иванович",
        "Получатель ИП Петров Иван",
    ):
        f_dets = [
            d
            for d in detector.detect(_seg(fio_sample), fio_sample)
            if d.category == EntityCategory.RU_PERSON_NAME
        ]
        assert f_dets, f"Missed FIO in {fio_sample!r}"

    # Expanded address formats
    for addr_sample in (
        "г. Екатеринбург, пер. Банковский, д. 5",
        "г. москва, ул. ленина, д. 10, кв. 5",
        "г. Москва, ул. Тверская, 12, кв. 45",
        "ул. Кирпичная, д. 131, г. Токсово",
        "Московская обл., г. Химки, мкр. Сходня, д. 5, корп. 2",
        "Москва, Тверская 12, кв 45",
        "г. Казань, алл. Славы, д. 7, индекс 420000",
        "Почтовый индекс 473829",
    ):
        a_dets = [
            d
            for d in detector.detect(_seg(addr_sample), addr_sample)
            if d.category == EntityCategory.RU_ADDRESS
        ]
        assert a_dets, f"Missed address in {addr_sample!r}"

    # Legal citations with 'ст. 18' must NOT trigger RU_ADDRESS
    legal = "Согласно ч. 5 ст. 18 и ст. 13.11 КоАП РФ ответственность усилена."
    assert not [
        d for d in detector.detect(_seg(legal), legal) if d.category == EntityCategory.RU_ADDRESS
    ]


def test_ner_100kb_near_miss_address_latency_under_30ms() -> None:
    """100 KB near-miss address,prose, and 12,800-IP log must run well within budget."""
    detector = RuNerDetector(use_natasha=False)
    pii = RuPiiDetector()
    detector.detect(_seg("г. Москва, ул. Тверская, д. 1"), "г. Москва, ул. Тверская, д. 1")
    pii.detect(_seg("1.1.1.1"), "1.1.1.1")

    near_miss_100kb = ("г. Москва, ул. Промышленная Линия Корпусная Северная, " * 1_850)[:100_000]
    seg = _seg(near_miss_100kb)

    t0 = time.perf_counter()
    dets = detector.detect(seg, near_miss_100kb)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    scale = _timing_scale()
    ner_limit = 30.0 * scale
    assert dets == []
    assert elapsed_ms < ner_limit, (
        f"100 KB NER took {elapsed_ms:.2f} ms (limit: {ner_limit:.1f} ms)"
    )

    ip_log_100kb = ("1.1.1.1 " * 12_800)[:100_000]
    seg_ip = _seg(ip_log_100kb)
    t1 = time.perf_counter()
    ip_dets = pii.detect(seg_ip, ip_log_100kb)
    ip_elapsed_ms = (time.perf_counter() - t1) * 1000.0
    ip_limit = 60.0 * scale
    assert len(ip_dets) >= 12_400
    assert ip_elapsed_ms < ip_limit, (
        f"100 KB IPv4 log took {ip_elapsed_ms:.2f} ms (limit: {ip_limit:.1f} ms)"
    )


def test_ner_optional_natasha_lazy_load_and_mocked_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RuNerDetector must work without natasha and integrate cleanly if natasha is available."""
    detector_no_natasha = RuNerDetector(hmac_fn=_test_hmac)
    text = "Договор с Ивановым Петром Сергеевичем, ООО «Альфа»"
    dets = detector_no_natasha.detect(_seg(text), text)
    assert any(d.category == EntityCategory.RU_PERSON_NAME for d in dets)
    assert any(d.category == EntityCategory.RU_ORG_NAME for d in dets)

    class FakeSpan:
        def __init__(self, start: int, stop: int, span_type: str) -> None:
            self.start = start
            self.stop = stop
            self.type = span_type

    class FakeDoc:
        def __init__(self, doc_text: str) -> None:
            self.text = doc_text
            self.spans: list[FakeSpan] = []

        def segment(self, _seg_obj: object) -> None:
            return None

        def tag_ner(self, _tagger: object) -> None:
            idx = self.text.find("Смирнов")
            if idx != -1:
                self.spans = [FakeSpan(idx, idx + len("Смирнов"), "PER")]

    fake_natasha = types.SimpleNamespace(
        Doc=FakeDoc,
        Segmenter=lambda: object(),
        NewsEmbedding=lambda: object(),
        NewsNERTagger=lambda _emb: object(),
    )

    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "natasha":
            return fake_natasha
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    detector_with_natasha = RuNerDetector(hmac_fn=_test_hmac)
    sample = "Смирнов подписал приказ"
    nat_dets = detector_with_natasha.detect(_seg(sample), sample)
    assert len(nat_dets) == 1
    assert nat_dets[0].category == EntityCategory.RU_PERSON_NAME
    assert nat_dets[0].metadata["ner_backend"] == "natasha"
    assert sys.modules.get("natasha") is None


def test_evasion_span_backmapping_direct_and_decoded_blob() -> None:
    """Span back-mapping and evasion flags must propagate for DIRECT and DECODED_BLOB."""
    pii = RuPiiDetector(hmac_fn=_test_hmac)
    ner = RuNerDetector(hmac_fn=_test_hmac, use_natasha=False)

    raw_text = "СНИЛС: 112-\u200b233-445 95"
    norm_text = "СНИЛС: 112-233-445 95"
    index_map = tuple(i if i < 11 else i + 1 for i in range(len(norm_text)))
    seg_direct = NormalizedSegment(
        text=norm_text,
        span_source=SpanSource.DIRECT,
        index_map=index_map,
        evasion_techniques=("zero_width",),
    )
    dets = pii.detect(seg_direct, raw_text)
    snils_dets = [d for d in dets if d.category == EntityCategory.RU_SNILS]
    assert len(snils_dets) == 1
    assert raw_text[snils_dets[0].start : snils_dets[0].end] == "112-\u200b233-445 95"
    assert snils_dets[0].evasion_detected is True
    assert snils_dets[0].evasion_techniques == ("zero_width",)
    assert snils_dets[0].span_source == SpanSource.DIRECT

    seg_blob = NormalizedSegment(
        text="Иванов И. И., 112-233-445 95",
        span_source=SpanSource.DECODED_BLOB,
        outer_start=12,
        outer_end=56,
        evasion_techniques=("base64",),
    )
    blob_pii = pii.detect(seg_blob, "prefix data <base64_payload_here> suffix")
    blob_ner = ner.detect(seg_blob, "prefix data <base64_payload_here> suffix")
    assert len(blob_pii) == 1
    assert (blob_pii[0].start, blob_pii[0].end) == (12, 56)
    assert blob_pii[0].span_source == SpanSource.DECODED_BLOB
    assert blob_pii[0].evasion_detected is True
    assert len(blob_ner) == 1
    assert (blob_ner[0].start, blob_ner[0].end) == (12, 56)
    assert blob_ner[0].span_source == SpanSource.DECODED_BLOB


def test_zero_retention_json_scan_across_all_categories() -> None:
    """No raw PII character or value may ever appear in Detection.model_dump_json()."""
    secrets = [
        "112-233-445 95",
        "7707083893",
        "1027700132195",
        "2200 7007 1234 5676",
        "044525225",
        "40702810938000012345",
        "4512 654321",
        "7700001234567890",
        "+7 (916) 123-45-67",
        "К456МР199",
        "77 12 345678",
        "15.04.1988",
        "secret.user@corp.ru",
        "10.20.30.40",
        "ДОГ-2026-99887",
        "Петров Иван Сергеевич",
        "г. Москва, ул. Тверская, д. 12, кв. 34",
        "ООО «Секретный Проект»",
    ]
    text = (
        "Сотрудник Петров Иван Сергеевич (д.р. 15.04.1988), паспорт 4512 654321, "
        "СНИЛС 112-233-445 95, ИНН 7707083893, ОГРН 1027700132195, "
        "карта Мир 2200 7007 1234 5676, БИК 044525225, р/с 40702810938000012345, "
        "полис ОМС 7700001234567890, тел. +7 (916) 123-45-67, авто К456МР199, "
        "ВУ 77 12 345678, email secret.user@corp.ru, ip 10.20.30.40, "
        "контракт ДОГ-2026-99887, адрес г. Москва, ул. Тверская, д. 12, кв. 34, "
        "компания ООО «Секретный Проект»."
    )
    pii = RuPiiDetector(
        custom_patterns=[CustomPatternRule(name="contract", pattern=r"ДОГ-\d{4}-\d{5}")],
        hmac_fn=_test_hmac,
    )
    ner = RuNerDetector(hmac_fn=_test_hmac, use_natasha=False)
    seg = _seg(text)

    all_dets: list[BaseModel] = [*pii.detect(seg, text), *ner.detect(seg, text)]
    assert len(all_dets) >= len(secrets)

    for det_obj in pii.detect(seg, text) + ner.detect(seg, text):
        dumped = det_obj.model_dump_json()
        raw_slice = text[det_obj.start : det_obj.end]
        assert det_obj.masked_preview == f"[{det_obj.category.value}:len={len(raw_slice)}]"
        assert det_obj.value_hmac == _test_hmac(raw_slice)
        assert det_obj.compliance_refs == get_compliance_refs(det_obj.category)
        for sensitive in secrets:
            assert sensitive not in dumped, f"Raw PII leaked into Detection JSON: {sensitive}"
        assert raw_slice not in dumped


def test_speed_under_15ms_on_2000_char_russian_text() -> None:
    """Detection on 2000-char Russian text must run well below 15 ms."""
    sample_block = (
        "Клиент Иванов Иван Иванович (паспорт 4512 654321, СНИЛС 112-233-445 95, "
        "ИНН 7707083893, карта Мир 2200 7007 1234 5676, тел. +7 (916) 123-45-67, "
        "email client@example.ru, адрес г. Москва, ул. Тверская, д. 12, кв. 34, "
        "организация ООО «Ромашка») отправил запрос в информационную систему. "
    )
    text = (sample_block * 10)[:2050]
    assert len(text) >= 2000

    pii = RuPiiDetector(
        custom_patterns=[CustomPatternRule(name="contract", pattern=r"ДОГ-\d{4}-\d{5}")],
        hmac_fn=_test_hmac,
    )
    ner = RuNerDetector(hmac_fn=_test_hmac, use_natasha=False)
    seg = _seg(text)

    pii.detect(seg, text)
    ner.detect(seg, text)

    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        pii_dets = pii.detect(seg, text)
        ner_dets = ner.detect(seg, text)
    elapsed_ms = ((time.perf_counter() - start) / iterations) * 1000.0

    limit_ms = 15.0 * _timing_scale()
    assert len(pii_dets) > 0
    assert len(ner_dets) > 0
    assert elapsed_ms < limit_ms, f"Detection took {elapsed_ms:.2f} ms (limit: {limit_ms:.1f} ms)"


def test_adversarial_redos_and_no_hmac_default() -> None:
    """Adversarial ReDoS custom patterns and long inputs must complete rapidly without leaks."""
    pii = RuPiiDetector(
        custom_patterns=[
            CustomPatternRule(name="evil1", pattern=r"(a+)+b"),
            CustomPatternRule(name="evil2", pattern=r"(?:a|a)+$"),
            CustomPatternRule(name="evil3", pattern=r"^\d+\d+\d+\d+X$"),
        ],
        hmac_fn=None,
    )
    adversarial = ("a" * 30_000) + "!" + " 112-233-445 95"
    start = time.perf_counter()
    dets = pii.detect(_seg(adversarial), adversarial)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 50.0 * _timing_scale()
    assert len(dets) == 1
    assert dets[0].category == EntityCategory.RU_SNILS
    assert dets[0].value_hmac == ""
    assert "112-233-445 95" not in dets[0].model_dump_json()


def test_ru_pii_additional_edge_branches_for_audit_coverage() -> None:
    """Cover IPv6, standalone CBR accounts, foreign passport, CVV/expiry, and negative contexts."""
    pii = RuPiiDetector()

    # IPv6 real vs loopback/unspecified
    v6_text = "Server ::1 and 2001:db8:85a3::8a2e:370:7334 in production"
    v6_dets = pii.detect(_seg(v6_text), v6_text)
    ips = [v6_text[d.start : d.end] for d in v6_dets if d.category == EntityCategory.IP_ADDRESS]
    assert "2001:db8:85a3::8a2e:370:7334" in ips
    assert "::1" not in ips

    # Standalone 20-digit CBR account + standalone BIK + RU IBAN + foreign passport + KPP
    combo = (
        "БИК 044525225, к/с 30101810400000000225, расчётный счёт 40702810900000012345, "
        "IBAN RU0204452522540702810900000012345, КПП 773601001, "
        "загранпаспорт 75 № 1234567, карта 2202 2006 1234 5676 срок 05/28 CVV 749, "
        "дата рождения 15.04.1988, ОГРНИП 304500116000157."
    )
    dets = pii.detect(_seg(combo), combo)
    cats = {d.category for d in dets}
    assert {
        EntityCategory.RU_BIK_ACCOUNT,
        EntityCategory.RU_OGRN,
        EntityCategory.RU_PASSPORT,
        EntityCategory.RU_BANK_CARD,
        EntityCategory.RU_DATE_OF_BIRTH,
    } <= cats

    # Negative guards: loopback/RFC5737 doc/semver IPs and order numbers must stay quiet
    clean = "v2.14.8.12 релиз 1.2.3.4 документация 192.0.2.15 loopback 127.0.0.1 заказ 4512 654321"
    clean_dets = pii.detect(_seg(clean), clean)
    assert clean_dets == []


def test_ru_pii_helpers_regex_guard_and_validators() -> None:
    """Cover _is_unsafe_regex, _is_valid_email, checksum validators, and RFC5737 IPv4 edges."""
    from pokrov.detectors.ru_pii import _detect_card_brand, _is_valid_email

    # _is_unsafe_regex edge cases
    assert _is_unsafe_regex("") is True
    assert _is_unsafe_regex("a" * 2001) is True
    assert _is_unsafe_regex(r"(a+)+b") is True
    assert _is_unsafe_regex(r"(?:a|aa)+") is True
    assert _is_unsafe_regex(r"(a|aa){15}") is True
    assert _is_unsafe_regex(r"(a|b|)+") is True
    assert _is_unsafe_regex(r"(a+?)+?") is True
    assert _is_unsafe_regex(r"([a-z]+)+") is True
    assert _is_unsafe_regex(r"(\d+)+") is True
    assert _is_unsafe_regex(r"((a+)(b))+") is True
    assert _is_unsafe_regex(r"(a+){15}") is True
    assert _is_unsafe_regex(r"(a\|b|c\[d\]|(e|f))+") is True
    assert _is_unsafe_regex(r"^(\bfoo|\bfor)+$") is True
    assert _is_unsafe_regex(r"^([a-z]|[0-9])$") is False
    assert _is_unsafe_regex(r"ДОГ-\d{4}-\d{5}") is False
    assert _is_unsafe_regex(r"(?:foo|bar)[a|b]\d+") is False

    # _is_valid_email negatives & punycode
    assert _is_valid_email("not-an-email") is False
    assert _is_valid_email("user@localhost") is False
    assert _is_valid_email("user@sub.-bad.ru") is False
    assert _is_valid_email("user@domain.123") is False
    assert _is_valid_email("user@site.xn--p1ai") is True

    # Checksum invalid length / non-digit cases
    assert is_valid_snils("123") is False
    assert is_valid_inn("123456789") is False
    assert is_valid_ogrn("12345") is False
    assert is_valid_luhn("12") is False
    assert is_valid_cbr_bik_account("044525225", "123") is False

    # _detect_card_brand coverage (Mir, Visa, MC, Amex, UnionPay, Maestro, Discover, JCB, unknown)
    assert _detect_card_brand("2200700123456789") == "mir"
    assert _detect_card_brand("4111111111111111") == "visa"
    assert _detect_card_brand("5500000000000004") == "mastercard"
    assert _detect_card_brand("378282246310005") == "amex"
    assert _detect_card_brand("6234567890123451") == "unionpay"
    assert _detect_card_brand("676196000012345675") == "maestro"
    assert _detect_card_brand("6011000000000004") is None
    assert _detect_card_brand("not-digits") is None

    # RFC 5737 doc IP + SemVer prefix/trailing context
    pii = RuPiiDetector()
    doc_ex = "Например, IP 192.0.2.15 из RFC 5737 в документации"
    assert pii.detect(_seg(doc_ex), doc_ex) == []
    real_rfc = "Адрес 192.0.2.15"
    assert len(pii.detect(_seg(real_rfc), real_rfc)) == 1
    assert pii.detect(_seg("v1.2.3.4"), "v1.2.3.4") == []
    assert pii.detect(_seg("10.20.30.40 build version"), "10.20.30.40 build version") == []
    assert len(pii.detect(_seg("foo 10.20.30.40"), "foo 10.20.30.40")) == 1
    hmac_pii = RuPiiDetector(hmac_fn=_test_hmac)
    assert len(hmac_pii.detect(_seg("foo 10.20.30.40 v1.2.3.4"), "foo 10.20.30.40 v1.2.3.4")) == 1
