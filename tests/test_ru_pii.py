"""Comprehensive unit, privacy, accuracy, and performance tests for Russian PII and NER."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import sys
import time
import types

import pytest

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.config import CustomPatternRule
from pokrov.core.contracts import (
    Action,
    BaseDetector,
    EntityCategory,
    NormalizedSegment,
    SpanSource,
)
from pokrov.detectors.ner import RuNerDetector
from pokrov.detectors.ru_pii import (
    RuPiiDetector,
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


def test_snils_helper_and_phone_disambiguation() -> None:
    """11-digit mobile numbers starting with 89 must not false-positive as SNILS."""
    assert is_valid_snils("11223344595")
    assert is_valid_snils("08765430300")
    assert not is_valid_snils("00100199800")
    assert not is_valid_snils("11223344596")

    detector = RuPiiDetector()
    # 11-digit mobile numbers starting with 89 must never be tagged as SNILS without context
    phone_text = "Телефон клиента: 89161234567"
    dets = detector.detect(_seg(phone_text), phone_text)
    cats = {d.category for d in dets}
    assert EntityCategory.RU_PHONE in cats
    assert EntityCategory.RU_SNILS not in cats


@pytest.mark.parametrize(
    ("text", "expected_valid"),
    [
        ("ИНН 7707083893", True),  # 10-digit org (Sberbank)
        ("7707083893", True),
        ("ИНН 7707083894", False),  # bad 10-digit checksum
        ("ИНН 0007083893", False),  # 00 region invalid
        ("ИНН 500100732259", True),  # 12-digit individual
        ("500100732259", True),
        ("ИНН 500100732258", False),  # bad n12
        ("ИНН 500100732249", False),  # bad n11
    ],
)
def test_inn_10_and_12_checksums(text: str, expected_valid: bool) -> None:
    """10-digit and 12-digit INN must enforce FNS checksum weights."""
    assert is_valid_inn("7707083893")
    assert is_valid_inn("500100732259")
    assert not is_valid_inn("7707083894")
    assert not is_valid_inn("500100732258")

    detector = RuPiiDetector()
    dets = [d for d in detector.detect(_seg(text), text) if d.category == EntityCategory.RU_INN]
    assert bool(dets) is expected_valid


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
    ],
)
def test_ogrn_and_ogrnip_checksums(
    text: str,
    expected_valid: bool,
    expected_subtype: str,
) -> None:
    """OGRN (13) and OGRNIP (15) must enforce mod 11 % 10 and mod 13 % 10."""
    assert is_valid_ogrn("1027700132195")
    assert is_valid_ogrn("304500116000157")
    assert not is_valid_ogrn("1027700132196")
    assert not is_valid_ogrn("304500116000158")

    detector = RuPiiDetector()
    dets = [d for d in detector.detect(_seg(text), text) if d.category == EntityCategory.RU_OGRN]
    assert bool(dets) is expected_valid
    if expected_valid:
        assert dets[0].metadata["subtype"] == expected_subtype


def test_bank_cards_luhn_brands_and_confidence_boost() -> None:
    """Bank cards must enforce Luhn and boost confidence for Mir/Visa/MC + context."""
    assert is_valid_luhn("2200700712345676")
    assert is_valid_luhn("4276123456789014")
    assert is_valid_luhn("5469123456789018")
    assert is_valid_luhn("4111111111119")
    assert is_valid_luhn("378282246310005")
    assert not is_valid_luhn("2200700712345675")
    assert not is_valid_luhn("0000000000000000")

    detector = RuPiiDetector()

    # Invalid Luhn rejected
    bad_text = "Карта Мир 2200 7007 1234 5675, срок 12/28"
    assert not [
        d
        for d in detector.detect(_seg(bad_text), bad_text)
        if d.category == EntityCategory.RU_BANK_CARD
    ]

    # Bare Luhn vs brand + context boost
    generic_luhn = "6011000990139424"  # Discover valid Luhn, no brand/context
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

    mir_ctx = "Карта Мир 2200 7007 1234 5676, срок 08/29, cvv 999"
    mir_ctx_det = [
        d
        for d in detector.detect(_seg(mir_ctx), mir_ctx)
        if d.category == EntityCategory.RU_BANK_CARD
    ]
    assert len(mir_ctx_det) == 1
    assert mir_ctx_det[0].confidence > mir_plain_det[0].confidence

    visa_text = "Visa 4276-1234-5678-9014"
    visa_det = [
        d
        for d in detector.detect(_seg(visa_text), visa_text)
        if d.category == EntityCategory.RU_BANK_CARD
    ]
    assert len(visa_det) == 1
    assert visa_det[0].metadata["brand"] == "visa"

    mc_text = "MasterCard 5469 1234 5678 9018"
    mc_det = [
        d
        for d in detector.detect(_seg(mc_text), mc_text)
        if d.category == EntityCategory.RU_BANK_CARD
    ]
    assert len(mc_det) == 1
    assert mc_det[0].metadata["brand"] == "mastercard"


def test_bik_and_account_cbr_checksum() -> None:
    """9-digit BIK (04...) + 20-digit checking/corr account must satisfy CBR mod 10."""
    bik = "044525225"
    rs_valid = "40702810938000012345"
    ks_valid = "30101810400000000225"
    rs_invalid = "40702810938000012346"

    assert is_valid_cbr_bik_account(bik, rs_valid)
    assert is_valid_cbr_bik_account(bik, ks_valid)
    assert not is_valid_cbr_bik_account(bik, rs_invalid)
    assert not is_valid_cbr_bik_account("054525225", rs_valid)

    detector = RuPiiDetector()

    # Valid both checking and correspondent accounts
    good_text = f"Реквизиты: БИК {bik}, р/с {rs_valid}, к/с {ks_valid}"
    dets = [
        d
        for d in detector.detect(_seg(good_text), good_text)
        if d.category == EntityCategory.RU_BIK_ACCOUNT
    ]
    assert len(dets) == 2

    # Account appearing before BIK
    reversed_text = f"Расчётный счёт {rs_valid} открыт в банке с БИК {bik}"
    rev_dets = [
        d
        for d in detector.detect(_seg(reversed_text), reversed_text)
        if d.category == EntityCategory.RU_BIK_ACCOUNT
    ]
    assert len(rev_dets) == 1

    # Invalid account checksum or missing BIK -> no RU_BIK_ACCOUNT
    bad_text = f"БИК {bik}, р/с {rs_invalid}"
    assert not [
        d
        for d in detector.detect(_seg(bad_text), bad_text)
        if d.category == EntityCategory.RU_BIK_ACCOUNT
    ]
    no_bik = f"Счёт {rs_valid} без БИК"
    assert not [
        d
        for d in detector.detect(_seg(no_bik), no_bik)
        if d.category == EntityCategory.RU_BIK_ACCOUNT
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Паспорт РФ: 4512 654321", True),
        ("Данные: 45 12 654321, выдан ОВД", True),
        ("паспорт 4512654321", True),
        ("серия 45 12 номер 654321", True),
        ("4512 654321", True),
        ("Заказ № 1234567890", False),  # order ID FP suppression
        ("Заказ № 4512 654321", False),  # order context FP suppression
        ("Order 4512654321", False),  # plain 10-digit order ID
        ("Дата документа 01.02.2025 и 2025-11-23", False),  # dates
        ("Паспорт 0012 654321", False),  # invalid region 00
        ("Паспорт 4555 123456", False),  # invalid year 55
    ],
)
def test_passport_detection_and_false_positive_suppression(text: str, expected: bool) -> None:
    """Passport detector must catch valid passports and suppress order IDs, dates, bad series."""
    detector = RuPiiDetector()
    dets = [
        d for d in detector.detect(_seg(text), text) if d.category == EntityCategory.RU_PASSPORT
    ]
    assert bool(dets) is expected


def test_oms_phone_auto_plate_driver_license_email_ip_custom() -> None:
    """Test OMS, phones, auto plates, driver license, email, IPv4, and custom patterns."""
    custom_rules = [
        CustomPatternRule(
            name="internal_contract",
            pattern=r"ДОГ-\d{4}-\d{5}",
            category=EntityCategory.CUSTOM_PATTERN,
            action=Action.MASK,
            confidence=0.93,
        ),
        # Unsafe/invalid patterns must be skipped cleanly without crashing or hanging
        CustomPatternRule(name="broken_regex", pattern=r"[unclosed("),
        CustomPatternRule(name="redos_regex", pattern=r"(a+)+b"),
    ]
    detector = RuPiiDetector(custom_patterns=custom_rules, hmac_fn=_test_hmac)

    # OMS
    oms_good = "Полис ОМС: 7700001234567890"
    oms_dets = [
        d for d in detector.detect(_seg(oms_good), oms_good) if d.category == EntityCategory.RU_OMS
    ]
    assert len(oms_dets) == 1

    oms_no_ctx = "Идентификатор транзакции: 7700001234567890"
    assert not [
        d
        for d in detector.detect(_seg(oms_no_ctx), oms_no_ctx)
        if d.category == EntityCategory.RU_OMS
    ]

    # Phones (+7 / 8 mobile & landline)
    for phone in (
        "+7 (916) 123-45-67",
        "8 (495) 123-45-67",
        "+79161234567",
        "8-812-234-56-78",
        "8 903 555 11 22",
    ):
        p_dets = [
            d for d in detector.detect(_seg(phone), phone) if d.category == EntityCategory.RU_PHONE
        ]
        assert len(p_dets) == 1, f"Failed phone: {phone}"

    # Auto plates
    for plate in ("А123ВС77", "К 456 МР 199", "A777BC777", "в001ор178"):
        a_dets = [
            d
            for d in detector.detect(_seg(plate), plate)
            if d.category == EntityCategory.RU_AUTO_PLATE
        ]
        assert len(a_dets) == 1, f"Failed plate: {plate}"

    for non_plate in ("SHA256BC12", "А000АА77", "X123AB4567"):
        assert not [
            d
            for d in detector.detect(_seg(non_plate), non_plate)
            if d.category == EntityCategory.RU_AUTO_PLATE
        ]

    # Driver license (with context, no false-positive as passport)
    for dl_text in (
        "Водительское удостоверение: 77 12 345678",
        "ВУ 77 АЕ 123456",
        "права 5012 654321",
    ):
        dl_dets = detector.detect(_seg(dl_text), dl_text)
        assert any(d.category == EntityCategory.RU_DRIVER_LICENSE for d in dl_dets)
        assert not any(d.category == EntityCategory.RU_PASSPORT for d in dl_dets)

    dl_no_ctx = "Номер документа 77 АЕ 123456"
    assert not [
        d
        for d in detector.detect(_seg(dl_no_ctx), dl_no_ctx)
        if d.category == EntityCategory.RU_DRIVER_LICENSE
    ]

    # Email & IPv4
    contact_text = (
        "Пишите на ivan.petrov+work@company.ru, хост 192.168.10.25, не 999.999.1.1 и не 1.2.3.4.5"
    )
    c_dets = detector.detect(_seg(contact_text), contact_text)
    emails = [d for d in c_dets if d.category == EntityCategory.EMAIL]
    ips = [d for d in c_dets if d.category == EntityCategory.IP_ADDRESS]
    assert len(emails) == 1
    assert contact_text[emails[0].start : emails[0].end] == "ivan.petrov+work@company.ru"
    assert len(ips) == 1
    assert contact_text[ips[0].start : ips[0].end] == "192.168.10.25"

    # Custom pattern + skipped ReDoS safety
    custom_text = "Подписан договор ДОГ-2026-00142 и строка " + ("a" * 30) + "!"
    cust_dets = [
        d
        for d in detector.detect(_seg(custom_text), custom_text)
        if d.category == EntityCategory.CUSTOM_PATTERN
    ]
    assert len(cust_dets) == 1
    assert cust_dets[0].metadata["rule_name"] == "internal_contract"
    assert cust_dets[0].confidence == 0.93


def test_ner_fallback_names_addresses_and_orgs() -> None:
    """RuNerDetector pure-Python fallback must extract FIO, addresses, and organizations."""
    detector = RuNerDetector(hmac_fn=_test_hmac, use_natasha=False)
    text = (
        "Клиент Иванов Иван Иванович, представитель И. О. Сидоров и Петрова А. С., "
        "а также Мария Соколова проживают по адресу г. Москва, ул. Тверская, д. 12, кв. 34 "
        "и 191186, г. Санкт-Петербург, Невский пр., д. 28. "
        'Контрагенты: ООО «Ромашка», АО "Техно-Плюс" и ИП Ковалёв Д. Н. '
        "Ложные цели: Московский Государственный Университет, Новый Год, Нижний Новгород, "
        "ул. А. С. Пушкина, д. 10."
    )
    dets = detector.detect(_seg(text), text)

    names = [text[d.start : d.end] for d in dets if d.category == EntityCategory.RU_PERSON_NAME]
    addrs = [text[d.start : d.end] for d in dets if d.category == EntityCategory.RU_ADDRESS]
    orgs = [text[d.start : d.end] for d in dets if d.category == EntityCategory.RU_ORG_NAME]

    assert "Иванов Иван Иванович" in names
    assert "И. О. Сидоров" in names
    assert "Петрова А. С." in names
    assert "Мария Соколова" in names
    assert "Московский Государственный Университет" not in names
    assert "Новый Год" not in names
    assert "Нижний Новгород" not in names
    assert "А. С. Пушкина" not in names

    assert "г. Москва, ул. Тверская, д. 12, кв. 34" in addrs
    assert "191186, г. Санкт-Петербург, Невский пр., д. 28" in addrs
    assert "ул. А. С. Пушкина, д. 10" in addrs

    lane_text = "Офис находится по адресу г. Екатеринбург, пер. Банковский, д. 5"
    lane_dets = detector.detect(_seg(lane_text), lane_text)
    assert any(
        d.category == EntityCategory.RU_ADDRESS
        and "пер. Банковский, д. 5" in lane_text[d.start : d.end]
        for d in lane_dets
    )

    assert "ООО «Ромашка»" in orgs
    assert 'АО "Техно-Плюс"' in orgs
    assert "ИП Ковалёв Д. Н." in orgs


def test_ner_optional_natasha_lazy_load_and_mocked_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RuNerDetector must work without natasha and integrate cleanly if natasha is available."""
    # 1. Without natasha installed (or import failure) -> graceful fallback
    detector_no_natasha = RuNerDetector(hmac_fn=_test_hmac)
    text = "Договор с Ивановым Петром Сергеевичем, ООО «Альфа»"
    dets = detector_no_natasha.detect(_seg(text), text)
    assert any(d.category == EntityCategory.RU_PERSON_NAME for d in dets)
    assert any(d.category == EntityCategory.RU_ORG_NAME for d in dets)

    # 2. Mocked natasha module
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

    # Reset mocked import before finishing
    assert sys.modules.get("natasha") is None


def test_evasion_span_backmapping_direct_and_decoded_blob() -> None:
    """Span back-mapping and evasion flags must propagate for DIRECT and DECODED_BLOB."""
    pii = RuPiiDetector(hmac_fn=_test_hmac)
    ner = RuNerDetector(hmac_fn=_test_hmac, use_natasha=False)

    # Zero-width char inside SNILS
    raw_text = "СНИЛС: 112-\u200b233-445 95"
    norm_text = "СНИЛС: 112-233-445 95"
    # Map every char in norm_text back to raw_text (skipping index 11 = '\u200b')
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

    # DECODED_BLOB segment mapping to outer_start..outer_end
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
        "40702810938000012345",
        "4512 654321",
        "7700001234567890",
        "+7 (916) 123-45-67",
        "К456МР199",
        "77 12 345678",
        "secret.user@corp.ru",
        "10.20.30.40",
        "ДОГ-2026-99887",
        "Петров Иван Сергеевич",
        "г. Москва, ул. Тверская, д. 12, кв. 34",
        "ООО «Секретный Проект»",
    ]
    text = (
        "Сотрудник Петров Иван Сергеевич, паспорт 4512 654321, СНИЛС 112-233-445 95, "
        "ИНН 7707083893, ОГРН 1027700132195, карта Мир 2200 7007 1234 5676, "
        "БИК 044525225, р/с 40702810938000012345, полис ОМС 7700001234567890, "
        "тел. +7 (916) 123-45-67, авто К456МР199, ВУ 77 12 345678, "
        "email secret.user@corp.ru, ip 10.20.30.40, контракт ДОГ-2026-99887, "
        "адрес г. Москва, ул. Тверская, д. 12, кв. 34, компания ООО «Секретный Проект»."
    )
    pii = RuPiiDetector(
        custom_patterns=[CustomPatternRule(name="contract", pattern=r"ДОГ-\d{4}-\d{5}")],
        hmac_fn=_test_hmac,
    )
    ner = RuNerDetector(hmac_fn=_test_hmac, use_natasha=False)
    seg = _seg(text)

    all_dets = pii.detect(seg, text) + ner.detect(seg, text)
    assert len(all_dets) >= len(secrets)

    for det in all_dets:
        dumped = det.model_dump_json()
        raw_slice = text[det.start : det.end]
        assert det.masked_preview == f"[{det.category.value}:len={len(raw_slice)}]"
        assert det.value_hmac == _test_hmac(raw_slice)
        assert det.compliance_refs == get_compliance_refs(det.category)
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

    # Warm-up run
    pii.detect(seg, text)
    ner.detect(seg, text)

    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        pii_dets = pii.detect(seg, text)
        ner_dets = ner.detect(seg, text)
    elapsed_ms = ((time.perf_counter() - start) / iterations) * 1000.0

    assert len(pii_dets) > 0
    assert len(ner_dets) > 0
    assert elapsed_ms < 15.0, f"Detection took {elapsed_ms:.2f} ms (limit: 15.0 ms)"


def test_adversarial_redos_and_no_hmac_default() -> None:
    """Adversarial ReDoS custom patterns and long inputs must complete rapidly without leaks."""
    pii = RuPiiDetector(
        custom_patterns=[CustomPatternRule(name="evil", pattern=r"(a+)+b")],
        hmac_fn=None,
    )
    adversarial = ("a" * 30_000) + "!" + " 112-233-445 95"
    start = time.perf_counter()
    dets = pii.detect(_seg(adversarial), adversarial)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 50.0
    assert len(dets) == 1
    assert dets[0].category == EntityCategory.RU_SNILS
    assert dets[0].value_hmac == ""
    assert "112-233-445 95" not in dets[0].model_dump_json()
