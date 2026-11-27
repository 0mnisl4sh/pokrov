"""Deterministic Russian PII, financial identifier, contact, and custom regex detectors."""

from __future__ import annotations

import ipaddress
import logging
import re
from collections.abc import Callable, Sequence

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.config import CustomPatternRule
from pokrov.core.contracts import (
    ComplianceRef,
    Detection,
    EntityCategory,
    NormalizedSegment,
    SpanSource,
)

logger = logging.getLogger(__name__)

_COMPLIANCE_CACHE: dict[EntityCategory, tuple[ComplianceRef, ...]] = {
    cat: get_compliance_refs(cat) for cat in EntityCategory
}
_DETECTION_FIELDS_SET: frozenset[str] = frozenset(Detection.model_fields)
_FIELDS_SET_TEMPLATE: set[str] = set(Detection.model_fields)
_HAS_2_DIGITS_RE = re.compile(r"\d{2}")
_HAS_ALPHA_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")

# ---------------------------------------------------------------------------
# Pre-compiled linear-time patterns & boundaries
# ---------------------------------------------------------------------------

_ALNUM_BOUND_L = r"(?<![0-9A-Za-zА-Яа-яЁё_])"
_ALNUM_BOUND_R = r"(?![0-9A-Za-zА-Яа-яЁё_])"
_NUM_SEP = r"[ \u00a0\u202f\u2009.\-–—]"
_PHONE_SEP = r"[ \t\u00a0\u202f\u2009.\-–—]"
_DOC_SEP = r"[ \u00a0\u202f\u2009.\-/–—]"

_SNILS_RE = re.compile(
    rf"{_ALNUM_BOUND_L}"
    rf"(?:\d{{3}}{_NUM_SEP}\d{{3}}{_NUM_SEP}\d{{3}}{_NUM_SEP}\d{{2}}|\d{{11}})"
    rf"{_ALNUM_BOUND_R}",
)

_INN_RE = re.compile(rf"{_ALNUM_BOUND_L}(?:\d{{12}}|\d{{10}}){_ALNUM_BOUND_R}")

_OGRN_RE = re.compile(rf"{_ALNUM_BOUND_L}(?:\d{{15}}|\d{{13}}){_ALNUM_BOUND_R}")

_KPP_RE = re.compile(rf"{_ALNUM_BOUND_L}(?!00)\d{{4}}[0-9A-Z]{{2}}\d{{3}}{_ALNUM_BOUND_R}")

_BANK_CARD_RE = re.compile(
    rf"{_ALNUM_BOUND_L}(?:"
    rf"\d{{4}}{_NUM_SEP}\d{{4}}{_NUM_SEP}\d{{4}}{_NUM_SEP}\d{{4}}(?:{_NUM_SEP}\d{{1,3}})?"
    rf"|\d{{4}}{_NUM_SEP}\d{{6}}{_NUM_SEP}\d{{5}}"
    rf"|\d{{13,19}}"
    rf"){_ALNUM_BOUND_R}",
)

_CARD_CVV_RE = re.compile(
    rf"(?i){_ALNUM_BOUND_L}"
    r"(?:cvv2?|cvc2?|код\s+безопасности|защитный\s+код)[ \t:=№#\-]*\d{3,4}"
    rf"{_ALNUM_BOUND_R}",
)

_CARD_EXPIRY_RE = re.compile(
    rf"(?i){_ALNUM_BOUND_L}"
    r"(?:срок(?:\s+действия)?|действует\s+до|valid\s+thru|exp(?:iry|ires)?)"
    r"[ \t:=]*(?:0[1-9]|1[0-2])/[0-9]{2}(?:[0-9]{2})?"
    r"(?![0-9A-Za-zА-Яа-яЁё_/])",
)

_BIK_RE = re.compile(rf"{_ALNUM_BOUND_L}04\d{{7}}{_ALNUM_BOUND_R}")

_ACCOUNT_RE = re.compile(
    rf"{_ALNUM_BOUND_L}"
    r"(?:\d{20}|"
    r"\d{5}[. \u00a0\u202f]\d{3}[. \u00a0\u202f]\d[. \u00a0\u202f]\d{4}[. \u00a0\u202f]\d{7})"
    rf"{_ALNUM_BOUND_R}",
)

_IBAN_RU_RE = re.compile(
    rf"{_ALNUM_BOUND_L}RU\d{{2}}[A-Z0-9]{{4}}\d{{6,29}}{_ALNUM_BOUND_R}",
)

_PASSPORT_CONTIGUOUS_RE = re.compile(
    rf"{_ALNUM_BOUND_L}"
    rf"(?:\d{{2}}{_DOC_SEP}\d{{2}}{_DOC_SEP}\d{{6}}|\d{{4}}{_DOC_SEP}\d{{6}}|\d{{10}})"
    rf"{_ALNUM_BOUND_R}",
)

_PASSPORT_LABELED_RE = re.compile(
    rf"{_ALNUM_BOUND_L}"
    r"(\d{2}[ \u00a0\u202f\u2009\-–—]?\d{2})"
    r"(?:[\s,;]+(?:номер|№)[ \t:.\-–—]*|[ \t]*№[ \t:.\-–—]*)"
    r"(\d{6})"
    rf"{_ALNUM_BOUND_R}",
    re.IGNORECASE,
)

_FOREIGN_PASSPORT_RE = re.compile(
    rf"{_ALNUM_BOUND_L}\d{{2}}[ \u00a0\u202f\u2009№:\-/–—]*\d{{7}}{_ALNUM_BOUND_R}",
)

_OMS_RE = re.compile(
    rf"{_ALNUM_BOUND_L}"
    rf"(?:\d{{16}}|\d{{4}}{_NUM_SEP}\d{{4}}{_NUM_SEP}\d{{4}}{_NUM_SEP}\d{{4}})"
    rf"{_ALNUM_BOUND_R}",
)

_PHONE_RE = re.compile(
    rf"{_ALNUM_BOUND_L}"
    r"(?:\+7|8|7)"
    r"(?:"
    rf"{_PHONE_SEP}?(?:\([34789]\d{{2}}\)|[34789]\d{{2}}){_PHONE_SEP}?\d{{3}}"
    rf"(?:{_PHONE_SEP}?\d{{2}}{_PHONE_SEP}?\d{{2}}|{_PHONE_SEP}?\d{{4}})"
    rf"|{_PHONE_SEP}?(?:\([348]\d{{3}}\)|[348]\d{{3}}){_PHONE_SEP}?\d{{2}}{_PHONE_SEP}?\d{{2}}{_PHONE_SEP}?\d{{2}}"
    rf"|{_PHONE_SEP}?(?:\([348]\d{{4}}\)|[348]\d{{4}}){_PHONE_SEP}?\d{_PHONE_SEP}?\d{{2}}{_PHONE_SEP}?\d{{2}}"
    r")"
    rf"{_ALNUM_BOUND_R}",
)

_PHONE_CIS_RE = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё_+])"
    r"\+(?:375|998|374|996|992|994)"
    rf"{_PHONE_SEP}?(?:\(\d{{2,3}}\)|\d{{2,3}})"
    rf"{_PHONE_SEP}?\d{{3}}{_PHONE_SEP}?\d{{2}}{_PHONE_SEP}?\d{{2}}"
    rf"{_ALNUM_BOUND_R}",
)

_PHONE_10_RE = re.compile(
    rf"{_ALNUM_BOUND_L}"
    r"(?:"
    rf"\([34789]\d{{2}}\){_PHONE_SEP}?\d{{3}}(?:{_PHONE_SEP}?\d{{2}}{_PHONE_SEP}?\d{{2}}|{_PHONE_SEP}?\d{{4}})"
    rf"|[34789]\d{{2}}{_PHONE_SEP}?\d{{3}}(?:{_PHONE_SEP}?\d{{2}}{_PHONE_SEP}?\d{{2}}|{_PHONE_SEP}?\d{{4}})"
    r")"
    rf"{_ALNUM_BOUND_R}",
)

_PLATE_LETTERS = "АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx"
_PLATE_SEP = r"[ \u00a0\u202f\u2009\-–—]"
_AUTO_PLATE_TYPE1_RE = re.compile(
    rf"{_ALNUM_BOUND_L}"
    rf"[{_PLATE_LETTERS}]{_PLATE_SEP}?(?!000)\d{{3}}{_PLATE_SEP}?[{_PLATE_LETTERS}]{{2}}"
    rf"{_PLATE_SEP}?(?:[1279]\d{{2}}|(?!00)\d{{2}})"
    rf"(?:{_PLATE_SEP}?(?:RUS|rus))?"
    rf"{_ALNUM_BOUND_R}",
)
_AUTO_PLATE_ALT_RE = re.compile(
    rf"{_ALNUM_BOUND_L}"
    rf"(?:[{_PLATE_LETTERS}]{{2}}{_PLATE_SEP}?\d{{3,4}}|\d{{4}}{_PLATE_SEP}?[{_PLATE_LETTERS}]{{2}})"
    rf"{_PLATE_SEP}?(?:[1279]\d{{2}}|(?!00)\d{{2}})"
    rf"(?:{_PLATE_SEP}?(?:RUS|rus))?"
    rf"{_ALNUM_BOUND_R}",
)

_DRIVER_LICENSE_RE = re.compile(
    rf"{_ALNUM_BOUND_L}(?:"
    rf"\d{{2}}{_PLATE_SEP}?\d{{2}}{_PLATE_SEP}?\d{{6}}"
    rf"|\d{{2}}{_PLATE_SEP}?[{_PLATE_LETTERS}]{{2}}{_PLATE_SEP}?\d{{6}}"
    rf"){_ALNUM_BOUND_R}",
)

_EMAIL_LOCAL_CHAR = r"[A-Za-z0-9А-Яа-яЁё_%+-]"
_EMAIL_DOM_CHAR = r"[A-Za-z0-9А-Яа-яЁё-]"
_EMAIL_RE = re.compile(
    rf"(?<![A-Za-z0-9А-Яа-яЁё._%+-])"
    rf"{_EMAIL_LOCAL_CHAR}{{1,64}}(?:\.{_EMAIL_LOCAL_CHAR}{{1,64}}){{0,8}}"
    rf"@"
    rf"{_EMAIL_DOM_CHAR}{{1,63}}(?:\.{_EMAIL_DOM_CHAR}{{1,63}})+"
    rf"(?![A-Za-z0-9А-Яа-яЁё_%+-]|\.{_EMAIL_DOM_CHAR})",
)

_DOB_RE = re.compile(
    rf"{_ALNUM_BOUND_L}(?:"
    r"(?:0[1-9]|[12]\d|3[01])[./-](?:0[1-9]|1[0-2])[./-](?:19\d{2}|20[0-2]\d)"
    r"|(?:19\d{2}|20[0-2]\d)[./-](?:0[1-9]|1[0-2])[./-](?:0[1-9]|[12]\d|3[01])"
    rf"){_ALNUM_BOUND_R}",
)

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4_RE = re.compile(
    rf"(?<![0-9A-Za-zА-Яа-яЁё_.]){_OCTET}(?:\.{_OCTET}){{3}}(?![0-9A-Za-zА-Яа-яЁё_])(?!\.\d)",
)
_IPV4_SENSITIVE_FAST_RE = re.compile(
    rf"(?<![0-9_.])(?!(?:0|127|255)\.|(?:192\.0\.2|198\.51\.100|203\.0\.113)\.)"
    rf"{_OCTET}(?:\.{_OCTET}){{3}}(?![0-9_])(?!\.\d)",
)
_IPV4_PREVIEWS: dict[int, str] = {n: f"[ip_address:len={n}]" for n in range(7, 16)}
_IPV4_METADATA: dict[str, str] = {"subtype": "ipv4"}

_IPV6_CANDIDATE_RE = re.compile(
    r"(?<![0-9A-Za-z:])(?:"
    r"(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}"
    r"|(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{1,4}){0,5}"
    r"|::(?:[0-9A-Fa-f]{1,4}:){0,5}[0-9A-Fa-f]{1,4}"
    r")(?![0-9A-Za-z:])",
)

_ASCII_NON_DIGITS_RE = re.compile(r"[^0-9]+")

# ---------------------------------------------------------------------------
# Context keywords (all lowercase; checked within bounded local windows)
# ---------------------------------------------------------------------------

_SNILS_POS_RE = re.compile(r"(?<![а-яёa-z])(?:снилс|snils|пфр|сфр|страховой\s+номер)(?![а-яёa-z])")
_INN_POS_RE = re.compile(r"(?<![а-яёa-z])(?:инн|inn|фнс|егрюл|егрип|реквизит\w*)(?![а-яёa-z])")
_OGRN_POS_RE = re.compile(r"(?<![а-яёa-z])(?:огрн|огрнип|ogrn|ogrnip|егрюл|егрип)(?![а-яёa-z])")
_OGRN_NEG_RE = re.compile(
    r"(?<![а-яёa-z])(?:накладн\w*|сдэк|cdek|трек\w*|track\w*|заказ\w*|timestamp|unix|"
    r"интервал\s+выборки|seed|штрихкод\w*|ean)(?![а-яёa-z])",
)
_KPP_POS_RE = re.compile(r"(?<![а-яёa-z])(?:кпп|kpp)(?![а-яёa-z])")
_CARD_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:карт[аыеуой]*|card\w*|cvv2?|cvc2?|срок|valid\s+thru|мир|mir|visa|"
    r"mastercard|unionpay|maestro|amex|pan|payment_pan)(?![а-яёa-z])",
)
_CARD_NEG_RE = re.compile(
    r"(?<![а-яёa-z])(?:артикул\w*|sku|штрихкод\w*|ean|isbn|трек\w*|накладн\w*|сдэк|cdek|"
    r"timestamp|unix|seed)(?![а-яёa-z])",
)
_BIK_POS_RE = re.compile(r"(?<![а-яёa-z])(?:бик|bik)(?![а-яёa-z])")
_BANK_ACC_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:р/с|к/с|расч[её]тн\w*|корр\w*\s+сч[её]т\w*|лицев\w*\s+сч[её]т\w*|"
    r"сч[её]т[ауеом]*|реквизит\w*|банк\w*|получател\w*|account\w*|iban)(?![а-яёa-z])",
)
_PASSPORT_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:паспорт\w*|загранпаспорт\w*|passport\w*|doc_num\w*|document_number|"
    r"сери[яиею]|выдан\w*|код\s+подразделения|уфмс|мвд|овд|увд|гувм)(?![а-яёa-z])",
)
_FOREIGN_PASSPORT_STRICT_RE = re.compile(
    r"(?<![а-яёa-z])(?:загранпаспорт\w*|заграничн\w+\s+паспорт\w*|foreign\s+passport)(?![а-яёa-z])",
)
_PASSPORT_WEAK_POS_RE = re.compile(r"(?<![а-яёa-z])(?:номер|№)(?![а-яёa-z])")
_PASSPORT_NEG_RE = re.compile(
    r"(?<![а-яёa-z])(?:заказ\w*|order|артикул\w*|трек\w*|track\w*|накладн\w*|квитанц\w*|"
    r"инвойс\w*|invoice|парти[яиюей]|склад\w*|кадастр\w*|окпд\d?)(?![а-яёa-z])",
)
_OMS_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:омс|oms|полис\w*|медполис\w*|страхов\w*|фомс|тфомс|енп)(?![а-яёa-z])",
)
_PHONE_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:тел(?:ефон\w*)?|phone\w*|моб(?:ильн\w*)?|whatsapp|telegram|viber|"
    r"звон\w*|контакт\w*)(?![а-яёa-z])",
)
_AUTO_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:авто\w*|машин\w*|госномер\w*|грз|птс|стс|тс|такси)(?![а-яёa-z])",
)
_DL_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:ву|в/у|права|водител\w*|гибдд|driver|license)(?![а-яёa-z])",
)
_DOB_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:дат[аыеу]\s+рождени\w*|год\s+рождени\w*|д\.\s*р\.|"
    r"родился|родилась|урожен\w*|dob|date[_\s-]*of[_\s-]*birth|birth[_\s-]*date|birthday)(?![а-яёa-z])",
)
_IP_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:ip|ipv4|ipv6|хост\w*|host|сервер\w*|server|адрес\w*|подключ\w*)(?![а-яёa-z])",
)
_IP_DOC_EXAMPLE_RE = re.compile(
    r"(?<![а-яёa-z])(?:rfc\s*5737|пример\w*|example|документац\w*)(?![а-яёa-z])",
)
_IP_SEMVER_NEG_RE = re.compile(
    r"(?<![а-яёa-z])(?:верси\w*|version|релиз\w*|release|сборк\w*|build|odbc|openssl|"
    r"runtime|драйвер\w*|пакет\w*|прошивк\w*|ревизи\w*|плагин\w*|модул\w*\s+ядр\w*|kernel|"
    r"пункт\w*|разд\w*)(?![а-яёa-z])"
    r"|(?<![а-яёa-z])(?:п|ст|гл|ч|табл|прил)\.\s*$"
    r"|\.net\b",
)

# ---------------------------------------------------------------------------
# Pure checksum & validation helpers
# ---------------------------------------------------------------------------

_INN_WEIGHTS_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_CBR_WEIGHTS_23 = (7, 1, 3) * 7 + (7, 1)

_CBR_STANDALONE_PREFIXES: frozenset[str] = frozenset(
    {
        "40701",
        "40702",
        "40703",
        "40802",
        "40817",
        "40820",
        "42301",
        "42601",
        "30101",
        "30232",
    }
)
_CBR_CURRENCIES: frozenset[str] = frozenset({"810", "643", "840", "978", "156"})

_MAX_CUSTOM_PATTERN_LEN = 2000
_BIK_ACCOUNT_PROXIMITY = 450


def _digits(value: str) -> str:
    return _ASCII_NON_DIGITS_RE.sub("", value)


def _is_ascii_digits(value: str) -> bool:
    return value.isascii() and value.isdigit()


def is_valid_snils(digits: str) -> bool:
    """Validate 11-digit Russian SNILS via PFR mod-101 rule (> 001-001-998)."""
    if len(digits) != 11 or not _is_ascii_digits(digits):
        return False
    number_part = digits[:9]
    if int(number_part) <= 1_001_998:
        return False
    total = sum(int(d) * w for d, w in zip(number_part, range(9, 0, -1), strict=True))
    rem = total % 101
    expected = 0 if rem == 100 else rem
    return expected == int(digits[9:11])


def is_valid_inn(digits: str) -> bool:
    """Validate 10-digit (org) or 12-digit (individual) Russian INN via FNS weights."""
    if not _is_ascii_digits(digits) or len(set(digits)) == 1:
        return False
    if len(digits) == 10:
        if digits[:2] == "00":
            return False
        total = sum(int(digits[i]) * w for i, w in enumerate(_INN_WEIGHTS_10))
        return (total % 11) % 10 == int(digits[9])
    if len(digits) == 12:
        if digits[:2] == "00":
            return False
        n11 = sum(int(digits[i]) * w for i, w in enumerate(_INN_WEIGHTS_11))
        n12 = sum(int(digits[i]) * w for i, w in enumerate(_INN_WEIGHTS_12))
        return (n11 % 11) % 10 == int(digits[10]) and (n12 % 11) % 10 == int(digits[11])
    return False


def is_valid_ogrn(digits: str) -> bool:
    """Validate 13-digit OGRN (mod 11 % 10) or 15-digit OGRNIP (mod 13 % 10)."""
    if not _is_ascii_digits(digits) or digits[0] == "0" or len(set(digits)) == 1:
        return False
    if len(digits) == 13:
        return (int(digits[:12]) % 11) % 10 == int(digits[12])
    if len(digits) == 15:
        return (int(digits[:14]) % 13) % 10 == int(digits[14])
    return False


def is_valid_luhn(digits: str) -> bool:
    """Validate 13-19 digit PAN using the Luhn checksum algorithm."""
    if not (13 <= len(digits) <= 19) or not _is_ascii_digits(digits) or len(set(digits)) == 1:
        return False
    total = 0
    for idx, ch in enumerate(reversed(digits)):
        num = ord(ch) - 48
        if idx % 2 == 1:
            num *= 2
            if num > 9:
                num -= 9
        total += num
    return total % 10 == 0


def _detect_card_brand(digits: str) -> str | None:
    if not _is_ascii_digits(digits):
        return None
    if 16 <= len(digits) <= 19 and digits[:4] in {"2200", "2201", "2202", "2203", "2204"}:
        return "mir"
    if digits.startswith("4") and len(digits) in {13, 16, 19}:
        return "visa"
    if len(digits) == 16 and (
        digits[:2] in {"51", "52", "53", "54", "55"} or 2221 <= int(digits[:4]) <= 2720
    ):
        return "mastercard"
    if len(digits) == 15 and digits[:2] in {"34", "37"}:
        return "amex"
    if 16 <= len(digits) <= 19 and digits.startswith("62"):
        return "unionpay"
    if 13 <= len(digits) <= 19 and (
        digits[:2] in {"50", "56", "57", "58"} or digits.startswith(("63", "67"))
    ):
        return "maestro"
    return None


def is_valid_cbr_bik_account(bik: str, account: str) -> bool:
    """Validate 9-digit BIK (starting with 04) + 20-digit account via CBR mod-10 weights."""
    if len(bik) != 9 or not bik.startswith("04") or not _is_ascii_digits(bik):
        return False
    if len(account) != 20 or not _is_ascii_digits(account):
        return False
    prefix = f"0{bik[4:6]}" if bik[6:8] == "00" or account.startswith("30101") else bik[6:9]
    control_str = f"{prefix}{account}"
    total = sum(int(ch) * w for ch, w in zip(control_str, _CBR_WEIGHTS_23, strict=True))
    return total % 10 == 0


def _is_valid_passport_series(series: str) -> bool:
    if len(series) != 4 or not _is_ascii_digits(series):
        return False
    region = int(series[:2])
    year = int(series[2:4])
    if not (1 <= region <= 99):
        return False
    return 97 <= year <= 99 or 0 <= year <= 30


def _is_valid_email(value: str) -> bool:
    local, _, domain = value.partition("@")
    if not local or not domain or len(local) > 64 or len(domain) > 253:
        return False
    if ".." in local or ".." in domain or local.startswith(".") or local.endswith("."):
        return False
    labels = domain.split(".")
    if len(labels) < 2:
        return False
    for label in labels:
        if not label or len(label) > 63 or label.startswith("-") or label.endswith("-"):
            return False
    tld = labels[-1]
    return len(tld) >= 2 and (tld.isalpha() or tld.lower().startswith("xn--"))


def _split_top_level_alternation(group_body: str) -> list[str]:
    """Split a regex group body on top-level ``|`` tokens."""
    branches: list[str] = []
    depth = 0
    in_class = False
    escaped = False
    start = 0
    for idx, ch in enumerate(group_body):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if in_class:
            if ch == "]":
                in_class = False
            continue
        if ch == "[":
            in_class = True
            continue
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == "|" and depth == 0:
            branches.append(group_body[start:idx])
            start = idx + 1
    branches.append(group_body[start:])
    return branches


def _leading_atom_signature(branch: str) -> str:
    """Return a conservative signature of a branch's leading matchable token."""
    s = branch.lstrip("^")
    if s.startswith("?:"):
        s = s[2:]
    if not s:
        return "*"
    if s[0] in {".", "[", "("}:
        return "*"
    if s[0] == "\\":
        if len(s) < 2:
            return "*"
        esc = s[1]
        if esc in {"d", "D", "w", "W", "s", "S"}:
            return f"\\{esc}"
        if esc in {"b", "B", "A", "Z"}:
            return _leading_atom_signature(s[2:])
        return esc
    return s[0]


def _has_ambiguous_alternation(group_body: str) -> bool:
    branches = _split_top_level_alternation(group_body)
    if len(branches) < 2:
        return False
    seen_sigs: set[str] = set()
    cleaned_branches = [b[2:] if b.startswith("?:") else b for b in branches]
    for idx, br in enumerate(cleaned_branches):
        if not br:
            return True
        sig = _leading_atom_signature(br)
        if sig == "*":
            return True
        if sig in seen_sigs:
            return True
        seen_sigs.add(sig)
        # Also catch prefix overlap across branches even if first chars matched via classes
        for prev in cleaned_branches[:idx]:
            if br.startswith(prev) or prev.startswith(br):
                return True
    return False


def _parse_quantifier_at(pattern: str, idx: int) -> tuple[bool, bool, int]:
    """Return ``(is_unbounded, is_large_repeat, end_idx_exclusive)`` at ``pattern[idx]``."""
    length = len(pattern)
    if idx >= length:
        return (False, False, idx)
    ch = pattern[idx]
    if ch in {"*", "+"}:
        end = idx + 1
        if end < length and pattern[end] == "?":
            end += 1
        return (True, True, end)
    if ch == "{":
        close = pattern.find("}", idx + 1)
        if close != -1:
            body = pattern[idx + 1 : close].strip()
            end = close + 1
            if end < length and pattern[end] == "?":
                end += 1
            if "," in body:
                _, _, right_s = body.partition(",")
                if not right_s.strip():
                    return (True, True, end)
                if right_s.strip().isdigit() and int(right_s.strip()) >= 12:
                    return (False, True, end)
                return (False, False, end)
            if body.isdigit() and int(body) >= 12:
                return (False, True, end)
    return (False, False, idx)


def _is_unsafe_regex(pattern: str) -> bool:
    """Detect ReDoS risks: nested/adjacent unbounded quantifiers and ambiguous alternations."""
    if not pattern or len(pattern) > _MAX_CUSTOM_PATTERN_LEN:
        return True

    # Stack entries: (group_content_start_idx, has_inner_unbounded)
    group_stack: list[tuple[int, bool]] = []
    # Per scope (0 = root, 1+ = inside group): whether immediately preceding atom was unbounded
    prev_atom_unbounded: list[bool] = [False]

    escaped = False
    in_char_class = False
    idx = 0
    length = len(pattern)

    while idx < length:
        ch = pattern[idx]
        if escaped:
            escaped = False
            unbounded, _, next_i = _parse_quantifier_at(pattern, idx + 1)
            if unbounded:
                if prev_atom_unbounded[-1]:
                    return True
                prev_atom_unbounded[-1] = True
                if group_stack:
                    g_start, _ = group_stack[-1]
                    group_stack[-1] = (g_start, True)
                idx = next_i
                continue
            prev_atom_unbounded[-1] = False
            idx += 1
            continue

        if ch == "\\":
            escaped = True
            idx += 1
            continue

        if in_char_class:
            if ch == "]":
                in_char_class = False
                unbounded, _, next_i = _parse_quantifier_at(pattern, idx + 1)
                if unbounded:
                    if prev_atom_unbounded[-1]:
                        return True
                    prev_atom_unbounded[-1] = True
                    if group_stack:
                        g_start, _ = group_stack[-1]
                        group_stack[-1] = (g_start, True)
                    idx = next_i
                    continue
                prev_atom_unbounded[-1] = False
            idx += 1
            continue

        if ch == "[":
            in_char_class = True
            idx += 1
            continue

        if ch == "(":
            group_stack.append((idx + 1, False))
            prev_atom_unbounded.append(False)
            idx += 1
            continue

        if ch == ")":
            g_start, inner_unbounded = group_stack.pop() if group_stack else (0, False)
            if len(prev_atom_unbounded) > 1:
                prev_atom_unbounded.pop()
            group_body = pattern[g_start:idx]
            outer_unbounded, outer_large, next_i = _parse_quantifier_at(pattern, idx + 1)
            if inner_unbounded and (outer_unbounded or outer_large):
                return True
            if (outer_unbounded or outer_large) and _has_ambiguous_alternation(group_body):
                return True
            if outer_unbounded:
                if prev_atom_unbounded[-1]:
                    return True
                prev_atom_unbounded[-1] = True
                if group_stack:
                    pg_start, _ = group_stack[-1]
                    group_stack[-1] = (pg_start, True)
            else:
                if inner_unbounded and group_stack:
                    pg_start, _ = group_stack[-1]
                    group_stack[-1] = (pg_start, True)
                prev_atom_unbounded[-1] = False
            idx = next_i if next_i > idx + 1 else idx + 1
            continue

        if ch == "|":
            prev_atom_unbounded[-1] = False
            idx += 1
            continue

        if ch in {"^", "$"}:
            idx += 1
            continue

        unbounded, _, next_i = _parse_quantifier_at(pattern, idx + 1)
        if unbounded:
            if prev_atom_unbounded[-1]:
                return True
            prev_atom_unbounded[-1] = True
            if group_stack:
                g_start, _ = group_stack[-1]
                group_stack[-1] = (g_start, True)
            idx = next_i
            continue

        prev_atom_unbounded[-1] = False
        idx += 1

    return False


def _context_window(
    text_lower: str, start: int, end: int, before: int = 80, after: int = 40
) -> str:
    return text_lower[max(0, start - before) : min(len(text_lower), end + after)]


def _overlaps_span(span: tuple[int, int], existing: set[tuple[int, int]]) -> bool:
    s, e = span
    return any(s < ex_e and ex_s < e for ex_s, ex_e in existing)


def _is_non_sensitive_ipv4(
    ip_str: str,
    text: str,
    text_lower: str,
    start: int,
    end: int,
    *,
    has_ip_semver: bool,
    has_ip_pos: bool,
    has_ip_doc: bool,
) -> bool:
    """Return True if an IPv4 match is loopback, wildcard, netmask, RFC 5737 doc, or SemVer."""
    dot1 = ip_str.find(".")
    p0 = ip_str[:dot1]
    # Wildcard 0.0.0.0/8, loopback 127.0.0.0/8, broadcast/netmask 255.*
    if p0 in {"0", "127", "255"}:
        return True

    # Preceded immediately by 'v'/'V' (e.g., v2.14.8.12)
    if start > 0 and text[start - 1] in {"v", "V"}:
        return True

    # RFC 5737 documentation ranges: 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24
    if p0 in {"192", "198", "203"} and (
        ip_str.startswith(("192.0.2.", "198.51.100.", "203.0.113."))
    ):
        if not has_ip_pos:
            return True
        win = _context_window(text_lower, start, end, before=64, after=32)
        if (has_ip_doc and _IP_DOC_EXAMPLE_RE.search(win)) or not _IP_POS_RE.search(win):
            return True

    # SemVer / OS build / legal clause context without explicit host/IP keyword
    if has_ip_semver:
        before_slice = text_lower[max(0, start - 64) : start]
        if _IP_SEMVER_NEG_RE.search(before_slice):
            return True
        win = _context_window(text_lower, start, end, before=64, after=32)
        if _IP_SEMVER_NEG_RE.search(win) and not (has_ip_pos and _IP_POS_RE.search(win)):
            return True

    return False


# ---------------------------------------------------------------------------
# Detector implementation
# ---------------------------------------------------------------------------


class RuPiiDetector:
    """Layer-1 deterministic detector for Russian PII, financial, and custom regexes."""

    def __init__(
        self,
        custom_patterns: Sequence[CustomPatternRule] = (),
        hmac_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._hmac_fn = hmac_fn
        self._compiled_custom: list[tuple[CustomPatternRule, re.Pattern[str]]] = []
        for rule in custom_patterns:
            if _is_unsafe_regex(rule.pattern):
                logger.warning("Skipping unsafe or oversized custom pattern %r", rule.name)
                continue
            try:
                compiled = re.compile(rule.pattern)
            except re.error as exc:
                logger.warning("Skipping invalid custom pattern %r: %s", rule.name, exc)
                continue
            self._compiled_custom.append((rule, compiled))

    @property
    def name(self) -> str:
        """Unique detector identifier."""
        return "ru_pii"

    @property
    def layer(self) -> int:
        """Pipeline execution layer (1 = fast deterministic)."""
        return 1

    def _build_detection(
        self,
        segment: NormalizedSegment,
        category: EntityCategory,
        norm_start: int,
        norm_end: int,
        confidence: float,
        metadata: dict[str, str] | None = None,
    ) -> Detection:
        raw_match = segment.text[norm_start:norm_end]
        if not segment.index_map and segment.span_source == SpanSource.DIRECT:
            mapped_start, mapped_end = norm_start, norm_end
        else:
            mapped_start, mapped_end = segment.map_span(norm_start, norm_end)
        clamped_conf = min(1.0, max(0.0, round(confidence, 4)))
        det = Detection.__new__(Detection)
        object.__setattr__(
            det,
            "__dict__",
            {
                "category": category,
                "start": mapped_start,
                "end": mapped_end,
                "confidence": clamped_conf,
                "detector": self.name,
                "span_source": segment.span_source,
                "evasion_detected": bool(segment.evasion_techniques),
                "evasion_techniques": segment.evasion_techniques,
                "compliance_refs": _COMPLIANCE_CACHE[category],
                "masked_preview": f"[{category.value}:len={len(raw_match)}]",
                "value_hmac": self._hmac_fn(raw_match) if self._hmac_fn else "",
                "source_message_indices": (),
                "metadata": metadata if metadata is not None else {},
            },
        )
        object.__setattr__(det, "__pydantic_fields_set__", _FIELDS_SET_TEMPLATE.copy())
        object.__setattr__(det, "__pydantic_extra__", None)
        object.__setattr__(det, "__pydantic_private__", None)
        return det

    def _detect_multi_digit_entities(
        self,
        segment: NormalizedSegment,
        text: str,
        text_lower: str,
        results: list[Detection],
    ) -> tuple[
        set[tuple[int, int]],
        set[tuple[int, int]],
        set[tuple[int, int]],
        set[tuple[int, int]],
        set[tuple[int, int]],
    ]:
        """Detect SNILS, OGRN/KPP, OMS, cards, BIK/accounts, DL, passports, and INN."""
        snils_spans: set[tuple[int, int]] = set()
        ogrn_spans: set[tuple[int, int]] = set()
        kpp_spans: set[tuple[int, int]] = set()
        oms_spans: set[tuple[int, int]] = set()
        dl_spans: set[tuple[int, int]] = set()
        passport_spans: set[tuple[int, int]] = set()
        inn_spans: set[tuple[int, int]] = set()
        bik_spans: set[tuple[int, int]] = set()
        account_spans: set[tuple[int, int]] = set()

        # 1. SNILS
        for m in _SNILS_RE.finditer(text):
            raw = m.group(0)
            digits = _digits(raw)
            if not is_valid_snils(digits):
                continue
            win = _context_window(text_lower, m.start(), m.end())
            has_snils_ctx = bool(_SNILS_POS_RE.search(win))
            if raw.count(".") == 3 and not has_snils_ctx and _IPV4_RE.fullmatch(raw):
                continue
            is_bare = len(raw) == 11
            if is_bare and digits[:2] in {"79", "89"} and not has_snils_ctx:
                continue
            conf = 0.95 if not is_bare else 0.85
            if has_snils_ctx:
                conf = 0.99
            snils_spans.add((m.start(), m.end()))
            results.append(
                self._build_detection(segment, EntityCategory.RU_SNILS, m.start(), m.end(), conf)
            )

        # 2. OGRN (13) / OGRNIP (15) + KPP (9)
        for m in _OGRN_RE.finditer(text):
            digits = m.group(0)
            if not is_valid_ogrn(digits):
                continue
            win = _context_window(text_lower, m.start(), m.end())
            has_ogrn_ctx = bool(_OGRN_POS_RE.search(win))
            if not has_ogrn_ctx and _OGRN_NEG_RE.search(win):
                continue
            if len(digits) == 13:
                if digits[0] not in {"1", "2", "5"} and not has_ogrn_ctx:
                    continue
                subtype = "ogrn"
            else:
                if digits[0] not in {"3", "4"} and not has_ogrn_ctx:
                    continue
                subtype = "ogrnip"
            conf = 0.98 if has_ogrn_ctx else 0.85
            ogrn_spans.add((m.start(), m.end()))
            results.append(
                self._build_detection(
                    segment,
                    EntityCategory.RU_OGRN,
                    m.start(),
                    m.end(),
                    conf,
                    metadata={"subtype": subtype},
                )
            )

        if "кпп" in text_lower or "kpp" in text_lower:
            for m in _KPP_RE.finditer(text):
                span = (m.start(), m.end())
                win = _context_window(text_lower, m.start(), m.end(), before=60, after=30)
                if not _KPP_POS_RE.search(win):
                    continue
                if m.group(0).startswith("04") and _BIK_POS_RE.search(
                    _context_window(text_lower, m.start(), m.end(), before=25, after=10)
                ):
                    continue
                kpp_spans.add(span)
                results.append(
                    self._build_detection(
                        segment,
                        EntityCategory.RU_OGRN,
                        m.start(),
                        m.end(),
                        0.96,
                        metadata={"subtype": "kpp"},
                    )
                )

        # 3. OMS (16 digits + context)
        for m in _OMS_RE.finditer(text):
            win = _context_window(text_lower, m.start(), m.end(), before=140, after=60)
            if not _OMS_POS_RE.search(win):
                continue
            conf = 0.96 if "омс" in win or "oms" in win else 0.92
            oms_spans.add((m.start(), m.end()))
            results.append(
                self._build_detection(segment, EntityCategory.RU_OMS, m.start(), m.end(), conf)
            )

        # 4. Bank cards (13-19 digits, Luhn + brand/context boost + nearby CVV/expiry)
        card_aux_spans: set[tuple[int, int]] = set()
        for m in _BANK_CARD_RE.finditer(text):
            span = (m.start(), m.end())
            raw = m.group(0)
            digits = _digits(raw)
            win = _context_window(text_lower, m.start(), m.end(), before=100, after=80)
            has_card_ctx = bool(_CARD_POS_RE.search(win))

            if not has_card_ctx and _CARD_NEG_RE.search(win):
                continue
            if span in oms_spans and not has_card_ctx:
                continue
            is_amex = len(digits) == 15 and digits[:2] in {"34", "37"}
            is_formatted = len(raw) > len(digits)
            if span in ogrn_spans and not has_card_ctx and not is_formatted:
                if is_amex:
                    if _OGRN_POS_RE.search(win):
                        continue
                else:
                    continue
            elif len(digits) == 13 and digits[0] in {"1", "2", "5"} and not has_card_ctx:
                continue

            if not is_valid_luhn(digits):
                continue

            brand = _detect_card_brand(digits)
            conf = 0.88 if is_formatted else (0.82 if is_amex else 0.72)
            if brand is not None:
                conf += 0.08
            if has_card_ctx:
                conf += 0.10
            meta: dict[str, str] | None = {"brand": brand} if brand is not None else None
            results.append(
                self._build_detection(
                    segment,
                    EntityCategory.RU_BANK_CARD,
                    m.start(),
                    m.end(),
                    min(0.99, conf),
                    metadata=meta,
                )
            )

            # Nearby CVV/CVC and expiry around a matched PAN
            aux_left = max(0, m.start() - 60)
            aux_right = min(len(text), m.end() + 80)
            aux_slice = text[aux_left:aux_right]
            for aux_re, aux_conf in ((_CARD_EXPIRY_RE, 0.95), (_CARD_CVV_RE, 0.96)):
                for am in aux_re.finditer(aux_slice):
                    a_span = (aux_left + am.start(), aux_left + am.end())
                    if a_span not in card_aux_spans and not (
                        a_span[0] < span[1] and span[0] < a_span[1]
                    ):
                        card_aux_spans.add(a_span)
                        results.append(
                            self._build_detection(
                                segment,
                                EntityCategory.RU_BANK_CARD,
                                a_span[0],
                                a_span[1],
                                aux_conf,
                                metadata={"subtype": "card_cvv_expiry"},
                            )
                        )

        # 5. BIK + 20-digit checking/correspondent account + standalone CBR accounts + RU IBAN
        biks = [(m.start(), m.end(), m.group(0)) for m in _BIK_RE.finditer(text)]
        rejected_by_bik: set[tuple[int, int]] = set()

        for m in _ACCOUNT_RE.finditer(text):
            acc_raw = m.group(0)
            acc_digits = _digits(acc_raw)
            if len(acc_digits) != 20:
                continue
            acc_span = (m.start(), m.end())

            if biks:
                acc_center = (m.start() + m.end()) // 2
                nearby_biks = sorted(
                    biks,
                    key=lambda b: abs(((b[0] + b[1]) // 2) - acc_center),
                )
                matched_bik_span: tuple[int, int] | None = None
                saw_explicit_bik = False
                for b_start, b_end, bik_val in nearby_biks:
                    dist = max(0, max(b_start - m.end(), m.start() - b_end))
                    if dist > _BIK_ACCOUNT_PROXIMITY:
                        break
                    if _BIK_POS_RE.search(
                        _context_window(text_lower, b_start, b_end, before=30, after=15)
                    ):
                        saw_explicit_bik = True
                    if is_valid_cbr_bik_account(bik_val, acc_digits):
                        matched_bik_span = (b_start, b_end)
                        break

                if matched_bik_span is not None:
                    account_spans.add(acc_span)
                    results.append(
                        self._build_detection(
                            segment,
                            EntityCategory.RU_BIK_ACCOUNT,
                            m.start(),
                            m.end(),
                            0.98,
                            metadata={"subtype": "account"},
                        )
                    )
                    if matched_bik_span not in bik_spans:
                        bik_spans.add(matched_bik_span)
                        results.append(
                            self._build_detection(
                                segment,
                                EntityCategory.RU_BIK_ACCOUNT,
                                matched_bik_span[0],
                                matched_bik_span[1],
                                0.97,
                                metadata={"subtype": "bik"},
                            )
                        )
                    continue
                if saw_explicit_bik:
                    rejected_by_bik.add(acc_span)
                    continue

            # Standalone 20-digit CBR account check
            if acc_digits[:5] in _CBR_STANDALONE_PREFIXES and acc_digits[5:8] in _CBR_CURRENCIES:
                win = _context_window(text_lower, m.start(), m.end(), before=120, after=60)
                if "." in acc_raw or _BANK_ACC_POS_RE.search(win):
                    account_spans.add(acc_span)
                    results.append(
                        self._build_detection(
                            segment,
                            EntityCategory.RU_BIK_ACCOUNT,
                            m.start(),
                            m.end(),
                            0.92,
                            metadata={"subtype": "standalone_account"},
                        )
                    )

        # Standalone BIK with explicit 'бик'/'bik' context
        for b_start, b_end, _ in biks:
            b_span = (b_start, b_end)
            if b_span in bik_spans or b_span in kpp_spans:
                continue
            win = _context_window(text_lower, b_start, b_end, before=40, after=20)
            if _BIK_POS_RE.search(win):
                bik_spans.add(b_span)
                results.append(
                    self._build_detection(
                        segment,
                        EntityCategory.RU_BIK_ACCOUNT,
                        b_start,
                        b_end,
                        0.93,
                        metadata={"subtype": "bik"},
                    )
                )

        # RU IBANs
        if "RU" in text:
            for m in _IBAN_RU_RE.finditer(text):
                results.append(
                    self._build_detection(
                        segment,
                        EntityCategory.RU_BIK_ACCOUNT,
                        m.start(),
                        m.end(),
                        0.95,
                        metadata={"subtype": "iban"},
                    )
                )

        # 6. Driver license (requires explicit DL context)
        for m in _DRIVER_LICENSE_RE.finditer(text):
            win = _context_window(text_lower, m.start(), m.end(), before=140, after=60)
            if not _DL_POS_RE.search(win):
                continue
            dl_spans.add((m.start(), m.end()))
            conf = 0.95 if ("права" in win or "водител" in win or "в/у" in win) else 0.90
            results.append(
                self._build_detection(
                    segment, EntityCategory.RU_DRIVER_LICENSE, m.start(), m.end(), conf
                )
            )

        # 7. Russian Passport (4-digit series + 6-digit number + 9-digit foreign passport)
        for m in _PASSPORT_LABELED_RE.finditer(text):
            series = _digits(m.group(1))
            if not _is_valid_passport_series(series):
                continue
            win = _context_window(text_lower, m.start(), m.end(), before=160, after=80)
            if _PASSPORT_NEG_RE.search(win) and not _PASSPORT_POS_RE.search(win):
                continue
            if (m.start(), m.end()) in dl_spans and not _PASSPORT_POS_RE.search(win):
                continue
            passport_spans.add((m.start(), m.end()))
            results.append(
                self._build_detection(segment, EntityCategory.RU_PASSPORT, m.start(), m.end(), 0.97)
            )

        for m in _PASSPORT_CONTIGUOUS_RE.finditer(text):
            span = (m.start(), m.end())
            if _overlaps_span(span, passport_spans):
                continue
            raw = m.group(0)
            digits = _digits(raw)
            if not _is_valid_passport_series(digits[:4]):
                continue
            win = _context_window(text_lower, m.start(), m.end(), before=160, after=80)
            has_pos = bool(_PASSPORT_POS_RE.search(win))
            has_weak = bool(_PASSPORT_WEAK_POS_RE.search(win))
            has_neg = bool(_PASSPORT_NEG_RE.search(win))

            if span in dl_spans and not has_pos:
                continue
            if has_neg and not has_pos:
                continue
            if not has_pos and _INN_POS_RE.search(win) and is_valid_inn(digits):
                continue

            is_bare_10 = len(raw) == 10
            if is_bare_10 and not has_pos:
                continue

            if has_pos:
                conf = 0.97 if not is_bare_10 else 0.90
            elif has_weak and not is_bare_10:
                conf = 0.85
            elif not is_bare_10:
                # Formatted 2+2+6 or 4+6 with valid region/year and no negative order context
                conf = 0.80 if len(raw) >= 12 else 0.75
            else:
                continue

            passport_spans.add(span)
            results.append(
                self._build_detection(segment, EntityCategory.RU_PASSPORT, m.start(), m.end(), conf)
            )

        if "загран" in text_lower or "foreign" in text_lower or "паспорт" in text_lower:
            for m in _FOREIGN_PASSPORT_RE.finditer(text):
                span = (m.start(), m.end())
                if (
                    _overlaps_span(span, passport_spans)
                    or _overlaps_span(span, bik_spans)
                    or _overlaps_span(span, kpp_spans)
                    or _overlaps_span(span, dl_spans)
                ):
                    continue
                digits = _digits(m.group(0))
                if len(digits) != 9:
                    continue
                win_wide = _context_window(text_lower, m.start(), m.end(), before=120, after=60)
                win_tight = _context_window(text_lower, m.start(), m.end(), before=36, after=16)
                if _FOREIGN_PASSPORT_STRICT_RE.search(win_wide) or (
                    len(m.group(0)) > 9
                    and "паспорт" in win_tight
                    and not _BIK_POS_RE.search(win_tight)
                    and not _KPP_POS_RE.search(win_tight)
                ):
                    passport_spans.add(span)
                    results.append(
                        self._build_detection(
                            segment,
                            EntityCategory.RU_PASSPORT,
                            m.start(),
                            m.end(),
                            0.96,
                            metadata={"subtype": "foreign_passport"},
                        )
                    )

        # 8. INN (10 / 12 digits)
        for m in _INN_RE.finditer(text):
            span = (m.start(), m.end())
            digits = m.group(0)
            if not is_valid_inn(digits):
                continue
            win = _context_window(text_lower, m.start(), m.end(), before=120, after=60)
            has_inn_ctx = bool(_INN_POS_RE.search(win))
            if span in passport_spans and not has_inn_ctx:
                continue
            if (
                len(digits) == 10
                and not has_inn_ctx
                and (_PASSPORT_NEG_RE.search(win) or _PASSPORT_POS_RE.search(win))
            ):
                continue
            if len(digits) == 12:
                conf = 0.98 if has_inn_ctx else 0.92
                inn_subtype = "person_12"
            else:
                conf = 0.96 if has_inn_ctx else 0.78
                inn_subtype = "org_10"
            inn_spans.add(span)
            results.append(
                self._build_detection(
                    segment,
                    EntityCategory.RU_INN,
                    m.start(),
                    m.end(),
                    conf,
                    metadata={"subtype": inn_subtype},
                )
            )
        return snils_spans, dl_spans, passport_spans, inn_spans, account_spans

    def detect(self, segment: NormalizedSegment, raw_text: str) -> list[Detection]:
        """Detect Russian PII, financial identifiers, contacts, and custom patterns."""
        _ = raw_text
        text = segment.text
        if not text or not text.strip():
            return []

        has_two_digits = _HAS_2_DIGITS_RE.search(text) is not None
        has_alpha = _HAS_ALPHA_RE.search(text) is not None
        text_lower = text.lower() if (has_two_digits or has_alpha) else text
        results: list[Detection] = []

        phone_spans: set[tuple[int, int]] = set()
        if has_two_digits:
            snils_spans, dl_spans, passport_spans, inn_spans, account_spans = (
                self._detect_multi_digit_entities(segment, text, text_lower, results)
            )
        else:
            snils_spans = dl_spans = passport_spans = inn_spans = account_spans = set()

        # 9. Russian & CIS Phones (+7 / 8 / 7 / +375 / +998 / 10-digit)
        if has_two_digits:
            for m in _PHONE_RE.finditer(text):
                span = (m.start(), m.end())
                win = _context_window(text_lower, m.start(), m.end())
                has_phone_ctx = bool(_PHONE_POS_RE.search(win))
                if span in snils_spans and _SNILS_POS_RE.search(win) and not has_phone_ctx:
                    continue
                raw = m.group(0)
                digits = _digits(raw)
                if len(digits) != 11:
                    continue
                is_formatted = len(raw) > 11
                if raw[0] == "7" and not is_formatted and digits[1] != "9" and not has_phone_ctx:
                    continue
                conf = 0.94 if is_formatted else 0.86
                if has_phone_ctx:
                    conf = min(0.99, conf + 0.05)
                phone_spans.add(span)
                results.append(
                    self._build_detection(
                        segment, EntityCategory.RU_PHONE, m.start(), m.end(), conf
                    )
                )

            if "+" in text:
                for m in _PHONE_CIS_RE.finditer(text):
                    span = (m.start(), m.end())
                    if _overlaps_span(span, phone_spans):
                        continue
                    phone_spans.add(span)
                    results.append(
                        self._build_detection(
                            segment,
                            EntityCategory.RU_PHONE,
                            m.start(),
                            m.end(),
                            0.95,
                            metadata={"subtype": "cis"},
                        )
                    )

            has_any_phone_kw = has_alpha and bool(_PHONE_POS_RE.search(text_lower))
            if "(" in text or has_any_phone_kw:
                for m in _PHONE_10_RE.finditer(text):
                    span = (m.start(), m.end())
                    if (
                        _overlaps_span(span, phone_spans)
                        or _overlaps_span(span, passport_spans)
                        or _overlaps_span(span, inn_spans)
                        or _overlaps_span(span, snils_spans)
                        or _overlaps_span(span, dl_spans)
                    ):
                        continue
                    raw = m.group(0)
                    digits = _digits(raw)
                    if len(digits) != 10:
                        continue
                    win = _context_window(text_lower, m.start(), m.end())
                    has_phone_ctx = bool(_PHONE_POS_RE.search(win))
                    if not raw.startswith("(") and not has_phone_ctx:
                        continue
                    conf = 0.93 if (raw.startswith("(") and has_phone_ctx) else 0.88
                    phone_spans.add(span)
                    results.append(
                        self._build_detection(
                            segment, EntityCategory.RU_PHONE, m.start(), m.end(), conf
                        )
                    )

        # 10. Russian Auto Plate
        if has_two_digits and has_alpha:
            plate_spans: set[tuple[int, int]] = set()
            for m in _AUTO_PLATE_TYPE1_RE.finditer(text):
                win = _context_window(text_lower, m.start(), m.end())
                conf = 0.96 if _AUTO_POS_RE.search(win) else 0.88
                plate_spans.add((m.start(), m.end()))
                results.append(
                    self._build_detection(
                        segment, EntityCategory.RU_AUTO_PLATE, m.start(), m.end(), conf
                    )
                )
            for m in _AUTO_PLATE_ALT_RE.finditer(text):
                span = (m.start(), m.end())
                if span in plate_spans:
                    continue
                win = _context_window(text_lower, m.start(), m.end())
                if not _AUTO_POS_RE.search(win):
                    continue
                results.append(
                    self._build_detection(
                        segment, EntityCategory.RU_AUTO_PLATE, m.start(), m.end(), 0.90
                    )
                )

        # 11. Date of Birth (contextual)
        if has_two_digits and has_alpha and _DOB_POS_RE.search(text_lower):
            for m in _DOB_RE.finditer(text):
                win = _context_window(text_lower, m.start(), m.end(), before=80, after=40)
                if _DOB_POS_RE.search(win):
                    results.append(
                        self._build_detection(
                            segment,
                            EntityCategory.RU_DATE_OF_BIRTH,
                            m.start(),
                            m.end(),
                            0.95,
                        )
                    )

        # 12. Email
        if "@" in text:
            for m in _EMAIL_RE.finditer(text):
                if not _is_valid_email(m.group(0)):
                    continue
                results.append(
                    self._build_detection(segment, EntityCategory.EMAIL, m.start(), m.end(), 0.95)
                )

        # 13. IPv4 & IPv6
        if "." in text:
            has_ip_semver = has_alpha and bool(_IP_SEMVER_NEG_RE.search(text_lower))
            has_ip_pos = has_alpha and bool(_IP_POS_RE.search(text_lower))
            has_ip_doc = has_alpha and bool(_IP_DOC_EXAMPLE_RE.search(text_lower))
            check_ip_overlaps = bool(account_spans or phone_spans)
            if not has_ip_semver and not has_ip_pos and not check_ip_overlaps:
                det_new = Detection.__new__
                obj_setattr = object.__setattr__
                fields_copy = _FIELDS_SET_TEMPLATE.copy
                cat_ip = EntityCategory.IP_ADDRESS
                ip_refs = _COMPLIANCE_CACHE[cat_ip]
                hmac_fn = self._hmac_fn
                detector_name = self.name
                span_source = segment.span_source
                evasion_detected = bool(segment.evasion_techniques)
                evasion_techs = segment.evasion_techniques
                direct_span = not segment.index_map and span_source == SpanSource.DIRECT
                append_result = results.append
                skip_prefixes = ("0.", "127.", "255.", "192.0.2.", "198.51.100.", "203.0.113.")
                v_chars = {"v", "V"}
                if direct_span and hmac_fn is None and not has_alpha:
                    previews = _IPV4_PREVIEWS
                    base_d: dict[str, object] = {
                        "category": cat_ip,
                        "start": 0,
                        "end": 0,
                        "confidence": 0.85,
                        "detector": detector_name,
                        "span_source": span_source,
                        "evasion_detected": evasion_detected,
                        "evasion_techniques": evasion_techs,
                        "compliance_refs": ip_refs,
                        "masked_preview": "",
                        "value_hmac": "",
                        "source_message_indices": (),
                        "metadata": _IPV4_METADATA,
                    }
                    fast_ips: list[Detection] = []
                    fast_append = fast_ips.append
                    for m in _IPV4_SENSITIVE_FAST_RE.finditer(text):
                        start, end = m.span()
                        det = det_new(Detection)
                        obj_setattr(
                            det,
                            "__dict__",
                            {
                                **base_d,
                                "start": start,
                                "end": end,
                                "masked_preview": previews[end - start],
                            },
                        )
                        obj_setattr(det, "__pydantic_fields_set__", fields_copy())
                        obj_setattr(det, "__pydantic_extra__", None)
                        obj_setattr(det, "__pydantic_private__", None)
                        fast_append(det)
                    if not results and ":" not in text and not self._compiled_custom:
                        return fast_ips
                    results.extend(fast_ips)
                elif direct_span and hmac_fn is None:
                    for m in _IPV4_RE.finditer(text):
                        start, end = m.span()
                        ip_str = m.group(0)
                        if ip_str.startswith(skip_prefixes) or (
                            start > 0 and text[start - 1] in v_chars
                        ):
                            continue
                        det = det_new(Detection)
                        obj_setattr(
                            det,
                            "__dict__",
                            {
                                "category": cat_ip,
                                "start": start,
                                "end": end,
                                "confidence": 0.85,
                                "detector": detector_name,
                                "span_source": span_source,
                                "evasion_detected": evasion_detected,
                                "evasion_techniques": evasion_techs,
                                "compliance_refs": ip_refs,
                                "masked_preview": f"[ip_address:len={end - start}]",
                                "value_hmac": "",
                                "source_message_indices": (),
                                "metadata": {"subtype": "ipv4"},
                            },
                        )
                        obj_setattr(det, "__pydantic_fields_set__", fields_copy())
                        obj_setattr(det, "__pydantic_extra__", None)
                        obj_setattr(det, "__pydantic_private__", None)
                        append_result(det)
                else:
                    for m in _IPV4_RE.finditer(text):
                        start, end = m.span()
                        ip_str = m.group(0)
                        if ip_str.startswith(skip_prefixes) or (
                            start > 0 and text[start - 1] in v_chars
                        ):
                            continue
                        mapped_start, mapped_end = (
                            (start, end) if direct_span else segment.map_span(start, end)
                        )
                        det = det_new(Detection)
                        obj_setattr(
                            det,
                            "__dict__",
                            {
                                "category": cat_ip,
                                "start": mapped_start,
                                "end": mapped_end,
                                "confidence": 0.85,
                                "detector": detector_name,
                                "span_source": span_source,
                                "evasion_detected": evasion_detected,
                                "evasion_techniques": evasion_techs,
                                "compliance_refs": ip_refs,
                                "masked_preview": f"[ip_address:len={end - start}]",
                                "value_hmac": hmac_fn(ip_str) if hmac_fn else "",
                                "source_message_indices": (),
                                "metadata": {"subtype": "ipv4"},
                            },
                        )
                        obj_setattr(det, "__pydantic_fields_set__", fields_copy())
                        obj_setattr(det, "__pydantic_extra__", None)
                        obj_setattr(det, "__pydantic_private__", None)
                        append_result(det)
            else:
                for m in _IPV4_RE.finditer(text):
                    start, end = m.span()
                    if check_ip_overlaps:
                        span = (start, end)
                        if _overlaps_span(span, account_spans) or _overlaps_span(span, phone_spans):
                            continue
                    ip_str = m.group(0)
                    if _is_non_sensitive_ipv4(
                        ip_str,
                        text,
                        text_lower,
                        start,
                        end,
                        has_ip_semver=has_ip_semver,
                        has_ip_pos=has_ip_pos,
                        has_ip_doc=has_ip_doc,
                    ):
                        continue
                    if has_ip_pos:
                        win = _context_window(text_lower, start, end)
                        conf = 0.95 if _IP_POS_RE.search(win) else 0.85
                    else:
                        conf = 0.85
                    results.append(
                        self._build_detection(
                            segment,
                            EntityCategory.IP_ADDRESS,
                            start,
                            end,
                            conf,
                            metadata={"subtype": "ipv4"},
                        )
                    )

        if ":" in text:
            for m in _IPV6_CANDIDATE_RE.finditer(text):
                cand = m.group(0)
                try:
                    addr = ipaddress.IPv6Address(cand)
                except ValueError:
                    continue
                if addr.is_loopback or addr.is_unspecified:
                    continue
                win = _context_window(text_lower, m.start(), m.end())
                conf = 0.95 if _IP_POS_RE.search(win) else 0.88
                results.append(
                    self._build_detection(
                        segment,
                        EntityCategory.IP_ADDRESS,
                        m.start(),
                        m.end(),
                        conf,
                        metadata={"subtype": "ipv6"},
                    )
                )

        # 14. User custom patterns
        for rule, compiled in self._compiled_custom:
            for m in compiled.finditer(text):
                if m.start() == m.end():
                    continue
                results.append(
                    self._build_detection(
                        segment,
                        rule.category,
                        m.start(),
                        m.end(),
                        rule.confidence,
                        metadata={"rule_name": rule.name, "rule_action": rule.action.value},
                    )
                )

        if len(results) > 1:
            results.sort(key=lambda det: (det.start, det.end, -det.confidence))
        return results


__all__ = [
    "RuPiiDetector",
    "is_valid_cbr_bik_account",
    "is_valid_inn",
    "is_valid_luhn",
    "is_valid_ogrn",
    "is_valid_snils",
]
