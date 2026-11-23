"""Deterministic Russian PII, financial identifier, contact, and custom regex detectors."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.config import CustomPatternRule
from pokrov.core.contracts import (
    Detection,
    EntityCategory,
    NormalizedSegment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-compiled linear-time patterns
# ---------------------------------------------------------------------------

_SNILS_RE = re.compile(
    r"(?<!\d)(?:\d{3}[ -]\d{3}[ -]\d{3}[ -]\d{2}|\d{11})(?!\d)",
)

_INN_RE = re.compile(r"(?<!\d)(?:\d{12}|\d{10})(?!\d)")

_OGRN_RE = re.compile(r"(?<!\d)(?:\d{15}|\d{13})(?!\d)")

_BANK_CARD_RE = re.compile(
    r"(?<!\d)(?:"
    r"\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}(?:[ -]\d{1,3})?"
    r"|\d{4}[ -]\d{6}[ -]\d{5}"
    r"|\d{13,19}"
    r")(?!\d)",
)

_BIK_RE = re.compile(r"(?<!\d)04\d{7}(?!\d)")
_ACCOUNT_RE = re.compile(
    r"(?<!\d)(?:\d{20}|\d{5}[. ]\d{3}[. ]\d[. ]\d{4}[. ]\d{7})(?!\d)",
)

_PASSPORT_CONTIGUOUS_RE = re.compile(
    r"(?<!\d)(?:\d{2}[ -]\d{2}[ -]\d{6}|\d{4}[ -]\d{6}|\d{10})(?!\d)",
)
_PASSPORT_LABELED_RE = re.compile(
    r"(?<!\d)(\d{2}[ -]?\d{2})(?:[ ,;]+(?:номер|№)[ :.-]*|[ ]+(?:номер|№)[ :.-]*)(\d{6})(?!\d)",
    re.IGNORECASE,
)

_OMS_RE = re.compile(
    r"(?<!\d)(?:\d{16}|\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4})(?!\d)",
)

_PHONE_RE = re.compile(
    r"(?<![\dA-Za-zА-Яа-яЁё])"
    r"(?:\+7|8)"
    r"(?:"
    r"[ -]?(?:\([3489]\d{2}\)|[3489]\d{2})[ -]?\d{3}(?:[ -]?\d{2}[ -]?\d{2}|[ -]?\d{4})"
    r"|[ -]?(?:\([348]\d{3}\)|[348]\d{3})[ -]?\d{2}[ -]?\d{2}[ -]?\d{2}"
    r"|[ -]?(?:\([348]\d{4}\)|[348]\d{4})[ -]?\d[ -]?\d{2}[ -]?\d{2}"
    r")"
    r"(?!\d)",
)

_PLATE_LETTERS = "АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx"
_AUTO_PLATE_TYPE1_RE = re.compile(
    rf"(?<![0-9A-Za-zА-Яа-яЁё])"
    rf"[{_PLATE_LETTERS}][ -]?(?!000)\d{{3}}[ -]?[{_PLATE_LETTERS}]{{2}}"
    rf"[ -]?(?:[1279]\d{{2}}|(?!00)\d{{2}})"
    rf"(?:[ -]?(?:RUS|rus))?"
    rf"(?![0-9A-Za-zА-Яа-яЁё])",
)
_AUTO_PLATE_ALT_RE = re.compile(
    rf"(?<![0-9A-Za-zА-Яа-яЁё])"
    rf"(?:[{_PLATE_LETTERS}]{{2}}[ -]?\d{{3,4}}|\d{{4}}[ -]?[{_PLATE_LETTERS}]{{2}})"
    rf"[ -]?(?:[1279]\d{{2}}|(?!00)\d{{2}})"
    rf"(?:[ -]?(?:RUS|rus))?"
    rf"(?![0-9A-Za-zА-Яа-яЁё])",
)

_DRIVER_LICENSE_RE = re.compile(
    rf"(?<!\d)(?:"
    rf"\d{{2}}[ -]?\d{{2}}[ -]?\d{{6}}"
    rf"|\d{{2}}[ -]?[{_PLATE_LETTERS}]{{2}}[ -]?\d{{6}}"
    rf")(?!\d)",
)

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
    r"(?![A-Za-z0-9._%+-])",
)

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4_RE = re.compile(
    rf"(?<![\d.])({_OCTET}(?:\.{_OCTET}){{3}})(?!\d)(?!\.\d)",
)

_DIGITS_ONLY_RE = re.compile(r"\D+")

# ---------------------------------------------------------------------------
# Context keywords (all lowercase; checked within bounded local windows)
# ---------------------------------------------------------------------------

_SNILS_POS_RE = re.compile(r"(?<![а-яёa-z])(?:снилс|snils|пфр|сфр|страховой\s+номер)(?![а-яёa-z])")
_INN_POS_RE = re.compile(r"(?<![а-яёa-z])(?:инн|inn|кпп|фнс|егрюл|егрип|реквизит\w*)(?![а-яёa-z])")
_OGRN_POS_RE = re.compile(r"(?<![а-яёa-z])(?:огрн|огрнип|ogrn|ogrnip|егрюл|егрип)(?![а-яёa-z])")
_CARD_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:карт[аыеуой]*|card|cvv|cvc|срок|valid\s+thru|мир|mir|visa|mastercard|pan)(?![а-яёa-z])",
)
_PASSPORT_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:паспорт\w*|сери[яиею]|выдан\w*|код\s+подразделения|уфмс|мвд|овд|увд)(?![а-яёa-z])",
)
_PASSPORT_WEAK_POS_RE = re.compile(r"(?<![а-яёa-z])(?:номер|№)(?![а-яёa-z])")
_PASSPORT_NEG_RE = re.compile(
    r"(?<![а-яёa-z])(?:заказ\w*|order|артикул\w*|трек\w*|track\w*|накладн\w*|квитанц\w*|инвойс\w*|invoice)(?![а-яёa-z])",
)
_OMS_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:омс|oms|полис\w*|медполис\w*|страхов\w*|фомс|тфомс|енп)(?![а-яёa-z])",
)
_PHONE_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:тел(?:ефон)?|phone|моб(?:ильный)?|whatsapp|telegram|viber|звон\w*|контакт\w*)(?![а-яёa-z])",
)
_AUTO_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:авто\w*|машин\w*|госномер\w*|грз|птс|стс|тс|такси)(?![а-яёa-z])",
)
_DL_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:ву|в/у|права|водител\w*|гибдд|driver|license)(?![а-яёa-z])",
)
_IP_POS_RE = re.compile(
    r"(?<![а-яёa-z])(?:ip|ipv4|хост\w*|host|сервер\w*|server|адрес\w*|подключ\w*)(?![а-яёa-z])",
)

# ---------------------------------------------------------------------------
# Pure checksum & validation helpers
# ---------------------------------------------------------------------------

_INN_WEIGHTS_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_CBR_WEIGHTS_23 = (7, 1, 3) * 7 + (7, 1)

_MAX_CUSTOM_PATTERN_LEN = 2000
_BIK_ACCOUNT_PROXIMITY = 450


def _digits(value: str) -> str:
    return _DIGITS_ONLY_RE.sub("", value)


def is_valid_snils(digits: str) -> bool:
    """Validate 11-digit Russian SNILS via PFR mod-101 rule (> 001-001-998)."""
    if len(digits) != 11 or not digits.isdigit():
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
    if not digits.isdigit() or len(set(digits)) == 1:
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
    if not digits.isdigit() or digits[0] == "0" or len(set(digits)) == 1:
        return False
    if len(digits) == 13:
        return (int(digits[:12]) % 11) % 10 == int(digits[12])
    if len(digits) == 15:
        return (int(digits[:14]) % 13) % 10 == int(digits[14])
    return False


def is_valid_luhn(digits: str) -> bool:
    """Validate 13-19 digit PAN using the Luhn checksum algorithm."""
    if not (13 <= len(digits) <= 19) or not digits.isdigit() or len(set(digits)) == 1:
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
    if 16 <= len(digits) <= 19 and digits[:4] in {"2200", "2201", "2202", "2203", "2204"}:
        return "mir"
    if digits.startswith("4") and len(digits) in {13, 16, 19}:
        return "visa"
    if len(digits) == 16 and (
        digits[:2] in {"51", "52", "53", "54", "55"} or 2221 <= int(digits[:4]) <= 2720
    ):
        return "mastercard"
    return None


def is_valid_cbr_bik_account(bik: str, account: str) -> bool:
    """Validate 9-digit BIK (starting with 04) + 20-digit account via CBR mod-10 weights."""
    if len(bik) != 9 or not bik.startswith("04") or not bik.isdigit():
        return False
    if len(account) != 20 or not account.isdigit():
        return False
    prefix = f"0{bik[4:6]}" if bik[6:8] == "00" or account.startswith("30101") else bik[6:9]
    control_str = f"{prefix}{account}"
    total = sum(int(ch) * w for ch, w in zip(control_str, _CBR_WEIGHTS_23, strict=True))
    return total % 10 == 0


def _is_valid_passport_series(series: str) -> bool:
    if len(series) != 4 or not series.isdigit():
        return False
    region = int(series[:2])
    year = int(series[2:4])
    if not (1 <= region <= 99):
        return False
    return 97 <= year <= 99 or 0 <= year <= 30


def _is_valid_email(value: str) -> bool:
    local, _, domain = value.partition("@")
    if not local or not domain or ".." in local or ".." in domain:
        return False
    if local.startswith(".") or local.endswith("."):
        return False
    labels = domain.split(".")
    if len(labels) < 2:
        return False
    for label in labels:
        if not label or label.startswith("-") or label.endswith("-"):
            return False
    tld = labels[-1]
    return len(tld) >= 2 and (tld.isalpha() or tld.lower().startswith("xn--"))


def _is_unsafe_regex(pattern: str) -> bool:
    """Detect invalid length or nested unbounded quantifiers that risk ReDoS."""
    if not pattern or len(pattern) > _MAX_CUSTOM_PATTERN_LEN:
        return True

    # Track whether each open parenthesis group contains an unbounded repetition
    group_has_unbounded: list[bool] = []
    escaped = False
    in_char_class = False
    idx = 0
    length = len(pattern)

    while idx < length:
        ch = pattern[idx]
        if escaped:
            escaped = False
            idx += 1
            continue
        if ch == "\\":
            escaped = True
            idx += 1
            continue
        if in_char_class:
            if ch == "]":
                in_char_class = False
            idx += 1
            continue
        if ch == "[":
            in_char_class = True
            idx += 1
            continue
        if ch == "(":
            group_has_unbounded.append(False)
            idx += 1
            continue
        if ch == ")":
            inner_unbounded = group_has_unbounded.pop() if group_has_unbounded else False
            # Look ahead to see if this group has an outer multiplier
            next_idx = idx + 1
            if next_idx < length:
                nxt = pattern[next_idx]
                outer_repeats = nxt in {"*", "+"}
                if nxt == "{":
                    close_brace = pattern.find("}", next_idx + 1)
                    if close_brace != -1:
                        body = pattern[next_idx + 1 : close_brace]
                        if "," in body:
                            outer_repeats = True
                if inner_unbounded and outer_repeats:
                    return True
                if outer_repeats and group_has_unbounded:
                    group_has_unbounded[-1] = True
            elif inner_unbounded and group_has_unbounded:
                group_has_unbounded[-1] = True
            idx += 1
            continue

        if ch in {"*", "+"} and group_has_unbounded:
            group_has_unbounded[-1] = True
        elif ch == "{" and group_has_unbounded:
            close_brace = pattern.find("}", idx + 1)
            if close_brace != -1 and "," in pattern[idx + 1 : close_brace]:
                group_has_unbounded[-1] = True

        idx += 1

    return False


def _context_window(
    text_lower: str, start: int, end: int, before: int = 80, after: int = 40
) -> str:
    return text_lower[max(0, start - before) : min(len(text_lower), end + after)]


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
        mapped_start, mapped_end = segment.map_span(norm_start, norm_end)
        clamped_conf = min(1.0, max(0.0, round(confidence, 4)))
        return Detection(
            category=category,
            start=mapped_start,
            end=mapped_end,
            confidence=clamped_conf,
            detector=self.name,
            span_source=segment.span_source,
            evasion_detected=bool(segment.evasion_techniques),
            evasion_techniques=segment.evasion_techniques,
            compliance_refs=get_compliance_refs(category),
            masked_preview=f"[{category.value}:len={len(raw_match)}]",
            value_hmac=self._hmac_fn(raw_match) if self._hmac_fn else "",
            metadata=metadata or {},
        )

    def detect(self, segment: NormalizedSegment, raw_text: str) -> list[Detection]:
        """Detect Russian PII, financial identifiers, contacts, and custom patterns."""
        _ = raw_text
        text = segment.text
        if not text or not text.strip():
            return []

        text_lower = text.lower()
        results: list[Detection] = []

        snils_spans: set[tuple[int, int]] = set()
        ogrn_spans: set[tuple[int, int]] = set()
        oms_spans: set[tuple[int, int]] = set()
        dl_spans: set[tuple[int, int]] = set()
        passport_spans: set[tuple[int, int]] = set()

        # 1. SNILS
        for m in _SNILS_RE.finditer(text):
            raw = m.group(0)
            digits = _digits(raw)
            if not is_valid_snils(digits):
                continue
            win = _context_window(text_lower, m.start(), m.end())
            has_snils_ctx = bool(_SNILS_POS_RE.search(win))
            is_bare = len(raw) == 11
            if is_bare and digits[:2] in {"79", "89"} and not has_snils_ctx:
                if _PHONE_POS_RE.search(win):
                    continue
                # Bare 11-digit 89.../79... without SNILS context is likely a mobile number.
                continue
            conf = 0.95 if not is_bare else 0.85
            if has_snils_ctx:
                conf = 0.99
            snils_spans.add((m.start(), m.end()))
            results.append(
                self._build_detection(segment, EntityCategory.RU_SNILS, m.start(), m.end(), conf)
            )

        # 2. OGRN (13) / OGRNIP (15)
        for m in _OGRN_RE.finditer(text):
            digits = m.group(0)
            if not is_valid_ogrn(digits):
                continue
            win = _context_window(text_lower, m.start(), m.end())
            has_ogrn_ctx = bool(_OGRN_POS_RE.search(win))
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

        # 3. OMS (16 digits + context)
        for m in _OMS_RE.finditer(text):
            win = _context_window(text_lower, m.start(), m.end())
            if not _OMS_POS_RE.search(win):
                continue
            conf = 0.96 if "омс" in win or "oms" in win else 0.92
            oms_spans.add((m.start(), m.end()))
            results.append(
                self._build_detection(segment, EntityCategory.RU_OMS, m.start(), m.end(), conf)
            )

        # 4. Bank cards (13-19 digits, Luhn + brand/context boost)
        for m in _BANK_CARD_RE.finditer(text):
            span = (m.start(), m.end())
            raw = m.group(0)
            digits = _digits(raw)
            win = _context_window(text_lower, m.start(), m.end())
            has_card_ctx = bool(_CARD_POS_RE.search(win))

            if span in oms_spans and not has_card_ctx:
                continue
            if span in ogrn_spans and not has_card_ctx and " " not in raw and "-" not in raw:
                continue
            if not is_valid_luhn(digits):
                continue

            brand = _detect_card_brand(digits)
            is_formatted = " " in raw or "-" in raw
            conf = 0.88 if is_formatted else 0.72
            if brand is not None:
                conf += 0.08
            if has_card_ctx:
                conf += 0.10
            meta = {"brand": brand} if brand is not None else None
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

        # 5. BIK + 20-digit checking/correspondent account
        biks = [(m.start(), m.end(), m.group(0)) for m in _BIK_RE.finditer(text)]
        if biks:
            for m in _ACCOUNT_RE.finditer(text):
                acc_digits = _digits(m.group(0))
                if len(acc_digits) != 20:
                    continue
                acc_center = (m.start() + m.end()) // 2
                nearby_biks = sorted(
                    biks,
                    key=lambda b: abs(((b[0] + b[1]) // 2) - acc_center),
                )
                matched_bik = False
                for b_start, b_end, bik_val in nearby_biks:
                    dist = max(0, max(b_start - m.end(), m.start() - b_end))
                    if dist > _BIK_ACCOUNT_PROXIMITY:
                        break
                    if is_valid_cbr_bik_account(bik_val, acc_digits):
                        matched_bik = True
                        break
                if matched_bik:
                    results.append(
                        self._build_detection(
                            segment,
                            EntityCategory.RU_BIK_ACCOUNT,
                            m.start(),
                            m.end(),
                            0.98,
                        )
                    )

        # 6. Driver license (requires explicit DL context)
        for m in _DRIVER_LICENSE_RE.finditer(text):
            win = _context_window(text_lower, m.start(), m.end())
            if not _DL_POS_RE.search(win):
                continue
            dl_spans.add((m.start(), m.end()))
            conf = 0.95 if ("права" in win or "водител" in win or "в/у" in win) else 0.90
            results.append(
                self._build_detection(
                    segment, EntityCategory.RU_DRIVER_LICENSE, m.start(), m.end(), conf
                )
            )

        # 7. Russian Passport (4-digit series + 6-digit number)
        for m in _PASSPORT_LABELED_RE.finditer(text):
            series = _digits(m.group(1))
            if not _is_valid_passport_series(series):
                continue
            win = _context_window(text_lower, m.start(), m.end())
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
            if any(ps <= span[0] and span[1] <= pe for ps, pe in passport_spans):
                continue
            raw = m.group(0)
            digits = _digits(raw)
            if not _is_valid_passport_series(digits[:4]):
                continue
            win = _context_window(text_lower, m.start(), m.end())
            has_pos = bool(_PASSPORT_POS_RE.search(win))
            has_weak = bool(_PASSPORT_WEAK_POS_RE.search(win))
            has_neg = bool(_PASSPORT_NEG_RE.search(win))

            if span in dl_spans and not has_pos:
                continue
            if has_neg and not has_pos:
                continue

            is_bare_10 = len(raw) == 10
            if is_bare_10 and not has_pos:
                continue

            if has_pos:
                conf = 0.97 if not is_bare_10 else 0.90
            elif has_weak and not is_bare_10:
                conf = 0.85
            elif " " in raw or "-" in raw:
                # Formatted 2+2+6 or 4+6 with valid region/year and no negative order context
                conf = 0.80 if raw.count(" ") == 2 or raw.count("-") == 2 else 0.75
            else:
                continue

            passport_spans.add(span)
            results.append(
                self._build_detection(segment, EntityCategory.RU_PASSPORT, m.start(), m.end(), conf)
            )

        # 8. INN (10 / 12 digits)
        for m in _INN_RE.finditer(text):
            span = (m.start(), m.end())
            digits = m.group(0)
            if not is_valid_inn(digits):
                continue
            win = _context_window(text_lower, m.start(), m.end())
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
            else:
                conf = 0.96 if has_inn_ctx else 0.78
            results.append(
                self._build_detection(segment, EntityCategory.RU_INN, m.start(), m.end(), conf)
            )

        # 9. Russian Phone (+7 / 8)
        for m in _PHONE_RE.finditer(text):
            span = (m.start(), m.end())
            win = _context_window(text_lower, m.start(), m.end())
            if span in snils_spans and _SNILS_POS_RE.search(win) and not _PHONE_POS_RE.search(win):
                continue
            raw = m.group(0)
            digits = _digits(raw)
            if len(digits) != 11:
                continue
            is_formatted = raw.startswith("+") or "(" in raw or " " in raw or "-" in raw
            conf = 0.94 if is_formatted else 0.84
            if _PHONE_POS_RE.search(win):
                conf = min(0.99, conf + 0.05)
            results.append(
                self._build_detection(segment, EntityCategory.RU_PHONE, m.start(), m.end(), conf)
            )

        # 10. Russian Auto Plate
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

        # 11. Email
        for m in _EMAIL_RE.finditer(text):
            if not _is_valid_email(m.group(0)):
                continue
            results.append(
                self._build_detection(segment, EntityCategory.EMAIL, m.start(), m.end(), 0.95)
            )

        # 12. IPv4
        for m in _IPV4_RE.finditer(text):
            win = _context_window(text_lower, m.start(), m.end())
            conf = 0.95 if _IP_POS_RE.search(win) else 0.85
            results.append(
                self._build_detection(segment, EntityCategory.IP_ADDRESS, m.start(), m.end(), conf)
            )

        # 13. User custom patterns
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
                        metadata={"rule_name": rule.name},
                    )
                )

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
