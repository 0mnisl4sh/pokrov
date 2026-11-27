"""Anti-evasion text normalizer for Pokrov.

Handles invisible/bidi/tag/variation-selector/combining character stripping,
Unicode whitespace/dash/digit/fullwidth canonicalization, context-aware
Cyrillic/Latin/Greek and numeric-lookalike homoglyph normalization, bounded
recursive decoding of URL/hex/Base64 (including RFC 2045 multiline Base64)
blobs, and split-digit normalization while preserving exact character back-mappings.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from collections import deque
from dataclasses import dataclass
from urllib.parse import unquote_to_bytes

from pokrov.core.contracts import (
    NormalizationResult,
    NormalizedSegment,
    SpanSource,
)

# ---------------------------------------------------------------------------
# Character sets & precompiled linear-time patterns
# ---------------------------------------------------------------------------

# Zero-width, bidi controls, soft hyphen, combining diacritics (U+0300..U+036F;
# precomposed NFC й/ё are U+0439/U+0451 and remain untouched), CGJ (U+034F),
# Hangul/Mongolian fillers, Variation Selectors (U+FE00..U+FE0F, U+E0100..U+E01EF),
# and Unicode Tags (U+E0000..U+E007F).
_INVISIBLE_RE = re.compile(
    r"[\u00ad\u0300-\u036f\u034f\u061c\u115f\u1160\u180b-\u180e"
    r"\u200b-\u200f\u202a-\u202e\u2060-\u206f"
    r"\u3164\ufe00-\ufe0f\ufeff\uffa0"
    r"\U000e0000-\U000e007f\U000e0100-\U000e01ef]+"
)

_BIDI_OVERRIDE_CHARS: frozenset[str] = frozenset(
    "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)

_UNICODE_TAG_RUN_RE = re.compile(r"[\U000e0009\U000e000a\U000e000d\U000e0020-\U000e007e]{6,}")
_ROT13_HINT_RE = re.compile(r"\brot[\s_-]?13\b", re.IGNORECASE)
_ROT13_TRANS = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
)

# Non-ASCII Unicode Zs space separators -> ASCII ' ' (1:1 index_map)
_UNICODE_SPACES: frozenset[str] = frozenset(
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
)

# Unicode hyphen/dash/minus variants -> ASCII '-' (1:1 index_map)
_UNICODE_DASHES: frozenset[str] = frozenset("\u2010\u2011\u2012\u2013\u2014\u2015\u2212")

# Unicode No (superscript, subscript, circled, dingbat) single digits 0-9 -> ASCII '0'-'9'
_UNICODE_NO_DIGITS: dict[str, str] = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    **{chr(0x2074 + i): str(i + 4) for i in range(6)},
    **{chr(0x2080 + i): str(i) for i in range(10)},
    "\u24ea": "0",
    "\u24ff": "0",
    **{chr(0x2460 + i): str(i + 1) for i in range(9)},
    **{chr(0x24F5 + i): str(i + 1) for i in range(9)},
    **{chr(0x2776 + i): str(i + 1) for i in range(9)},
    **{chr(0x2780 + i): str(i + 1) for i in range(9)},
    **{chr(0x278A + i): str(i + 1) for i in range(9)},
}

_UNICODE_CANON_RE = re.compile(
    r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000"
    r"\u2010-\u2015\u2212"
    r"\uff21-\uff3a\uff41-\uff5a"
    r"\u00b2\u00b3\u00b9\u2070\u2074-\u2079"
    r"\u2080-\u2089"
    r"\u2460-\u2468\u24ea\u24f5-\u24fd\u24ff"
    r"\u2776-\u277e\u2780-\u2788\u278a-\u2792"
    r"]|(?![0-9])\d"
)

# Candidate tokens for script-mixing or numeric-lookalike homoglyph normalization.
# Strictly linear-time: disjoint prefixes anchored at token start, greedy tail [^\W_]*.
_CYR_CLASS = r"А-Яа-яЁёІіЅѕЈјӀӏ"
_GRK_CLASS = r"Α-Ωα-ω"
_EXT_SCRIPT_CLASS = _CYR_CLASS + _GRK_CLASS

_HOMOGLYPH_TOKEN_RE = re.compile(
    r"(?<![^\W_])(?:"
    r"[A-Za-z0-9]+[" + _EXT_SCRIPT_CLASS + r"][^\W_]*"
    r"|[" + _EXT_SCRIPT_CLASS + r"]+[A-Za-z0-9][^\W_]*"
    r"|[" + _CYR_CLASS + r"]+[" + _GRK_CLASS + r"][^\W_]*"
    r"|[" + _GRK_CLASS + r"]+[" + _CYR_CLASS + r"][^\W_]*"
    r"|(?:[0-9]+[OolI]|[OolI]+[0-9])[^\W_]*"
    r")"
)

_ASCII_DIGIT_LOOKALIKE_TOKEN_RE = re.compile(r"(?<![^\W_])(?:[0-9]+[OolI]|[OolI]+[0-9])[^\W_]*")

# Symmetric Cyrillic <-> Latin confusables (including lowercase к <-> k)
_CYR_CHARS = "АВЕКМНОРСТХУаекорсух"
_LAT_CHARS = "ABEKMHOPCTXYaekopcyx"

_CYR_TO_LAT: dict[str, str] = {
    **dict(zip(_CYR_CHARS, _LAT_CHARS, strict=True)),
    "І": "I",
    "і": "i",
    "Ѕ": "S",
    "ѕ": "s",
    "Ј": "J",
    "ј": "j",
}
_LAT_TO_CYR: dict[str, str] = dict(zip(_LAT_CHARS, _CYR_CHARS, strict=True))

# Greek confusables -> Latin
_GRK_CHARS = "ΑΒΕΖΗΙΚΜΝΟΡΤΧΥαεικορυ"
_GRK_LAT_CHARS = "ABEZHIKMNOPTXYaeikopu"
_GRK_TO_LAT: dict[str, str] = dict(zip(_GRK_CHARS, _GRK_LAT_CHARS, strict=True))

_CYR_ALL_SET: frozenset[str] = frozenset(
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяІіЅѕЈјӀӏ"
)
_LAT_ALL_SET: frozenset[str] = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_GRK_ALL_SET: frozenset[str] = frozenset("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρςστυφχψω")

# Non-Latin characters whose shapes mimic ASCII digits in obfuscated numeric tokens
_NON_LAT_DIGIT_LOOKALIKES: dict[str, str] = {
    "О": "0",
    "о": "0",
    "З": "3",
    "з": "3",
    "І": "1",
    "і": "1",
    "Ӏ": "1",
    "ӏ": "1",
    "Ο": "0",
    "ο": "0",
    "Ι": "1",
    "ι": "1",
}

# Latin characters whose shapes mimic ASCII digits inside numeric-dominant tokens
_LAT_DIGIT_LOOKALIKES: dict[str, str] = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
}

_ALL_DIGIT_LOOKALIKES: dict[str, str] = {
    **_NON_LAT_DIGIT_LOOKALIKES,
    **_LAT_DIGIT_LOOKALIKES,
}

# Preserve genuine Russian vehicle registration plates (Cyrillic or Latin lookalike letters)
_RU_AUTO_PLATE_RE = re.compile(
    r"^[АВЕКМНОРСТУХABEKMHOPCTYX]\d{3}[АВЕКМНОРСТУХABEKMHOPCTYX]{2}(?:\d{2,3})?$",
    re.IGNORECASE,
)

# Encoded blob candidate finders (strictly linear-time, no nested backtracking)
_PCT_PAIR_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_URL_STOP_CHARS: frozenset[str] = frozenset(" \t\r\n\"'<>()[]{}")

_HEX_CONTIGUOUS_RE = re.compile(r"(?<![A-Za-z0-9])(?:0[xX])?[0-9A-Fa-f]{8,}(?![A-Za-z0-9])")
_HEX_ESCAPED_RE = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){4,}")
_HEX_SPACED_RE = re.compile(r"(?<![A-Za-z0-9])(?:[0-9A-Fa-f]{2} ){3,}[0-9A-Fa-f]{2}(?![A-Za-z0-9])")
_HEX_LETTER_RE = re.compile(r"[A-Fa-f]")

# Possessive quantifiers (Python 3.11+) guarantee strict O(N) non-backtracking multiline scan
_B64_MULTILINE_STANDARD_RE = re.compile(
    r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}+"
    r"(?:\r?\n[ \t]*+[A-Za-z0-9+/]{2,}+)++={0,2}+(?![A-Za-z0-9+/=])"
)
_B64_MULTILINE_URLSAFE_RE = re.compile(
    r"(?<![A-Za-z0-9_=-])[A-Za-z0-9_-]{16,}+"
    r"(?:\r?\n[ \t]*+[A-Za-z0-9_-]{2,}+)++={0,2}+(?![A-Za-z0-9_=-])"
)
_B64_LINE_SPLIT_RE = re.compile(r"\r?\n[ \t]*")
_B64_WS_STRIP_RE = re.compile(r"[\r\n \t]+")
_B64_NON_DIGIT_RE = re.compile(r"[A-Za-z+/=_-]")

_B64_URLSAFE_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}={0,2}(?![A-Za-z0-9_=-])")
_B64_STANDARD_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{8,}={0,2}(?![A-Za-z0-9+/=])")

# Split/spaced digit separator matcher: ASCII/Unicode spaces, tabs, and dash variants between digits
_SEP_CHARS_CLASS = r" \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000\-\u2010-\u2015\u2212"
_DIGIT_SEP_RE = re.compile(rf"(?<=\d)[{_SEP_CHARS_CLASS}]+(?=\d)")
_SPLIT_3PLUS_TRIGGER_RE = re.compile(
    rf"\d\d[{_SEP_CHARS_CLASS}]+\d"
    rf"|\d[{_SEP_CHARS_CLASS}]+\d\d"
    rf"|\d[{_SEP_CHARS_CLASS}]+\d[{_SEP_CHARS_CLASS}]+\d"
)
_DASH_CHARS_SET: frozenset[str] = frozenset("-\u2010\u2011\u2012\u2013\u2014\u2015\u2212")

_ALLOWED_EXTRA_CHARS: frozenset[str] = frozenset("\t\r\n№₽«»—–•…°")


@dataclass(frozen=True, slots=True)
class _DecodeCandidate:
    start: int
    end: int
    decoded_text: str
    decoded_bytes: int
    technique: str
    oversized: bool = False


@dataclass(frozen=True, slots=True)
class _QueueItem:
    text: str
    depth: int
    outer_start: int
    outer_end: int
    techniques: tuple[str, ...]


def _dedup_tuple(items: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return unique items preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _strip_invisible(text: str) -> tuple[str, tuple[int, ...], tuple[str, ...]]:
    """Strip invisible, tag, variation-selector, combining, and bidi chars with ``index_map``."""
    if not text:
        return ("", (), ())

    if text.isascii():
        return (text, tuple(range(len(text))), ())

    first = _INVISIBLE_RE.search(text)
    if first is None:
        return (text, tuple(range(len(text))), ())

    chunks: list[str] = []
    indices: list[int] = []
    has_zero_width = False
    has_bidi = False
    cursor = 0

    for match in _INVISIBLE_RE.finditer(text, first.start()):
        start, end = match.span()
        if start > cursor:
            chunks.append(text[cursor:start])
            indices.extend(range(cursor, start))
        for ch in match.group(0):
            if ch in _BIDI_OVERRIDE_CHARS:
                has_bidi = True
            else:
                has_zero_width = True
        cursor = end

    if cursor < len(text):
        chunks.append(text[cursor:])
        indices.extend(range(cursor, len(text)))

    techniques: list[str] = []
    if has_zero_width:
        techniques.append("zero_width")
    if has_bidi:
        techniques.append("bidi_override")

    return ("".join(chunks), tuple(indices), tuple(techniques))


def _canonicalize_unicode_chars(text: str) -> tuple[str, bool, bool]:
    """Map Unicode Zs spaces, dashes, Nd/No digits, and fullwidth Latin 1:1 (preserving index_map).

    Returns ``(canonical_text, has_unicode_digits, has_fullwidth_alpha)``.
    """
    if not text or text.isascii():
        return (text, False, False)

    first = _UNICODE_CANON_RE.search(text)
    if first is None:
        return (text, False, False)

    chunks: list[str] = []
    cursor = 0
    has_unicode_digits = False
    has_fullwidth_alpha = False

    for match in _UNICODE_CANON_RE.finditer(text, first.start()):
        start, end = match.span()
        ch = match.group(0)
        if ch in _UNICODE_SPACES:
            repl = " "
        elif ch in _UNICODE_DASHES:
            repl = "-"
        elif ("\uff21" <= ch <= "\uff3a") or ("\uff41" <= ch <= "\uff5a"):
            repl = chr(ord(ch) - 0xFEE0)
            has_fullwidth_alpha = True
        elif ch in _UNICODE_NO_DIGITS:
            repl = _UNICODE_NO_DIGITS[ch]
            has_unicode_digits = True
        else:
            digit_val = unicodedata.digit(ch, -1)
            if 0 <= digit_val <= 9:
                repl = chr(48 + digit_val)
                has_unicode_digits = True
            else:
                continue

        if start > cursor:
            chunks.append(text[cursor:start])
        chunks.append(repl)
        cursor = end

    if cursor == 0:
        return (text, False, False)
    if cursor < len(text):
        chunks.append(text[cursor:])
    return ("".join(chunks), has_unicode_digits, has_fullwidth_alpha)


def _normalize_token_homoglyphs(token: str) -> str | None:
    """Normalize a single alphanumeric token if it exhibits script mixing or digit spoofing.

    Returns ``None`` when the token is already clean (avoids allocation on the fast path).
    """
    cyr_count = 0
    lat_count = 0
    grk_count = 0
    digit_count = 0
    uniq_cyr = 0
    uniq_lat = 0
    uniq_grk = 0
    other_alpha_in_num = False
    non_lat_digit_lookalikes = 0
    lat_digit_lookalikes = 0

    for ch in token:
        if "0" <= ch <= "9":
            digit_count += 1
        elif ch in _CYR_ALL_SET:
            cyr_count += 1
            if ch not in _CYR_TO_LAT:
                uniq_cyr += 1
            if ch in _NON_LAT_DIGIT_LOOKALIKES:
                non_lat_digit_lookalikes += 1
            else:
                other_alpha_in_num = True
        elif ch in _LAT_ALL_SET:
            lat_count += 1
            if ch not in _LAT_TO_CYR:
                uniq_lat += 1
            if ch in _LAT_DIGIT_LOOKALIKES:
                lat_digit_lookalikes += 1
            else:
                other_alpha_in_num = True
        elif ch in _GRK_ALL_SET:
            grk_count += 1
            if ch not in _GRK_TO_LAT:
                uniq_grk += 1
            if ch in _NON_LAT_DIGIT_LOOKALIKES:
                non_lat_digit_lookalikes += 1
            else:
                other_alpha_in_num = True
        else:
            other_alpha_in_num = True

    # Case 1a: Numeric token spoofed with Cyrillic/Greek digit lookalikes (e.g. 45І2, 654З21, О1)
    if (
        digit_count > 0
        and non_lat_digit_lookalikes > 0
        and not other_alpha_in_num
        and not _RU_AUTO_PLATE_RE.match(token)
    ):
        return "".join(_ALL_DIGIT_LOOKALIKES.get(ch, ch) for ch in token)

    # Case 1b: Numeric-dominant token spoofed with Latin O/o -> 0 or l/I -> 1
    # (e.g. 451O, 65432l, 65432I, 2O3) without altering short flags (-O2, l1), hex, or auto plates.
    if (
        non_lat_digit_lookalikes == 0
        and lat_digit_lookalikes > 0
        and not other_alpha_in_num
        and len(token) >= 3
        and digit_count >= 2
        and (
            digit_count > lat_digit_lookalikes
            or (len(token) >= 4 and digit_count >= lat_digit_lookalikes)
        )
        and not _RU_AUTO_PLATE_RE.match(token)
    ):
        return "".join(_LAT_DIGIT_LOOKALIKES.get(ch, ch) for ch in token)

    # Case 2: Pure single-script tokens must never be altered
    scripts_present = (
        (1 if cyr_count > 0 else 0) + (1 if lat_count > 0 else 0) + (1 if grk_count > 0 else 0)
    )
    if scripts_present < 2:
        return None

    # Non-confusable Greek mixed in or bilingual identifier (e.g. clientПаспорт) -> keep intact
    if uniq_grk > 0 or (uniq_cyr > 0 and uniq_lat > 0):
        return None

    if uniq_lat > 0:
        target_latin = True
    elif uniq_cyr > 0:
        target_latin = False
    elif cyr_count == 0:
        target_latin = True
    else:
        # All letters are confusable across scripts; break ties by Latin+Greek vs Cyrillic
        target_latin = (lat_count + grk_count) >= cyr_count

    if target_latin:
        normalized = "".join(_CYR_TO_LAT.get(ch, _GRK_TO_LAT.get(ch, ch)) for ch in token)
    else:
        normalized = "".join(
            _LAT_TO_CYR.get(_GRK_TO_LAT.get(ch, ch), _GRK_TO_LAT.get(ch, ch)) for ch in token
        )
    return normalized if normalized != token else None


_HAS_HOMOGLYPH_TRIGGER_RE = re.compile(r"[A-Za-z0-9\u0370-\u03ff]")
_HEX_TRIGGER_RE = re.compile(r"(?:\\x|[0-9A-Fa-f]{4}|[0-9A-Fa-f]{2} [0-9A-Fa-f]{2})")


def _normalize_homoglyphs(text: str) -> tuple[str, bool]:
    """Normalize Cyrillic/Latin/Greek & numeric homoglyphs without altering clean prose."""
    if not text:
        return (text, False)

    is_ascii = text.isascii()
    if not is_ascii and _HAS_HOMOGLYPH_TRIGGER_RE.search(text) is None:
        return (text, False)

    pattern = _ASCII_DIGIT_LOOKALIKE_TOKEN_RE if is_ascii else _HOMOGLYPH_TOKEN_RE
    first = pattern.search(text)
    if first is None:
        return (text, False)

    chunks: list[str] = []
    cursor = 0
    changed = False
    token_cache: dict[str, str | None] = {}

    for match in pattern.finditer(text, first.start()):
        token = match.group(0)
        if len(token) > 48:
            continue
        if token in token_cache:
            replacement = token_cache[token]
        else:
            replacement = _normalize_token_homoglyphs(token)
            if len(token) <= 64 and len(token_cache) < 2048:
                token_cache[token] = replacement
        if replacement is not None:
            start, end = match.span()
            if start > cursor:
                chunks.append(text[cursor:start])
            chunks.append(replacement)
            cursor = end
            changed = True

    if not changed:
        return (text, False)

    if cursor < len(text):
        chunks.append(text[cursor:])
    return ("".join(chunks), True)


def _normalize_text_stage(text: str) -> tuple[str, tuple[int, ...], tuple[str, ...]]:
    """Run invisible stripping, Unicode canonicalization, and homoglyph normalization."""
    stripped, index_map, strip_techniques = _strip_invisible(text)
    canon_text, has_unicode_digits, has_fullwidth_alpha = _canonicalize_unicode_chars(stripped)
    norm_text, homoglyph_changed = _normalize_homoglyphs(canon_text)

    techniques = list(strip_techniques)
    if has_unicode_digits:
        techniques.append("unicode_digits")
    if homoglyph_changed or has_fullwidth_alpha:
        techniques.append("homoglyph")

    return (norm_text, index_map, _dedup_tuple(techniques))


def _is_meaningful_decoded_text(decoded: str, *, allow_partial_tail: bool = False) -> bool:
    """Return True if ``decoded`` is printable, valid text rather than random byte noise."""
    if not decoded:
        return False

    total = 0
    standard_chars = 0
    has_alnum = False

    for ch in decoded:
        code = ord(ch)
        if code < 0x20:
            if ch not in ("\t", "\n", "\r"):
                return False
        elif 0x7F <= code <= 0x9F:
            return False

        total += 1
        if ch.isalnum():
            has_alnum = True

        if (0x20 <= code <= 0x7E) or (0x0400 <= code <= 0x04FF) or ch in _ALLOWED_EXTRA_CHARS:
            standard_chars += 1

    if not has_alnum and not allow_partial_tail:
        return False

    return (standard_chars / total) >= 0.85


def _decode_utf8_prefix_check(raw_bytes: bytes) -> str | None:
    """Validate UTF-8 for a bounded prefix (trimming up to 3 split trailing bytes)."""
    for trim in range(min(4, len(raw_bytes))):
        chunk = raw_bytes if trim == 0 else raw_bytes[:-trim]
        if not chunk:
            return None
        try:
            text = chunk.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        if _is_meaningful_decoded_text(text, allow_partial_tail=True):
            return text
        return None
    return None


def _overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    """Return True if ``[start, end)`` overlaps any interval in ``occupied``."""
    return any(start < occ_end and end > occ_start for occ_start, occ_end in occupied)


def _scan_url_candidates(
    text: str,
    remaining_bytes: int,
    occupied: list[tuple[int, int]],
    max_candidates: int,
) -> list[_DecodeCandidate]:
    """Detect valid UTF-8 ``%XX`` URL-encoded substrings in linear time."""
    candidates: list[_DecodeCandidate] = []
    n = len(text)
    pos = 0

    while pos < n and len(candidates) < max_candidates:
        match = _PCT_PAIR_RE.search(text, pos)
        if match is None:
            break

        left = match.start()
        while left > 0 and text[left - 1] not in _URL_STOP_CHARS:
            left -= 1

        right = match.end()
        while right < n and text[right] not in _URL_STOP_CHARS:
            right += 1

        pos = right
        if _overlaps(left, right, occupied):
            continue

        raw_span = text[left:right]
        pct_count = len(_PCT_PAIR_RE.findall(raw_span))
        if len(raw_span) > 4096 and len(raw_span) > remaining_bytes * 3:
            sample = raw_span[:4096]
            last_pct = sample.rfind("%")
            if last_pct > len(sample) - 3:
                sample = sample[:last_pct]
            sample_bytes = unquote_to_bytes(sample)
            if _decode_utf8_prefix_check(sample_bytes) is not None:
                candidates.append(
                    _DecodeCandidate(
                        start=left,
                        end=right,
                        decoded_text="",
                        decoded_bytes=len(sample_bytes),
                        technique="url_encoding",
                        oversized=True,
                    )
                )
                occupied.append((left, right))
            continue

        decoded_bytes = unquote_to_bytes(raw_span)
        try:
            decoded_text = decoded_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue

        if decoded_text == raw_span:
            continue
        if pct_count < 2 and len(decoded_text) < 4:
            continue
        if not _is_meaningful_decoded_text(decoded_text):
            continue

        oversized = len(decoded_bytes) > remaining_bytes
        candidates.append(
            _DecodeCandidate(
                start=left,
                end=right,
                decoded_text="" if oversized else decoded_text,
                decoded_bytes=len(decoded_bytes),
                technique="url_encoding",
                oversized=oversized,
            )
        )
        occupied.append((left, right))

    return candidates


def _try_decode_hex_payload(
    hex_digits: str,
    *,
    has_prefix_or_sep: bool,
    remaining_bytes: int,
) -> tuple[str, int, bool] | None:
    """Decode validated hex digits into UTF-8 text or flag an oversized valid hex bomb."""
    if len(hex_digits) < 8 or len(hex_digits) % 2 != 0:
        return None

    byte_len = len(hex_digits) // 2
    if not has_prefix_or_sep and _HEX_LETTER_RE.search(hex_digits) is None:
        if len(hex_digits) < 16:
            return None
        sample_end = min(len(hex_digits), max(32, remaining_bytes * 2))
        sample_end -= sample_end % 2
        try:
            sample_bytes = bytes.fromhex(hex_digits[:sample_end])
        except ValueError:
            return None
        if not all(b == 0x20 or 0x30 <= b <= 0x39 for b in sample_bytes):
            return None

    if byte_len > remaining_bytes and len(hex_digits) > 4096:
        sample_end = min(len(hex_digits), 4096)
        sample_end -= sample_end % 2
        try:
            sample_bytes = bytes.fromhex(hex_digits[:sample_end])
        except ValueError:
            return None
        if _decode_utf8_prefix_check(sample_bytes) is not None:
            return ("", byte_len, True)
        return None

    try:
        raw_bytes = bytes.fromhex(hex_digits)
        decoded_text = raw_bytes.decode("utf-8", errors="strict")
    except (ValueError, UnicodeDecodeError):
        return None

    if not _is_meaningful_decoded_text(decoded_text):
        return None
    if (
        not has_prefix_or_sep
        and _HEX_LETTER_RE.search(hex_digits) is None
        and not all(ch == " " or ("0" <= ch <= "9") for ch in decoded_text)
    ):
        return None

    if len(raw_bytes) > remaining_bytes:
        return ("", len(raw_bytes), True)

    return (decoded_text, len(raw_bytes), False)


def _scan_hex_candidates(
    text: str,
    remaining_bytes: int,
    occupied: list[tuple[int, int]],
    max_candidates: int,
) -> list[_DecodeCandidate]:
    """Detect ``\\xXX``, space-separated, ``0x``-prefixed, and contiguous hex payloads."""
    candidates: list[_DecodeCandidate] = []
    invalid_tokens: set[str] = set()

    def _collect(
        pattern: re.Pattern[str],
        extract: str,
    ) -> None:
        for match in pattern.finditer(text):
            if len(candidates) >= max_candidates:
                return
            start, end = match.span()
            if _overlaps(start, end, occupied):
                continue
            raw_span = match.group(0)
            if len(raw_span) <= 256 and raw_span in invalid_tokens:
                continue

            if extract == "escaped":
                hex_digits = raw_span.replace("\\x", "")
                has_prefix_or_sep = True
            elif extract == "spaced":
                hex_digits = raw_span.replace(" ", "")
                has_prefix_or_sep = _HEX_LETTER_RE.search(hex_digits) is not None
                if not has_prefix_or_sep and len(hex_digits) < 12:
                    if len(raw_span) <= 256 and len(invalid_tokens) < 2048:
                        invalid_tokens.add(raw_span)
                    continue
            else:
                has_prefix_or_sep = raw_span.startswith(("0x", "0X"))
                hex_digits = raw_span[2:] if has_prefix_or_sep else raw_span

            res = _try_decode_hex_payload(
                hex_digits,
                has_prefix_or_sep=has_prefix_or_sep,
                remaining_bytes=remaining_bytes,
            )
            if res is None:
                if len(raw_span) <= 256 and len(invalid_tokens) < 2048:
                    invalid_tokens.add(raw_span)
                continue

            decoded_text, byte_len, oversized = res
            candidates.append(
                _DecodeCandidate(start, end, decoded_text, byte_len, "hex", oversized)
            )
            occupied.append((start, end))

    if "\\x" in text:
        _collect(_HEX_ESCAPED_RE, "escaped")
    if len(candidates) < max_candidates and " " in text:
        _collect(_HEX_SPACED_RE, "spaced")
    if len(candidates) < max_candidates:
        _collect(_HEX_CONTIGUOUS_RE, "contiguous")

    return candidates


def _try_decode_b64_token(
    token: str,
    *,
    urlsafe: bool,
    remaining_bytes: int,
) -> tuple[str, int, bool] | None:
    """Validate and decode a Base64 (standard or URL-safe) candidate token."""
    pad_count = len(token) - len(token.rstrip("="))
    core = token[:-pad_count] if pad_count else token
    if not core or len(core) % 4 == 1:
        return None

    expected_pad = (4 - len(core) % 4) % 4
    if pad_count and pad_count != expected_pad:
        return None

    # Pure-digit strings (e.g. 00000000, 11111111) or single-case alpha words without '='
    if core.isdigit():
        return None
    if not pad_count and core.isalpha() and (core.islower() or core.isupper()):
        return None

    estimated_bytes = (len(core) * 3) // 4
    standard_core = core.replace("-", "+").replace("_", "/") if urlsafe else core

    if estimated_bytes > remaining_bytes and len(standard_core) > 4096:
        prefix_chars = min(len(standard_core), 4096)
        prefix_chars -= prefix_chars % 4
        if prefix_chars >= 4:
            try:
                sample_bytes = base64.b64decode(standard_core[:prefix_chars], validate=True)
            except binascii.Error:
                return None
            if _decode_utf8_prefix_check(sample_bytes) is not None:
                return ("", estimated_bytes, True)
        return None

    padded = standard_core + ("=" * expected_pad)
    try:
        raw_bytes = base64.b64decode(padded, validate=True)
    except binascii.Error:
        return None

    # Enforce canonical trailing zero padding bits to eliminate accidental text matches
    canonical_core = base64.b64encode(raw_bytes).decode("ascii").rstrip("=")
    if canonical_core != standard_core:
        return None

    try:
        decoded_text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None

    if not _is_meaningful_decoded_text(decoded_text):
        return None

    if len(raw_bytes) > remaining_bytes:
        return ("", len(raw_bytes), True)

    return (decoded_text, len(raw_bytes), False)


def _split_multiline_spans(raw_block: str, base_start: int) -> list[tuple[str, int, int]]:
    """Split a multiline Base64 match into ``(line_text, abs_start, abs_end)`` entries."""
    lines: list[tuple[str, int, int]] = []
    cursor = 0
    for sep_match in _B64_LINE_SPLIT_RE.finditer(raw_block):
        s_start, s_end = sep_match.span()
        lines.append((raw_block[cursor:s_start], base_start + cursor, base_start + s_start))
        cursor = s_end
    lines.append((raw_block[cursor:], base_start + cursor, base_start + len(raw_block)))
    return lines


def _scan_b64_candidates(
    text: str,
    remaining_bytes: int,
    occupied: list[tuple[int, int]],
    max_candidates: int,
) -> list[_DecodeCandidate]:
    """Detect RFC 2045 multiline, URL-safe, and standard Base64 payloads in linear time."""
    candidates: list[_DecodeCandidate] = []
    invalid_tokens: set[str] = set()

    def _collect_multiline(pattern: re.Pattern[str], *, urlsafe: bool) -> None:
        for match in pattern.finditer(text):
            if len(candidates) >= max_candidates:
                return
            raw_block = match.group(0)
            if urlsafe and "-" not in raw_block and "_" not in raw_block:
                continue
            start, end = match.span()
            if _overlaps(start, end, occupied):
                continue

            joined = _B64_WS_STRIP_RE.sub("", raw_block)
            res = _try_decode_b64_token(joined, urlsafe=urlsafe, remaining_bytes=remaining_bytes)
            cand_start, cand_end = start, end

            # Recover if a leading/trailing non-wrapped label line was adjacent across \n
            if res is None:
                line_spans = _split_multiline_spans(raw_block, start)
                if len(line_spans) >= 3:
                    for lo, hi in (
                        (1, len(line_spans)),
                        (0, len(line_spans) - 1),
                        (1, len(line_spans) - 1),
                    ):
                        if hi - lo < 2:
                            continue
                        sub_joined = "".join(item[0] for item in line_spans[lo:hi])
                        if len(sub_joined) < 18:
                            continue
                        sub_start = line_spans[lo][1]
                        sub_end = line_spans[hi - 1][2]
                        if _overlaps(sub_start, sub_end, occupied):
                            continue
                        res = _try_decode_b64_token(
                            sub_joined, urlsafe=urlsafe, remaining_bytes=remaining_bytes
                        )
                        if res is not None:
                            cand_start, cand_end = sub_start, sub_end
                            break

            if res is None:
                continue

            decoded_text, byte_len, oversized = res
            candidates.append(
                _DecodeCandidate(cand_start, cand_end, decoded_text, byte_len, "base64", oversized)
            )
            occupied.append((cand_start, cand_end))

    def _collect_single(pattern: re.Pattern[str], *, urlsafe: bool) -> None:
        for match in pattern.finditer(text):
            if len(candidates) >= max_candidates:
                return
            token = match.group(0)
            if urlsafe and "-" not in token and "_" not in token:
                continue
            if (
                not urlsafe
                and not token.endswith("=")
                and (token.isdigit() or (token.isalpha() and (token.islower() or token.isupper())))
            ):
                continue
            if len(token) <= 256 and token in invalid_tokens:
                continue
            start, end = match.span()
            if _overlaps(start, end, occupied):
                continue

            res = _try_decode_b64_token(token, urlsafe=urlsafe, remaining_bytes=remaining_bytes)
            if res is None:
                if len(token) <= 256 and len(invalid_tokens) < 2048:
                    invalid_tokens.add(token)
                continue

            decoded_text, byte_len, oversized = res
            candidates.append(
                _DecodeCandidate(start, end, decoded_text, byte_len, "base64", oversized)
            )
            occupied.append((start, end))

    has_urlsafe_chars = "-" in text or "_" in text
    if "\n" in text:
        if has_urlsafe_chars:
            _collect_multiline(_B64_MULTILINE_URLSAFE_RE, urlsafe=True)
        if len(candidates) < max_candidates:
            _collect_multiline(_B64_MULTILINE_STANDARD_RE, urlsafe=False)

    if len(candidates) < max_candidates and has_urlsafe_chars:
        _collect_single(_B64_URLSAFE_RE, urlsafe=True)
    if len(candidates) < max_candidates:
        _collect_single(_B64_STANDARD_RE, urlsafe=False)

    return candidates


def _scan_unicode_tag_candidates(
    raw_text: str,
    remaining_bytes: int,
    max_candidates: int,
) -> list[_DecodeCandidate]:
    """Decode ASCII characters smuggled inside Unicode Tag codepoints (U+E0000..U+E007F)."""
    if not raw_text or raw_text.isascii() or remaining_bytes <= 0 or max_candidates < 1:
        return []
    candidates: list[_DecodeCandidate] = []
    for match in _UNICODE_TAG_RUN_RE.finditer(raw_text):
        if len(candidates) >= max_candidates:
            break
        raw_run = match.group(0)
        byte_len = len(raw_run)
        if byte_len > remaining_bytes:
            candidates.append(
                _DecodeCandidate(
                    start=match.start(),
                    end=match.end(),
                    decoded_text="",
                    decoded_bytes=byte_len,
                    technique="unicode_tags",
                    oversized=True,
                )
            )
            continue
        decoded = "".join(chr(ord(ch) - 0xE0000) for ch in raw_run)
        if _is_meaningful_decoded_text(decoded):
            candidates.append(
                _DecodeCandidate(
                    start=match.start(),
                    end=match.end(),
                    decoded_text=decoded,
                    decoded_bytes=byte_len,
                    technique="unicode_tags",
                    oversized=False,
                )
            )
    return candidates


def _scan_rot13_candidates(
    text: str,
    remaining_bytes: int,
    max_candidates: int,
) -> list[_DecodeCandidate]:
    """Decode text with explicit ``ROT13`` instruction hint under bounded byte cap."""
    if max_candidates < 1 or remaining_bytes <= 0:
        return []
    if ("rot" not in text and "ROT" not in text and "Rot" not in text) or (
        _ROT13_HINT_RE.search(text) is None
    ):
        return []
    oversized = len(text) > remaining_bytes
    payload = text[:remaining_bytes] if oversized else text
    decoded = payload.translate(_ROT13_TRANS)
    return [
        _DecodeCandidate(
            start=0,
            end=len(payload),
            decoded_text="" if oversized else decoded,
            decoded_bytes=len(decoded.encode("utf-8")),
            technique="rot13",
            oversized=oversized,
        )
    ]


def _find_encoded_candidates(
    text: str,
    remaining_bytes: int,
    max_candidates: int = 17,
) -> list[_DecodeCandidate]:
    """Find non-overlapping URL, hex, Base64, and ROT13 candidates sorted by start offset."""
    if len(text) < 4 or max_candidates < 1:
        return []

    occupied: list[tuple[int, int]] = []
    candidates: list[_DecodeCandidate] = []

    if "%" in text:
        candidates.extend(_scan_url_candidates(text, remaining_bytes, occupied, max_candidates))

    if _HEX_TRIGGER_RE.search(text) is not None:
        candidates.extend(_scan_hex_candidates(text, remaining_bytes, occupied, max_candidates))
    if _B64_NON_DIGIT_RE.search(text) is not None:
        candidates.extend(_scan_b64_candidates(text, remaining_bytes, occupied, max_candidates))
    if len(candidates) < max_candidates:
        candidates.extend(
            _scan_rot13_candidates(text, remaining_bytes, max_candidates - len(candidates))
        )

    candidates.sort(key=lambda c: c.start)
    return candidates


def collapse_split_digits(
    source: str | NormalizedSegment,
    *,
    min_digits: int = 2,
) -> NormalizedSegment:
    """Collapse spaces and hyphens separating digits while preserving raw back-mapping.

    Useful for catching split-digit obfuscation such as ``"45 12 - 654 321"`` or
    ``"4 5 1 2  6 5 4 3 2 1"`` while keeping list indices such as
    ``"Строка 1 - 2202 20 06 1234 56 76"`` separate from the following identifier.
    """
    if isinstance(source, NormalizedSegment):
        text = source.text
        base_map = source.index_map
        span_source = source.span_source
        outer_start = source.outer_start
        outer_end = source.outer_end
        evasion_techniques = source.evasion_techniques
    else:
        text = source
        base_map = ()
        span_source = SpanSource.DIRECT
        outer_start = 0
        outer_end = 0
        evasion_techniques = ()

    if not text:
        return NormalizedSegment(
            text="",
            span_source=span_source,
            index_map=(),
            outer_start=outer_start,
            outer_end=outer_end,
            evasion_techniques=evasion_techniques,
        )

    spans_to_drop = _find_split_digit_drops(text, min_digits)
    if not spans_to_drop:
        final_map = base_map if base_map else tuple(range(len(text)))
        return NormalizedSegment(
            text=text,
            span_source=span_source,
            index_map=final_map,
            outer_start=outer_start,
            outer_end=outer_end,
            evasion_techniques=evasion_techniques,
        )

    chunks: list[str] = []
    indices: list[int] = []
    cursor = 0
    for drop_start, drop_end in spans_to_drop:
        if drop_start > cursor:
            chunks.append(text[cursor:drop_start])
            indices.extend(base_map[cursor:drop_start] if base_map else range(cursor, drop_start))
        cursor = drop_end

    if cursor < len(text):
        chunks.append(text[cursor:])
        indices.extend(base_map[cursor:] if base_map else range(cursor, len(text)))

    return NormalizedSegment(
        text="".join(chunks),
        span_source=span_source,
        index_map=tuple(indices),
        outer_start=outer_start,
        outer_end=outer_end,
        evasion_techniques=evasion_techniques,
    )


def _is_spaced_dash_sep(sep: str) -> bool:
    """Return True if ``sep`` contains both whitespace and a dash/hyphen."""
    has_ws = False
    has_dash = False
    for ch in sep:
        if ch in _DASH_CHARS_SET:
            has_dash = True
        else:
            has_ws = True
    return has_ws and has_dash


def _emit_group_drops(
    text: str,
    group: list[tuple[int, int]],
    min_digits: int,
    out: list[tuple[int, int]],
) -> None:
    """Append valid separator spans from ``group`` without merging list indices (`1 - ...`)."""
    if not group:
        return

    # Guard (P1-03): never merge a leading 1-2 digit list/row index separated by ' - '
    # from a complete (>= 9-digit) single or multi-group identifier that follows.
    first_start, first_end = group[0]
    if _is_spaced_dash_sep(text[first_start:first_end]):
        left = first_start
        while left > 0 and text[left - 1].isdigit():
            left -= 1
        left_len = first_start - left
        if 1 <= left_len <= 2:
            n = len(text)
            right = group[-1][1]
            while right < n and text[right].isdigit():
                right += 1
            trailing_digits = sum(1 for ch in text[first_end:right] if ch.isdigit())
            if trailing_digits >= 9:
                group = group[1:]
                if not group:
                    return

    if _count_group_digits(text, group) >= min_digits:
        out.extend(group)


def _find_split_digit_drops(text: str, min_digits: int) -> list[tuple[int, int]]:
    """Return separator spans inside digit groups that meet ``min_digits``."""
    if min_digits >= 3 and _SPLIT_3PLUS_TRIGGER_RE.search(text) is None:
        return []

    first = _DIGIT_SEP_RE.search(text)
    if first is None:
        return []

    spans_to_drop: list[tuple[int, int]] = []
    it = _DIGIT_SEP_RE.finditer(text, first.start())
    first_m = next(it, None)
    if first_m is None:
        return []

    group: list[tuple[int, int]] = [first_m.span()]
    for m in it:
        m_start, m_end = m.span()
        prev_end = group[-1][1]
        if text[prev_end:m_start].isdigit():
            group.append((m_start, m_end))
        else:
            if min_digits <= 2 or len(group) >= 2:
                _emit_group_drops(text, group, min_digits, spans_to_drop)
            else:
                # Single-separator fast path when min_digits >= 3 (avoids O(M) scans on '1.1.1.1 ')
                g_start, g_end = group[0]
                left = g_start - 1
                while left >= 0 and text[left].isdigit():
                    left -= 1
                right = g_end
                n = len(text)
                while right < n and text[right].isdigit():
                    right += 1
                if (g_start - 1 - left) + (right - g_end) >= min_digits:
                    _emit_group_drops(text, group, min_digits, spans_to_drop)
            group = [(m_start, m_end)]
    _emit_group_drops(text, group, min_digits, spans_to_drop)
    return spans_to_drop


def _count_group_digits(text: str, group: list[tuple[int, int]]) -> int:
    """Count digits belonging to a chained digit-separator span group."""
    left = group[0][0]
    while left > 0 and text[left - 1].isdigit():
        left -= 1
    right = group[-1][1]
    n = len(text)
    while right < n and text[right].isdigit():
        right += 1
    return sum(1 for ch in text[left:right] if ch.isdigit())


class AntiEvasionNormalizer:
    """Anti-evasion normalizer implementing the ``TextNormalizer`` Protocol."""

    def __init__(
        self,
        max_depth: int = 3,
        max_decoded_bytes: int = 65536,
        max_segments: int = 16,
    ) -> None:
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        if max_decoded_bytes < 1:
            raise ValueError(f"max_decoded_bytes must be >= 1, got {max_decoded_bytes}")
        if max_segments < 0:
            raise ValueError(f"max_segments must be >= 0, got {max_segments}")

        self._max_depth = max_depth
        self._max_decoded_bytes = max_decoded_bytes
        self._max_segments = max_segments

    @property
    def max_depth(self) -> int:
        """Maximum recursive decoding depth."""
        return self._max_depth

    @property
    def max_decoded_bytes(self) -> int:
        """Total decoded byte budget across all segments."""
        return self._max_decoded_bytes

    @property
    def max_segments(self) -> int:
        """Maximum number of emitted decoded segments."""
        return self._max_segments

    def collapse_split_digits(
        self,
        source: str | NormalizedSegment,
        *,
        min_digits: int = 2,
    ) -> NormalizedSegment:
        """Collapse spaces and hyphens between digits while preserving back-mapping."""
        return collapse_split_digits(source, min_digits=min_digits)

    def normalize(self, text: str) -> NormalizationResult:
        """Normalize raw input text and extract bounded recursively decoded blobs."""
        primary_text, index_map, primary_techniques = _normalize_text_stage(text)

        primary = NormalizedSegment(
            text=primary_text,
            span_source=SpanSource.DIRECT,
            index_map=index_map,
            evasion_techniques=primary_techniques,
        )

        decoded_segments, all_extra_techniques, truncated = self._decode_bounded(
            primary, raw_text=text
        )

        combined_techniques = _dedup_tuple([*primary_techniques, *all_extra_techniques])
        return NormalizationResult(
            original_length=len(text),
            primary=primary,
            decoded_segments=decoded_segments,
            evasion_detected=bool(combined_techniques) or truncated,
            evasion_techniques=combined_techniques,
            truncated=truncated,
        )

    def _decode_bounded(
        self,
        primary: NormalizedSegment,
        *,
        raw_text: str = "",
    ) -> tuple[tuple[NormalizedSegment, ...], tuple[str, ...], bool]:
        """Breadth-first bounded decoding of embedded encoded blobs."""
        source_raw = raw_text or primary.text
        if not primary.text and source_raw.isascii():
            return ((), (), False)

        segments: list[NormalizedSegment] = []
        extra_techniques: list[str] = []
        truncated = False
        used_bytes = 0

        queue: deque[_QueueItem] = deque()
        initial_limit = max(self._max_segments + 1, 1)
        tag_candidates = _scan_unicode_tag_candidates(
            source_raw,
            self._max_decoded_bytes,
            initial_limit,
        )
        initial_candidates = (
            _find_encoded_candidates(
                primary.text,
                self._max_decoded_bytes,
                max_candidates=initial_limit,
            )
            if primary.text
            else []
        )

        def _enqueue_initial(cand: _DecodeCandidate, outer_start: int, outer_end: int) -> bool:
            nonlocal truncated, used_bytes
            if self._max_depth == 0 or self._max_segments == 0:
                extra_techniques.append(cand.technique)
                truncated = True
                return True
            if cand.oversized or used_bytes + cand.decoded_bytes > self._max_decoded_bytes:
                extra_techniques.append(cand.technique)
                truncated = True
                return True
            if len(queue) >= self._max_segments:
                extra_techniques.append(cand.technique)
                truncated = True
                return False
            used_bytes += cand.decoded_bytes
            queue.append(
                _QueueItem(
                    text=cand.decoded_text,
                    depth=1,
                    outer_start=outer_start,
                    outer_end=outer_end,
                    techniques=(cand.technique,),
                )
            )
            return True

        for cand in tag_candidates:
            if not _enqueue_initial(cand, cand.start, cand.end):
                break
        for cand in initial_candidates:
            outer_start, outer_end = primary.map_span(cand.start, cand.end)
            if not _enqueue_initial(cand, outer_start, outer_end):
                break

        while queue:
            item = queue.popleft()
            if len(segments) >= self._max_segments:
                truncated = True
                break

            seg_text, _, seg_stage_techs = _normalize_text_stage(item.text)
            deduped_seg_techs = _dedup_tuple([*item.techniques, *seg_stage_techs])
            extra_techniques.extend(deduped_seg_techs)

            segments.append(
                NormalizedSegment(
                    text=seg_text,
                    span_source=SpanSource.DECODED_BLOB,
                    outer_start=item.outer_start,
                    outer_end=item.outer_end,
                    evasion_techniques=deduped_seg_techs,
                )
            )

            remaining = self._max_decoded_bytes - used_bytes
            slots_left = max(self._max_segments - len(segments) - len(queue) + 1, 1)
            next_candidates = _find_encoded_candidates(
                seg_text,
                max(remaining, 1),
                max_candidates=slots_left,
            )
            if not next_candidates:
                continue

            if item.depth >= self._max_depth:
                truncated = True
                for cand in next_candidates:
                    extra_techniques.append(cand.technique)
                continue

            for cand in next_candidates:
                if cand.oversized or used_bytes + cand.decoded_bytes > self._max_decoded_bytes:
                    extra_techniques.append(cand.technique)
                    truncated = True
                    continue
                if len(segments) + len(queue) >= self._max_segments:
                    extra_techniques.append(cand.technique)
                    truncated = True
                    break
                used_bytes += cand.decoded_bytes
                queue.append(
                    _QueueItem(
                        text=cand.decoded_text,
                        depth=item.depth + 1,
                        outer_start=item.outer_start,
                        outer_end=item.outer_end,
                        techniques=(*deduped_seg_techs, cand.technique),
                    )
                )

        if queue:
            truncated = True

        return (tuple(segments), tuple(extra_techniques), truncated)


__all__ = [
    "AntiEvasionNormalizer",
    "collapse_split_digits",
]
