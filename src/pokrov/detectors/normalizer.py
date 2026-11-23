"""Anti-evasion text normalizer for Pokrov.

Handles invisible/bidi-override character stripping, context-aware Cyrillic/Latin
homoglyph normalization, bounded recursive decoding of URL/hex/Base64 blobs, and
split-digit normalization while preserving exact character back-mappings.
"""

from __future__ import annotations

import base64
import binascii
import re
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

_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]")

_BIDI_OVERRIDE_CHARS: frozenset[str] = frozenset(
    "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)

_MIXED_TOKEN_RE = re.compile(
    r"(?<![^\W_])(?:[A-Za-z0-9]+[А-Яа-яЁёІіЅѕЈјӀӏ]|[А-Яа-яЁёІіЅѕЈјӀӏ]+[A-Za-z0-9])[^\W_]*"
)
_CYRILLIC_PRESENT_RE = re.compile(r"[А-Яа-яЁёІіЅѕЈјӀӏ]")
_LATIN_OR_DIGIT_PRESENT_RE = re.compile(r"[A-Za-z0-9]")

# Symmetric Cyrillic <-> Latin confusables
_CYR_CHARS = "АВЕКМНОРСТХУаеорсух"
_LAT_CHARS = "ABEKMHOPCTXYaeopcyx"

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

_CYR_ALL_SET: frozenset[str] = frozenset(
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяІіЅѕЈјӀӏ"
)
_LAT_ALL_SET: frozenset[str] = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# Cyrillic characters whose shapes mimic ASCII digits in obfuscated numeric tokens
_CYR_DIGIT_LOOKALIKES: dict[str, str] = {
    "О": "0",
    "о": "0",
    "З": "3",
    "з": "3",
    "І": "1",
    "і": "1",
    "Ӏ": "1",
    "ӏ": "1",
}

# Preserve genuine Russian vehicle registration plates (e.g. О777ОО77, А123ВС199)
_RU_AUTO_PLATE_RE = re.compile(
    r"^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}(?:\d{2,3})?$",
    re.IGNORECASE,
)

# Encoded blob candidate finders (strictly linear-time, no nested quantifiers)
_PCT_PAIR_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_URL_STOP_CHARS: frozenset[str] = frozenset(" \t\r\n\"'<>()[]{}")

_HEX_CONTIGUOUS_RE = re.compile(r"(?<![A-Za-z0-9])(?:0[xX])?[0-9A-Fa-f]{8,}(?![A-Za-z0-9])")
_HEX_ESCAPED_RE = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){4,}")
_HEX_SPACED_RE = re.compile(r"(?<![A-Za-z0-9])(?:[0-9A-Fa-f]{2} ){3,}[0-9A-Fa-f]{2}(?![A-Za-z0-9])")
_HEX_LETTER_RE = re.compile(r"[A-Fa-f]")

_B64_URLSAFE_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}={0,2}(?![A-Za-z0-9_=-])")
_B64_STANDARD_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{8,}={0,2}(?![A-Za-z0-9+/=])")

# Split/spaced digit separator matcher: spaces, tabs, and hyphen/dash variants between digits
_DIGIT_SEP_RE = re.compile(r"(?<=\d)[ \t\-\u2010-\u2015]+(?=\d)")

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
    """Strip zero-width and bidi-override characters while building ``index_map``."""
    if not text:
        return ("", (), ())

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
        ch = match.group(0)
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


def _normalize_token_homoglyphs(token: str) -> str | None:
    """Normalize a single alphanumeric token if it exhibits script mixing or digit spoofing.

    Returns ``None`` when the token is already clean (avoids allocation on the fast path).
    """
    cyr_count = 0
    lat_count = 0
    digit_count = 0
    uniq_cyr = 0
    uniq_lat = 0
    other_alpha_in_num = False
    has_cyr_digit_lookalike = False

    for ch in token:
        if "0" <= ch <= "9":
            digit_count += 1
        elif ch in _CYR_ALL_SET:
            cyr_count += 1
            if ch not in _CYR_TO_LAT:
                uniq_cyr += 1
            if ch in _CYR_DIGIT_LOOKALIKES:
                has_cyr_digit_lookalike = True
            else:
                other_alpha_in_num = True
        elif ch in _LAT_ALL_SET:
            lat_count += 1
            if ch not in _LAT_TO_CYR:
                uniq_lat += 1
            other_alpha_in_num = True

    # Case 1: Obfuscated numeric token (digits + Cyrillic digit lookalikes like О/0, З/3, І/1)
    if (
        digit_count > 0
        and has_cyr_digit_lookalike
        and lat_count == 0
        and not other_alpha_in_num
        and not _RU_AUTO_PLATE_RE.match(token)
    ):
        return "".join(_CYR_DIGIT_LOOKALIKES.get(ch, ch) for ch in token)

    # Case 2: Pure single-script tokens must never be altered
    if cyr_count == 0 or lat_count == 0:
        return None

    # If both scripts contribute non-confusable characters (e.g. clientПаспорт), keep intact
    if uniq_cyr > 0 and uniq_lat > 0:
        return None

    if uniq_lat > 0:
        target_latin = True
    elif uniq_cyr > 0:
        target_latin = False
    else:
        # All letters are confusable across both scripts; break ties by majority, else Latin
        target_latin = lat_count >= cyr_count

    mapping = _CYR_TO_LAT if target_latin else _LAT_TO_CYR
    normalized = "".join(mapping.get(ch, ch) for ch in token)
    return normalized if normalized != token else None


def _normalize_homoglyphs(text: str) -> tuple[str, bool]:
    """Normalize Cyrillic/Latin homoglyphs without altering pure-Russian or pure-English text."""
    if not text or text.isascii():
        return (text, False)

    if _CYRILLIC_PRESENT_RE.search(text) is None:
        return (text, False)

    if _LATIN_OR_DIGIT_PRESENT_RE.search(text) is None:
        return (text, False)

    chunks: list[str] = []
    cursor = 0
    changed = False
    token_cache: dict[str, str | None] = {}

    for match in _MIXED_TOKEN_RE.finditer(text):
        token = match.group(0)
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
            # Trim incomplete %X tail if cut mid-escape
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
        # Guard against false positives on pure decimal IDs/passports/cards:
        # allow long (>= 16 hex chars) pure-digit hex whose decoded bytes are ASCII digits/spaces.
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
    if (
        not pad_count
        and not any(ch in "+/_-" for ch in core)
        and core.isalpha()
        and (core.islower() or core.isupper())
    ):
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


def _scan_b64_candidates(
    text: str,
    remaining_bytes: int,
    occupied: list[tuple[int, int]],
    max_candidates: int,
) -> list[_DecodeCandidate]:
    """Detect URL-safe and standard Base64 payloads in linear time."""
    candidates: list[_DecodeCandidate] = []
    invalid_tokens: set[str] = set()

    def _collect(pattern: re.Pattern[str], *, urlsafe: bool) -> None:
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

    # Check URL-safe tokens containing '-' or '_' first so they are not split prematurely
    if "-" in text or "_" in text:
        _collect(_B64_URLSAFE_RE, urlsafe=True)
    if len(candidates) < max_candidates:
        _collect(_B64_STANDARD_RE, urlsafe=False)

    return candidates


def _find_encoded_candidates(
    text: str,
    remaining_bytes: int,
    max_candidates: int = 17,
) -> list[_DecodeCandidate]:
    """Find non-overlapping URL, hex, and Base64 candidates sorted by start offset."""
    if len(text) < 4 or max_candidates < 1:
        return []

    occupied: list[tuple[int, int]] = []
    candidates: list[_DecodeCandidate] = []

    if "%" in text:
        candidates.extend(_scan_url_candidates(text, remaining_bytes, occupied, max_candidates))

    candidates.extend(_scan_hex_candidates(text, remaining_bytes, occupied, max_candidates))
    candidates.extend(_scan_b64_candidates(text, remaining_bytes, occupied, max_candidates))

    candidates.sort(key=lambda c: c.start)
    return candidates


def collapse_split_digits(
    source: str | NormalizedSegment,
    *,
    min_digits: int = 2,
) -> NormalizedSegment:
    """Collapse spaces and hyphens separating digits while preserving raw back-mapping.

    Useful for catching split-digit obfuscation such as ``"45 12 - 654 321"`` or
    ``"4 5 1 2  6 5 4 3 2 1"``. Accepts either raw text or an existing
    ``NormalizedSegment`` (composing its ``index_map`` back to raw text).
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


def _find_split_digit_drops(text: str, min_digits: int) -> list[tuple[int, int]]:
    """Return separator spans inside digit groups that meet ``min_digits``."""
    first = _DIGIT_SEP_RE.search(text)
    if first is None:
        return []

    spans_to_drop: list[tuple[int, int]] = []
    matches = list(_DIGIT_SEP_RE.finditer(text, first.start()))
    group: list[re.Match[str]] = [matches[0]]
    for m in matches[1:]:
        between = text[group[-1].end() : m.start()]
        if between.isdigit():
            group.append(m)
        else:
            if _count_group_digits(text, group) >= min_digits:
                spans_to_drop.extend(gm.span() for gm in group)
            group = [m]
    if _count_group_digits(text, group) >= min_digits:
        spans_to_drop.extend(gm.span() for gm in group)
    return spans_to_drop


def _count_group_digits(text: str, group: list[re.Match[str]]) -> int:
    """Count ASCII digits belonging to a chained digit-separator match group."""
    left = group[0].start()
    while left > 0 and "0" <= text[left - 1] <= "9":
        left -= 1
    right = group[-1].end()
    n = len(text)
    while right < n and "0" <= text[right] <= "9":
        right += 1
    return sum(1 for ch in text[left:right] if "0" <= ch <= "9")


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
        stripped, index_map, strip_techniques = _strip_invisible(text)
        primary_text, homoglyph_changed = _normalize_homoglyphs(stripped)

        primary_techniques_list = list(strip_techniques)
        if homoglyph_changed:
            primary_techniques_list.append("homoglyph")
        primary_techniques = tuple(primary_techniques_list)

        primary = NormalizedSegment(
            text=primary_text,
            span_source=SpanSource.DIRECT,
            index_map=index_map,
            evasion_techniques=primary_techniques,
        )

        decoded_segments, all_extra_techniques, truncated = self._decode_bounded(primary)

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
    ) -> tuple[tuple[NormalizedSegment, ...], tuple[str, ...], bool]:
        """Breadth-first bounded decoding of embedded encoded blobs."""
        if not primary.text:
            return ((), (), False)

        segments: list[NormalizedSegment] = []
        extra_techniques: list[str] = []
        truncated = False
        used_bytes = 0

        queue: deque[_QueueItem] = deque()
        initial_limit = max(self._max_segments + 1, 1)
        initial_candidates = _find_encoded_candidates(
            primary.text,
            self._max_decoded_bytes,
            max_candidates=initial_limit,
        )
        for cand in initial_candidates:
            outer_start, outer_end = primary.map_span(cand.start, cand.end)
            if self._max_depth == 0 or self._max_segments == 0:
                extra_techniques.append(cand.technique)
                truncated = True
                continue
            if cand.oversized or used_bytes + cand.decoded_bytes > self._max_decoded_bytes:
                extra_techniques.append(cand.technique)
                truncated = True
                continue
            if len(queue) >= self._max_segments:
                extra_techniques.append(cand.technique)
                truncated = True
                break
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

        while queue:
            item = queue.popleft()
            if len(segments) >= self._max_segments:
                truncated = True
                break

            seg_stripped, _, seg_strip_tech = _strip_invisible(item.text)
            seg_text, seg_homo_changed = _normalize_homoglyphs(seg_stripped)
            seg_techs = list(item.techniques)
            seg_techs.extend(seg_strip_tech)
            if seg_homo_changed:
                seg_techs.append("homoglyph")
            deduped_seg_techs = _dedup_tuple(seg_techs)
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
