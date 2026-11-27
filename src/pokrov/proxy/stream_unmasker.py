"""Sliding-window SSE stream unmasker, EchoLeak guard, and span masking helpers."""

from __future__ import annotations

import html
import re
import urllib.parse
from collections.abc import Mapping, Sequence

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.contracts import (
    PLACEHOLDER_PREFIX,
    PLACEHOLDER_SUFFIX,
    Action,
    ControlPoint,
    Detection,
    DetectionPipeline,
    EntityCategory,
    InspectionRequest,
    InspectionResult,
    PolicyDecision,
    PolicyEngine,
    StreamChunkDecision,
    VaultBackend,
)

MAX_PLACEHOLDER_LEN = 128
MAX_EXFIL_HOLD = 1024
_MAX_NONSTREAM_EXFIL_SCAN = 65_536
_ROLLING_WINDOW_CHARS = 512
_LOOKAHEAD_HOLD_CHARS = 96
_MAX_LOOKAHEAD_HOLD_CHARS = 256

_CONTROL_STRIP_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_EXTERNAL_URL_CANDIDATE_RE = re.compile(
    r"(?:https?://[^\s\"'<>)\]]+|(?<![A-Za-z0-9:/])//[A-Za-z0-9][-A-Za-z0-9.]*(?::\d+)?(?:/[^\s\"'<>)\]]*)?)",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    re.escape(PLACEHOLDER_PREFIX) + r"[A-Z0-9_]+:[0-9a-f]{8,16}" + re.escape(PLACEHOLDER_SUFFIX)
)
_ALLOWED_TOKEN_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_:-."
)

_ACTIVE_EXFIL_TAGS = frozenset(
    {
        "img",
        "iframe",
        "svg",
        "image",
        "object",
        "embed",
        "link",
        "meta",
        "video",
        "audio",
        "source",
        "input",
    }
)

_URL_ATTR_WITH_PKR_RE = re.compile(
    r"""\b(?:src|srcset|href|xlink:href|action|formaction|data|poster|ping|content)\s*=\s*"""
    r"""(?:"[^"]*<PKR:[^"]*"|'[^']*<PKR:[^']*'|[^\s>]*<PKR:[^\s>]*)"""
    r"""|\bstyle\s*=\s*(?:"[^"]*<PKR:[^"]*"|'[^']*<PKR:[^']*'|[^\s>]*<PKR:[^\s>]*)""",
    re.IGNORECASE,
)

_SAFE_ALT_ATTR_RE = re.compile(
    r"""(\b(?:alt|title)\s*=\s*)(?:"([^"]*)"|'([^']*)')""",
    re.IGNORECASE,
)

_REF_DEF_LINE_RE = re.compile(
    r"^[ \t]{0,3}\[([^\]]{1,256})\]:[ \t]*([^\r\n]+)",
    re.MULTILINE,
)

_CSS_URL_PKR_RE = re.compile(
    r"url\(\s*(?:\"[^\"]*<PKR:[^\"]*\"|'[^']*<PKR:[^']*'|[^)]*<PKR:[^)]*)\)",
    re.IGNORECASE,
)


def _normalize_for_url_scan(text: str) -> str:
    """Decode HTML character references and strip obfuscating control characters."""
    unescaped = html.unescape(text)
    return _CONTROL_STRIP_RE.sub("", unescaped)


def _is_domain_allowed(url_str: str, allowed_set: tuple[str, ...]) -> bool:
    if not allowed_set or PLACEHOLDER_PREFIX in url_str:
        return False
    candidate = f"https:{url_str}" if url_str.startswith("//") else url_str
    try:
        parsed = urllib.parse.urlsplit(candidate)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False
    return any(host == dom or host.endswith(f".{dom}") for dom in allowed_set)


def has_external_exfil_url(
    text: str,
    *,
    allowed_domains: Sequence[str] = (),
) -> bool:
    """Return True if ``text`` contains a non-allowlisted external HTTP(S) or ``//`` URL."""
    if not text:
        return False
    normalized = _normalize_for_url_scan(text)
    matches = [m.group(0) for m in _EXTERNAL_URL_CANDIDATE_RE.finditer(normalized)]
    if not matches:
        return False
    if not allowed_domains:
        return True
    allowed_set = tuple(d.strip().lower().lstrip(".") for d in allowed_domains if d.strip())
    if not allowed_set:
        return True
    return any(not _is_domain_allowed(u, allowed_set) for u in matches)


def _extract_html_tag_name(text: str, lt_pos: int, max_end: int) -> tuple[str, int]:
    """Return ``(lowercased_tag_name, name_end_idx)`` if ``text[lt_pos:]`` opens an HTML tag."""
    idx = lt_pos + 1
    if idx >= max_end or not text[idx].isalpha():
        return "", lt_pos + 1
    while idx < max_end and (text[idx].isalnum() or text[idx] in ("-", ":")):
        idx += 1
    return text[lt_pos + 1 : idx].lower(), idx


def _find_markdown_close_bracket(text: str, start: int, max_end: int) -> int:
    """Find closing ``]`` for ``![alt]``; return -1 if incomplete, -2 on paragraph break."""
    idx = start
    while idx < max_end:
        ch = text[idx]
        if ch == "\\" and idx + 1 < max_end:
            idx += 2
            continue
        if ch == "]":
            return idx
        if ch == "\n" and idx + 1 < max_end and text[idx + 1] == "\n":
            return -2
        idx += 1
    return -1


def _find_markdown_close_paren(text: str, start: int, max_end: int) -> int:
    """Find matching ``)`` for ``(url)``; return -1 if incomplete, -2 on paragraph break."""
    depth = 1
    idx = start
    while idx < max_end:
        ch = text[idx]
        if ch == "\\" and idx + 1 < max_end:
            idx += 2
            continue
        if ch == "<" and text.startswith(PLACEHOLDER_PREFIX, idx):
            gt_idx = text.find(">", idx + len(PLACEHOLDER_PREFIX), max_end)
            if gt_idx != -1:
                idx = gt_idx + 1
                continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
        elif ch == "\n" and idx + 1 < max_end and text[idx + 1] == "\n":
            return -2
        idx += 1
    return -1


def _find_html_tag_end(text: str, start: int, max_end: int) -> int:
    """Find closing ``>`` for an HTML tag respecting quotes and ``<PKR:...>`` placeholders."""
    quote: str | None = None
    idx = start
    while idx < max_end:
        ch = text[idx]
        if quote is not None:
            if ch == quote:
                quote = None
            idx += 1
            continue
        if ch in ('"', "'"):
            quote = ch
        elif ch == "<" and text.startswith(PLACEHOLDER_PREFIX, idx):
            gt_idx = text.find(">", idx + len(PLACEHOLDER_PREFIX), max_end)
            if gt_idx != -1:
                idx = gt_idx + 1
                continue
        elif ch == ">":
            return idx
        idx += 1
    return -1


def _make_exfil_detection(
    length: int,
    *,
    start: int = 0,
    end: int | None = None,
    detector: str = "stream_unmasker",
) -> Detection:
    """Build a Zero-Retention Detection for EchoLeak Markdown/HTML exfiltration."""
    eff_end = end if end is not None else max(start, length)
    return Detection(
        category=EntityCategory.MARKDOWN_EXFIL,
        start=start,
        end=max(start, eff_end),
        confidence=0.99,
        detector=detector,
        masked_preview=f"[markdown_exfil:len={max(0, length)}]",
        compliance_refs=get_compliance_refs(EntityCategory.MARKDOWN_EXFIL),
    )


def _collect_reference_exfil_ids(
    text: str,
    *,
    allowed_image_domains: Sequence[str] = (),
) -> set[str]:
    """Identify Markdown reference IDs tied to external image URLs or ``<PKR:...>`` leaks."""
    used_image_refs: set[str] = set()
    length = len(text)
    pos = 0
    while pos < length:
        bang = text.find("![", pos)
        if bang == -1:
            break
        close_br = _find_markdown_close_bracket(
            text, bang + 2, min(length, bang + _MAX_NONSTREAM_EXFIL_SCAN)
        )
        if close_br < 0:
            pos = bang + 2
            continue
        alt_part = text[bang + 2 : close_br]
        after = close_br + 1
        if after < length and text[after] == "[":
            ref_close = text.find("]", after + 1, min(length, after + 260))
            if ref_close != -1:
                ref_raw = text[after + 1 : ref_close].strip() or alt_part.strip()
                if ref_raw:
                    used_image_refs.add(ref_raw.lower())
                pos = ref_close + 1
                continue
        elif (after >= length or text[after] != "(") and alt_part.strip():
            used_image_refs.add(alt_part.strip().lower())
        pos = close_br + 1

    exfil_ids: set[str] = set()
    for match in _REF_DEF_LINE_RE.finditer(text):
        ref_id = match.group(1).strip().lower()
        url_tail = match.group(2).strip()
        if PLACEHOLDER_PREFIX in url_tail or (
            ref_id in used_image_refs
            and has_external_exfil_url(url_tail, allowed_domains=allowed_image_domains)
        ):
            exfil_ids.add(ref_id)
    return exfil_ids


def sanitize_echoleak(
    text: str,
    *,
    allowed_image_domains: Sequence[str] = (),
) -> tuple[str, list[Detection]]:
    """Detect and strip EchoLeak Markdown/HTML exfiltration and ``<PKR:...>`` URL leaks."""
    if not text:
        return "", []

    exfil_ref_ids = _collect_reference_exfil_ids(text, allowed_image_domains=allowed_image_domains)
    parts: list[str] = []
    detections: list[Detection] = []
    pos = 0
    length = len(text)

    while pos < length:
        lt_idx = text.find("<", pos)
        bang_idx = text.find("!", pos)
        lbr_idx = text.find("[", pos)
        css_m = _CSS_URL_PKR_RE.search(text, pos)
        css_idx = css_m.start() if css_m is not None else -1

        candidates = [idx for idx in (lt_idx, bang_idx, lbr_idx, css_idx) if idx != -1]
        if not candidates:
            parts.append(text[pos:])
            break

        next_idx = min(candidates)
        if next_idx > pos:
            parts.append(text[pos:next_idx])
            pos = next_idx

        if css_m is not None and pos == css_idx:
            span_len = css_m.end() - css_m.start()
            detections.append(
                _make_exfil_detection(
                    span_len,
                    start=css_m.start(),
                    end=css_m.end(),
                    detector="echoleak_guard",
                )
            )
            pos = css_m.end()
            continue

        if text[pos] == "!":
            if pos + 1 < length and text[pos + 1] == "[":
                scan_end = min(length, pos + _MAX_NONSTREAM_EXFIL_SCAN)
                close_br = _find_markdown_close_bracket(text, pos + 2, scan_end)
                if close_br >= 0:
                    alt_part = text[pos + 2 : close_br]
                    after_br = close_br + 1
                    while after_br < scan_end and text[after_br] in " \t":
                        after_br += 1
                    if after_br < scan_end and text[after_br] == "(":
                        close_par = _find_markdown_close_paren(text, after_br + 1, scan_end)
                        if close_par >= 0:
                            url_part = text[after_br + 1 : close_par]
                            if (
                                has_external_exfil_url(
                                    url_part, allowed_domains=allowed_image_domains
                                )
                                or PLACEHOLDER_PREFIX in url_part
                            ):
                                payload_len = close_par + 1 - pos
                                detections.append(
                                    _make_exfil_detection(
                                        payload_len,
                                        start=pos,
                                        end=close_par + 1,
                                        detector="echoleak_guard",
                                    )
                                )
                                pos = close_par + 1
                                continue
                    elif after_br < scan_end and text[after_br] == "[":
                        ref_close = text.find("]", after_br + 1, min(length, after_br + 260))
                        if ref_close != -1:
                            ref_id = (
                                text[after_br + 1 : ref_close].strip() or alt_part.strip()
                            ).lower()
                            if ref_id in exfil_ref_ids:
                                payload_len = ref_close + 1 - pos
                                detections.append(
                                    _make_exfil_detection(
                                        payload_len,
                                        start=pos,
                                        end=ref_close + 1,
                                        detector="echoleak_guard",
                                    )
                                )
                                pos = ref_close + 1
                                continue
                    else:
                        ref_id = alt_part.strip().lower()
                        if ref_id and ref_id in exfil_ref_ids:
                            payload_len = close_br + 1 - pos
                            detections.append(
                                _make_exfil_detection(
                                    payload_len,
                                    start=pos,
                                    end=close_br + 1,
                                    detector="echoleak_guard",
                                )
                            )
                            pos = close_br + 1
                            continue
            parts.append(text[pos])
            pos += 1
            continue

        if text[pos] == "[":
            # Check line-start reference definitions `[id]: url`
            if (
                pos == 0
                or text[pos - 1] == "\n"
                or (text[pos - 1] in " \t" and (pos < 4 or "\n" in text[max(0, pos - 4) : pos]))
            ):
                line_end = text.find("\n", pos)
                if line_end == -1:
                    line_end = length
                line_slice = text[pos:line_end]
                ref_m = _REF_DEF_LINE_RE.match(line_slice)
                if ref_m is not None:
                    ref_id = ref_m.group(1).strip().lower()
                    url_tail = ref_m.group(2).strip()
                    if ref_id in exfil_ref_ids or PLACEHOLDER_PREFIX in url_tail:
                        detections.append(
                            _make_exfil_detection(
                                len(line_slice),
                                start=pos,
                                end=line_end,
                                detector="echoleak_guard",
                            )
                        )
                        pos = line_end
                        continue
            # Check `[link_text](url)` carrying `<PKR:...>` inside URL target
            scan_end = min(length, pos + _MAX_NONSTREAM_EXFIL_SCAN)
            close_br = _find_markdown_close_bracket(text, pos + 1, scan_end)
            if close_br >= 0 and close_br + 1 < scan_end and text[close_br + 1] == "(":
                close_par = _find_markdown_close_paren(text, close_br + 2, scan_end)
                if close_par >= 0:
                    url_part = text[close_br + 2 : close_par]
                    if PLACEHOLDER_PREFIX in url_part:
                        payload_len = close_par + 1 - pos
                        detections.append(
                            _make_exfil_detection(
                                payload_len,
                                start=pos,
                                end=close_par + 1,
                                detector="echoleak_guard",
                            )
                        )
                        pos = close_par + 1
                        continue
            parts.append(text[pos])
            pos += 1
            continue

        # text[pos] == "<"
        if not text.startswith(PLACEHOLDER_PREFIX, pos):
            scan_end = min(length, pos + _MAX_NONSTREAM_EXFIL_SCAN)
            tag_name, name_end = _extract_html_tag_name(text, pos, scan_end)
            if tag_name and (
                name_end >= scan_end or text[name_end].isspace() or text[name_end] in ("/", ">")
            ):
                gt_idx = _find_html_tag_end(text, name_end, scan_end)
                tag_end = gt_idx + 1 if gt_idx >= 0 else scan_end
                tag_slice = text[pos:tag_end]
                is_active_exfil = tag_name in _ACTIVE_EXFIL_TAGS and has_external_exfil_url(
                    tag_slice, allowed_domains=allowed_image_domains
                )
                has_attr_pkr = _URL_ATTR_WITH_PKR_RE.search(tag_slice) is not None
                if is_active_exfil or has_attr_pkr:
                    detections.append(
                        _make_exfil_detection(
                            len(tag_slice),
                            start=pos,
                            end=tag_end,
                            detector="echoleak_guard",
                        )
                    )
                    pos = tag_end
                    continue

        parts.append(text[pos])
        pos += 1

    return "".join(parts), detections


def _collect_unsafe_unmask_ranges(text: str) -> list[tuple[int, int]]:
    """Collect URL and HTML non-alt ``[start, end)`` ranges where placeholders must not unmask."""
    ranges: list[tuple[int, int]] = []
    length = len(text)

    # 1. Markdown `](url)` targets
    pos = 0
    while pos < length:
        idx = text.find("](", pos)
        if idx == -1:
            break
        close_par = _find_markdown_close_paren(
            text, idx + 2, min(length, idx + _MAX_NONSTREAM_EXFIL_SCAN)
        )
        end_idx = close_par + 1 if close_par >= 0 else length
        ranges.append((idx + 2, end_idx))
        pos = idx + 2

    # 2. Markdown `[id]: url` reference definitions
    for m in _REF_DEF_LINE_RE.finditer(text):
        ranges.append((m.start(2), m.end(2)))

    # 3. CSS `url(...)`
    for m in _CSS_URL_PKR_RE.finditer(text):
        ranges.append((m.start(), m.end()))

    # 4. HTML tags: forbid unmasking everywhere inside `<tag ...>` EXCEPT safe `alt=`/`title=`
    pos = 0
    while pos < length:
        lt = text.find("<", pos)
        if lt == -1:
            break
        if text.startswith(PLACEHOLDER_PREFIX, lt):
            pos = lt + len(PLACEHOLDER_PREFIX)
            continue
        tag_name, name_end = _extract_html_tag_name(
            text, lt, min(length, lt + _MAX_NONSTREAM_EXFIL_SCAN)
        )
        if not tag_name:
            pos = lt + 1
            continue
        gt = _find_html_tag_end(text, name_end, min(length, lt + _MAX_NONSTREAM_EXFIL_SCAN))
        tag_end = gt + 1 if gt >= 0 else length
        tag_slice = text[lt:tag_end]
        safe_sub_ranges: list[tuple[int, int]] = []
        for attr_m in _SAFE_ALT_ATTR_RE.finditer(tag_slice):
            val_group = 2 if attr_m.group(2) is not None else 3
            s_off, e_off = attr_m.span(val_group)
            if s_off >= 0 and e_off > s_off:
                safe_sub_ranges.append((lt + s_off, lt + e_off))

        cursor = lt
        for s_safe, e_safe in safe_sub_ranges:
            if s_safe > cursor:
                ranges.append((cursor, s_safe))
            cursor = e_safe
        if cursor < tag_end:
            ranges.append((cursor, tag_end))
        pos = tag_end

    return ranges


def restore_text_outside_urls(
    vault: VaultBackend,
    session_id: str,
    text: str,
) -> tuple[str, int]:
    """Restore ``<PKR:...>`` tokens outside URL and active HTML attribute contexts."""
    if PLACEHOLDER_PREFIX not in text:
        return text, 0

    unsafe_ranges = _collect_unsafe_unmask_ranges(text)
    restored_count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal restored_count
        m_start, m_end = match.start(), match.end()
        if any(u_start <= m_start and m_end <= u_end for u_start, u_end in unsafe_ranges):
            return match.group(0)
        placeholder = match.group(0)
        val = vault.retrieve(session_id, placeholder)
        if val is None:
            return placeholder
        restored_count += 1
        return val

    restored = _PLACEHOLDER_RE.sub(_replace, text)
    return restored, restored_count


def _resolve_detection_action(det: Detection, decision: PolicyDecision) -> Action:
    """Resolve effective per-detection action from ``PolicyDecision``."""
    rule_name = det.metadata.get("rule_name", "")
    if rule_name and rule_name in decision.rule_actions:
        return decision.rule_actions[rule_name]
    if decision.category_actions:
        return decision.category_actions.get(det.category, Action.ALLOW)
    if det.category in decision.categories:
        return decision.action
    return Action.ALLOW


def mask_text_spans(
    text: str,
    detections: Sequence[Detection],
    *,
    action: Action,
    vault: VaultBackend,
    session_id: str,
    allowed_categories: set[EntityCategory] | None = None,
    category_actions: Mapping[EntityCategory, Action] | None = None,
    rule_actions: Mapping[str, Action] | None = None,
) -> tuple[str, int]:
    """Replace detected spans in ``text`` honoring per-category and per-rule actions."""
    if not text or not detections:
        return text, 0

    text_len = len(text)
    valid_spans: list[tuple[int, int, EntityCategory, Action]] = []
    has_granular_map = bool(category_actions or rule_actions)

    for det in detections:
        if allowed_categories is not None and det.category not in allowed_categories:
            continue

        det_action: Action
        rule_name = det.metadata.get("rule_name", "")
        if rule_actions and rule_name and rule_name in rule_actions:
            det_action = rule_actions[rule_name]
        elif category_actions is not None and det.category in category_actions:
            det_action = category_actions[det.category]
        elif has_granular_map:
            continue
        else:
            det_action = action

        if det_action not in (Action.MASK, Action.MASK_REVERSIBLE):
            continue

        start = max(0, min(det.start, text_len))
        end = max(start, min(det.end, text_len))
        if end > start:
            valid_spans.append((start, end, det.category, det_action))

    if not valid_spans:
        return text, 0

    valid_spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    merged: list[tuple[int, int, EntityCategory, Action]] = []
    for start, end, category, det_action in valid_spans:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, prev_cat, prev_act = merged[-1]
            # Prefer irreversible static MASK when two overlapping spans conflict
            eff_act = (
                Action.MASK if Action.MASK in (prev_act, det_action) else Action.MASK_REVERSIBLE
            )
            merged[-1] = (prev_start, max(prev_end, end), prev_cat, eff_act)
        else:
            merged.append((start, end, category, det_action))

    parts: list[str] = []
    cursor = 0
    replaced = 0
    for start, end, category, span_action in merged:
        if start > cursor:
            parts.append(text[cursor:start])
        raw_val = text[start:end]
        if span_action == Action.MASK_REVERSIBLE:
            replacement = vault.store(session_id, category, raw_val)
        else:
            replacement = f"[REDACTED:{category.value.upper()}]"
        parts.append(replacement)
        cursor = end
        replaced += 1

    if cursor < text_len:
        parts.append(text[cursor:])

    return "".join(parts), replaced


def _redact_local_detections(
    text: str,
    detections: Sequence[Detection],
    offset: int,
) -> str:
    """Statically redact detection spans mapped into a flushed text segment."""
    seg_len = len(text)
    spans: list[tuple[int, int, EntityCategory]] = []
    for det in detections:
        local_start = max(0, min(det.start - offset, seg_len))
        local_end = max(local_start, min(det.end - offset, seg_len))
        if local_end > local_start:
            spans.append((local_start, local_end, det.category))

    if not spans:
        return text

    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    merged: list[tuple[int, int, EntityCategory]] = []
    for start, end, category in spans:
        if merged and start < merged[-1][1]:
            prev_s, prev_e, prev_c = merged[-1]
            merged[-1] = (prev_s, max(prev_e, end), prev_c)
        else:
            merged.append((start, end, category))

    parts: list[str] = []
    cursor = 0
    for start, end, category in merged:
        if start > cursor:
            parts.append(text[cursor:start])
        parts.append(f"[REDACTED:{category.value.upper()}]")
        cursor = end
    if cursor < seg_len:
        parts.append(text[cursor:])
    return "".join(parts)


class SlidingWindowStreamGuard:
    """Sliding-window SSE stream unmasker and EchoLeak output guardrail."""

    def __init__(
        self,
        vault: VaultBackend,
        session_id: str,
        output_pipeline: DetectionPipeline | None = None,
        policy_engine: PolicyEngine | None = None,
        policy_profile: str = "balanced-152fz",
        principal: str = "anonymous",
        allowed_image_domains: Sequence[str] = (),
    ) -> None:
        self._vault = vault
        self._session_id = session_id
        self._output_pipeline = output_pipeline
        self._policy_engine = policy_engine
        self._policy_profile = policy_profile
        self._principal = principal
        self._allowed_domains: tuple[str, ...] = self._init_allowed_domains(
            allowed_image_domains, policy_engine, policy_profile
        )

        self._buffer = ""
        self._pending_out = ""
        self._rolling_raw = ""
        self._skip_until_delim: str | None = None
        self._aborted_reason = ""
        self._in_md_image_alt = False
        self._in_active_html_tag: str | None = None
        self._at_line_start = True
        self._seen_image_refs: set[str] = set()
        self._external_ref_defs: set[str] = set()

    @staticmethod
    def _init_allowed_domains(
        explicit_domains: Sequence[str],
        policy_engine: PolicyEngine | None,
        profile: str,
    ) -> tuple[str, ...]:
        if explicit_domains:
            return tuple(explicit_domains)
        cfg = getattr(policy_engine, "config", None)
        if cfg is not None:
            get_prof = getattr(cfg, "get_profile", None)
            if callable(get_prof):
                try:
                    prof_obj = get_prof(profile)
                    domains = getattr(prof_obj, "allowed_image_domains", None)
                    if isinstance(domains, list | tuple):
                        return tuple(str(d) for d in domains)
                except ValueError:
                    pass
        return ()

    def _vault_retrieve(self, token: str) -> str | None:
        if "\x1f" in self._session_id:
            return self._vault.retrieve(self._session_id, token)
        scoped = f"{self._principal}\x1f{self._session_id}"
        val = self._vault.retrieve(scoped, token)
        if val is not None:
            return val
        if self._principal == "anonymous":
            return self._vault.retrieve(self._session_id, token)
        return None

    def _vault_restore_safe_text(self, text: str) -> tuple[str, int]:
        if PLACEHOLDER_PREFIX not in text:
            return text, 0
        count = 0

        def _repl(match: re.Match[str]) -> str:
            nonlocal count
            tok = match.group(0)
            val = self._vault_retrieve(tok)
            if val is None:
                return tok
            count += 1
            return val

        return _PLACEHOLDER_RE.sub(_repl, text), count

    def _restore_html_tag_alt_only(self, tag: str) -> tuple[str, int]:
        if PLACEHOLDER_PREFIX not in tag:
            return tag, 0
        total = 0

        def _repl_attr(match: re.Match[str]) -> str:
            nonlocal total
            prefix = match.group(1)
            if match.group(2) is not None:
                restored, cnt = self._vault_restore_safe_text(match.group(2))
                total += cnt
                return f'{prefix}"{restored}"'
            restored, cnt = self._vault_restore_safe_text(match.group(3) or "")
            total += cnt
            return f"{prefix}'{restored}'"

        return _SAFE_ALT_ATTR_RE.sub(_repl_attr, tag), total

    def _evaluate_exfil_detection(
        self,
        det: Detection,
        raw_construct: str,
        default_reason: str,
    ) -> tuple[str, Detection, bool, str]:
        """Evaluate ``MARKDOWN_EXFIL`` via ``PolicyEngine`` so ``dev-audit`` flags safely."""
        if self._policy_engine is None:
            return "", det, True, default_reason

        decision = self._policy_engine.evaluate(
            InspectionResult(
                detections=(det,),
                control_point=ControlPoint.FINAL_OUTPUT,
            ),
            direction="output",
            principal=self._principal,
            profile=self._policy_profile,
        )
        act = _resolve_detection_action(det, decision)
        if act in (Action.BLOCK, Action.REQUIRE_APPROVAL):
            return "", det, True, (decision.reason or default_reason)
        if act in (Action.MASK, Action.MASK_REVERSIBLE):
            return "[REDACTED:MARKDOWN_EXFIL]", det, False, ""
        # Action.FLAG / ALLOW (e.g. dev-audit): emit construct without unmasking URLs
        return raw_construct, det, False, ""

    def feed(self, chunk: str) -> StreamChunkDecision:
        """Consume a streaming chunk and emit any verified safe/unmasked prefix."""
        if not chunk:
            return StreamChunkDecision(emit_text="")

        self._buffer += chunk
        return self._drain(final=False)

    def flush(self) -> StreamChunkDecision:
        """Flush remaining buffered characters at end-of-stream."""
        return self._drain(final=True)

    def _update_line_state(self, text: str) -> None:
        if not text:
            return
        nl = text.rfind("\n")
        if nl != -1:
            tail = text[nl + 1 :]
            self._at_line_start = tail.strip(" \t") == ""
        elif text.strip(" \t") != "":
            self._at_line_start = False

    def _drain(self, *, final: bool) -> StreamChunkDecision:
        segments: list[tuple[str, bool]] = []
        detections: list[Detection] = []
        unmasked_count = 0
        blocked = False
        block_reason = ""

        if self._skip_until_delim is not None:
            delim_idx = self._buffer.find(self._skip_until_delim)
            if delim_idx == -1:
                self._buffer = ""
                return StreamChunkDecision(
                    emit_text="",
                    blocked=True,
                    block_reason=self._aborted_reason,
                )
            self._buffer = self._buffer[delim_idx + 1 :]
            self._skip_until_delim = None
            blocked = True
            block_reason = self._aborted_reason
            if not self._buffer and not self._pending_out:
                return StreamChunkDecision(
                    emit_text="",
                    blocked=True,
                    block_reason=block_reason,
                )

        if self._in_active_html_tag is not None:
            gt_idx = _find_html_tag_end(self._buffer, 0, len(self._buffer))
            check_slice = self._buffer[: gt_idx + 1] if gt_idx >= 0 else self._buffer
            tag_name = self._in_active_html_tag
            is_exfil = (
                tag_name in _ACTIVE_EXFIL_TAGS
                and has_external_exfil_url(check_slice, allowed_domains=self._allowed_domains)
            ) or (_URL_ATTR_WITH_PKR_RE.search(check_slice) is not None)
            if is_exfil:
                det = _make_exfil_detection(len(check_slice))
                seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                    det,
                    check_slice,
                    f"EchoLeak protection: external URL blocked in HTML <{tag_name}> tag",
                )
                detections.append(det)
                if gt_idx >= 0:
                    self._buffer = self._buffer[gt_idx + 1 :]
                    self._in_active_html_tag = None
                else:
                    self._buffer = ""
                    if hit_block and not final:
                        self._skip_until_delim = ">"
                        self._aborted_reason = reason
                    elif final:
                        self._in_active_html_tag = None
                if hit_block:
                    self._pending_out = ""
                    return StreamChunkDecision(
                        emit_text="",
                        blocked=True,
                        block_reason=reason,
                        detections=tuple(detections),
                    )
                if seg_out:
                    segments.append((seg_out, False))
            elif gt_idx >= 0:
                closed_part = self._buffer[: gt_idx + 1]
                self._buffer = self._buffer[gt_idx + 1 :]
                self._in_active_html_tag = None
                restored_part, cnt = self._restore_html_tag_alt_only(closed_part)
                unmasked_count += cnt
                segments.append((restored_part, cnt > 0))
            elif not final and len(self._buffer) <= 96:
                emit_text, pipe_dets, pipe_blocked, pipe_reason = self._inspect_flushed_segments(
                    segments, final=False
                )
                detections.extend(pipe_dets)
                return StreamChunkDecision(
                    emit_text=emit_text,
                    blocked=pipe_blocked,
                    block_reason=pipe_reason,
                    unmasked_count=unmasked_count,
                    detections=tuple(detections),
                )
            elif not final:
                flush_cut = len(self._buffer) - 64
                safe_part = self._buffer[:flush_cut]
                self._buffer = self._buffer[flush_cut:]
                segments.append((safe_part, False))

        buf = self._buffer
        length = len(buf)
        pos = 0

        while pos < length:
            lt_idx = buf.find("<", pos)
            bang_idx = buf.find("!", pos)
            lbr_idx = buf.find("[", pos)
            rbr_idx = buf.find("]", pos)
            candidates = [idx for idx in (lt_idx, bang_idx, lbr_idx, rbr_idx) if idx != -1]
            if not candidates:
                plain = buf[pos:]
                if self._in_md_image_alt and "\n\n" in plain:
                    self._in_md_image_alt = False
                self._update_line_state(plain)
                segments.append((plain, False))
                pos = length
                break

            next_idx = min(candidates)
            if next_idx > pos:
                plain = buf[pos:next_idx]
                if self._in_md_image_alt and "\n\n" in plain:
                    self._in_md_image_alt = False
                self._update_line_state(plain)
                segments.append((plain, False))
                pos = next_idx

            rem = buf[pos:]
            ch = rem[0]
            tail_det: Detection | None = None
            if ch == "<":
                held, step, seg, tail_det, unm_delta, is_restored, hit_block, reason = (
                    self._process_lt(rem, final=final)
                )
            elif ch == "!":
                held, step, seg, tail_det, unm_delta, is_restored, hit_block, reason = (
                    self._process_bang(rem, final=final)
                )
            elif ch == "[":
                held, step, seg, tail_det, unm_delta, is_restored, hit_block, reason = (
                    self._process_lbracket(rem, final=final)
                )
            else:
                held, step, seg, tail_det, unm_delta, is_restored, hit_block, reason = (
                    self._process_rbracket(rem, final=final)
                )

            if held:
                break

            if tail_det is not None:
                detections.append(tail_det)
            unmasked_count += unm_delta
            if hit_block:
                blocked = True
                block_reason = reason
                pos += step
                if self._skip_until_delim is not None:
                    self._aborted_reason = reason
                break

            if seg:
                self._update_line_state(seg)
                segments.append((seg, is_restored))
            pos += step

        self._buffer = buf[pos:]

        if blocked:
            # Flush only already-verified safe segments prior to the blocked construct
            emit_text, pipe_dets, pipe_blocked, pipe_reason = self._inspect_flushed_segments(
                segments, final=True
            )
            if pipe_dets:
                detections.extend(pipe_dets)
            if pipe_blocked:
                block_reason = pipe_reason
            self._buffer = ""
            self._pending_out = ""
            return StreamChunkDecision(
                emit_text=emit_text,
                blocked=True,
                block_reason=block_reason,
                unmasked_count=unmasked_count,
                detections=tuple(detections),
            )

        emit_text, pipe_dets, pipe_blocked, pipe_reason = self._inspect_flushed_segments(
            segments, final=final
        )
        if pipe_dets:
            detections.extend(pipe_dets)
        if pipe_blocked:
            blocked = True
            block_reason = pipe_reason
            self._buffer = ""
            self._pending_out = ""

        return StreamChunkDecision(
            emit_text=emit_text,
            blocked=blocked,
            block_reason=block_reason,
            unmasked_count=unmasked_count,
            detections=tuple(detections),
        )

    def _process_lt(
        self, rem: str, *, final: bool
    ) -> tuple[bool, int, str, Detection | None, int, bool, bool, str]:
        """Handle ``<`` for ``<PKR:...>`` placeholders or active/attribute HTML tags."""
        rem_len = len(rem)
        prefix_len = len(PLACEHOLDER_PREFIX)

        if rem_len == 1:
            if not final:
                return True, 0, "", None, 0, False, False, ""
            return False, 1, "<", None, 0, False, False, ""

        if rem_len < prefix_len and PLACEHOLDER_PREFIX.startswith(rem):
            if not final:
                return True, 0, "", None, 0, False, False, ""
            return False, rem_len, rem, None, 0, False, False, ""

        if rem.startswith(PLACEHOLDER_PREFIX):
            limit = min(rem_len, MAX_PLACEHOLDER_LEN)
            for idx in range(prefix_len, limit):
                ch = rem[idx]
                if ch == ">":
                    if idx == prefix_len:
                        return False, 1, "<", None, 0, False, False, ""
                    token = rem[: idx + 1]
                    restored = self._vault_retrieve(token)
                    if restored is not None:
                        return False, idx + 1, restored, None, 1, True, False, ""
                    return False, idx + 1, token, None, 0, False, False, ""
                if ch not in _ALLOWED_TOKEN_CHARS:
                    return False, 1, "<", None, 0, False, False, ""
            if rem_len >= MAX_PLACEHOLDER_LEN:
                return False, 1, "<", None, 0, False, False, ""
            if not final:
                return True, 0, "", None, 0, False, False, ""
            return False, rem_len, rem, None, 0, False, False, ""

        if rem[1].isalpha():
            max_end = min(rem_len, MAX_EXFIL_HOLD)
            tag_name, name_end = _extract_html_tag_name(rem, 0, max_end)
            if not tag_name:
                return False, 1, "<", None, 0, False, False, ""
            if name_end == rem_len and not final:
                return True, 0, "", None, 0, False, False, ""
            if name_end < rem_len and not (rem[name_end].isspace() or rem[name_end] in ("/", ">")):
                return False, 1, "<", None, 0, False, False, ""

            gt_idx = _find_html_tag_end(rem, name_end, max_end)
            if gt_idx >= 0:
                tag = rem[: gt_idx + 1]
                is_exfil = (
                    tag_name in _ACTIVE_EXFIL_TAGS
                    and has_external_exfil_url(tag, allowed_domains=self._allowed_domains)
                ) or (_URL_ATTR_WITH_PKR_RE.search(tag) is not None)
                if is_exfil:
                    det = _make_exfil_detection(len(tag))
                    seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                        det,
                        tag,
                        f"EchoLeak protection: external URL blocked in HTML <{tag_name}> tag",
                    )
                    return False, gt_idx + 1, seg_out, det, 0, False, hit_block, reason

                restored_tag, cnt = self._restore_html_tag_alt_only(tag)
                return False, gt_idx + 1, restored_tag, None, cnt, cnt > 0, False, ""

            if rem_len < MAX_EXFIL_HOLD and not final:
                return True, 0, "", None, 0, False, False, ""

            window = rem[:max_end]
            is_exfil_win = (
                tag_name in _ACTIVE_EXFIL_TAGS
                and has_external_exfil_url(window, allowed_domains=self._allowed_domains)
            ) or (_URL_ATTR_WITH_PKR_RE.search(window) is not None)
            if is_exfil_win:
                det = _make_exfil_detection(len(window))
                seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                    det,
                    window,
                    f"EchoLeak protection: external URL blocked in HTML <{tag_name}> tag",
                )
                if hit_block and not final and rem_len >= MAX_EXFIL_HOLD:
                    self._skip_until_delim = ">"
                return False, len(window), seg_out, det, 0, False, hit_block, reason

            if final:
                restored_win, cnt = self._restore_html_tag_alt_only(rem)
                return False, rem_len, restored_win, None, cnt, cnt > 0, False, ""

            # Oversized HTML tag (>1024 chars before '>'): keep active-tag state across chunks!
            self._in_active_html_tag = tag_name
            flush_step = max(1, max_end - 64)
            return False, flush_step, rem[:flush_step], None, 0, False, False, ""

        return False, 1, "<", None, 0, False, False, ""

    def _process_bang(
        self, rem: str, *, final: bool
    ) -> tuple[bool, int, str, Detection | None, int, bool, bool, str]:
        """Handle ``!`` for ``![alt](url)`` and reference-style ``![alt][ref]`` images."""
        rem_len = len(rem)
        if rem_len == 1:
            if not final:
                return True, 0, "", None, 0, False, False, ""
            return False, 1, "!", None, 0, False, False, ""

        if rem[1] != "[":
            return False, 1, "!", None, 0, False, False, ""

        max_end = min(rem_len, MAX_EXFIL_HOLD)
        close_br = _find_markdown_close_bracket(rem, 2, max_end)
        if close_br == -2:
            return False, 1, "!", None, 0, False, False, ""
        if close_br == -1:
            if rem_len < MAX_EXFIL_HOLD and not final:
                return True, 0, "", None, 0, False, False, ""
            if final:
                return False, rem_len, rem, None, 0, False, False, ""
            # Oversized alt (>1024 chars): track _in_md_image_alt so closing ](url) is still caught!
            self._in_md_image_alt = True
            flush_step = max(2, max_end - 8)
            return False, flush_step, rem[:flush_step], None, 0, False, False, ""

        alt_part = rem[2:close_br]
        after_br = close_br + 1
        if after_br >= rem_len:
            if not final:
                return True, 0, "", None, 0, False, False, ""
            # Shortcut reference image `![id]` at EOF
            ref_id = alt_part.strip().lower()
            if ref_id:
                self._seen_image_refs.add(ref_id)
                if ref_id in self._external_ref_defs:
                    det = _make_exfil_detection(rem_len)
                    seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                        det,
                        rem,
                        "EchoLeak protection: external reference URL blocked in Markdown image",
                    )
                    return False, rem_len, seg_out, det, 0, False, hit_block, reason
            restored_alt, cnt = self._vault_restore_safe_text(alt_part)
            return False, rem_len, f"![{restored_alt}]", None, cnt, cnt > 0, False, ""

        if rem[after_br] == "[":
            ref_max = min(rem_len, after_br + 260)
            ref_close = rem.find("]", after_br + 1, ref_max)
            if ref_close == -1:
                if rem_len < after_br + 260 and not final:
                    return True, 0, "", None, 0, False, False, ""
                return False, 1, "!", None, 0, False, False, ""
            ref_label = rem[after_br + 1 : ref_close]
            ref_id = (ref_label.strip() or alt_part.strip()).lower()
            full_ref_img = rem[: ref_close + 1]
            if ref_id:
                self._seen_image_refs.add(ref_id)
                if ref_id in self._external_ref_defs:
                    det = _make_exfil_detection(len(full_ref_img))
                    seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                        det,
                        full_ref_img,
                        "EchoLeak protection: external reference URL blocked in Markdown image",
                    )
                    return False, ref_close + 1, seg_out, det, 0, False, hit_block, reason
            restored_alt, cnt = self._vault_restore_safe_text(alt_part)
            return (
                False,
                ref_close + 1,
                f"![{restored_alt}][{ref_label}]",
                None,
                cnt,
                cnt > 0,
                False,
                "",
            )

        if rem[after_br] != "(":
            # Shortcut reference image `![id]` followed by space/newline/text
            ref_id = alt_part.strip().lower()
            full_short = rem[: close_br + 1]
            if ref_id:
                self._seen_image_refs.add(ref_id)
                if ref_id in self._external_ref_defs:
                    det = _make_exfil_detection(len(full_short))
                    seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                        det,
                        full_short,
                        "EchoLeak protection: external reference URL blocked in Markdown image",
                    )
                    return False, close_br + 1, seg_out, det, 0, False, hit_block, reason
            restored_alt, cnt = self._vault_restore_safe_text(alt_part)
            return False, close_br + 1, f"![{restored_alt}]", None, cnt, cnt > 0, False, ""

        paren_max = min(rem_len, after_br + MAX_EXFIL_HOLD)
        close_par = _find_markdown_close_paren(rem, after_br + 1, paren_max)
        if close_par == -2:
            return False, 1, "!", None, 0, False, False, ""
        if close_par >= 0:
            md_img = rem[: close_par + 1]
            url_part = rem[after_br + 1 : close_par]
            if (
                has_external_exfil_url(url_part, allowed_domains=self._allowed_domains)
                or PLACEHOLDER_PREFIX in url_part
            ):
                det = _make_exfil_detection(len(md_img))
                seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                    det,
                    md_img,
                    "EchoLeak protection: external URL blocked in Markdown image",
                )
                return False, close_par + 1, seg_out, det, 0, False, hit_block, reason

            restored_alt, cnt = self._vault_restore_safe_text(alt_part)
            return (
                False,
                close_par + 1,
                f"![{restored_alt}]({url_part})",
                None,
                cnt,
                cnt > 0,
                False,
                "",
            )

        if rem_len < after_br + MAX_EXFIL_HOLD and not final:
            return True, 0, "", None, 0, False, False, ""

        window = rem[:paren_max]
        url_slice = window[after_br + 1 :]
        if (
            has_external_exfil_url(url_slice, allowed_domains=self._allowed_domains)
            or PLACEHOLDER_PREFIX in url_slice
        ):
            det = _make_exfil_detection(len(window))
            seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                det,
                window,
                "EchoLeak protection: external URL blocked in Markdown image",
            )
            if hit_block and not final and rem_len >= after_br + MAX_EXFIL_HOLD:
                self._skip_until_delim = ")"
            return False, len(window), seg_out, det, 0, False, hit_block, reason
        if final:
            return False, rem_len, rem, None, 0, False, False, ""
        return False, 1, "!", None, 0, False, False, ""

    def _process_lbracket(
        self, rem: str, *, final: bool
    ) -> tuple[bool, int, str, Detection | None, int, bool, bool, str]:
        """Handle line-start ``[id]: url`` reference definitions."""
        if not self._at_line_start:
            return False, 1, "[", None, 0, False, False, ""

        rem_len = len(rem)
        close_br = rem.find("]", 1, min(rem_len, 260))
        if close_br == -1:
            if "\n" in rem or rem_len >= 260 or final:
                return False, 1, "[", None, 0, False, False, ""
            return True, 0, "", None, 0, False, False, ""

        if close_br + 1 >= rem_len:
            if not final:
                return True, 0, "", None, 0, False, False, ""
            return False, 1, "[", None, 0, False, False, ""

        if rem[close_br + 1] != ":":
            return False, 1, "[", None, 0, False, False, ""

        nl_idx = rem.find("\n", close_br + 2, min(rem_len, MAX_EXFIL_HOLD))
        if nl_idx == -1 and rem_len < MAX_EXFIL_HOLD and not final:
            return True, 0, "", None, 0, False, False, ""

        line_end = nl_idx if nl_idx != -1 else min(rem_len, MAX_EXFIL_HOLD)
        full_def = rem[:line_end]
        ref_id = rem[1:close_br].strip().lower()
        url_part = rem[close_br + 2 : line_end].strip()

        is_ext = has_external_exfil_url(url_part, allowed_domains=self._allowed_domains)
        if is_ext and ref_id:
            self._external_ref_defs.add(ref_id)

        if PLACEHOLDER_PREFIX in url_part or (is_ext and ref_id in self._seen_image_refs):
            det = _make_exfil_detection(len(full_def))
            seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                det,
                full_def,
                "EchoLeak protection: external reference definition blocked",
            )
            if hit_block and nl_idx == -1 and not final and rem_len >= MAX_EXFIL_HOLD:
                self._skip_until_delim = "\n"
            return False, line_end, seg_out, det, 0, False, hit_block, reason

        return False, line_end, full_def, None, 0, False, False, ""

    def _process_rbracket(
        self, rem: str, *, final: bool
    ) -> tuple[bool, int, str, Detection | None, int, bool, bool, str]:
        """Handle ``](url)`` for link ``<PKR:...>`` protection and oversized-alt images."""
        rem_len = len(rem)
        if rem_len == 1:
            if not final:
                return True, 0, "", None, 0, False, False, ""
            self._in_md_image_alt = False
            return False, 1, "]", None, 0, False, False, ""

        if self._in_md_image_alt and rem[1] == "[":
            ref_close = rem.find("]", 2, min(rem_len, 262))
            if ref_close == -1:
                if rem_len < 262 and not final:
                    return True, 0, "", None, 0, False, False, ""
                self._in_md_image_alt = False
                return False, 1, "]", None, 0, False, False, ""
            self._in_md_image_alt = False
            ref_id = rem[2:ref_close].strip().lower()
            target_seg = rem[: ref_close + 1]
            if ref_id:
                self._seen_image_refs.add(ref_id)
                if ref_id in self._external_ref_defs:
                    det = _make_exfil_detection(len(target_seg))
                    seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                        det,
                        target_seg,
                        "EchoLeak protection: external reference URL blocked in Markdown image",
                    )
                    return False, ref_close + 1, seg_out, det, 0, False, hit_block, reason
            return False, ref_close + 1, target_seg, None, 0, False, False, ""

        if rem[1] != "(":
            self._in_md_image_alt = False
            return False, 1, "]", None, 0, False, False, ""

        was_image_alt = self._in_md_image_alt
        max_end = min(rem_len, MAX_EXFIL_HOLD)
        close_par = _find_markdown_close_paren(rem, 2, max_end)
        if close_par == -2:
            self._in_md_image_alt = False
            return False, 1, "]", None, 0, False, False, ""

        if close_par >= 0:
            self._in_md_image_alt = False
            target_slice = rem[: close_par + 1]
            url_part = rem[2:close_par]
            is_exfil = PLACEHOLDER_PREFIX in url_part or (
                was_image_alt
                and has_external_exfil_url(url_part, allowed_domains=self._allowed_domains)
            )
            if is_exfil:
                det = _make_exfil_detection(len(target_slice))
                seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                    det,
                    target_slice,
                    "EchoLeak protection: sensitive token or external URL blocked in Markdown link",
                )
                return False, close_par + 1, seg_out, det, 0, False, hit_block, reason
            return False, close_par + 1, target_slice, None, 0, False, False, ""

        if rem_len < MAX_EXFIL_HOLD and not final:
            return True, 0, "", None, 0, False, False, ""

        self._in_md_image_alt = False
        window = rem[:max_end]
        url_slice = window[2:]
        if PLACEHOLDER_PREFIX in url_slice or (
            was_image_alt
            and has_external_exfil_url(url_slice, allowed_domains=self._allowed_domains)
        ):
            det = _make_exfil_detection(len(window))
            seg_out, det, hit_block, reason = self._evaluate_exfil_detection(
                det,
                window,
                "EchoLeak protection: sensitive token or external URL blocked in Markdown link",
            )
            if hit_block and not final and rem_len >= MAX_EXFIL_HOLD:
                self._skip_until_delim = ")"
            return False, len(window), seg_out, det, 0, False, hit_block, reason

        if final:
            return False, rem_len, rem, None, 0, False, False, ""
        return False, 1, "]", None, 0, False, False, ""

    def _flush_pending_out(self, *, force_all: bool) -> tuple[str, list[Detection], bool, str]:
        """Inspect and flush ready prefix of ``_pending_out`` using bounded lookahead holdback."""
        if not self._pending_out:
            return "", [], False, ""

        pending_len = len(self._pending_out)
        if not force_all:
            if pending_len <= _LOOKAHEAD_HOLD_CHARS:
                return "", [], False, ""
            flush_upto = pending_len - _LOOKAHEAD_HOLD_CHARS
            if (
                flush_upto > 0
                and not self._pending_out[flush_upto - 1].isspace()
                and not self._pending_out[flush_upto].isspace()
            ):
                ws_idx = max(
                    self._pending_out.rfind(" ", 0, flush_upto),
                    self._pending_out.rfind("\n", 0, flush_upto),
                    self._pending_out.rfind("\t", 0, flush_upto),
                )
                if ws_idx >= 0:
                    flush_upto = ws_idx + 1
                elif pending_len <= _MAX_LOOKAHEAD_HOLD_CHARS:
                    return "", [], False, ""
        else:
            flush_upto = pending_len

        if self._output_pipeline is None:
            emitted = self._pending_out
            self._pending_out = ""
            return emitted, [], False, ""

        prev_tail = self._rolling_raw[-_ROLLING_WINDOW_CHARS:]
        combined = prev_tail + self._pending_out
        inspection: InspectionResult = self._output_pipeline.inspect(
            InspectionRequest(
                text=combined,
                control_point=ControlPoint.FINAL_OUTPUT,
                session_id=self._session_id,
                principal=self._principal,
                policy_profile=self._policy_profile,
                direction="output",
            )
        )
        new_dets = [det for det in inspection.detections if det.end > len(prev_tail)]

        if new_dets and self._policy_engine is not None:
            effective_inspection = inspection.model_copy(update={"detections": tuple(new_dets)})
            decision = self._policy_engine.evaluate(
                effective_inspection,
                direction="output",
                principal=self._principal,
                profile=self._policy_profile,
            )
            if decision.action in (Action.BLOCK, Action.REQUIRE_APPROVAL):
                self._pending_out = ""
                return "", new_dets, True, (decision.reason or "Blocked by output guardrail")

            if not force_all:
                for det in new_dets:
                    local_s = max(0, det.start - len(prev_tail))
                    local_e = max(local_s, det.end - len(prev_tail))
                    if (
                        local_s < flush_upto
                        and (local_e > flush_upto or local_e >= pending_len)
                        and pending_len <= _MAX_LOOKAHEAD_HOLD_CHARS
                    ):
                        flush_upto = local_s
                if flush_upto <= 0:
                    return "", [], False, ""

            emit_dets = [d for d in new_dets if (d.start - len(prev_tail)) < flush_upto]
            mask_dets = [
                d
                for d in emit_dets
                if _resolve_detection_action(d, decision) in (Action.MASK, Action.MASK_REVERSIBLE)
            ]
            to_emit = self._pending_out[:flush_upto]
            self._pending_out = self._pending_out[flush_upto:]
            redacted_emit = (
                _redact_local_detections(to_emit, mask_dets, len(prev_tail))
                if mask_dets
                else to_emit
            )
            self._rolling_raw = (prev_tail + redacted_emit)[-_ROLLING_WINDOW_CHARS:]
            return redacted_emit, emit_dets, False, ""

        emit_dets = [d for d in new_dets if (d.start - len(prev_tail)) < flush_upto]
        to_emit = self._pending_out[:flush_upto]
        self._pending_out = self._pending_out[flush_upto:]
        self._rolling_raw = (prev_tail + to_emit)[-_ROLLING_WINDOW_CHARS:]
        return to_emit, emit_dets, False, ""

    def _inspect_flushed_segments(
        self,
        segments: list[tuple[str, bool]],
        *,
        final: bool = False,
    ) -> tuple[str, list[Detection], bool, str]:
        """Run optional output pipeline and policy engine with lookahead holdback."""
        if self._output_pipeline is None:
            return "".join(seg for seg, _ in segments), [], False, ""

        out_parts: list[str] = []
        collected_dets: list[Detection] = []

        for seg_text, is_restored in segments:
            if not seg_text:
                continue
            if is_restored:
                flushed, f_dets, f_blocked, f_reason = self._flush_pending_out(force_all=True)
                if f_dets:
                    collected_dets.extend(f_dets)
                if f_blocked:
                    return "".join(out_parts), collected_dets, True, f_reason
                if flushed:
                    out_parts.append(flushed)
                out_parts.append(seg_text)
                self._rolling_raw = (self._rolling_raw + "[PKR]")[-_ROLLING_WINDOW_CHARS:]
            else:
                self._pending_out += seg_text

        tail_out, tail_dets, tail_blocked, tail_reason = self._flush_pending_out(force_all=final)
        if tail_dets:
            collected_dets.extend(tail_dets)
        if tail_blocked:
            return "".join(out_parts), collected_dets, True, tail_reason
        if tail_out:
            out_parts.append(tail_out)

        return "".join(out_parts), collected_dets, False, ""


__all__ = [
    "MAX_EXFIL_HOLD",
    "MAX_PLACEHOLDER_LEN",
    "SlidingWindowStreamGuard",
    "has_external_exfil_url",
    "mask_text_spans",
    "restore_text_outside_urls",
    "sanitize_echoleak",
]
