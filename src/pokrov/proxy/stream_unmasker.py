"""Sliding-window SSE stream unmasker, EchoLeak guard, and span masking helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.contracts import (
    PLACEHOLDER_PREFIX,
    Action,
    ControlPoint,
    Detection,
    DetectionPipeline,
    EntityCategory,
    InspectionRequest,
    InspectionResult,
    PolicyEngine,
    StreamChunkDecision,
    VaultBackend,
)

MAX_PLACEHOLDER_LEN = 128
MAX_EXFIL_HOLD = 1024
_ROLLING_WINDOW_CHARS = 512

_EXTERNAL_URL_RE = re.compile(r"https?://", re.IGNORECASE)
_ALLOWED_TOKEN_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_:-."
)


def has_external_exfil_url(text: str) -> bool:
    """Return True if text contains an external HTTP/HTTPS URL."""
    return _EXTERNAL_URL_RE.search(text) is not None


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


def _find_html_img_end(text: str, start: int, max_end: int) -> int:
    """Find closing ``>`` for an ``<img ...>`` tag respecting quotes and placeholders."""
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


def _make_exfil_detection(length: int) -> Detection:
    """Build a Zero-Retention Detection for EchoLeak Markdown/HTML exfiltration."""
    return Detection(
        category=EntityCategory.MARKDOWN_EXFIL,
        start=0,
        end=max(0, length),
        confidence=0.99,
        detector="stream_unmasker",
        masked_preview=f"[markdown_exfil:len={length}]",
        compliance_refs=get_compliance_refs(EntityCategory.MARKDOWN_EXFIL),
    )


def sanitize_echoleak(text: str) -> tuple[str, list[Detection]]:
    """Detect and strip EchoLeak ``![...](https?://...)`` and ``<img ...>`` payloads."""
    if not text:
        return "", []

    parts: list[str] = []
    detections: list[Detection] = []
    pos = 0
    length = len(text)

    while pos < length:
        lt_idx = text.find("<", pos)
        bang_idx = text.find("!", pos)
        candidates = [idx for idx in (lt_idx, bang_idx) if idx != -1]
        if not candidates:
            parts.append(text[pos:])
            break

        next_idx = min(candidates)
        if next_idx > pos:
            parts.append(text[pos:next_idx])
            pos = next_idx

        if text[pos] == "!":
            if pos + 1 < length and text[pos + 1] == "[":
                max_end = min(length, pos + MAX_EXFIL_HOLD)
                close_br = _find_markdown_close_bracket(text, pos + 2, max_end)
                if close_br >= 0 and close_br + 1 < max_end and text[close_br + 1] == "(":
                    close_par = _find_markdown_close_paren(text, close_br + 2, max_end)
                    if close_par >= 0:
                        url_part = text[close_br + 2 : close_par]
                        if has_external_exfil_url(url_part):
                            payload_len = close_par + 1 - pos
                            detections.append(
                                Detection(
                                    category=EntityCategory.MARKDOWN_EXFIL,
                                    start=pos,
                                    end=close_par + 1,
                                    confidence=0.99,
                                    detector="echoleak_guard",
                                    masked_preview=f"[markdown_exfil:len={payload_len}]",
                                    compliance_refs=get_compliance_refs(
                                        EntityCategory.MARKDOWN_EXFIL
                                    ),
                                )
                            )
                            pos = close_par + 1
                            continue
            parts.append(text[pos])
            pos += 1
            continue

        # text[pos] == "<"
        if (
            pos + 4 < length
            and text[pos : pos + 4].lower() == "<img"
            and (text[pos + 4].isspace() or text[pos + 4] in ("/", ">"))
        ):
            max_end = min(length, pos + MAX_EXFIL_HOLD)
            gt_idx = _find_html_img_end(text, pos + 4, max_end)
            tag_end = gt_idx + 1 if gt_idx >= 0 else max_end
            tag_slice = text[pos:tag_end]
            if has_external_exfil_url(tag_slice):
                detections.append(
                    Detection(
                        category=EntityCategory.MARKDOWN_EXFIL,
                        start=pos,
                        end=tag_end,
                        confidence=0.99,
                        detector="echoleak_guard",
                        masked_preview=f"[markdown_exfil:len={len(tag_slice)}]",
                        compliance_refs=get_compliance_refs(EntityCategory.MARKDOWN_EXFIL),
                    )
                )
                pos = tag_end
                continue

        parts.append(text[pos])
        pos += 1

    return "".join(parts), detections


def mask_text_spans(
    text: str,
    detections: Sequence[Detection],
    *,
    action: Action,
    vault: VaultBackend,
    session_id: str,
    allowed_categories: set[EntityCategory] | None = None,
) -> tuple[str, int]:
    """Replace detected spans in ``text`` via Vault or static redaction."""
    if not text or not detections:
        return text, 0

    text_len = len(text)
    valid_spans: list[tuple[int, int, EntityCategory]] = []
    for det in detections:
        if allowed_categories and det.category not in allowed_categories:
            continue
        start = max(0, min(det.start, text_len))
        end = max(start, min(det.end, text_len))
        if end > start:
            valid_spans.append((start, end, det.category))

    if not valid_spans:
        return text, 0

    valid_spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    merged: list[tuple[int, int, EntityCategory]] = []
    for start, end, category in valid_spans:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, prev_cat = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_cat)
        else:
            merged.append((start, end, category))

    parts: list[str] = []
    cursor = 0
    replaced = 0
    for start, end, category in merged:
        if start > cursor:
            parts.append(text[cursor:start])
        raw_val = text[start:end]
        if action == Action.MASK_REVERSIBLE:
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
    ) -> None:
        self._vault = vault
        self._session_id = session_id
        self._output_pipeline = output_pipeline
        self._policy_engine = policy_engine
        self._policy_profile = policy_profile
        self._principal = principal

        self._buffer = ""
        self._rolling_raw = ""
        self._skip_until_delim: str | None = None
        self._aborted_reason = ""

    def feed(self, chunk: str) -> StreamChunkDecision:
        """Consume a streaming chunk and emit any verified safe/unmasked prefix."""
        if not chunk:
            return StreamChunkDecision(emit_text="")

        self._buffer += chunk
        return self._drain(final=False)

    def flush(self) -> StreamChunkDecision:
        """Flush remaining buffered characters at end-of-stream."""
        return self._drain(final=True)

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
            if not self._buffer:
                return StreamChunkDecision(
                    emit_text="",
                    blocked=True,
                    block_reason=block_reason,
                )

        buf = self._buffer
        length = len(buf)
        pos = 0

        while pos < length:
            lt_idx = buf.find("<", pos)
            bang_idx = buf.find("!", pos)
            candidates = [idx for idx in (lt_idx, bang_idx) if idx != -1]
            if not candidates:
                segments.append((buf[pos:], False))
                pos = length
                break

            next_idx = min(candidates)
            if next_idx > pos:
                segments.append((buf[pos:next_idx], False))
                pos = next_idx

            rem = buf[pos:]
            if rem[0] == "<":
                held, step, seg, det, was_unmasked, hit_block, reason = self._process_lt(
                    rem, final=final
                )
            else:
                held, step, seg, det, was_unmasked, hit_block, reason = self._process_bang(
                    rem, final=final
                )

            if held:
                break

            if det is not None:
                detections.append(det)
            if was_unmasked:
                unmasked_count += 1
            if hit_block:
                blocked = True
                block_reason = reason
                pos += step
                if self._skip_until_delim is not None:
                    self._aborted_reason = reason
                break

            if seg:
                segments.append((seg, was_unmasked))
            pos += step

        self._buffer = buf[pos:]

        emit_text, pipe_dets, pipe_blocked, pipe_reason = self._inspect_flushed_segments(segments)
        if pipe_dets:
            detections.extend(pipe_dets)
        if pipe_blocked:
            blocked = True
            block_reason = pipe_reason
            self._buffer = ""

        return StreamChunkDecision(
            emit_text=emit_text,
            blocked=blocked,
            block_reason=block_reason,
            unmasked_count=unmasked_count,
            detections=tuple(detections),
        )

    def _process_lt(
        self, rem: str, *, final: bool
    ) -> tuple[bool, int, str, Detection | None, bool, bool, str]:
        """Handle buffer starting at ``<`` for ``<PKR:...>`` placeholders or ``<img ...>``."""
        rem_len = len(rem)
        prefix_len = len(PLACEHOLDER_PREFIX)

        if rem_len < prefix_len:
            is_pkr_prefix = PLACEHOLDER_PREFIX.startswith(rem)
            is_img_prefix = "<img".startswith(rem.lower())
            if (is_pkr_prefix or is_img_prefix) and not final:
                return True, 0, "", None, False, False, ""
            if final and (is_pkr_prefix or is_img_prefix):
                return False, rem_len, rem, None, False, False, ""
            return False, 1, "<", None, False, False, ""

        if rem.startswith(PLACEHOLDER_PREFIX):
            limit = min(rem_len, MAX_PLACEHOLDER_LEN)
            for idx in range(prefix_len, limit):
                ch = rem[idx]
                if ch == ">":
                    if idx == prefix_len:
                        return False, 1, "<", None, False, False, ""
                    token = rem[: idx + 1]
                    restored = self._vault.retrieve(self._session_id, token)
                    if restored is not None:
                        return False, idx + 1, restored, None, True, False, ""
                    return False, idx + 1, token, None, False, False, ""
                if ch not in _ALLOWED_TOKEN_CHARS:
                    return False, 1, "<", None, False, False, ""
            if rem_len >= MAX_PLACEHOLDER_LEN:
                return False, 1, "<", None, False, False, ""
            if not final:
                return True, 0, "", None, False, False, ""
            return False, rem_len, rem, None, False, False, ""

        if rem[:4].lower() == "<img" and (rem[4].isspace() or rem[4] in ("/", ">")):
            max_end = min(rem_len, MAX_EXFIL_HOLD)
            gt_idx = _find_html_img_end(rem, 4, max_end)
            if gt_idx >= 0:
                tag = rem[: gt_idx + 1]
                if has_external_exfil_url(tag):
                    return (
                        False,
                        gt_idx + 1,
                        "",
                        _make_exfil_detection(len(tag)),
                        False,
                        True,
                        "EchoLeak protection: external HTTP/HTTPS URL blocked in HTML <img> tag",
                    )
                return False, gt_idx + 1, tag, None, False, False, ""

            if rem_len < MAX_EXFIL_HOLD and not final:
                return True, 0, "", None, False, False, ""

            window = rem[:max_end]
            if has_external_exfil_url(window):
                if not final and rem_len >= MAX_EXFIL_HOLD:
                    self._skip_until_delim = ">"
                return (
                    False,
                    len(window),
                    "",
                    _make_exfil_detection(len(window)),
                    False,
                    True,
                    "EchoLeak protection: external HTTP/HTTPS URL blocked in HTML <img> tag",
                )
            if final:
                return False, rem_len, rem, None, False, False, ""
            return False, 1, "<", None, False, False, ""

        return False, 1, "<", None, False, False, ""

    def _process_bang(
        self, rem: str, *, final: bool
    ) -> tuple[bool, int, str, Detection | None, bool, bool, str]:
        """Handle buffer starting at ``!`` for ``![alt](url)`` Markdown exfiltration."""
        rem_len = len(rem)
        if rem_len == 1:
            if not final:
                return True, 0, "", None, False, False, ""
            return False, 1, "!", None, False, False, ""

        if rem[1] != "[":
            return False, 1, "!", None, False, False, ""

        max_end = min(rem_len, MAX_EXFIL_HOLD)
        close_br = _find_markdown_close_bracket(rem, 2, max_end)
        if close_br == -2:
            return False, 1, "!", None, False, False, ""
        if close_br == -1:
            if rem_len < MAX_EXFIL_HOLD and not final:
                return True, 0, "", None, False, False, ""
            if final:
                return False, rem_len, rem, None, False, False, ""
            return False, 1, "!", None, False, False, ""

        after_br = close_br + 1
        if after_br >= rem_len:
            if rem_len < MAX_EXFIL_HOLD and not final:
                return True, 0, "", None, False, False, ""
            return False, rem_len, rem, None, False, False, ""

        if rem[after_br] != "(":
            return False, 1, "!", None, False, False, ""

        close_par = _find_markdown_close_paren(rem, after_br + 1, max_end)
        if close_par == -2:
            return False, 1, "!", None, False, False, ""
        if close_par >= 0:
            md_img = rem[: close_par + 1]
            url_part = rem[after_br + 1 : close_par]
            if has_external_exfil_url(url_part):
                return (
                    False,
                    close_par + 1,
                    "",
                    _make_exfil_detection(len(md_img)),
                    False,
                    True,
                    "EchoLeak protection: external HTTP/HTTPS URL blocked in Markdown image",
                )
            return False, close_par + 1, md_img, None, False, False, ""

        if rem_len < MAX_EXFIL_HOLD and not final:
            return True, 0, "", None, False, False, ""

        window = rem[:max_end]
        url_slice = window[after_br + 1 :]
        if has_external_exfil_url(url_slice):
            if not final and rem_len >= MAX_EXFIL_HOLD:
                self._skip_until_delim = ")"
            return (
                False,
                len(window),
                "",
                _make_exfil_detection(len(window)),
                False,
                True,
                "EchoLeak protection: external HTTP/HTTPS URL blocked in Markdown image",
            )
        if final:
            return False, rem_len, rem, None, False, False, ""
        return False, 1, "!", None, False, False, ""

    def _inspect_flushed_segments(
        self,
        segments: list[tuple[str, bool]],
    ) -> tuple[str, list[Detection], bool, str]:
        """Run optional output pipeline and policy engine on non-restored segments."""
        if not segments:
            return "", [], False, ""

        if self._output_pipeline is None:
            return "".join(seg for seg, _ in segments), [], False, ""

        out_parts: list[str] = []
        collected_dets: list[Detection] = []
        blocked = False
        block_reason = ""

        for seg_text, is_restored in segments:
            if not seg_text:
                continue
            if is_restored:
                out_parts.append(seg_text)
                continue

            prev_tail = self._rolling_raw[-_ROLLING_WINDOW_CHARS:]
            combined = prev_tail + seg_text
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
            self._rolling_raw = combined[-_ROLLING_WINDOW_CHARS:]

            if not new_dets:
                out_parts.append(seg_text)
                continue

            self._rolling_raw = ""
            collected_dets.extend(new_dets)

            if self._policy_engine is None:
                out_parts.append(seg_text)
                continue

            effective_inspection = inspection.model_copy(update={"detections": tuple(new_dets)})
            decision = self._policy_engine.evaluate(
                effective_inspection,
                direction="output",
                principal=self._principal,
                profile=self._policy_profile,
            )
            if decision.action in (Action.BLOCK, Action.REQUIRE_APPROVAL):
                blocked = True
                block_reason = decision.reason or "Blocked by output guardrail"
                break
            if decision.action in (Action.MASK, Action.MASK_REVERSIBLE):
                out_parts.append(_redact_local_detections(seg_text, new_dets, len(prev_tail)))
            else:
                out_parts.append(seg_text)

        return "".join(out_parts), collected_dets, blocked, block_reason


__all__ = [
    "MAX_EXFIL_HOLD",
    "MAX_PLACEHOLDER_LEN",
    "SlidingWindowStreamGuard",
    "has_external_exfil_url",
    "mask_text_spans",
    "sanitize_echoleak",
]
