"""Multi-layer detection pipeline with SHA-256 cache, early-exit, and multi-turn window."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pokrov.core.compliance_map import aggregate_compliance_refs, get_category_severity
from pokrov.core.config import PokrovConfig, ProfileConfig, policy_hash
from pokrov.core.contracts import (
    SEVERITY_ORDER,
    Action,
    BaseDetector,
    ControlPoint,
    Detection,
    EntityCategory,
    InspectionRequest,
    InspectionResult,
    MessageItem,
    NormalizedSegment,
    SpanSource,
    TextNormalizer,
    max_severity,
)
from pokrov.detectors.ner import RuNerDetector
from pokrov.detectors.normalizer import AntiEvasionNormalizer, collapse_split_digits
from pokrov.detectors.prompt_injection import OnnxPromptInjectionDetector, PromptInjectionDetector
from pokrov.detectors.ru_pii import RuPiiDetector
from pokrov.detectors.secrets import SecretsDetector

# Categories belonging to the injection / exfiltration / policy-violation wrapper domain.
# Distinct from entity/credential categories so e.g. a Markdown exfil URL carrying a
# secret or passport preserves both the exfil finding and the leaked entity finding.
_ATTACK_WRAPPER_CATEGORIES: frozenset[EntityCategory] = frozenset(
    {
        EntityCategory.PROMPT_INJECTION,
        EntityCategory.INDIRECT_INJECTION,
        EntityCategory.MARKDOWN_EXFIL,
        EntityCategory.TOXIC_OR_LEAK,
        EntityCategory.MCP_POLICY_VIOLATION,
        EntityCategory.INFRA_MISCONFIG,
    }
)

# Budget safety margin: skip Layer 2 once >= 90% of time_budget_ms has been spent.
_BUDGET_EXIT_RATIO = 0.9

# Block size for O(M log M) chunked disjoint interval index.
_INTERVAL_CHUNK_SIZE = 512

_CacheKey = tuple[str, str, str, bool, bool, str, str, str]
_MultiTurnCacheKey = tuple[str, bool, tuple[tuple[int, str, str], ...]]


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    """Zero-Retention cache entry holding only detection metadata and expiry."""

    detections: tuple[Detection, ...]
    evasion_detected: bool
    layers_executed: tuple[int, ...]
    early_exited: bool
    expires_at: float


@dataclass(frozen=True, slots=True)
class _MultiTurnCacheEntry:
    """Zero-Retention multi-turn window cache entry (metadata + expiry only)."""

    detections: tuple[Detection, ...]
    layer2_ran: bool
    expires_at: float


@dataclass(frozen=True, slots=True)
class _WindowTurn:
    """Normalized turn metadata for in-memory multi-turn sliding window."""

    position: int
    message_index: int
    role: str
    raw_text: str
    primary: NormalizedSegment


@dataclass(frozen=True, slots=True)
class _DefaultLayer1Bundle:
    """Shared default Layer 1 detector instances."""

    secrets: SecretsDetector
    prompt_injection: PromptInjectionDetector
    ru_pii_empty: RuPiiDetector


class _DisjointIntervalSet:
    """Chunked non-overlapping interval set supporting O(log M) overlap checks and inserts."""

    __slots__ = ("_block_first_starts", "_ends_blocks", "_starts_blocks", "_zero_points")

    def __init__(self) -> None:
        self._block_first_starts: list[int] = []
        self._starts_blocks: list[list[int]] = []
        self._ends_blocks: list[list[int]] = []
        self._zero_points: set[int] = set()

    def overlaps(self, start: int, end: int) -> bool:
        """Return True if ``[start, end)`` overlaps any stored interval."""
        if start == end:
            return start in self._zero_points

        if not self._block_first_starts:
            return False

        b_idx = bisect_right(self._block_first_starts, start) - 1
        if b_idx < 0:
            return self._starts_blocks[0][0] < end

        starts = self._starts_blocks[b_idx]
        ends = self._ends_blocks[b_idx]
        pos = bisect_right(starts, start)
        if pos > 0 and ends[pos - 1] > start:
            return True
        if pos < len(starts) and starts[pos] < end:
            return True
        return bool(
            b_idx + 1 < len(self._starts_blocks) and self._starts_blocks[b_idx + 1][0] < end
        )

    def add(self, start: int, end: int) -> None:
        """Insert a non-overlapping interval ``[start, end)``."""
        if start == end:
            self._zero_points.add(start)
            return

        if not self._block_first_starts:
            self._block_first_starts.append(start)
            self._starts_blocks.append([start])
            self._ends_blocks.append([end])
            return

        b_idx = bisect_right(self._block_first_starts, start) - 1
        if b_idx < 0:
            b_idx = 0

        starts = self._starts_blocks[b_idx]
        ends = self._ends_blocks[b_idx]
        pos = bisect_right(starts, start)
        starts.insert(pos, start)
        ends.insert(pos, end)
        if pos == 0:
            self._block_first_starts[b_idx] = start

        if len(starts) > _INTERVAL_CHUNK_SIZE:
            mid = len(starts) // 2
            right_starts = starts[mid:]
            right_ends = ends[mid:]
            del starts[mid:]
            del ends[mid:]
            self._starts_blocks.insert(b_idx + 1, right_starts)
            self._ends_blocks.insert(b_idx + 1, right_ends)
            self._block_first_starts.insert(b_idx + 1, right_starts[0])


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detection_rank_key(det: Detection) -> tuple[int, int, float, int, int, str]:
    """Rank detections preferring built-in over custom, higher severity, confidence, then length."""
    is_custom = 1 if det.category == EntityCategory.CUSTOM_PATTERN else 0
    sev = SEVERITY_ORDER[get_category_severity(det.category)]
    return (is_custom, -sev, -det.confidence, -(det.end - det.start), det.start, det.category.value)


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    if a_start == a_end or b_start == b_end:
        return a_start == b_start and a_end == b_end
    return a_start < b_end and b_start < a_end


def _parse_inner_span(meta: dict[str, str]) -> tuple[int, int] | None:
    raw = meta.get("inner_span")
    if not raw or ":" not in raw:
        return None
    left, _, right = raw.partition(":")
    if left.isdigit() and right.isdigit():
        return (int(left), int(right))
    return None


def _decoded_blobs_conflict(candidate: Detection, existing: Detection) -> bool:
    """Return True if two overlapping DECODED_BLOB detections represent the same entity (P2-02)."""
    cand_inner = _parse_inner_span(candidate.metadata)
    ex_inner = _parse_inner_span(existing.metadata)
    if (
        cand_inner is not None
        and ex_inner is not None
        and candidate.evasion_techniques == existing.evasion_techniques
    ):
        if _spans_overlap(cand_inner[0], cand_inner[1], ex_inner[0], ex_inner[1]):
            return candidate.category == existing.category or bool(
                candidate.value_hmac
                and existing.value_hmac
                and candidate.value_hmac == existing.value_hmac
            )
        # Disjoint spans inside the same decoded segment are distinct entities
        if candidate.value_hmac and existing.value_hmac:
            return candidate.value_hmac == existing.value_hmac
        return False

    if candidate.value_hmac and existing.value_hmac:
        return candidate.value_hmac == existing.value_hmac

    if candidate.category != existing.category:
        return False

    if candidate.masked_preview and existing.masked_preview:
        return candidate.masked_preview == existing.masked_preview

    return True


def _pre_deduplicate_exact(detections: Sequence[Detection]) -> list[Detection]:
    """Collapse exact duplicate spans in O(M) time before sorting and interval sweeping."""
    best: dict[tuple[EntityCategory, int, int, SpanSource, str, str], Detection] = {}
    for det in detections:
        if det.span_source == SpanSource.DECODED_BLOB:
            inner = det.metadata.get("inner_span", "")
            disc = inner if inner else f"{det.value_hmac}|{det.masked_preview}"
            blob_disc = (disc, "|".join(det.evasion_techniques))
        else:
            blob_disc = ("", "")
        key = (det.category, det.start, det.end, det.span_source, *blob_disc)
        prev = best.get(key)
        if prev is None or _detection_rank_key(det) < _detection_rank_key(prev):
            best[key] = det
    return list(best.values())


def _deduplicate_detections(detections: Sequence[Detection]) -> list[Detection]:
    """Resolve overlapping detections in O(M log M) time preserving distinct domains & blobs."""
    if len(detections) <= 1:
        return list(detections)

    # Zero-sort O(M) fast path when input is already sorted by start and strictly disjoint
    prev_start = -1
    prev_end = -1
    already_sorted_disjoint = True
    for d in detections:
        if d.start == d.end or d.start < prev_end or d.start < prev_start:
            already_sorted_disjoint = False
            break
        prev_start = d.start
        prev_end = d.end
    if already_sorted_disjoint:
        return list(detections)

    # O(M log M) fast path when detections are strictly disjoint after sorting
    ordered = sorted(detections, key=lambda d: (d.start, d.end, -d.confidence, d.category.value))
    prev_end = -1
    all_disjoint = True
    for d in ordered:
        if d.start == d.end or d.start < prev_end:
            all_disjoint = False
            break
        prev_end = d.end
    if all_disjoint:
        return ordered

    unique = _pre_deduplicate_exact(detections)
    if len(unique) <= 1:
        return unique

    ranked = sorted(unique, key=_detection_rank_key)

    domain_pii = _DisjointIntervalSet()
    domain_attack = _DisjointIntervalSet()
    domain_custom = _DisjointIntervalSet()
    by_category_direct: dict[EntityCategory, _DisjointIntervalSet] = {}
    by_category_blob: dict[EntityCategory, _DisjointIntervalSet] = {}
    exact_non_blob_spans: set[tuple[int, int]] = set()
    accepted_blobs: list[Detection] = []
    accepted: list[Detection] = []

    for cand in ranked:
        s, e = cand.start, cand.end
        cat = cand.category

        if cand.span_source == SpanSource.DECODED_BLOB:
            cat_direct = by_category_direct.get(cat)
            if cat_direct is not None and cat_direct.overlaps(s, e):
                continue
            if any(
                _spans_overlap(s, e, b.start, b.end) and _decoded_blobs_conflict(cand, b)
                for b in accepted_blobs
            ):
                continue
            accepted_blobs.append(cand)
            blob_set = by_category_blob.get(cat)
            if blob_set is None:
                blob_set = _DisjointIntervalSet()
                by_category_blob[cat] = blob_set
            if not blob_set.overlaps(s, e):
                blob_set.add(s, e)
            accepted.append(cand)
            continue

        # Direct or Cross-Turn candidate: check same-category overlap (both direct & blob)
        cat_direct = by_category_direct.get(cat)
        if cat_direct is not None and cat_direct.overlaps(s, e):
            continue
        cat_blob = by_category_blob.get(cat)
        if cat_blob is not None and cat_blob.overlaps(s, e):
            continue

        # Custom patterns live in their own domain so they NEVER suppress built-in PII/secrets.
        if cat == EntityCategory.CUSTOM_PATTERN:
            if domain_custom.overlaps(s, e):
                continue
            domain_custom.add(s, e)
        elif cand.detector == "prompt_injection_onnx":
            # Whole-message ONNX classifier shouldn't suppress or be suppressed by localized PII.
            if (s, e) in exact_non_blob_spans:
                continue
            exact_non_blob_spans.add((s, e))
        elif cat in _ATTACK_WRAPPER_CATEGORIES:
            if domain_attack.overlaps(s, e) or (s, e) in exact_non_blob_spans:
                continue
            domain_attack.add(s, e)
            exact_non_blob_spans.add((s, e))
        else:
            if domain_pii.overlaps(s, e) or (s, e) in exact_non_blob_spans:
                continue
            domain_pii.add(s, e)
            exact_non_blob_spans.add((s, e))

        if cat_direct is None:
            cat_direct = _DisjointIntervalSet()
            by_category_direct[cat] = cat_direct
        cat_direct.add(s, e)
        accepted.append(cand)

    accepted.sort(key=lambda d: (d.start, d.end, -d.confidence, d.category.value))
    return accepted


def _is_budget_exhausted(elapsed_ms: float, time_budget_ms: float) -> bool:
    if time_budget_ms <= 0.0:
        return True
    return elapsed_ms >= time_budget_ms * _BUDGET_EXIT_RATIO


def _resolve_message_indices(messages: Sequence[MessageItem]) -> list[int]:
    """Return effective message indices, falling back to positional indices when non-unique."""
    if not messages:
        return []
    raw_indices = [msg.message_index for msg in messages]
    if len(set(raw_indices)) == len(raw_indices):
        return raw_indices
    return list(range(len(messages)))


def _build_joined_segment(
    turns: Sequence[_WindowTurn],
    separator: str,
) -> tuple[str, NormalizedSegment, list[tuple[int, int, int]]]:
    """Join normalized window turns and build an exact index_map back to raw_joined."""
    raw_parts: list[str] = []
    norm_parts: list[str] = []
    index_map: list[int] = []
    turn_ranges: list[tuple[int, int, int]] = []
    evasions: list[str] = []

    raw_cursor = 0
    sep_len = len(separator)
    for idx, turn in enumerate(turns):
        if idx > 0 and sep_len > 0:
            raw_parts.append(separator)
            norm_parts.append(separator)
            index_map.extend(range(raw_cursor, raw_cursor + sep_len))
            raw_cursor += sep_len

        raw_len = len(turn.raw_text)
        raw_start = raw_cursor
        raw_end = raw_start + raw_len
        turn_ranges.append((turn.message_index, raw_start, raw_end))

        raw_parts.append(turn.raw_text)
        norm_text = turn.primary.text
        if norm_text:
            norm_parts.append(norm_text)
            if turn.primary.index_map:
                index_map.extend(raw_start + offset for offset in turn.primary.index_map)
            else:
                index_map.extend(range(raw_start, raw_start + len(norm_text)))

        for tech in turn.primary.evasion_techniques:
            if tech not in evasions:
                evasions.append(tech)

        raw_cursor = raw_end

    raw_joined = "".join(raw_parts)
    segment = NormalizedSegment(
        text="".join(norm_parts),
        span_source=SpanSource.DIRECT,
        index_map=tuple(index_map),
        evasion_techniques=tuple(evasions),
    )
    return raw_joined, segment, turn_ranges


def _run_detector_preserving_blob_inner(
    detector: BaseDetector,
    seg: NormalizedSegment,
    raw_text: str,
) -> list[Detection]:
    """Run ``detector`` on ``seg``, attaching ``inner_span`` metadata for DECODED_BLOBs."""
    if seg.span_source != SpanSource.DECODED_BLOB:
        return detector.detect(seg, raw_text)

    probe = NormalizedSegment(
        text=seg.text,
        span_source=SpanSource.DIRECT,
        index_map=seg.index_map,
        evasion_techniques=seg.evasion_techniques,
    )
    raw_dets = detector.detect(probe, seg.text)
    if not raw_dets:
        return []

    remapped: list[Detection] = []
    for det in raw_dets:
        meta = dict(det.metadata)
        meta["inner_span"] = f"{det.start}:{det.end}"
        remapped.append(
            det.model_copy(
                update={
                    "start": seg.outer_start,
                    "end": seg.outer_end,
                    "span_source": SpanSource.DECODED_BLOB,
                    "metadata": meta,
                }
            )
        )
    return remapped


class PokrovDetectionPipeline:
    """Multi-layer detection pipeline satisfying ``DetectionPipeline`` Protocol."""

    def __init__(
        self,
        config: PokrovConfig,
        normalizer: TextNormalizer | None = None,
        layer1_detectors: Sequence[BaseDetector] | None = None,
        layer2_detectors: Sequence[BaseDetector] | None = None,
        hmac_fn: Callable[[str], str] | None = None,
        max_cache_entries: int = 4096,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        if max_cache_entries < 0:
            raise ValueError(f"max_cache_entries must be >= 0, got {max_cache_entries}")

        self._config = config
        self._normalizer: TextNormalizer = (
            normalizer if normalizer is not None else AntiEvasionNormalizer()
        )
        self._hmac_fn = hmac_fn
        self._max_cache_entries = max_cache_entries
        self._time_fn: Callable[[], float] = time_fn if time_fn is not None else time.perf_counter

        self._lock = threading.RLock()
        self._cache: OrderedDict[_CacheKey, _CacheEntry] = OrderedDict()
        self._mt_cache: OrderedDict[_MultiTurnCacheKey, _MultiTurnCacheEntry] = OrderedDict()
        self._policy_hash_cache: dict[str, tuple[ProfileConfig, str]] = {}

        self._custom_layer1 = layer1_detectors is not None
        self._fixed_layer1: tuple[BaseDetector, ...] = (
            tuple(layer1_detectors) if layer1_detectors is not None else ()
        )
        self._default_l1: _DefaultLayer1Bundle | None = None
        self._layer1_by_profile: dict[str, tuple[Sequence[object], tuple[BaseDetector, ...]]] = {}

        if not self._custom_layer1:
            self._default_l1 = _DefaultLayer1Bundle(
                secrets=SecretsDetector(hmac_fn=hmac_fn),
                prompt_injection=PromptInjectionDetector(hmac_fn=hmac_fn),
                ru_pii_empty=RuPiiDetector(custom_patterns=(), hmac_fn=hmac_fn),
            )
            active_name = config.active_profile
            if active_name in config.profiles or not config.profiles:
                active_prof = config.get_profile(active_name if config.profiles else None)
                self._get_layer1_detectors(active_name, active_prof)

        self._custom_layer2 = layer2_detectors is not None
        self._layer2_detectors: tuple[BaseDetector, ...] = (
            tuple(layer2_detectors)
            if layer2_detectors is not None
            else (
                RuNerDetector(hmac_fn=hmac_fn),
                OnnxPromptInjectionDetector(hmac_fn=hmac_fn),
            )
        )

    def _resolve_profile(self, request: InspectionRequest) -> tuple[str, ProfileConfig]:
        fields_set = getattr(request, "model_fields_set", None)
        explicit_profile = (
            fields_set is not None
            and "policy_profile" in fields_set
            and bool(request.policy_profile)
        )
        if explicit_profile:
            name = request.policy_profile
            return name, self._config.get_profile(name)

        active = self._config.active_profile
        if active in self._config.profiles:
            return active, self._config.get_profile(active)
        if not self._config.profiles:
            return active or request.policy_profile, ProfileConfig()
        if request.policy_profile in self._config.profiles:
            return request.policy_profile, self._config.get_profile(request.policy_profile)
        return active, self._config.get_profile(active)

    def _get_policy_hash(self, profile_name: str, profile: ProfileConfig) -> str:
        with self._lock:
            cached = self._policy_hash_cache.get(profile_name)
            if cached is not None and cached[0] is profile:
                return cached[1]
            digest = policy_hash(self._config, profile_name)
            self._policy_hash_cache[profile_name] = (profile, digest)
            return digest

    def _get_layer1_detectors(
        self,
        profile_name: str,
        profile: ProfileConfig,
    ) -> tuple[BaseDetector, ...]:
        if self._custom_layer1 or self._default_l1 is None:
            return self._fixed_layer1

        bundle = self._default_l1
        with self._lock:
            cached = self._layer1_by_profile.get(profile_name)
            if cached is not None and cached[0] == profile.custom_patterns:
                return cached[1]

            pii_detector = (
                RuPiiDetector(custom_patterns=profile.custom_patterns, hmac_fn=self._hmac_fn)
                if profile.custom_patterns
                else bundle.ru_pii_empty
            )
            detectors: tuple[BaseDetector, ...] = (
                pii_detector,
                bundle.secrets,
                bundle.prompt_injection,
            )
            self._layer1_by_profile[profile_name] = (
                tuple(profile.custom_patterns),
                detectors,
            )
            return detectors

    def _select_layer2_detectors(
        self,
        *,
        run_ner: bool,
        run_ml: bool,
    ) -> tuple[BaseDetector, ...]:
        if not (run_ner or run_ml):
            return ()

        selected: list[BaseDetector] = []
        for det in self._layer2_detectors:
            if isinstance(det, RuNerDetector):
                if run_ner:
                    selected.append(det)
            elif isinstance(det, OnnxPromptInjectionDetector):
                if run_ml:
                    selected.append(det)
            else:
                selected.append(det)
        return tuple(selected)

    def _has_early_exit_trigger(
        self,
        detections: Sequence[Detection],
        profile: ProfileConfig,
        request: InspectionRequest,
    ) -> bool:
        """Return True ONLY when an enabled, above-threshold rule resolves to Action.BLOCK."""
        if not detections:
            return False

        override_rules = profile.principal_overrides.get(request.principal, {})
        custom_by_name = (
            {r.name: r for r in profile.custom_patterns} if profile.custom_patterns else {}
        )

        for det in detections:
            rule_name = det.metadata.get("rule_name")
            if rule_name and rule_name in custom_by_name:
                custom_rule = custom_by_name[rule_name]
                if det.category in override_rules:
                    ovr = override_rules[det.category]
                    if not ovr.enabled or det.confidence < ovr.min_confidence:
                        continue
                    eff_action = (
                        ovr.output_action
                        if request.direction == "output" and ovr.output_action is not None
                        else ovr.action
                    )
                else:
                    cat_rule = profile.rules.get(det.category)
                    if cat_rule is not None and not cat_rule.enabled:
                        continue
                    if det.confidence + 1e-9 < custom_rule.confidence:
                        continue
                    eff_action = custom_rule.action
                if eff_action == Action.BLOCK:
                    return True
                continue

            rule = (
                override_rules[det.category]
                if det.category in override_rules
                else profile.rules.get(det.category)
            )
            if rule is None or not rule.enabled or det.confidence < rule.min_confidence:
                continue
            effective_action = (
                rule.output_action
                if request.direction == "output" and rule.output_action is not None
                else rule.action
            )
            if effective_action == Action.BLOCK:
                return True
        return False

    def _cache_get(self, key: _CacheKey, now: float) -> _CacheEntry | None:
        if self._max_cache_entries <= 0:
            return None
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if now >= entry.expires_at:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry

    def _cache_put(self, key: _CacheKey, entry: _CacheEntry) -> None:
        if self._max_cache_entries <= 0:
            return
        with self._lock:
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
                return
            while len(self._cache) >= self._max_cache_entries:
                self._cache.popitem(last=False)
            self._cache[key] = entry

    def _mt_cache_get(self, key: _MultiTurnCacheKey, now: float) -> _MultiTurnCacheEntry | None:
        if self._max_cache_entries <= 0:
            return None
        with self._lock:
            entry = self._mt_cache.get(key)
            if entry is None:
                return None
            if now >= entry.expires_at:
                del self._mt_cache[key]
                return None
            self._mt_cache.move_to_end(key)
            return entry

    def _mt_cache_put(self, key: _MultiTurnCacheKey, entry: _MultiTurnCacheEntry) -> None:
        if self._max_cache_entries <= 0:
            return
        with self._lock:
            if key in self._mt_cache:
                self._mt_cache[key] = entry
                self._mt_cache.move_to_end(key)
                return
            while len(self._mt_cache) >= self._max_cache_entries:
                self._mt_cache.popitem(last=False)
            self._mt_cache[key] = entry

    def _run_layer1_on_segments(
        self,
        segments: Sequence[NormalizedSegment],
        raw_text: str,
        layer1_detectors: Sequence[BaseDetector],
    ) -> list[Detection]:
        detections: list[Detection] = []
        for seg in segments:
            for detector in layer1_detectors:
                detections.extend(_run_detector_preserving_blob_inner(detector, seg, raw_text))
                if isinstance(detector, RuPiiDetector) and seg.text:
                    collapsed = collapse_split_digits(seg, min_digits=3)
                    if collapsed.text != seg.text:
                        detections.extend(
                            _run_detector_preserving_blob_inner(detector, collapsed, raw_text)
                        )
        return detections

    def _inspect_single_message(
        self,
        *,
        msg: MessageItem,
        request: InspectionRequest,
        profile: ProfileConfig,
        p_hash: str,
        layer1_detectors: Sequence[BaseDetector],
        eligible_layer2: Sequence[BaseDetector],
        layer2_enabled: bool,
        run_ner: bool,
        run_ml: bool,
        started: float,
    ) -> tuple[list[Detection], bool, tuple[int, ...], bool, bool]:
        """Inspect one message using cache -> normalizer -> Layer 1 -> conditional Layer 2."""
        if not msg.content:
            return ([], False, (1,), False, False)

        ttl = profile.cache_ttl_seconds
        use_cache = self._max_cache_entries > 0 and ttl > 0
        override_scope = (
            request.principal if request.principal in profile.principal_overrides else ""
        )
        cache_key: _CacheKey | None = None
        if use_cache:
            cache_key = (
                p_hash,
                msg.role,
                msg.control_point.value,
                run_ner,
                run_ml,
                request.direction,
                override_scope,
                _sha256_text(msg.content),
            )
            cached = self._cache_get(cache_key, started)
            if cached is not None:
                return (
                    list(cached.detections),
                    cached.evasion_detected,
                    cached.layers_executed,
                    cached.early_exited,
                    True,
                )

        norm_res = self._normalizer.normalize(msg.content)
        segments: tuple[NormalizedSegment, ...] = (norm_res.primary, *norm_res.decoded_segments)

        l1_detections = self._run_layer1_on_segments(segments, msg.content, layer1_detectors)
        evasion_detected = norm_res.evasion_detected

        all_detections: list[Detection] = l1_detections
        layers_executed: tuple[int, ...] = (1,)
        early_exited = False
        budget_aborted = False

        if layer2_enabled and eligible_layer2:
            trigger_exit = self._has_early_exit_trigger(l1_detections, profile, request)
            elapsed_ms = max(0.0, (self._time_fn() - started) * 1000.0)
            budget_exit = _is_budget_exhausted(elapsed_ms, request.time_budget_ms)
            if trigger_exit or budget_exit:
                early_exited = True
                budget_aborted = budget_exit and not trigger_exit
            else:
                all_detections = list(l1_detections)
                ran_any_l2 = False
                for seg in segments:
                    seg_elapsed = max(0.0, (self._time_fn() - started) * 1000.0)
                    if ran_any_l2 and _is_budget_exhausted(seg_elapsed, request.time_budget_ms):
                        budget_aborted = True
                        break
                    for detector in eligible_layer2:
                        all_detections.extend(
                            _run_detector_preserving_blob_inner(detector, seg, msg.content)
                        )
                        ran_any_l2 = True
                if ran_any_l2:
                    layers_executed = (1, 2)

        deduped = _deduplicate_detections(all_detections)
        if not evasion_detected and any(d.evasion_detected for d in deduped):
            evasion_detected = True

        # Do not cache incomplete results when Layer 2 was skipped solely due to time budget.
        if use_cache and cache_key is not None and not budget_aborted:
            now_store = self._time_fn()
            self._cache_put(
                cache_key,
                _CacheEntry(
                    detections=tuple(deduped),
                    evasion_detected=evasion_detected,
                    layers_executed=layers_executed,
                    early_exited=early_exited,
                    expires_at=now_store + ttl,
                ),
            )

        return (deduped, evasion_detected, layers_executed, early_exited, False)

    def _scan_joined_window_view(
        self,
        *,
        turns: Sequence[_WindowTurn],
        separators: Sequence[str],
        per_message_detections: Sequence[Sequence[Detection]],
        layer1_detectors: Sequence[BaseDetector],
        mt_layer2: Sequence[BaseDetector],
        request: InspectionRequest,
        started: float,
    ) -> tuple[list[Detection], bool]:
        """Scan a sequence of window turns across ``separators`` (plus decoded blobs & Layer 2)."""
        if len(turns) < 2:
            return ([], False)

        _, _, canonical_ranges = _build_joined_segment(turns, " ")
        canon_start_by_msg: dict[int, int] = {
            msg_idx: r_start for msg_idx, r_start, _ in canonical_ranges
        }

        existing_in_window: list[tuple[EntityCategory, int, int]] = []
        for turn, (_, r_start, _) in zip(turns, canonical_ranges, strict=True):
            for det in per_message_detections[turn.position]:
                if det.span_source in (SpanSource.DIRECT, SpanSource.DECODED_BLOB):
                    existing_in_window.append(
                        (det.category, r_start + det.start, r_start + det.end)
                    )

        pass_results: list[tuple[str, list[tuple[int, int, int]], list[Detection]]] = []
        ran_l2 = False

        for sep in separators:
            raw_joined, seg, pass_ranges = _build_joined_segment(turns, sep)
            norm_joined = self._normalizer.normalize(raw_joined)
            all_segs: tuple[NormalizedSegment, ...] = (seg, *norm_joined.decoded_segments)

            dets = self._run_layer1_on_segments(all_segs, raw_joined, layer1_detectors)
            if mt_layer2 and sep in (" ", "\n"):
                elapsed_ms = max(0.0, (self._time_fn() - started) * 1000.0)
                if not _is_budget_exhausted(elapsed_ms, request.time_budget_ms):
                    for s_item in all_segs:
                        for l2_det in mt_layer2:
                            dets.extend(
                                _run_detector_preserving_blob_inner(l2_det, s_item, raw_joined)
                            )
                            ran_l2 = True
            pass_results.append((raw_joined, pass_ranges, dets))

        cross_turn_out: list[Detection] = []
        for raw_joined, pass_ranges, raw_candidates in pass_results:
            joined_len = len(raw_joined)
            for cand in raw_candidates:
                c_start = max(0, min(cand.start, joined_len))
                c_end = max(c_start, min(cand.end, joined_len))
                if c_end <= c_start:
                    continue

                touched: list[tuple[int, int, int, int]] = []
                for idx, (msg_idx, r_start, r_end) in enumerate(pass_ranges):
                    if c_start < r_end and c_end > r_start:
                        loc_s = max(0, c_start - r_start)
                        loc_e = min(r_end - r_start, c_end - r_start)
                        if loc_e > loc_s:
                            touched.append((idx, msg_idx, loc_s, loc_e))

                if not touched:
                    continue

                first_msg_idx, first_loc_s = touched[0][1], touched[0][2]
                last_msg_idx, last_loc_e = touched[-1][1], touched[-1][3]
                canon_start = canon_start_by_msg[first_msg_idx] + first_loc_s
                canon_end = canon_start_by_msg[last_msg_idx] + last_loc_e

                covered_single_turn = any(
                    ex_cat == cand.category
                    and _spans_overlap(canon_start, canon_end, ex_start, ex_end)
                    for ex_cat, ex_start, ex_end in existing_in_window
                )
                if len(touched) == 1 and covered_single_turn:
                    continue

                if len(touched) >= 2:
                    source_indices = tuple(msg_idx for _, msg_idx, _, _ in touched)
                else:
                    turn_pos = touched[0][0]
                    # Verify the single touched turn did not already trigger on its own
                    if self._single_turn_triggers(
                        turn=turns[turn_pos],
                        category=cand.category,
                        layer1_detectors=layer1_detectors,
                        mt_layer2=mt_layer2,
                    ):
                        continue
                    source_indices = self._attribute_context_turns(
                        turns=turns,
                        target_pos=turn_pos,
                        category=cand.category,
                        layer1_detectors=layer1_detectors,
                        mt_layer2=mt_layer2,
                    )

                turn_spans_json = json.dumps(
                    [[msg_idx, loc_s, loc_e] for _, msg_idx, loc_s, loc_e in touched],
                    separators=(",", ":"),
                )
                meta = dict(cand.metadata)
                meta.pop("inner_span", None)
                meta["turn_spans"] = turn_spans_json

                raw_slice = raw_joined[c_start:c_end]
                val_hmac = (
                    cand.value_hmac
                    if (cand.span_source == SpanSource.DECODED_BLOB and cand.value_hmac)
                    else (self._hmac_fn(raw_slice) if self._hmac_fn else "")
                )
                preview = (
                    cand.masked_preview
                    if (cand.span_source == SpanSource.DECODED_BLOB and cand.masked_preview)
                    else f"[{cand.category.value}:len={c_end - c_start}]"
                )

                cross_turn_out.append(
                    cand.model_copy(
                        update={
                            "start": canon_start,
                            "end": canon_end,
                            "span_source": SpanSource.CROSS_TURN,
                            "source_message_indices": source_indices,
                            "masked_preview": preview,
                            "value_hmac": val_hmac,
                            "metadata": meta,
                        }
                    )
                )

        return (cross_turn_out, ran_l2)

    def _single_turn_triggers(
        self,
        *,
        turn: _WindowTurn,
        category: EntityCategory,
        layer1_detectors: Sequence[BaseDetector],
        mt_layer2: Sequence[BaseDetector],
    ) -> bool:
        """Return True if ``turn`` alone triggers ``category`` without adjacent context."""
        dets = self._run_layer1_on_segments((turn.primary,), turn.raw_text, layer1_detectors)
        for l2_det in mt_layer2:
            dets.extend(l2_det.detect(turn.primary, turn.raw_text))
        return any(d.category == category for d in dets)

    def _run_multi_turn_window(
        self,
        *,
        messages: Sequence[MessageItem],
        effective_indices: Sequence[int],
        per_message_detections: Sequence[Sequence[Detection]],
        profile: ProfileConfig,
        layer1_detectors: Sequence[BaseDetector],
        eligible_layer2: Sequence[BaseDetector],
        run_ner: bool,
        request: InspectionRequest,
        p_hash: str,
        now: float,
    ) -> tuple[list[Detection], bool]:
        """Run sliding window over user/tool turns plus assistant context turns."""
        window_size = profile.multi_turn_window
        if window_size < 2 or len(messages) < 2:
            return ([], False)

        ut_entries: list[tuple[int, int, MessageItem]] = [
            (pos, eff_idx, msg)
            for pos, (eff_idx, msg) in enumerate(zip(effective_indices, messages, strict=True))
            if msg.role.lower() in {"user", "tool"} and msg.content
        ]
        if not ut_entries:
            return ([], False)

        recent_ut = ut_entries[-window_size:]
        first_ut_pos = recent_ut[0][0]

        # Include immediately preceding and interleaved assistant messages as context (P1-02)
        ctx_start_pos = first_ut_pos
        while ctx_start_pos > 0 and messages[ctx_start_pos - 1].role.lower() == "assistant":
            ctx_start_pos -= 1

        ctx_entries: list[tuple[int, int, MessageItem]] = [
            (pos, effective_indices[pos], messages[pos])
            for pos in range(ctx_start_pos, len(messages))
            if messages[pos].role.lower() in {"user", "tool", "assistant"} and messages[pos].content
        ]

        if len(recent_ut) < 2 and len(ctx_entries) < 2:
            return ([], False)

        mt_layer2 = tuple(
            det for det in eligible_layer2 if not isinstance(det, OnnxPromptInjectionDetector)
        )
        l2_active = bool(mt_layer2)

        ttl = profile.cache_ttl_seconds
        use_cache = self._max_cache_entries > 0 and ttl > 0
        mt_key: _MultiTurnCacheKey | None = None
        if use_cache:
            mt_key = (
                p_hash,
                l2_active and run_ner,
                tuple(
                    (eff_idx, msg.role.lower(), _sha256_text(msg.content))
                    for _pos, eff_idx, msg in ctx_entries
                ),
            )
            cached_mt = self._mt_cache_get(mt_key, now)
            if cached_mt is not None:
                return (list(cached_mt.detections), cached_mt.layer2_ran)

        turn_by_pos: dict[int, _WindowTurn] = {}
        for pos, eff_idx, msg in ctx_entries:
            turn_by_pos[pos] = _WindowTurn(
                position=pos,
                message_index=eff_idx,
                role=msg.role.lower(),
                raw_text=msg.content,
                primary=self._normalizer.normalize(msg.content).primary,
            )

        ut_turns = [turn_by_pos[pos] for pos, _, _ in recent_ut]
        ctx_turns = [turn_by_pos[pos] for pos, _, _ in ctx_entries]

        combined_candidates: list[Detection] = []
        any_mt_l2_ran = False

        if len(ut_turns) >= 2:
            ut_cands, ut_l2 = self._scan_joined_window_view(
                turns=ut_turns,
                separators=(" ", "\n", ""),
                per_message_detections=per_message_detections,
                layer1_detectors=layer1_detectors,
                mt_layer2=mt_layer2,
                request=request,
                started=now,
            )
            combined_candidates.extend(ut_cands)
            if ut_l2:
                any_mt_l2_ran = True

        if len(ctx_turns) >= 2 and [t.position for t in ctx_turns] != [
            t.position for t in ut_turns
        ]:
            ctx_cands, ctx_l2 = self._scan_joined_window_view(
                turns=ctx_turns,
                separators=(" ", "\n"),
                per_message_detections=per_message_detections,
                layer1_detectors=layer1_detectors,
                mt_layer2=mt_layer2,
                request=request,
                started=now,
            )
            # Filter out ctx_cands whose (category, turn_spans) was already found in ut_turns
            seen_ut_keys = {
                (d.category, d.metadata.get("turn_spans", "")) for d in combined_candidates
            }
            for cand in ctx_cands:
                if (cand.category, cand.metadata.get("turn_spans", "")) not in seen_ut_keys:
                    combined_candidates.append(cand)
            if ctx_l2:
                any_mt_l2_ran = True

        if not combined_candidates:
            if use_cache and mt_key is not None:
                self._mt_cache_put(
                    mt_key,
                    _MultiTurnCacheEntry(
                        detections=(),
                        layer2_ran=any_mt_l2_ran,
                        expires_at=self._time_fn() + ttl,
                    ),
                )
            return ([], any_mt_l2_ran)

        # Deduplicate across views; note canonical start/end differs if ut_turns != ctx_turns,
        # so deduplicate per view or re-project onto ctx_turns canonical ranges first.
        if len(ctx_turns) >= 2 and [t.position for t in ctx_turns] != [
            t.position for t in ut_turns
        ]:
            _, _, full_canon_ranges = _build_joined_segment(ctx_turns, " ")
            full_canon_map = {m_idx: r_s for m_idx, r_s, _ in full_canon_ranges}
            reprojected: list[Detection] = []
            for det in combined_candidates:
                raw_ts = det.metadata.get("turn_spans")
                if raw_ts:
                    spans_parsed: list[list[int]] = json.loads(raw_ts)
                    c_s = full_canon_map[spans_parsed[0][0]] + spans_parsed[0][1]
                    c_e = full_canon_map[spans_parsed[-1][0]] + spans_parsed[-1][2]
                    reprojected.append(det.model_copy(update={"start": c_s, "end": c_e}))
                else:
                    reprojected.append(det)
            combined_candidates = reprojected

        deduped_mt = _deduplicate_detections(combined_candidates)
        if use_cache and mt_key is not None:
            self._mt_cache_put(
                mt_key,
                _MultiTurnCacheEntry(
                    detections=tuple(deduped_mt),
                    layer2_ran=any_mt_l2_ran,
                    expires_at=self._time_fn() + ttl,
                ),
            )
        return (deduped_mt, any_mt_l2_ran)

    def _attribute_context_turns(
        self,
        *,
        turns: Sequence[_WindowTurn],
        target_pos: int,
        category: EntityCategory,
        layer1_detectors: Sequence[BaseDetector],
        mt_layer2: Sequence[BaseDetector],
    ) -> tuple[int, ...]:
        """Identify which adjacent turn(s) provided context for a newly uncovered detection."""

        def _pair_triggers(pair: tuple[_WindowTurn, _WindowTurn]) -> bool:
            for sep in (" ", "\n"):
                raw_pair, seg_pair, _ = _build_joined_segment(pair, sep)
                dets = self._run_layer1_on_segments((seg_pair,), raw_pair, layer1_detectors)
                for l2_det in mt_layer2:
                    dets.extend(l2_det.detect(seg_pair, raw_pair))
                if any(d.category == category for d in dets):
                    return True
            return False

        if target_pos > 0:
            pair_prev = (turns[target_pos - 1], turns[target_pos])
            if _pair_triggers(pair_prev):
                return (pair_prev[0].message_index, pair_prev[1].message_index)

        if target_pos + 1 < len(turns):
            pair_next = (turns[target_pos], turns[target_pos + 1])
            if _pair_triggers(pair_next):
                return (pair_next[0].message_index, pair_next[1].message_index)

        return tuple(t.message_index for t in turns)

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        """Inspect a single text or multi-turn message list across Pokrov detection layers."""
        started = self._time_fn()
        profile_name, profile = self._resolve_profile(request)
        p_hash = self._get_policy_hash(profile_name, profile)

        run_ner = profile.enable_ner if request.enable_ner is None else request.enable_ner
        run_ml = profile.enable_ml if request.enable_ml is None else request.enable_ml
        layer2_enabled = run_ner or run_ml

        if request.messages:
            items = list(request.messages)
            from_messages = True
        elif request.text:
            items = [
                MessageItem(
                    role="user",
                    content=request.text,
                    control_point=request.control_point,
                    message_index=0,
                )
            ]
            from_messages = False
        else:
            finished = self._time_fn()
            return InspectionResult(
                detections=(),
                evasion_detected=False,
                highest_severity=None,
                layers_executed=(1,),
                early_exited=False,
                cache_hits=0,
                cache_misses=0,
                multi_turn_triggered=False,
                latency_ms=round(max(0.0, (finished - started) * 1000.0), 4),
                control_point=request.control_point,
                compliance_refs=(),
            )

        effective_indices = _resolve_message_indices(items)
        layer1_detectors = self._get_layer1_detectors(profile_name, profile)
        eligible_layer2 = self._select_layer2_detectors(run_ner=run_ner, run_ml=run_ml)

        per_message_detections: list[list[Detection]] = []
        final_detections: list[Detection] = []
        evasion_detected = False
        cache_hits = 0
        cache_misses = 0
        any_layer2_ran = False
        any_early_exit = False
        detected_control_point: ControlPoint | None = None

        for msg, eff_idx in zip(items, effective_indices, strict=True):
            msg_dets, msg_evasion, msg_layers, msg_early_exit, is_hit = (
                self._inspect_single_message(
                    msg=msg,
                    request=request,
                    profile=profile,
                    p_hash=p_hash,
                    layer1_detectors=layer1_detectors,
                    eligible_layer2=eligible_layer2,
                    layer2_enabled=layer2_enabled,
                    run_ner=run_ner,
                    run_ml=run_ml,
                    started=started,
                )
            )
            if msg.content:
                if is_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1

            if msg_evasion:
                evasion_detected = True
            if 2 in msg_layers:
                any_layer2_ran = True
            if msg_early_exit:
                any_early_exit = True

            if from_messages and msg_dets:
                tagged = [
                    det.model_copy(update={"source_message_indices": (eff_idx,)})
                    for det in msg_dets
                ]
                per_message_detections.append(tagged)
                final_detections.extend(tagged)
            else:
                per_message_detections.append(msg_dets)
                final_detections.extend(msg_dets)

            if msg_dets and (
                msg.control_point == ControlPoint.TOOL_RESULT or detected_control_point is None
            ):
                detected_control_point = msg.control_point

        cross_turn_dets: list[Detection] = []
        if from_messages and len(items) >= 2:
            cross_turn_dets, mt_l2_ran = self._run_multi_turn_window(
                messages=items,
                effective_indices=effective_indices,
                per_message_detections=per_message_detections,
                profile=profile,
                layer1_detectors=layer1_detectors,
                eligible_layer2=eligible_layer2,
                run_ner=run_ner,
                request=request,
                p_hash=p_hash,
                now=started,
            )
            if mt_l2_ran:
                any_layer2_ran = True
            if cross_turn_dets:
                final_detections.extend(cross_turn_dets)
                if any(d.evasion_detected for d in cross_turn_dets):
                    evasion_detected = True

        multi_turn_triggered = bool(cross_turn_dets)
        layers_executed: tuple[int, ...] = (1, 2) if any_layer2_ran else (1,)
        early_exited = bool(any_early_exit and not any_layer2_ran)

        if final_detections:
            seen_cats: set[EntityCategory] = set()
            unique_cats: list[EntityCategory] = []
            for d in final_detections:
                if d.category not in seen_cats:
                    seen_cats.add(d.category)
                    unique_cats.append(d.category)
            highest_sev = max_severity([get_category_severity(cat) for cat in unique_cats])
            compliance_refs = aggregate_compliance_refs(unique_cats)
        else:
            highest_sev = None
            compliance_refs = ()

        fields_set = getattr(request, "model_fields_set", None)
        explicit_cp = fields_set is not None and "control_point" in fields_set
        result_cp = (
            request.control_point
            if (explicit_cp or not from_messages or detected_control_point is None)
            else detected_control_point
        )

        finished = self._time_fn()
        latency_ms = round(max(0.0, (finished - started) * 1000.0), 4)

        return InspectionResult.model_construct(
            detections=tuple(final_detections),
            evasion_detected=evasion_detected,
            highest_severity=highest_sev,
            layers_executed=layers_executed,
            early_exited=early_exited,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            multi_turn_triggered=multi_turn_triggered,
            latency_ms=latency_ms,
            control_point=result_cp,
            compliance_refs=compliance_refs,
        )


__all__ = ["PokrovDetectionPipeline"]
