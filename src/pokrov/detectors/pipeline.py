"""Multi-layer detection pipeline with SHA-256 cache, early-exit, and multi-turn window."""

from __future__ import annotations

import hashlib
import threading
import time
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

# Categories that immediately short-circuit Layer 2 (NER/ML) when matched in Layer 1.
_HARD_EARLY_EXIT_CATEGORIES: frozenset[EntityCategory] = frozenset(
    {
        EntityCategory.SECRET_KEY,
        EntityCategory.PROMPT_INJECTION,
        EntityCategory.INDIRECT_INJECTION,
        EntityCategory.MARKDOWN_EXFIL,
    }
)

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

_CacheKey = tuple[str, str, str, bool, bool, str, str, str]
_MultiTurnCacheKey = tuple[str, tuple[tuple[int, str], ...]]


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
    expires_at: float


@dataclass(frozen=True, slots=True)
class _WindowTurn:
    """Normalized turn metadata for in-memory multi-turn sliding window."""

    position: int
    message_index: int
    raw_text: str
    primary: NormalizedSegment


@dataclass(frozen=True, slots=True)
class _DefaultLayer1Bundle:
    """Shared default Layer 1 detector instances."""

    secrets: SecretsDetector
    prompt_injection: PromptInjectionDetector
    ru_pii_empty: RuPiiDetector


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detection_rank_key(det: Detection) -> tuple[float, int, int, int, str]:
    sev = SEVERITY_ORDER[get_category_severity(det.category)]
    return (-det.confidence, -(det.end - det.start), -sev, det.start, det.category.value)


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    if a_start == a_end or b_start == b_end:
        return a_start == b_start and a_end == b_end
    return a_start < b_end and b_start < a_end


def _same_conflict_domain(a: EntityCategory, b: EntityCategory) -> bool:
    return (a in _ATTACK_WRAPPER_CATEGORIES) == (b in _ATTACK_WRAPPER_CATEGORIES)


def _detections_conflict(candidate: Detection, existing: Detection) -> bool:
    """Return True if two detections represent conflicting claims on an overlapping span."""
    if not _spans_overlap(candidate.start, candidate.end, existing.start, existing.end):
        return False

    if (
        candidate.span_source == SpanSource.DECODED_BLOB
        and existing.span_source == SpanSource.DECODED_BLOB
    ):
        if candidate.category == existing.category:
            return True
        # Multiple distinct entities inside one decoded blob share (outer_start, outer_end);
        # only resolve across categories when both matched the exact same underlying value.
        return bool(
            candidate.value_hmac
            and existing.value_hmac
            and candidate.value_hmac == existing.value_hmac
        )

    if candidate.category == existing.category:
        return True

    if (
        candidate.span_source == SpanSource.DECODED_BLOB
        or existing.span_source == SpanSource.DECODED_BLOB
    ):
        return False

    # Both are DIRECT (or both CROSS_TURN):
    # - Whole-message ONNX classifier shouldn't suppress or be suppressed by localized PII.
    if (
        candidate.detector == "prompt_injection_onnx"
        or existing.detector == "prompt_injection_onnx"
    ) and (candidate.start, candidate.end) != (existing.start, existing.end):
        return False

    if _same_conflict_domain(candidate.category, existing.category):
        return True

    # Across entity vs attack-wrapper domains, resolve only exact span collisions.
    return (candidate.start, candidate.end) == (existing.start, existing.end)


def _deduplicate_detections(detections: Sequence[Detection]) -> list[Detection]:
    """Resolve overlapping detections preferring higher confidence, longer span, then severity."""
    if len(detections) <= 1:
        return list(detections)

    ranked = sorted(detections, key=_detection_rank_key)
    accepted: list[Detection] = []
    for candidate in ranked:
        if any(_detections_conflict(candidate, existing) for existing in accepted):
            continue
        accepted.append(candidate)

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
    for idx, turn in enumerate(turns):
        if idx > 0:
            raw_parts.append(separator)
            norm_parts.append(separator)
            index_map.append(raw_cursor)
            raw_cursor += len(separator)

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
        override_rules = profile.principal_overrides.get(request.principal, {})
        for det in detections:
            if det.category in _HARD_EARLY_EXIT_CATEGORIES:
                return True
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
            if effective_action in (Action.BLOCK, Action.ROUTE_LOCAL):
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
                detections.extend(detector.detect(seg, raw_text))
                if isinstance(detector, RuPiiDetector) and seg.text:
                    collapsed = collapse_split_digits(seg)
                    if collapsed.text != seg.text:
                        detections.extend(detector.detect(collapsed, raw_text))
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
        evasion_detected = norm_res.evasion_detected or any(
            d.evasion_detected for d in l1_detections
        )

        all_detections: list[Detection] = list(l1_detections)
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
                ran_any_l2 = False
                for seg in segments:
                    seg_elapsed = max(0.0, (self._time_fn() - started) * 1000.0)
                    if ran_any_l2 and _is_budget_exhausted(seg_elapsed, request.time_budget_ms):
                        budget_aborted = True
                        break
                    for detector in eligible_layer2:
                        all_detections.extend(detector.detect(seg, msg.content))
                        ran_any_l2 = True
                if ran_any_l2:
                    layers_executed = (1, 2)

        deduped = _deduplicate_detections(all_detections)
        if any(d.evasion_detected for d in deduped):
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

    def _run_multi_turn_window(
        self,
        *,
        messages: Sequence[MessageItem],
        effective_indices: Sequence[int],
        per_message_detections: Sequence[Sequence[Detection]],
        profile: ProfileConfig,
        layer1_detectors: Sequence[BaseDetector],
        p_hash: str,
        now: float,
    ) -> list[Detection]:
        """Run in-memory sliding window over the last ``multi_turn_window`` user/tool turns."""
        window_size = profile.multi_turn_window
        if window_size < 2 or len(messages) < 2:
            return []

        user_tool_entries: list[tuple[int, int, MessageItem]] = [
            (pos, eff_idx, msg)
            for pos, (eff_idx, msg) in enumerate(zip(effective_indices, messages, strict=True))
            if msg.role.lower() in {"user", "tool"}
        ]
        recent_entries = [entry for entry in user_tool_entries[-window_size:] if entry[2].content]
        if len(recent_entries) < 2:
            return []

        ttl = profile.cache_ttl_seconds
        use_cache = self._max_cache_entries > 0 and ttl > 0
        mt_key: _MultiTurnCacheKey | None = None
        if use_cache:
            mt_key = (
                p_hash,
                tuple(
                    (eff_idx, _sha256_text(msg.content)) for _pos, eff_idx, msg in recent_entries
                ),
            )
            cached_mt = self._mt_cache_get(mt_key, now)
            if cached_mt is not None:
                return list(cached_mt.detections)

        turns = [
            _WindowTurn(
                position=pos,
                message_index=eff_idx,
                raw_text=msg.content,
                primary=self._normalizer.normalize(msg.content).primary,
            )
            for pos, eff_idx, msg in recent_entries
        ]

        raw_joined_space, seg_space, turn_ranges = _build_joined_segment(turns, " ")
        raw_joined_nl, seg_nl, _ = _build_joined_segment(turns, "\n")

        existing_in_window: list[tuple[EntityCategory, int, int]] = []
        for turn, (_, r_start, _) in zip(turns, turn_ranges, strict=True):
            for det in per_message_detections[turn.position]:
                if det.span_source == SpanSource.DIRECT:
                    existing_in_window.append(
                        (det.category, r_start + det.start, r_start + det.end)
                    )

        raw_candidates: list[Detection] = []
        for raw_joined, seg in ((raw_joined_space, seg_space), (raw_joined_nl, seg_nl)):
            raw_candidates.extend(
                self._run_layer1_on_segments((seg,), raw_joined, layer1_detectors)
            )

        cross_turn_candidates: list[Detection] = []
        joined_len = len(raw_joined_space)

        for cand in raw_candidates:
            c_start = max(0, min(cand.start, joined_len))
            c_end = max(c_start, min(cand.end, joined_len))
            if c_end <= c_start:
                continue

            touched_turns = [
                (idx, msg_idx)
                for idx, (msg_idx, r_start, r_end) in enumerate(turn_ranges)
                if c_start < r_end and c_end > r_start
            ]
            if not touched_turns:
                continue

            covered_single_turn = any(
                ex_cat == cand.category and _spans_overlap(c_start, c_end, ex_start, ex_end)
                for ex_cat, ex_start, ex_end in existing_in_window
            )
            if len(touched_turns) == 1 and covered_single_turn:
                continue
            # Also skip if it touches one turn and simply sits strictly inside one turn's range
            # while overlapping an existing same-category detection inside that turn.
            if len(touched_turns) >= 2:
                # Check if the candidate just spans two turns accidentally? No, spanning >=2 turns
                # means it crosses a turn boundary; only verify it isn't strictly identical to a
                # single-turn span (which len >= 2 already rules out).
                source_indices = tuple(msg_idx for _, msg_idx in touched_turns)
            else:
                # Newly uncovered detection inside one turn enabled by context in adjacent turns.
                turn_pos = touched_turns[0][0]
                source_indices = self._attribute_context_turns(
                    turns=turns,
                    target_pos=turn_pos,
                    category=cand.category,
                    layer1_detectors=layer1_detectors,
                )

            raw_slice = raw_joined_space[c_start:c_end]
            updated = cand.model_copy(
                update={
                    "start": c_start,
                    "end": c_end,
                    "span_source": SpanSource.CROSS_TURN,
                    "source_message_indices": source_indices,
                    "masked_preview": f"[{cand.category.value}:len={c_end - c_start}]",
                    "value_hmac": self._hmac_fn(raw_slice) if self._hmac_fn else "",
                }
            )
            cross_turn_candidates.append(updated)

        if not cross_turn_candidates:
            if use_cache and mt_key is not None:
                self._mt_cache_put(
                    mt_key,
                    _MultiTurnCacheEntry(detections=(), expires_at=self._time_fn() + ttl),
                )
            return []

        deduped_mt = _deduplicate_detections(cross_turn_candidates)
        if use_cache and mt_key is not None:
            self._mt_cache_put(
                mt_key,
                _MultiTurnCacheEntry(
                    detections=tuple(deduped_mt),
                    expires_at=self._time_fn() + ttl,
                ),
            )
        return deduped_mt

    def _attribute_context_turns(
        self,
        *,
        turns: Sequence[_WindowTurn],
        target_pos: int,
        category: EntityCategory,
        layer1_detectors: Sequence[BaseDetector],
    ) -> tuple[int, ...]:
        """Identify which adjacent turn(s) provided context for a newly uncovered detection."""
        if target_pos > 0:
            pair = (turns[target_pos - 1], turns[target_pos])
            raw_pair, seg_pair, _ = _build_joined_segment(pair, " ")
            if any(
                d.category == category
                for d in self._run_layer1_on_segments((seg_pair,), raw_pair, layer1_detectors)
            ):
                return (pair[0].message_index, pair[1].message_index)

        if target_pos + 1 < len(turns):
            pair = (turns[target_pos], turns[target_pos + 1])
            raw_pair, seg_pair, _ = _build_joined_segment(pair, " ")
            if any(
                d.category == category
                for d in self._run_layer1_on_segments((seg_pair,), raw_pair, layer1_detectors)
            ):
                return (pair[0].message_index, pair[1].message_index)

        return tuple(t.message_index for t in turns)

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        """Inspect a single text or multi-turn message list across Pokrov detection layers."""
        started = self._time_fn()
        profile_name, profile = self._resolve_profile(request)
        p_hash = self._get_policy_hash(profile_name, profile)

        run_ner = bool(request.enable_ner or profile.enable_ner)
        run_ml = bool(request.enable_ml or profile.enable_ml)
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
            cross_turn_dets = self._run_multi_turn_window(
                messages=items,
                effective_indices=effective_indices,
                per_message_detections=per_message_detections,
                profile=profile,
                layer1_detectors=layer1_detectors,
                p_hash=p_hash,
                now=started,
            )
            if cross_turn_dets:
                final_detections.extend(cross_turn_dets)
                if any(d.evasion_detected for d in cross_turn_dets):
                    evasion_detected = True

        multi_turn_triggered = bool(cross_turn_dets)
        layers_executed: tuple[int, ...] = (1, 2) if any_layer2_ran else (1,)
        early_exited = bool(any_early_exit and not any_layer2_ran)

        highest_sev = (
            max_severity([get_category_severity(d.category) for d in final_detections])
            if final_detections
            else None
        )
        compliance_refs = aggregate_compliance_refs(d.category for d in final_detections)

        fields_set = getattr(request, "model_fields_set", None)
        explicit_cp = fields_set is not None and "control_point" in fields_set
        result_cp = (
            request.control_point
            if (explicit_cp or not from_messages or detected_control_point is None)
            else detected_control_point
        )

        finished = self._time_fn()
        latency_ms = round(max(0.0, (finished - started) * 1000.0), 4)

        return InspectionResult(
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
