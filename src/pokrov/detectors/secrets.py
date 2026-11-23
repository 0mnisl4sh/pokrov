"""Deterministic and entropy-based secret detector for Pokrov.

Detects API keys, cloud credentials, JWTs, PEM private keys, database
connection strings with embedded credentials, Bearer tokens, and high-entropy
secret assignments in strict linear time while preserving Zero-Retention.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.contracts import Detection, EntityCategory, NormalizedSegment

# ---------------------------------------------------------------------------
# Linear-time provider and format patterns (no nested quantifiers)
# ---------------------------------------------------------------------------

_ANTHROPIC_RE = re.compile(
    r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_OPENAI_SCOPED_RE = re.compile(
    r"(?<![A-Za-z0-9_-])sk-(?:proj|svcacct|admin|None)-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_OPENAI_LEGACY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])sk-(?!ant-|proj-|svcacct-|admin-|None-)[A-Za-z0-9]{16,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_GITHUB_CLASSIC_RE = re.compile(
    r"(?<![A-Za-z0-9_-])gh[pousr]_[A-Za-z0-9]{16,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_GITHUB_PAT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])github_pat_[A-Za-z0-9_]{16,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_GITLAB_RE = re.compile(
    r"(?<![A-Za-z0-9_-])glpat-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
# Allow '/bot<token>' in Telegram Bot API URLs by anchoring only against preceding digits.
_TELEGRAM_RE = re.compile(
    r"(?<!\d)\d{8,11}:[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_AWS_ACCESS_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9_])",
    re.ASCII,
)
_YANDEX_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])AQVN[A-Za-z0-9_-]{30,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_YANDEX_IAM_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])t1\.[A-Za-z0-9_-]{12,}(?:\.[A-Za-z0-9_-]{8,}=*)?(?![A-Za-z0-9_-])",
    re.ASCII,
)
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{3,}\.[A-Za-z0-9_-]{4,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_PEM_BEGIN_RE = re.compile(
    r"-----BEGIN (?:[A-Z0-9]{1,16} ){0,4}PRIVATE KEY(?: BLOCK)?-----",
    re.ASCII,
)
_PEM_END_RE = re.compile(
    r"-----END (?:[A-Z0-9]{1,16} ){0,4}PRIVATE KEY(?: BLOCK)?-----",
    re.ASCII,
)
_DB_CONN_RE = re.compile(
    r"(?<![A-Za-z0-9+.-])"
    r"(?:postgres(?:ql)?|mysql|rediss?|mongodb(?:\+srv)?|amqps?)://"
    r"[^\s:@/\"'<>]{0,64}:[^\s@/\"'<>]{1,128}@[^\s\"'<>]{1,256}",
    re.ASCII,
)
_BEARER_RE = re.compile(
    r"(?<![A-Za-z0-9_-])Bearer[ \t]+(?P<token>[A-Za-z0-9._~+/=-]{16,512})(?![A-Za-z0-9._~+/=-])",
    re.ASCII | re.IGNORECASE,
)
_ASSIGNMENT_RE = re.compile(
    r"""(?<![A-Za-z0-9_.-])"""
    r"""(?:["'](?P<key_q>[A-Za-z_][A-Za-z0-9_.-]{1,48})["']|(?P<key_u>[A-Za-z_][A-Za-z0-9_.-]{1,48}))"""
    r"""[ \t]{0,16}[:=][ \t]{0,16}"""
    r"""(?:"(?P<val_dq>[^"\s\r\n]{8,512})"|'(?P<val_sq>[^'\s\r\n]{8,512})'|(?P<val_uq>[^\s"'`,;)}\]\r\n]{8,512}))""",
    re.ASCII,
)

_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])", re.ASCII)
_PEM_SCAN_LIMIT = 16_384

_SECRET_SINGLE_TOKENS: frozenset[str] = frozenset(
    {
        "apikey",
        "passphrase",
        "passwd",
        "password",
        "pwd",
        "secret",
        "token",
    }
)
_SECRET_PAIR_TOKENS: frozenset[tuple[str, str]] = frozenset(
    {
        ("access", "key"),
        ("api", "key"),
        ("auth", "key"),
        ("client", "secret"),
        ("encryption", "key"),
        ("jwt", "secret"),
        ("master", "key"),
        ("private", "key"),
        ("secret", "key"),
        ("signing", "key"),
        ("webhook", "secret"),
    }
)
_VALUE_SUFFIX_TOKENS: frozenset[str] = frozenset({"val", "value"})
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]", re.ASCII)


@dataclass(frozen=True, slots=True)
class _RuleSpec:
    kind: str
    pattern: re.Pattern[str]
    confidence: float
    priority: int
    marker: str | None = None


_PROVIDER_RULES: tuple[_RuleSpec, ...] = (
    _RuleSpec("anthropic_api_key", _ANTHROPIC_RE, 0.99, 100, "sk-ant-"),
    _RuleSpec("openai_api_key", _OPENAI_SCOPED_RE, 0.98, 95, "sk-"),
    _RuleSpec("openai_api_key", _OPENAI_LEGACY_RE, 0.92, 90, "sk-"),
    _RuleSpec("github_token", _GITHUB_CLASSIC_RE, 0.99, 95, "gh"),
    _RuleSpec("github_token", _GITHUB_PAT_RE, 0.99, 95, "github_pat_"),
    _RuleSpec("gitlab_token", _GITLAB_RE, 0.99, 95, "glpat-"),
    _RuleSpec("telegram_bot_token", _TELEGRAM_RE, 0.97, 95, ":"),
    _RuleSpec("aws_access_key", _AWS_ACCESS_KEY_RE, 0.98, 95, "IA"),
    _RuleSpec("yandex_api_key", _YANDEX_API_KEY_RE, 0.98, 95, "AQVN"),
    _RuleSpec("yandex_iam_token", _YANDEX_IAM_RE, 0.95, 92, "t1."),
    _RuleSpec("jwt", _JWT_RE, 0.96, 90, "eyJ"),
    _RuleSpec("db_connection_string", _DB_CONN_RE, 0.97, 95, "://"),
)


@dataclass(frozen=True, slots=True)
class _CandidateSpan:
    start: int
    end: int
    confidence: float
    priority: int
    kind: str


def shannon_entropy(value: str) -> float:
    """Compute Shannon entropy in bits per character for ``value``."""
    if not value:
        return 0.0
    length = len(value)
    counts = Counter(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _is_valid_yandex_iam(token: str) -> bool:
    """Reject SQL table aliases like ``t1.customer_account_id`` while keeping IAM tokens."""
    body = token[3:]
    if "." in body:
        return len(body) >= 20
    has_digit = any(ch.isdigit() for ch in body)
    has_mixed_case = any(ch.isupper() for ch in body) and any(ch.islower() for ch in body)
    return has_digit or has_mixed_case


@lru_cache(maxsize=512)
def _is_secret_key_name(key: str) -> bool:
    """Return True if an assignment key name denotes a secret/credential."""
    snake = _CAMEL_BOUNDARY_RE.sub(r"\1_\2", key).lower()
    raw_parts = [part for part in re.split(r"[_.-]+", snake) if part]
    while raw_parts and raw_parts[-1].isdigit():
        raw_parts.pop()
    if len(raw_parts) >= 2 and raw_parts[-1] in _VALUE_SUFFIX_TOKENS:
        raw_parts.pop()
    if not raw_parts:
        return False
    if raw_parts[-1] in _SECRET_SINGLE_TOKENS:
        return True
    return bool(len(raw_parts) >= 2 and (raw_parts[-2], raw_parts[-1]) in _SECRET_PAIR_TOKENS)


def _is_non_secret_literal(value: str) -> bool:
    """Filter obvious placeholders, template variables, and local file paths."""
    if value.startswith("<PKR:") and value.endswith(">"):
        return True
    if value.startswith(("${", "{{", "$")):
        return True
    if value.startswith(("/", "./", "../")) or _WINDOWS_DRIVE_RE.match(value):
        return True
    return bool(value.startswith(("http://", "https://")) and "@" not in value)


def _trim_db_url_end(text: str, start: int, end: int) -> int:
    """Strip trailing sentence punctuation from a matched DB connection URI."""
    while end > start and text[end - 1] in ".,;":
        end -= 1
    return end


def _select_non_overlapping(candidates: list[_CandidateSpan]) -> list[_CandidateSpan]:
    """Resolve overlapping spans, keeping higher-priority and higher-confidence findings."""
    ranked = sorted(
        candidates,
        key=lambda item: (-item.priority, -item.confidence, -(item.end - item.start), item.start),
    )
    accepted: list[_CandidateSpan] = []
    for candidate in ranked:
        if any(
            not (candidate.end <= existing.start or candidate.start >= existing.end)
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    accepted.sort(key=lambda item: (item.start, item.end))
    return accepted


class SecretsDetector:
    """Layer-1 detector for API keys, credentials, private keys, and high-entropy secrets."""

    def __init__(
        self,
        hmac_fn: Callable[[str], str] | None = None,
        entropy_threshold: float = 3.8,
    ) -> None:
        self._hmac_fn = hmac_fn
        self._entropy_threshold = entropy_threshold

    @property
    def name(self) -> str:
        """Return detector identifier."""
        return "secrets"

    @property
    def layer(self) -> int:
        """Return cascade layer index (1 = fast deterministic)."""
        return 1

    def detect(self, segment: NormalizedSegment, raw_text: str) -> list[Detection]:
        """Scan a normalized segment for secrets while enforcing Zero-Retention."""
        text = segment.text
        if not text:
            return []

        candidates: list[_CandidateSpan] = []
        self._collect_provider_matches(text, candidates)
        if "-----BEGIN " in text:
            self._collect_pem_matches(text, candidates)
        if "bearer" in text or "Bearer" in text or "BEARER" in text:
            self._collect_bearer_matches(text, candidates)
        if "=" in text or ":" in text:
            self._collect_assignment_matches(text, candidates)

        if not candidates:
            return []

        selected = _select_non_overlapping(candidates)
        compliance_refs = get_compliance_refs(EntityCategory.SECRET_KEY)
        evasion_detected = bool(segment.evasion_techniques)

        detections: list[Detection] = []
        for item in selected:
            raw_secret = text[item.start : item.end]
            mapped_start, mapped_end = segment.map_span(item.start, item.end)
            detections.append(
                Detection(
                    category=EntityCategory.SECRET_KEY,
                    start=mapped_start,
                    end=mapped_end,
                    confidence=item.confidence,
                    detector=self.name,
                    span_source=segment.span_source,
                    evasion_detected=evasion_detected,
                    evasion_techniques=segment.evasion_techniques,
                    compliance_refs=compliance_refs,
                    masked_preview=f"[{EntityCategory.SECRET_KEY.value}:len={len(raw_secret)}]",
                    value_hmac=self._hmac_fn(raw_secret) if self._hmac_fn else "",
                    metadata={"secret_type": item.kind},
                )
            )
        return detections

    def _collect_provider_matches(self, text: str, out: list[_CandidateSpan]) -> None:
        for rule in _PROVIDER_RULES:
            if rule.marker is not None and rule.marker not in text:
                continue
            for match in rule.pattern.finditer(text):
                start, end = match.span()
                if rule.kind == "yandex_iam_token" and not _is_valid_yandex_iam(match.group(0)):
                    continue
                if rule.kind == "db_connection_string":
                    end = _trim_db_url_end(text, start, end)
                    if end <= start:
                        continue
                out.append(
                    _CandidateSpan(
                        start=start,
                        end=end,
                        confidence=rule.confidence,
                        priority=rule.priority,
                        kind=rule.kind,
                    )
                )

    def _collect_pem_matches(self, text: str, out: list[_CandidateSpan]) -> None:
        for match in _PEM_BEGIN_RE.finditer(text):
            start, end = match.span()
            search_limit = min(len(text), end + _PEM_SCAN_LIMIT)
            next_begin = text.find("-----BEGIN ", end, search_limit)
            bound = next_begin if next_begin != -1 else search_limit
            end_match = _PEM_END_RE.search(text, end, bound)
            if end_match is not None:
                end = end_match.end()
            out.append(
                _CandidateSpan(
                    start=start,
                    end=end,
                    confidence=0.99,
                    priority=110,
                    kind="pem_private_key",
                )
            )

    def _collect_bearer_matches(self, text: str, out: list[_CandidateSpan]) -> None:
        for match in _BEARER_RE.finditer(text):
            token = match.group("token")
            if _is_non_secret_literal(token):
                continue
            entropy = shannon_entropy(token)
            if entropy < self._entropy_threshold:
                continue
            start, end = match.span("token")
            confidence = min(
                0.95, round(0.84 + max(0.0, entropy - self._entropy_threshold) * 0.1, 3)
            )
            out.append(
                _CandidateSpan(
                    start=start,
                    end=end,
                    confidence=confidence,
                    priority=70,
                    kind="bearer_token",
                )
            )

    def _collect_assignment_matches(self, text: str, out: list[_CandidateSpan]) -> None:
        # Mathematically, H(s) <= log2(len(set(s))), so fewer unique chars cannot pass threshold.
        min_unique = math.ceil(2**self._entropy_threshold) if self._entropy_threshold > 0 else 1
        for match in _ASSIGNMENT_RE.finditer(text):
            key = match.group("key_q") or match.group("key_u")
            if not key or not _is_secret_key_name(key):
                continue

            if match.group("val_dq") is not None:
                val_group = "val_dq"
            elif match.group("val_sq") is not None:
                val_group = "val_sq"
            else:
                val_group = "val_uq"

            value = match.group(val_group)
            if not value or _is_non_secret_literal(value):
                continue
            if len(set(value)) < min_unique:
                continue

            entropy = shannon_entropy(value)
            if entropy < self._entropy_threshold:
                continue

            start, end = match.span(val_group)
            confidence = min(
                0.94, round(0.80 + max(0.0, entropy - self._entropy_threshold) * 0.12, 3)
            )
            out.append(
                _CandidateSpan(
                    start=start,
                    end=end,
                    confidence=confidence,
                    priority=60,
                    kind="high_entropy_assignment",
                )
            )


__all__ = ["SecretsDetector", "shannon_entropy"]
