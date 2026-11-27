"""Deterministic and entropy-based secret detector for Pokrov.

Detects API keys, cloud credentials, JWTs, PEM private keys, database
connection strings with embedded credentials, Bearer/Basic tokens, password
hashes, and secret assignments in O(N + M log M) time with Zero-Retention.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.contracts import Detection, EntityCategory, NormalizedSegment, SpanSource

_SECRET_COMPLIANCE_REFS = get_compliance_refs(EntityCategory.SECRET_KEY)
_FIELDS_SET_TEMPLATE: set[str] = set(Detection.model_fields)
_NON_SCHEME_COLON_RE = re.compile(r":(?!//)")
_ANGLE_TEMPLATE_RE = re.compile(r"^<[A-Za-z_][A-Za-z _-]{0,31}>$", re.ASCII)

# ---------------------------------------------------------------------------
# Linear-time provider and format patterns (no nested quantifiers)
# ---------------------------------------------------------------------------

_ANTHROPIC_RE = re.compile(
    r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_OPENROUTER_RE = re.compile(
    r"(?<![A-Za-z0-9_-])sk-or-v1-[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_OPENAI_SCOPED_RE = re.compile(
    r"(?<![A-Za-z0-9_-])sk-(?:proj|svcacct|admin|None)-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_OPENAI_LEGACY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])sk-(?!ant-|proj-|svcacct-|admin-|None-|or-|live_|test_)[A-Za-z0-9]{16,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_HUGGINGFACE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_STRIPE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])[spr]k_live_[A-Za-z0-9]{16,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_SLACK_RE = re.compile(
    r"(?<![A-Za-z0-9_-])xox[baprs]-[0-9A-Za-z-]{16,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_GOOGLE_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{32,40}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_SENDGRID_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_NPM_RE = re.compile(
    r"(?<![A-Za-z0-9_-])npm_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_PYPI_RE = re.compile(
    r"(?<![A-Za-z0-9_-])pypi-[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_VAULT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])hvs\.[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_DIGITALOCEAN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])dop_v1_[0-9a-fA-F]{32,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_DATABRICKS_RE = re.compile(
    r"(?<![A-Za-z0-9_-])dapi[0-9a-f]{32}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_GROQ_RE = re.compile(
    r"(?<![A-Za-z0-9_-])gsk_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_PERPLEXITY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])pplx-[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_REPLICATE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])r8_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])",
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
_YANDEX_OAUTH_RE = re.compile(
    r"(?<![A-Za-z0-9_-])y0_[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_VK_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])vk1\.a\.[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_TBANK_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])t\.[A-Za-z0-9_-]{40,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_AGE_SECRET_RE = re.compile(
    r"(?<![A-Za-z0-9_-])AGE-SECRET-KEY-1[0-9A-Z]{50,}(?![A-Za-z0-9_-])",
    re.ASCII,
)
_PASSWORD_HASH_RE = re.compile(
    r"(?<![A-Za-z0-9./$])(?:"
    r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}"
    r"|\$argon2(?:id|i|d)\$v=\d+\$[^\s\"'`<>]{16,160}"
    r"|\$(?:6|5|1)\$(?:rounds=\d+\$)?[./A-Za-z0-9]{1,16}\$[./A-Za-z0-9]{22,86}"
    r")(?![./A-Za-z0-9])",
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
_PEM_INLINE_BODY_RE = re.compile(r"^[A-Za-z0-9+/=]{12,}", re.ASCII)
_PEM_B64_LINE_RE = re.compile(r"^[A-Za-z0-9+/=-]+$", re.ASCII)

_DB_CONN_RE = re.compile(
    r"(?<![A-Za-z0-9+.-])"
    r"(?P<scheme>postgres(?:ql)?(?:\+[a-z0-9_]+)?|mysql(?:\+[a-z0-9_]+)?|mariadb(?:\+[a-z0-9_]+)?"
    r"|rediss?|mongodb(?:\+srv)?|amqps?|clickhouse(?:\+[a-z0-9_]+)?|mssql(?:\+[a-z0-9_]+)?"
    r"|https?|s?ftp)://"
    r"(?P<user>[^\s:@/\"'<>]{0,64}):(?P<pw>[^\s@/\"'<>]{3,128})@[^\s\"'<>]{1,256}",
    re.ASCII | re.IGNORECASE,
)

_BEARER_RE = re.compile(
    r"(?<![A-Za-z0-9_-])Bearer[ \t]+(?P<cred>[A-Za-z0-9._~+/=-]{16,512})(?![A-Za-z0-9._~+/=-])",
    re.ASCII | re.IGNORECASE,
)
_BASIC_AUTH_RE = re.compile(
    r"(?<![A-Za-z0-9_-])Authorization[ \t]*:[ \t]*Basic[ \t]+"
    r"(?P<cred>[A-Za-z0-9+/]{12,}={0,2})(?![A-Za-z0-9+/=])",
    re.ASCII | re.IGNORECASE,
)
_CLI_CRED_RE = re.compile(
    r"(?:(?<![A-Za-z0-9_-])--(?P<flag>password|passwd|api-key|secret|access-token)"
    r"|(?<![A-Za-z0-9_-])-u)[ \t=]+"
    r"(?:\"(?P<dq>[^\"\r\n]{4,256})\"|'(?P<sq>[^'\r\n]{4,256})'|(?P<uq>[^\s\"'\r\n]{4,256}))",
    re.ASCII | re.IGNORECASE,
)

# P0-07: Exclude '://' after ':' and '&' from val_uq so query params inside URLs always match!
_ASSIGNMENT_RE = re.compile(
    r"""(?<![A-Za-zА-Яа-яЁё0-9_.-])"""
    r"""(?:["'](?P<key_q>[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_.-]{1,48})["']|(?P<key_u>[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_.-]{1,48}))"""
    r"""[ \t]{0,16}(?::(?!//)|=)(?:[ \t]*[|>]-?[ \t]*\r?\n[ \t]+|[ \t]{0,16})"""
    r"""(?:"(?P<val_dq>[^"\r\n]{4,512})"|'(?P<val_sq>[^'\r\n]{4,512})'|(?P<val_uq>[^\s"'`,;&)}\]\r\n]{4,512}))""",
)

_RU_PROSE_PASSWORD_RE = re.compile(
    r"(?i)(?<![а-яёa-z0-9_])парол[ьяюем]"
    r"(?:[ \t]+(?:от|для|к|в|на)[ \t]+[а-яёa-z0-9_.-]{2,24})?"
    r"[ \t]+(?P<pw>[^\s\"'`,;:()<>]{6,128})",
)

_CAMEL_BOUNDARY_1_RE = re.compile(r"([A-Z]+)([A-Z][a-z])", re.ASCII)
_CAMEL_BOUNDARY_2_RE = re.compile(r"([a-z0-9])([A-Z])", re.ASCII)
_PEM_SCAN_LIMIT = 16_384

_PKR_PLACEHOLDER_RE = re.compile(r"^<PKR:[A-Z0-9_]+:[0-9a-f]{8,16}>$", re.ASCII)
_SHELL_VAR_RE = re.compile(r"^\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)$", re.ASCII)
_MUSTACHE_VAR_RE = re.compile(r"^\{\{[ \t]*[A-Za-z_][A-Za-z0-9_.-]*[ \t]*\}\}$", re.ASCII)
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]", re.ASCII)

_PASSWORD_KEY_TOKENS: frozenset[str] = frozenset(
    {
        "passphrase",
        "passwd",
        "password",
        "pwd",
        "пароль",
        "пароля",
    }
)

_SECRET_SINGLE_TOKENS: frozenset[str] = frozenset(
    {
        "apikey",
        "auth",
        "cookie",
        "cookies",
        "cred",
        "credential",
        "credentials",
        "creds",
        "key",
        "keys",
        "secret",
        "secrets",
        "token",
        "tokens",
        "ключ",
        "секрет",
        "токен",
    }
)

_CYRILLIC_SECRET_HEADS: frozenset[str] = frozenset({"ключ", "секрет", "токен"})

_NON_SECRET_SUFFIX_TOKENS: frozenset[str] = frozenset(
    {
        "age",
        "alg",
        "algorithm",
        "arn",
        "column",
        "count",
        "desc",
        "description",
        "dir",
        "disabled",
        "enabled",
        "endpoint",
        "env",
        "error",
        "exp",
        "expires",
        "expiry",
        "field",
        "file",
        "format",
        "header",
        "hint",
        "host",
        "id",
        "ids",
        "kind",
        "label",
        "len",
        "length",
        "limit",
        "max",
        "message",
        "min",
        "mode",
        "msg",
        "name",
        "path",
        "pattern",
        "port",
        "prefix",
        "regex",
        "required",
        "status",
        "suffix",
        "timeout",
        "title",
        "ttl",
        "type",
        "uri",
        "url",
        "version",
        "длина",
        "имя",
        "путь",
        "тип",
        "файл",
    }
)

_NON_SECRET_KEY_PREFIXES: frozenset[str] = frozenset(
    {
        "attr",
        "attribute",
        "bucket",
        "cache",
        "candidate",
        "composite",
        "config",
        "dedup",
        "dict",
        "foreign",
        "group",
        "hash",
        "hotkey",
        "i18n",
        "idempotency",
        "index",
        "join",
        "json",
        "kms",
        "label",
        "locale",
        "lookup",
        "map",
        "message",
        "meta",
        "metadata",
        "natural",
        "object",
        "order",
        "partition",
        "primary",
        "property",
        "pub",
        "public",
        "redis",
        "registry",
        "routing",
        "s3",
        "setting",
        "shard",
        "shortcut",
        "sort",
        "ssh_pub",
        "storage",
        "surrogate",
        "tag",
        "translation",
        "unique",
        "verification",
        "verify",
        "внешний",
        "открытый",
        "первичный",
        "публичный",
    }
)

_NON_SECRET_TOKEN_PREFIXES: frozenset[str] = frozenset(
    {
        "bos",
        "cls",
        "completion",
        "eos",
        "estimated",
        "first",
        "input",
        "last",
        "mask",
        "max",
        "min",
        "next",
        "num",
        "output",
        "pad",
        "prev",
        "prompt",
        "sep",
        "special",
        "stop",
        "total",
        "unk",
    }
)

_OBVIOUS_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "change-me",
        "change_me",
        "changeme",
        "dummy",
        "example",
        "false",
        "fixme",
        "masked",
        "none",
        "null",
        "password",
        "passwd",
        "placeholder",
        "redacted",
        "sample",
        "secret",
        "todo",
        "true",
        "undefined",
        "your-api-key",
        "your-password",
        "your-secret",
        "your-token",
        "your_api_key",
        "your_password",
        "your_secret",
        "your_token",
        "yourpassword",
    }
)

_VALUE_SUFFIX_TOKENS: frozenset[str] = frozenset({"val", "value"})


@dataclass(frozen=True, slots=True)
class _RuleSpec:
    kind: str
    pattern: re.Pattern[str]
    confidence: float
    priority: int
    marker: str | None = None


_PROVIDER_RULES: tuple[_RuleSpec, ...] = (
    _RuleSpec("anthropic_api_key", _ANTHROPIC_RE, 0.99, 100, "sk-ant-"),
    _RuleSpec("openrouter_api_key", _OPENROUTER_RE, 0.99, 98, "sk-or-v1-"),
    _RuleSpec("openai_api_key", _OPENAI_SCOPED_RE, 0.98, 95, "sk-"),
    _RuleSpec("openai_api_key", _OPENAI_LEGACY_RE, 0.92, 90, "sk-"),
    _RuleSpec("huggingface_token", _HUGGINGFACE_RE, 0.98, 95, "hf_"),
    _RuleSpec("stripe_api_key", _STRIPE_RE, 0.99, 96, "k_live_"),
    _RuleSpec("slack_token", _SLACK_RE, 0.99, 96, "xox"),
    _RuleSpec("google_api_key", _GOOGLE_API_KEY_RE, 0.98, 95, "AIza"),
    _RuleSpec("sendgrid_api_key", _SENDGRID_RE, 0.99, 96, "SG."),
    _RuleSpec("npm_token", _NPM_RE, 0.98, 95, "npm_"),
    _RuleSpec("pypi_token", _PYPI_RE, 0.99, 96, "pypi-"),
    _RuleSpec("vault_token", _VAULT_TOKEN_RE, 0.98, 95, "hvs."),
    _RuleSpec("digitalocean_token", _DIGITALOCEAN_RE, 0.99, 96, "dop_v1_"),
    _RuleSpec("databricks_token", _DATABRICKS_RE, 0.97, 94, "dapi"),
    _RuleSpec("groq_api_key", _GROQ_RE, 0.98, 95, "gsk_"),
    _RuleSpec("perplexity_api_key", _PERPLEXITY_RE, 0.98, 95, "pplx-"),
    _RuleSpec("replicate_token", _REPLICATE_RE, 0.97, 94, "r8_"),
    _RuleSpec("github_token", _GITHUB_CLASSIC_RE, 0.99, 95, "gh"),
    _RuleSpec("github_token", _GITHUB_PAT_RE, 0.99, 95, "github_pat_"),
    _RuleSpec("gitlab_token", _GITLAB_RE, 0.99, 95, "glpat-"),
    _RuleSpec("telegram_bot_token", _TELEGRAM_RE, 0.97, 95, ":"),
    _RuleSpec("aws_access_key", _AWS_ACCESS_KEY_RE, 0.98, 95, "IA"),
    _RuleSpec("yandex_api_key", _YANDEX_API_KEY_RE, 0.98, 95, "AQVN"),
    _RuleSpec("yandex_iam_token", _YANDEX_IAM_RE, 0.95, 92, "t1."),
    _RuleSpec("yandex_oauth_token", _YANDEX_OAUTH_RE, 0.98, 95, "y0_"),
    _RuleSpec("vk_access_token", _VK_TOKEN_RE, 0.98, 95, "vk1.a."),
    _RuleSpec("tbank_token", _TBANK_TOKEN_RE, 0.95, 92, "t."),
    _RuleSpec("age_secret_key", _AGE_SECRET_RE, 0.99, 98, "AGE-SECRET-KEY-"),
    _RuleSpec("password_hash", _PASSWORD_HASH_RE, 0.97, 94, "$"),
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


@lru_cache(maxsize=2048)
def _shannon_entropy_cached(value: str) -> float:
    length = len(value)
    counts = Counter(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def shannon_entropy(value: str) -> float:
    """Compute Shannon entropy in bits per character for ``value``."""
    if not value:
        return 0.0
    if len(value) <= 128:
        return _shannon_entropy_cached(value)
    length = len(value)
    counts = Counter(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _is_valid_yandex_iam(raw_val: str) -> bool:
    """Reject SQL table aliases like ``t1.customer_account_id`` while keeping IAM tokens."""
    body = raw_val[3:]
    if "." in body:
        return len(body) >= 20
    has_digit = any(ch.isdigit() for ch in body)
    has_mixed_case = any(ch.isupper() for ch in body) and any(ch.islower() for ch in body)
    return has_digit or has_mixed_case


def _is_valid_tbank_token(raw_val: str) -> bool:
    """Reject SQL aliases ``t.column`` while keeping T-Bank tokens ``t.<40+ mixed chars>``."""
    body = raw_val[2:]
    has_digit = any(ch.isdigit() for ch in body)
    has_mixed_case = any(ch.isupper() for ch in body) and any(ch.islower() for ch in body)
    return has_digit and has_mixed_case


def _strip_numeric_key_suffix(key: str) -> str:
    """Strip trailing delimiter-separated digits (e.g. ``API_SECRET_123`` -> ``API_SECRET``)."""
    if not key or not ("0" <= key[-1] <= "9"):
        return key
    idx = len(key)
    while idx > 0 and key[idx - 1].isdigit():
        num_start = idx
        while num_start > 0 and key[num_start - 1].isdigit():
            num_start -= 1
        if 0 < num_start < idx and key[num_start - 1] in "_.-":
            sep_start = num_start
            while sep_start > 0 and key[sep_start - 1] in "_.-":
                sep_start -= 1
            if sep_start == 0:
                break
            idx = sep_start
        else:
            break
    return key[:idx] if idx != len(key) else key


@lru_cache(maxsize=1024)
def _classify_secret_key_name_cached(key: str) -> Literal["none", "password", "secret"]:
    """Classify an assignment key name as ``password``, ``secret``, or ``none``."""
    step1 = _CAMEL_BOUNDARY_1_RE.sub(r"\1_\2", key)
    snake = _CAMEL_BOUNDARY_2_RE.sub(r"\1_\2", step1).lower()
    raw_parts = [part for part in re.split(r"[_.-]+", snake) if part]
    while raw_parts and raw_parts[-1].isdigit():
        raw_parts.pop()
    if len(raw_parts) >= 2 and raw_parts[-1] in _VALUE_SUFFIX_TOKENS:
        raw_parts.pop()
    if not raw_parts:
        return "none"

    last = raw_parts[-1]
    first = raw_parts[0]
    if last in _NON_SECRET_SUFFIX_TOKENS:
        return "none"

    if last in _PASSWORD_KEY_TOKENS or first in _PASSWORD_KEY_TOKENS:
        return "password"

    if last in {"key", "keys", "ключ"}:
        if len(raw_parts) >= 2 and raw_parts[-2] in _NON_SECRET_KEY_PREFIXES:
            return "none"
        return "secret"

    if last in {"token", "tokens", "токен"}:
        if len(raw_parts) >= 2 and raw_parts[-2] in _NON_SECRET_TOKEN_PREFIXES:
            return "none"
        return "secret"

    if last in _SECRET_SINGLE_TOKENS or first in _CYRILLIC_SECRET_HEADS:
        return "secret"

    return "none"


def _classify_secret_key_name(key: str) -> Literal["none", "password", "secret"]:
    """Classify an assignment key name after collapsing numeric suffixes for cache reuse."""
    return _classify_secret_key_name_cached(_strip_numeric_key_suffix(key))


def _is_secret_key_name(key: str) -> bool:
    """Return True if an assignment key name denotes a secret or password."""
    return _classify_secret_key_name(key) != "none"


def _is_non_secret_literal(value: str) -> bool:
    """Filter Pokrov placeholders, strict shell/mustache variables, and local file paths."""
    if _PKR_PLACEHOLDER_RE.match(value):
        return True
    if _SHELL_VAR_RE.match(value) or _MUSTACHE_VAR_RE.match(value):
        return True
    if value.startswith(("/", "./", "../")) or _WINDOWS_DRIVE_RE.match(value):
        return True
    return bool(value.startswith(("http://", "https://")) and "@" not in value)


def _is_obvious_placeholder(value: str) -> bool:
    """Reject placeholder strings like ``changeme``, ``example``, ``***``, or ``xxxx``."""
    stripped = value.strip()
    if not stripped:
        return True
    low = stripped.lower()
    if low in _OBVIOUS_PLACEHOLDERS:
        return True
    if _ANGLE_TEMPLATE_RE.match(stripped):
        return True
    return set(low) <= {"*", "x", ".", "-", "_"}


def _trim_db_url_end(text: str, start: int, end: int) -> int:
    """Strip trailing sentence punctuation from a matched DB/HTTP connection URI."""
    while end > start and text[end - 1] in ".,;":
        end -= 1
    return end


def _extend_unclosed_pem_end(text: str, header_end: int, bound: int) -> int:
    """Extend an unclosed PEM header across contiguous Base64 key lines up to ``bound``."""
    inline_m = _PEM_INLINE_BODY_RE.match(text, header_end, bound)
    if inline_m is not None:
        return inline_m.end()

    pos = header_end
    last_key_end = header_end
    saw_key_line = False

    while pos < bound:
        # Advance across single newline (\r\n or \n)
        if text[pos] == "\r":
            pos += 1
        if pos < bound and text[pos] == "\n":
            pos += 1
        else:
            break

        nl = text.find("\n", pos, bound)
        line_end = nl if nl != -1 else bound
        raw_line = text[pos:line_end].rstrip("\r")
        stripped = raw_line.strip(" \t")

        if not stripped:
            if not saw_key_line:
                pos = line_end
                continue
            break

        # Optional RFC 1421 encrypted PEM metadata headers right after BEGIN
        if (
            not saw_key_line
            and ":" in stripped
            and stripped.startswith(("Proc-Type:", "DEK-Info:"))
        ):
            pos = line_end
            continue

        if stripped.startswith("-----"):
            break

        if _PEM_B64_LINE_RE.match(stripped) and (
            len(stripped) >= 12 or (saw_key_line and len(stripped) >= 4 and stripped.endswith("="))
        ):
            saw_key_line = True
            # Trim trailing horizontal whitespace on this line
            r_trimmed = len(raw_line.rstrip(" \t"))
            last_key_end = pos + r_trimmed
            pos = line_end
            continue

        break

    return last_key_end


def _select_non_overlapping(candidates: list[_CandidateSpan]) -> list[_CandidateSpan]:
    """Resolve overlapping spans in O(M log M), keeping higher-priority findings."""
    if len(candidates) <= 1:
        return list(candidates)

    prev_end = -1
    already_disjoint = True
    for candidate in candidates:
        if candidate.start < prev_end:
            already_disjoint = False
            break
        prev_end = candidate.end
    if already_disjoint:
        return list(candidates)

    ranked = sorted(
        candidates,
        key=lambda item: (-item.priority, -item.confidence, -(item.end - item.start), item.start),
    )
    starts: list[int] = []
    ends: list[int] = []
    accepted: list[_CandidateSpan] = []

    for candidate in ranked:
        c_start = candidate.start
        c_end = candidate.end
        idx = bisect_right(starts, c_start)
        if idx > 0 and ends[idx - 1] > c_start:
            continue
        if idx < len(starts) and starts[idx] < c_end:
            continue
        starts.insert(idx, c_start)
        ends.insert(idx, c_end)
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
        _ = raw_text
        text = segment.text
        if not text:
            return []

        candidates: list[_CandidateSpan] = []
        self._collect_provider_matches(text, candidates)
        if "-----BEGIN " in text:
            self._collect_pem_matches(text, candidates)
        if "bearer" in text or "Bearer" in text or "BEARER" in text:
            self._collect_bearer_matches(text, candidates)
        if "Basic" in text or "basic" in text or "BASIC" in text:
            self._collect_basic_auth_matches(text, candidates)
        if "--" in text or "-u " in text:
            self._collect_cli_matches(text, candidates)
        if "=" in text or (":" in text and _NON_SCHEME_COLON_RE.search(text) is not None):
            self._collect_assignment_matches(text, candidates)
        if "парол" in text or "Парол" in text or "ПАРОЛ" in text:
            self._collect_ru_prose_passwords(text, candidates)

        if not candidates:
            return []

        selected = _select_non_overlapping(candidates)
        evasion_detected = bool(segment.evasion_techniques)
        evasion_techs = segment.evasion_techniques
        span_source = segment.span_source
        direct_span = not segment.index_map and span_source == SpanSource.DIRECT
        det_new = Detection.__new__
        obj_setattr = object.__setattr__
        fields_copy = _FIELDS_SET_TEMPLATE.copy
        hmac_fn = self._hmac_fn
        detector_name = self.name
        cat_secret = EntityCategory.SECRET_KEY
        secret_refs = _SECRET_COMPLIANCE_REFS

        detections: list[Detection] = []
        append_det = detections.append
        for item in selected:
            raw_val = text[item.start : item.end]
            if direct_span:
                mapped_start, mapped_end = item.start, item.end
            else:
                mapped_start, mapped_end = segment.map_span(item.start, item.end)
            det = det_new(Detection)
            obj_setattr(
                det,
                "__dict__",
                {
                    "category": cat_secret,
                    "start": mapped_start,
                    "end": mapped_end,
                    "confidence": item.confidence,
                    "detector": detector_name,
                    "span_source": span_source,
                    "evasion_detected": evasion_detected,
                    "evasion_techniques": evasion_techs,
                    "compliance_refs": secret_refs,
                    "masked_preview": f"[secret_key:len={len(raw_val)}]",
                    "value_hmac": hmac_fn(raw_val) if hmac_fn else "",
                    "source_message_indices": (),
                    "metadata": {"secret_type": item.kind},
                },
            )
            obj_setattr(det, "__pydantic_fields_set__", fields_copy())
            obj_setattr(det, "__pydantic_extra__", None)
            obj_setattr(det, "__pydantic_private__", None)
            append_det(det)
        return detections

    def _collect_provider_matches(self, text: str, out: list[_CandidateSpan]) -> None:
        for rule in _PROVIDER_RULES:
            if rule.marker is not None and rule.marker not in text:
                continue
            for match in rule.pattern.finditer(text):
                start, end = match.span()
                matched = match.group(0)
                if rule.kind == "yandex_iam_token" and not _is_valid_yandex_iam(matched):
                    continue
                if rule.kind == "tbank_token" and not _is_valid_tbank_token(matched):
                    continue
                if rule.kind == "db_connection_string":
                    scheme = match.group("scheme").lower()
                    user = match.group("user")
                    pw = match.group("pw")
                    if _is_non_secret_literal(pw) or _is_obvious_placeholder(pw):
                        continue
                    if scheme in {"http", "https"} and (not user or user.isdigit()):
                        continue
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
            else:
                end = _extend_unclosed_pem_end(text, end, bound)
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
            cred = match.group("cred")
            if _is_non_secret_literal(cred) or _is_obvious_placeholder(cred):
                continue
            entropy = shannon_entropy(cred)
            if entropy < self._entropy_threshold:
                continue
            start, end = match.span("cred")
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

    def _collect_basic_auth_matches(self, text: str, out: list[_CandidateSpan]) -> None:
        for match in _BASIC_AUTH_RE.finditer(text):
            cred = match.group("cred")
            if _is_non_secret_literal(cred) or _is_obvious_placeholder(cred):
                continue
            start, end = match.span("cred")
            out.append(
                _CandidateSpan(
                    start=start,
                    end=end,
                    confidence=0.96,
                    priority=85,
                    kind="basic_auth",
                )
            )

    def _collect_cli_matches(self, text: str, out: list[_CandidateSpan]) -> None:
        for match in _CLI_CRED_RE.finditer(text):
            val_group = (
                "dq"
                if match.group("dq") is not None
                else ("sq" if match.group("sq") is not None else "uq")
            )
            val = match.group(val_group)
            if not val or _is_non_secret_literal(val) or _is_obvious_placeholder(val):
                continue
            is_u_flag = match.group("flag") is None
            if is_u_flag:
                if ":" not in val or len(val) < 8:
                    continue
                _, _, pw = val.partition(":")
                if len(pw) < 4 or _is_obvious_placeholder(pw):
                    continue
            elif shannon_entropy(val) < 1.5:
                continue
            start, end = match.span(val_group)
            out.append(
                _CandidateSpan(
                    start=start,
                    end=end,
                    confidence=0.90,
                    priority=65,
                    kind="cli_credential",
                )
            )

    def _collect_assignment_matches(self, text: str, out: list[_CandidateSpan]) -> None:
        base_min_unique = (
            math.ceil(2**self._entropy_threshold) if self._entropy_threshold > 0 else 1
        )
        key_cache: dict[str, Literal["none", "password", "secret"]] = {}
        val_conf_cache: dict[tuple[str, str], float | None] = {}
        entropy_thr = self._entropy_threshold
        out_append = out.append
        for match in _ASSIGNMENT_RE.finditer(text):
            key = match.group("key_q") or match.group("key_u")
            if not key:
                continue
            base_key = _strip_numeric_key_suffix(key)
            key_kind = key_cache.get(base_key)
            if key_kind is None:
                key_kind = _classify_secret_key_name_cached(base_key)
                if len(key_cache) < 2048:
                    key_cache[base_key] = key_kind
            if key_kind == "none":
                continue

            if match.group("val_dq") is not None:
                val_group = "val_dq"
            elif match.group("val_sq") is not None:
                val_group = "val_sq"
            else:
                val_group = "val_uq"

            value = match.group(val_group)
            if not value:
                continue
            cache_key = (key_kind, value)
            if cache_key in val_conf_cache:
                confidence = val_conf_cache[cache_key]
                if confidence is None:
                    continue
            else:
                if _is_non_secret_literal(value) or _is_obvious_placeholder(value):
                    if len(value) <= 128 and len(val_conf_cache) < 2048:
                        val_conf_cache[cache_key] = None
                    continue
                if key_kind == "password":
                    if len(value) < 4 or len(set(value)) < 3:
                        if len(value) <= 128 and len(val_conf_cache) < 2048:
                            val_conf_cache[cache_key] = None
                        continue
                    entropy = shannon_entropy(value)
                    if entropy < 1.5:
                        if len(value) <= 128 and len(val_conf_cache) < 2048:
                            val_conf_cache[cache_key] = None
                        continue
                    confidence = min(0.94, round(0.84 + max(0.0, entropy - 2.5) * 0.06, 3))
                else:
                    if len(value) < 8 or " " in value:
                        if len(value) <= 128 and len(val_conf_cache) < 2048:
                            val_conf_cache[cache_key] = None
                        continue
                    eff_threshold = min(entropy_thr, 3.3) if len(value) >= 28 else entropy_thr
                    req_unique = min(base_min_unique, 10) if len(value) >= 28 else base_min_unique
                    if len(set(value)) < req_unique:
                        if len(value) <= 128 and len(val_conf_cache) < 2048:
                            val_conf_cache[cache_key] = None
                        continue
                    entropy = shannon_entropy(value)
                    if entropy < eff_threshold:
                        if len(value) <= 128 and len(val_conf_cache) < 2048:
                            val_conf_cache[cache_key] = None
                        continue
                    confidence = min(
                        0.94, round(0.80 + max(0.0, entropy - eff_threshold) * 0.12, 3)
                    )
                if len(value) <= 128 and len(val_conf_cache) < 2048:
                    val_conf_cache[cache_key] = confidence

            start, end = match.span(val_group)
            out_append(
                _CandidateSpan(
                    start=start,
                    end=end,
                    confidence=confidence,
                    priority=60,
                    kind="high_entropy_assignment",
                )
            )

    def _collect_ru_prose_passwords(self, text: str, out: list[_CandidateSpan]) -> None:
        for match in _RU_PROSE_PASSWORD_RE.finditer(text):
            pw = match.group("pw").rstrip(".,;!?")
            if len(pw) < 6 or not pw.isascii():
                continue
            if _is_non_secret_literal(pw) or _is_obvious_placeholder(pw):
                continue
            has_alpha = any(ch.isalpha() for ch in pw)
            has_digit = any(ch.isdigit() for ch in pw)
            has_sym = any(not ch.isalnum() for ch in pw)
            entropy = shannon_entropy(pw)
            if not ((has_alpha and (has_digit or has_sym)) or entropy >= 3.6):
                continue
            if entropy < 2.2:
                continue
            start = match.start("pw")
            end = start + len(pw)
            out.append(
                _CandidateSpan(
                    start=start,
                    end=end,
                    confidence=0.90,
                    priority=62,
                    kind="password_prose_ru",
                )
            )


__all__ = ["SecretsDetector", "shannon_entropy"]
