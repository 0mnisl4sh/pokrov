"""Prompt injection, indirect injection, EchoLeak Markdown exfil, and ONNX hook detectors.

Layer-1 deterministic rules cover Russian and English direct prompt injections,
delimiter attacks, hidden HTML/Markdown instruction comments, tool hijacking,
and EchoLeak (CVE-2025-32711) Markdown/HTML image exfiltration in linear time.
Layer-2 ``OnnxPromptInjectionDetector`` provides an optional CPU ONNX hook
(or custom ``infer_fn``) that degrades gracefully when ``[ml]`` is absent.
"""

from __future__ import annotations

import base64
import importlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.contracts import Detection, EntityCategory, NormalizedSegment

# ---------------------------------------------------------------------------
# Linear-time direct prompt injection & delimiter patterns (RU + EN)
# ---------------------------------------------------------------------------

_HAS_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")

_EN_OVERRIDE_VERBS = ("ignore", "disregard", "forget", "override")
_EN_OVERRIDE_NOUNS = (
    "instruction",
    "rule",
    "prompt",
    "directive",
    "constraint",
    "guideline",
    "policy",
    "filter",
    "guardrail",
)
_EN_EXTRACT_VERBS = (
    "repeat",
    "reveal",
    "show",
    "print",
    "output",
    "leak",
    "display",
    "dump",
    "share",
    "expose",
    "tell",
)
_EN_JAILBREAK_HINTS = (
    "jailbreak",
    "mode",
    "without",
    "unrestricted",
    "unfiltered",
    "uncensored",
)

_EN_DIRECT_RE = re.compile(
    r"(?P<en_instruction_override>(?i:\b(?:ignore|disregard|forget|override)\b\s{1,8}"
    r"(?:(?:all|any|the|your|previous|prior|above|preceding|earlier|old|safety|system)\s{1,8}){0,4}"
    r"(?:instructions?|rules?|prompts?|directives?|constraints?|guidelines?|policy|filters?|guardrails?|system\s{1,4}prompt)\b))"
    r"|(?P<en_forget_above>(?i:\bforget\s{1,4}(?:everything|all)\s{1,4}(?:above|before|prior)\b))"
    r"|(?P<en_new_system>(?i:\b(?:new\s{1,4}system\s{1,4}instructions?|"
    r"from\s{1,4}now\s{1,4}on,?\s{1,4}(?:you\s{1,4}(?:are|will|must)\s{1,4})?"
    r"(?:ignore|disregard|act\s{1,4}as|behave\s{1,4}as|follow\s{1,4}only))\b))"
    r"|(?P<en_prompt_extract>(?i:\b(?:repeat|reveal|show|print|output|leak|display|dump|share|expose|tell\s{1,4}me)\s{1,8}"
    r"(?:(?:all|me|your|the|full|complete|exact|original|hidden|initial)\s{1,8}){0,4}"
    r"(?:system\s{1,4}prompt|system\s{1,4}instructions|initial\s{1,4}instructions|"
    r"developer\s{1,4}prompt|hidden\s{1,4}prompt)\b))"
    r"|(?P<en_what_prompt>(?i:\bwhat\s{1,4}(?:are|is|was)\s{1,4}your\s{1,4}"
    r"(?:(?:exact|full|original|hidden)\s{1,4}){0,2}"
    r"(?:system\s{1,4}prompt|system\s{1,4}instructions|initial\s{1,4}instructions)\b))"
    r"|(?P<en_dan_jailbreak>\bDAN\b|(?i:\bdo\s{1,4}anything\s{1,4}now\b))"
    r"|(?P<en_jailbreak_mode>(?i:\b(?:jailbreak(?:ing|ed|s)?|"
    r"(?:developer|unrestricted|unfiltered|uncensored|god|sudo)\s{1,4}mode|"
    r"(?:act|respond|answer|operate|run)\s{1,4}(?:as\s{1,4}(?:an?\s{1,4})?"
    r"(?:unrestricted|unfiltered|uncensored)\s{1,4}(?:ai|assistant|model)|"
    r"without\s{1,4}(?:any\s{1,4})?(?:safety\s{1,4})?"
    r"(?:restrictions|limitations|filters|guardrails|rules)))\b))",
    re.ASCII,
)

# Pre-lowercased case-sensitive ASCII sub-patterns for O(N) fast scanning.
_EN_OVERRIDE_LOWER_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b\s{1,8}"
    r"(?:(?:all|any|the|your|previous|prior|above|preceding|earlier|old|safety|system)\s{1,8}){0,4}"
    r"(?:instructions?|rules?|prompts?|directives?|constraints?|guidelines?|policy|filters?|guardrails?|system\s{1,4}prompt)\b",
    re.ASCII,
)
_EN_FORGET_ABOVE_LOWER_RE = re.compile(
    r"\bforget\s{1,4}(?:everything|all)\s{1,4}(?:above|before|prior)\b",
    re.ASCII,
)
_EN_NEW_SYSTEM_LOWER_RE = re.compile(
    r"\b(?:new\s{1,4}system\s{1,4}instructions?|"
    r"from\s{1,4}now\s{1,4}on,?\s{1,4}(?:you\s{1,4}(?:are|will|must)\s{1,4})?"
    r"(?:ignore|disregard|act\s{1,4}as|behave\s{1,4}as|follow\s{1,4}only))\b",
    re.ASCII,
)
_EN_PROMPT_EXTRACT_LOWER_RE = re.compile(
    r"\b(?:repeat|reveal|show|print|output|leak|display|dump|share|expose|tell\s{1,4}me)\s{1,8}"
    r"(?:(?:all|me|your|the|full|complete|exact|original|hidden|initial)\s{1,8}){0,4}"
    r"(?:system\s{1,4}prompt|system\s{1,4}instructions|initial\s{1,4}instructions|"
    r"developer\s{1,4}prompt|hidden\s{1,4}prompt)\b",
    re.ASCII,
)
_EN_WHAT_PROMPT_LOWER_RE = re.compile(
    r"\bwhat\s{1,4}(?:are|is|was)\s{1,4}your\s{1,4}"
    r"(?:(?:exact|full|original|hidden)\s{1,4}){0,2}"
    r"(?:system\s{1,4}prompt|system\s{1,4}instructions|initial\s{1,4}instructions)\b",
    re.ASCII,
)
_EN_DAN_UPPER_RE = re.compile(r"\bDAN\b", re.ASCII)
_EN_DO_ANYTHING_LOWER_RE = re.compile(r"\bdo\s{1,4}anything\s{1,4}now\b", re.ASCII)
_EN_JAILBREAK_MODE_LOWER_RE = re.compile(
    r"\b(?:jailbreak(?:ing|ed|s)?|"
    r"(?:developer|unrestricted|unfiltered|uncensored|god|sudo)\s{1,4}mode|"
    r"(?:act|respond|answer|operate|run)\s{1,4}(?:as\s{1,4}(?:an?\s{1,4})?"
    r"(?:unrestricted|unfiltered|uncensored)\s{1,4}(?:ai|assistant|model)|"
    r"without\s{1,4}(?:any\s{1,4})?(?:safety\s{1,4})?"
    r"(?:restrictions|limitations|filters|guardrails|rules)))\b",
    re.ASCII,
)

_RU_BOUND_L = r"(?<![а-яёА-ЯЁa-zA-Z0-9_])"
_RU_BOUND_R = r"(?![а-яёА-ЯЁa-zA-Z0-9_])"

_RU_DIRECT_RE = re.compile(
    r"(?i)"
    + _RU_BOUND_L
    + r"(?:"
    + r"(?P<ru_instruction_override>"
    + r"(?:забудь|забудьте)\s{1,8}(?:(?:все|любые|свои|предыдущие|прежние|старые)\s{1,8}){0,3}"
    + r"(?:инструкции|инструкций|правила|правил|указания|ограничения|системный\s{1,4}промпт)"
    + r"|(?:игнорируй|игнорируйте|проигнорируй|проигнорируйте)\s{1,8}"
    + r"(?:(?:все|любые|свои|предыдущие|прежние)\s{1,8}){0,3}"
    + r"(?:правила|правил|инструкции|инструкций|ограничения|"
    + r"запреты|указания|системный\s{1,4}промпт)"
    + r"|(?:отмени|отмените|отбрось)\s{1,8}(?:(?:все|предыдущие|старые)\s{1,8}){0,3}"
    + r"(?:инструкции|правила|ограничения|запреты)"
    + r"|новы[ех]\s{1,4}системны[ех]\s{1,4}"
    + r"(?:инструкци[ийя]|правил[а]?|указани[яй]|промпт[а-яё]{0,3})"
    + r")"
    + r"|(?P<ru_prompt_extract>"
    + r"(?:раскрой|раскройте|покажи|покажите|выведи|выведите|"
    + r"повтори|повторите|напечатай|озвучь|сообщи|напиши)\s{1,8}"
    + r"(?:мне\s{1,4})?(?:(?:свой|исходный|полный|точный|скрытый)\s{1,8}){0,2}"
    + r"(?:системный\s{1,4}промпт|системные\s{1,4}инструкции|"
    + r"скрытые\s{1,4}инструкции|системное\s{1,4}сообщение)"
    + r"|(?:какой|каков)\s{1,4}(?:твой|ваш)\s{1,4}(?:системный|исходный)\s{1,4}промпт"
    + r")"
    + r"|(?P<ru_jailbreak_mode>"
    + r"в\s{1,4}режиме\s{1,4}разработчика(?:\s{1,4}без\s{1,4}ограничений)?"
    + r"|(?:включи|перейди\s{1,4}в|активируй)\s{1,4}режим\s{1,4}"
    + r"(?:разработчика|бога|без\s{1,4}ограничений|без\s{1,4}цензуры)"
    + r"|(?:отвечай|работай|действуй|пиши)\s{1,4}без\s{1,4}"
    + r"(?:каких-либо\s{1,4}|всяких\s{1,4})?(?:ограничений|правил|фильтров|цензуры|запретов)"
    + r"|джейлбрейк[а-яёa-z0-9]{0,4}"
    + r")"
    + r")"
    + _RU_BOUND_R
)

_DELIMITER_RE = re.compile(
    r"(?i:</?system>)"
    r"|(?i:<\|(?:im_start|im_end|endoftext|system|assistant|user)\|>)"
    r"|(?i:\[/?INST\])"
    r"|(?i:<</?SYS>>)"
    r"|###[ \t]{0,4}SYSTEM\b"
    r"|(?i:###[ \t]{0,4}SYSTEM[ \t]{0,4}:)",
    re.ASCII,
)
_DELIMITER_LOWER_RE = re.compile(
    r"</?system>"
    r"|<\|(?:im_start|im_end|endoftext|system|assistant|user)\|>"
    r"|\[/?inst\]"
    r"|<</?sys>>"
    r"|###[ \t]{0,4}system[ \t]{0,4}:",
    re.ASCII,
)
_DELIMITER_UPPER_SYSTEM_RE = re.compile(r"###[ \t]{0,4}SYSTEM\b", re.ASCII)

# ---------------------------------------------------------------------------
# Linear-time indirect prompt injection & tool-hijacking patterns
# ---------------------------------------------------------------------------

_MD_COMMENT_RE = re.compile(
    r"(?m)^[ \t]{0,3}\[(?://|comment)\]:[ \t]{1,4}(?:#|<>)[ \t]{1,4}(?P<body>[^\r\n]{1,1024})"
)
_INDIRECT_ASCII_RE = re.compile(
    r"(?P<sql_tool_hijack>(?i:\b(?:DROP\s{1,4}(?:TABLE|DATABASE|SCHEMA)|TRUNCATE\s{1,4}TABLE)\b))"
    r"|(?P<shell_read_hijack>(?i:\b(?:cat|head|tail|less|more|type|Get-Content|base64|xxd)\s{1,4}"
    r"(?:~?/(?:\.ssh|\.aws|\.gnupg)(?:/[^\s;|&]{0,64})?|[^\s;|&]{0,64}\.env\b|/etc/(?:passwd|shadow)|[^\s;|&]{0,64}id_(?:rsa|ed25519))))"
    r"|(?P<curl_pipe_shell>(?i:\b(?:curl|wget)\s{1,4}[^\r\n|]{1,256}\|\s{0,4}(?:ba)?sh\b))"
    r"|(?P<rm_rf_hijack>(?i:\brm\s{1,4}-rf\s{1,4}(?:/|~)))"
    r"|(?P<exfiltrate_en>(?i:\b(?:exfiltrate(?:\s{1,4}(?:[^\s\r\n]{1,32}\s{1,4}){0,4}"
    r"(?:data|secrets?|keys?|tokens?|files?|credentials?))?|"
    r"(?:send|upload|post|forward|transmit|leak)\s{1,4}(?:all\s{1,4}|the\s{1,4})?"
    r"(?:contents?|data|secrets?|credentials?|keys?|tokens?|files?|\.env)\s{1,4}"
    r"(?:to|via)\s{1,4}(?:https?://\S{1,256}|\S{1,64}@\S{1,64}|external|attacker))))",
    re.ASCII,
)
_SQL_HIJACK_LOWER_RE = re.compile(
    r"\b(?:drop\s{1,4}(?:table|database|schema)|truncate\s{1,4}table)\b",
    re.ASCII,
)
_SHELL_READ_LOWER_RE = re.compile(
    r"\b(?:cat|head|tail|less|more|type|get-content|base64|xxd)\s{1,4}"
    r"(?:~?/(?:\.ssh|\.aws|\.gnupg)(?:/[^\s;|&]{0,64})?|[^\s;|&]{0,64}\.env\b|/etc/(?:passwd|shadow)|[^\s;|&]{0,64}id_(?:rsa|ed25519))",
    re.ASCII,
)
_CURL_PIPE_LOWER_RE = re.compile(
    r"\b(?:curl|wget)\s{1,4}[^\r\n|]{1,256}\|\s{0,4}(?:ba)?sh\b",
    re.ASCII,
)
_RM_RF_LOWER_RE = re.compile(r"\brm\s{1,4}-rf\s{1,4}(?:/|~)", re.ASCII)
_EXFIL_EN_LOWER_RE = re.compile(
    r"\b(?:exfiltrate(?:\s{1,4}(?:[^\s\r\n]{1,32}\s{1,4}){0,4}"
    r"(?:data|secrets?|keys?|tokens?|files?|credentials?))?|"
    r"(?:send|upload|post|forward|transmit|leak)\s{1,4}(?:all\s{1,4}|the\s{1,4})?"
    r"(?:contents?|data|secrets?|credentials?|keys?|tokens?|files?|\.env)\s{1,4}"
    r"(?:to|via)\s{1,4}(?:https?://\S{1,256}|\S{1,64}@\S{1,64}|external|attacker))",
    re.ASCII,
)
_SHELL_READ_MARKERS = (".ssh", ".aws", ".gnupg", ".env", "/etc/", "id_rsa", "id_ed25519")
_EXFIL_SEND_VERBS = ("send", "upload", "post", "forward", "transmit", "leak")
_SQL_HIJACK_RU_RE = re.compile(
    r"(?i)"
    + _RU_BOUND_L
    + r"выполни\s{1,8}(?:DROP\s{1,4}(?:TABLE|DATABASE|SCHEMA)|TRUNCATE\s{1,4}TABLE)\b"
)
_EXFIL_RU_RE = re.compile(
    r"(?i)"
    + _RU_BOUND_L
    + r"(?:отправь|перешли|передай|слей|выгрузи)\s{1,4}"
    + r"(?:все\s{1,4})?(?:содержимое|данные|файлы|секреты|ключи|пароли|токены|переписку)\s{1,4}"
    + r"(?:[^\s\r\n]{1,32}\s{1,4}){0,4}(?:на|в|по)"
    + _RU_BOUND_R
    + r"(?:\s{1,4}\S{1,256})?"
)

_COMMENT_DIRECTIVE_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:ignore|disregard|forget|override|system\s+prompt|new\s+instructions?|jailbreak|"
    r"execute|drop\s+table|truncate\s+table|delete\s+from|exfiltrate|steal|leak|"
    r"curl|wget|bash|cat\s+|~/.ssh|\.env|/etc/passwd|https?://|send\s+to|upload\s+to|"
    r"post\s+to|assistant|instructions?|secret|password|token)\b"
    r"|!\["
    r"|<img\b"
    r"|(?:игнорируй|забудь|выполни|отправь|перешли|слей|прочитай|удали|"
    r"инструкц|системн|секрет|пароль|токен)"
    r")"
)
_MAX_COMMENT_LEN = 16_384

# ---------------------------------------------------------------------------
# EchoLeak Markdown & HTML image exfiltration patterns (CVE-2025-32711)
# ---------------------------------------------------------------------------

_INLINE_MD_IMG_RE = re.compile(
    r"!\[(?P<alt>[^\]\r\n]{0,200})\]"
    r"\(<?(?P<url>https?://[^\s>)\r\n]{1,1024})>?"
    r"(?:[ \t]{1,8}(?:\"[^\"\r\n]{0,200}\"|'[^'\r\n]{0,200}'))?\)",
    re.ASCII,
)
_REF_DEF_RE = re.compile(
    r"(?m)^[ \t]{0,3}\[(?P<id>[^\]\r\n]{1,100})\]:[ \t]{0,8}<?(?P<url>https?://[^\s>\r\n]{1,1024})>?",
    re.ASCII,
)
_REF_IMG_FULL_RE = re.compile(
    r"!\[(?P<alt>[^\]\r\n]{0,200})\][ \t]?\[(?P<id>[^\]\r\n]{0,100})\]",
    re.ASCII,
)
_REF_IMG_SHORTCUT_RE = re.compile(r"!\[(?P<id>[^\]\r\n]{1,100})\](?![ \t]?[(\[])", re.ASCII)
_HTML_IMG_TAG_RE = re.compile(r"<img\b(?P<attrs>[^>\r\n]{1,1024})>", re.ASCII | re.IGNORECASE)
_HTML_IMG_SRC_RE = re.compile(
    r"""\bsrc\s{0,4}=\s{0,4}(?:"(?P<dq>https?://[^"\r\n]{1,1024})"|'(?P<sq>https?://[^'\r\n]{1,1024})'|(?P<uq>https?://[^\s>"']{1,1024}))""",
    re.ASCII | re.IGNORECASE,
)
_HTML_TRACKING_ATTR_RE = re.compile(
    r"""(?:\b(?:width|height)\s{0,4}=\s{0,4}["']?[01](?:px)?["']?(?:\s|$)|display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\s|;|$))""",
    re.ASCII | re.IGNORECASE,
)
_EXFIL_PARAM_HINT_RE = re.compile(
    r"(?:^|[?&#/])(?:data|secret|token|key|leak|exfil|steal|payload|q|p|d|s|b64|msg|body|content|track|pixel|beacon|collect|spy)=",
    re.ASCII | re.IGNORECASE,
)
_TRACKING_URL_HINT_RE = re.compile(
    r"/(?:track|pixel|beacon|collect|exfil(?:trate)?|steal|leak|spy|log)(?:[/?#._-]|$)",
    re.ASCII | re.IGNORECASE,
)
_TEMPLATE_HINT_RE = re.compile(r"(?:\{\{|\}\}|\$\{|%7b|%7d)", re.ASCII | re.IGNORECASE)
_IMAGE_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|svg|webp|ico|avif|bmp)$", re.ASCII | re.IGNORECASE)
_BASE64_SEG_RE = re.compile(r"^[A-Za-z0-9+/_-]{16,}={0,2}$", re.ASCII)

_DIRECT_CONFIDENCE: dict[str, float] = {
    "en_instruction_override": 0.96,
    "en_forget_above": 0.94,
    "en_new_system": 0.92,
    "en_prompt_extract": 0.95,
    "en_what_prompt": 0.90,
    "en_dan_jailbreak": 0.93,
    "en_jailbreak_mode": 0.94,
    "ru_instruction_override": 0.96,
    "ru_prompt_extract": 0.95,
    "ru_jailbreak_mode": 0.94,
}

_INDIRECT_CONFIDENCE: dict[str, float] = {
    "sql_tool_hijack": 0.96,
    "shell_read_hijack": 0.97,
    "curl_pipe_shell": 0.98,
    "rm_rf_hijack": 0.97,
    "exfiltrate_en": 0.94,
}


@dataclass(frozen=True, slots=True)
class _FindingSpan:
    start: int
    end: int
    category: EntityCategory
    confidence: float
    rule_id: str


def _looks_like_base64_segment(segment: str) -> bool:
    if _IMAGE_EXT_RE.search(segment) or not _BASE64_SEG_RE.match(segment):
        return False
    if segment.endswith("="):
        return True
    padded = segment.replace("-", "+").replace("_", "/")
    pad_len = (-len(padded)) % 4
    try:
        decoded = base64.b64decode(padded + ("=" * pad_len), validate=True)
    except Exception:
        return False
    if len(decoded) < 8:
        return False
    printable = sum(32 <= b <= 126 for b in decoded)
    return printable / len(decoded) >= 0.85


def _is_suspicious_exfil_url(url: str) -> bool:
    """Return True if an image URL carries query strings, templates, or encoded payloads."""
    if "?" in url or _TEMPLATE_HINT_RE.search(url) or _EXFIL_PARAM_HINT_RE.search(url):
        return True
    parsed = urlsplit(url)
    if parsed.query:
        return True
    return any(part and _looks_like_base64_segment(part) for part in parsed.path.split("/"))


def _deduplicate_findings(candidates: list[_FindingSpan]) -> list[_FindingSpan]:
    """Deduplicate overlapping same-category spans and drop spans inside HTML comments."""
    comment_spans = [
        (item.start, item.end) for item in candidates if item.rule_id == "hidden_html_comment"
    ]
    filtered: list[_FindingSpan] = []
    for item in candidates:
        if (
            item.rule_id != "hidden_html_comment"
            and item.category
            in (
                EntityCategory.PROMPT_INJECTION,
                EntityCategory.INDIRECT_INJECTION,
            )
            and any(cs <= item.start and item.end <= ce for cs, ce in comment_spans)
        ):
            continue
        filtered.append(item)

    ranked = sorted(
        filtered,
        key=lambda item: (-item.confidence, -(item.end - item.start), item.start),
    )
    accepted: list[_FindingSpan] = []
    for item in ranked:
        if any(
            existing.category == item.category
            and not (item.end <= existing.start or item.start >= existing.end)
            for existing in accepted
        ):
            continue
        accepted.append(item)
    accepted.sort(key=lambda item: (item.start, item.end))
    return accepted


class PromptInjectionDetector:
    """Layer-1 detector for RU/EN prompt injections, indirect hijackings, and EchoLeak exfil."""

    def __init__(self, hmac_fn: Callable[[str], str] | None = None) -> None:
        self._hmac_fn = hmac_fn

    @property
    def name(self) -> str:
        """Return detector identifier."""
        return "prompt_injection"

    @property
    def layer(self) -> int:
        """Return cascade layer index (1 = fast deterministic)."""
        return 1

    def detect(self, segment: NormalizedSegment, raw_text: str) -> list[Detection]:
        """Scan a normalized segment for prompt injections and exfiltration payloads."""
        text = segment.text
        if not text:
            return []

        lower_text = text.lower()
        use_lower = len(lower_text) == len(text)
        has_cyrillic = (not text.isascii()) and (_HAS_CYRILLIC_RE.search(text) is not None)
        candidates: list[_FindingSpan] = []

        self._collect_direct_rules(text, lower_text, use_lower, has_cyrillic, candidates)
        self._collect_indirect_rules(text, lower_text, use_lower, has_cyrillic, candidates)
        self._collect_hidden_comments(text, candidates)
        self._collect_markdown_exfil(text, lower_text, candidates)

        if not candidates:
            return []

        selected = _deduplicate_findings(candidates)
        evasion_detected = bool(segment.evasion_techniques)

        detections: list[Detection] = []
        for item in selected:
            matched = text[item.start : item.end]
            mapped_start, mapped_end = segment.map_span(item.start, item.end)
            detections.append(
                Detection(
                    category=item.category,
                    start=mapped_start,
                    end=mapped_end,
                    confidence=item.confidence,
                    detector=self.name,
                    span_source=segment.span_source,
                    evasion_detected=evasion_detected,
                    evasion_techniques=segment.evasion_techniques,
                    compliance_refs=get_compliance_refs(item.category),
                    masked_preview=f"[{item.category.value}:len={len(matched)}]",
                    value_hmac=self._hmac_fn(matched) if self._hmac_fn else "",
                    metadata={"rule": item.rule_id},
                )
            )
        return detections

    def _collect_direct_rules(
        self,
        text: str,
        lower_text: str,
        use_lower: bool,
        has_cyrillic: bool,
        out: list[_FindingSpan],
    ) -> None:
        if not use_lower:
            for match in _EN_DIRECT_RE.finditer(text):
                rule_id = match.lastgroup or "en_prompt_injection"
                out.append(
                    _FindingSpan(
                        start=match.start(),
                        end=match.end(),
                        category=EntityCategory.PROMPT_INJECTION,
                        confidence=_DIRECT_CONFIDENCE.get(rule_id, 0.92),
                        rule_id=rule_id,
                    )
                )
        else:
            has_prompt_or_instr = "prompt" in lower_text or "instructions" in lower_text
            if any(v in lower_text for v in _EN_OVERRIDE_VERBS) and any(
                n in lower_text for n in _EN_OVERRIDE_NOUNS
            ):
                for match in _EN_OVERRIDE_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_instruction_override"],
                            rule_id="en_instruction_override",
                        )
                    )
            if "forget" in lower_text and (
                "above" in lower_text or "before" in lower_text or "prior" in lower_text
            ):
                for match in _EN_FORGET_ABOVE_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_forget_above"],
                            rule_id="en_forget_above",
                        )
                    )
            if ("new" in lower_text and "system" in lower_text) or (
                "from" in lower_text and "now" in lower_text
            ):
                for match in _EN_NEW_SYSTEM_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_new_system"],
                            rule_id="en_new_system",
                        )
                    )
            if has_prompt_or_instr and any(v in lower_text for v in _EN_EXTRACT_VERBS):
                for match in _EN_PROMPT_EXTRACT_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_prompt_extract"],
                            rule_id="en_prompt_extract",
                        )
                    )
            if has_prompt_or_instr and "what" in lower_text:
                for match in _EN_WHAT_PROMPT_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_what_prompt"],
                            rule_id="en_what_prompt",
                        )
                    )
            if "DAN" in text:
                for match in _EN_DAN_UPPER_RE.finditer(text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_dan_jailbreak"],
                            rule_id="en_dan_jailbreak",
                        )
                    )
            if "do" in lower_text and "now" in lower_text:
                for match in _EN_DO_ANYTHING_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_dan_jailbreak"],
                            rule_id="en_dan_jailbreak",
                        )
                    )
            if any(hint in lower_text for hint in _EN_JAILBREAK_HINTS):
                for match in _EN_JAILBREAK_MODE_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_jailbreak_mode"],
                            rule_id="en_jailbreak_mode",
                        )
                    )

        if has_cyrillic:
            for match in _RU_DIRECT_RE.finditer(text):
                rule_id = match.lastgroup or "ru_prompt_injection"
                out.append(
                    _FindingSpan(
                        start=match.start(),
                        end=match.end(),
                        category=EntityCategory.PROMPT_INJECTION,
                        confidence=_DIRECT_CONFIDENCE.get(rule_id, 0.94),
                        rule_id=rule_id,
                    )
                )

        if "<" in text or "[" in text or "#" in text:
            if use_lower:
                if (
                    "system>" in lower_text
                    or "<|" in lower_text
                    or "inst]" in lower_text
                    or "sys>>" in lower_text
                    or ("###" in text and ":" in text)
                ):
                    for match in _DELIMITER_LOWER_RE.finditer(lower_text):
                        out.append(
                            _FindingSpan(
                                start=match.start(),
                                end=match.end(),
                                category=EntityCategory.PROMPT_INJECTION,
                                confidence=0.97,
                                rule_id="delimiter_attack",
                            )
                        )
                if "###" in text and "SYSTEM" in text:
                    for match in _DELIMITER_UPPER_SYSTEM_RE.finditer(text):
                        out.append(
                            _FindingSpan(
                                start=match.start(),
                                end=match.end(),
                                category=EntityCategory.PROMPT_INJECTION,
                                confidence=0.97,
                                rule_id="delimiter_attack",
                            )
                        )
            else:
                for match in _DELIMITER_RE.finditer(text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=0.97,
                            rule_id="delimiter_attack",
                        )
                    )

    def _collect_indirect_rules(
        self,
        text: str,
        lower_text: str,
        use_lower: bool,
        has_cyrillic: bool,
        out: list[_FindingSpan],
    ) -> None:
        if not use_lower:
            for match in _INDIRECT_ASCII_RE.finditer(text):
                rule_id = match.lastgroup or "indirect_injection"
                out.append(
                    _FindingSpan(
                        start=match.start(),
                        end=match.end(),
                        category=EntityCategory.INDIRECT_INJECTION,
                        confidence=_INDIRECT_CONFIDENCE.get(rule_id, 0.94),
                        rule_id=rule_id,
                    )
                )
        else:
            if "table" in lower_text or "database" in lower_text or "schema" in lower_text:
                for match in _SQL_HIJACK_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.INDIRECT_INJECTION,
                            confidence=_INDIRECT_CONFIDENCE["sql_tool_hijack"],
                            rule_id="sql_tool_hijack",
                        )
                    )
            if any(m in lower_text for m in _SHELL_READ_MARKERS):
                for match in _SHELL_READ_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.INDIRECT_INJECTION,
                            confidence=_INDIRECT_CONFIDENCE["shell_read_hijack"],
                            rule_id="shell_read_hijack",
                        )
                    )
            if "|" in text and ("curl" in lower_text or "wget" in lower_text):
                for match in _CURL_PIPE_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.INDIRECT_INJECTION,
                            confidence=_INDIRECT_CONFIDENCE["curl_pipe_shell"],
                            rule_id="curl_pipe_shell",
                        )
                    )
            if "-rf" in lower_text and "rm" in lower_text:
                for match in _RM_RF_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.INDIRECT_INJECTION,
                            confidence=_INDIRECT_CONFIDENCE["rm_rf_hijack"],
                            rule_id="rm_rf_hijack",
                        )
                    )
            if "exfiltrate" in lower_text or (
                any(v in lower_text for v in _EXFIL_SEND_VERBS)
                and ("to" in lower_text or "via" in lower_text)
            ):
                for match in _EXFIL_EN_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.INDIRECT_INJECTION,
                            confidence=_INDIRECT_CONFIDENCE["exfiltrate_en"],
                            rule_id="exfiltrate_en",
                        )
                    )

        if has_cyrillic:
            for match in _SQL_HIJACK_RU_RE.finditer(text):
                out.append(
                    _FindingSpan(
                        start=match.start(),
                        end=match.end(),
                        category=EntityCategory.INDIRECT_INJECTION,
                        confidence=0.96,
                        rule_id="sql_tool_hijack_ru",
                    )
                )
            for match in _EXFIL_RU_RE.finditer(text):
                out.append(
                    _FindingSpan(
                        start=match.start(),
                        end=match.end(),
                        category=EntityCategory.INDIRECT_INJECTION,
                        confidence=0.94,
                        rule_id="exfiltrate_ru",
                    )
                )

    def _collect_hidden_comments(self, text: str, out: list[_FindingSpan]) -> None:
        if "<!--" in text and "-->" in text:
            pos = 0
            text_len = len(text)
            while pos < text_len:
                start = text.find("<!--", pos)
                if start == -1:
                    break
                search_end = min(text_len, start + 4 + _MAX_COMMENT_LEN)
                end_marker = text.find("-->", start + 4, search_end)
                if end_marker == -1:
                    pos = start + 4
                    continue
                body = text[start + 4 : end_marker]
                if _COMMENT_DIRECTIVE_RE.search(body):
                    out.append(
                        _FindingSpan(
                            start=start,
                            end=end_marker + 3,
                            category=EntityCategory.INDIRECT_INJECTION,
                            confidence=0.95,
                            rule_id="hidden_html_comment",
                        )
                    )
                pos = end_marker + 3

        if "]:" in text:
            for match in _MD_COMMENT_RE.finditer(text):
                if _COMMENT_DIRECTIVE_RE.search(match.group("body")):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.INDIRECT_INJECTION,
                            confidence=0.93,
                            rule_id="hidden_markdown_comment",
                        )
                    )

    def _collect_markdown_exfil(
        self,
        text: str,
        lower_text: str,
        out: list[_FindingSpan],
    ) -> None:
        if "![" in text:
            if "](" in text and ")" in text:
                for match in _INLINE_MD_IMG_RE.finditer(text):
                    url = match.group("url")
                    if _is_suspicious_exfil_url(url) or _TRACKING_URL_HINT_RE.search(url):
                        out.append(
                            _FindingSpan(
                                start=match.start(),
                                end=match.end(),
                                category=EntityCategory.MARKDOWN_EXFIL,
                                confidence=0.97,
                                rule_id="echoleak_inline_markdown_image",
                            )
                        )

            # Reference-style markdown image exfil: ![alt][ref] / ![ref][] / ![ref] -> [ref]: url
            if "]:" in text:
                ref_defs: dict[str, tuple[int, int, str]] = {}
                for match in _REF_DEF_RE.finditer(text):
                    ref_id = match.group("id").strip().lower()
                    ref_defs[ref_id] = (match.start(), match.end(), match.group("url"))

                if ref_defs:
                    used_refs: set[str] = set()
                    for match in _REF_IMG_FULL_RE.finditer(text):
                        ref_id = (match.group("id") or match.group("alt")).strip().lower()
                        if ref_id in ref_defs:
                            used_refs.add(ref_id)
                    for match in _REF_IMG_SHORTCUT_RE.finditer(text):
                        ref_id = match.group("id").strip().lower()
                        if ref_id in ref_defs:
                            used_refs.add(ref_id)
                    for ref_id in used_refs:
                        def_start, def_end, url = ref_defs[ref_id]
                        if _is_suspicious_exfil_url(url) or _TRACKING_URL_HINT_RE.search(url):
                            out.append(
                                _FindingSpan(
                                    start=def_start,
                                    end=def_end,
                                    category=EntityCategory.MARKDOWN_EXFIL,
                                    confidence=0.96,
                                    rule_id="echoleak_reference_markdown_image",
                                )
                            )

        if "<img" in lower_text and ">" in text:
            for match in _HTML_IMG_TAG_RE.finditer(text):
                attrs = match.group("attrs")
                src_match = _HTML_IMG_SRC_RE.search(attrs)
                if src_match is None:
                    continue
                url = src_match.group("dq") or src_match.group("sq") or src_match.group("uq") or ""
                if (
                    _is_suspicious_exfil_url(url)
                    or _TRACKING_URL_HINT_RE.search(url)
                    or _HTML_TRACKING_ATTR_RE.search(attrs)
                ):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.MARKDOWN_EXFIL,
                            confidence=0.96,
                            rule_id="echoleak_html_img_tag",
                        )
                    )


def _softmax_positive_prob(logits: list[float]) -> float:
    """Compute positive-class probability from 1D or 2D model logits."""
    if not logits:
        return 0.0
    if len(logits) == 1:
        val = max(-60.0, min(60.0, logits[0]))
        return 1.0 / (1.0 + math.exp(-val))
    max_logit = max(logits)
    exps = [math.exp(max(-60.0, min(60.0, val - max_logit))) for val in logits]
    total = sum(exps)
    if total <= 0.0:
        return 0.0
    return exps[1] / total


class OnnxPromptInjectionDetector:
    """Layer-2 optional ONNX prompt injection classifier wrapper.

    Gracefully returns ``[]`` when ``onnxruntime`` / ``tokenizers`` (``[ml]``)
    are not installed and no custom ``infer_fn`` is provided.
    """

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.85,
        infer_fn: Callable[[str], float] | None = None,
        hmac_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._infer_fn = infer_fn
        self._hmac_fn = hmac_fn
        self._session: Any = None
        self._tokenizer: Any = None
        self._np_mod: Any = None
        self._load_attempted = False

    @property
    def name(self) -> str:
        """Return detector identifier."""
        return "prompt_injection_onnx"

    @property
    def layer(self) -> int:
        """Return cascade layer index (2 = ML/ONNX fallback)."""
        return 2

    def _ensure_loaded(self) -> bool:
        if self._session is not None and self._tokenizer is not None:
            return True
        if self._load_attempted or not self._model_path:
            return False
        self._load_attempted = True

        try:
            ort_mod = importlib.import_module("onnxruntime")
            tok_mod = importlib.import_module("tokenizers")
            np_mod = importlib.import_module("numpy")
        except ImportError:
            return False

        model_target = Path(self._model_path)
        if model_target.is_dir():
            onnx_path = model_target / "model.onnx"
            if not onnx_path.is_file():
                candidates = sorted(model_target.glob("*.onnx"))
                if not candidates:
                    return False
                onnx_path = candidates[0]
            tok_path = model_target / "tokenizer.json"
        else:
            onnx_path = model_target
            tok_path = model_target.parent / "tokenizer.json"

        if not onnx_path.is_file() or not tok_path.is_file():
            return False

        try:
            tokenizer_cls = tok_mod.Tokenizer
            tokenizer = tokenizer_cls.from_file(str(tok_path))
            if hasattr(tokenizer, "enable_truncation"):
                tokenizer.enable_truncation(max_length=512)
            session = ort_mod.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        except Exception:
            return False

        self._tokenizer = tokenizer
        self._session = session
        self._np_mod = np_mod
        return True

    def _run_onnx_inference(self, text: str) -> float | None:
        if not self._ensure_loaded():
            return None
        try:
            encoding = self._tokenizer.encode(text)
            input_ids = self._np_mod.asarray([encoding.ids], dtype=self._np_mod.int64)
            attention_mask = self._np_mod.asarray(
                [encoding.attention_mask], dtype=self._np_mod.int64
            )
            input_names = {inp.name for inp in self._session.get_inputs()}
            feeds: dict[str, Any] = {}
            if "input_ids" in input_names:
                feeds["input_ids"] = input_ids
            if "attention_mask" in input_names:
                feeds["attention_mask"] = attention_mask
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = self._np_mod.zeros_like(input_ids)
            outputs = self._session.run(None, feeds)
            logits = [float(x) for x in outputs[0][0]]
            return _softmax_positive_prob(logits)
        except Exception:
            return None

    def detect(self, segment: NormalizedSegment, raw_text: str) -> list[Detection]:
        """Score a normalized segment with ``infer_fn`` or an optional CPU ONNX model."""
        text = segment.text
        if not text.strip():
            return []

        score: float | None
        if self._infer_fn is not None:
            score = float(self._infer_fn(text))
        else:
            score = self._run_onnx_inference(text)

        if score is None or score < self._threshold:
            return []

        confidence = min(1.0, max(0.0, float(score)))
        mapped_start, mapped_end = segment.map_span(0, len(text))
        category = EntityCategory.PROMPT_INJECTION
        return [
            Detection(
                category=category,
                start=mapped_start,
                end=mapped_end,
                confidence=confidence,
                detector=self.name,
                span_source=segment.span_source,
                evasion_detected=bool(segment.evasion_techniques),
                evasion_techniques=segment.evasion_techniques,
                compliance_refs=get_compliance_refs(category),
                masked_preview=f"[{category.value}:len={len(text)}]",
                value_hmac=self._hmac_fn(text) if self._hmac_fn else "",
                metadata={
                    "model": "custom_infer_fn" if self._infer_fn is not None else "onnx",
                },
            )
        ]


__all__ = ["OnnxPromptInjectionDetector", "PromptInjectionDetector"]
