"""Prompt injection, indirect injection, EchoLeak Markdown exfil, and ONNX hook detectors.

Layer-1 deterministic rules cover Russian and English direct prompt injections,
paraphrase/word-order overrides, delimiter attacks, hidden HTML/Markdown comments,
tool hijacking, and EchoLeak (CVE-2025-32711) Markdown/HTML exfiltration in
O(N + M log M) time. Layer-2 ``OnnxPromptInjectionDetector`` provides an optional
CPU ONNX hook (or custom ``infer_fn``) that degrades gracefully when ``[ml]`` is absent.
"""

from __future__ import annotations

import base64
import html
import importlib
import math
import re
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.contracts import (
    ComplianceRef,
    Detection,
    EntityCategory,
    NormalizedSegment,
    SpanSource,
)

_COMPLIANCE_CACHE: dict[EntityCategory, tuple[ComplianceRef, ...]] = {
    cat: get_compliance_refs(cat) for cat in EntityCategory
}
_DETECTION_FIELDS_SET: frozenset[str] = frozenset(Detection.model_fields)

# ---------------------------------------------------------------------------
# Linear-time direct prompt injection & delimiter patterns (RU + EN)
# ---------------------------------------------------------------------------

_HAS_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")

_LEET_DIGIT_TRIGGER_RE = re.compile(r"[013457@$]")
_LEET_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Za-z]+[013457@$][A-Za-z0-9@$]*|[013457@$]+[A-Za-z][A-Za-z0-9@$]*)"
    r"(?![A-Za-z0-9])"
)
_LEET_TRANS = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)
_SPACED_RUN_RE = re.compile(r"(?<![^\W_])(?:[A-Za-zА-Яа-яЁё][ \t]){2,}[A-Za-zА-Яа-яЁё](?![^\W_])")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

_EN_OVERRIDE_VERBS = ("ignore", "disregard", "forget", "override", "discard")
_EN_OVERRIDE_NOUNS = (
    "instruction",
    "rule",
    "prompt",
    "directive",
    "constraint",
    "guideline",
    "policy",
    "policies",
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
    "quote",
    "echo",
)
_EN_JAILBREAK_HINTS = (
    "jailbreak",
    "mode",
    "without",
    "unrestricted",
    "unfiltered",
    "uncensored",
    "suspended",
    "revoked",
)

_EN_MODIFIER_WORDS = (
    r"(?:all|any|of|the|your|very|previous|prior|above|preceding|earlier|old|"
    r"outdated|existing|safety|system|initial|content)"
)

_EN_DIRECT_RE = re.compile(
    rf"(?P<en_instruction_override>(?i:\b(?:ignore|disregard|forget|override|discard)\b\s{{1,8}}"
    rf"(?:{_EN_MODIFIER_WORDS}\s{{1,8}}){{0,10}}"
    r"(?:instructions?|rules?|prompts?|directives?|constraints?|guidelines?|polic(?:y|ies)|"
    r"filters?|guardrails?|system\s{1,4}prompt)\b))"
    r"|(?P<en_passive_override>(?i:\b(?:all\s{1,4}|any\s{1,4})?(?:prior|previous|preceding|"
    r"existing|system|safety|content)\s{1,4}(?:(?:safety|system|content)\s{1,4}){0,2}"
    r"(?:instructions?|rules?|directives?|polic(?:y|ies)|constraints?|guardrails?|filters?)\s{1,4}"
    r"(?:are|is|have\s{1,4}been)\s{1,4}(?:hereby\s{1,4}|now\s{1,4}|temporarily\s{1,4})?"
    r"(?:revoked|cancelled|canceled|void(?:ed)?|suspended|disabled|overridden|ignored|lifted)\b))"
    r"|(?P<en_forget_above>(?i:\b(?:forget|discard|disregard|ignore)\s{1,4}"
    r"(?:everything|all(?:\s{1,4}text)?|anything)\s{1,4}(?:written\s{1,4}|appearing\s{1,4})?"
    r"(?:above|before|prior(?:\s{1,4}to)?)(?:\s{1,4}this(?:\s{1,4}line)?)?\b))"
    r"|(?P<en_new_system>(?i:\b(?:new\s{1,4}system\s{1,4}instructions?|"
    r"from\s{1,4}now\s{1,4}on,?\s{1,4}(?:you\s{1,4}(?:are|will|must)\s{1,4})?"
    r"(?:ignore|disregard|act\s{1,4}as|behave\s{1,4}as|follow\s{1,4}only))\b))"
    r"|(?P<en_prompt_extract>(?i:\b(?:repeat|reveal|show|print|output|leak|display|dump|"
    r"share|expose|quote|echo|tell\s{1,4}me)\s{1,8}"
    r"(?:(?:verbatim|all|me|your|the|full|complete|exact|original|hidden|initial|every\s{1,4}word)\s{1,8}){0,5}"
    r"(?:system\s{1,4}prompt|system\s{1,4}instructions|initial\s{1,4}instructions|"
    r"developer\s{1,4}prompt|hidden\s{1,4}prompt|(?:appearing\s{1,4}|written\s{1,4})?(?:above|before)\s{1,4}this\s{1,4}(?:user\s{1,4})?message)\b))"
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
    rf"\b(?:ignore|disregard|forget|override|discard)\b\s{{1,8}}"
    rf"(?:{_EN_MODIFIER_WORDS}\s{{1,8}}){{0,10}}"
    r"(?:instructions?|rules?|prompts?|directives?|constraints?|guidelines?|polic(?:y|ies)|"
    r"filters?|guardrails?|system\s{1,4}prompt)\b",
    re.ASCII,
)
_EN_PASSIVE_OVERRIDE_LOWER_RE = re.compile(
    r"\b(?:all\s{1,4}|any\s{1,4})?(?:prior|previous|preceding|existing|system|safety|content)\s{1,4}"
    r"(?:(?:safety|system|content)\s{1,4}){0,2}"
    r"(?:instructions?|rules?|directives?|polic(?:y|ies)|constraints?|guardrails?|filters?)\s{1,4}"
    r"(?:are|is|have\s{1,4}been)\s{1,4}(?:hereby\s{1,4}|now\s{1,4}|temporarily\s{1,4})?"
    r"(?:revoked|cancelled|canceled|void(?:ed)?|suspended|disabled|overridden|ignored|lifted)\b",
    re.ASCII,
)
_EN_FORGET_ABOVE_LOWER_RE = re.compile(
    r"\b(?:forget|discard|disregard|ignore)\s{1,4}"
    r"(?:everything|all(?:\s{1,4}text)?|anything)\s{1,4}(?:written\s{1,4}|appearing\s{1,4})?"
    r"(?:above|before|prior(?:\s{1,4}to)?)(?:\s{1,4}this(?:\s{1,4}line)?)?\b",
    re.ASCII,
)
_EN_NEW_SYSTEM_LOWER_RE = re.compile(
    r"\b(?:new\s{1,4}system\s{1,4}instructions?|"
    r"from\s{1,4}now\s{1,4}on,?\s{1,4}(?:you\s{1,4}(?:are|will|must)\s{1,4})?"
    r"(?:ignore|disregard|act\s{1,4}as|behave\s{1,4}as|follow\s{1,4}only))\b",
    re.ASCII,
)
_EN_PROMPT_EXTRACT_LOWER_RE = re.compile(
    r"\b(?:repeat|reveal|show|print|output|leak|display|dump|share|expose|quote|echo|tell\s{1,4}me)\s{1,8}"
    r"(?:(?:verbatim|all|me|your|the|full|complete|exact|original|hidden|initial|every\s{1,4}word)\s{1,8}){0,5}"
    r"(?:system\s{1,4}prompt|system\s{1,4}instructions|initial\s{1,4}instructions|"
    r"developer\s{1,4}prompt|hidden\s{1,4}prompt|(?:appearing\s{1,4}|written\s{1,4})?(?:above|before)\s{1,4}this\s{1,4}(?:user\s{1,4})?message)\b",
    re.ASCII,
)
_EN_WHAT_PROMPT_LOWER_RE = re.compile(
    r"\bwhat\s{1,4}(?:are|is|was)\s{1,4}your\s{1,4}"
    r"(?:(?:exact|full|original|hidden)\s{1,4}){0,2}"
    r"(?:system\s{1,4}prompt|system\s{1,4}instructions|initial\s{1,4}instructions)\b",
    re.ASCII,
)
_EN_SPLIT_VARS_LOWER_RE = re.compile(
    r"\b(?:let|var|const|set)\s+[a-z0-9_]+\s*=\s*[\"'][^\"'\r\n]{0,64}"
    r"\b(?:ignore|disregard|forget|override)\b[^\"'\r\n]{0,64}[\"'][^\r\n]{1,120}"
    r"\b(?:instructions?|rules?|system\s+prompt)\b[^\r\n]{0,80}\b(?:execute|eval|run|follow)\b",
    re.ASCII,
)
_EN_SENTENCE_INVERTED_LOWER_RE = re.compile(
    r"\b(?:all\s+|any\s+|your\s+)?(?:previous|prior|preceding|system|safety)\s+"
    r"(?:(?:system|safety)\s+)?(?:instructions?|rules?|directives?|prompts?)"
    r"[^.!?\r\n]{1,60}\b(?:must\s+be\s+)?(?:ignore[d]?|disregard(?:ed)?|forget|forgotten|overridden|discard(?:ed)?)\b",
    re.ASCII,
)

_EN_DAN_UPPER_RE = re.compile(r"\bDAN\b", re.ASCII)
_DAN_JAILBREAK_CTX_RE = re.compile(
    r"(?i)(?:\b(?:you\s+are(?:\s+now)?|act\s+as|behave\s+as|enable|enter|mode)\b[^\r\n]{0,24}\bDAN\b"
    r"|\bDAN\b[^\r\n]{0,24}\b(?:mode|jailbreak|do\s+anything\s+now|\d+(?:\.\d+)?)\b"
    r"|(?:ты\s+(?:теперь\s+)?|режим\w*\s+)DAN\b)",
)
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

_DEFENSIVE_EDUCATIONAL_PREFIX_RE = re.compile(
    r"(?i)(?:как\s+защитить[^\r\n]{0,60}от\s+атак\s+(?:вида|типа)\s*$"
    r"|защит\w+\s+от\s+(?:prompt\s+injection\s+|атак\s+)(?:вида|типа)\s*$"
    r"|how\s+to\s+(?:defend|protect)[^\r\n]{0,60}against\s+attacks?\s+(?:like|such\s+as)\s*$)",
)

_RU_BOUND_L = r"(?<![а-яёА-ЯЁa-zA-Z0-9_])"
_RU_BOUND_R = r"(?![а-яёА-ЯЁa-zA-Z0-9_])"

_RU_DIRECT_RE = re.compile(
    r"(?i)"
    + _RU_BOUND_L
    + r"(?:"
    + r"(?P<ru_instruction_override>"
    + r"(?:забудь|забудьте)\s{1,8}(?:(?:все|любые|свои|предыдущие|прежние|старые)\s{1,8}){0,5}"
    + r"(?:инструкци[ийя]|правил[а]?|указани[яй]|установк[и]|ограничени[яй]|системный\s{1,4}промпт)"
    + r"|(?:игнорируй|игнорируйте|проигнорируй|проигнорируйте)\s{1,8}"
    + r"(?:(?:все|любые|свои|предыдущие|прежние|старые|системные)\s{1,8}){0,5}"
    + r"(?:правил[а]?|инструкци[ийя]|ограничени[яй]|запреты|указани[яй]|"
    + r"установк[и]|системный\s{1,4}промпт)"
    + r"|(?:отмени|отмените|отбрось|сбрось|обнули|наплюй\s{1,4}на)\s{1,8}"
    + r"(?:(?:все|свои|предыдущие|прежние|старые|системные)\s{1,8}){0,5}"
    + r"(?:инструкци[ийя]|правил[а]?|ограничени[яй]|запреты|установк[и]|"
    + r"указани[яй]|системный\s{1,4}контекст|контекст)"
    + r"|не\s{1,4}обращай(?:те)?\s{1,4}внимания\s{1,4}на\s{1,4}"
    + r"(?:(?:все|любые|предыдущие|прежние|системные)\s{1,4}){0,4}"
    + r"(?:инструкци[ийя]|правил[а]?|указани[яй]|ограничени[яй]|установк[и])"
    + r"|(?:все\s{1,4}|любые\s{1,4})?(?:предыдущие|прежние|старые|системные)\s{1,4}"
    + r"(?:инструкци[ийя]|правил[а]?|указани[яй]|установк[и]|ограничени[яй])"
    + r"[^.!?\r\n]{1,50}(?:проигнорируй(?:те)?|игнорируй(?:те)?|забудь(?:те)?|"
    + r"отмени(?:те)?|отбрось(?:те)?)"
    + r"|с\s{1,4}этого\s{1,4}момента\s{1,4}ты\s{1,4}(?:обязан|должен)\s{1,4}"
    + r"(?:выполнять\s{1,4}только|забыть|игнорировать|подчиняться)"
    + r"|новы[ех]\s{1,4}системны[ех]\s{1,4}"
    + r"(?:инструкци[ийя]|правил[а]?|указани[яй]|промпт[а-яё]{0,3})"
    + r")"
    + r"|(?P<ru_prompt_extract>"
    + r"(?:раскрой|раскройте|покажи|покажите|выведи|выведите|"
    + r"повтори|повторите|напечатай|озвучь|сообщи|напиши|процитируй|воспроизведи)\s{1,8}"
    + r"(?:мне\s{1,4}|дословно\s{1,4}|слово\s{1,4}в\s{1,4}слово\s{1,4})?"
    + r"(?:(?:свой|исходный|полный|точный|скрытый)\s{1,8}){0,3}"
    + r"(?:системный\s{1,4}промпт|системные\s{1,4}инструкции|"
    + r"скрытые\s{1,4}инструкции|системное\s{1,4}сообщение)"
    + r"|(?:дословно\s{1,4})?(?:процитируй|воспроизведи|напиши\s{1,4}слово\s{1,4}в\s{1,4}слово)"
    + r"[^.!?\r\n]{1,90}(?:до\s{1,4}моего\s{1,4}сообщения|в\s{1,4}блоке\s{1,4}разработчика|"
    + r"в\s{1,4}самом\s{1,4}начале\s{1,4}(?:нашего\s{1,4})?диалога)"
    + r"|(?:какой|каков)\s{1,4}(?:твой|ваш)\s{1,4}(?:системный|исходный)\s{1,4}промпт"
    + r")"
    + r"|(?P<ru_jailbreak_mode>"
    + r"в\s{1,4}режиме\s{1,4}разработчика(?:\s{1,4}без\s{1,4}ограничений)?"
    + r"|(?:включи|перейди\s{1,4}в|активируй)\s{1,4}режим\s{1,4}"
    + r"(?:разработчика|бога|без\s{1,4}ограничений|без\s{1,4}цензуры)"
    + r"|(?:отвечай|работай|действуй|пиши)\s{1,4}без\s{1,4}"
    + r"(?:каких-либо\s{1,4}|всяких\s{1,4})?(?:ограничений|правил|фильтров|цензуры|запретов)"
    + r"|нет\s{1,4}(?:никаких\s{1,4})?(?:этических|моральных|защитных)\s{1,4}"
    + r"(?:рамок|ограничений|фильтров)"
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

_TOKENIZER_DOC_CTX_RE = re.compile(
    r"(?i)(?:токенизатор\w*|tokenizer\w*|stop_token|eos_token|bos_token|pad_token|"
    r"документаци\w*|шаблон\w*\s+чата|chat\s+template|системн\w+\s+тег\w*|special\s+tokens?)",
)
_DELIMITER_ATTACK_CUE_RE = re.compile(
    r"(?i)(?:\b(?:ignore|override|reveal|secrets?|evil|root|admin|new\s+instructions?|act\s+as)\b"
    r"|(?:\[inst\][^\r\n]{1,120}\[/inst\])"
    r"|(?:<\|im_start\|>\s*system)"
    r"|(?:игнорир|забудь|раскрой|секрет|парол))",
)

# ---------------------------------------------------------------------------
# Linear-time indirect prompt injection & tool-hijacking patterns
# ---------------------------------------------------------------------------

_MD_COMMENT_RE = re.compile(
    r"(?m)^[ \t]{0,3}\[(?://|comment)\]:[ \t]{1,4}(?:#|<>)[ \t]{1,4}(?P<body>[^\r\n]{1,1024})"
)

_ENV_EXAMPLE_NEG = r"(?![.\-](?:example|sample|template|defaults|dist|schema|bak|md|txt)\b)"

_INDIRECT_ASCII_RE = re.compile(
    r"(?P<sql_tool_hijack>(?i:\b(?:DROP\s{1,4}(?:TABLE|DATABASE|SCHEMA)|TRUNCATE\s{1,4}TABLE|"
    r"DELETE\s{1,4}FROM\s{1,4}[A-Za-z0-9_.]+\s{1,4}WHERE\s{1,4}(?:1\s*=\s*1|true))\b))"
    rf"|(?P<shell_read_hijack>(?i:\b(?:cat|head|tail|less|more|type|Get-Content|base64|xxd)\s{{1,4}}"
    rf"(?:~?/(?:\.ssh|\.aws|\.gnupg)(?:/[^\s;|&]{{0,64}})?|[^\s;|&]{{0,64}}\.env\b{_ENV_EXAMPLE_NEG}|"
    r"/etc/(?:passwd|shadow)|[^\s;|&]{0,64}id_(?:rsa|ed25519))))"
    r"|(?P<curl_pipe_shell>(?i:\b(?:curl|wget)\s{1,4}[^\r\n|]{1,256}\|\s{0,4}(?:ba)?sh\b))"
    r"|(?P<rm_rf_hijack>(?i:\brm\s{1,4}-rf\s{1,4}(?:/|~)))"
    r"|(?P<exfiltrate_en>(?i:\b(?:exfiltrate(?:\s{1,4}(?:[^\s\r\n]{1,32}\s{1,4}){0,4}"
    r"(?:data|secrets?|keys?|tokens?|files?|credentials?))?|"
    r"(?:send|upload|post|forward|transmit|leak)\s{1,4}(?:all\s{1,4}|the\s{1,4})?(?:user(?:'s)?\s{1,4})?"
    r"(?:contents?|data|secrets?|credentials?|keys?|tokens?|files?|chat\s{1,4}history|\.env)\s{1,4}"
    r"(?:to|via)\s{1,4}(?:https?://\S{1,256}|\S{1,64}@\S{1,64}|[a-z0-9.-]+\.[a-z]{2,12}|external|attacker))))"
    r"|(?P<tool_hijack_en>(?i:\b(?:(?:important\s{1,4}for\s{1,4}(?:ai\s{1,4})?assistant|ai\s{1,4}assistant\s*:)"
    r"[^\r\n]{1,120}\b(?:call|send|forward|upload|execute|append)\b"
    r"|call\s{1,4}(?:run_python|execute_command|bash|shell_exec|send_email)\b[^\r\n]{1,120}"
    r"\b(?:os\.system|subprocess|whoami|/bin/sh|@)\b"
    r"|<instructions>[^\r\n]{4,400}</instructions>)))",
    re.ASCII,
)

_SQL_HIJACK_LOWER_RE = re.compile(
    r"\b(?:drop\s{1,4}(?:table|database|schema)|truncate\s{1,4}table|"
    r"delete\s{1,4}from\s{1,4}[a-z0-9_.]+\s{1,4}where\s{1,4}(?:1\s*=\s*1|true))\b",
    re.ASCII,
)
_SHELL_READ_LOWER_RE = re.compile(
    rf"\b(?:cat|head|tail|less|more|type|get-content|base64|xxd)\s{{1,4}}"
    rf"(?:~?/(?:\.ssh|\.aws|\.gnupg)(?:/[^\s;|&]{{0,64}})?|[^\s;|&]{{0,64}}\.env\b{_ENV_EXAMPLE_NEG}|"
    r"/etc/(?:passwd|shadow)|[^\s;|&]{0,64}id_(?:rsa|ed25519))",
    re.ASCII,
)
_CURL_PIPE_LOWER_RE = re.compile(
    r"\b(?:curl|wget)\s{1,4}[^\r\n|]{1,256}\|\s{0,4}(?:ba)?sh\b",
    re.ASCII,
)
_RM_RF_LOWER_RE = re.compile(r"\brm\s{1,4}-rf\s{1,4}(?:/|~)", re.ASCII)
_EXFIL_EN_LOWER_RE = re.compile(
    r"\b(?:exfiltrate(?:\s{1,4}(?>[^\s\r\n]{1,32}\s{1,4}){0,4}"
    r"(?:data|secrets?|keys?|tokens?|files?|credentials?))?|"
    r"(?:send|upload|post|forward|transmit|leak)\s{1,4}(?:all\s{1,4}|the\s{1,4})?(?:user(?:'s)?\s{1,4})?"
    r"(?:contents?|data|secrets?|credentials?|keys?|tokens?|files?|chat\s{1,4}history|\.env)\s{1,4}"
    r"(?:to|via)\s{1,4}(?:https?://\S{1,256}|\S{1,64}@\S{1,64}|[a-z0-9.-]+\.[a-z]{2,12}|external|attacker))",
    re.ASCII,
)
_TOOL_HIJACK_EN_LOWER_RE = re.compile(
    r"\b(?:important\s{1,4}for\s{1,4}(?:ai\s{1,4})?assistant|ai\s{1,4}assistant\s*:)"
    r"[^\r\n]{1,120}\b(?:call|send|forward|upload|execute|append)\b"
    r"|\bcall\s{1,4}(?:run_python|execute_command|bash|shell_exec|send_email)\b[^\r\n]{1,120}"
    r"(?:os\.system|subprocess|whoami|/bin/sh|@)"
    r"|<instructions>[^\r\n]{4,400}</instructions>",
    re.ASCII,
)
_CSS_HIDDEN_INJECTION_RE = re.compile(
    r"(?i)<(?:div|span|p|a|section)\b[^>\r\n]{1,256}"
    r"(?:font-size\s*:\s*0|color\s*:\s*transparent|display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0)"
    r"[^>\r\n]*>[^<]{1,512}</(?:div|span|p|a|section)>",
)

_SQL_EDUCATIONAL_QUESTION_RE = re.compile(
    r"(?i)(?:чем\s+отличается|в\s+ч[её]м\s+разница|что\s+делает|как\s+работает|"
    r"what\s+is\s+the\s+difference|difference\s+between|explain\s+how)",
)
_SQL_IMPERATIVE_RE = re.compile(r"(?i)(?:выполни|запусти|\b(?:execute|run)\b|;)")
_NEGATED_WARNING_RE = re.compile(
    r"(?i)(?:никогда\s+не\s+(?:запускай\w*|выполняй\w*|используй\w*|вводи\w*)"
    r"|не\s+(?:запускайте|выполняйте)"
    r"|запрещено\s+(?:запускать|выполнять)"
    r"|\b(?:never|do\s+not|don't|avoid)\s+(?:run(?:ning)?|execut(?:e|ing)|us(?:e|ing))\b)",
)

_SHELL_READ_MARKERS = (".ssh", ".aws", ".gnupg", ".env", "/etc/", "id_rsa", "id_ed25519")
_EXFIL_SEND_VERBS = ("send", "upload", "post", "forward", "transmit", "leak")

_SQL_HIJACK_RU_RE = re.compile(
    r"(?i)"
    + _RU_BOUND_L
    + r"выполни\s{1,8}(?:DROP\s{1,4}(?:TABLE|DATABASE|SCHEMA)|"
    + r"TRUNCATE\s{1,4}TABLE|DELETE\s{1,4}FROM)\b"
)
_EXFIL_RU_RE = re.compile(
    r"(?i)"
    + _RU_BOUND_L
    + r"(?:(?:скопируй|собери)\s{1,4}(?:все\s{1,4})?"
    + r"(?:содержимое|данные|файлы|секреты|ключи|пароли|токены|переписку)\s{1,4}и\s{1,4})?"
    + r"(?:отправь|перешли|передай|слей|выгрузи|вышли|отошли)\s{1,4}"
    + r"(?:(?:все\s{1,4})?"
    + r"(?:содержимое|данные|файлы|секреты|ключи|пароли|токены|переписку)\s{1,4})?"
    + r"(?:[^\s\r\n]{1,32}\s{1,4}){0,4}(?:на|в|по(?:\s{1,4}адресу)?)"
    + _RU_BOUND_R
    + r"(?:\s{1,4}\S{1,256})?"
)

_COMMENT_STRONG_DIRECTIVE_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:ignore|disregard|forget|override|system\s+prompt|new\s+instructions?|jailbreak|"
    r"drop\s+table|truncate\s+table|delete\s+from|exfiltrate|steal|"
    r"curl\s+|wget\s+|~/.ssh|\.env\b|/etc/passwd|send\s+to|upload\s+to|post\s+to)\b"
    r"|!\["
    r"|<img\b"
    r"|(?:игнорируй|проигнорируй|забудь|отправь|перешли|слей|выгрузи|"
    r"системн\w+\s+(?:промпт|инструкц))"
    r")"
)
_COMMENT_WEAK_NOUN_RE = re.compile(
    r"(?i)(?:\b(?:assistant|instructions?|secret|password|token)\b"
    r"|(?:инструкц|секрет|пароль|токен))"
)
_COMMENT_IMPERATIVE_CUE_RE = re.compile(
    r"(?i)(?:\b(?:execute|run|call|leak|reveal|print|output|dump|send|upload|post|forward|read|bash)\b"
    r"|https?://"
    r"|(?:выполни|раскрой|покажи|выведи|передай|прочитай|удали|укради|ассистент\s*:))"
)
_MAX_COMMENT_SCAN_SLICE = 8_192

# ---------------------------------------------------------------------------
# EchoLeak Markdown & HTML image exfiltration patterns (CVE-2025-32711)
# ---------------------------------------------------------------------------

_URL_SCHEME_PART = r"(?:https?(?::|&#0*58;|&#[xX]0*3[aA];)//|//)"

_INLINE_MD_IMG_RE = re.compile(
    r"!\[(?P<alt>[^\[\]]{0,4096})\]\s{0,16}"
    rf"\(\s{{0,8}}<?(?P<url>{_URL_SCHEME_PART}[^\s>)\r\n]{{1,1024}})>?"
    r"(?:[ \t]{1,8}(?:\"[^\"\r\n]{0,200}\"|'[^'\r\n]{0,200}'))?\s{0,8}\)",
)
_MD_CLICKABLE_LINK_RE = re.compile(
    r"(?<!!)\[(?P<label>[^\[\]]{0,512})\]\s{0,8}"
    rf"\(\s{{0,8}}<?(?P<url>{_URL_SCHEME_PART}[^\s>)\r\n]{{1,1024}})>?\s{{0,8}}\)",
)
_REF_DEF_RE = re.compile(
    r"(?m)^[ \t]{0,3}\[(?P<id>[^\]\r\n]{1,100})\]:[ \t]{0,8}"
    rf"<?(?P<url>{_URL_SCHEME_PART}[^\s>\r\n]{{1,1024}})>?",
)
_REF_IMG_FULL_RE = re.compile(
    r"!\[(?P<alt>[^\[\]]{0,1024})\][ \t]?\[(?P<id>[^\]\r\n]{0,100})\]",
)
_REF_IMG_SHORTCUT_RE = re.compile(r"!\[(?P<id>[^\]\r\n]{1,100})\](?![ \t]?[(\[])")
_HTML_IMG_TAG_RE = re.compile(r"<img\b(?P<attrs>[^>]{1,2048})>", re.IGNORECASE)
_HTML_IMG_SRC_RE = re.compile(
    r"""\bsrc\s{0,4}=\s{0,4}(?:"(?P<dq>(?:https?:)?//[^"\r\n]{1,1024})"|'(?P<sq>(?:https?:)?//[^'\r\n]{1,1024})'|(?P<uq>(?:https?:)?//[^\s>"']{1,1024}))""",
    re.IGNORECASE,
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
_TEMPLATE_HINT_RE = re.compile(r"(?:\{\{|\}\}|\$\{|%7b|%7d|<pkr:)", re.ASCII | re.IGNORECASE)
_IMAGE_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|svg|webp|ico|avif|bmp)$", re.ASCII | re.IGNORECASE)
_BASE64_SEG_RE = re.compile(r"^[A-Za-z0-9+/_-]{16,}={0,2}$", re.ASCII)
_HEX_SEG_RE = re.compile(r"^[0-9a-fA-F]{16,}$", re.ASCII)
_KEBAB_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$", re.ASCII)
_BENIGN_QUERY_VAL_RE = re.compile(r"^[A-Za-z0-9._x%-]{1,16}$", re.ASCII)
_BENIGN_IMG_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "auto",
        "crop",
        "dpr",
        "fit",
        "fm",
        "format",
        "h",
        "height",
        "quality",
        "raw",
        "scale",
        "size",
        "theme",
        "v",
        "ver",
        "version",
        "w",
        "width",
    }
)
_SUBDOMAIN_SECRET_HINTS = (
    "sk-",
    "ghp_",
    "gho_",
    "glpat-",
    "akia",
    "aqvn",
    "hf_",
    "xoxb-",
    "eyj",
    "secret",
    "token",
    "password",
    "stolen",
    "exfil",
    "leak",
)

_DIRECT_CONFIDENCE: dict[str, float] = {
    "en_instruction_override": 0.96,
    "en_passive_override": 0.95,
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
    "tool_hijack_en": 0.96,
    "css_hidden_injection": 0.95,
}


@dataclass(frozen=True, slots=True)
class _FindingSpan:
    start: int
    end: int
    category: EntityCategory
    confidence: float
    rule_id: str


def _has_comment_directive(body: str) -> bool:
    if _COMMENT_STRONG_DIRECTIVE_RE.search(body):
        return True
    return bool(_COMMENT_WEAK_NOUN_RE.search(body) and _COMMENT_IMPERATIVE_CUE_RE.search(body))


def _looks_like_base64_segment(segment: str) -> bool:
    if _IMAGE_EXT_RE.search(segment) or not _BASE64_SEG_RE.match(segment):
        return False
    if _KEBAB_SLUG_RE.match(segment):
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


def _is_benign_static_image_query(query: str) -> bool:
    """Return True if query contains ONLY benign static sizing/versioning parameters."""
    if not query:
        return True
    pairs = query.split("&")
    if len(pairs) > 5:
        return False
    for pair in pairs:
        if "=" not in pair:
            return False
        k, _, v = pair.partition("=")
        if k.lower() not in _BENIGN_IMG_QUERY_KEYS:
            return False
        if not _BENIGN_QUERY_VAL_RE.match(v):
            return False
    return True


def _is_suspicious_exfil_url(raw_url: str) -> bool:
    """Return True if an image URL carries exfil queries, templates, subdomains, or encodings."""
    url = html.unescape(raw_url).strip() if "&" in raw_url else raw_url.strip()
    if _TEMPLATE_HINT_RE.search(url) or _EXFIL_PARAM_HINT_RE.search(url):
        return True

    parse_target = f"https:{url}" if url.startswith("//") else url
    parsed = urlsplit(parse_target)

    if parsed.query and not _is_benign_static_image_query(parsed.query):
        return True

    if parsed.fragment and _EXFIL_PARAM_HINT_RE.search(f"?{parsed.fragment}"):
        return True

    hostname = (parsed.hostname or "").lower()
    if hostname:
        host_labels = hostname.split(".")
        subdomains = host_labels[:-2] if len(host_labels) > 2 else ()
        for sub in subdomains:
            if (
                len(sub) >= 20
                or _HEX_SEG_RE.match(sub)
                or any(hint in sub for hint in _SUBDOMAIN_SECRET_HINTS)
            ):
                return True

    for part in parsed.path.split("/"):
        if not part:
            continue
        stem = _IMAGE_EXT_RE.sub("", part)
        if _HEX_SEG_RE.match(stem) or _looks_like_base64_segment(part):
            return True

    return False


def _merge_sorted_intervals(spans: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    if not spans:
        return ([], [])
    ordered = sorted(spans)
    m_starts: list[int] = [ordered[0][0]]
    m_ends: list[int] = [ordered[0][1]]
    for s, e in ordered[1:]:
        if s <= m_ends[-1]:
            if e > m_ends[-1]:
                m_ends[-1] = e
        else:
            m_starts.append(s)
            m_ends.append(e)
    return (m_starts, m_ends)


def _is_contained_in_merged(start: int, end: int, c_starts: list[int], c_ends: list[int]) -> bool:
    if not c_starts:
        return False
    idx = bisect_right(c_starts, start)
    return idx > 0 and end <= c_ends[idx - 1]


def _fold_leetspeak_if_triggered(text: str) -> str | None:
    """Fold leetspeak tokens (1:1 length) when >= 2 alphanumeric leet tokens exist."""
    if _LEET_DIGIT_TRIGGER_RE.search(text) is None:
        return None
    matches = list(_LEET_TOKEN_RE.finditer(text))
    if len(matches) < 2:
        return None
    parts: list[str] = []
    cursor = 0
    for match in matches:
        s, e = match.span()
        if s > cursor:
            parts.append(text[cursor:s])
        parts.append(match.group(0).translate(_LEET_TRANS))
        cursor = e
    if cursor < len(text):
        parts.append(text[cursor:])
    return "".join(parts)


def _collapse_spaced_letters(text: str) -> tuple[str, list[int]] | None:
    """Collapse ``I g n o r e   a l l`` runs with exact character back-mapping."""
    if " " not in text and "\t" not in text:
        return None
    runs = list(_SPACED_RUN_RE.finditer(text))
    if not runs:
        return None

    joined_parts: list[str] = []
    idx_map: list[int] = []
    cursor = 0
    for run in runs:
        r_start, r_end = run.span()
        if r_start > cursor:
            segment = text[cursor:r_start]
            s_pos = 0
            for ws_m in _MULTI_SPACE_RE.finditer(segment):
                ws_s, ws_e = ws_m.span()
                if ws_s > s_pos:
                    joined_parts.append(segment[s_pos:ws_s])
                    idx_map.extend(range(cursor + s_pos, cursor + ws_s))
                joined_parts.append(" ")
                idx_map.append(cursor + ws_s)
                s_pos = ws_e
            if s_pos < len(segment):
                joined_parts.append(segment[s_pos:])
                idx_map.extend(range(cursor + s_pos, cursor + len(segment)))
        for idx in range(r_start, r_end):
            ch = text[idx]
            if ch not in (" ", "\t"):
                joined_parts.append(ch)
                idx_map.append(idx)
        cursor = r_end

    if cursor < len(text):
        tail = text[cursor:]
        s_pos = 0
        for ws_m in _MULTI_SPACE_RE.finditer(tail):
            ws_s, ws_e = ws_m.span()
            if ws_s > s_pos:
                joined_parts.append(tail[s_pos:ws_s])
                idx_map.extend(range(cursor + s_pos, cursor + ws_s))
            joined_parts.append(" ")
            idx_map.append(cursor + ws_s)
            s_pos = ws_e
        if s_pos < len(tail):
            joined_parts.append(tail[s_pos:])
            idx_map.extend(range(cursor + s_pos, len(text)))

    return ("".join(joined_parts), idx_map)


def _deduplicate_findings(candidates: list[_FindingSpan]) -> list[_FindingSpan]:
    """Deduplicate overlapping same-category spans in O(M log M) and drop comment subspans."""
    if len(candidates) <= 1:
        return list(candidates)

    raw_comments = [
        (item.start, item.end) for item in candidates if item.rule_id == "hidden_html_comment"
    ]
    c_starts, c_ends = _merge_sorted_intervals(raw_comments)

    filtered: list[_FindingSpan] = []
    for item in candidates:
        if (
            c_starts
            and item.rule_id != "hidden_html_comment"
            and item.category
            in (
                EntityCategory.PROMPT_INJECTION,
                EntityCategory.INDIRECT_INJECTION,
            )
            and _is_contained_in_merged(item.start, item.end, c_starts, c_ends)
        ):
            continue
        filtered.append(item)

    ranked = sorted(
        filtered,
        key=lambda item: (-item.confidence, -(item.end - item.start), item.start),
    )
    starts_by_cat: dict[EntityCategory, list[int]] = {}
    ends_by_cat: dict[EntityCategory, list[int]] = {}
    accepted: list[_FindingSpan] = []

    for item in ranked:
        starts = starts_by_cat.setdefault(item.category, [])
        ends = ends_by_cat.setdefault(item.category, [])
        idx = bisect_right(starts, item.start)
        if idx > 0 and ends[idx - 1] > item.start:
            continue
        if idx < len(starts) and starts[idx] < item.end:
            continue
        starts.insert(idx, item.start)
        ends.insert(idx, item.end)
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
        _ = raw_text
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

        leet_text = _fold_leetspeak_if_triggered(text)
        if leet_text is not None:
            leet_lower = leet_text.lower()
            leet_use_lower = len(leet_lower) == len(leet_text)
            leet_cands: list[_FindingSpan] = []
            self._collect_direct_rules(
                leet_text, leet_lower, leet_use_lower, has_cyrillic, leet_cands
            )
            self._collect_indirect_rules(
                leet_text, leet_lower, leet_use_lower, has_cyrillic, leet_cands
            )
            for lc in leet_cands:
                if text[lc.start : lc.end] != leet_text[lc.start : lc.end]:
                    candidates.append(lc)

        spaced_collapsed = _collapse_spaced_letters(text)
        if spaced_collapsed is not None:
            sp_text, sp_map = spaced_collapsed
            sp_lower = sp_text.lower()
            sp_use_lower = len(sp_lower) == len(sp_text)
            sp_cands: list[_FindingSpan] = []
            self._collect_direct_rules(sp_text, sp_lower, sp_use_lower, has_cyrillic, sp_cands)
            self._collect_indirect_rules(sp_text, sp_lower, sp_use_lower, has_cyrillic, sp_cands)
            map_len = len(sp_map)
            for sc in sp_cands:
                if map_len > 0 and sc.end > sc.start >= 0:
                    raw_s = sp_map[min(sc.start, map_len - 1)]
                    raw_e = sp_map[min(sc.end - 1, map_len - 1)] + 1
                    candidates.append(
                        _FindingSpan(
                            start=raw_s,
                            end=raw_e,
                            category=sc.category,
                            confidence=sc.confidence,
                            rule_id=sc.rule_id,
                        )
                    )

        if not candidates:
            return []

        selected = _deduplicate_findings(candidates)
        evasion_detected = bool(segment.evasion_techniques)
        direct_span = not segment.index_map and segment.span_source == SpanSource.DIRECT

        detections: list[Detection] = []
        for item in selected:
            matched = text[item.start : item.end]
            if direct_span:
                mapped_start, mapped_end = item.start, item.end
            else:
                mapped_start, mapped_end = segment.map_span(item.start, item.end)
            det = Detection.__new__(Detection)
            object.__setattr__(
                det,
                "__dict__",
                {
                    "category": item.category,
                    "start": mapped_start,
                    "end": mapped_end,
                    "confidence": item.confidence,
                    "detector": self.name,
                    "span_source": segment.span_source,
                    "evasion_detected": evasion_detected,
                    "evasion_techniques": segment.evasion_techniques,
                    "compliance_refs": _COMPLIANCE_CACHE[item.category],
                    "masked_preview": f"[{item.category.value}:len={len(matched)}]",
                    "value_hmac": self._hmac_fn(matched) if self._hmac_fn else "",
                    "source_message_indices": (),
                    "metadata": {"rule": item.rule_id},
                },
            )
            object.__setattr__(det, "__pydantic_fields_set__", set(_DETECTION_FIELDS_SET))
            object.__setattr__(det, "__pydantic_extra__", None)
            object.__setattr__(det, "__pydantic_private__", None)
            detections.append(det)
        return detections

    def _is_defensive_mention_only(self, text: str, start: int, end: int) -> bool:
        before = text[max(0, start - 96) : start]
        after = text[end : min(len(text), end + 24)].strip()
        return bool(
            _DEFENSIVE_EDUCATIONAL_PREFIX_RE.search(before) and after in {"", ".", "?", "!"}
        )

    def _is_dan_with_jailbreak_ctx(self, text: str, start: int, end: int) -> bool:
        matched = text[start:end]
        if matched.lower().startswith("do "):
            return True
        win = text[max(0, start - 40) : min(len(text), end + 40)]
        return _DAN_JAILBREAK_CTX_RE.search(win) is not None

    def _is_benign_tokenizer_doc_delimiter(self, text: str, start: int, end: int) -> bool:
        win = text[max(0, start - 96) : min(len(text), end + 96)]
        if not _TOKENIZER_DOC_CTX_RE.search(win):
            return False
        return _DELIMITER_ATTACK_CUE_RE.search(win) is None

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
                if rule_id == "en_dan_jailbreak" and not self._is_dan_with_jailbreak_ctx(
                    text, match.start(), match.end()
                ):
                    continue
                if rule_id == "en_instruction_override" and self._is_defensive_mention_only(
                    text, match.start(), match.end()
                ):
                    continue
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
            has_any_override_verb = any(v in lower_text for v in _EN_OVERRIDE_VERBS)
            has_any_override_noun = any(n in lower_text for n in _EN_OVERRIDE_NOUNS)

            if has_any_override_verb and has_any_override_noun:
                for match in _EN_OVERRIDE_LOWER_RE.finditer(lower_text):
                    if self._is_defensive_mention_only(text, match.start(), match.end()):
                        continue
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_instruction_override"],
                            rule_id="en_instruction_override",
                        )
                    )
                for match in _EN_SENTENCE_INVERTED_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=0.93,
                            rule_id="en_instruction_override",
                        )
                    )
                if "=" in lower_text:
                    for match in _EN_SPLIT_VARS_LOWER_RE.finditer(lower_text):
                        out.append(
                            _FindingSpan(
                                start=match.start(),
                                end=match.end(),
                                category=EntityCategory.PROMPT_INJECTION,
                                confidence=0.94,
                                rule_id="en_instruction_override",
                            )
                        )

            if has_any_override_noun and (
                "are " in lower_text or "is " in lower_text or "been " in lower_text
            ):
                for match in _EN_PASSIVE_OVERRIDE_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=_DIRECT_CONFIDENCE["en_passive_override"],
                            rule_id="en_passive_override",
                        )
                    )

            if has_any_override_verb and (
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

            has_extract_target = (
                "prompt" in lower_text
                or "instructions" in lower_text
                or "above this" in lower_text
                or "before this" in lower_text
            )
            if has_extract_target and any(v in lower_text for v in _EN_EXTRACT_VERBS):
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
            if ("prompt" in lower_text or "instructions" in lower_text) and "what" in lower_text:
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
                    if self._is_dan_with_jailbreak_ctx(text, match.start(), match.end()):
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
                        if self._is_benign_tokenizer_doc_delimiter(
                            text, match.start(), match.end()
                        ):
                            continue
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
                    if self._is_benign_tokenizer_doc_delimiter(text, match.start(), match.end()):
                        continue
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.PROMPT_INJECTION,
                            confidence=0.97,
                            rule_id="delimiter_attack",
                        )
                    )

    def _is_sql_educational_question(self, text: str, start: int, end: int) -> bool:
        win = text[max(0, start - 80) : min(len(text), end + 80)]
        return bool(_SQL_EDUCATIONAL_QUESTION_RE.search(win) and not _SQL_IMPERATIVE_RE.search(win))

    def _is_negated_safety_warning(self, text: str, start: int) -> bool:
        before = text[max(0, start - 96) : start]
        return _NEGATED_WARNING_RE.search(before) is not None

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
                if rule_id == "sql_tool_hijack" and self._is_sql_educational_question(
                    text, match.start(), match.end()
                ):
                    continue
                if rule_id in {
                    "rm_rf_hijack",
                    "curl_pipe_shell",
                } and self._is_negated_safety_warning(text, match.start()):
                    continue
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
            if (
                "table" in lower_text
                or "database" in lower_text
                or "schema" in lower_text
                or "delete" in lower_text
            ):
                for match in _SQL_HIJACK_LOWER_RE.finditer(lower_text):
                    if self._is_sql_educational_question(text, match.start(), match.end()):
                        continue
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
                    if self._is_negated_safety_warning(text, match.start()):
                        continue
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
                    if self._is_negated_safety_warning(text, match.start()):
                        continue
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
                    if self._is_negated_safety_warning(text, match.start()):
                        continue
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
            if "assistant" in lower_text or "call " in lower_text or "<instructions>" in lower_text:
                for match in _TOOL_HIJACK_EN_LOWER_RE.finditer(lower_text):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.INDIRECT_INJECTION,
                            confidence=_INDIRECT_CONFIDENCE["tool_hijack_en"],
                            rule_id="tool_hijack_en",
                        )
                    )

        if "style=" in lower_text and ("<" in text and "</" in text):
            for match in _CSS_HIDDEN_INJECTION_RE.finditer(text):
                if _has_comment_directive(match.group(0)):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.INDIRECT_INJECTION,
                            confidence=_INDIRECT_CONFIDENCE["css_hidden_injection"],
                            rule_id="css_hidden_injection",
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
                end_marker = text.find("-->", start + 4)
                if end_marker == -1:
                    break
                body_start = start + 4
                body_len = end_marker - body_start
                if body_len <= _MAX_COMMENT_SCAN_SLICE * 2:
                    body_sample = text[body_start:end_marker]
                else:
                    # Scan head + tail of oversized (>16 KB) HTML comments in O(1)
                    head = text[body_start : body_start + _MAX_COMMENT_SCAN_SLICE]
                    tail = text[end_marker - _MAX_COMMENT_SCAN_SLICE : end_marker]
                    body_sample = f"{head}\n{tail}"

                if _has_comment_directive(body_sample):
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
                if _has_comment_directive(match.group("body")):
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
            if "]" in text and "(" in text and ")" in text:
                for match in _INLINE_MD_IMG_RE.finditer(text):
                    raw_url = match.group("url")
                    unescaped_url = html.unescape(raw_url) if "&" in raw_url else raw_url
                    if _is_suspicious_exfil_url(unescaped_url) or _TRACKING_URL_HINT_RE.search(
                        unescaped_url
                    ):
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
                    raw_url = match.group("url")
                    ref_defs[ref_id] = (
                        match.start(),
                        match.end(),
                        html.unescape(raw_url) if "&" in raw_url else raw_url,
                    )

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

        if (
            "[" in text
            and ")" in text
            and ("{{" in text or "${" in text or "%7b" in lower_text or "<pkr:" in lower_text)
        ):
            for match in _MD_CLICKABLE_LINK_RE.finditer(text):
                raw_url = match.group("url")
                unescaped_url = html.unescape(raw_url) if "&" in raw_url else raw_url
                if _TEMPLATE_HINT_RE.search(unescaped_url):
                    out.append(
                        _FindingSpan(
                            start=match.start(),
                            end=match.end(),
                            category=EntityCategory.MARKDOWN_EXFIL,
                            confidence=0.96,
                            rule_id="echoleak_markdown_link_template",
                        )
                    )

        if "<img" in lower_text and ">" in text:
            for match in _HTML_IMG_TAG_RE.finditer(text):
                raw_attrs = match.group("attrs")
                attrs = html.unescape(raw_attrs) if "&" in raw_attrs else raw_attrs
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
        _ = raw_text
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
