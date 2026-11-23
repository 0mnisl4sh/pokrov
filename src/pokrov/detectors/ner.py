"""Layer-2 Russian NER detector with optional Natasha backend and fast pure-Python fallback."""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Callable
from typing import Any

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.contracts import (
    Detection,
    EntityCategory,
    NormalizedSegment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure-Python Russian NER heuristics (FIO, Addresses, Organizations)
# ---------------------------------------------------------------------------

_SURNAME_INITIALS_RE = re.compile(
    r"(?<![А-ЯЁа-яёA-Za-z])"
    r"([А-ЯЁ][а-яё]{1,30}(?:-[А-ЯЁ][а-яё]{1,30})?)"
    r"[ ]+([А-ЯЁ]\.[ ]?[А-ЯЁ]\.)"
    r"(?![А-ЯЁа-яё.])",
)

_INITIALS_SURNAME_RE = re.compile(
    r"(?<![А-ЯЁа-яёA-Za-z])"
    r"([А-ЯЁ]\.[ ]?[А-ЯЁ]\.)"
    r"[ ]?([А-ЯЁ][а-яё]{1,30}(?:-[А-ЯЁ][а-яё]{1,30})?)"
    r"(?![А-ЯЁа-яё])",
)

_CAP_WORD_PART = r"[А-ЯЁ][а-яё]{1,28}(?:-[А-ЯЁ][а-яё]{1,28})?"
_CAP_RUN_RE = re.compile(
    rf"(?<![А-ЯЁа-яёA-Za-z]){_CAP_WORD_PART}(?:[ ]+{_CAP_WORD_PART})+(?![А-ЯЁа-яё])",
)
_CAP_TOKEN_RE = re.compile(_CAP_WORD_PART)

_PATRONYMIC_SUFFIXES = (
    "ович",
    "евич",
    "ич",
    "овича",
    "евича",
    "ича",
    "овичу",
    "евичу",
    "ичу",
    "овичем",
    "евичем",
    "ичем",
    "овиче",
    "евиче",
    "иче",
    "овна",
    "евна",
    "ична",
    "инична",
    "овны",
    "евны",
    "ичны",
    "иничны",
    "овне",
    "евне",
    "ичне",
    "иничне",
    "овну",
    "евну",
    "ичну",
    "иничну",
    "овной",
    "евной",
    "ичной",
    "иничной",
    "овною",
    "евною",
    "ичною",
)

_SURNAME_SUFFIXES = (
    "ов",
    "ова",
    "ову",
    "овым",
    "овой",
    "ове",
    "ев",
    "ева",
    "еву",
    "евым",
    "евой",
    "еве",
    "ёв",
    "ёва",
    "ёву",
    "ёвым",
    "ёвой",
    "ёве",
    "ин",
    "ина",
    "ину",
    "иным",
    "иной",
    "ине",
    "ын",
    "ына",
    "ыну",
    "ыным",
    "ыной",
    "ыне",
    "ский",
    "ская",
    "ского",
    "ской",
    "скому",
    "ским",
    "ском",
    "цкий",
    "цкая",
    "цкого",
    "цкой",
    "енко",
    "чук",
    "ук",
    "юк",
    "ских",
    "цких",
    "ых",
    "их",
    "дзе",
    "швили",
    "ян",
)

# Common Russian first names (stems/nominative + common oblique endings checked via stem)
_COMMON_FIRST_NAMES: frozenset[str] = frozenset(
    {
        "александр",
        "алексей",
        "анатолий",
        "андрей",
        "антон",
        "аркадий",
        "арсений",
        "артём",
        "артем",
        "борис",
        "вадим",
        "валентин",
        "валерий",
        "василий",
        "виктор",
        "виталий",
        "владимир",
        "владислав",
        "вячеслав",
        "геннадий",
        "георгий",
        "глеб",
        "григорий",
        "даниил",
        "данила",
        "денис",
        "дмитрий",
        "евгений",
        "егор",
        "иван",
        "игорь",
        "илья",
        "кирилл",
        "константин",
        "лев",
        "леонид",
        "максим",
        "марат",
        "матвей",
        "михаил",
        "никита",
        "николай",
        "олег",
        "павел",
        "пётр",
        "петр",
        "роман",
        "руслан",
        "семён",
        "семен",
        "сергей",
        "станислав",
        "степан",
        "тимур",
        "фёдор",
        "федор",
        "филипп",
        "эдуард",
        "юрий",
        "ярослав",
        "александра",
        "алёна",
        "алена",
        "алина",
        "алла",
        "анастасия",
        "анна",
        "валентина",
        "валерия",
        "варвара",
        "вера",
        "вероника",
        "виктория",
        "галина",
        "дарья",
        "диана",
        "евгения",
        "екатерина",
        "елена",
        "елизавета",
        "жанна",
        "зоя",
        "инна",
        "ирина",
        "карина",
        "кристина",
        "ксения",
        "лариса",
        "лидия",
        "любовь",
        "людмила",
        "маргарита",
        "марина",
        "мария",
        "надежда",
        "наталья",
        "наталия",
        "нина",
        "оксана",
        "ольга",
        "полина",
        "светлана",
        "софия",
        "софья",
        "тамара",
        "татьяна",
        "ульяна",
        "юлия",
        "яна",
    }
)

_NON_NAME_WORDS: frozenset[str] = frozenset(
    {
        "москва",
        "санкт",
        "петербург",
        "россия",
        "федерация",
        "республика",
        "область",
        "край",
        "город",
        "улица",
        "проспект",
        "переулок",
        "бульвар",
        "площадь",
        "шоссе",
        "приказ",
        "закон",
        "кодекс",
        "постановление",
        "указ",
        "договор",
        "контракт",
        "паспорт",
        "снилс",
        "полис",
        "банк",
        "сбербанк",
        "газпром",
        "яндекс",
        "ростелеком",
        "совет",
        "министерство",
        "департамент",
        "управление",
        "комитет",
        "служба",
        "центр",
        "институт",
        "университет",
        "академия",
        "театр",
        "музей",
        "вокзал",
        "новый",
        "старый",
        "большой",
        "малый",
        "нижний",
        "верхний",
        "великий",
        "новгород",
        "ростов",
        "казань",
        "самара",
        "новосибирск",
        "екатеринбург",
        "январь",
        "февраль",
        "март",
        "апрель",
        "май",
        "июнь",
        "июль",
        "август",
        "сентябрь",
        "октябрь",
        "ноябрь",
        "декабрь",
        "компания",
        "организация",
        "общество",
        "предприятие",
        "клиент",
        "сотрудник",
        "директор",
        "адрес",
        "телефон",
        "информация",
    }
)

# Address building blocks (strictly bounded quantifiers, linear-time)
_ADDR_POSTAL = r"(?:\d{6}[,]?[ ]*)"
_ADDR_REGION = (
    r"(?:(?:(?:Респ\.|Республика)[ ]+[А-ЯЁ][а-яё-]+(?:[ ]+[А-ЯЁ][а-яё-]+)?"
    r"|[А-ЯЁ][а-яё-]+(?:-[А-ЯЁа-яё]+)?[ ]+(?:обл\.|область|край|АО))"
    r"[,]?[ ]*)"
)
_ADDR_CITY = (
    r"(?:г\.|гор\.|город|пгт|пос\.|посёлок|поселок|с\.|село|дер\.|деревня|ст-ца)"
    r"[ ]*[А-ЯЁ][а-яёА-ЯЁ-]+(?:-[А-ЯЁа-яё][а-яё]+){0,2}"
)
_ADDR_STREET_NEXT_GUARD = (
    r"(?!д\.|дом\b|зд\.|здание\b|корп\.|корпус\b|к\.|стр\.|строение\b|с\."
    r"|кв\.|квартира\b|оф\.|офис\b|пом\.|помещение\b)"
)
_ADDR_STREET_PREFIX = (
    r"(?:ул\.|улица|пр\.|пр-т|пр-кт|просп\.|проспект|пер\.|переулок|б-р|бул\.|бульвар"
    r"|ш\.|шоссе|наб\.|набережная|пл\.|площадь|проезд|аллея|туп\.|тупик|линия)"
    r"[ ]*(?:им\.?[ ]*)?"
    rf"[А-ЯЁ0-9«\"][А-ЯЁа-яё0-9«»\".-]{{0,30}}"
    rf"(?:[ ]+{_ADDR_STREET_NEXT_GUARD}[А-ЯЁа-яё0-9«»\".-]{{1,30}}){{0,3}}"
)
_ADDR_STREET_SUFFIX = (
    r"[А-ЯЁ][а-яёА-ЯЁ0-9-]+(?:[ ]+[А-ЯЁа-яё0-9-]+){0,2}[ ]+"
    r"(?:ул\.|улица|пр\.|пр-т|пр-кт|просп\.|проспект|пер\.|переулок|б-р|бульвар"
    r"|ш\.|шоссе|наб\.|набережная|площадь|проезд|аллея|тупик)"
)
_ADDR_STREET = rf"(?:{_ADDR_STREET_PREFIX}|{_ADDR_STREET_SUFFIX})"
_ADDR_HOUSE = (
    r"(?:(?:д\.|дом|зд\.|здание)[ ]*\d+[А-Яа-яA-Za-z]?(?:/\d+[А-Яа-яA-Za-z]?)?"
    r"(?:[,]?[ ]*(?:корп\.|корпус|к\.|стр\.|строение|с\.|лит\.|литера)[ ]*[0-9А-Яа-яA-Za-z]+){0,3})"
)
_ADDR_APT = (
    r"(?:[,]?[ ]*(?:кв\.|квартира|оф\.|офис|пом\.|помещение|ком\.|комната)"
    r"[ ]*\d+[А-Яа-яA-Za-z]?)"
)

_ADDRESS_RE = re.compile(
    rf"(?<![А-ЯЁа-яёA-Za-z0-9])"
    rf"(?:"
    rf"{_ADDR_POSTAL}?{_ADDR_REGION}?{_ADDR_CITY}[,]?[ ]*"
    rf"(?:{_ADDR_STREET}(?:[,]?[ ]*{_ADDR_HOUSE})?{_ADDR_APT}?|{_ADDR_HOUSE}{_ADDR_APT}?)"
    rf"|{_ADDR_POSTAL}?{_ADDR_STREET}[,]?[ ]*{_ADDR_HOUSE}{_ADDR_APT}?"
    rf")"
    rf"(?![А-ЯЁа-яёA-Za-z0-9])",
)

# Organization patterns (ООО/АО/ПАО/ИП ...)
_LEGAL_CORP = r"(?:ООО|ОАО|ЗАО|ПАО|НАО|АО|НКО|АНО|ФГУП|ГУП|МУП|ФГБУ|ГБУ|МБУ|ТСЖ|ЖСК|СНТ|КФХ|ГК)"
_QUOTED_NAME = r"(?:«[^»\n]{1,80}»|\"[^\"\n]{1,80}\"|„[^“\"\n]{1,80}[“\"]|'[^'\n]{1,80}')"
_ORG_QUOTED_RE = re.compile(
    rf"(?<![А-ЯЁа-яёA-Za-z0-9])(?:{_LEGAL_CORP}|ИП)[ ]+{_QUOTED_NAME}",
)
_ORG_IP_RE = re.compile(
    r"(?<![А-ЯЁа-яёA-Za-z0-9])ИП[ ]+"
    r"([А-ЯЁ][а-яё]{1,30}(?:-[А-ЯЁ][а-яё]{1,30})?)"
    r"(?:[ ]+(?:[А-ЯЁ]\.[ ]?[А-ЯЁ]\.|[А-ЯЁ][а-яё]{1,30}(?:[ ]+[А-ЯЁ][а-яё]{1,30})?))?"
    r"(?![А-ЯЁа-яёA-Za-z])",
)
_ORG_UNQUOTED_RE = re.compile(
    rf"(?<![А-ЯЁа-яёA-Za-z0-9]){_LEGAL_CORP}[ ]+"
    r"([А-ЯЁA-Z][А-ЯЁа-яёA-Za-z0-9-]{1,30}(?:[ ]+[А-ЯЁA-Z][А-ЯЁа-яёA-Za-z0-9-]{1,30}){0,2})"
    r"(?![А-ЯЁа-яёA-Za-z0-9])",
)


def _is_patronymic(word: str) -> bool:
    low = word.lower()
    return len(low) >= 4 and low.endswith(_PATRONYMIC_SUFFIXES)


def _is_likely_surname(word: str) -> bool:
    low = word.lower()
    return len(low) >= 3 and low.endswith(_SURNAME_SUFFIXES) and low not in _NON_NAME_WORDS


def _is_common_first_name(word: str) -> bool:
    low = word.lower()
    if low in _COMMON_FIRST_NAMES:
        return True
    # Match common inflected forms by stripping a single Russian case suffix
    if len(low) >= 4 and low[-1] in {"а", "у", "е", "ы", "и", "ю", "я"}:
        stem = low[:-1]
        if (
            stem in _COMMON_FIRST_NAMES
            or f"{stem}а" in _COMMON_FIRST_NAMES
            or f"{stem}я" in _COMMON_FIRST_NAMES
        ):
            return True
    if len(low) >= 5 and low[-2:] in {"ом", "ем", "ём", "ой", "ей", "ию", "ии", "ие"}:
        stem2 = low[:-2]
        if (
            stem2 in _COMMON_FIRST_NAMES
            or f"{stem2}а" in _COMMON_FIRST_NAMES
            or f"{stem2}ий" in _COMMON_FIRST_NAMES
            or f"{stem2}ия" in _COMMON_FIRST_NAMES
            or f"{stem2}ей" in _COMMON_FIRST_NAMES
        ):
            return True
    return False


def _overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(not (end <= s or start >= e) for s, e in spans)


class RuNerDetector:
    """Layer-2 Russian NER detector with lazy optional ``natasha`` and pure-Python fallback."""

    def __init__(
        self,
        hmac_fn: Callable[[str], str] | None = None,
        *,
        use_natasha: bool = True,
    ) -> None:
        self._hmac_fn = hmac_fn
        self._use_natasha = use_natasha
        self._natasha_attempted = False
        self._natasha_components: tuple[Any, Any, Any, Any] | None = None

    @property
    def name(self) -> str:
        """Unique detector identifier."""
        return "ru_ner"

    @property
    def layer(self) -> int:
        """Pipeline execution layer (2 = NER / semantic layer)."""
        return 2

    def _ensure_natasha(self) -> tuple[Any, Any, Any, Any] | None:
        if not self._use_natasha:
            return None
        if self._natasha_attempted:
            return self._natasha_components
        self._natasha_attempted = True
        try:
            natasha_mod = importlib.import_module("natasha")
            doc_cls = natasha_mod.Doc
            segmenter = natasha_mod.Segmenter()
            emb = natasha_mod.NewsEmbedding()
            ner_tagger = natasha_mod.NewsNERTagger(emb)
            self._natasha_components = (doc_cls, segmenter, ner_tagger, natasha_mod)
        except Exception as exc:
            logger.debug("Optional natasha NER unavailable (%s); using heuristic fallback", exc)
            self._natasha_components = None
        return self._natasha_components

    def _build_detection(
        self,
        segment: NormalizedSegment,
        category: EntityCategory,
        norm_start: int,
        norm_end: int,
        confidence: float,
        source: str,
    ) -> Detection:
        raw_match = segment.text[norm_start:norm_end]
        mapped_start, mapped_end = segment.map_span(norm_start, norm_end)
        clamped_conf = min(1.0, max(0.0, round(confidence, 4)))
        return Detection(
            category=category,
            start=mapped_start,
            end=mapped_end,
            confidence=clamped_conf,
            detector=self.name,
            span_source=segment.span_source,
            evasion_detected=bool(segment.evasion_techniques),
            evasion_techniques=segment.evasion_techniques,
            compliance_refs=get_compliance_refs(category),
            masked_preview=f"[{category.value}:len={len(raw_match)}]",
            value_hmac=self._hmac_fn(raw_match) if self._hmac_fn else "",
            metadata={"ner_backend": source},
        )

    def _detect_natasha(self, text: str) -> list[tuple[EntityCategory, int, int, float, str]]:
        components = self._ensure_natasha()
        if components is None:
            return []
        doc_cls, segmenter, ner_tagger, _ = components
        findings: list[tuple[EntityCategory, int, int, float, str]] = []
        try:
            doc = doc_cls(text)
            doc.segment(segmenter)
            doc.tag_ner(ner_tagger)
            type_map = {
                "PER": EntityCategory.RU_PERSON_NAME,
                "LOC": EntityCategory.RU_ADDRESS,
                "ORG": EntityCategory.RU_ORG_NAME,
            }
            for span in getattr(doc, "spans", ()):
                cat = type_map.get(getattr(span, "type", ""))
                if cat is None:
                    continue
                start = int(span.start)
                stop = int(span.stop)
                if stop > start:
                    findings.append((cat, start, stop, 0.88, "natasha"))
        except Exception as exc:
            logger.debug("Natasha runtime NER failed (%s); falling back to heuristics", exc)
            return []
        return findings

    def _detect_heuristics(self, text: str) -> list[tuple[EntityCategory, int, int, float, str]]:
        findings: list[tuple[EntityCategory, int, int, float, str]] = []

        # 1. Addresses first (so person names inside street names are not misclassified)
        address_spans: list[tuple[int, int]] = []
        for m in _ADDRESS_RE.finditer(text):
            span = (m.start(), m.end())
            address_spans.append(span)
            snippet = m.group(0)
            has_apt = any(kw in snippet.lower() for kw in ("кв.", "квартира", "оф.", "офис"))
            conf = 0.94 if has_apt else 0.89
            findings.append((EntityCategory.RU_ADDRESS, span[0], span[1], conf, "heuristic"))

        # 2. Organizations
        org_spans: list[tuple[int, int]] = []
        for m in _ORG_QUOTED_RE.finditer(text):
            span = (m.start(), m.end())
            org_spans.append(span)
            findings.append((EntityCategory.RU_ORG_NAME, span[0], span[1], 0.92, "heuristic"))

        for m in _ORG_IP_RE.finditer(text):
            span = (m.start(), m.end())
            if _overlaps_any(span[0], span[1], org_spans):
                continue
            surname = m.group(1)
            if surname.lower() in _NON_NAME_WORDS:
                continue
            org_spans.append(span)
            findings.append((EntityCategory.RU_ORG_NAME, span[0], span[1], 0.90, "heuristic"))

        for m in _ORG_UNQUOTED_RE.finditer(text):
            span = (m.start(), m.end())
            if _overlaps_any(span[0], span[1], org_spans):
                continue
            org_spans.append(span)
            findings.append((EntityCategory.RU_ORG_NAME, span[0], span[1], 0.85, "heuristic"))

        # 3. Person names (FIO, initials + surname, surname + initials, First + Last)
        person_spans: list[tuple[int, int]] = []

        for run_m in _CAP_RUN_RE.finditer(text):
            base = run_m.start()
            tokens = [
                (tm.group(0), base + tm.start(), base + tm.end())
                for tm in _CAP_TOKEN_RE.finditer(run_m.group(0))
            ]
            claimed: set[int] = set()
            n_tok = len(tokens)

            # Pass 1: 3-word FIO ("Фамилия Имя Отчество" or "Имя Отчество Фамилия")
            idx = 0
            while idx + 3 <= n_tok:
                w1, s1, _ = tokens[idx]
                w2, _, _ = tokens[idx + 1]
                w3, _, e3 = tokens[idx + 2]
                if not any(w.lower() in _NON_NAME_WORDS for w in (w1, w2, w3)) and (
                    _is_patronymic(w3) or _is_patronymic(w2)
                ):
                    span = (s1, e3)
                    if not _overlaps_any(span[0], span[1], address_spans):
                        person_spans.append(span)
                        findings.append(
                            (EntityCategory.RU_PERSON_NAME, span[0], span[1], 0.92, "heuristic")
                        )
                        claimed.update((idx, idx + 1, idx + 2))
                        idx += 3
                        continue
                idx += 1

            # Pass 2: 2-word Name + Surname / Surname + Name on unclaimed tokens
            idx = 0
            while idx + 2 <= n_tok:
                if idx in claimed or (idx + 1) in claimed:
                    idx += 1
                    continue
                w1, s1, _ = tokens[idx]
                w2, _, e2 = tokens[idx + 1]
                if w1.lower() not in _NON_NAME_WORDS and w2.lower() not in _NON_NAME_WORDS:
                    span = (s1, e2)
                    if not (
                        _overlaps_any(span[0], span[1], address_spans)
                        or _overlaps_any(span[0], span[1], person_spans)
                        or _overlaps_any(span[0], span[1], org_spans)
                    ):
                        is_fn_sn = _is_common_first_name(w1) and _is_likely_surname(w2)
                        is_sn_fn = _is_likely_surname(w1) and _is_common_first_name(w2)
                        if is_fn_sn or is_sn_fn:
                            person_spans.append(span)
                            findings.append(
                                (
                                    EntityCategory.RU_PERSON_NAME,
                                    span[0],
                                    span[1],
                                    0.80,
                                    "heuristic",
                                )
                            )
                            claimed.update((idx, idx + 1))
                            idx += 2
                            continue
                idx += 1

        for m in _SURNAME_INITIALS_RE.finditer(text):
            surname = m.group(1)
            if surname.lower() in _NON_NAME_WORDS:
                continue
            span = (m.start(), m.end())
            if _overlaps_any(span[0], span[1], address_spans) or _overlaps_any(
                span[0], span[1], person_spans
            ):
                continue
            person_spans.append(span)
            findings.append((EntityCategory.RU_PERSON_NAME, span[0], span[1], 0.90, "heuristic"))

        for m in _INITIALS_SURNAME_RE.finditer(text):
            surname = m.group(2)
            if surname.lower() in _NON_NAME_WORDS:
                continue
            span = (m.start(), m.end())
            if _overlaps_any(span[0], span[1], address_spans) or _overlaps_any(
                span[0], span[1], person_spans
            ):
                continue
            person_spans.append(span)
            findings.append((EntityCategory.RU_PERSON_NAME, span[0], span[1], 0.90, "heuristic"))

        return findings

    def detect(self, segment: NormalizedSegment, raw_text: str) -> list[Detection]:
        """Detect Russian person names, addresses, and organization names."""
        _ = raw_text
        text = segment.text
        if not text or not text.strip():
            return []

        natasha_hits = self._detect_natasha(text)
        heuristic_hits = self._detect_heuristics(text)

        # Merge natasha + heuristic spans, keeping higher-confidence/longer non-duplicates
        merged: list[tuple[EntityCategory, int, int, float, str]] = list(natasha_hits)
        for cat, start, end, conf, src in heuristic_hits:
            overlaps_same = any(
                m_cat == cat and not (end <= m_start or start >= m_end)
                for m_cat, m_start, m_end, _, _ in merged
            )
            if not overlaps_same:
                merged.append((cat, start, end, conf, src))

        detections = [
            self._build_detection(segment, cat, start, end, conf, src)
            for cat, start, end, conf, src in merged
        ]
        detections.sort(key=lambda det: (det.start, det.end, -det.confidence))
        return detections


__all__ = ["RuNerDetector"]
