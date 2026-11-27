"""Layer-2 Russian NER detector with optional Natasha backend and fast pure-Python fallback."""

from __future__ import annotations

import importlib
import logging
import re
from bisect import bisect_right
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.contracts import (
    ComplianceRef,
    Detection,
    EntityCategory,
    NormalizedSegment,
    SpanSource,
)

logger = logging.getLogger(__name__)

_COMPLIANCE_CACHE: dict[EntityCategory, tuple[ComplianceRef, ...]] = {
    cat: get_compliance_refs(cat) for cat in EntityCategory
}
_DETECTION_FIELDS_SET: frozenset[str] = frozenset(Detection.model_fields)

# ---------------------------------------------------------------------------
# Pure-Python Russian NER heuristics (FIO, Addresses, Organizations)
# ---------------------------------------------------------------------------

_WSEP = r"[ \u00a0\u202f\u2009]"

_SURNAME_INITIALS_RE = re.compile(
    r"(?<![А-ЯЁа-яёA-Za-z])"
    r"([А-ЯЁ][а-яё]{1,30}(?:-[А-ЯЁ][а-яё]{1,30})?)"
    rf"{_WSEP}+([А-ЯЁ]\.(?:{_WSEP}?[А-ЯЁ]\.)?)"
    r"(?![А-ЯЁа-яё.])",
)

_INITIALS_SURNAME_RE = re.compile(
    r"(?<![А-ЯЁа-яёA-Za-z.])"
    rf"([А-ЯЁ]\.{_WSEP}?[А-ЯЁ]\.)"
    rf"{_WSEP}?([А-ЯЁ][а-яё]{{1,30}}(?:-[А-ЯЁ][а-яё]{{1,30}})?)"
    r"(?![А-ЯЁа-яё])",
)

_SINGLE_INITIAL_SURNAME_RE = re.compile(
    r"(?<![А-ЯЁа-яёA-Za-z.])"
    r"([А-ЯЁ]\.)"
    rf"{_WSEP}+([А-ЯЁ][а-яё]{{2,30}}(?:-[А-ЯЁ][а-яё]{{2,30}})?)"
    r"(?![А-ЯЁа-яё])",
)

_CAP_WORD_PART = r"[А-ЯЁ][а-яё]{1,28}(?:-[А-ЯЁ][а-яё]{1,28})?"
_CAP_RUN_RE = re.compile(
    rf"(?<![А-ЯЁа-яёA-Za-z]){_CAP_WORD_PART}(?:{_WSEP}+{_CAP_WORD_PART})+(?![А-ЯЁа-яё])",
)
_CAP_TOKEN_RE = re.compile(_CAP_WORD_PART)

_UPPER_WORD_PART = r"[А-ЯЁ]{2,28}(?:-[А-ЯЁ]{2,28})?"
_UPPER_RUN_RE = re.compile(
    rf"(?<![А-ЯЁа-яёA-Za-z]){_UPPER_WORD_PART}(?:{_WSEP}+{_UPPER_WORD_PART})+(?![А-ЯЁа-яё])",
)
_UPPER_TOKEN_RE = re.compile(_UPPER_WORD_PART)
_TWO_UPPER_RE = re.compile(r"[А-ЯЁ]{2}")

_LOWER_FIO_CTX_RE = re.compile(
    r"(?<![а-яёa-z])(?:клиент\w*|фио|ф\.и\.о\.|сотрудник\w*|заявител\w*|получател\w*|"
    r"плательщик\w*|гражданин\w*|пациент\w*|контактн\w+\s+лиц\w*)[ \t:—\-]+"
    r"([а-яё]{2,28}(?:-[а-яё]{2,28})?)[ ]+([а-яё]{2,28})[ ]+([а-яё]{3,28})"
    r"(?![а-яёa-z])",
)

_TURKIC_PATRONYMIC_PARTICLES: frozenset[str] = frozenset({"оглы", "кызы", "угли", "кизи"})
_TRAILING_PARTICLE_RE = re.compile(
    r"^[ \u00a0]+(?:оглы|кызы|угли|кизи)(?![А-ЯЁа-яёA-Za-z])", re.IGNORECASE
)
_ART_WORK_CTX_RE = re.compile(
    r"(?:опер[аыеу]|спектакл\w*|пьес\w*|балет\w*|роман\w*|фильм\w*|памятник\w*)[ «\"']{1,4}$",
)

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
    "оглы",
    "кызы",
    "угли",
    "кизи",
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
    "аев",
    "аева",
    "аеву",
    "аевым",
    "аевой",
    "иев",
    "иева",
    "иеву",
    "иевым",
    "уллин",
    "уллина",
    "уллину",
    "уллиным",
    "уллиной",
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
    "цкому",
    "цким",
    "ой",
    "ый",
    "ого",
    "ому",
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
    "ия",
    "ян",
    "янц",
    "заде",
    "оглы",
    "кызы",
    "угли",
)

_SHORT_EXACT_SURNAMES: frozenset[str] = frozenset({"ким", "пак", "цой", "хан", "тен", "чен", "ли"})

# Common Russian & RF/CIS first names (nominative forms)
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
        "артур",
        "альберт",
        "айдар",
        "амир",
        "аслан",
        "армен",
        "богдан",
        "борис",
        "вадим",
        "валентин",
        "валерий",
        "василий",
        "виктор",
        "виталий",
        "владимир",
        "владислав",
        "всеволод",
        "вячеслав",
        "геннадий",
        "георгий",
        "герман",
        "глеб",
        "гремислав",
        "григорий",
        "давид",
        "дамир",
        "даниил",
        "данила",
        "денис",
        "дмитрий",
        "евгений",
        "егор",
        "заур",
        "захар",
        "иван",
        "игорь",
        "ильдар",
        "илья",
        "камиль",
        "карен",
        "кирилл",
        "константин",
        "лаврентий",
        "лев",
        "леонид",
        "магомед",
        "макар",
        "максим",
        "марат",
        "марк",
        "матвей",
        "мирослав",
        "михаил",
        "мурад",
        "мурат",
        "никита",
        "николай",
        "олег",
        "павел",
        "пётр",
        "петр",
        "платон",
        "рамзан",
        "рашид",
        "рашит",
        "роберт",
        "роман",
        "ростислав",
        "руслан",
        "рустам",
        "святослав",
        "семён",
        "семен",
        "сергей",
        "станислав",
        "степан",
        "султан",
        "тамерлан",
        "тарас",
        "тигран",
        "тимофей",
        "тимур",
        "фёдор",
        "федор",
        "филипп",
        "шамиль",
        "эдуард",
        "эльдар",
        "эрнст",
        "юрий",
        "ярослав",
        "александра",
        "алёна",
        "алена",
        "алина",
        "алиса",
        "алла",
        "алсу",
        "альбина",
        "амина",
        "анастасия",
        "анжелика",
        "анна",
        "айгуль",
        "валентина",
        "валерия",
        "варвара",
        "василиса",
        "венера",
        "вера",
        "вероника",
        "виктория",
        "галина",
        "гузель",
        "гульназ",
        "дарья",
        "диана",
        "диляра",
        "евгения",
        "евдокия",
        "екатерина",
        "елена",
        "елизавета",
        "жанна",
        "зарина",
        "зоя",
        "зульфия",
        "инна",
        "ирина",
        "камилла",
        "карина",
        "кристина",
        "ксения",
        "лариса",
        "лейла",
        "лидия",
        "любовь",
        "людмила",
        "мадина",
        "маргарита",
        "марина",
        "мария",
        "милана",
        "надежда",
        "наиля",
        "наталья",
        "наталия",
        "нина",
        "оксана",
        "ольга",
        "полина",
        "рената",
        "светлана",
        "серафима",
        "снежана",
        "софия",
        "софья",
        "тамара",
        "татьяна",
        "ульяна",
        "фатима",
        "элина",
        "эльвира",
        "юлия",
        "яна",
    }
)

_IRREGULAR_OBLIQUE_FIRST_NAMES: frozenset[str] = frozenset(
    {
        "павла",
        "павлу",
        "павлом",
        "павле",
        "льва",
        "льву",
        "львом",
        "льве",
        "петра",
        "петру",
        "петром",
        "петре",
        "любви",
        "любовью",
        "ильёй",
        "ильей",
    }
)

_NON_NAME_WORDS: frozenset[str] = frozenset(
    {
        "москва",
        "санкт",
        "петербург",
        "россия",
        "российский",
        "российская",
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
        "район",
        "округ",
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
        "парк",
        "сад",
        "океан",
        "море",
        "канал",
        "группа",
        "система",
        "портал",
        "отдел",
        "форум",
        "год",
        "век",
        "новый",
        "старый",
        "большой",
        "малый",
        "нижний",
        "верхний",
        "великий",
        "северный",
        "южный",
        "западный",
        "восточный",
        "центральный",
        "единый",
        "первый",
        "рабочий",
        "рабочая",
        "генеральный",
        "государственный",
        "федеральный",
        "федеральная",
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

# ---------------------------------------------------------------------------
# Fast linear-time Address patterns (< 30 ms on 100 KB inputs)
# ---------------------------------------------------------------------------

_ADDR_FAST_MARKERS = (
    "г.",
    "гор.",
    "город",
    "пгт",
    "пос.",
    "посёлок",
    "поселок",
    "дер.",
    "деревня",
    "ст-ца",
    "ст.",
    "станица",
    "мкр",
    "микрорайон",
    "ул.",
    "улиц",
    "пр.",
    "пр-т",
    "пр-кт",
    "просп",
    "пер.",
    "переул",
    "б-р",
    "бул.",
    "бульвар",
    "ш.",
    "шоссе",
    "наб.",
    "набережн",
    "пл.",
    "площад",
    "проезд",
    "алл.",
    "аллея",
    "туп.",
    "тупик",
    "индекс",
    "москва",
    "петербург",
    "новосибирск",
    "екатеринбург",
    "казань",
)

_ADDR_POSTAL_LEAD = rf"(?:(?:(?:почтовый{_WSEP}+)?индекс[ :№]*)?\d{{6}}[,]?{_WSEP}*)"
_ADDR_POSTAL_TAIL = (
    rf"(?:[,]?{_WSEP}*(?:почтовый{_WSEP}+)?индекс[ :№]*\d{{6}}"
    rf"|[,]?{_WSEP}+\d{{6}}(?![0-9]))"
)
_ADDR_REGION = (
    rf"(?:(?:(?:респ\.|республика){_WSEP}+[А-ЯЁа-яё][А-ЯЁа-яё-]+(?:{_WSEP}+[А-ЯЁа-яё][А-ЯЁа-яё-]+)?"
    rf"|[А-ЯЁа-яё][А-ЯЁа-яё-]+(?:-[А-ЯЁа-яё]+)?{_WSEP}+(?:обл\.|область|край|ао))"
    rf"[,]?{_WSEP}*)"
)
_ADDR_CITY = (
    r"(?:(?:г\.|гор\.|город|пгт|пос\.|посёлок|поселок|с\.|село|дер\.|деревня|ст-ца|станица)"
    rf"{_WSEP}*[А-ЯЁа-яё][А-ЯЁа-яё-]+(?:-[А-ЯЁа-яё]+)?"
    rf"|ст\.{_WSEP}*(?!\d)[А-ЯЁа-яё]{{3,}}[А-ЯЁа-яё-]*)"
)
_ADDR_NEXT_STOP = (
    r"(?!(?:д\.|дом\b|зд\.|здание\b|вл\.|корп\.|корпус\b|к\.|стр\.|строение\b"
    r"|кв\.?|квартира\b|оф\.?|офис\b|пом\.?|помещение\b|г\.|гор\.|город\b|с\.|пгт\b"
    r"|ст\.|ст-ца|пос\.|дер\.|индекс\b))"
)
_STREET_TYPE = (
    r"(?:ул\.|улица|пр\.|пр-т|пр-кт|просп\.|проспект|пер\.|переулок|б-р|бул\.|бульвар"
    r"|ш\.|шоссе|наб\.|набережная|пл\.|площадь|проезд|алл\.|аллея|туп\.|тупик|линия"
    r"|мкр\.?|микрорайон|кв-л|квартал)"
)
_ADDR_STREET_PREFIX = (
    rf"{_STREET_TYPE}{_WSEP}*(?:(?:им\.?|{_STREET_TYPE}){_WSEP}*)?"
    rf"(?:[А-ЯЁа-яё]\.{_WSEP}*){{0,2}}"
    rf"[А-ЯЁа-яё0-9«\"][А-ЯЁа-яё0-9«»\".-]{{0,28}}"
    rf"(?:{_WSEP}+{_ADDR_NEXT_STOP}[А-ЯЁа-яё0-9«»\".-]{{1,28}}){{0,2}}"
)
_ADDR_STREET_SUFFIX = (
    rf"[А-ЯЁа-яё][А-ЯЁа-яё0-9-]{{1,28}}(?:{_WSEP}+{_ADDR_NEXT_STOP}[А-ЯЁа-яё0-9-]{{1,24}})?{_WSEP}+"
    r"(?:ул\.|улица|пр\.|пр-т|пр-кт|просп\.|проспект|пер\.|переулок|б-р|бул\.|бульвар"
    r"|ш\.|шоссе|наб\.|набережная|площадь|проезд|алл\.|аллея|тупик)"
)
_ADDR_CORP = (
    rf"(?:[,]?{_WSEP}*(?:корп\.?|корпус|к\.|стр\.?|строение|лит\.?|литера)"
    rf"{_WSEP}*[0-9А-Яа-яA-Za-z]+){{0,2}}"
)
_ADDR_HOUSE_EXPLICIT = (
    rf"(?:[,]?{_WSEP}*(?:д\.|дом|зд\.|здание|вл\.|владение){_WSEP}*\d{{1,4}}[А-Яа-яA-Za-z]?"
    rf"(?:/\d{{1,4}}[А-Яа-яA-Za-z]?)?{_ADDR_CORP})"
)
_ADDR_HOUSE_OPT_PREFIX = (
    rf"(?:(?:[,]?{_WSEP}*(?:д\.|дом|зд\.|здание|вл\.){_WSEP}*|[,]?{_WSEP}+)"
    rf"\d{{1,4}}[А-Яа-яA-Za-z]?(?:/\d{{1,4}}[А-Яа-яA-Za-z]?)?(?![0-9]|\.\d|-фз|-ФЗ){_ADDR_CORP})"
)
_ADDR_APT = (
    rf"(?:[,]?{_WSEP}*(?:кв\.?|квартира|оф\.?|офис|пом\.?|помещение|ком\.?|комната)"
    rf"{_WSEP}*\d{{1,5}}[А-Яа-яA-Za-z]?)"
)

_ADDRESS_CITY_FIRST_RE = re.compile(
    rf"(?i)(?<![А-ЯЁа-яёA-Za-z0-9])"
    rf"{_ADDR_POSTAL_LEAD}?{_ADDR_REGION}?{_ADDR_CITY}[,]?{_WSEP}+"
    rf"(?:(?:{_ADDR_STREET_PREFIX}|{_ADDR_STREET_SUFFIX}){_ADDR_HOUSE_OPT_PREFIX}{_ADDR_APT}?"
    rf"|{_ADDR_HOUSE_EXPLICIT}{_ADDR_APT}?)"
    rf"{_ADDR_POSTAL_TAIL}?"
    rf"(?![А-ЯЁа-яёA-Za-z0-9])",
)

_ADDRESS_STREET_FIRST_RE = re.compile(
    rf"(?i)(?<![А-ЯЁа-яёA-Za-z0-9])"
    rf"{_ADDR_POSTAL_LEAD}?{_ADDR_STREET_PREFIX}{_ADDR_HOUSE_OPT_PREFIX}{_ADDR_APT}?"
    rf"(?:[,]?{_WSEP}+{_ADDR_CITY})?"
    rf"{_ADDR_POSTAL_TAIL}?"
    rf"(?![А-ЯЁа-яёA-Za-z0-9])",
)

_MAJOR_CITIES = (
    r"(?:Москва|Санкт-Петербург|Новосибирск|Екатеринбург|Казань|Нижний[ ]+Новгород|"
    r"Челябинск|Самара|Омск|Ростов-на-Дону|Уфа|Красноярск|Воронеж|Пермь|Волгоград|"
    r"Краснодар|Саратов|Тюмень|Тольятти|Ижевск|Барнаул|Ульяновск|Иркутск|Хабаровск|"
    r"Ярославль|Владивосток|Махачкала|Томск|Оренбург|Кемерово|Рязань|Астрахань|"
    r"Калининград|Тула|Тверь|Курск|Сочи|Севастополь|Симферополь)"
)
_SHORT_ADDRESS_RE = re.compile(
    rf"(?i)(?<![А-ЯЁа-яёA-Za-z0-9])"
    rf"{_ADDR_POSTAL_LEAD}?{_MAJOR_CITIES}[,]{_WSEP}+"
    rf"[А-ЯЁа-яё][А-ЯЁа-яё0-9-]{{2,28}}(?:{_WSEP}+{_ADDR_NEXT_STOP}[А-ЯЁа-яё0-9-]{{2,24}})?"
    rf"{_ADDR_HOUSE_OPT_PREFIX}{_ADDR_APT}?{_ADDR_POSTAL_TAIL}?"
    rf"(?![А-ЯЁа-яёA-Za-z0-9])",
)

_STANDALONE_POSTAL_RE = re.compile(
    rf"(?i)(?<![А-ЯЁа-яёA-Za-z0-9])(?:почтовый{_WSEP}+)?индекс[ :№]+\d{{6}}(?![0-9А-ЯЁа-яёA-Za-z])",
)

_HAS_DIGIT_RE = re.compile(r"\d")

# Organization patterns (ООО/АО/ПАО/ИП ...)
_LEGAL_CORP = r"(?:ООО|ОАО|ЗАО|ПАО|НАО|АО|НКО|АНО|ФГУП|ГУП|МУП|ФГБУ|ГБУ|МБУ|ТСЖ|ЖСК|СНТ|КФХ|ГК)"
_QUOTED_NAME = r"(?:«[^»\n]{1,80}»|\"[^\"\n]{1,80}\"|„[^“\"\n]{1,80}[“\"]|'[^'\n]{1,80}')"
_ORG_QUOTED_RE = re.compile(
    rf"(?<![А-ЯЁа-яёA-Za-z0-9])(?:{_LEGAL_CORP}|ИП){_WSEP}+{_QUOTED_NAME}",
)
_ORG_IP_RE = re.compile(
    rf"(?<![А-ЯЁа-яёA-Za-z0-9])ИП{_WSEP}+"
    r"(?P<fio>(?P<surname>[А-ЯЁ][а-яё]{1,30}(?:-[А-ЯЁ][а-яё]{1,30})?)"
    rf"(?:{_WSEP}+(?:[А-ЯЁ]\.(?:{_WSEP}?[А-ЯЁ]\.)?|[А-ЯЁ][а-яё]{{1,30}}(?:{_WSEP}+[А-ЯЁ][а-яё]{{1,30}})?))?)"
    r"(?![А-ЯЁа-яёA-Za-z.])",
)
_ORG_UNQUOTED_RE = re.compile(
    rf"(?<![А-ЯЁа-яёA-Za-z0-9]){_LEGAL_CORP}{_WSEP}+"
    rf"([А-ЯЁA-Z][А-ЯЁа-яёA-Za-z0-9-]{{1,30}}(?:{_WSEP}+[А-ЯЁA-Z][А-ЯЁа-яёA-Za-z0-9-]{{1,30}}){{0,2}})"
    r"(?![А-ЯЁа-яёA-Za-z0-9])",
)


@lru_cache(maxsize=4096)
def _is_patronymic_low(low: str) -> bool:
    if low in _TURKIC_PATRONYMIC_PARTICLES:
        return True
    return len(low) >= 4 and low.endswith(_PATRONYMIC_SUFFIXES)


def _is_patronymic(word: str) -> bool:
    return _is_patronymic_low(word.lower())


@lru_cache(maxsize=4096)
def _is_likely_surname_low(low: str) -> bool:
    if low in _NON_NAME_WORDS:
        return False
    if low in _SHORT_EXACT_SURNAMES:
        return True
    if "-" in low:
        parts = low.split("-")
        return all(len(p) >= 2 for p in parts) and any(
            p.endswith(_SURNAME_SUFFIXES) or p in _SHORT_EXACT_SURNAMES for p in parts
        )
    return len(low) >= 3 and low.endswith(_SURNAME_SUFFIXES)


def _is_likely_surname(word: str) -> bool:
    return _is_likely_surname_low(word.lower())


@lru_cache(maxsize=4096)
def _is_common_first_name_low(low: str) -> bool:
    if low in _COMMON_FIRST_NAMES or low in _IRREGULAR_OBLIQUE_FIRST_NAMES:
        return True
    # Match common inflected forms by stripping a single Russian case suffix
    if len(low) >= 4 and low[-1] in {"а", "у", "е", "ы", "и", "ю", "я"}:
        stem = low[:-1]
        if (
            stem in _COMMON_FIRST_NAMES
            or f"{stem}й" in _COMMON_FIRST_NAMES
            or f"{stem}а" in _COMMON_FIRST_NAMES
            or f"{stem}я" in _COMMON_FIRST_NAMES
            or f"{stem}ь" in _COMMON_FIRST_NAMES
        ):
            return True
    if len(low) >= 5 and low[-2:] in {
        "ом",
        "ем",
        "ём",
        "ой",
        "ей",
        "ёй",
        "ью",
        "юю",
        "ию",
        "ии",
        "ие",
    }:
        stem2 = low[:-2]
        if (
            stem2 in _COMMON_FIRST_NAMES
            or f"{stem2}й" in _COMMON_FIRST_NAMES
            or f"{stem2}а" in _COMMON_FIRST_NAMES
            or f"{stem2}я" in _COMMON_FIRST_NAMES
            or f"{stem2}ь" in _COMMON_FIRST_NAMES
            or f"{stem2}ий" in _COMMON_FIRST_NAMES
            or f"{stem2}ия" in _COMMON_FIRST_NAMES
            or f"{stem2}ей" in _COMMON_FIRST_NAMES
        ):
            return True
    return False


def _is_common_first_name(word: str) -> bool:
    return _is_common_first_name_low(word.lower())


class _DisjointIntervals:
    """Fast O(log M) interval overlap tracker using sorted parallel endpoint arrays."""

    __slots__ = ("_ends", "_starts")

    def __init__(self) -> None:
        self._starts: list[int] = []
        self._ends: list[int] = []

    def overlaps(self, start: int, end: int) -> bool:
        if not self._starts:
            return False
        idx = bisect_right(self._starts, start)
        if idx > 0 and self._ends[idx - 1] > start:
            return True
        return idx < len(self._starts) and self._starts[idx] < end

    def add_if_disjoint(self, start: int, end: int) -> bool:
        idx = bisect_right(self._starts, start)
        if idx > 0 and self._ends[idx - 1] > start:
            return False
        if idx < len(self._starts) and self._starts[idx] < end:
            return False
        self._starts.insert(idx, start)
        self._ends.insert(idx, end)
        return True


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
        if not segment.index_map and segment.span_source == SpanSource.DIRECT:
            mapped_start, mapped_end = norm_start, norm_end
        else:
            mapped_start, mapped_end = segment.map_span(norm_start, norm_end)
        clamped_conf = min(1.0, max(0.0, round(confidence, 4)))
        det = Detection.__new__(Detection)
        object.__setattr__(
            det,
            "__dict__",
            {
                "category": category,
                "start": mapped_start,
                "end": mapped_end,
                "confidence": clamped_conf,
                "detector": self.name,
                "span_source": segment.span_source,
                "evasion_detected": bool(segment.evasion_techniques),
                "evasion_techniques": segment.evasion_techniques,
                "compliance_refs": _COMPLIANCE_CACHE[category],
                "masked_preview": f"[{category.value}:len={len(raw_match)}]",
                "value_hmac": self._hmac_fn(raw_match) if self._hmac_fn else "",
                "source_message_indices": (),
                "metadata": {"ner_backend": source},
            },
        )
        object.__setattr__(det, "__pydantic_fields_set__", set(_DETECTION_FIELDS_SET))
        object.__setattr__(det, "__pydantic_extra__", None)
        object.__setattr__(det, "__pydantic_private__", None)
        return det

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

    def _scan_fio_tokens(
        self,
        text: str,
        text_lower: str,
        tokens: list[tuple[str, int, int]],
        address_spans: _DisjointIntervals,
        org_spans: _DisjointIntervals,
        person_spans: _DisjointIntervals,
        findings: list[tuple[EntityCategory, int, int, float, str]],
    ) -> bool:
        claimed: set[int] = set()
        n_tok = len(tokens)
        lows = [w.lower() for w, _, _ in tokens]
        saw_candidate = False

        # Pass 1: 3-word FIO ("Фамилия Имя Отчество", "Имя Отчество Фамилия", or "... Оглы")
        idx = 0
        while idx + 3 <= n_tok:
            _, s1, _ = tokens[idx]
            _, _, e3 = tokens[idx + 2]
            l1 = lows[idx]
            l2 = lows[idx + 1]
            l3 = lows[idx + 2]
            if (
                l1 not in _NON_NAME_WORDS
                and l2 not in _NON_NAME_WORDS
                and l3 not in _NON_NAME_WORDS
                and (_is_patronymic_low(l3) or _is_patronymic_low(l2))
            ):
                saw_candidate = True
                # Extend trailing Turkic particle ("... Рустамович угли") if present
                tail_m = _TRAILING_PARTICLE_RE.match(text, e3)
                if tail_m is not None:
                    e3 = tail_m.end()
                    if idx + 3 < n_tok and tokens[idx + 3][2] <= e3:
                        claimed.add(idx + 3)
                if not address_spans.overlaps(s1, e3) and person_spans.add_if_disjoint(s1, e3):
                    findings.append((EntityCategory.RU_PERSON_NAME, s1, e3, 0.92, "heuristic"))
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
            _, s1, _ = tokens[idx]
            _, _, e2 = tokens[idx + 1]
            l1 = lows[idx]
            l2 = lows[idx + 1]
            if (
                l1 not in _NON_NAME_WORDS
                and l2 not in _NON_NAME_WORDS
                and not (address_spans.overlaps(s1, e2) or org_spans.overlaps(s1, e2))
            ):
                is_fn_sn = _is_common_first_name_low(l1) and _is_likely_surname_low(l2)
                is_sn_fn = _is_likely_surname_low(l1) and _is_common_first_name_low(l2)
                if is_fn_sn or is_sn_fn:
                    saw_candidate = True
                    before_ctx = text_lower[max(0, s1 - 48) : s1]
                    if not _ART_WORK_CTX_RE.search(before_ctx) and person_spans.add_if_disjoint(
                        s1, e2
                    ):
                        findings.append((EntityCategory.RU_PERSON_NAME, s1, e2, 0.82, "heuristic"))
                        claimed.update((idx, idx + 1))
                        idx += 2
                        continue
            idx += 1
        return saw_candidate

    def _detect_heuristics(self, text: str) -> list[tuple[EntityCategory, int, int, float, str]]:
        findings: list[tuple[EntityCategory, int, int, float, str]] = []
        text_lower = text.lower()

        # 1. Addresses first (fast marker + digit pre-check keeps 100 KB prose < 5 ms)
        address_spans = _DisjointIntervals()
        if _HAS_DIGIT_RE.search(text) and any(
            marker in text_lower for marker in _ADDR_FAST_MARKERS
        ):
            addr_candidates: list[tuple[int, int, float]] = []
            for addr_re in (_ADDRESS_CITY_FIRST_RE, _ADDRESS_STREET_FIRST_RE, _SHORT_ADDRESS_RE):
                for m in addr_re.finditer(text):
                    snippet_low = m.group(0).lower()
                    has_apt = any(
                        kw in snippet_low for kw in ("кв", "квартира", "оф", "офис", "индекс")
                    )
                    addr_candidates.append((m.start(), m.end(), 0.94 if has_apt else 0.89))

            if "индекс" in text_lower:
                for m in _STANDALONE_POSTAL_RE.finditer(text):
                    addr_candidates.append((m.start(), m.end(), 0.88))

            # Prefer longer/higher-confidence address spans first
            addr_candidates.sort(key=lambda item: (-(item[1] - item[0]), -item[2], item[0]))
            for a_start, a_end, a_conf in addr_candidates:
                if address_spans.add_if_disjoint(a_start, a_end):
                    findings.append(
                        (EntityCategory.RU_ADDRESS, a_start, a_end, a_conf, "heuristic")
                    )

        # 2. Organizations & Sole-Proprietor (ИП) citizen FIO extraction
        org_spans = _DisjointIntervals()
        person_spans = _DisjointIntervals()

        for m in _ORG_QUOTED_RE.finditer(text):
            if org_spans.add_if_disjoint(m.start(), m.end()):
                findings.append((EntityCategory.RU_ORG_NAME, m.start(), m.end(), 0.92, "heuristic"))

        if "ИП" in text:
            for m in _ORG_IP_RE.finditer(text):
                if org_spans.overlaps(m.start(), m.end()) or address_spans.overlaps(
                    m.start(), m.end()
                ):
                    continue
                surname = m.group("surname")
                if surname.lower() in _NON_NAME_WORDS:
                    continue
                # Emit both RU_ORG_NAME ('ИП ...') and RU_PERSON_NAME (higher conf) for citizen FIO
                findings.append((EntityCategory.RU_ORG_NAME, m.start(), m.end(), 0.88, "heuristic"))
                fio_start, fio_end = m.span("fio")
                if person_spans.add_if_disjoint(fio_start, fio_end):
                    findings.append(
                        (EntityCategory.RU_PERSON_NAME, fio_start, fio_end, 0.93, "heuristic")
                    )

        for m in _ORG_UNQUOTED_RE.finditer(text):
            if not address_spans.overlaps(m.start(), m.end()) and org_spans.add_if_disjoint(
                m.start(), m.end()
            ):
                findings.append((EntityCategory.RU_ORG_NAME, m.start(), m.end(), 0.85, "heuristic"))

        # 3. Person names (Title-Case runs, ALL-CAPS runs, initials, context lowercase)
        miss_cap_runs: set[str] = set()
        for run_m in _CAP_RUN_RE.finditer(text):
            run_str = run_m.group(0)
            if run_str in miss_cap_runs:
                continue
            base, run_end = run_m.span()
            if (
                "\u00a0" not in run_str
                and "\u202f" not in run_str
                and "\u2009" not in run_str
                and "  " not in run_str
            ):
                tokens = []
                tok_pos = base
                for part in run_str.split(" "):
                    tok_end = tok_pos + len(part)
                    tokens.append((part, tok_pos, tok_end))
                    tok_pos = tok_end + 1
            else:
                tokens = [
                    (tm.group(0), base + tm.start(), base + tm.end())
                    for tm in _CAP_TOKEN_RE.finditer(run_str)
                ]
            saw_cand = self._scan_fio_tokens(
                text, text_lower, tokens, address_spans, org_spans, person_spans, findings
            )
            if (
                not saw_cand
                and len(miss_cap_runs) < 1024
                and not address_spans.overlaps(base, run_end)
                and not org_spans.overlaps(base, run_end)
            ):
                miss_cap_runs.add(run_str)

        if _TWO_UPPER_RE.search(text):
            for run_m in _UPPER_RUN_RE.finditer(text):
                base = run_m.start()
                tokens = [
                    (tm.group(0), base + tm.start(), base + tm.end())
                    for tm in _UPPER_TOKEN_RE.finditer(run_m.group(0))
                ]
                self._scan_fio_tokens(
                    text, text_lower, tokens, address_spans, org_spans, person_spans, findings
                )

        if "." in text:
            for m in _SURNAME_INITIALS_RE.finditer(text):
                surname = m.group(1)
                initials = m.group(2)
                if surname.lower() in _NON_NAME_WORDS:
                    continue
                if initials.count(".") == 1 and not _is_likely_surname(surname):
                    continue
                s_start, s_end = m.start(), m.end()
                if not address_spans.overlaps(s_start, s_end) and person_spans.add_if_disjoint(
                    s_start, s_end
                ):
                    findings.append(
                        (EntityCategory.RU_PERSON_NAME, s_start, s_end, 0.90, "heuristic")
                    )

            for m in _INITIALS_SURNAME_RE.finditer(text):
                surname = m.group(2)
                if surname.lower() in _NON_NAME_WORDS:
                    continue
                s_start, s_end = m.start(), m.end()
                if not address_spans.overlaps(s_start, s_end) and person_spans.add_if_disjoint(
                    s_start, s_end
                ):
                    findings.append(
                        (EntityCategory.RU_PERSON_NAME, s_start, s_end, 0.90, "heuristic")
                    )

            for m in _SINGLE_INITIAL_SURNAME_RE.finditer(text):
                surname = m.group(2)
                if not _is_likely_surname(surname):
                    continue
                s_start, s_end = m.start(), m.end()
                if not address_spans.overlaps(s_start, s_end) and person_spans.add_if_disjoint(
                    s_start, s_end
                ):
                    findings.append(
                        (EntityCategory.RU_PERSON_NAME, s_start, s_end, 0.88, "heuristic")
                    )

        for m in _LOWER_FIO_CTX_RE.finditer(text):
            w1, w2, w3 = m.group(1), m.group(2), m.group(3)
            if any(w in _NON_NAME_WORDS for w in (w1, w2, w3)):
                continue
            if _is_patronymic(w3) or _is_patronymic(w2):
                s_start, s_end = m.start(1), m.end(3)
                if not address_spans.overlaps(s_start, s_end) and person_spans.add_if_disjoint(
                    s_start, s_end
                ):
                    findings.append(
                        (EntityCategory.RU_PERSON_NAME, s_start, s_end, 0.89, "heuristic")
                    )

        return findings

    def detect(self, segment: NormalizedSegment, raw_text: str) -> list[Detection]:
        """Detect Russian person names, addresses, and organization names."""
        _ = raw_text
        text = segment.text
        if not text or not text.strip():
            return []

        natasha_hits = self._detect_natasha(text)
        heuristic_hits = self._detect_heuristics(text)

        # Merge natasha + heuristic spans in O(M log M) per category
        intervals_by_cat: dict[EntityCategory, _DisjointIntervals] = {
            EntityCategory.RU_PERSON_NAME: _DisjointIntervals(),
            EntityCategory.RU_ADDRESS: _DisjointIntervals(),
            EntityCategory.RU_ORG_NAME: _DisjointIntervals(),
        }
        merged: list[tuple[EntityCategory, int, int, float, str]] = []
        for cat, start, end, conf, src in (*natasha_hits, *heuristic_hits):
            tracker = intervals_by_cat.get(cat)
            if tracker is None or tracker.add_if_disjoint(start, end):
                merged.append((cat, start, end, conf, src))

        detections = [
            self._build_detection(segment, cat, start, end, conf, src)
            for cat, start, end, conf, src in merged
        ]
        detections.sort(key=lambda det: (det.start, det.end, -det.confidence))
        return detections


__all__ = ["RuNerDetector"]
