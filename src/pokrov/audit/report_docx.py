"""Pure-stdlib OpenXML (.docx) compliance report generator for Pokrov."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from xml.sax.saxutils import escape as xml_escape

from pokrov.audit.report_html import (
    REGULATORY_OVERVIEW_ITEMS,
    build_report_context,
    format_timestamp,
)
from pokrov.core.contracts import AuditEvent, AuditStats

_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# XML 1.0 forbids C0 control characters other than HT (\x09), LF (\x0A), and CR (\x0D),
# as well as unpaired surrogates and U+FFFE/U+FFFF.
_INVALID_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff\ufffe\uffff]")

_XML_ENTITY_MAP: dict[str, str] = {
    '"': "&quot;",
    "'": "&apos;",
}

_CONTENT_TYPES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.'
    'wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.'
    'wordprocessingml.styles+xml"/>'
    "</Types>"
)

_ROOT_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/>'
    "</Relationships>"
)

_DOCUMENT_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
    'Target="styles.xml"/>'
    "</Relationships>"
)

_STYLES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<w:styles xmlns:w="{_WORD_NS}">'
    "<w:docDefaults>"
    "<w:rPrDefault><w:rPr>"
    '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>'
    '<w:sz w:val="20"/><w:szCs w:val="20"/>'
    '<w:color w:val="0F172A"/>'
    "</w:rPr></w:rPrDefault>"
    '<w:pPrDefault><w:pPr><w:spacing w:after="80" w:line="260" w:lineRule="auto"/>'
    "</w:pPr></w:pPrDefault>"
    "</w:docDefaults>"
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
    '<w:name w:val="Normal"/>'
    "</w:style>"
    '<w:style w:type="paragraph" w:styleId="Title">'
    '<w:name w:val="Title"/>'
    '<w:basedOn w:val="Normal"/>'
    '<w:pPr><w:spacing w:before="0" w:after="100"/></w:pPr>'
    '<w:rPr><w:b/><w:color w:val="1E3A5F"/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr>'
    "</w:style>"
    '<w:style w:type="paragraph" w:styleId="Subtitle">'
    '<w:name w:val="Subtitle"/>'
    '<w:basedOn w:val="Normal"/>'
    '<w:pPr><w:spacing w:before="0" w:after="140"/></w:pPr>'
    '<w:rPr><w:color w:val="475569"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
    "</w:style>"
    '<w:style w:type="paragraph" w:styleId="Heading1">'
    '<w:name w:val="Heading 1"/>'
    '<w:basedOn w:val="Normal"/>'
    '<w:pPr><w:keepNext/><w:spacing w:before="220" w:after="100"/></w:pPr>'
    '<w:rPr><w:b/><w:color w:val="1E3A5F"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>'
    "</w:style>"
    '<w:style w:type="paragraph" w:styleId="Heading2">'
    '<w:name w:val="Heading 2"/>'
    '<w:basedOn w:val="Normal"/>'
    '<w:pPr><w:keepNext/><w:spacing w:before="160" w:after="80"/></w:pPr>'
    '<w:rPr><w:b/><w:color w:val="2B4C7E"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
    "</w:style>"
    '<w:style w:type="table" w:styleId="PokrovTable">'
    '<w:name w:val="PokrovTable"/>'
    "<w:tblPr>"
    "<w:tblCellMar>"
    '<w:top w:w="70" w:type="dxa"/>'
    '<w:left w:w="100" w:type="dxa"/>'
    '<w:bottom w:w="70" w:type="dxa"/>'
    '<w:right w:w="100" w:type="dxa"/>'
    "</w:tblCellMar>"
    "</w:tblPr>"
    "</w:style>"
    "</w:styles>"
)


def _xml_text(value: str) -> str:
    """Strip invalid XML 1.0 characters and escape XML special characters."""
    cleaned = _INVALID_XML_RE.sub("", value)
    return xml_escape(cleaned, _XML_ENTITY_MAP)


def _build_run(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    color: str = "",
    size_half_pt: int = 0,
) -> str:
    """Build a <w:r> element supporting multi-line text via <w:br/>."""
    rpr_parts: list[str] = []
    if bold:
        rpr_parts.append("<w:b/>")
    if italic:
        rpr_parts.append("<w:i/>")
    if color:
        rpr_parts.append(f'<w:color w:val="{_xml_text(color)}"/>')
    if size_half_pt > 0:
        rpr_parts.append(f'<w:sz w:val="{size_half_pt}"/><w:szCs w:val="{size_half_pt}"/>')

    rpr = f"<w:rPr>{''.join(rpr_parts)}</w:rPr>" if rpr_parts else ""
    lines = text.split("\n")
    content_parts: list[str] = []
    for idx, line in enumerate(lines):
        if idx > 0:
            content_parts.append("<w:br/>")
        content_parts.append(f'<w:t xml:space="preserve">{_xml_text(line)}</w:t>')
    return f"<w:r>{rpr}{''.join(content_parts)}</w:r>"


def _build_paragraph(
    text: str,
    *,
    style: str = "Normal",
    bold: bool = False,
    italic: bool = False,
    color: str = "",
    size_half_pt: int = 0,
    shade: str = "",
    left_border_color: str = "",
    space_before: int | None = None,
    space_after: int | None = None,
) -> str:
    """Build a <w:p> element with optional shading and left accent border."""
    ppr_parts: list[str] = [f'<w:pStyle w:val="{_xml_text(style)}"/>']
    if left_border_color:
        border_color = _xml_text(left_border_color)
        ppr_parts.append(
            "<w:pBdr>"
            f'<w:left w:val="single" w:sz="18" w:space="8" w:color="{border_color}"/>'
            "</w:pBdr>"
        )
    if shade:
        ppr_parts.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{_xml_text(shade)}"/>')
    if space_before is not None or space_after is not None:
        attrs: list[str] = []
        if space_before is not None:
            attrs.append(f'w:before="{space_before}"')
        if space_after is not None:
            attrs.append(f'w:after="{space_after}"')
        ppr_parts.append(f"<w:spacing {' '.join(attrs)}/>")
    ppr = f"<w:pPr>{''.join(ppr_parts)}</w:pPr>"
    run = _build_run(
        text,
        bold=bold,
        italic=italic,
        color=color,
        size_half_pt=size_half_pt,
    )
    return f"<w:p>{ppr}{run}</w:p>"


def _build_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    col_widths: Sequence[int],
    font_size_half_pt: int = 18,
) -> str:
    """Build a styled <w:tbl> with fixed grid column widths and repeating header row."""
    if len(headers) != len(col_widths):
        raise ValueError("Header count must match col_widths length")

    total_width = sum(col_widths)
    grid_cols = "".join(f'<w:gridCol w:w="{width}"/>' for width in col_widths)
    borders = (
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        "</w:tblBorders>"
    )
    tbl_pr = (
        "<w:tblPr>"
        '<w:tblStyle w:val="PokrovTable"/>'
        f'<w:tblW w:w="{total_width}" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        f"{borders}"
        "</w:tblPr>"
    )

    header_cells: list[str] = []
    for width, header in zip(col_widths, headers, strict=True):
        cell_p = _build_paragraph(
            header,
            bold=True,
            color="FFFFFF",
            size_half_pt=font_size_half_pt,
            space_before=0,
            space_after=0,
        )
        header_cells.append(
            "<w:tc>"
            "<w:tcPr>"
            f'<w:tcW w:w="{width}" w:type="dxa"/>'
            '<w:shd w:val="clear" w:color="auto" w:fill="1E3A5F"/>'
            '<w:vAlign w:val="center"/>'
            "</w:tcPr>"
            f"{cell_p}"
            "</w:tc>"
        )
    header_row = (
        f"<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>{''.join(header_cells)}</w:tr>"
    )

    body_rows: list[str] = []
    for row_idx, row in enumerate(rows):
        if len(row) != len(col_widths):
            raise ValueError("Row cell count must match col_widths length")
        fill = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        cells: list[str] = []
        for width, cell_text in zip(col_widths, row, strict=True):
            cell_p = _build_paragraph(
                cell_text,
                size_half_pt=font_size_half_pt,
                space_before=0,
                space_after=0,
            )
            cells.append(
                "<w:tc>"
                "<w:tcPr>"
                f'<w:tcW w:w="{width}" w:type="dxa"/>'
                f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>'
                '<w:vAlign w:val="top"/>'
                "</w:tcPr>"
                f"{cell_p}"
                "</w:tc>"
            )
        body_rows.append(f"<w:tr><w:trPr><w:cantSplit/></w:trPr>{''.join(cells)}</w:tr>")

    return (
        f"<w:tbl>{tbl_pr}<w:tblGrid>{grid_cols}</w:tblGrid>{header_row}{''.join(body_rows)}</w:tbl>"
    )


class DocxComplianceReportGenerator:
    """Pure-stdlib OpenXML (.docx) compliance report generator (ReportGenerator Protocol)."""

    def __init__(self, *, now_fn: Callable[[], datetime] | None = None) -> None:
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def render_bytes(
        self,
        events: Sequence[AuditEvent],
        stats: AuditStats,
        *,
        title: str = "Отчёт Покров (Pokrov)",
    ) -> bytes:
        """Render a Zero-Retention compliance report as a valid OpenXML .docx archive."""
        document_xml = self._build_document_xml(events, stats, title=title)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", _CONTENT_TYPES_XML.encode("utf-8"))
            zf.writestr("_rels/.rels", _ROOT_RELS_XML.encode("utf-8"))
            zf.writestr("word/_rels/document.xml.rels", _DOCUMENT_RELS_XML.encode("utf-8"))
            zf.writestr("word/styles.xml", _STYLES_XML.encode("utf-8"))
            zf.writestr("word/document.xml", document_xml.encode("utf-8"))
        return buffer.getvalue()

    def _build_document_xml(
        self,
        events: Sequence[AuditEvent],
        stats: AuditStats,
        *,
        title: str,
    ) -> str:
        ctx = build_report_context(events, stats, title=title, generated_at=self._now_fn())
        body_parts: list[str] = [
            _build_paragraph(ctx.title, style="Title"),
            _build_paragraph(
                f"Сформирован: {ctx.generated_at_str} | Период: {ctx.period_str} | "
                f"Событий в выборке: {ctx.events_count} | Режим: Zero-Retention",
                style="Subtitle",
            ),
            _build_paragraph(
                f"Правовой дисклеймер: {ctx.legal_disclaimer}",
                color="78350F",
                size_half_pt=18,
                shade="FFFBEB",
                left_border_color="D97706",
                space_before=60,
                space_after=160,
            ),
            _build_paragraph("1. Ключевые показатели безопасности (KPI)", style="Heading1"),
            _build_table(
                ("Показатель", "Код метрики", "Значение"),
                [(k.label, k.key, k.value) for k in ctx.kpis],
                col_widths=(5700, 2600, 1800),
            ),
            _build_paragraph(
                "2. Разбивка по категориям угроз (EntityCategory) и контрольным точкам",
                style="Heading1",
            ),
            _build_paragraph("Категории обнаружений (EntityCategory)", style="Heading2"),
        ]

        if ctx.category_rows:
            body_parts.append(
                _build_table(
                    ("Категория", "Количество", "Нормативные привязки"),
                    [(row.label_ru, str(row.count), row.frameworks) for row in ctx.category_rows],
                    col_widths=(3600, 1300, 5200),
                )
            )
        else:
            body_parts.append(
                _build_paragraph(
                    "Обнаружения по категориям за отчётный период отсутствуют.",
                    italic=True,
                    color="475569",
                )
            )

        body_parts.append(
            _build_paragraph("Контрольные точки трафика (ControlPoint)", style="Heading2")
        )
        if ctx.control_point_rows:
            body_parts.append(
                _build_table(
                    ("Контрольная точка", "Количество событий"),
                    [(row.label_ru, str(row.count)) for row in ctx.control_point_rows],
                    col_widths=(7800, 2300),
                )
            )
        else:
            body_parts.append(
                _build_paragraph(
                    "События по контрольным точкам за отчётный период отсутствуют.",
                    italic=True,
                    color="475569",
                )
            )

        body_parts.append(
            _build_paragraph(
                "3. Нормативное покрытие: 152-ФЗ, ст. 13.11 КоАП РФ (420-ФЗ), "
                "Приказ ФСТЭК России № 117 (п. 3.18, РСБ)",
                style="Heading1",
            )
        )
        for reg_name, reg_desc in REGULATORY_OVERVIEW_ITEMS:
            body_parts.append(
                _build_paragraph(
                    f"• {reg_name}: {reg_desc}",
                    size_half_pt=18,
                    space_after=60,
                )
            )

        if ctx.regulatory_rows:
            body_parts.append(
                _build_table(
                    (
                        "Стандарт / закон",
                        "Статья / пункт",
                        "Требование и мера",
                        "Категории",
                        "Событий / риск",
                    ),
                    [
                        (
                            row.framework,
                            row.article,
                            row.title,
                            row.categories,
                            f"{row.hits}: {row.risk_hint}" if row.risk_hint else str(row.hits),
                        )
                        for row in ctx.regulatory_rows
                    ],
                    col_widths=(1500, 1400, 2700, 1800, 2700),
                    font_size_half_pt=16,
                )
            )
        else:
            body_parts.append(
                _build_paragraph(
                    "Инциденты с нормативными срабатываниями за отчётный период не зафиксированы.",
                    italic=True,
                    color="475569",
                )
            )

        body_parts.append(
            _build_paragraph(
                "4. Журнал инцидентов (Zero-Retention — без хранения сырых ПДн)",
                style="Heading1",
            )
        )
        if events:
            incident_rows: list[tuple[str, ...]] = []
            for ev in events:
                cats = ", ".join(c.value for c in ev.categories) or "—"
                hmacs = ", ".join(ev.value_hmacs) or "—"
                incident_rows.append(
                    (
                        ev.event_id,
                        format_timestamp(ev.timestamp),
                        ev.control_point.value,
                        ev.action.value,
                        ev.severity.value,
                        cats,
                        hmacs,
                    )
                )
            body_parts.append(
                _build_table(
                    (
                        "event_id",
                        "timestamp",
                        "control_point",
                        "action",
                        "severity",
                        "categories",
                        "HMAC digests",
                    ),
                    incident_rows,
                    col_widths=(1150, 1500, 1200, 1150, 950, 1550, 2600),
                    font_size_half_pt=16,
                )
            )
        else:
            body_parts.append(
                _build_paragraph(
                    "Инциденты безопасности за отчётный период не зафиксированы.",
                    italic=True,
                    color="475569",
                )
            )

        sect_pr = (
            "<w:sectPr>"
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" '
            'w:header="708" w:footer="708" w:gutter="0"/>'
            "</w:sectPr>"
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{_WORD_NS}">'
            f"<w:body>{''.join(body_parts)}{sect_pr}</w:body>"
            "</w:document>"
        )


__all__ = ["DocxComplianceReportGenerator"]
