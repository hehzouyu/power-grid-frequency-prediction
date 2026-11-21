"""Set Chinese journal typography after Pandoc's editable LaTeX-to-Word conversion."""
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor


def standard_font(style, east_asia, size):
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(size)
    font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().rFonts
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        fonts.attrib.pop(qn(f"w:{attr}"), None)
    fonts.set(qn("w:eastAsia"), east_asia)


def alter_table(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_properties = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        for attr, value in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
            border.set(qn(f"w:{attr}"), value)
        borders.append(border)
    previous = table_properties.find(qn("w:tblBorders"))
    if previous is not None:
        table_properties.remove(previous)
    table_properties.append(borders)
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Mm(width)
    for row_number, row in enumerate(table.rows):
        props = row._tr.get_or_add_trPr()
        props.append(OxmlElement("w:cantSplit"))
        if row_number == 0 and props.find(qn("w:tblHeader")) is None:
            props.append(OxmlElement("w:tblHeader"))
        for index, (cell, width) in enumerate(zip(row.cells, widths, strict=True)):
            cell.width = Mm(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_number == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "F3F3F3")
                cell._tc.get_or_add_tcPr().append(shade)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_after = Pt(3)
                paragraph.paragraph_format.space_before = Pt(3)
                paragraph.paragraph_format.keep_with_next = row_number < len(table.rows) - 1
                for run in paragraph.runs:
                    run.font.size = Pt(10)


def format_document(path):
    document = Document(path)
    for section in document.sections:
        section.page_width, section.page_height = Mm(210), Mm(297)
        section.left_margin = section.right_margin = Mm(23)
        section.top_margin = section.bottom_margin = Mm(23)
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        style = document.styles[name]
        standard_font(style, "SimSun", 11)
        style.paragraph_format.line_spacing = 1.27
        style.paragraph_format.space_after = Pt(6)
    title = document.styles["Title"]
    standard_font(title, "SimHei", 18)
    title.font.bold = True
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(12)
    border = title.element.get_or_add_pPr().find(qn("w:pBdr"))
    if border is not None:
        title.element.pPr.remove(border)
    for name, size in [("Heading 1", 13), ("Heading 2", 11), ("Caption", 10),
                       ("Table Caption", 10), ("Image Caption", 10)]:
        if name in document.styles:
            style = document.styles[name]
            standard_font(style, "SimHei" if name.startswith("Heading") else "SimSun", size)
            style.font.italic = False
            style.paragraph_format.keep_with_next = name != "Image Caption"
    for table, widths in zip(document.tables, [(52, 28, 28, 27, 29), (26, 48, 23, 23, 22, 22),
                                               (36, 23, 23, 23, 23, 36)], strict=True):
        alter_table(table, widths)
    for picture in document.inline_shapes:
        if picture.width > Mm(116):
            picture.height = int(picture.height * Mm(116) / picture.width)
            picture.width = Mm(116)
    for paragraph in document.paragraphs:
        if paragraph._p.xpath(".//w:drawing"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = True
    document.core_properties.author = ""
    document.core_properties.last_modified_by = ""
    document.core_properties.keywords = ""
    document.save(path)


if __name__ == "__main__":
    format_document(Path(sys.argv[1]))
