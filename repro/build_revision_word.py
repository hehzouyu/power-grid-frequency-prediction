"""Convert the revised LaTeX source into editable named and anonymous Word drafts."""
from pathlib import Path
import argparse
import re
import subprocess
import tempfile

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ART = Path(__file__).resolve().parent.parent / "artical"


def expanded_source(stem="ARTICAL-revised"):
    source = (ART / f"{stem}.tex").read_text(encoding="utf-8")
    metrics = (ART / "revision-assets" / "metrics.tex").read_text(encoding="utf-8")
    for name, value in re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]+)\}", metrics):
        source = source.replace("\\" + name + "{}", value)
    source = re.sub(
        r"\\resultrows\{([^{}]+)\}",
        lambda match: (ART / match[1]).read_text(encoding="utf-8"),
        source,
    )
    refs = {}
    for kind in ("figure", "table", "equation"):
        for i, block in enumerate(re.findall(
            rf"\\begin\{{{kind}\}}(.*?)\\end\{{{kind}\}}", source, re.S
        ), 1):
            label = re.search(r"\\label\{([^{}]+)\}", block)
            if label:
                refs[label[1]] = i
    source = re.sub(r"\\eqref\{([^{}]+)\}", lambda m: f"（{refs[m[1]]}）", source)
    source = re.sub(r"\\ref\{([^{}]+)\}", lambda m: str(refs[m[1]]), source)
    source = re.sub(r"\\label\{[^{}]+\}", "", source)
    bib = re.findall(r"\\bibitem\{([^{}]+)\}", source)
    source = re.sub(
        r"\\cite\{([^{}]+)\}",
        lambda m: "[" + ", ".join(str(bib.index(key) + 1) for key in m[1].split(",")) + "]",
        source,
    )
    source = source.replace(r"\begin{thebibliography}{99}", r"\section*{参考文献}")
    source = source.replace(r"\end{thebibliography}", "")
    source = re.sub(r"\\bibitem\{([^{}]+)\}", lambda m: f"\n\n[{bib.index(m[1])+1}] ", source)
    def number_captions(kind, label, content):
        counter = 0

        def caption(match):
            nonlocal counter
            counter += 1
            return match[0].replace(r"\caption{", rf"\caption{{{label}{counter}  ", 1)

        return re.sub(rf"\\begin\{{{kind}\}}.*?\\end\{{{kind}\}}", caption, content, flags=re.S)

    source = number_captions("table", "表", source)
    source = number_captions("figure", "图", source)
    source = source.replace(r"\quad ", "~")
    counters = {"section": -1, "subsection": 0}

    def headings(match):
        kind, name = match.groups()
        if kind == "section":
            counters["section"] += 1
            counters["subsection"] = 0
            number = str(counters["section"])
        else:
            counters["subsection"] += 1
            number = f'{counters["section"]}.{counters["subsection"]}'
        return rf"\{kind}*{{{number}  {name}}}"

    return re.sub(r"\\(section|subsection)\{([^{}]+)\}", headings, source)


def font_style(style, east_asian, size):
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asian)


def table_format(table):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    props = table._tbl.tblPr
    borders = props.find(qn("w:tblBorders"))
    if borders is not None:
        props.remove(borders)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single" if edge in ("top","bottom") else "nil")
        border.set(qn("w:sz"), "8")
        border.set(qn("w:color"), "000000")
        borders.append(border)
    props.append(borders)
    for index, row in enumerate(table.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if not index:
            row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for cell in row.cells:
            cell_props = cell._tc.get_or_add_tcPr()
            existing = cell_props.find(qn("w:tcBorders"))
            if existing is not None:
                cell_props.remove(existing)
            if not index:
                line = OxmlElement("w:tcBorders")
                bottom = OxmlElement("w:bottom")
                bottom.set(qn("w:val"), "single")
                bottom.set(qn("w:sz"), "4")
                line.append(bottom)
                cell_props.append(line)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.space_before = Pt(2)
                paragraph.paragraph_format.first_line_indent = Pt(0)
                paragraph.paragraph_format.keep_with_next = index < len(table.rows) - 1
                for run in paragraph.runs:
                    run.font.size = Pt(9)


def format_docx(path, anonymous, expected_equations=2, title="聚合电力系统扰动后最低频率的早期预测"):
    document = Document(path)
    styles = document.styles
    for name, face, size in (
        ("Normal","宋体",10.5), ("Body Text","宋体",10.5), ("First Paragraph","宋体",10.5),
        ("Title","黑体",18), ("Author","宋体",11), ("Heading 1","黑体",13),
        ("Heading 2","黑体",11.5), ("Caption","宋体",9.5), ("Image Caption","宋体",9.5),
        ("Table Caption","宋体",9.5),
    ):
        if name in styles:
            font_style(styles[name],face,size)
    for name in ("Normal","Body Text","First Paragraph"):
        p = styles[name].paragraph_format
        p.space_after = Pt(5)
        p.line_spacing = 1.16
    for name in ("Title","Author"):
        if name in styles:
            styles[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for section in document.sections:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.left_margin = section.right_margin = Cm(2.35)
        section.top_margin = section.bottom_margin = Cm(2.15)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        page = OxmlElement("w:fldSimple")
        page.set(qn("w:instr"), "PAGE")
        footer._p.append(page)
    equations = 0
    for paragraph in document.paragraphs:
        style = paragraph.style.name
        if anonymous:
            for run in paragraph.runs:
                for token in ("匿名占位","ANONYMOUSPLACEHOLDER"):
                    if token in run.text:
                        run.text = run.text.replace(token, "\u00a0")
        display_math = paragraph._p.find(qn("m:oMathPara"))
        if display_math is not None:
            equations += 1
            math = display_math.find(qn("m:oMath"))
            paragraph._p.remove(display_math)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            tabs = paragraph.paragraph_format.tab_stops
            tabs.add_tab_stop(Cm(8.15), WD_TAB_ALIGNMENT.CENTER)
            tabs.add_tab_stop(Cm(16.30), WD_TAB_ALIGNMENT.RIGHT)
            paragraph.add_run("\t")
            paragraph._p.append(math)
            paragraph.add_run(f"\t({equations})")
        if style in ("Table Caption","Image Caption","Caption","Author"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = style == "Table Caption"
        if paragraph._p.xpath(".//w:drawing"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = True
            for item in paragraph._p.xpath(".//wp:inline"):
                extent = item.find(qn("wp:extent"))
                if extent is not None and int(extent.get("cx")) > Cm(15.5):
                    factor = Cm(15.5) / int(extent.get("cx"))
                    for node in [extent, *item.xpath(".//a:xfrm/a:ext")]:
                        for dimension in ("cx","cy"):
                            node.set(dimension,str(round(int(node.get(dimension))*factor)))
        if paragraph.text.startswith(("Early Prediction", "HE Zhouyu")):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if re.match(r"^\[\d+\]",paragraph.text):
            paragraph.paragraph_format.keep_together = True
    assert equations == expected_equations
    for table in document.tables:
        table_format(table)
    document.core_properties.author = "" if anonymous else "何周俣"
    document.core_properties.last_modified_by = ""
    document.core_properties.title = title
    document.core_properties.comments = ""
    document.save(path)


def build_nadir():
    stem = "ARTICAL-nadir-restored"
    source = expanded_source(stem)
    with tempfile.TemporaryDirectory(prefix="nadir-word-") as temp:
        folder = Path(temp)
        for number, graphic in enumerate(re.findall(r"\\includegraphics\[[^\]]*\]\{([^{}]+\.pdf)\}",source),1):
            png = f"figure-{number}"
            subprocess.run(["pdftoppm","-singlefile","-r","180","-png",str(ART/graphic),str(folder/png)],check=True)
            source = source.replace(graphic, png+".png")
        intermediate = folder/"source.tex"
        intermediate.write_text(source,encoding="utf-8")
        output = ART/f"{stem}.docx"
        import os
        subprocess.run(["pandoc",str(intermediate),"-f","latex","-t","docx","--resource-path",
                        os.pathsep.join(map(str,(ART,folder))),"-o",str(output)],cwd=ART,check=True)
        title = re.search(r"\\title\{([^{}]+)\}",source)[1]
        format_docx(output,True,source.count(r"\begin{equation}"),title)
        doc = Document(output)
        assert len(doc.tables) == source.count(r"\begin{table}")
        assert len(doc.inline_shapes) == source.count(r"\includegraphics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nadir",action="store_true")
    if parser.parse_args().nadir:
        build_nadir()
        return
    source = expanded_source()
    with tempfile.TemporaryDirectory(prefix="nature-word-") as temp:
        for anonymous in (False, True):
            content = source
            suffix = "-anonymous" if anonymous else ""
            if anonymous:
                content = content.replace(r"\author{何周俣}", r"\author{匿名占位}")
                content = content.replace("HE Zhouyu", "ANONYMOUSPLACEHOLDER")
            intermediate = Path(temp) / "source.tex"
            intermediate.write_text(content, encoding="utf-8")
            out = ART / f"ARTICAL-revised{suffix}.docx"
            subprocess.run(["pandoc", str(intermediate), "-f", "latex", "-t", "docx",
                            "--resource-path", str(ART), "-o", str(out)], cwd=ART, check=True)
            format_docx(out, anonymous)
            document = Document(out)
            assert len(document.tables) == 7 and len(document.inline_shapes) == 2
            print(f"Built {out.name}: {len(document.tables)} tables, {len(document.inline_shapes)} figures.")


if __name__ == "__main__":
    main()
