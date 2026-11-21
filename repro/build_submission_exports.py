"""Build the editable submission copy and the matching supplement from local sources."""
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent.parent
ARTICLE = ROOT / "artical"
WORK = ROOT / "repro/out/submission-audit"


def build_figure():
    font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=10)
    data = pd.read_csv(ROOT / "repro/out/rolling/warning-summary.csv").query("Dataset=='aggregate' and Condition=='clean'")
    figure, axis = plt.subplots(figsize=(6.1, 3.25), layout="constrained")
    for policy, label, marker, color, line in (
        ("persistence_rolling", "持续性滚动", "o", "#222222", "--"),
        ("quadratic_rolling", "二次拟合滚动", "s", "#175b83", "-"),
        ("quadratic_fixed", "二次拟合固定时刻", "^", "#864421", ":"),
    ):
        rows = data[data.Policy.eq(policy)].sort_values("TargetFAR")
        axis.plot(100 * rows.FAR, 100 * rows.Hits / rows.Risk, color=color, marker=marker,
                  linestyle=line, linewidth=1.25, markersize=4, label=label)
    axis.set_xlabel("实际虚警率 / %", fontproperties=font)
    axis.set_ylabel("及时命中率 / %", fontproperties=font)
    axis.set_ylim(0, 103)
    axis.legend(loc="lower right", frameon=False, prop=font)
    axis.spines[["right", "top"]].set_visible(False)
    figure.savefig(WORK / "figure-1.pdf")
    plt.close(figure)


def build_word():
    source = (ARTICLE / "ARTICAL-submission.tex").read_text(encoding="utf-8")
    WORK.mkdir(parents=True, exist_ok=True)
    chart = "../repro/out/submission-audit/figure-1.pdf"
    subprocess.run(["pdftoppm", "-singlefile", "-r", "180", "-png", str(ARTICLE / chart),
                    str(WORK / "aggregate-tradeoff")], check=True)
    source = source.replace(chart, "../repro/out/submission-audit/aggregate-tradeoff.png")
    captions = {"table": 0, "figure": 0}

    def number_caption(match):
        kind = match[1]
        captions[kind] += 1
        label = "表" if kind == "table" else "图"
        return match[0].replace(r"\caption{", rf"\caption{{{label}{captions[kind]}　", 1)

    source = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", number_caption, source, flags=re.S)
    references = re.findall(r"\\bibitem\{([^}]+)\} ([^\r\n]+)", source)
    for number, (key, _) in enumerate(references, start=1):
        source = source.replace(r"\cite{" + key + "}", rf"\textsuperscript{{[{number}]}}")
    bibliography = "\n\n".join(f"[{i}] {entry}" for i, (_, entry) in enumerate(references, start=1))
    source = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                    lambda _: "\\section*{参考文献}\n" + bibliography, source, flags=re.S)
    word_tex = WORK / "submission-word.tex"
    word_tex.write_text(source, encoding="utf-8")
    subprocess.run(["pandoc", "--from=latex", "--to=docx", "--number-sections", str(word_tex),
                    "--output", str(ARTICLE / "ARTICAL-submission.docx")], cwd=ARTICLE, check=True)


def build_supplement():
    source = (ARTICLE / "ARTICAL-rolling-supplement.tex").read_text(encoding="utf-8")
    source = source.replace("主文表9", "主文表3").replace("按表9", "按表3")
    source = source.replace(r"\item \path{repro/README-rolling.md}",
                            "\\item \\path{repro/out/submission-audit/} 保存正式稿数字和时间轴核对结果。\n"
                            r"\item \path{repro/README-rolling.md}")
    (ARTICLE / "ARTICAL-submission-supplement.tex").write_text(source, encoding="utf-8")


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    build_figure()
    build_word()
    build_supplement()
