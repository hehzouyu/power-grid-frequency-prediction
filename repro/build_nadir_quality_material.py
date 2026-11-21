"""Render the passed quality-gate study as a separate LaTeX manuscript insert."""
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"repro/out/nadir-quality/confirm"
OUT=ROOT/"artical/nadir-quality-method"
CURRENT=ROOT/"artical/nadir-paired-v2/ARTICAL-nadir-paired-v2.tex"
REPORT=ROOT/"reports/nadir-quality-method.md"
BASELINE="87edac8f9ff3fb75b0352e188073b464c4ebc8045a9607d0714552a8a281d164"


def compile_material():
    assert hashlib.sha256(CURRENT.read_bytes()).hexdigest()==BASELINE
    with (RUN/"acceptance.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==20 and all(row["Pass"]=="True" for row in rows)
    log=(RUN/"readback.log").read_text(encoding="utf-8-sig",errors="replace")
    assert "PASS five fixed strategies" in log
    (OUT/"figures").mkdir(parents=True,exist_ok=True)
    (OUT/"evidence").mkdir(exist_ok=True)
    copy=["metrics.csv","main-contrasts.csv","mis-selection.csv","selection-confusion.csv",
          "pair-selection-confusion.csv","acceptance.csv","status-counts.csv","readback.log",
          "worst-single-jump.csv","worst-pair-jump.csv"]
    for name in copy:
        shutil.copyfile(RUN/name,OUT/"evidence"/name)
    shutil.copyfile(RUN.parent/"frozen-selector.json",OUT/"evidence/frozen-selector.json")
    header=r"""\documentclass[UTF8,a4paper,11pt]{ctexart}
\usepackage[margin=27mm]{geometry}
\usepackage{amsmath,graphicx,booktabs}
\usepackage[hidelinks]{hyperref}
\title{量测质量条件化的短窗形状表示选择\\截止后最低频率预测的方法增强材料}
\date{}
\author{}
\begin{document}
\maketitle
"""
    report=REPORT.read_text(encoding="utf-8-sig")
    body=subprocess.run(["pandoc","--from=markdown+tex_math_single_backslash","--shift-heading-level-by=-1","--to=latex"],
        input="\n".join(report.splitlines()[1:]),encoding="utf-8",capture_output=True,check=True).stdout
    body=body[:body.find(r"\section{裁定与适用范围}")] if r"\section{裁定与适用范围}" in body else body
    # Keep 12 principal paired/point cells in the manuscript; the complete
    # five-baseline MAE/RMSE/P95 matrix is attached as metrics.csv.
    with (RUN/"metrics.csv").open() as f:
        m={(x["Kind"],x["Condition"],x["Feature"]):x for x in csv.DictReader(f)}
    body+=r"""
\section{结论与适用范围}
量测质量门控在同一冻结估计器池中保留了干净条件下ECGR的幅值收益，并在所测30 dB模拟白噪声下通过配对尾部保护。相对开发集锁定的跨条件固定策略，两类预测的等条件平均绝对误差均下降。白噪声及相关噪声下的单场景结果和突变下的误选说明，不同评价目标对应不同的特征收益。本结论限于二状态聚合仿真、预设扰动及模拟观测条件。干净仿真的扰动前基线为平坦稳态，结论的工程适用性仍待独立验证。
\begin{center}
\small
\begin{tabular}{llrrrr}
\toprule
任务 & 观测条件 & 门控MAE & 端点MAE & 门控P95 & 端点P95\\
\midrule
"""
    names={"clean":"干净","white40":"白噪声40 dB","white30":"白噪声30 dB",
           "white20":"白噪声20 dB","ar30":"AR(1) 30 dB","jump":"扰动时突变"}
    for kind in ("single","pair"):
        for key,name in names.items():
            numbers=[float(m[(kind,key,model)][field]) for field in ("MAE_mHz","P95_mHz") for model in ("gate","endpoint")]
            body+=("单场景" if kind=="single" else "配对差值")+" & "+name+" & "+" & ".join(f"{v:.2f}" for v in numbers)+r"\\"+"\n"
    body+=r"""\bottomrule
\end{tabular}
\par\smallskip 表中误差单位均为mHz。完整五基线及门控对照、RMSE和方向指标见随附证据表。
\end{center}
"""
    figs={"quality-selection-performance":"可观测质量、所选表示和单场景预测误差。两条虚线为已冻结的门槛。",
          "quality-strategy-errors":"五种固定策略与质量门控的单场景及配对MAE、RMSE与P95。",
          "quality-misselection-matrix":"实际选择与事后最小幅值误差配置的混淆矩阵，数值为判断次数。",
          "quality-misselection-loss":"相对不可在线实现的事后选择的误选比例与平均损失。"}
    for name,caption in figs.items():
        shutil.copyfile(RUN/f"{name}.pdf",OUT/"figures"/f"{name}.pdf")
        body+=("\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/"+name+".pdf}\n"
                +"\\caption{"+caption+"}\n\\end{figure}\n")
    tex=OUT/"nadir-quality-enhancement.tex"
    tex.write_text(header+body+"\n\\end{document}\n",encoding="utf-8")
    with ZipFile(OUT/"nadir-quality-LaTeX.zip","w",ZIP_DEFLATED) as zipfile:
        for f in [tex,*sorted((OUT/"figures").glob("*")),*sorted((OUT/"evidence").glob("*"))]:
            zipfile.write(f,f.relative_to(OUT))


if __name__=="__main__":
    compile_material()
