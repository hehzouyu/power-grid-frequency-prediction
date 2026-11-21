"""Assemble the original nadir manuscript and frozen ECGR/gate evidence."""
from pathlib import Path
from shutil import copy2, copytree
import hashlib
import re
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "artical/nadir-paired-v2"
QUALITY = ROOT / "artical/nadir-quality-integrated-v2"
DEST = ROOT / "artical/nadir-original-integrated-v3"
STEM = "ARTICAL-nadir-integrated-v3"
BASE = "ARTICAL-nadir-paired-v2"


def read(path):
    return path.read_text(encoding="utf-8-sig")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old[:90]}")
    return text.replace(old, new, 1)


def before(text, anchor, addition):
    return replace_once(text, anchor, addition + "\n\n" + anchor)


def extend_line(text, prefix, addition):
    matches = [row for row in text.splitlines() if row.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one paragraph starting: {prefix}")
    return replace_once(text, matches[0], matches[0] + addition)


def assemble_main(base, gate):
    method = gate[gate.index(r"\section{端点约束"):gate.index(r"\section{独立确认结果}")]
    method = method.replace(r"\section{端点约束协方差加权的弦线残差估计}",
                            r"\subsection{端点约束协方差加权的弦线残差估计}")
    for heading in ("零端点形状与工作协方差", "整窗估计和三次项收缩"):
        method = method.replace(r"\subsection{" + heading + "}", r"\subsubsection{" + heading + "}")
    method = method.replace(r"\section{可观测质量门控}", r"\subsection{可观测质量门控}")
    method = method.replace(
        r"以 \(f_i\) 表示频率观测，\(u_i=i/25\)。",
        r"为检验式\eqref{eq:chord-residual}中的窗内信息能否形成可用预测表示，构造端点约束的协方差加权弦线残差估计（ECGR）。本小节以 \(f_i\) 表示窗内第 \(i\) 个频率观测（\(f_0\) 在本小节专指窗首观测），\(u_i=i/25\)。")
    method = method.replace(
        r"记扰动前观测均值为",
        r"沿用式\eqref{eq:target}的截止后目标。五种固定点预测器均使用原0.50 s窗；门控另使用扰动前1 s的50点量测判断质量，这些数据仅供选择表示。记扰动前观测均值为")
    method = method.replace(
        "新参数与开发、训练、原近端点和ECGR确认档案逐项检查重合。",
        "最终确认样本与训练、开发、原独立配对和ECGR独立确认样本隔离。条件评价沿用式\\eqref{eq:pair-loss}的差值误差方向。")
    method = method.replace(
        "匹配构造和样本流转见补充材料。",
        "ECGR先在自己的独立确认集验证，使用干净及30 dB模拟白噪声；门控另用新的确认集。各批匹配构造和样本流转见补充材料。")
    result = gate[gate.index(r"\section{独立确认结果}"):gate.index(r"\section{讨论}")]
    result = result.replace(r"\section{独立确认结果}",
                            r"\section{ECGR与质量门控的独立确认}" + "\n" + r"\label{sec:quality-confirm}")
    result = result.replace(
        "ECGR独立确认由",
        "第\\ref{sec:pair-results}节检验原二次和分段表示。本节先检验ECGR整体估计，再用另一批场景检验质量门控。ECGR独立确认由")
    result = result.replace(
        "另一批确认集包含",
        "质量门控确认集包含")
    additions = (method + result).replace("{figures/", "{quality-assets/figures/")
    # The assembled sections retain original figure numbering via LaTeX counters.
    method, result = additions.split(r"\section{ECGR与质量门控的独立确认}", 1)
    result = r"\section{ECGR与质量门控的独立确认}" + result
    base = before(base, r"\input{assets/pair-metrics.tex}", r"\input{quality-assets/numbers.tex}")
    base = before(base, r"\subsection{预测区间的独立校准}", method.rstrip())
    base = before(base, r"\section{讨论与适用范围}", result.rstrip())
    abstracts = {
        r"\noindent\textbf{摘要：}": r"基于含调频限幅的二状态聚合模型，用扰动后前0.50 s的26点频率预测截止后最低频率。弦线残差和调频动力学解释了同端点可对应不同预测目标的原因。原独立近端点试验中，干净条件下二次与分段拟合的差值平均绝对误差较端点分别降低\CleanQuadGain{}和\CleanSegGain{} mHz；30 dB模拟白噪声下二次拟合幅值增益不稳定。另一批干净配对中，端点约束协方差加权弦线残差估计（ECGR）将差值误差由\QcleanMae{}降至\EcleanMae{} mHz，但未通过30 dB尾部保护。采用可观测量测质量选择表示后，独立确认中的等条件单场景与配对误差相对固定端点分别降低\Sgain{}\%和\Pgain{}\%，通过预设尾部保护；噪声突变时单场景误差略升。时序分组、窗口和结构迁移分析限定了短窗形状的仿真适用范围。",
        r"\noindent\textbf{Abstract.}": r"Using 26 frequency samples from the first 0.50 s after a disturbance, we predict the post-cutoff minimum frequency in a two-state aggregate model. Chord residuals explain why similar endpoints can precede different minima. In the original independent near-endpoint pairs, quadratic and segment fitting reduced clean pairwise mean absolute error (MAE) by \CleanQuadGain{} and \CleanSegGain{} mHz relative to the endpoint baseline. Quadratic fitting showed no stable MAE gain under 30 dB simulated white noise. In a separate confirmation, endpoint-constrained covariance-weighted chord residual estimation (ECGR) reduced clean pairwise MAE from \QcleanMae{} to \EcleanMae{} mHz but failed the 30 dB tail guard. On a further independent set, gating by observable measurement quality lowered equal-condition single-event and pairwise MAE by \Sgain{}\% and \Pgain{}\% relative to the fixed endpoint and met the prespecified tail guards. Single-event MAE increased slightly under a simulated noise jump.",
    }
    for prefix, paragraph in abstracts.items():
        original = next(line for line in base.splitlines() if line.startswith(prefix))
        base = replace_once(base, original, prefix+" "+paragraph)
    base = replace_once(base, "分段斜率；量测噪声", "分段斜率；量测质量门控")
    base = replace_once(base, "segment slope; measurement noise", "segment slope; quality gating")
    base = extend_line(base, "本文首先用严格的",
        " 进一步构造端点约束的残差估计与可观测量测质量门控，分别在两批独立样本中评价固定ECGR与按质量选择表示。扰动前量测只供门控使用，所有预测仍在同一截止时刻作出。")
    base = before(base, r"\section{结论}",
        r"""原固定特征的噪声和结构迁移结果与后续门控结果来自不同样本。干净ECGR的幅值收益与方向判对下降同时成立，且无法由整体预测比较分离单个基函数或物理参数的贡献。门控的优势是所设六条件等权下的平均幅值改善与尾部保护。30 dB下固定二次拟合仍有较低的单场景MAE，噪声突变中的误选也会带来额外损失。扰动前动态和相关噪声均可影响质量代理，表中结果限于所设聚合模型及模拟量测；门控本身的调频结构迁移尚未验证。""")
    base = before(base, r"\noindent\textbf{数据与代码可用性：}",
        r"""ECGR在干净独立配对中进一步降低了差值幅度误差，其30 dB尾部失败表明固定启用存在风险。基于可观测质量的门控保留了干净收益，在新确认集上相对固定端点降低等条件MAE并通过预设尾部保护。固定策略在部分条件下仍有较低的误差，噪声突变下门控可能略劣于端点。""")
    base = extend_line(base, r"\noindent\textbf{数据与代码可用性：}",
        " ECGR和质量门控另附两批确认的逐场景、逐配对输出及冻结配置。")
    base = replace_once(base, r"\usepackage[margin=25mm]{geometry}",
                        r"\usepackage[margin=25mm]{geometry}" + "\n" + r"\raggedbottom" + "\n" + r"\setlength{\emergencystretch}{2em}")
    base = re.sub(r"Fig\. \d+(?=\\quad)", lambda _: r"Fig. \thefigure", base)
    for old, english in (
        ("ECGR的估计流程。端点同时提供弦线约束和点预测输入。", "Estimation workflow of ECGR"),
        ("负值表示降低。", "Changes in pairwise MAE for ECGR with clustered confidence intervals"),
        ("虚线为两档冻结门槛。", "Observable quality, representation selection, and single-event errors"),
        ("上行为独立单场景，下行为配对差值。", "Errors of fixed and gated predictors by observation condition"),
    ):
        base = replace_once(base, old+"}", old+r"\\Fig. \thefigure\quad "+english+"}")
    return base


def assemble_supplement(base, gate):
    extra = gate[gate.index(r"\section{独立确认的样本流转"):gate.index(r"\end{document}")]
    extra = extra.replace(r"\section{独立确认的样本流转", r"\section{ECGR与门控的独立确认样本流转")
    extra = extra.replace("{figures/", "{quality-assets/figures/")
    for name in ("ecgr-table", "quality-table", "mis-table"):
        extra = extra.replace(r"\input{" + name + "}", r"\input{quality-assets/" + name + "}")
    for index in range(1, 4):
        extra = extra.replace(r"\captionofdummy{S" + str(index) + r"\quad",
                              r"\captionofdummy{附图" + str(index) + r"\quad")
    base = before(base, r"\begin{document}",
        r"\usepackage{longtable}" + "\n" + r"\input{quality-assets/numbers}" + "\n" +
        r"\newcommand{\captionofdummy}[1]{\par\smallskip\small #1}")
    return before(base, r"\end{document}", r"\clearpage" + "\n" + extra.rstrip())


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    original_main = ORIGINAL / (BASE + ".tex")
    frozen_hash = hashlib.sha256(original_main.read_bytes()).hexdigest()
    assert frozen_hash == "87edac8f9ff3fb75b0352e188073b464c4ebc8045a9607d0714552a8a281d164"
    for name in ("assets", "revision-assets", "pair-evidence"):
        copytree(ORIGINAL / name, DEST / name, dirs_exist_ok=True)
    for name in ("figures", "evidence"):
        copytree(QUALITY / name, DEST / "quality-assets" / name, dirs_exist_ok=True,
                 ignore=lambda _, names: [name for name in names if Path(name).suffix in {".log", ".aux", ".txt"}] if name == "figures" else [])
    for name in ("numbers.tex", "ecgr-table.tex", "quality-table.tex", "mis-table.tex"):
        copy2(QUALITY / name, DEST / "quality-assets" / name)
    copy2(ORIGINAL / "数字—源文件对照表.csv", DEST / "原稿数字追溯.csv")
    copy2(QUALITY / "数字追溯.csv", DEST / "ECGR与门控数字追溯.csv")
    for suffix, assembler, quality_source in (
        ("", assemble_main, "quality-integrated.tex"),
        ("-supplement", assemble_supplement, "quality-integrated-supplement.tex"),
    ):
        content = assembler(read(ORIGINAL / f"{BASE}{suffix}.tex"), read(QUALITY / quality_source))
        (DEST / f"{STEM}{suffix}.tex").write_text(content, encoding="utf-8")
    assert hashlib.sha256(original_main.read_bytes()).hexdigest() == frozen_hash


def package():
    archive = DEST / f"{STEM}-LaTeX.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=6) as output:
        for path in DEST.rglob("*"):
            if not path.is_file() or "render" in path.relative_to(DEST).parts:
                continue
            if path.suffix.lower() not in {".tex", ".pdf", ".png", ".jpg", ".csv", ".md", ".json", ".log"}:
                continue
            if path.suffix.lower() == ".log" and "evidence" not in path.parts:
                continue
            output.write(path, path.relative_to(DEST))
        output.write(Path(__file__), "repro/build_original_nadir_integrated.py")
    with ZipFile(archive) as output:
        assert output.testzip() is None
        names = set(output.namelist())
        for stem in (STEM, STEM + "-supplement"):
            src = output.read(stem+".tex").decode("utf-8")
            for path in re.findall(r"\\(?:input|resultrows|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", src):
                assert path in names or path+".tex" in names, path
        print(f"ZIP PASS: {len(names)} files")


if __name__ == "__main__":
    import sys
    package() if sys.argv[1:] == ["--package"] else build()
