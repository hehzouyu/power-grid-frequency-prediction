"""Build the self-contained paired-test paper edition from the audited outputs."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artical"
OUT = ART / "nadir-paired-v2"
RUN = ROOT / "repro/out/nadir-pair-tests/independent-v1"
MAIN = ART / "ARTICAL-nadir-restored.tex"
SUPP = ART / "ARTICAL-nadir-supplement.tex"
NAMES = {"endpoint":"端点","initial_rate":"端点＋初始变化率",
         "quadratic":"端点＋二次拟合","fitted_segments":"端点＋分段拟合"}
CONDITIONS = {"clean":"干净","white30":"白噪声"}
BASE = {"endpoint":"端点","initial_rate":"端点＋初始变化率"}
METRICS = {}
AUDIT = []


def table(name):
    with (RUN / name).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def single(rows, **keys):
    (row,) = [row for row in rows if all(row[k] == str(v) for k,v in keys.items())]
    return row


def put(name, value, file, key, fields, rule, fmt):
    shown = format(float(value),fmt)
    METRICS[name] = shown
    AUDIT.append([name, shown, file, key, fields, rule])
    return shown


def fields(name, row, file, key, columns, kind=".2f", scale=1):
    return [put(name+field, float(row[field])*scale, file, key, field,
                "原值" if scale==1 else f"原值乘 {scale}",kind) for field in columns]


def interval(row, name, key):
    point, lo, hi = fields(name,row,"paired-contrasts-verified.csv",key,
                           ("ChangeMAE_mHz","Low95_mHz","High95_mHz"),"+.2f")
    return r"$"+point+r"\ ["+lo+r",\, "+hi+r"]$"


def replace_once(text, old, new):
    if text.count(old)!=1:
        raise ValueError(f"Expected one source anchor: {old[:100]}")
    return text.replace(old,new,1)


def paragraph(text, prefix, replacement):
    pattern = "(?m)^"+re.escape(prefix)+r".*"
    text,count = re.subn(pattern,lambda _: replacement,text,count=1)
    if count!=1:
        raise ValueError(prefix)
    return text


def copy_images(text):
    def convert(match):
        path = match.group(1)
        source = (ART / path).resolve()
        destination = OUT / "assets" / source.name
        if destination.exists() and destination.read_bytes()!=source.read_bytes():
            raise ValueError(f"Asset collision: {source}")
        shutil.copyfile(source,destination)
        return "{assets/"+source.name+"}"
    return re.sub(r"(?<=\])\{([^{}]+\.(?:png|pdf))\}",convert,text)


def data():
    assert "PASS exact seeded reference-cluster bootstrap" in (RUN/"verification.log").read_text(encoding="utf-8-sig")
    assert (RUN/"paper-sections.md").is_file()
    reference_text = (RUN/"paper-sections.md").read_text(encoding="utf-8-sig")
    assert all(term in reference_text for term in ("1 325/1 920","−4.89","−3.80","41.35"))
    flow_file = "sample-flow-verified.csv"
    flow = table(flow_file)
    total = single(flow,RBand="all",ChangedFactor="all")
    fields("Pair",total,flow_file,"RBand=all; ChangedFactor=all",
           ("Attempted","GeometricMatches","PrimaryPairs","TrueTies","SmallNonTies","MatchFailures","FeatureFailures"),".0f")
    fields("Pair",total,flow_file,"RBand=all; ChangedFactor=all",("MatchFailureRate",),".1f",100)
    fields("PairLow",single(flow,RBand=1,ChangedFactor="all"),flow_file,"RBand=1; ChangedFactor=all",("PrimaryPairs",),".0f")
    fields("PairHigh",single(flow,RBand=2,ChangedFactor="all"),flow_file,"RBand=2; ChangedFactor=all",("PrimaryPairs",),".0f")
    ledger = table("pair-index.csv")
    assert len(ledger)==int(total["Attempted"]) and all(r["EventA"]==r["EventB"]=="after" for r in ledger)
    gap = max(abs(float(r["EndpointA_Hz"])-float(r["EndpointB_Hz"])) for r in ledger)
    put("PairMaxEndMantissa", gap*1e10,"pair-index.csv","全部配对","EndpointA_Hz; EndpointB_Hz",
        "最大绝对差乘 10^10，原单位 Hz",".2f")
    put("PairReferences",len(table("references.csv")),"references.csv","全部参考事件","ReferenceID","参考事件行数",".0f")

    metrics = table("main-results-verified.csv")
    contrasts = table("paired-contrasts-verified.csv")
    out_rows, effect_rows = [], []
    for cond, tag in (("clean","Clean"),("white30","White")):
        for feature, short in zip(NAMES,("End","Rate","Quad","Seg"),strict=True):
            key = f"Condition={cond}; Feature={feature}"
            row = single(metrics,Condition=cond,Feature=feature)
            vals = fields(tag+short,row,"main-results-verified.csv",key,
                          ("DeltaMAE_mHz","DeltaRMSE_mHz","DeltaP95_mHz"))
            counts = fields(tag+short,row,"main-results-verified.csv",key,
                            ("CorrectDirections","DirectionTrials","PredictedTieTrials","TrueTieTrials"),".0f")
            fields(tag+short,row,"main-results-verified.csv",key,("DirectionAccuracy",),".1f",100)
            out_rows.append(" & ".join([CONDITIONS[cond],NAMES[feature],*vals,counts[0]+"/"+counts[1],*counts[2:]])+r"\\")
            if feature=="endpoint":
                continue
            for comparator in (("endpoint",) if feature=="initial_rate" else ("endpoint","initial_rate")):
                effect = single(contrasts,Condition=cond,Feature=feature,Comparator=comparator)
                kind = "VsEndpoint" if comparator=="endpoint" else "VsRate"
                effect_rows.append(" & ".join([CONDITIONS[cond],NAMES[feature],BASE[comparator],
                    interval(effect,tag+short+kind,key+"; Comparator="+comparator)])+r"\\")
    (OUT/"assets/pair-summary-rows.tex").write_text("\n".join(out_rows)+"\n",encoding="utf-8")
    (OUT/"assets/pair-effect-rows.tex").write_text("\n".join(effect_rows)+"\n",encoding="utf-8")

    band_rows = []
    for row in table("results-by-r-band.csv"):
        band, cond, feat = (row[k] for k in ("RBand","Condition","Feature"))
        key = f"RBand={band}; Condition={cond}; Feature={feat}"
        vals = [format(float(row[col]),".2f") for col in
                ("DeltaMAE_mHz","DeltaRMSE_mHz","DeltaP95_mHz","ChangeVsEndpoint_mHz")]
        band_rows.append(" & ".join([band,CONDITIONS[cond],NAMES[feat],row["PrimaryPairs"],*vals,
                                   row["CorrectDirections"]+"/"+row["DirectionTrials"],row["PredictedTieTrials"]])+r"\\")
        AUDIT.append(["补充材料分带表","；".join(vals),"results-by-r-band.csv",key,
                      "PrimaryPairs; DeltaMAE_mHz; DeltaRMSE_mHz; DeltaP95_mHz; ChangeVsEndpoint_mHz; CorrectDirections; DirectionTrials; PredictedTieTrials",
                      "每行按 CSV 字段取值；含噪方向判断数含复本"])
    (OUT/"assets/pair-band-rows.tex").write_text("\n".join(band_rows)+"\n",encoding="utf-8")
    flow_rows = []
    for row in flow:
        b,f = row["RBand"],row["ChangedFactor"]
        cols = ("Attempted","GeometricMatches","PrimaryPairs","TrueTies","SmallNonTies","MatchFailures","FeatureFailures")
        flow_rows.append(" & ".join([{"all":"汇总"}.get(b,b),{"all":"汇总","Tg":"$T_g$"}.get(f,f),*(row[k] for k in cols)])+r"\\")
        AUDIT.append(["补充材料流转表","/".join(row[k] for k in cols),flow_file,f"RBand={b}; ChangedFactor={f}",
                      "; ".join(cols),"直接列示，失败分原因另见附带账本"])
    (OUT/"assets/pair-flow-rows.tex").write_text("\n".join(flow_rows)+"\n",encoding="utf-8")
    for file in ("fig-pair-deltas.pdf","fig-paired-contrasts.pdf"):
        shutil.copyfile(RUN/file,OUT/"assets"/file)
        AUDIT.append([file,"正式图原样复制","pair-predictions.csv" if "deltas" in file else "paired-contrasts-verified.csv",
                      "全部模型和条件","真值差、预测差、配对状态或 MAE 变化与区间","来自已核验正式输出"])
    protocol = (RUN/"protocol.json").read_bytes()
    assert hashlib.sha256(protocol).hexdigest() == "7f9cb24f3da243ee5915ff0412a4c670e93c0f788e8a25536d7a521a6f048b44"
    settings = json.loads(protocol)
    for key in ("r_bands","references_per_r_band","candidate_factors","candidate_multipliers","deficit_bracket",
                "cutoff_relative_s","step_s","disturbance_s","end_s","endpoint_tolerance_hz",
                "tie_hz","minimum_target_gap_hz","white_snr_db","noise_repeats","bootstrap_draws"):
        AUDIT.append(["方法预设 "+key,str(settings[key]),"protocol.json",key,key,
                      "冻结协议；Hz 阈值乘 1000 转为 mHz"])
    original_flows = ("main-results-verified.csv","paired-contrasts-verified.csv","results-by-r-band.csv",
                      "sample-flow-verified.csv","failure-reasons.csv","pair-index.csv","excluded-and-failed.csv",
                      "pair-predictions.csv","references.csv","protocol.json","freeze.json","verification.log",
                      "historical-evidence-audit.log","anchor-check.csv")
    for name in original_flows:
        shutil.copyfile(RUN/name,OUT/"pair-evidence"/name)
    lines = [r"% Generated directly from the formal paired-test CSV files."]
    for name,value in METRICS.items():
        if re.fullmatch("[A-Za-z]+",name):
            lines.append("\\newcommand{\\%s}{%s}" % (name,value))
    (OUT/"assets/pair-metrics.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    with (OUT/"数字—源文件对照表.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["稿件变量或位置","显示值","正式源文件（pair-evidence）","筛选键","字段","计算与单位"])
        writer.writerows(AUDIT)


def main_source():
    tex = MAIN.read_text(encoding="utf-8")
    tex = replace_once(tex,r"\input{revision-assets/metrics.tex}",
                       r"\input{revision-assets/metrics.tex}"+"\n"+r"\input{assets/pair-metrics.tex}")
    tex = replace_once(tex,r"\usepackage{caption}",r"\usepackage{caption,placeins}")
    tex = paragraph(tex,r"\noindent\textbf{摘要：}",r"""\noindent\textbf{摘要：}基于含调频限幅的二状态聚合模型，仅用扰动后前0.50 s的26个频率样本预测截止后的最低频率。扣除端点弦线后，窗内残差仍可反映阻尼和调频动态。在\PairGeometricMatches{}个独立近端点配对中，\PairPrimaryPairs{}对进入差值幅度评价。干净条件下，二次拟合与分段拟合相对端点分别降低配对差值平均绝对误差\CleanQuadVsEndpointChangeMAE_mHz{}和\CleanSegVsEndpointChangeMAE_mHz{} mHz的绝对量，两种表示也优于端点加初始变化率。30 dB模拟白噪声下，二次拟合的方向判对率提高，但相对端点的差值平均绝对误差变化区间跨零，均方根误差和第95百分位误差升高，分段拟合及初始变化率配置的平均绝对误差增加。局部灵敏度、边界事件和调频结构迁移结果进一步限定了短窗形状的适用范围。""")
    tex = paragraph(tex,r"\noindent\textbf{Abstract.}",r"""\noindent\textbf{Abstract.} A two-state aggregate model with limited primary response is used to predict the minimum frequency after a fixed cutoff from 26 samples in the first 0.50 s after a disturbance. Subtracting the endpoint chord exposes within-window shape associated with damping and primary frequency response. Of \PairGeometricMatches{} independent near-endpoint pairs, \PairPrimaryPairs{} met the criterion for magnitude evaluation. With clean observations, quadratic fitting and segment fitting both reduced mean absolute error (MAE) of predicted pairwise differences relative to the endpoint-only and endpoint-plus-initial-rate baselines. The reductions relative to the endpoint were \CleanQuadGain{} and \CleanSegGain{} mHz. At 30 dB simulated white noise, quadratic fitting improved direction accuracy, but the MAE change interval crossed zero and both root mean square error and the 95th percentile absolute error increased relative to the endpoint. Segment fitting and the initial-rate configuration increased MAE. Local sensitivity, boundary events, and transfer between governor structures delimit the findings under aggregate simulation.""")
    tex = replace_once(tex,r"分别降低配对差值平均绝对误差\CleanQuadVsEndpointChangeMAE_mHz{}和\CleanSegVsEndpointChangeMAE_mHz{} mHz的绝对量",
                       r"分别降低配对差值平均绝对误差\CleanQuadGain{}和\CleanSegGain{} mHz")
    tex = paragraph(tex,"本文首先定义",r"""本文首先用严格的0.50 s信息截止和截止后最低频率标签分离输入与预测目标，并按全局最低点时序评价边界事件。其次，通过端点冗余关系、弦线残差和调频动力学解释同一端点为何可以对应不同的后续最低频率。冻结原模型后，在独立近端点配对中检验二次拟合和分段拟合相对端点及端点加初始变化率的差值预测收益。最后，用配对噪声、惯量外推和调频结构变化考察点预测和预测区间的适用边界。""")
    tex = replace_once(tex,r"\subsection{端点相同条件下的形状差异}",r"\subsection{端点信息不足与弦线残差}")
    tex = replace_once(tex,r"有限窗口的二次系数还包含更高阶动态。在区间$[a,b]$对摆动方程积分可得",r"""令$\tau$为窗口长度，定义标幺频偏的弦线残差$r_\tau(t)=x(t)-(t/\tau)x(\tau)$。在包含所考察短窗的线性调频段内，由扰动右侧的局部展开得到
\begin{equation}
\begin{aligned}
r_\tau(t)={}&\tfrac{1}{2}\ddot x(0^+)(t^2-t\tau)
 +\tfrac{1}{6}x^{(3)}(0^+)(t^3-t\tau^2)+O(\tau^4),\qquad 0\le t\le\tau .
\end{aligned}
\label{eq:chord-residual}
\end{equation}
初始一阶项在弦线残差中消去，阻尼和一阶调频滞后的影响仍在。式\eqref{eq:chord-residual}是短时局部展开，有限窗的二次拟合系数受整个窗口及高阶项影响。本文用它解释形状来源，预测模型沿用原有频率特征。在区间$[a,b]$对摆动方程积分可得""")
    tex = replace_once(tex,r"\subsection{扰动与泛化评估}",r"""\subsection{独立近端点配对协议}
\label{sec:pair-design}
在原训练网格的下垂系数取值之外，预设$R$带$[0.041,0.044]$和$[0.056,0.059]$，每带抽取32个参考场景，各参数在预设范围内独立均匀抽样。每个参考场景分别将$H$、$D$或$T_g$乘以0.8、1.5或1.5，各形成一个候选。每对保留相同的$R$和其余未干预参数，仅在$[0.02,0.08]$ p.u.内调整候选缺额，使干净轨迹的绝对端点差不超过0.10 mHz。原网格训练组拟合标准化器和提升树，验证与历史测试维持原划分。归档近端点扫描只用于开发，正式配对使用独立种子，并检查与历史网格、连续确认、区间试验及开发扫描的参数重合。模型和规则在配对生成前冻结。各候选及其失败原因逐一登记，失败候选不补抽。

四种输入依次为端点、端点＋初始变化率、端点＋二次拟合、端点＋分段拟合。所有配置使用式\eqref{eq:target}及同一0.50 s输入窗。以$B$为候选、$A$为参考，定义
\begin{equation}
\Delta y=y_B-y_A,\qquad \Delta\hat y=\hat y_B-\hat y_A,\qquad
E_\Delta=|\Delta\hat y-\Delta y|.
\label{eq:pair-loss}
\end{equation}
幅度评价纳入可推理且$|\Delta y|\ge 2.00$ mHz的配对，报告$E_\Delta$的平均值（MAE）、均方根值（RMSE）和第95百分位数（P95）。方向评价纳入全部可推理匹配，真值与预测的绝对差各自在0.50 mHz以内或恰为0.50 mHz时计为并列，候选较低、并列、参考较低三个类别一致才计正确。

干净输入为主试验。同批配对另施加30 dB模拟高斯白噪声，每对10个复本。噪声标准差按各成员的干净窗频偏均方根确定。同一成员输入供四模型共用，同一参考的噪声在其候选间复用，两成员噪声独立。MAE变化按待评估模型减基线模型计算。按两个$R$带分层，以参考事件为簇有放回重采样5\,000次，同一参考的候选与复本一起抽取，取2.5和97.5百分位数作为95\%区间。

\subsection{扰动与泛化评估}""")
    tex = paragraph(tex,r"表\ref{tab:windows}保留",r"""在固定0.50 s之后的标签和232个条件测试事件上，图\ref{fig:matrix}对比各窗相对同窗端点的MAE降幅，表\ref{tab:windows}列出端点模型的绝对误差。名义0.25 s窗的实际末样本为0.24 s。0.10 s窗按2、2、1个样本间隔分段，末段仅含两点。""")
    tex = paragraph(tex,"为比较窗口之间",r"""同一事件内比较各窗形状增益，再按事件重采样。由0.10 s延长至0.50 s时，二次拟合的MAE降幅增加8.2 mHz，95\%区间为$[5.2,11.2]$ mHz；由0.10 s延长至0.24 s的区间跨零。分段配置的跨窗结果见图\ref{fig:matrix}，各窗的分段长度和样本数随窗长变化。""")
    tex = replace_once(tex,r"\subsection{边界事件与示例风险阈值}",r"""\subsection{独立近端点配对的幅度与方向结果}
\label{sec:pair-results}
\PairReferences{}个参考场景生成\PairAttempted{}个候选，全部匹配，最大绝对端点差为$\PairMaxEndMantissa\times10^{-10}$ Hz。幅度分析纳入\PairPrimaryPairs{}对，\PairTrueTies{}个真实并列及\PairSmallNonTies{}个小差非并列对全部参加方向评价。所有正式配对成员的全局最低点都在截止后形成。分参数带与排除账本见补充材料。

干净条件下，端点、二次拟合和分段拟合的差值MAE分别为\CleanEndMae{}、\CleanQuadMae{}和\CleanSegMae{} mHz。相对端点，二次拟合和分段拟合的MAE变化分别为\CleanQuadVsEnd{}和\CleanSegVsEnd{} mHz；相对端点＋初始变化率，两者分别为\CleanQuadVsRate{}和\CleanSegVsRate{} mHz。括号内为95\%区间。仅加入初始变化率的变化为\CleanRateVsEnd{} mHz，区间跨零。端点基线把所有干净配对判为并列，只在真实并列时判对，体现了近同端点设计下该基线的行为。

30 dB模拟白噪声下，相对端点，二次拟合的MAE变化为\WhiteQuadVsEnd{} mHz，区间跨零；分段拟合和初始变化率配置分别变化\WhiteSegVsEnd{}和\WhiteRateVsEnd{} mHz，均为增加。二次拟合的差值RMSE和P95为\WhiteQuadRmse{}、\WhiteQuadTail{} mHz，高于端点的\WhiteEndRmse{}、\WhiteEndTail{} mHz。二次拟合与端点的方向判对率分别为\WhiteQuadAcc{}\%和\WhiteEndAcc{}\%，方向判断的提高未伴随差值幅度误差的明确改善。干净与白噪声的方向评价分母分别为\CleanDirections{}和\WhiteDirections{}，白噪声分母按复本计。

\begin{figure}[!htbp]
\centering
\includegraphics[width=.92\linewidth]{assets/fig-paired-contrasts.pdf}
\caption{近端点配对的差值MAE变化。左侧以端点为基线，右侧以端点＋初始变化率为基线。负值表示待评估模型误差下降，线段为按下垂系数带分层、以参考事件为簇的95\%重采样区间。\\
Fig. \thefigure\quad Changes in MAE of pairwise differences relative to the two baselines}
\label{fig:pair-effects}
\end{figure}
\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{assets/fig-pair-deltas.pdf}
\caption{近端点配对的最低频率预测差与真实差。横纵轴均为候选减参考。彩点为主幅度评价配对，灰点为其余可推理匹配。淡色横纵带为$\pm0.50$ mHz并列区，虚线为等值线。上排为干净量测，下排为30 dB模拟白噪声下的逐复本结果。\\
Fig. \thefigure\quad Predicted versus true nadir differences for near-endpoint pairs}
\label{fig:pair-deltas}
\end{figure}

图\ref{fig:pair-deltas}给出真值差与预测差的分布，图\ref{fig:pair-effects}汇总相对两条基线的差值MAE变化。
\subsection{边界事件与示例风险阈值}""")
    tex = replace_once(tex,r"\subsection{量测噪声的影响}",r"\FloatBarrier"+"\n\n"+r"\subsection{量测噪声的影响}")
    # Preserve the repeated-split evidence as an appendix table.
    start = tex.index(r"表\ref{tab:repeat}汇总")
    end = tex.index(r"\subsection{预测区间的覆盖与宽度}",start)
    tex = tex[:start] + "十次随机划分结果见补充材料，各划分复用同一原始网格。\n\n" + tex[end:]
    tex = tex.replace(r"\\Table 8\quad Prediction intervals",r"\\Table \thetable\quad Prediction intervals")
    tex = paragraph(tex,r"表\ref{tab:general}给出",r"""表\ref{tab:general}分别报告连续参数抽样、惯量留出及两级调速器结构测试，三者对应新参数组合、惯量范围外推和结构变化。单级与两级的同参场景配对用于量化结构变化的影响。""")
    tex = tex.replace("二次拟合在30至40 dB下误差最低，20 dB下端点误差最低。","在原条件网格的单场景RMSE中，二次拟合在30至40 dB下最低，20 dB下端点最低。该网格结果采用单场景误差，近端点试验采用配对差值误差。")
    start = tex.index(r"\section{适用范围}")
    end = tex.index(r"\begin{thebibliography}{99}",start)
    tex = tex[:start] + r"""\section{讨论与适用范围}
独立配对结果表明，二次拟合与分段拟合的整体表示在干净条件下提供了超出端点以及所测初始变化率配置的预测信息。有限窗二次系数反映整窗动态，参数缩放等价性仍限定各原始参数的单独反演。正式配对均属于截止后形成最低点的场景，主幅度集合以惯量和调频时间常数变化为主。边界事件仍由原扩展测试评价。30 dB下的方向判对和幅度误差反映不同的性能维度。

本文的证据限于所设二状态聚合模型、参数范围和离线数值仿真条件。点预测结果和区间覆盖均随噪声与调频结构变化。若将最低频率估计用于时序告警，需另行定义首次越限前的决策时效和虚警评价。多机迁移仍需沿用相同截止与标签，并检查局部量测和系统参考频率之间的差异。

\section{结论}
固定扰动后0.50 s信息截止并只预测其后的最低频率时，相同端点可对应不同预测目标。弦线残差和分段功率平衡为这种差异提供物理解释。独立近端点配对中，干净曲线的二次拟合和分段拟合均降低差值MAE，比较基线包括端点和端点＋初始变化率。

30 dB模拟白噪声下，二次拟合相对端点的MAE变化区间跨零，RMSE与P95升高。分段拟合和初始变化率配置的MAE增加。原单场景噪声对照及两级调频迁移结果进一步表明，短窗特征的收益与预测区间的覆盖需要按噪声、评价指标和调频结构分别解释。

\noindent\textbf{数据与代码可用性：}轨迹及特征由MATLAB生成，冻结模型由scikit-learn运行。随稿材料包含场景索引、逐场景预测、正式近端点配对账本及逐对预测、分参数带结果、灵敏度检查以及区间校准记录。

""" + tex[end:]
    tex = tex.replace(r"\setlength{\itemsep}{0pt}",r"\setlength{\itemsep}{0pt}"+"\n"+r"\interlinepenalty=10000")
    tex = copy_images(tex.replace("{assets/fig-pair-deltas.pdf}","{../repro/out/nadir-pair-tests/independent-v1/fig-pair-deltas.pdf}").replace(
        "{assets/fig-paired-contrasts.pdf}","{../repro/out/nadir-pair-tests/independent-v1/fig-paired-contrasts.pdf}"))
    (OUT/"ARTICAL-nadir-paired-v2.tex").write_text(tex,encoding="utf-8")


def add_aliases():
    names = {}
    for label,typ in (("CleanEnd","CleanEnd"),("CleanQuad","CleanQuad"),("CleanSeg","CleanSeg"),
                      ("WhiteQuad","WhiteQuad"),("WhiteEnd","WhiteEnd")):
        for short, key in (("Mae","DeltaMAE_mHz"),("Rmse","DeltaRMSE_mHz"),("Tail","DeltaP95_mHz")):
            names[label+short] = METRICS[typ+key]
    names.update(CleanDirections=METRICS["CleanEndDirectionTrials"],WhiteDirections=METRICS["WhiteEndDirectionTrials"],
                 CleanQuadGain=METRICS["CleanQuadVsEndpointChangeMAE_mHz"].lstrip("-"),
                 CleanSegGain=METRICS["CleanSegVsEndpointChangeMAE_mHz"].lstrip("-"),
                 WhiteQuadAcc=METRICS["WhiteQuadDirectionAccuracy"],WhiteEndAcc=METRICS["WhiteEndDirectionAccuracy"])
    for condition in ("Clean","White"):
        for feature in ("Rate","Quad","Seg"):
            for suffix, comparator in (("VsEnd","VsEndpoint"),("VsRate","VsRate")):
                key = condition+feature+comparator
                if key+"ChangeMAE_mHz" not in METRICS:
                    continue
                names[condition+feature+suffix] = ("$"+METRICS[key+"ChangeMAE_mHz"]+r"\ ["+
                        METRICS[key+"Low95_mHz"]+r",\, "+METRICS[key+"High95_mHz"]+" ]$")
    with (OUT/"assets/pair-metrics.tex").open("a",encoding="utf-8") as stream:
        for name,value in names.items():
            stream.write("\\newcommand{\\%s}{%s}\n" % (name,value))


def supplement_source():
    tex = SUPP.read_text(encoding="utf-8")
    tex = replace_once(tex,r"\usepackage{amsmath,graphicx,booktabs}",r"""\usepackage{amsmath,graphicx,booktabs,caption}
\captionsetup{font=small,hypcap=false}
\makeatletter
\newcommand{\resultrows}[1]{\@@input #1 }
\makeatother
\renewcommand{\thetable}{S\arabic{table}}
\input{assets/pair-metrics.tex}""")
    tex = replace_once(tex,r"\section{复现入口}",r"""\clearpage
\section{独立近端点配对的完整结果}
\subsection{筛选流转与失败账本}
$R$带1为$[0.041,0.044]$，带2为$[0.056,0.059]$。表\ref{tab:s-flow}中，“主幅度”指可推理且真实绝对差不小于2.00 mHz的配对，“小差”指高于0.50 mHz且低于2.00 mHz的非并列对。真实并列和小差配对均参加方向评价。正式试验的几何匹配失败率为\PairMatchFailureRate{}\%。缺额无法括根、求根失败、仿真失败、超端点容差、与历史参数重合以及特征失败的计数均列在随附原因表和逐对账本中。

\begin{center}
\small
\captionof{table}{近端点配对的样本流转}
\label{tab:s-flow}
\begin{tabular}{llrrrrrrr}
\toprule
$R$带 & 因素 & 尝试 & 匹配 & 主幅度 & 真并列 & 小差 & 匹配失败 & 特征失败\\
\midrule
\resultrows{assets/pair-flow-rows.tex}
\bottomrule
\end{tabular}
\end{center}

\subsection{两种条件下的差值与方向评价}
表\ref{tab:s-metrics}的误差均按主幅度集合计算，方向分母纳入全部可推理配对。白噪声行按每对10个复本计算。“真并列”列在白噪声行也按次数报告，对应同一个真实并列事件。模型栏沿用主稿定义。白噪声均指30 dB模拟高斯白噪声。

\begin{center}
\small
\captionof{table}{总体误差与方向指标}
\label{tab:s-metrics}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrr}
\toprule
条件 & 输入 & MAE/mHz & RMSE/mHz & P95/mHz & 判对/总数 & 预测并列 & 真并列\\
\midrule
\resultrows{assets/pair-summary-rows.tex}
\bottomrule
\end{tabular}
\end{center}

\clearpage
\subsection{配对效应与下垂系数分带统计}
表\ref{tab:s-effects}列出同一候选配对上的差值MAE变化，正数表示误差增加。各区间用主稿的分层参考事件重采样口径。表\ref{tab:s-bands}用于描述两参数带的结果。

\begin{center}
\small
\captionof{table}{相对两个基线的差值MAE变化及95\%区间（mHz）}
\label{tab:s-effects}
\begin{tabular}{llll}
\toprule
条件 & 输入 & 比较基线 & 变化及区间\\
\midrule
\resultrows{assets/pair-effect-rows.tex}
\bottomrule
\end{tabular}
\end{center}

\begin{center}
\small
\captionof{table}{下垂系数分带结果}
\label{tab:s-bands}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lllrrrrrr}
\toprule
$R$带 & 条件 & 输入 & 主对数 & MAE & RMSE & P95 & 相对端点变化 & 判对/总数 & 预测并列\\
\midrule
\resultrows{assets/pair-band-rows.tex}
\bottomrule
\end{tabular}
\par\smallskip
\footnotesize 误差及误差变化单位为mHz。每带的真并列物理对数见表\ref{tab:s-flow}。
\end{center}

\clearpage
\subsection{原网格随机划分与复算记录}
原网格的十次随机划分复用同一仿真场景集。表\ref{tab:s-repeat}中的标准差反映划分敏感性。

\begin{center}
\small
\captionof{table}{十次随机划分的梯度提升树结果}
\label{tab:s-repeat}
\begin{tabular}{lrr}
\toprule
特征 & 均方根误差/Hz & $R^2$\\
\midrule
\resultrows{revision-assets/repeat.tex}
\bottomrule
\end{tabular}
\end{center}

近端点正式试验的参考参数表和配对账本记录参考与候选标识、配对参数、截点及目标频差、匹配状态和排除原因。逐对预测表保留成员预测、真实差和预测差、方向及并列字段。复算按参考事件和候选标识对齐成员，严格从观测截止后采样重新核对标签，重新判定三类方向与幅度纳入状态。重采样在两个下垂系数带内分别抽取参考事件，同一参考的候选和噪声复本共同进出。随附核验日志记载逐指标校验与5\,000个重采样结果的复核。历史基准检查表记录固定0.50 s窗口特征及逐场景预测相对原档案的误差。

\section{复现入口}
数字对照表列出正式CSV的筛选键、字段、计算式和单位。配对证据文件与补充材料一并提供。原有实验的复现入口如下。""")
    tex=tex.replace(r"\begin{tabular}{lllrrrrrr}",r"\begin{tabular}{lllrrrrrrr}")
    start = tex.index(r"\begin{itemize}",tex.index(r"\section{复现入口}"))
    end = tex.index(r"\end{itemize}",start)+len(r"\end{itemize}")
    tex = tex[:start]+("随稿说明给出原有仿真与模型训练的复现入口。短窗矩阵、边界事件、扩展噪声、"
                      "局部可辨识性及预测区间均保留原始实验记录。")+tex[end:]
    tex=copy_images(tex)
    (OUT/"ARTICAL-nadir-paired-v2-supplement.tex").write_text(tex,encoding="utf-8")


def prepare():
    for folder in ("assets","revision-assets","pair-evidence"):
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    for file in (ART/"revision-assets").iterdir():
        if file.is_file():
            shutil.copyfile(file,OUT/"revision-assets"/file.name)
    data()
    add_aliases()
    main_source()
    supplement_source()


def package():
    name = OUT/"nadir-paired-v2-LaTeX.zip"
    files = list(OUT.glob("*.tex"))+[p for folder in ("assets","revision-assets","pair-evidence")
                                    for p in (OUT/folder).rglob("*") if p.is_file()]
    files.extend(p for p in (OUT/"数字—源文件对照表.csv",OUT/"修改说明.md") if p.is_file())
    with zipfile.ZipFile(name,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p,p.relative_to(OUT))
    with zipfile.ZipFile(name) as archive:
        assert archive.testzip() is None


if __name__=="__main__":
    prepare()
    package()
