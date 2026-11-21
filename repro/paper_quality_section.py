"""Render the measurement-quality experiment from its result tables."""
import pandas as pd


def render_quality(root):
    stats=pd.read_csv(root/"quality-summary.csv")
    pairs=pd.read_csv(root/"quality-paired.csv")
    estimates=pd.read_csv(root/"quality-estimates.csv")
    events=pd.read_csv(root/"quality-events.csv")
    design=pd.read_csv(root/"ieee39quality-design.csv")
    s=stats[stats.TargetFAR==.10]
    main=s[s.Condition=="mixed"]
    assert len(main)==12
    translations={"persistence":"持续性","quadratic":"二次拟合","fixed":"固定",
                  "estimated":"估计分级","oracle":"真值等级","aggregate":"聚合","ieee39_multi":"39节点"}
    pct=lambda v: f"{100*v:.1f}"

    def value(ds,policy,condition="mixed"):
        return s.query("Dataset==@ds and Model=='quadratic' and Policy==@policy and Condition==@condition").iloc[0]

    rows=[]
    for r in main.itertuples():
        pair=lambda v,lo,hi: f"{pct(v)} [{pct(lo)}, {pct(hi)}]"
        rows.append(" & ".join([translations[r.Dataset],translations[r.Model],translations[r.Policy],
          str(r.Hits),str(r.FalseAlarms),pair(r.Timely,r.TimelyLow,r.TimelyHigh),
          pair(r.FAR,r.FARLow,r.FARHigh),f"{r.LeadMedian_ms:.0f}" if pd.notna(r.LeadMedian_ms) else "--"])+r" \\")
    parts=[]
    for ds in ("aggregate","ieee39_multi"):
        a,b=value(ds,"fixed"),value(ds,"estimated")
        d=pairs.query("Dataset==@ds and Model=='quadratic' and TargetFAR==0.10").iloc[0]
        parts.append(f"{translations[ds]}算例中，二次拟合的及时命中率由{pct(a.Timely)}\\%变为"
                     f"{pct(b.Timely)}\\%，实际虚警率由{pct(a.FAR)}\\%变为{pct(b.FAR)}\\%。"
                     f"命中率配对差为{pct(d.DeltaTimely)}个百分点，95\\%区间为"
                     f"[{pct(d.TimelyLow)}, {pct(d.TimelyHigh)}]。")
    primary=["clean","white40","white20","correlated20"]
    test=estimates.query("Role=='test' and Condition in @primary")
    wrong=(test.EstimatedLow!=test.TrueLow).sum()
    near=estimates.query("Role=='test' and Dataset=='aggregate' and Condition=='white30'").EstimatedLow.mean()
    jump=value("aggregate","estimated","noise_jump")
    ideal=value("aggregate","oracle","noise_jump")
    examples=events.query("Dataset=='aggregate' and Model=='quadratic' and TargetFAR==.10 "
                         "and Condition in @primary")
    cases=examples.pivot(index=["Scenario","Condition"],columns="Policy",values="Outcome")
    observations=[]
    for description,mask in [
        ("固定阈值虚警，分级阈值未触发",(cases.fixed=="false_alarm")&(cases.estimated=="true_negative")),
        ("固定阈值及时命中，分级阈值未触发",(cases.fixed=="hit")&(cases.estimated=="no_alarm")),
        ("分级阈值在越限后触发",cases.estimated=="late"),
    ]:
        if mask.any():
            i,c=cases[mask].index[0]
            observations.append(f"场景{i}（{c}）{description}")
    return r"""
\section{基于扰动前量测的告警阈值校准}
量测噪声与短窗估计误差会影响频率变化率的解释\cite{pmu}。本节固定预测器，以扰动前1 s的50个量测样本估计频率标准差。以开发参考频偏下40 dB和20 dB噪声标准差的几何均值划分高、低质量等级。固定阈值在本地校准集的完整混合上选择，分级阈值则在对应等级的未越限校准场景上选择。每个事件以四个决策时刻的预测最小值校准，告警评价仍以首次触发为准。受虚警约束的分类框架参见文献\cite{np}。告警阈值采用经验分位数；目标虚警率是校准参数。

主试验包含干净量测、40 dB白噪声、20 dB白噪声和与20 dB白噪声同方差的相关噪声。相关噪声用系数0.8的一阶自回归过程模拟。各法共享事件与噪声实现。聚合新增600组校准和1800组封存场景。IEEE 39节点计划ANDESPLAN组，完成ANDESOK组；校准和测试分别使用4个负荷水平，封存测试有ANDEStest组。每个物理事件生成4个量测版本。混合结果将四种量测等权合并，95\%区间以物理事件为单位重采样，保留同一事件的全部量测版本。区间反映给定试验设计内的场景变化。

\newpage
\begin{center}\textbf{表6\quad 量测质量分级的封存测试}\\
\small 10\%目标虚警率，四种量测等权混合
\end{center}
\begin{center}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lllrrrrr}
\toprule
系统 & 模型 & 阈值 & 命中 & 虚警 & 命中率/\%（95\%区间） & 虚警率/\%（95\%区间） & 中位提前量/ms\\
\midrule
QUALITYROWS
\bottomrule
\end{tabular}
\end{center}

RESULTTEXT
表6与图4应结合实际虚警率及持续性基线解读。提前量的统计对象为及时命中事件。

\begin{center}
\includegraphics[width=.93\linewidth]{../repro/out/rolling/quality-tradeoff.pdf}\\
图4\quad 固定与分级策略的实际虚警率和及时命中率
\end{center}

主试验两档噪声强度相隔较大，封存集有WRONG次等级误分。另以冻结阈值开展探索性压力试验。30 dB接近分级边界，聚合试验中NEAR\%事件被划为低质量。扰动发生时噪声从40 dB突变至20 dB，扰动前估计仍反映初始质量。此时聚合试验中二次拟合估计分级与真值分级的虚警率分别为JUMP\%和IDEAL\%。本实验假定扰动起点已知且扰动前真实频率保持额定值，实际测量中的背景漂移、起点误差与突发噪声仍需验证。

EXAMPLES
""".replace("QUALITYROWS","\n".join(rows)).replace("ANDESPLAN",str(len(design))).replace(
    "ANDESOK",str((design.Status=="ok").sum())).replace(
    "ANDEStest",str(int(value("ieee39_multi","fixed").PhysicalEvents))).replace(
    "RESULTTEXT","".join(parts)).replace("WRONG",str(wrong)).replace("NEAR",pct(near)).replace(
    "JUMP",pct(jump.FAR)).replace("IDEAL",pct(ideal.FAR)).replace(
    "EXAMPLES","逐场景记录可核查以下例子。"+"；".join(observations)+"。" if observations else "")
