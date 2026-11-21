"""Read frozen R2 evidence and optional R3 scores into the rolling manuscript."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
OUTPUT=ROOT/"repro/out"
R3=OUTPUT/"round3/sealed"
R4=OUTPUT/"round4"
R5=OUTPUT/"round5"


def render_round3():
    r2=OUTPUT/"round2/multi-noise/per-seed-summary.csv"
    if not r2.exists():
        return ""
    stats=pd.read_csv(r2).query("TargetFAR==.10").pivot(index="k",columns="Model",values="Hits")
    assert len(stats)==40
    paired=stats.quadratic-stats.persistence
    fig,ax=plt.subplots(figsize=(6.5,2.2),layout="constrained")
    ax.axhline(0,color="0.35",linewidth=1)
    ax.bar(stats.index+1,paired,color=["#a45434" if x>0 else "#356681" for x in paired])
    ax.set(xlabel="Noise implementation",ylabel="Timely hits difference\n(quadratic − persistence)")
    fig.savefig(OUTPUT/"round3/r2-paired-noise.pdf")
    plt.close(fig)
    section=r"""
\section{噪声实现敏感性与未见工况检验}
\subsection{固定物理场景上的噪声实现敏感性}
在一组固定的IEEE 39节点校准与测试场景上，更换40个随机噪声实现，逐实现重新选择告警阈值。二次拟合使用目标系统校准数据训练，其阈值基于校准集的五折折外预测。各实现包含21个物理越限事件及每个事件的6种量测条件，每实现共有126个越限场景与量测条件组合。四条件混合试验仍采用前文口径。

\begin{center}
\textbf{表7\quad 固定场景上40个噪声实现的平均及时命中}\\
\small 10\%目标虚警率，每实现126个越限组合
\end{center}
\begin{center}
\begin{tabular}{lrrr}
\toprule
比较 & 持续性 & 二次拟合 & 残差 \\
\midrule
平均命中数 & PERSIST & QUAD & RESID \\
\bottomrule
\end{tabular}
\end{center}

二次拟合较持续性平均少5.00个及时命中，按实现配对的95\%区间为$[-6.75,-3.25]$。二次拟合在6/40个实现中反超，表明平均方向未在全部实现保持。按实际虚警率对齐后的插值比较得到平均差$-14.37$个命中，95\%区间为$[-17.61,-11.14]$；其中4个实现需使用曲线端点处理。这项插值用于诊断工作点差异。按21个物理越限事件进行簇重采样估计，持续性的设计效应为1.48（95\%区间$[1.26,1.61]$），区间跨越离散度判据所需的1.36。结果限于固定的物理场景、模型及噪声族。

\begin{center}
\includegraphics[width=.82\linewidth]{../repro/out/round3/r2-paired-noise.pdf}\\
图5\quad 每个噪声实现中二次拟合相对持续性的及时命中差
\end{center}
""".replace("PERSIST",f"{stats.persistence.mean():.2f}").replace(
    "QUAD",f"{stats.quadratic.mean():.2f}").replace("RESID",f"{stats.residual.mean():.2f}")
    if not (R3/"summary.csv").exists():
        return section
    d=pd.read_csv(R3/"design-results.csv")
    summary=pd.read_csv(R3/"summary.csv").query("TargetFAR==.10 and Condition=='mixed'")
    labels={"load_holdout":"负荷外推","pattern_holdout":"扰动位置留出"}
    models={"persistence":"持续性","quadratic":"二次拟合","residual":"残差"}
    def percent(v):
        return f"{100*v:.1f}" if pd.notna(v) else "--"
    lines=[]
    for x in summary.itertuples():
        lead=f"{x.LeadMedian_ms:.0f}" if pd.notna(x.LeadMedian_ms) else "--"
        lines.append(" & ".join([labels[x.Block],models[x.Model],str(x.PhysicalEvents),str(x.PhysicalRisk),
            str(x.Hits),str(x.FalseAlarms),percent(x.Timely),percent(x.FAR),lead])+r" \\")
    differences=pd.read_csv(R3/"paired.csv")
    primary=summary.query("Block=='load_holdout'").set_index("Model")
    diff=differences.query("Block=='load_holdout' and VersusPersistence=='quadratic' and Metric=='Timely'").iloc[0]
    a,b,c=(primary.loc[m] for m in ("persistence","quadratic","residual"))
    narrative=(f"在负荷外推主测试中，持续性和二次拟合的实际虚警率分别为{percent(a.FAR)}\\%和"
        f"{percent(b.FAR)}\\%。二次拟合相对持续性的及时命中率配对差为{percent(diff.Difference)}个百分点"
        f"（95\\%事件簇重采样区间$[{percent(diff.Low95)},{percent(diff.High95)}]$）。残差的实际虚警率为"
        f"{percent(c.FAR)}\\%，与持续性的工作点存在差异。"
        "次测试块只有少量物理越限事件，且新负荷水平与新扰动位置同时变化，结果用于诊断联合迁移。")
    papers=r"""
\subsection{负荷外推与未见扰动位置}
另外生成144组新的IEEE 39节点场景，完成OKCOUNT组。阈值仅由目标系统校准块确定。主测试块的四个负荷水平全部位于校准范围之外，且扰动位置沿用校准设计。次测试块在校准负荷范围内使用新负荷水平和未见母线组合。二次拟合与残差模型均在原聚合开发网格上训练，三类策略共享R3的量测、分级阈值规则和四个决策时刻。残差模型从测得的当前频率减去预测的额外跌幅。与上一小节的二次拟合训练口径存在差异，各轮结果独立解释。

\newpage
\begin{center}\textbf{表8\quad 未见物理场景上的事件级滚动预警}\\
\small 10\%目标虚警率，四种量测条件等权混合
\end{center}
\begin{center}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrrrr}
\toprule
测试块 & 策略 & 物理场景 & 物理越限 & 命中 & 虚警 & 命中率/\% & 虚警率/\% & 中位提前量/ms\\
\midrule
RTHREEROWS
\bottomrule
\end{tabular}
\end{center}

表8的命中及虚警次数以场景与量测条件组合为单位。实际虚警率不同的工作点需结合另两档预设目标及按物理事件重采样的区间比较。试验结果仅代表本设计范围，收敛失败场景保留在设计记录中。

RTHREETEXT
"""
    result=section+papers.replace("OKCOUNT",str((d.Status=="ok").sum())).replace(
        "RTHREEROWS","\n".join(lines)).replace("RTHREETEXT",narrative)
    return result+render_round4()+render_round5()


def render_round4():
    if not (R4/"summary.csv").exists():
        return ""
    table=pd.read_csv(R4/"summary.csv")
    main=table.query("Condition=='mixed' and TargetFAR==.10").set_index("Model")
    contrasts=pd.read_csv(R4/"paired.csv")
    design=pd.read_csv(R4/"design-results.csv")
    labels={"persistence":"持续性","quadratic":"二次拟合","residual":"残差"}
    def pct(v):
        return f"{100*v:.1f}"
    rows=[]
    for model,label in labels.items():
        r=main.loc[model]
        rows.append(" & ".join([label,f"{int(r.Hits)}/{int(r.RiskVariants)}",
            f"{int(r.FalseAlarms)}/{int(r.NonriskVariants)}",pct(r.Timely),pct(r.FAR),
            f"{r.LeadMedian_ms:.0f}" if pd.notna(r.LeadMedian_ms) else "--"])+r" \\")
    a = main.loc["persistence"]
    timely = contrasts.query("Metric=='Timely'").set_index("VersusPersistence")
    differences = []
    for model in ("quadratic", "residual"):
        delta = timely.loc[model]
        differences.append(f"{labels[model]}低{pct(-delta.Difference)}个百分点"
                           f"（95\\%区间$[{pct(-delta.High95)},{pct(-delta.Low95)}]$）")
    observed = ("在预设主设置下，持续性的实际虚警率最低，及时命中率最高。以持续性为参照，"
                "及时命中率差为" + "、".join(differences) + "。表中的提前量仅针对及时命中组合。")
    return r"""
\subsection{冻结阈值下的高负荷外推确认}
为增加可评分的物理越限事件，另锁定128个高负荷场景。采用八个未用于上轮的新负荷水平，全部位于校准范围上方，保留两类扰动位置，提高较深阶跃的抽样比例。三个模型、分级规则和全部阈值均沿用上一轮，噪声种子独立。计划场景中SUCCESS组收敛，其中RISK组越限。此处越限比例描述定向压力设计。

\begin{center}\textbf{表9\quad 冻结阈值下的高负荷外推结果}\\
\small 10\%目标虚警率，四种量测条件等权混合
\end{center}
\begin{center}
\small
\begin{tabular}{lrrrrr}
\toprule
策略 & 及时命中 & 虚警 & 命中率/\% & 虚警率/\% & 中位提前量/ms\\
\midrule
RFOURROWS
\bottomrule
\end{tabular}
\end{center}

RFOURTEXT

\begin{center}
\includegraphics[width=.73\linewidth]{../repro/out/round4/tradeoff.pdf}\\
图6\quad 高负荷外推中三档冻结阈值的实际工作点
\end{center}

三个预设阈值的实际虚警率与及时命中率见图6。区间按物理事件重采样，每次保留同一事件的全部噪声版本。按负荷分组后，持续性在八个负荷水平的及时命中数均较高，分组虚警排序随负荷变化。噪声突变条件下，主阈值对应的实际虚警率升至18.8\%、52.1\%和52.1\%（按表9顺序）。结果限于本批高负荷工况及已设置的量测条件。收敛失败记录完整保留，评估对象为可评分场景。
""".replace("SUCCESS",str((design.Status=="ok").sum())).replace("RISK",str(int(a.PhysicalRisk))).replace(
        "RFOURROWS","\n".join(rows)).replace("RFOURTEXT",observed)


def render_round5(*, detailed=False):
    if not (R5 / "summary.csv").exists():
        return ""
    summary = pd.read_csv(R5 / "summary.csv")
    main = summary.query("Condition=='mixed' and TargetFAR==.10").set_index("Model")
    design = pd.read_csv(R5 / "design-results.csv")
    methods = ("persistence", "quadratic", "residual")
    false_alarms = "、".join(str(int(main.loc[method, "FalseAlarms"])) for method in methods)
    base = main.loc["persistence"]
    text = rf"""
\subsection{{扰动位置留出与求解收敛性}}
在四档高负荷下，采用两个未见的四母线扰动组合，沿用冻结模型和告警阈值。{len(design)}组场景中，{int(base.PhysicalEvents)}个低幅阶跃场景收敛，且均未越限。其余{int((design.Status == 'failed').sum())}个较深阶跃场景在积分步长缩至零后终止。按表9的策略顺序，10\%目标虚警设置下的虚警次数为{false_alarms}次，分母均为{int(base.NonriskVariants)}个量测组合。由于正式批次缺少可判定的越限事件，及时命中率无法估计。
"""
    if not detailed:
        return (text + "补充材料报告中间阶跃扫描、不同积分方法的交叉复算及静态潮流核查。"
                + render_independent_time_check() + "\n")
    transition_file = R5 / "transition" / "summary.csv"
    if transition_file.exists():
        trial = pd.read_csv(transition_file).query("Condition=='mixed' and TargetFAR==.10").set_index("Model")
        trial_design = pd.read_csv(R5 / "transition" / "design-results.csv")
        first = trial.loc["persistence"]
        text += (f"另在两个预设幅值区间之间补做{len(trial_design)}组探索性扫描，"
                 f"其中{int(first.PhysicalEvents)}组完成，{int(first.PhysicalRisk)}组越限。")
        if first.PhysicalRisk:
            trial_hits = "、".join(str(int(trial.loc[method, "Hits"])) for method in methods)
            text += (f"三种策略分别及时命中{trial_hits}次，分母均为{int(first.RiskVariants)}个量测组合。"
                     "此项扫描用于定位可评分的阶跃范围，结果与正式批次分别报告。")
        else:
            trial_false = "、".join(str(int(trial.loc[method, "FalseAlarms"])) for method in methods)
            text += (f"按表9的策略顺序，虚警为{trial_false}次，分母均为{int(first.NonriskVariants)}。"
                     "这些计数仅描述探索性扫描中的收敛样本。")
    audit_file = R5 / "r4-audit" / "solver-check.csv"
    if audit_file.exists():
        audit = pd.read_csv(audit_file)
        old = pd.read_csv(R4 / "design-results.csv").query("Status=='failed'")
        failed = audit[audit.R4Status.eq("failed")]
        controls = audit[audit.R4Status.eq("ok")]
        per_id = audit.groupby("DesignID").IntegrationStep_s.agg(set)
        complete = (set(failed.DesignID) == set(old.DesignID) and len(controls) > 0
                    and per_id.apply(lambda steps: steps == {.02, .01, .005}).all())
        if complete:
            recovered = failed[failed.Status.eq("ok")].DesignID.nunique()
            stable = int(controls.groupby("DesignID").Status.apply(lambda states: states.eq("ok").all()).sum())
            text += (f" 对上一轮{len(old)}个失败场景及{controls.DesignID.nunique()}个配对对照采用20、10、5 ms"
                     f"三档积分设置复算，原失败场景中有{recovered}组在复算中完成，"
                     f"{stable}个对照在三档设置下均完成。"
                     "求解中止场景的完整轨迹缺失，其动力学状态尚需独立仿真核实。")
    return (text + render_solver_crosscheck() + render_static_flow_check()
            + render_independent_time_check(detailed=True) + "\n")


def render_solver_crosscheck():
    source = OUTPUT / "solver-crosscheck" / "summary.json"
    if not source.exists():
        return ""
    result = json.loads(source.read_text(encoding="utf-8"))
    r4 = result["prior_completion"]["round4"]
    r5 = result["prior_completion"]["round5/transition"]
    completion = result["backeuler_completion_by_ms"]
    r5_crossings = r5["any_method_observed_crossing"]
    observed = (f"其中{r5_crossings}个在至少一种方法的已接受轨迹中观察到越限。"
                if r5_crossings else "三种方法在中止前均未观察到低频越限。")
    return (f"\n\n对同一模型的{result['cases']}个代表性场景分别采用隐式梯形、后向欧拉和QNDF复算。"
            f"高负荷外推测试中选取的{r4['failed_cases']}个初始失败场景，有"
            f"{r4['all_methods_observed_crossing']}个在三种方法的已接受轨迹中均观察到低频越限。"
            f"在10 ms步长参数下，有{len(result['only_backeuler_at_10ms'])}个场景只有后向欧拉完成。"
            f"后向欧拉在5和2.5 ms下分别完成其中{len(completion.get('5', []))}和{len(completion.get('2.5', []))}个。"
            f"新位置中选取的{r5['failed_cases']}个12.1\\%阶跃场景，"
            f"有{r5['all_methods_stopped']}个在三种方法下均中止。" + observed +
            "截断前未观察到越限时，完整事件的风险状态仍未知。以上复算仅用于诊断。\n\n")


def render_static_flow_check():
    source = OUTPUT / "static-flow-check" / "summary.csv"
    if not source.exists():
        return ""
    result = pd.read_csv(source)
    full = result[result.PFStatus.eq("ok") & result.StepShare.eq(1)]
    original_failures = full[full.PriorDynamicStatus.eq("failed")]
    return (f"使用同一网络参数独立组装交流潮流方程，在恒定PQ负荷、固定PV电压给定和不施加无功限幅的条件下，"
            f"{len(result)}个复算场景中有{len(full)}个在完整阶跃幅值处找到潮流解，"
            f"其中{len(original_failures)}个为原动态仿真失败场景。"
            "该静态结果说明所设控制条件下存在潮流平衡点。动态轨迹能否到达该平衡点，仍需由多机时域模型检验。\n")


def render_independent_time_check(*, detailed=False):
    directory = OUTPUT / "independent-time-check"
    summary = directory / "summary.json"
    if not summary.exists():
        return ""
    result = json.loads(summary.read_text(encoding="utf-8"))
    data = pd.read_csv(directory / "summary.csv")
    completed_controls = int((data.Group.eq("controls") & data.Status.eq("ok")).sum())
    sampling_file = directory / "sampling-diagnostic.csv"
    text = (f"另使用相同多机方程，以独立实现的BDF2和稀疏线性求解器复算6个场景。"
            f"两个积分步长下，低幅对照完成{completed_controls}/{result['control_runs']}次，"
            f"有{result['controls_accepted']}次通过预设验收。"
            f"原失败场景有{result['failures_completed']}/{result['failure_runs']}次完成，"
            f"有{result['failures_observed_crossing']}次观察到越限。"
            "截断且未观察到越限的结果保留为风险未知。"
            "复算结果用于求解诊断。")
    if not detailed:
        if sampling_file.exists():
            text += "补充材料中的格点对照显示，采样口径解释了低幅对照的大部分频差偏差。"
        return text
    if sampling_file.exists():
        comparison = pd.read_csv(sampling_file)
        held_max = comparison.HeldCOIMaxError_pu.max() * 1e5
        interpolated_max = comparison.InterpolatedCOIMaxError_pu.max() * 1e7
        text += (f"另对原积分轨迹开展事后采样对照。前样本保持下两套轨迹的最大频差为"
                 f"${held_max:.2f}\\times10^{{-5}}$ p.u.，"
                 f"原轨迹插值到相同格点后为${interpolated_max:.2f}\\times10^{{-7}}$ p.u.。"
                 "采样对照用于解释偏差，预设验收及告警评分沿用原口径。")
    labels = {"observed_crossing": "已观察到越限", "complete_no_crossing": "完整未越限",
              "truncated_unknown": "截断，风险未知"}
    rows = []
    for row in data.itertuples():
        crossing = f"{row.CrossingAfterEvent_s:.3f}" if pd.notna(row.CrossingAfterEvent_s) else "--"
        rows.append(f"{row.DesignID} & {1000 * row.Step_s:g} & {row.LastAccepted_s:.4f} & "
                    f"{row.MinVoltage_pu:.3f} & {crossing} & {labels[row.Outcome]}" + r" \\")
    return r"""
\subsection{六场景时域复算}
""" + text + r"""

使用5和2.5 ms积分步长。从原动态初始化出发，在1 s施加完全相同的负荷阶跃。用20 ms采样格点计算惯量中心频差、各母线电压和抗积分饱和状态。对照验收要求21 s轨迹完整、惯量中心频差最大偏差不超过$5\times10^{-5}$ p.u.，且相对QNDF结果的最低电压差不超过0.015 p.u.。表中最后接受时刻以仿真开始计，首次越限时间以扰动时刻计。

\begin{center}
\small
\begin{tabular}{rrrrrl}
\toprule
场景 & 步长/ms & 最后接受时刻/s & 最低电压/p.u. & 首次越限/s & 判定\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{center}
"""
