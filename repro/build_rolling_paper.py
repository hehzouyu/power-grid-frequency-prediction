from pathlib import Path
import json
import pandas as pd
from paper_quality_section import render_quality
from paper_round3_section import render_round3, render_round5
root=Path('repro/out/rolling')
s=pd.read_csv(root/'warning-summary.csv')
m=json.loads((root/'protocol-results.json').read_text())
a=s.query("Dataset=='aggregate' and TargetFAR==.05")
names=dict(persistence_rolling='持续性滚动',endpoint_rolling='端点滚动',quadratic_rolling='二次拟合滚动',segmented_rolling='分段斜率滚动',quadratic_fixed='二次拟合固定时刻',quadratic_lower='二次拟合下界滚动')
def row(r,labels):
 nums=[r.N,r.Risk,r.Hits, f'{100*r.FAR:.1f}',f'{100*r.Miss:.1f}' if pd.notna(r.Miss) else '--',f'{r.LeadMedian_ms:.0f}' if pd.notna(r.LeadMedian_ms) else '--']
 return ' & '.join(map(str,labels+nums))+r' \\'
def table(rows):
 return r'''
\begin{center}
\small
\begin{tabular}{lrrrrrr}
\toprule
配置 & 场景数 & 越限 & 按时命中 & 虚警率/\% & 未按时告警/\% & 中位提前量/ms\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{center}
'''
clean=table([row(r,[names[r.Policy]]) for r in a.query("Condition=='clean'").itertuples()])
stress=table([row(r,[r.Condition.replace('_',r'\_')]) for r in a.query("Policy=='quadratic_rolling'").itertuples()])
joint=f'{100*m["joint_test_coverage"]:.1f}'
dynamic=s.query("Dataset!='aggregate' and Condition=='clean' and TargetFAR==.05 and Policy=='quadratic_rolling'")
cross=table([row(r,[r.Dataset.replace('_',r'\_')]) for r in dynamic.itertuples()]) if len(dynamic) else '多机动态算例尚无通过验收的结果，以下结论限于聚合仿真。'
distributed=root/'distributed-summary.csv'
supplement=''
focused=''
if (root/'focused-summary.csv').exists():
 f=pd.read_csv(root/'focused-summary.csv')
 design=pd.read_csv(root/'ieee39focus-design.csv')
 top=f.query("Channel=='multi' and TargetFAR==0.10")
 if len(top)!=6:
  raise ValueError("Focused main comparison must contain six rows")
 methods={'persistence':'持续性','quadratic':'二次拟合'}
 conditions={'clean':'干净量测','20dB':'等效20 dB'}
 rows=[]
 for r in top.itertuples():
  method=r.Policy.split('_',1)[0]
  mode='同噪声校准' if 'matched' in r.Policy else '干净校准'
  rows.append(row(r,[methods[method]+'／'+conditions[r.Condition]+'／'+mode]))
 focused=r'''
\subsection{IEEE 39节点追加场景与噪声校准}
依据前轮结果，在两个较易产生可评分越限事件的多母线负荷分布上补充试验。采用6个初始负荷水平，每组分别在负荷阶跃比0.093至0.121及0.128至0.147的区间内随机抽取4个幅值。计划96组，完成FOCUSN组。0.945、0.985和1.035倍负荷仅用于阈值选择，0.965、1.015和1.055倍负荷用于评估。持续性和二次拟合的模型结构保持固定。

表5以多点均值量测、10\%目标虚警率为主要对照。5\%与20\%目标的结果同时保存在附件。噪声匹配阈值由同噪声水平的独立校准场景得到，且量测噪声水平在本试验中已知。IEEE 39节点的非收敛场景完整记录，该组评价针对特意增强越限覆盖的压力测试。

\begin{center}\textbf{表5\quad IEEE 39节点新增场景的预警与噪声校准}\end{center}
FOCUSTABLE

新增场景共92组可评分，其中35组越限。封存测试含45组，其中17组越限。干净量测下，持续性与二次拟合分别及时命中17和14组，虚警分别为3和2次。等效20 dB噪声下，匹配噪声校准使二次拟合虚警从10次降至5次，及时命中从11组降至10组。持续性的虚警从5次降至0次，及时命中由15组降至7组。相同目标虚警率未产生相同的实际虚警率，现有样本体现的是漏报与虚警之间的权衡。定向算例结果仅适用于本组扰动分布。
'''.replace('FOCUSN',str((design.Status=='ok').sum())).replace('FOCUSTABLE',table(rows).replace(r'\begin{tabular}{lrrrrrr}',r'\begin{tabular}{p{3.95cm}rrrrrr}'))
if distributed.exists():
 d=pd.read_csv(distributed)
 records=d.query("Condition=='clean' and Policy in ['quadratic_aggregate','quadratic_local']")
 selected=records[records.Dataset.str.endswith(('_coi','_multi'))]
 labels={'ieee14dist_coi':'14节点惯量中心','ieee14dist_multi':'14节点多点均值',
         'ieee39dist_coi':'39节点惯量中心','ieee39dist_multi':'39节点多点均值'}
 display=[row(r,[labels[r.Dataset]+('（本地）' if r.Policy.endswith('local') else '（迁移）')])
          for r in selected.itertuples()]
 completed=[]
 for case in ('ieee14dist','ieee39dist'):
  design=pd.read_csv(root/f'{case}-design.csv')
  completed.append(f"{(design.Status=='ok').sum()}/{len(design)}")
 supplement=r'''
\subsection{分布式负荷阶跃与阈值迁移}
为增加可评分的越限事件，另在多个负荷母线上按原有负荷比例施加同步阶跃，按0.94、1.00和1.06倍基准负荷组织场景，阶跃总量为扰动前系统负荷的8.5\%至17.5\%。该试验为有意加重的压力测试。14和39节点分别完成COUNTA和COUNTB组。模型保持冻结。以1.00倍负荷的未越限事件校准本地阈值，在0.94和1.06倍负荷的独立场景报告结果。表4同时列出沿用聚合阈值与应用本地校准阈值的表现。

\begin{center}\textbf{表4\quad 分布式阶跃的多机测试与本地阈值校准}\end{center}
DISTTABLE

以多点均值为输入时，直接移植聚合阈值在两个算例中均无及时命中。使用本地阈值后，二次拟合在14节点命中27/32并产生1次虚警，在39节点命中4/6并产生1次虚警。同样校准的持续性基线分别命中32/32和6/6，各产生2次虚警。现有样本支持本地校准对直接迁移的改善，尚不足以判定二次拟合在相同虚警水平下优于持续性。加入等效20 dB噪声后，二次拟合本地阈值的虚警分别增至7次和10次。

两个算例的本地校准仅分别包含14和15个未越限事件，无法精细匹配5\%虚警目标。测试结果应结合逐场景计数和汇总文件中的区间解释。未收敛场景不参与性能计算，39节点可评分结果可能受收敛性筛选影响。
'''.replace('COUNTA',completed[0]).replace('COUNTB',completed[1]).replace('DISTTABLE',table(display))
src=r'''
\documentclass[UTF8,a4paper,11pt]{ctexart}
\usepackage[margin=23mm]{geometry}
\usepackage{amsmath,booktabs,graphicx}
\usepackage[hidelinks]{hyperref}
\title{聚合电力系统的短窗滚动低频预警}
\author{何周俣}
\date{}
\begin{document}
\maketitle
\noindent\textbf{摘要：}针对扰动后固定时刻预测可能错过低频越限的问题，建立事件级滚动预警评价框架。利用扰动后0.20、0.30、0.40和0.50 s的逐步观测，比较端点频率、二次拟合和分段斜率；以首次越限和首次告警的时间差评估有效提前量。独立聚合仿真测试含1600组场景和291个越限事件。在校准集采用相同的5\%目标虚警率时，二次拟合滚动和固定时刻策略分别于272和189个事件中及时告警，实际虚警率分别为6.4\%和5.7\%。滚动策略可增加及时命中，但性能受量测噪声、起点偏差、告警时延和迁移条件制约。

\noindent\textbf{关键词：}低频预警；滚动预测；频率最低点；事件级校准；量测噪声

\begin{center}
\textbf{Short-window Rolling Underfrequency Warning in Aggregated Power Systems}
\end{center}
\noindent\textbf{Abstract.} An event-level evaluation framework compares fixed-time and rolling underfrequency warning using observations available at 0.20, 0.30, 0.40, and 0.50 seconds after a disturbance. Lead time is measured against the first threshold crossing. In 1,600 held-out aggregate simulations, 291 events cross the test threshold. With thresholds calibrated to the same target false-alarm rate of 5\%, rolling quadratic-fit forecasts warn in time for 272 events, compared with 189 for the fixed-time forecast. The realized false-alarm rates are 6.4\% and 5.7\%, respectively. Sensitivity to measurement noise and disturbance timing is assessed separately.

\section{问题与方法}
低阶模型可用于扰动后最低频率预测\cite{anderson,liu}。预警还需要确认预测发出时目标阈值是否已经越限。本文以第一次严格低于额定频率0.5\%定义示例越限，并将告警发出时间与首次越限时间比较。50 Hz系统的试验阈值为49.75 Hz。阈值用于仿真评价。

聚合仿真沿用摆动与限幅调频模型：
\[
2H\dot x=-d-Dx+p,\qquad
T_g\dot p=-p+\operatorname{clip}(-x/R,0,0.1).
\]
其中$x$为频率标幺偏差，初值为零。新场景以四阶龙格库塔法积分，步长0.005 s，输出间隔0.02 s，扰动后跟踪20 s。开发集使用既有1200组参数网格。新增参数在$H\in[2,6]$ s、$D\in[0.5,2]$、$R\in[0.04,0.06]$、$T_g\in[0.2,1]$ s和$d\in[0.02,0.08]$范围内独立均匀抽样。独立下界校准集、告警阈值校准集及最终测试集分别为600、600、1600组。

分别在四个观测时刻拟合100棵、最大深度3的梯度提升树，目标为该时刻后的最低频率。特征包括窗口端点、端点加二次拟合的两个系数、端点加三段首尾斜率。持续性基线用端点直接表示后续最低频率。固定时刻策略只在0.50 s判断。每个滚动策略以事件内预测最小值在未越限校准场景上确定告警阈值。下界校准对四时刻标准化误差取事件内最大值，以独立校准集的有限样本分位数构造同时下界\cite{romano}。标准化尺度由开发集确定。

同一事件只计首次告警。越限时刻及之后发出的提示计为延迟，未及时告警比例以全部越限事件为分母。虚警率以未越限场景为分母。提前量只对按时告警计算，并同时给出命中数。噪声以开发集频偏均方根设置幅度，加于输入，标签保持干净。起点分别偏移20或40 ms，另考察20 dB、偏晚40 ms和20 ms处理时延的组合扰动。

\section{聚合仿真实验}
表1给出同一目标虚警率下的结果。二次拟合滚动相较固定时刻增加83次及时命中，中位有效提前量由240 ms增至390 ms。固定时刻前或恰在决策时刻已越限的事件有98个。各策略实测虚警率不同，因此同时用图1展示多个预设校准水平下的权衡。下界的四时刻联合测试覆盖率为JOINT\%，覆盖率与告警性能分别评价。

冻结干净工况的告警阈值后，二次拟合滚动策略在20 dB条件下产生573次虚警，占1309个未越限场景的43.8\%。组合扰动下虚警增至686次，说明及时命中较多并不足以表明策略在量测条件变化后仍适用。

\begin{center}\textbf{表1\quad 干净聚合测试的事件级预警结果}\end{center}
CLEANTABLE

\begin{center}
\includegraphics[width=.72\linewidth]{../repro/out/rolling/tradeoff.pdf}\\
图1\quad 不同校准水平下的实际虚警率与未按时告警比例
\end{center}

\newpage
\begin{center}\textbf{表2\quad 二次拟合滚动策略的扰动敏感性}\end{center}
STRESSTABLE

\begin{center}
\includegraphics[width=.68\linewidth]{../repro/out/rolling/lead.pdf}\\
图2\quad 及时命中事件的提前量经验分布
\end{center}
\begin{center}
\includegraphics[width=.68\linewidth]{../repro/out/rolling/stress.pdf}\\
图3\quad 冻结阈值下的噪声和起点偏差试验
\end{center}

\section{多机动态验证}
CROSS

\section{结论}
独立聚合试验表明，较早读入频率曲线的形状信息，可在接近的实测虚警水平下提高及时命中。负荷、量测、起点误差与计算时延均应纳入评价。现有试验针对合成扰动，现场应用还需量测数据和独立事件验证。
DISTCONCLUSION

\noindent\textbf{数据与代码可用性。}随稿材料包含场景参数、逐时刻观测索引、逐场景告警、汇总结果、图件和复现入口。

\begin{thebibliography}{99}
\bibitem{anderson} Anderson P M, Mirheydar M. A low-order system frequency response model. IEEE Transactions on Power Systems, 1990, 5(3):720--729.
\bibitem{liu} Liu L, Li W, Ba Y, et al. An analytical model for frequency nadir prediction following a major disturbance. IEEE Transactions on Power Systems, 2020, 35(4):2527--2536.
\bibitem{romano} Romano Y, Patterson E, Candès E J. Conformalized quantile regression. Advances in Neural Information Processing Systems, 2019, 32.
\end{thebibliography}
\end{document}
'''
if len(dynamic):
 cross=r'采用 ANDES 2.0 对公开 IEEE 动态算例进行负荷阶跃试验。扰动幅度按初始总负荷的1.5\%至12\%设置。以惯量中心频率定义越限，分别使用惯量中心频率、一个母线量测及三个母线量测的均值作为输入。模型及阈值从聚合试验直接迁移。IEEE 14节点完成90/90组，IEEE 39节点完成80/90组。其余10组时域积分未收敛，保留在场景表中，表3统计仅使用完成的仿真。大扰动的结果可能受可收敛场景选择影响。14节点只有2个越限事件，39节点可评分场景没有越限事件，因而本组数据无法确认跨系统漏报表现。'+cross
dist_conclusion='多机压力测试显示，聚合阈值的直接迁移可能遗漏越限事件。本地校准有助于恢复及时命中，其效果仍受目标系统数据量、量测噪声和仿真收敛性的限制。' if supplement else ''
if (root/'quality-summary.csv').exists() and (root/'ieee39quality-design.csv').exists():
 quality=render_quality(root)
 q=pd.read_csv(root/'quality-summary.csv').query("Condition=='mixed' and TargetFAR==0.10 and Model=='quadratic'")
 def key_metric(dataset,policy,metric):
  return f"{100*q.query('Dataset==@dataset and Policy==@policy').iloc[0][metric]:.1f}"
 agg_before=key_metric("aggregate","fixed","Timely")
 agg_after=key_metric("aggregate","estimated","Timely")
 multi_false=key_metric("ieee39_multi","estimated","FAR")
 src=src.replace(r'\section{结论}',quality+r'\section{结论}')
 src=src.replace('聚合电力系统的短窗滚动低频预警','量测质量变化下的电力系统短窗滚动低频预警')
 src=src.replace('Short-window Rolling Underfrequency Warning in Aggregated Power Systems',
                 'Short-window Rolling Underfrequency Warning under Varying Measurement Quality')
 src=src.replace('滚动策略可增加及时命中，但性能受量测噪声、起点偏差、告警时延和迁移条件制约。',
   f'在4类量测等权混合的新增聚合封存测试中，以10\\%为校准目标，扰动前量测分级将二次拟合及时命中率由{agg_before}\\%提高至{agg_after}\\%。'
   f'IEEE 39节点的实际虚警率为{multi_false}\\%。扰动时噪声突变使分级策略失效风险增加。')
 src=src.replace('Sensitivity to measurement noise and disturbance timing is assessed separately.',
   f'Under mixed measurement conditions in a new aggregate test set, quality-stratified calibration '
   f'increases the timely hit rate of quadratic-fit forecasts from {agg_before}\\% to {agg_after}\\% '
   f'at a target false-alarm rate of 10\\%. The realized false-alarm rate on the IEEE 39-bus test is '
   f'{multi_false}\\%. An abrupt noise increase at the disturbance can invalidate the pre-event quality estimate.')
 src=src.replace('DISTCONCLUSION','DISTCONCLUSION 扰动前量测分级改善了混合噪声下二次拟合的及时命中。'
   '在IEEE 39节点，命中改善伴随实际虚警上升，持续性基线仍有竞争力。噪声突变试验表明，以扰动前噪声表征整个告警窗口的适用性取决于量测条件的稳定性。')
 src=src.replace(r'\end{thebibliography}',r'''
\bibitem{pmu} Rietveld G, Wright P S, Roscoe A J. Reliable rate of change of frequency measurements: Use cases and test conditions. IEEE Transactions on Instrumentation and Measurement, 2020, 69(9):6657--6666.
\bibitem{np} Rigollet P, Tong X. Neyman-Pearson classification under a strict constraint. Proceedings of Machine Learning Research, 2011, 19:595--614.
\bibitem{polynomial} Li H, Wang L, Qi S, et al. Power system frequency nadir prediction based on data-driven and power-frequency polynomial fitting. Frontiers in Energy Research, 2024, 12:1501181.
\end{thebibliography}''')
 src=src.replace('预警还需要确认预测发出时目标阈值是否已经越限。','已有研究结合功率—频率拟合与数据回归估计最低频率，并使用改进IEEE 39节点算例验证'+r'\cite{polynomial}'+'。本文进一步评价告警发出时目标阈值是否已经越限。')
src=src.replace(r'\section{结论}',render_round3()+r'\section{结论}')
if (root.parent/'round3'/'sealed'/'summary.csv').exists():
 dist_conclusion += (' 固定物理场景的噪声复核显示，持续性与二次拟合的排序会随噪声实现翻转。'
  '在越限样本较少的独立负荷外推块中，持续性以相近的实际虚警率取得更多及时命中。')
 if (root.parent/'round4'/'summary.csv').exists():
  dist_conclusion += '新增的高负荷压力测试沿用冻结阈值，在预设主设置下，持续性同时取得最高的及时命中率和最低的虚警率。'
 if (root.parent/'round5'/'summary.csv').exists():
  dist_conclusion += '扰动位置留出的正式批次缺少可评分的越限事件，及时命中率在新位置的适用性仍待验证。'
  if (root.parent/'round4'/'summary.csv').exists() and (root/'quality-summary.csv').exists():
   dist_conclusion = ('本地校准在已有多机算例中恢复了部分及时命中。扰动前量测分级可以改善混合噪声下二次拟合的及时命中，实际虚警也须同时检查。'
    '在沿用冻结阈值的IEEE 39节点高负荷试验中，持续性基线在预设主设置下同时取得最高及时命中率和最低虚警率。'
    '新扰动位置的正式批次缺少可评分的越限事件，及时命中率在新位置的适用性仍待验证。')
   src=src.replace('DISTCONCLUSION 扰动前量测分级改善了混合噪声下二次拟合的及时命中。'
    '在IEEE 39节点，命中改善伴随实际虚警上升，持续性基线仍有竞争力。噪声突变试验表明，以扰动前噪声表征整个告警窗口的适用性取决于量测条件的稳定性。',
    'DISTCONCLUSION')
  if (root/'quality-summary.csv').exists() and (root/'ieee39quality-design.csv').exists():
   src=src.replace('扰动时噪声突变使分级策略失效风险增加。',
    '扰动时噪声突变会降低分级的可靠性。新扰动位置的正式测试缺少可评分的越限事件，及时命中表现仍待验证。')
   src=src.replace('An abrupt noise increase at the disturbance can invalidate the pre-event quality estimate.',
    'An abrupt noise increase can invalidate the pre-event quality estimate. The held-out disturbance locations yielded no scorable threshold crossings, so timely detection at those locations remains unverified.')
  src=src.replace('图件和复现入口。','图件和复现入口。求解诊断详见随稿补充材料。')
  supplement_tex = r'''
\documentclass[UTF8,a4paper,11pt]{ctexart}
\usepackage[margin=23mm]{geometry}
\usepackage{amsmath,booktabs}
\usepackage[hidelinks]{hyperref}
\title{量测质量变化下的电力系统短窗滚动低频预警\\补充材料}
\author{}
\date{}
\begin{document}
\maketitle
\section{场景筛选与求解诊断}
以下诊断沿用主文表9所列策略顺序（持续性、二次拟合、残差）。正式性能以冻结数据和告警阈值为准。
''' + render_round5(detailed=True) + r'''
\section{复现文件}
\begin{itemize}
\item \path{repro/README-rolling.md} 记录运行命令。
\item \path{repro/out/round5/} 保存位置留出清单、评分、中间阶跃扫描及R4收敛复算。
\item \path{repro/out/solver-crosscheck/} 保存三种积分方法及步长细化结果。
\item \path{repro/out/static-flow-check/} 保存静态潮流推进结果。
\item \path{repro/out/independent-time-check/} 保存六场景复算汇总以及各母线电压和限幅状态。
\end{itemize}
\end{document}
'''
  Path('artical/ARTICAL-rolling-supplement.tex').write_text(supplement_tex, encoding='utf-8')
Path('artical/ARTICAL-rolling.tex').write_text(src.replace('JOINT',joint).replace('CLEANTABLE',clean).replace('STRESSTABLE',stress).replace('CROSS',cross+supplement+focused).replace('DISTCONCLUSION',dist_conclusion),encoding='utf-8')





