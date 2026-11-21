# 滚动低频预警复现

在项目根目录运行。聚合实验读取已有修订版的 `repro/out/revision/trajectories.mat`，新增校准和测试轨迹由脚本生成。ANDES 动态算例使用独立 Python 环境中的 ANDES 2.0.0。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$andes='repro/.andes-env/Scripts/python.exe'
& $andes -c 'import andes; andes.prepare(quick=True,nomp=True,pycode_path="repro/out/rolling/andes-cache")'
& $andes repro/run_andes_warning.py ieee14
& $andes repro/run_andes_warning.py ieee39
python repro/run_rolling_warning.py
& $andes repro/run_andes_warning.py ieee14 --distributed
& $andes repro/run_andes_warning.py ieee39 --distributed
python repro/analyze_distributed_warning.py
& $andes repro/run_andes_warning.py ieee39 --focused
python repro/analyze_focused_warning.py
& $andes repro/run_andes_warning.py ieee39 --quality
python repro/analyze_quality_warning.py
python repro/round2_cluster_audit.py
& $andes repro/run_round3_scenarios.py --smoke
& $andes repro/run_round3_scenarios.py
python repro/score_round3.py
& $andes repro/run_round4_scenarios.py pilot
& $andes repro/run_round4_scenarios.py 0
& $andes repro/run_round4_scenarios.py 1
python repro/score_round4.py
& $andes repro/run_round5_scenarios.py pilot
& $andes repro/run_round5_scenarios.py 0
& $andes repro/run_round5_scenarios.py 1
& $andes repro/run_round5_scenarios.py audit
python repro/score_round5.py
& $andes repro/run_round5_scenarios.py transition-0
& $andes repro/run_round5_scenarios.py transition-1
python repro/score_round5.py --transition
& $andes repro/run_solver_crosscheck.py trapezoid
& $andes repro/run_solver_crosscheck.py backeuler
& $andes repro/run_solver_crosscheck.py qndf
& $andes repro/run_solver_crosscheck.py backeuler --refine 0.005
& $andes repro/run_solver_crosscheck.py backeuler --refine 0.0025
& $andes repro/run_solver_crosscheck.py summarize
& $andes repro/check_static_power_flow.py
& $andes repro/run_independent_time_check.py controls --step 0.005
& $andes repro/run_independent_time_check.py controls --step 0.0025
& $andes repro/run_independent_time_check.py failures --step 0.005
& $andes repro/run_independent_time_check.py failures --step 0.0025
& $andes repro/run_independent_time_check.py summarize
& $andes repro/run_independent_time_check.py sampling
python repro/build_rolling_paper.py
Push-Location artical
latexmk -xelatex -interaction=nonstopmode -halt-on-error ARTICAL-rolling.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error ARTICAL-rolling-supplement.tex
Pop-Location
```

`out/rolling/` 下的 `scenarios.csv` 记录聚合场景参数和数据角色，`observations.csv` 记录决策使用的采样索引、告警发出时间轴与逐时刻预测，`events.csv` 记录逐策略首次告警和真实越限时间，`warning-summary.csv` 和 `point-summary.csv` 提供汇总。无限值表示没有越限或没有告警，空提前量表示没有及时告警。多机算例的 `*-design.csv` 记录每个场景的仿真状态和量测点，`*-config.json` 记录运行版本与案例文件指纹。

`distributed-summary.csv` 与 `distributed-events.csv` 记录补充试验。模型沿用聚合开发集训练结果。本地阈值使用对应系统 1.00 倍负荷的未越限事件校准，0.94 和 1.06 倍负荷只作评估。噪声设置沿用聚合开发集定义的信号幅度参考。试验对比固定聚合阈值与本地阈值，保留全部不收敛记录并对实际计数报告 Wilson 区间。

`ieee39focus-design.csv` 记录96个新的定向压力场景。设计基于前轮筛出的负荷分布，按负荷水平隔离校准与测试。 `focused-scoring-index.csv` 给出角色与越限标记，`focused-summary.csv` 和 `focused-events.csv` 记录干净校准及已知噪声水平下的匹配校准结果。比较5%、10%与20%三个目标虚警水平，实际虚警率另行报告。

量测质量分级实验的固定口径见 [实验协议](../reports/quality-warning-protocol.md)。`quality-summary.csv` 汇总逐噪声和等权混合结果；`quality-paired.csv` 为分级相对固定阈值的物理事件配对差。`quality-estimates.csv` 保存测得的扰动前标准差及分级，`quality-thresholds.csv` 保存校准阈值，`quality-events.csv` 保存逐场景结果。`ieee39quality-design.csv` 含非收敛记录。执行 `--aggregate-only` 会先生成仅聚合的诊断结果，正式构建论文前运行完整分析。

R3 采用 [冻结协议](../reports/R3-frozen-protocol.md)。校准试算与正式批次分别写入 `out/round3/smoke/` 和 `out/round3/sealed/`，场景生成器会拒绝覆盖存在的批次目录。正式批次的 `frozen-design.csv` 记录预定场景，`design-results.csv` 保留全部状态，`trajectories.npz` 的 `ids` 对应成功场景。评分入口只读取正式批次，输出 `summary.csv`、`events.csv` 和 `thresholds.csv`。R2 统计补算见 [补充说明](../reports/round3/R2-statistical-addendum.md)。

R4 采用 [高负荷外推确认方案](../reports/R4-frozen-protocol.md)。正式分片 0 和 1 为同一冻结设计的不同物理场景，试算目录只用于环境检查。评分按 DesignID 合并正式分片并直接读取 R3 的 `thresholds.csv`。输出位于 `out/round4/`，包含逐事件数据、实际工作点、按物理事件聚类的配对差、按负荷水平分组结果和输入文件指纹。按负荷水平的分组结果只涉及该水平的收敛场景。

R5 采用 [扰动位置留出方案](../reports/R5-frozen-protocol.md)。`out/round5/by-pattern.csv` 按新扰动位置组合报告主分析。`out/round5/r4-audit/solver-check.csv` 单独记录 R4 的三档积分步长复算结果。`out/round5/transition/` 为正式结果出来后追加的探索性中间阶跃网格，沿用相同阈值，噪声种子为 39195021。

[求解方法交叉复算](../reports/solver-crosscheck-protocol.md) 将完整性与已接受轨迹上的频率越限分别记录。`out/solver-crosscheck/cases.csv` 汇总三种积分方法及后向欧拉细化结果。截断轨迹中未观察到越限时，`ObservedCrossingAfterStep_s` 留空，整个事件风险未知。

`out/static-flow-check/` 对相同场景记录独立交流潮流筛查。用 ANDES 算例参数组装导纳矩阵，以 SciPy 求解潮流方程。从扰动前工作点起，以阶跃目标的 5% 为增量推进至完整阶跃。负荷为恒定 PQ，发电机 PV 电压给定，设平衡母线补足功率，未施加无功限幅。`continuation.csv` 记录每步收敛性与母线电压，`summary.csv` 记录终点或首个失败步。该筛查以稳态可解性为检查对象。

[六场景时域复算](../reports/independent-time-check-protocol.md) 沿用 ANDES 动态方程、初始化和限幅逻辑，采用单独实现的 BDF2 时间推进和 SciPy 稀疏线性求解。`out/independent-time-check/summary.csv` 记录两档步长的验收结果，分目录记录 20 ms 采样下的各母线电压和抗积分饱和状态。`sampling-diagnostic.csv` 记录原积分轨迹按两种采样口径进行的对照，只用于解释差异。

投稿稿从 `artical/ARTICAL-submission.tex` 生成。执行 `python repro/audit_rolling_submission.py` 核对冻结数字，再执行 `python repro/build_submission_exports.py` 生成图 1、可编辑 Word 稿及配套补充材料源稿。用包含 `python-docx` 的文档运行环境执行 `repro/format_submission_word.py artical/ARTICAL-submission.docx` 设置 Word 样式。用 XeLaTeX 编译主稿和补充材料。复核范围和主要数字来源见 [投稿稿证据核对表](../reports/submission-evidence-audit.md)。
