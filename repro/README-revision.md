# 修订稿复现实验

本目录的新版流程对应 `artical/ARTICAL-revised.tex`。历史稿件对应的 MAT、CSV 和核对脚本仍用于查阅旧结果。新版校正扰动积分起点，并使用 scikit-learn 的平方损失梯度提升回归树，数值应与本目录的新版产物比对。

投稿复现附件 `revision-reproducibility.zip` 含 MATLAB 与 Python 入口、场景索引、逐场景预测、汇总结果、图表及校验清单。体积较大的两个 MAT 中间产物保存在本地项目，解压附件后运行 MATLAB 入口即可重新生成。

## 执行

在项目根目录执行下列命令。

```powershell
matlab -batch "addpath('repro'); run_revision"
python repro/analyze_revision.py
Push-Location artical
latexmk -xelatex -interaction=nonstopmode -halt-on-error ARTICAL-revised.tex
Pop-Location
```

MATLAB 需要可调用 `vmd`，本次运行版本为 R2025a。Python 需要 NumPy、SciPy、pandas、scikit-learn 和 Matplotlib。版本见 `out/revision/config.json` 与 `out/revision/python-config.json`。Word 稿另用 Pandoc 和 python-docx 构建，入口为 `python repro/build_revision_word.py`。

## 短窗形状主稿

以下命令读取已有的 `out/revision`，生成 `out/nadir-shape`，并编译当前主稿。MATLAB 导出只用基础数组和最小二乘运算。Python 入口在缺少短窗 MATLAB 导出时可按相同公式读取冻结轨迹，并核对原 0.50 s 结果。机理示例在既有 SciPy 环境中积分。

```powershell
matlab -batch "addpath('repro'); export_nadir_windows"
python repro/analyze_nadir_shape.py
Push-Location artical
latexmk -xelatex -interaction=nonstopmode -halt-on-error ARTICAL-nadir-restored.tex
Pop-Location
```

[固定标签短窗规程](../reports/nadir-window-protocol.md) 记录样本、特征和配对区间口径。`anchor-check.csv` 核对原0.50 s预测，`window-summary.csv` 与 `window-predictions.csv` 保存矩阵和逐事件值。`boundary-decomposition.csv`、`paired-white-noise.csv` 和 `paired-structure.csv` 保存三个边界分析的数字。`matched-endpoint.csv` 保存频率、调频增功及限幅判据，`derivative-check.csv` 保存导数步长核查。

已有 `window-predictions.csv` 时，可执行 `python repro/analyze_nadir_shape.py --window-contrast-only`，从相同场景的逐窗绝对误差计算 `cross-window-gain.csv`。

增强试验执行 `matlab -batch "addpath('repro'); run_nadir_enhancements"`，再执行 `python repro/analyze_nadir_enhancements.py`。输出位于 `out/nadir-enhancements/`。计时因运行环境而异，补充材料记录本次使用的测量。用 XeLaTeX 编译 `artical/ARTICAL-nadir-supplement.tex`。在含有 python-docx 和 Pandoc 的文档环境中执行 `repro/build_revision_word.py --nadir`，从当前 LaTeX 源稿生成 Word。

主稿的局部可辨识性与预测区间使用以下独立复现实验。先生成相应数据和图件，再编译主稿及补充材料。

```powershell
matlab -batch "addpath('repro'); run_nadir_identifiability"
python repro/analyze_nadir_identifiability.py
matlab -batch "addpath('repro'); run_nadir_intervals"
python repro/analyze_nadir_intervals.py
```

输出分别位于 `out/nadir-identifiability/` 和 `out/nadir-intervals/`。逐场景口径见[局部可辨识性结果](../reports/nadir-identifiability-results.md)和[区间结果](../reports/nadir-interval-results.md)。

## 数据与指标

| 文件 | 内容 |
|---|---|
| `out/revision/grid-index.csv` | 1 200 个参数组合、事件类别、标签及固定划分。编号从 1 起，`InFullTest` 指扩展测试。 |
| `out/revision/continuous-index.csv` | 500 个新连续参数场景。 |
| `out/revision/cascade-index.csv` | 同一组 500 个参数用于两级串联调速器，编号与连续场景逐一对应。 |
| `out/revision/trajectories.mat` | 上述三组仿真轨迹和时间网格。 |
| `out/revision/features.mat` | 干净、跨模型、加噪和窗口长度实验的输入特征。 |
| `out/revision/predictions.csv` | 场景、模型、特征、试验设置、真值与预测。数据集由 `Experiment` 区分。 |
| `out/revision/summary.csv` | 绝对误差、决定系数和 49.75 Hz 示例阈值混淆计数。噪声与划分重复逐次保存。 |
| `artical/revision-assets/` | 稿件使用的自动生成指标、表格和图件。 |

主条件测试由 232 个最低点晚于截止时刻的未见网格场景组成。扩展测试由这 232 个场景与全部 46 个边界事件组成，因此早期事件占比高于原始网格。连续参数与结构变化检验包含各自的全部 500 个场景。两级模型测试只验证一种受控结构变化。

网格划分使用 MATLAB 的种子 42。划分敏感性使用 NumPy 的种子 1 至 10，各次复用同一网格。惯量外推仅以低于 6 s 的条件场景训练。训练集统计量用于标准化。所有预测使用严格晚于观察截止时刻的采样点来定义标签，场景分组则使用完整仿真轨迹。
