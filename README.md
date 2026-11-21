# 量测质量变化下的电力系统短窗频率预测与滚动低频预警

Short-window frequency-nadir prediction and rolling underfrequency warning under varying measurement quality. This archive accompanies the verified November 2026 manuscript (v4).

## 源码与样本目录

| 目录 | 内容 |
| --- | --- |
| [Project/](Project/) | 早期 MATLAB 仿真、VMD、回归基线、扩展和测试，含早期 MAT 样本 |
| [repro/](repro/) | 封版论文各研究线的 MATLAB、Python 和 PowerShell 入口 |
| [repro/out/revision/](repro/out/revision/) | 原固定截止轨迹、特征、场景索引、逐场景预测和配置 |
| [repro/out/](repro/out/) | 近端点配对、ECGR、固定门控、风险、原滚动和多机数值压力试验等的样本及账本 |
| [滚动 ECGR 冻结档案](NewWorkspace/2026-11-21-ECGR滚动接入-v1/) | 场景、逐窗与逐事件记录、冻结模型、校准阈值、协议和复算入口 |
| [audit/](audit/) | 审计脚本、数值复算表、诊断规则及追溯索引 |
| [reports/](reports/) | 样本划分和实验协议 |

MAT、NPZ 和逐事件记录来自数值仿真。早期 MATLAB 样本与论文正式样本按原目录区分。保留正式数据原始文件名和相对路径，以便检查冻结哈希。

从仓库根目录运行 python repro/verify_public_samples.py 可只读验证关键文件、滚动接入的八项冻结哈希和主要工作点计数。此命令不会重新仿真、训练或写入样本。历史完整运行入口参见 [固定截止](repro/README-revision.md) 和 [滚动](repro/README-rolling.md) 说明。旧稿编译、部分完整复跑和历史审计入口仍引用原项目的本地中间产物及软件环境。源文件发布不代表已验证所有历史命令可在新机器完整重跑。

## 论文与复算

| 文件 | 内容 |
| --- | --- |
| [主稿](paper.pdf) · [补充材料](supplement.pdf) | 封版论文和附表 |
| [LaTeX 源码](latex-source.zip) | 主稿、补充材料和全部图件。使用 XeLaTeX，先编译 frequency-project-supplement.tex 再编译 frequency-project.tex |
| [代码及复算表](code-and-recomputed-tables.zip) | MATLAB 和 Python 研究代码，证据审计脚本、复算 CSV、数字追溯索引和文献核验表 |
| [固定截止证据](evidence-nadir.zip) | 近端点配对、ECGR、质量门控及其他固定截止预测记录 |
| [原滚动预警证据](evidence-rolling.zip) | 聚合仿真和 IEEE 数值压力检验账本 |
| [ECGR 滚动接入证据](evidence-rolling-ecgr.zip) | 逐窗、逐事件预测，冻结配置、告警阈值及验收 |
| [单侧下界证据](evidence-risk.zip) | 风险预测和未通过验收的记录 |
| [文件校验](SHA256SUMS.txt) | 下载文件的 SHA-256 |

公开证据为数值仿真归档。审计代码参考原目录结构，所依赖的本机历史中间产物并未全部随公开快照发布；已发布逐事件账本、复算表和对应脚本可用于检查论文统计。

固定截止质量门控通过全部 20 项预设条件。固定 ECGR 在 30 dB 白噪声条件下未通过尾部保护。ECGR 质量门控接入滚动的 10% 主验收未通过，单侧风险下界整体未通过。结论限定于论文所定义的数值仿真。
