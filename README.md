# 量测质量变化下的电力系统短窗频率预测与滚动低频预警

Short-window frequency-nadir prediction and rolling underfrequency warning under varying measurement quality. This archive accompanies the verified November 2026 manuscript (v4).

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
