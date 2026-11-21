"""生成 reports/wp-member-agent-11-T1.json（第三层专家结算兜底，含三阶段证据与 report）。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def meta(rel: str) -> dict:
    p = ROOT / rel
    data = p.read_bytes()
    return {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


artifacts = [
    meta("reports/r2/record-field-spec.md"),
    meta("reports/r2/record-schema.json"),
    meta("reports/r2/field-list.csv"),
    meta("reports/r2/record-template.json"),
    meta("repro/validate_r2_record.py"),
    meta("reports/r2/_bitwise_probe.py"),
]

doc = {
    "task": "T1 产出 R2 多噪声实现实验协议 —— 设计 R2 逐实现可复现的记录字段规范并交付字段清单",
    "member": "可复现工程专家 · 陈默",
    "delegatedBy": "研究分析部门负责人",
    "subtaskId": "T1",
    "workingDir": str(ROOT),
    "status": "done",
    "artifacts": artifacts,
    "acceptance": [
        {
            "criterion": "质量要求 1：凭种子可逐位重放任一噪声实现",
            "result": "pass",
            "evidence": "reports/r2/_bitwise_probe.py 用真实 observe() 代码路径实测：同进程双跑 sha256 同为 241f6040...f89d8，max|Δ|=0.0；两个独立进程输出逐字相同。Schema 对 replay.level=bitwise 强制 verification.mode=double-run 且双跑哈希逐项相同，校验器实测会拒掉哈希不一致的记录。",
        },
        {
            "criterion": "质量要求 2：字段含随机源、输入输出哈希、模型版本与失败分母",
            "result": "pass",
            "evidence": "field-list.csv 159 项分 11 组：B 随机源与种子 24 项、D 输入与哈希 9 项、E 模型与版本 11 项（含 sourceSha256）、G 分母与失败 19 项（eventsScored/Failed、failures[] 7 字段）、K 输出与哈希 4 项。",
        },
        {
            "criterion": "质量要求 3：字段与现有 quality 表、design 表可对接",
            "result": "pass",
            "evidence": "metrics.columns 强制覆盖 quality-summary 全部 19 列（校验器逐列比对，缺列即 FAIL）；thresholds[] 强制带 LowQuality/CalibrationVariants/Threshold_pu 三个同名列；dataset.design.joinKey 固定 DesignID；compatibility.block 声明仅新增 NoiseImpl 一列、join 键与 Existing 一致且不改原列序。",
        },
        {
            "criterion": "交付字段清单（主产物）",
            "result": "pass",
            "evidence": "reports/r2/field-list.csv：11 组 A10 B24 C14 D9 E11 F15 G19 H26 I13 J14 K4 = 159 行，逐行标 group/groupName/field/type/required/source/note。",
        },
        {
            "criterion": "规范可被机器校验、样例可通过",
            "result": "pass",
            "evidence": "python repro/validate_r2_record.py reports/r2/record-template.json → PASS（1 条提示：单条实现记录 1/5 未跑满，属预期）；4 个变异体全部 FAIL 并给出定位。",
        },
        {
            "criterion": "存量记录差距有实测依据",
            "result": "pass",
            "evidence": "python repro/validate_r2_record.py --audit-legacy → 6 份存量 run-record.json，覆盖 3.9%–23.7%，均未达 R2；差异项已列清单，并给出 seed.observe→seeds.observationNoise 等命名映射。",
        },
    ],
    "phases": {
        "planning": {
            "problemAnalysis": [
                "目标与验收：不是写一段“要记什么”的说明，而是产出可被机器校验、且能让第三方凭种子位级重放的记录契约；验收落在三条质量要求（位级可重放、含随机源/哈希/版本/失败分母、可与 quality 表与 design 表对接）。",
                "现状与事实：项目已有 run-record.json 单实现记录，噪声由 analyze_quality_warning.py:20-33 observe() 生成，种子 39192514 硬编码在函数内；quality-summary 19 列、quality-thresholds 7 列、design 表 144 行、split-v1 144 行（72/72）已稳定。",
                "约束：动手不能碰既有产物（guard.protectedRoots 覆盖 repro/out/rolling 等），故一切验证只用只读读取与临时样例；R2 只允许观测种子变，其余种子冻结，否则归因混淆；字段必须能与现有扁平 CSV 列名对齐。",
                "未知项：一个种子覆盖哪一级（场景级还是整块张量）在代码里没有显式声明；失败分母在现有汇总里没有单独列；“位级”在跨环境时是否还成立无证据。",
                "主要矛盾：可重放要求记录“足够细”（种子、抽取顺序、张量形状、库版本全记），而可用性要求记录“足够薄”（能塞进现有 CSV 体系、能与 quality 表 join）。过细则无人填，过薄则重放不了。",
                "边界：只定义记录与校验，不实现 R2 实验本身（≥5 次重跑由后续任务执行）；只覆盖噪声研究这一条链路，不泛化为全项目 schema；不修改既有脚本的随机行为。",
            ],
            "approach": [
                "第一步先读后写：把 run-record.json、quality-summary/thresholds/estimates/events 表头、design 表与 split 表头读齐，字段先归入“可复用/需扩展/缺失”三类，避免另造同义字段。",
                "第二步钉死复现单元与种子语义：以“一次实现 × 一次运行”为记录粒度；用一个种子 = 一整块 (n, T+50) 噪声张量、6 条件共享同一张量只改调制系数的配对结构，作为种子契约写进 rng.shapeContract 与 drawOrder，并明确“形状变则同种子≠同实现”。",
                "第三步编字段：分 A–K 十一组，每组只回答一个问题（是谁、随机源、场景、输入、模型、运行、分母、指标、跨实现、复现、输出），逐字段标 required 与 source（reuse/extend/new）以暴露与旧记录的差异。",
                "第四步把“位级”做成可执行判据而不是形容词：level 三档 bitwise/numeric/parameter-only，bitwise 强制 double-run 且双跑哈希逐项相同，跨环境降级 numeric 并用 1e-9 pu 容差。",
                "第五步先验证再交付：写 JSON Schema 与校验器（结构 + 语义十项），生成样例跑通，再做四类变异体证明校验器会拒，最后跑存量覆盖审计确认差距清单。",
                "第六步自证位级：写 _bitwise_probe.py 直接调用真实 observe()，同进程双跑与跨进程各跑一次比对 sha256，用实测而非声明支撑 bitwise 档位。",
            ],
            "approachBasis": [
                "第一轮的“现状与事实”指出种子硬编码在 observe() 内、quality 表列名稳定 → 决定第一步先读齐表头并按组归类，而不是先设计字段再回头找表。",
                "第一轮识别的主要矛盾（足够细 vs 足够薄）→ 决定用“组 + required/source 标注”承载细度，用“不新增 CSV 列、仅加 NoiseImpl 一列”保住薄度。",
                "第一轮的未知项“种子作用域”与代码里 6 条件共享 noise 的事实 → 决定把种子契约写成张量形状 + 4 步抽取顺序，并禁止按条件分种子。",
                "第一轮未知项“跨环境位级是否成立” → 决定引入 numeric 档与 1e-9 容差，并在本环境实测位级成立，从而 bitwise 有依据。",
                "第一轮约束“不能碰既有产物” → 决定验证全部走只读调用 + reports/r2 下的样例与变异体，_bitwise_probe 只读调用 observe 写内存。",
            ],
            "constraints": [
                "不修改既有脚本随机行为，不写入 repro/out/rolling 等受保护目录。",
                "R2 实现间只允许 observationNoise 变；scenarioPopulation/modelFit/bootstrap 冻结。",
                "字段须与 quality 表 19 列、design 表 DesignID 零改动对接。",
            ],
            "unknowns": [
                "跨环境（不同 numpy/BLAS）位级是否仍成立 —— 本规范按降级 numeric 处理。",
                "aggregate 数据集无 design 对应，故 scenarioSource=simulated 时 design 必须为 null。",
            ],
        },
        "execution": {
            "steps": [
                "读取 repro/out/round2/persistence/run-record.json、quadratic-alarm/run-record.json 与 quality-* 表头、design/split 表头，完成字段三分类。",
                "写 reports/r2/record-field-spec.md（v1.0.0）：随机源语义、单点可变、失败分母恒等式、对接契约、复现流程、A–K 字段分组。",
                "写 reports/r2/record-schema.json（JSON Schema 2020-12）：recordId pattern、bitwise 强制 double-run、design vs simulated 二选一、series 块。",
                "写 reports/r2/field-list.csv 共 159 行，并修掉 3 处 note 内逗号导致的错列。",
                "写 repro/validate_r2_record.py：结构校验 + 语义十项 + --audit-legacy 覆盖审计。",
                "写 repro/make_r2_template.py 生成 reports/r2/record-template.json（真实 split/design/trajectories/模型脚本哈希），迭代修正 algorithm=PCG64、heldFixed 不含 split 两处约束不符，直至 PASS。",
                "写 reports/r2/_bitwise_probe.py 调用真实 observe()，同进程双跑 + 两进程各跑一次，比对 sha256。",
                "构造 4 个变异体（分母破裂、parameter-only、双跑哈希不一致、删 thresholds）证明校验器会拒。",
                "跑 --audit-legacy 得存量 6 份记录覆盖率 3.9%–23.7%，并据 seed.observe/doubleRun 实测补写命名映射。",
            ],
            "deviations": [
                "样例生成时曾把 rng.algorithm 写成解释性长串、heldFixed 里多放 split，被 Schema 拒；按契约改为 PCG64、heldFixed 仅四类种子，未放宽 Schema（不为了让样例过而改判据）。",
                "字段数初稿按 145 计，实测 159（D 9、F 15、H 26、I 13、J 14、K 4），以实测为准回改规范表格，不迁就初稿数字。",
            ],
        },
        "verification": {
            "allPassed": True,
            "requirementCheck": "三条质量要求逐条命中：1 位级可重放（实测 sha256 相同 + Schema 强制双跑）；2 随机源/哈希/版本/失败分母四类字段齐备（B/D/E/G/K）；3 quality 表 19 列与 thresholds 同名列、design 表 DesignID 零改动对接（校验器逐列比对）。",
            "checks": [
                {
                    "criterion": "质量要求 1 位级重放",
                    "method": "python reports/r2/_bitwise_probe.py 连跑两次 + diff",
                    "evidence": "exit=0/0；run1=run2=241f604081815a177bbe6245c915fa5993dd41c0cf47f4289099cb288e9f89d8；identical=True；est max|Δ|=0.0；两进程输出完全一致。",
                },
                {
                    "criterion": "质量要求 2 字段齐备",
                    "method": "awk 统计 field-list.csv 分组与必填数",
                    "evidence": "A10 B24 C14 D9 E11 F15 G19 H26 I13 J14 K4 = 159；必填 152 项。",
                },
                {
                    "criterion": "质量要求 3 表对接",
                    "method": "python repro/validate_r2_record.py reports/r2/record-template.json",
                    "evidence": "PASS；check_metric_columns 对 19 列逐列比对通过，check_threshold_join 对 LowQuality/CalibrationVariants/Threshold_pu 通过，compatibility 声明 NoiseImpl 单列新增。",
                },
                {
                    "criterion": "规范可校验且校验器非空转",
                    "method": "4 个变异体分别跑校验器",
                    "evidence": "denominator_break→2 项 FAIL；param_only→FAIL；bitwise_mismatch→FAIL；missing_thresholds→结构 FAIL。",
                },
                {
                    "criterion": "存量差距有实测",
                    "method": "python repro/validate_r2_record.py --audit-legacy",
                    "evidence": "6 份记录覆盖 36/152=23.7%、36/152=23.7%、6/152=3.9%、7/152=4.6%、7/152=4.6%、8/152=5.3%。",
                },
            ],
            "retrospective": {
                "problems": [
                    "field-list.csv 初稿有 3 行 note 内含英文逗号，导致列错位（NF=8/9），机器解析会错位取值；根因是手写 CSV 时把 note 当自然语言而非字段。",
                    "样例生成脚本引用了不存在的 reports/fix/t6_quadratic_alarm.py，FileNotFoundError；根因是凭印象写路径而未先用 glob 核对真实文件名。",
                    "Schema 与样例两处约束不一致（algorithm 期望常量 PCG64、heldFixed 枚举不含 split）；根因是先写 Schema 后写样例，未在写样例时回读枚举。",
                    "校验器初版把 drawOrder 的第 4 步按 lfilter 匹配，而 Schema 里该步 name 是 correlated，语义校验会误报；根因是拿代码里的函数名当字段名。",
                ],
                "detours": [
                    "先写了无哈希的样例再回头补真实哈希，中间返工一次生成器。",
                    "曾想把 split 放进 heldFixed 表达“划分冻结”，被枚举拒绝后改用 series.frozenSplitPreserved 承载，属一次无效尝试。",
                ],
                "resolutions": [
                    "用 awk NF!=7 定位错列行，把 note 内逗号改为分号/顿号并重跑统计确认 159 行 7 列。",
                    "改用 ops_glob 列出 reports/fix/*.py，按真实文件名 t6_fit_quadratic.py 修正并重新生成。",
                    "用脚本打印 Schema 的 required 与 enum，逐条对齐后重生成样例，直至 PASS。",
                    "把 drawOrder 校验改为按 name 集合（noise/noise_ar/zi/correlated）判定，并注明 correlated 即 lfilter 那一步。",
                ],
                "prevention": [
                    "遇到手写 CSV，先跑 `awk -F, 'NF!=预期列数'` 自查列数，再统计行数，最后才声明字段数。",
                    "遇到引用项目内脚本路径，先用 ops_glob 列目录核对真实文件名，不凭印象写路径。",
                    "遇到“Schema + 样例”成对交付，先打印 Schema 的 required/enum 再写样例，样例必须跑校验器 PASS 才能计入产物。",
                    "遇到校验规则，先分清“代码里的函数名”和“契约里的字段名”，校验规则一律以 Schema 里的字段名为准。",
                ],
            },
        },
    },
    "report": {
        "keyResults": [
            "产出 R2 逐实现记录字段规范 v1.0.0（159 字段分 A–K 十一组）与可校验 JSON Schema、完整样例、校验器，样例实测 PASS。",
            "实测位级重放成立：真实 observe() 同进程双跑与跨进程各跑，输出 sha256 全同（241f6040…f89d8），max|Δ|=0.0。",
            "校验器非空转：分母破裂、parameter-only、双跑哈希不一致、缺 thresholds 四类变异体全部被拒并定位。",
            "存量差距量化：6 份 run-record 对本规范覆盖仅 3.9%–23.7%，均已不满足 R2。",
        ],
        "keyInfo": [
            "一个种子 = 一整块 (n, 1051) 噪声张量，6 条件共享同一张量只改调制系数，故禁止按条件分种子；形状变则同种子≠同实现。",
            "R2 实现间只允许 observationNoise 变，scenarioPopulation(39192512/39192513)、modelFit(42)、bootstrap(39192515/39192517) 必须冻结。",
            "失败分母恒等式：eventsScored=eventsPlanned−eventsFailed，riskUnits+nonRiskUnits=eventsScored，零分母写 null+metricNullReason 而非 0。",
            "ieee39_multi 测试分母实测：432 事件单元、risk 126、nonRisk 306、calibrationNegatives 55；mixed 行为 288/72/84。",
        ],
        "importantInfo": [
            "aggregate 数据集 2400 场景为 simulate(params) 合成编号，design 表无对应，故 scenarioSource=simulated 时 dataset.design 必须为 null。",
            "存量 quadratic-alarm 记录已有真实 doubleRun（两份产物 run1=run2），命名映射为 seed.observe→seeds.observationNoise、doubleRun→replay.verification。",
            "跨环境位级未验证，规范按 numeric 档 + 1e-9 pu 容差降级，不宣称跨环境字节相同。",
            "本规范只定义记录与校验，未实现 ≥5 次重跑本身；R2 实验执行需后续任务按 §7 复现流程进行。",
            "dsh_workplan_verify 在叶子上返回 NO_PLAN（父会话无可关联计划，叶子无法自核验），故三阶段证据以本文件与证据块提交，由负责人侧的 T1 计划核验。",
        ],
        "leadRequirements": [
            "字段清单：reports/r2/field-list.csv（159 行，含 required/source 标注）。",
            "规范正文：reports/r2/record-field-spec.md（含 §10 落地资产与实测证据）。",
            "机器可校验：reports/r2/record-schema.json + reports/r2/record-template.json（PASS）。",
            "验证工具：repro/validate_r2_record.py（结构+语义+存量审计，--audit-legacy 可复跑）。",
            "位级证据：reports/r2/_bitwise_probe.py 双跑 sha256 相同，报告见本文件 acceptance。",
        ],
    },
}

out = ROOT / "reports" / "wp-member-agent-11-T1.json"
out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# 自检：report 四项条数与单条长度
for key, items in doc["report"].items():
    assert 1 <= len(items) <= 6, f"{key} 条数越界 {len(items)}"
    for it in items:
        assert len(it) <= 200, f"{key} 超长 {len(it)}: {it[:30]}"
        assert len(set(items)) == len(items), f"{key} 有重复项"
print(f"wrote {out}  ({out.stat().st_size} bytes); report 四项自检通过")
