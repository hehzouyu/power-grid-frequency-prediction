#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# REBUILD-SCRIPT: yes
# ORIGINAL-AUTHOR-SCRIPT: partial
# AUTHOR-CONFIRMATION-REQUIRED: yes
# MATLAB-NOT-RUN: yes
# =============================================================================
# -----------------------------------------------------------------------------
# 前提纠正（2026-11-19 自查）：工作区里并非“没有 .m 脚本” ——
#   * Project/ext/main_ta_pipeline.m 会写 ta_main_results / ta_ablation / ta_window /
#     ta_noise / ta_extrapolation / ta_grouped 六份 CSV；
#   * Project/final_wrapup.m 会写 ta_all1200_baseline / ta_noise_map95；
#   * 13 份 .m 调用 fitrensemble，原始数据 data/raw/dataset_{raw,ta}.mat 也在盘。
# 真正缺的是本项结果所依赖的部分 CSV 的生产脚本（COVERAGE 见下）与沙箱内可运行的
# MATLAB（/d/matlab 因写权限被拒无法启动）。故本文件仍为重建、非原始生成脚本。
# COVERAGE:
#   已有脚本：无
#   缺生产脚本：ta_independent.csv —— 全仓库无任何 .m 的 writetable 或引用
"""T10 / 重建脚本 2 of 4 —— 独立确认集（500 场景）已发表数值的聚合与对齐校验

【这是什么】
本文件是"重建脚本"（rebuild），不是作者的原始生成脚本。原始生成脚本不在本
工作区（详见文件头 COVERAGE 段）。

【重建哪一项结果：独立确认】
稿件表 12 / 4.9 节给出的 3 组配置在 500 组连续参数场景上的结果：
  I-1 Cfg1 RMSE 0.0327 Hz / R^2 0.806
  I-2 Cfg2 RMSE 0.0214 Hz / R^2 0.917
  I-3 Cfg3 RMSE 0.0134 Hz / R^2 0.967（端点+分段斜率，仍为最优）
并核对"500 场景"这一结构事实与"顺序不变"这一定性表述。

【重建口径】
  * 一行 CSV = 某一配置在独立确认集上的一次整集评估（N_test 列给出场景数）。
  * 重建层为：整集指标取值、场景数一致性、三配置的相对顺序。
  * 训练层（冻结主测试集上选定的配置、在独立集上重放）不在本脚本内重建：
    需要原始场景生成器（连续参数）与 MATLAB，本工作区不具备。
  * X-1 为记录项而非对齐项：独立集 R^2 0.96749 相对主测试 0.98194 低 0.01445，
    稿件据此表述"保持最优、无性能坍塌"。该差值不是稿件给出的数字，故不做
    通过/不通过判定，只登记事实并核对"顺序不变"这一可判定陈述。

【输入（只读）】
  Project/results/tables/ta_independent.csv
  Project/results/tables/ta_fair_compare.csv（仅用于取主测试同配置值做 X-1）
【输出】
  stdout: 对齐表（数值 / 来源 / 偏差）
  repro/out/rebuild-independent-confirm.json
【不声称】未运行 MATLAB；未生成任何新仿真数据；独立集场景由既有 CSV 记录。
【需作者确认】独立确认集的随机种子未在 CSV 中留存；Cfg 编号与特征的对应关系
              由数值与稿件顺序反推。
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import sys

# 控制台容错（2026-11-19 修）：Windows GBK 代码页无法编码 ⇒/≤/Δ 等符号，
# 会让脚本在打印结论行时抛 UnicodeEncodeError，把“已通过”变成退出码 1。
# 保持当前编码、只把不可编码字符替换掉，避免中文变乱码。
try:  # pragma: no cover - 仅影响输出，不影响计算
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, OSError):
    pass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = ROOT / "Project" / "results" / "tables"
OUTDIR = HERE / "out"

DECIMALS = {"RMSE": 4, "R2": 3}
HALF_ULP = 0.5

# 稿件表 12 的已给值（Cfg -> (RMSE, R^2)）
PUBLISHED = {"1": (0.0327, 0.806), "2": (0.0214, 0.917), "3": (0.0134, 0.967)}
N_SCENARIO_PUBLISHED = 500


def load(name: str) -> list:
    with (TABLES / name).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def row_at(rows, **conds):
    for lineno, row in enumerate(rows, start=2):
        if all(row.get(k) == v for k, v in conds.items()):
            return lineno, row
    raise LookupError(f"未找到满足 {conds} 的行")


def align(cid, metric, value, published, source, note=""):
    nd = DECIMALS[metric]
    dev = abs(float(value) - float(published))
    tol = HALF_ULP * 10 ** (-nd) + 1e-12
    return {
        "id": cid,
        "metric": metric,
        "rebuild_value": round(float(value), 12),
        "published_value": float(published),
        "deviation": float(f"{dev:.3g}"),
        "tolerance_half_ulp": float(f"{tol:.3g}"),
        "aligned": bool(dev <= tol),
        "source": source,
        "note": note,
    }


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    ind = load("ta_independent.csv")
    fair = load("ta_fair_compare.csv")

    checks = []
    for cfg in ("1", "2", "3"):
        ln, r = row_at(ind, Cfg=cfg)
        src = f"ta_independent.csv:{ln} Cfg={cfg}（N_test={r['N_test']}）"
        pub_rmse, pub_r2 = PUBLISHED[cfg]
        checks.append(align(f"I-{cfg}a", "RMSE", r["RMSE_Hz"], pub_rmse, src))
        checks.append(align(f"I-{cfg}b", "R2", r["R2"], pub_r2, src))

    # ---- 结构事实：场景数 500 -------------------------------------------
    n_vals = sorted({int(r["N_test"]) for r in ind})
    checks.append({
        "id": "I-7",
        "metric": "N_test",
        "rebuild_value": n_vals,
        "published_value": N_SCENARIO_PUBLISHED,
        "deviation": 0.0,
        "tolerance_half_ulp": 0.0,
        "aligned": n_vals == [N_SCENARIO_PUBLISHED],
        "source": "ta_independent.csv:2-4 N_test 列",
        "note": "稿件称独立确认集含 500 组连续参数场景；CSV 三行均为 500。",
    })

    # ---- X-1 记录项：独立集 vs 主测试同配置 -----------------------------
    _, fair32 = row_at(fair, FeatSet="2", Model="2")   # 端点+分段斜率 主测试
    _, ind3 = row_at(ind, Cfg="3")                     # 端点+分段斜率 独立集
    r2_main = float(fair32["R2"])
    r2_ind = float(ind3["R2"])
    order = [float(row_at(ind, Cfg=c)[1]["R2"]) for c in ("1", "2", "3")]
    order_ok = order == sorted(order)
    informational = [{
        "id": "X-1",
        "item": "独立确认集(端点+分段斜率) vs 主测试同配置 R^2",
        "main_test_R2": round(r2_main, 6),
        "independent_R2": round(r2_ind, 6),
        "delta": round(r2_ind - r2_main, 6),
        "published_statement": "独立确认集上仍为最优、无性能坍塌（稿件 4.9 节，定性表述）",
        "order_stable": bool(order_ok),
        "verdict": "三配置 R^2 在独立集上仍按 Cfg1<Cfg2<Cfg3 单调递增（顺序不变），"
                   "与稿件定性表述一致；0.0145 的绝对下降属跨集泛化差异，稿件未给出该差值，"
                   "故不判对齐、仅登记事实。",
    }]

    n_ok = sum(1 for c in checks if c["aligned"])
    print("=" * 112)
    print("rebuild-independent-confirm.py  |  对齐表：数值 / 来源 / 偏差（重建值 vs 稿件已给值）")
    print("=" * 112)
    print(f"{'ID':<7}{'指标':<7}{'重建值':>16}{'稿件值':>10}{'偏差':>12}{'容差':>10}  来源")
    print("-" * 112)
    for c in checks:
        rv = c["rebuild_value"]
        rvs = f"{rv:.10f}" if isinstance(rv, float) else str(rv)
        pv = c["published_value"]
        pvs = f"{pv:.4f}" if isinstance(pv, float) else str(pv)
        print(f"{c['id']:<7}{c['metric']:<7}{rvs:>16}{pvs:>10}{c['deviation']:>12.2e}"
              f"{c['tolerance_half_ulp']:>10.2e}  {c['source']}")
    print("-" * 112)
    print(f"对齐 {n_ok}/{len(checks)} 项；容差 = 半格舍入（R2 5e-4 / RMSE 5e-5）")
    for x in informational:
        print(f"[记录项] {x['item']}: 主测试 {x['main_test_R2']} vs 独立集 {x['independent_R2']} "
              f"(Δ={x['delta']:+.6f})；顺序不变={x['order_stable']}")

    payload = {
        "script": "rebuild-independent-confirm.py",
        "result": "独立确认集（500 场景）已发表数值",
        "rebuild": True,
        "original_author_script": "partial",
        "author_confirmation_required": True,
        "matlab_run": False,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": ["Project/results/tables/ta_independent.csv",
                   "Project/results/tables/ta_fair_compare.csv"],
        "checks": checks,
        "checks_total": len(checks),
        "checks_aligned": n_ok,
        "all_aligned": n_ok == len(checks),
        "informational": informational,
        "limitations": [
            "独立确认集的随机种子与场景生成器未留存，无法重建数据本身，只能核验整集指标；ta_independent.csv 的生产脚本全仓库无引用。",
            "Cfg 编号与特征配置的对应关系由稿件顺序与数值反推，需作者确认。",
            "未运行 MATLAB，未生成任何新仿真数据。",
        ],
    }
    out = OUTDIR / "rebuild-independent-confirm.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON 已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
