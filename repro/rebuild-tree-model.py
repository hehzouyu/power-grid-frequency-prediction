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
#   已有脚本：ta_main_results.csv、ta_ablation.csv —— Project/ext/main_ta_pipeline.m（writetable）
#   缺生产脚本：ta_fair_compare.csv、ta_seg_seed10.csv —— 全仓库无任何 .m 的 writetable 或引用
"""T10 / 重建脚本 1 of 4 —— 提升树（LSBoost）已发表数值的聚合与对齐校验

【这是什么】
本文件是"重建脚本"（rebuild），不是作者的原始生成脚本。原始生成脚本不在本
工作区（详见文件头 COVERAGE 段）。

【重建哪一项结果：树模型】
提升树（LSBoost：100 棵树、学习率 0.1）在稿件中的 5 处已给数值：
  T-1 主表   提升树(手工三特征)         RMSE 0.0364 Hz / R^2 0.817   [表 5]
  T-2 消融   提升树(手工+VMD 8 维)      RMSE 0.0184 Hz / R^2 0.953   [表 6]
  T-3 对比   提升树(端点+分段斜率 4 维) RMSE 0.0114 Hz / R^2 0.982   [表 6]
  T-4 对比   提升树(秩 1：仅端点)       RMSE 0.0368 Hz / R^2 0.813   [表 6]
  T-5 十种子 分段斜率提升树             R^2 0.987+-0.002 / RMSE 0.0112+-0.0009 Hz

【重建口径（依据稿件描述 + 残留代码 + CSV 结构）】
  * 一行 CSV = "一次已训练模型在固定测试集上的一次评估"（含 RMSE_Hz、R2）。
  * 本脚本重建其"聚合层"：单次评估直接对齐；十种子按稿件"均值±标准差"聚合，
    标准差取总体口径 ddof=0 —— 只有它同时再现稿中的 0.002 与 0.0009
    （ddof=1 时 RMSE 标准差为 0.000954，按四位小数应为 0.0010，与稿值不符），
    两种口径的差异已记入 diagnostics，最终以作者确认为准。
  * 训练层不在本脚本内重建：696/232/232 的划分、LSBoost 100 棵 lr=0.1、以及
    Project/final_wrapup.m:15-16 记录的分段斜率三段差分
        sl = (fw(:,[9 17 25]) - fw(:,[1 9 17])) ./ (tw([9 17 25]) - tw([1 9 17]))
    需要原始场景数据与 MATLAB fitrensemble，本工作区不具备。

【输入（只读）】
  Project/results/tables/ta_main_results.csv
  Project/results/tables/ta_ablation.csv
  Project/results/tables/ta_fair_compare.csv
  Project/results/tables/ta_seg_seed10.csv
【输出】
  stdout: 对齐表（数值 / 来源 / 偏差）
  repro/out/rebuild-tree-model.json
【不声称】未运行 MATLAB；未生成任何新仿真数据；CSV 均为既有产物。
【需作者确认】FeatSet/Model 与特征配置的对应关系（本脚本按数值逐位吻合反推，
              见每行的 source 字段）；十种子标准差的 ddof 口径。
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

# 稿件保留位数：RMSE 到 4 位小数，R^2 到 3 位小数
DECIMALS = {"RMSE": 4, "R2": 3}
HALF_ULP = 0.5


def load(name: str) -> list:
    with (TABLES / name).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def row_at(rows, **conds):
    """返回 (CSV 数据行号, 行)。文件第 1 行是表头，故数据行号从 2 起。"""
    for lineno, row in enumerate(rows, start=2):
        if all(row.get(k) == v for k, v in conds.items()):
            return lineno, row
    raise LookupError(f"未找到满足 {conds} 的行")


def mean(xs):
    return sum(xs) / len(xs)


def pstd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def sstd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


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

    main_tab = load("ta_main_results.csv")
    abl = load("ta_ablation.csv")
    fair = load("ta_fair_compare.csv")
    seg = load("ta_seg_seed10.csv")

    checks = []

    # ---- T-1 主表：提升树(手工特征) -------------------------------------
    ln, r = row_at(main_tab, Model="提升树(手工特征)")
    src = f"ta_main_results.csv:{ln} 提升树(手工特征)"
    checks.append(align("T-1a", "RMSE", r["RMSE_Hz"], 0.0364, src))
    checks.append(align("T-1b", "R2", r["R2"], 0.817, src))

    # ---- T-2 消融：提升树(手工+VMD) -------------------------------------
    ln, r = row_at(abl, Model="提升树(手工+VMD)")
    src = f"ta_ablation.csv:{ln} 提升树(手工+VMD)"
    checks.append(align("T-2a", "RMSE", r["RMSE_Hz"], 0.0184, src))
    checks.append(align("T-2b", "R2", r["R2"], 0.953, src))

    # ---- T-3 对比：提升树(端点+分段斜率) --------------------------------
    ln, r = row_at(fair, FeatSet="2", Model="2")
    src = f"ta_fair_compare.csv:{ln} FeatSet=2 Model=2（端点+分段斜率 4 维）"
    checks.append(align("T-3a", "RMSE", r["RMSE_Hz"], 0.0114, src))
    checks.append(align("T-3b", "R2", r["R2"], 0.982, src))

    # ---- T-4 对比：提升树(秩 1：仅端点) ---------------------------------
    ln, r = row_at(fair, FeatSet="1", Model="2")
    src = f"ta_fair_compare.csv:{ln} FeatSet=1 Model=2（秩 1：仅端点）"
    checks.append(align("T-4a", "RMSE", r["RMSE_Hz"], 0.0368, src))
    checks.append(align("T-4b", "R2", r["R2"], 0.813, src))

    # ---- T-5 十种子：分段斜率提升树 -------------------------------------
    seg2 = [(i, r) for i, r in enumerate(seg, start=2) if r.get("Model") == "2"]
    r2s = [float(r["R2"]) for _, r in seg2]
    es = [float(r["RMSE_Hz"]) for _, r in seg2]
    first, last = seg2[0][0], seg2[-1][0]
    src = f"ta_seg_seed10.csv:{first}-{last} Model=2（n={len(seg2)} 个种子）"
    checks.append(align("T-5a", "R2", mean(r2s), 0.987, src + " R2 均值", "标准差口径见 T-5b"))
    checks.append(align("T-5b", "R2", pstd(r2s), 0.002, src + " R2 总体标准差(ddof=0)",
                        f"ddof=1 时为 {sstd(r2s):.6f}（→0.002，亦对齐）"))
    checks.append(align("T-5c", "RMSE", mean(es), 0.0112, src + " RMSE 均值"))
    checks.append(align("T-5d", "RMSE", pstd(es), 0.0009, src + " RMSE 总体标准差(ddof=0)",
                        f"ddof=1 时为 {sstd(es):.6f}（按四位小数应为 0.0010，与稿值不符 → 支持 ddof=0）"))

    # ---- 非门禁诊断：十种子手工三特征的标准差口径 ------------------------
    hand = [float(r["R2"]) for r in load("ta_seed10.csv")
            if r.get("FeatCfg") == "1" and r.get("Model") == "2" and r.get("R2") not in ("", "NaN")]
    diagnostics = [{
        "item": "十种子手工三特征(ta_seed10.csv FeatCfg=1,Model=2) R2",
        "n": len(hand),
        "mean": round(mean(hand), 6),
        "std_ddof0": round(pstd(hand), 6),
        "std_ddof1": round(sstd(hand), 6),
        "published": "0.863 +- 0.018",
        "verdict": "均值 0.8631→0.863 对齐；标准差 0.0185(ddof0)/0.0195(ddof1) 与稿值 0.018 相差约 1e-3，"
                   "需作者确认是四舍五入方式还是种子集差异（非本脚本对齐项）",
    }]

    # ---- 输出 -----------------------------------------------------------
    n_ok = sum(1 for c in checks if c["aligned"])
    print("=" * 108)
    print("rebuild-tree-model.py  |  对齐表：数值 / 来源 / 偏差（重建值 vs 稿件已给值）")
    print("=" * 108)
    print(f"{'ID':<6}{'指标':<6}{'重建值':>16}{'稿件值':>10}{'偏差':>12}{'容差':>10}  来源")
    print("-" * 108)
    for c in checks:
        print(f"{c['id']:<6}{c['metric']:<6}{c['rebuild_value']:>16.10f}{c['published_value']:>10.4f}"
              f"{c['deviation']:>12.2e}{c['tolerance_half_ulp']:>10.2e}  {c['source']}")
    print("-" * 108)
    print(f"对齐 {n_ok}/{len(checks)} 项；容差 = 半格舍入（R2 5e-4 / RMSE 5e-5）")
    for d in diagnostics:
        print(f"[诊断] {d['item']}: n={d['n']} mean={d['mean']} std(ddof0)={d['std_ddof0']} "
              f"std(ddof1)={d['std_ddof1']} vs 稿件 {d['published']}")

    payload = {
        "script": "rebuild-tree-model.py",
        "result": "树模型（LSBoost）已发表数值",
        "rebuild": True,
        "original_author_script": "partial",
        "author_confirmation_required": True,
        "matlab_run": False,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": ["Project/results/tables/ta_main_results.csv",
                   "Project/results/tables/ta_ablation.csv",
                   "Project/results/tables/ta_fair_compare.csv",
                   "Project/results/tables/ta_seg_seed10.csv"],
        "checks": checks,
        "checks_total": len(checks),
        "checks_aligned": n_ok,
        "all_aligned": n_ok == len(checks),
        "diagnostics": diagnostics,
        "limitations": [
            "训练层（数据划分、LSBoost 训练、分段斜率特征构造）未重建：其生产脚本不全、MATLAB 在沙箱内不可启动（见报告 §一）。",
            "CSV 的 FeatSet/Model 编号含义由数值逐位吻合反推，需作者确认。",
            "未运行 MATLAB，未生成任何新仿真数据。",
        ],
    }
    out = OUTDIR / "rebuild-tree-model.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON 已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
