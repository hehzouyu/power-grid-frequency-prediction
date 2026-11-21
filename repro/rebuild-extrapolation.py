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
#   已有脚本：ta_extrapolation.csv —— Project/ext/main_ta_pipeline.m（writetable）
#   缺生产脚本：ta_extrap_vmd.csv、ta_seg_extrap.csv —— 全仓库无任何 .m 的 writetable 或引用
"""T10 / 重建脚本 3 of 4 —— H=6 s 单侧外推已发表数值的聚合与对齐校验

【这是什么】
本文件是"重建脚本"（rebuild），不是作者的原始生成脚本。原始生成脚本不在本
工作区（详见文件头 COVERAGE 段）。

【重建哪一项结果：外推检验（稿件 4.7 节）】
  E-1 提升树(手工三特征) 在 H=6 s 单侧外推：R^2 0.578 / RMSE 0.0464 Hz
  E-2 仅端点(秩 1)：                            R^2 0.574 / RMSE 0.0467 Hz
  E-3 端点+VMD，K=4(全网格选定)：               R^2 0.816
  E-4 端点+分段斜率：                           R^2 0.913 / RMSE 0.0211 Hz
  E-5 端点+VMD，K 在 H<=5 开发集重选：          R^2 0.835
  E-6 结构性陈述：0.913 > 0.835 > 0.574（三种配置的排序）
  E-7 结构性陈述：H=6 外推测试集共 240 组，全部未参与训练

【重建口径】
  * 一行 CSV = 某一配置在 H=6 外推集上的一次整集评估（N_test 列给出场景数）。
  * 重建层：整集指标取值、场景数一致性、三配置相对排序。
  * 训练层（把训练集限制在 H<=5、重新训练、在 H=6 上预测）不在本脚本内重建：
    需要原始 1200 场景数据与 MATLAB fitrensemble，本工作区不具备。
  * E-2 的 R^2 存在半格容差外 6e-7 的边界偏差（CSV 0.573494 按三位小数为 0.573，
    稿中 0.574 对应原始值 >=0.5735），已在 note 中如实登记，不掩盖。

【输入（只读）】
  Project/results/tables/ta_extrapolation.csv   （手工三特征，H=6 整集）
  Project/results/tables/ta_extrap_vmd.csv      （Cfg1=秩1，Cfg2=端点+VMD K=4）
  Project/results/tables/ta_seg_extrap.csv      （Cfg1=端点+分段斜率，Cfg2=端点+VMD K 重选）
【输出】
  stdout: 对齐表（数值 / 来源 / 偏差）
  repro/out/rebuild-extrapolation.json
【不声称】未运行 MATLAB；未生成任何新仿真数据。
【需作者确认】E-2 的 0.574 是否来自另一轮运行/不同舍入；Cfg 编号与配置的对应关系。
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
N_EXTRAP_PUBLISHED = 240


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

    hand = load("ta_extrapolation.csv")
    vmd = load("ta_extrap_vmd.csv")
    seg = load("ta_seg_extrap.csv")

    checks = []

    # ---- E-1 手工三特征（H=6 整集） -------------------------------------
    ln, r = row_at(hand, Extrapolation="H=6(训练未见)")
    src = f"ta_extrapolation.csv:{ln} H=6(训练未见) 提升树(手工三特征)"
    checks.append(align("E-1a", "RMSE", r["RMSE_Hz"], 0.0464, src))
    checks.append(align("E-1b", "R2", r["R2"], 0.578, src))

    # ---- E-2 仅端点（秩 1） ---------------------------------------------
    ln, r = row_at(vmd, Cfg="1")
    src = f"ta_extrap_vmd.csv:{ln} Cfg=1（仅端点/秩1，N_test={r['N_test']}）"
    boundary_note = (
        "边界偏差：CSV 的 R^2=0.573494 按三位小数应为 0.573，稿中 0.574 对应原始值 >=0.5735；"
        "偏差仅超半格容差 6e-7，但底层原始值差 >=5e-5 —— 可能源于另一次运行或不同舍入实现，需作者确认。"
        "不影响结论方向：0.574 仍远低于 0.835 / 0.913。"
    )
    checks.append(align("E-2a", "RMSE", r["RMSE_Hz"], 0.0467, src))
    checks.append(align("E-2b", "R2", r["R2"], 0.574, src, boundary_note))

    # ---- E-3 端点+VMD，K=4（全网格选定） --------------------------------
    ln, r = row_at(vmd, Cfg="2")
    src = f"ta_extrap_vmd.csv:{ln} Cfg=2（端点+VMD，K=4 全网格选定，N_test={r['N_test']}）"
    checks.append(align("E-3", "R2", r["R2"], 0.816, src,
                        "稿件未给出该配置的 RMSE，故仅对齐 R^2。"))

    # ---- E-4 端点+分段斜率 ----------------------------------------------
    ln, r = row_at(seg, Cfg="1")
    src = f"ta_seg_extrap.csv:{ln} Cfg=1（端点+分段斜率，N_test={r['N_test']}）"
    checks.append(align("E-4a", "RMSE", r["RMSE_Hz"], 0.0211, src))
    checks.append(align("E-4b", "R2", r["R2"], 0.913, src))

    # ---- E-5 端点+VMD，K 在 H<=5 开发集重选 -----------------------------
    ln, r = row_at(seg, Cfg="2")
    src = f"ta_seg_extrap.csv:{ln} Cfg=2（端点+VMD，K 重选，N_test={r['N_test']}）"
    checks.append(align("E-5", "R2", r["R2"], 0.835, src,
                        "稿件未给出该配置的 RMSE，故仅对齐 R^2。"))

    # ---- E-6 结构性陈述：排序 0.913 > 0.835 > 0.574 ----------------------
    seg_r2 = float(row_at(seg, Cfg="1")[1]["R2"])
    kresel_r2 = float(row_at(seg, Cfg="2")[1]["R2"])
    rank1_r2 = float(row_at(vmd, Cfg="1")[1]["R2"])
    order_ok = seg_r2 > kresel_r2 > rank1_r2
    checks.append({
        "id": "E-6",
        "metric": "ORDER",
        "rebuild_value": [round(seg_r2, 6), round(kresel_r2, 6), round(rank1_r2, 6)],
        "published_value": "0.913 > 0.835 > 0.574",
        "deviation": 0.0,
        "tolerance_half_ulp": 0.0,
        "aligned": bool(order_ok),
        "source": "ta_seg_extrap.csv:2-3 + ta_extrap_vmd.csv:2",
        "note": "三种特征配置在外推集上的排序与稿件陈述一致。",
    })

    # ---- E-7 结构性陈述：H=6 外推集 240 组 ------------------------------
    n_vals = sorted({int(r["N_test"]) for r in vmd + seg})
    checks.append({
        "id": "E-7",
        "metric": "N_test",
        "rebuild_value": n_vals,
        "published_value": N_EXTRAP_PUBLISHED,
        "deviation": 0.0,
        "tolerance_half_ulp": 0.0,
        "aligned": n_vals == [N_EXTRAP_PUBLISHED],
        "source": "ta_extrap_vmd.csv:2-3 + ta_seg_extrap.csv:2-3 的 N_test 列",
        "note": "稿件称全部 H=6 s 场景共 240 组均未参与训练；四行 CSV 的 N_test 均为 240。",
    })

    n_ok = sum(1 for c in checks if c["aligned"])
    mismatches = [c["id"] for c in checks if not c["aligned"]]

    print("=" * 118)
    print("rebuild-extrapolation.py  |  对齐表：数值 / 来源 / 偏差（重建值 vs 稿件已给值）")
    print("=" * 118)
    print(f"{'ID':<7}{'指标':<7}{'重建值':>30}{'稿件值':>22}{'偏差':>12}{'容差':>10}  来源")
    print("-" * 118)
    for c in checks:
        rv = c["rebuild_value"]
        rvs = f"{rv:.10f}" if isinstance(rv, float) else str(rv)
        pv = c["published_value"]
        pvs = f"{pv:.4f}" if isinstance(pv, float) else str(pv)
        print(f"{c['id']:<7}{c['metric']:<7}{rvs:>30}{pvs:>22}{c['deviation']:>12.2e}"
              f"{c['tolerance_half_ulp']:>10.2e}  {c['source']}")
    print("-" * 118)
    print(f"对齐 {n_ok}/{len(checks)} 项；容差 = 半格舍入（R2 5e-4 / RMSE 5e-5）")
    if mismatches:
        print(f"超容差项：{', '.join(mismatches)}（边界偏差，详见 JSON 的 note 字段）")
    for c in checks:
        if c["note"]:
            print(f"  [{c['id']}] {c['note']}")

    payload = {
        "script": "rebuild-extrapolation.py",
        "result": "H=6 s 单侧外推已发表数值",
        "rebuild": True,
        "original_author_script": "partial",
        "author_confirmation_required": True,
        "matlab_run": False,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": ["Project/results/tables/ta_extrapolation.csv",
                   "Project/results/tables/ta_extrap_vmd.csv",
                   "Project/results/tables/ta_seg_extrap.csv"],
        "checks": checks,
        "checks_total": len(checks),
        "checks_aligned": n_ok,
        "all_aligned": n_ok == len(checks),
        "mismatches": mismatches,
        "limitations": [
            "训练层（H<=5 限制训练、重新训练 LSBoost）未重建：1200 场景数据在盘（data/raw/dataset_ta.mat），但重训脚本不全且 MATLAB 不可启动。",
            "E-2 的 0.574 与 CSV 的 0.573494 存在边界偏差（5e-4），需作者确认取值来源。",
            "Cfg 编号与特征配置的对应关系由稿件叙述与数值反推，需作者确认。",
            "未运行 MATLAB，未生成任何新仿真数据。",
        ],
    }
    out = OUTDIR / "rebuild-extrapolation.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON 已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
