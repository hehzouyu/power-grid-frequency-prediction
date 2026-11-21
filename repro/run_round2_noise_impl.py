"""R2 多噪声实现的可参数化运行入口（T2 产出，T3 直接调用）。

三种等价调用方式::

    # 1) 记录规范里的单实现形式（T3 按实现循环调用）
    python repro/run_round2_noise_impl.py --impl N01 --seed 39192514 --out-dir repro/out/round2-noise

    # 2) 按实现数枚举：N01..N05（k=0..4）
    python repro/run_round2_noise_impl.py --count 5 --out-dir repro/out/round2-noise

    # 3) 显式种子集合：只补跑 k=20..21 的精度升级段
    python repro/run_round2_noise_impl.py --seeds 39192534,39192535 --out-dir repro/out/round2-noise

本入口只跑 **噪声阶段**（噪声 → 特征可用信号 → quality 分档），不跑告警判定，
不重训模型；告警长表由 T3 在拿到本阶段产物后按 R2 协议重算。这样职责边界清楚，
也不会把阈值/判定逻辑复制进第二个文件。默认参数（--impl N01）必须逐位复现 R1。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPRO_DIR = Path(__file__).resolve().parent
if str(REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(REPRO_DIR))

from r2_noise_impl import (  # noqa: E402
    CONDITIONS_ALL,
    enumerate_implementations,
    implementation_from_id,
    implementations_from_seeds,
    observe_implementation,
    quality_tier_counts,
)

ROOT = REPRO_DIR.parent
DEFAULT_TRAJECTORIES = ROOT / "repro/out/rolling/ieee39quality-trajectories.npz"
DEFAULT_REFERENCE = ROOT / "repro/out/rolling/protocol-results.json"
ANCHOR_BASELINE = ROOT / "reports/round2/t2/baseline-observe.json"


def digest(arr) -> str:
    arr = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def resolve_plan(args) -> tuple[tuple, str]:
    if args.impl and args.seed is not None:
        impl = implementation_from_id(args.impl)
        if impl.seed != args.seed:
            raise SystemExit(
                f"--seed {args.seed} 与 {args.impl} 的协议种子 {impl.seed} 不一致；"
                "种子必须落在 baseSeed+ 的格点上，请改用 --count/--seeds"
            )
        return (impl,), f"impl {impl.noise_impl_id}"
    if args.seeds:
        seeds = [int(x) for x in args.seeds.replace(" ", "").split(",") if x]
        return implementations_from_seeds(seeds), f"seeds {seeds}"
    if args.count:
        return enumerate_implementations(args.count, args.start_k), f"count {args.count} start_k {args.start_k}"
    raise SystemExit("必须给出 --count/--start-k、--seeds 或 --impl/--seed 三者之一")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="R2 multi-noise-implementation noise stage")
    parser.add_argument("--impl", help="noiseImplId，如 N01")
    parser.add_argument("--seed", type=int, help="与 --impl 配套的显式种子（必须等于 baseSeed+k）")
    parser.add_argument("--count", type=int, help="按实现数枚举")
    parser.add_argument("--start-k", type=int, default=0, help="枚举起始 k（默认 0 = R1 锚点）")
    parser.add_argument("--seeds", help="逗号分隔的显式种子集合")
    parser.add_argument("--out-dir", default="repro/out/round2-noise", help="产物目录")
    parser.add_argument("--trajectories", default=str(DEFAULT_TRAJECTORIES))
    parser.add_argument("--reference-json", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--key", default="multi", help="轨迹 npz 内的数组名")
    parser.add_argument("--no-double-run", action="store_true", help="跳过位级双跑校验（不建议）")
    parser.add_argument("--emit-npz", action="store_true", help="额外落盘每实现信号数组（体积大）")
    args = parser.parse_args(argv)

    plan, plan_label = resolve_plan(args)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with np.load(args.trajectories) as case:
        f = case[args.key]
    protocol = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))
    reference = protocol["noise_rms_reference_Hz"] / 50

    rows, details, all_bitwise = [], [], True
    for impl in plan:
        run = observe_implementation(f, reference, impl)
        signals, estimates, quality = run
        row = {
            "noiseImplId": impl.noise_impl_id,
            "k": impl.k,
            "seed": impl.seed,
            "variedBy": impl.varied_by,
            "scenarios": int(estimates.shape[0]),
            "conditions": int(estimates.shape[1]),
            "estimatesSha256": digest(estimates),
            "qualitySha256": digest(quality),
        }
        row.update(quality_tier_counts(quality))
        rows.append(row)

        detail = {
            "noiseImplId": impl.noise_impl_id,
            "k": impl.k,
            "seed": impl.seed,
            "signalSha256": {c: digest(s) for c, s in zip(CONDITIONS_ALL, signals)},
        }
        if not args.no_double_run:
            again = observe_implementation(f, reference, impl)
            identical = (
                all(digest(a) == digest(b) for a, b in zip(signals, again[0]))
                and digest(estimates) == digest(again[1])
                and digest(quality) == digest(again[2])
            )
            detail["doubleRunBitwiseIdentical"] = identical
            all_bitwise = all_bitwise and identical
        if args.emit_npz:
            npz_path = out_dir / f"noise-{impl.noise_impl_id}.npz"
            np.savez_compressed(
                npz_path,
                signals=np.stack(signals),
                estimates=estimates,
                quality=quality,
            )
            detail["npz"] = npz_path.name
        details.append(detail)

    csv_path = out_dir / "noise-implementations.csv"
    fieldnames = ["noiseImplId", "k", "seed", "variedBy", "scenarios", "conditions",
                  "estimatesSha256", "qualitySha256"]
    for condition in CONDITIONS_ALL:
        fieldnames += [f"{condition}_low", f"{condition}_high"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 锚点自检：只要 plan 覆盖 k=0，就必须与改动前基线逐位一致。
    anchor = {"checked": False, "bitwiseIdentical": None}
    if any(impl.k == 0 for impl in plan):
        if ANCHOR_BASELINE.exists():
            base = json.loads(ANCHOR_BASELINE.read_text(encoding="utf-8"))["outputs"]
            first = details[0]
            anchor = {
                "checked": True,
                "baselineFile": ANCHOR_BASELINE.relative_to(ROOT).as_posix(),
                "bitwiseIdentical": (
                    list(base["signal_sha256"]) == list(first["signalSha256"].values())
                    and base["estimates_sha256"] == rows[0]["estimatesSha256"]
                    and base["quality_sha256"] == rows[0]["qualitySha256"]
                ),
            }
        else:
            anchor = {"checked": False, "bitwiseIdentical": None, "reason": "baseline-observe.json 不存在"}

    record = {
        "task": "R2/T2 noise stage",
        "argv": sys.argv[1:] if argv is None else list(argv),
        "plan": plan_label,
        "outDir": out_dir.relative_to(ROOT).as_posix(),
        "inputs": {
            "trajectories": {"path": Path(args.trajectories).name, "key": args.key,
                             "shape": list(f.shape), "sha256": digest(f)},
            "referencePu": reference,
            "referenceJson": Path(args.reference_json).name,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "doubleRunBitwiseIdentical": all_bitwise,
        "anchorR1Bitwise": anchor,
        "implementations": details,
        "outputs": {"noiseImplementationsCsv": csv_path.name},
        "elapsedSeconds": round(time.time() - started, 3),
    }
    record_path = out_dir / "run-record.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"plan            : {plan_label}")
    print(f"implementations : {[i.noise_impl_id for i in plan]}")
    print(f"k / seed        : {[(i.k, i.seed) for i in plan]}")
    print(f"csv             : {(csv_path.relative_to(ROOT)).as_posix()} ({len(rows)} rows)")
    print(f"run-record      : {record_path.relative_to(ROOT).as_posix()}")
    print(f"double-run      : {all_bitwise if not args.no_double_run else 'skipped'}")
    print(f"anchor k=0      : {anchor}")
    ok = all_bitwise and anchor.get("bitwiseIdentical") is not False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
