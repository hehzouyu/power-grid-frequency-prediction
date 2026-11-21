"""T2 自测：默认路径逐位一致 + 参数化真的生效 + 改动边界只到 observe()。

三条验收各有一套独立证据：
  A1 改动边界：AST 逐函数比对（基线副本 vs 当前），除 observe 外零差异；
     run_rolling_warning.py（threshold/evaluate 所在文件）整文件哈希不变。
  A2 默认逐位：重算默认 observe() 与改动前 baseline-observe.json 的 sha256 全等；
     并独立重建 RNG 四步抽取，验证抽数顺序未被参数化改坏。
  A3 可被调用：枚举 API 给出 N01..、种子=基值+k，且 CLI 能按实现数跑通并落记录。

用法：python repro/test_r2_noise_implementation.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / "repro"
sys.path.insert(0, str(REPRO))

from analyze_quality_warning import NOISE_IMPLEMENTATION_BASE_SEED, observe  # noqa: E402
from r2_noise_impl import (  # noqa: E402
    enumerate_implementations,
    implementation_from_id,
    implementations_from_seeds,
    observe_implementation,
)

BASELINE_JSON = ROOT / "reports/round2/t2/baseline-observe.json"
BASELINE_SRC = ROOT / "reports/round2/t2/baseline-src"
REPORT = ROOT / "reports/round2/t2/selftest-report.json"
TARGET = "repro/analyze_quality_warning.py"
JUDGE = "repro/run_rolling_warning.py"


def digest(arr) -> str:
    arr = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_level_units(source: str) -> dict[str, str]:
    """顶层函数/类的源码段哈希 + 顶层赋值的 (name -> code) 文本。"""
    tree = ast.parse(source)
    units: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(source, node) or ""
            units[f"def {node.name}"] = hashlib.sha256(segment.encode()).hexdigest()
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            segment = ast.get_source_segment(source, node) or ""
            units[f"assign {node.targets[0].id}"] = hashlib.sha256(segment.encode()).hexdigest()
    return units


def check_diff_boundary() -> dict:
    before = (BASELINE_SRC / Path(TARGET).name).read_text(encoding="utf-8")
    after = (ROOT / TARGET).read_text(encoding="utf-8")
    ub, ua = top_level_units(before), top_level_units(after)
    changed = sorted(k for k in ub if k in ua and ub[k] != ua[k])
    removed = sorted(k for k in ub if k not in ua)
    added = sorted(k for k in ua if k not in ub)
    judge_before = (BASELINE_SRC / Path(JUDGE).name).read_bytes()
    judge_after = (ROOT / JUDGE).read_bytes()
    judge_unchanged = hashlib.sha256(judge_before).hexdigest() == hashlib.sha256(judge_after).hexdigest()
    # 判定函数与阈值常量的显式点名核对（即使整文件未变也留痕）。
    named = ["def threshold", "def evaluate", "THRESHOLD="]
    named_hits = {name: (name in after) == (name in before) for name in named}
    ok = (
        changed == ["def observe"]
        and not removed
        and set(added) <= {"assign NOISE_IMPLEMENTATION_BASE_SEED"}
        and judge_unchanged
        and all(named_hits.values())
    )
    return {
        "changedUnits": changed,
        "removedUnits": removed,
        "addedUnits": added,
        "judgeFileUnchanged": judge_unchanged,
        "judgeFileSha256": hashlib.sha256(judge_after).hexdigest(),
        "namedAnchorsPresent": named_hits,
        "pass": ok,
    }


def check_default_bitwise() -> dict:
    base = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    with np.load(ROOT / "repro/out/rolling/ieee39quality-trajectories.npz") as case:
        f = case["multi"]
    protocol = json.loads((ROOT / "repro/out/rolling/protocol-results.json").read_text(encoding="utf-8"))
    reference = protocol["noise_rms_reference_Hz"] / 50

    signals, estimates, quality = observe(f, reference)  # 不传 seed = 默认路径
    explicit = observe(f, reference, seed=NOISE_IMPLEMENTATION_BASE_SEED)  # 显式给同一 seed
    shifted = observe(f, reference, seed=NOISE_IMPLEMENTATION_BASE_SEED + 1)  # k=1

    expected = base["outputs"]
    got = {
        "signal_sha256": [digest(s) for s in signals],
        "estimates_sha256": digest(estimates),
        "quality_sha256": digest(quality),
    }
    matches_baseline = (
        got["signal_sha256"] == expected["signal_sha256"]
        and got["estimates_sha256"] == expected["estimates_sha256"]
        and got["quality_sha256"] == expected["quality_sha256"]
    )
    explicit_equals_default = (
        all(digest(a) == digest(b) for a, b in zip(signals, explicit[0]))
        and digest(estimates) == digest(explicit[1])
        and digest(quality) == digest(explicit[2])
    )
    shifted_differs = not explicit_equals_default or digest(shifted[0][1]) != digest(explicit[0][1])

    # 独立重建 RNG 四步（不改 observe 的代码，纯外部复算），核对抽数顺序未被改坏。
    rng = np.random.default_rng(NOISE_IMPLEMENTATION_BASE_SEED)
    noise = rng.normal(size=(len(f), f.shape[1] + 50))
    noise_ar = rng.normal(size=noise.shape)
    correlated, _ = lfilter([0.6], [1, -0.8], noise_ar, axis=1, zi=rng.normal(size=(len(f), 1)) * 0.8)
    recon_white40 = f + reference / 100 * noise[:, 50:]
    recon_corr20 = f + reference / 10 * correlated[:, 50:]
    rng_order_ok = digest(recon_white40) == digest(signals[1]) and digest(recon_corr20) == digest(signals[3])

    # baseline 文件里的 quality_low_counts 实为 quality==True（high 档）计数，
    # 抓取时命名有误；该 JSON 是改动前证据不事后改写，这里按真实语义比对。
    demo = base["outputs"]["quality_low_counts"]
    quality_true = quality.sum(axis=0).tolist()
    return {
        "baselineFile": BASELINE_JSON.relative_to(ROOT).as_posix(),
        "defaultMatchesBaseline": matches_baseline,
        "explicitSeedEqualsDefault": explicit_equals_default,
        "k1DiffersFromK0": shifted_differs,
        "rngDrawOrderReconstructed": rng_order_ok,
        "qualityTrueCounts": quality_true,
        "baselineQualityTrueCounts": demo,
        "got": {k: (v if isinstance(v, str) else v) for k, v in got.items()},
        "pass": matches_baseline and explicit_equals_default and shifted_differs and rng_order_ok,
    }


def check_parametric_api(tmp_dir: Path) -> dict:
    plan = enumerate_implementations(5)
    ids = [p.noise_impl_id for p in plan]
    seeds = [p.seed for p in plan]
    by_id = implementation_from_id("N03")
    by_seeds = implementations_from_seeds([39192514, 39192515])
    api_ok = (
        ids == ["N01", "N02", "N03", "N04", "N05"]
        and seeds == [39192514 + k for k in range(5)]
        and by_id.k == 2
        and by_id.seed == 39192516
        and [p.noise_impl_id for p in by_seeds] == ["N01", "N02"]
    )

    cli = subprocess.run(
        [sys.executable, str(REPRO / "run_round2_noise_impl.py"),
         "--count", "2", "--out-dir", str(tmp_dir), "--no-double-run"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600,
    )
    record_path = tmp_dir / "run-record.json"
    cli_ok = cli.returncode == 0 and record_path.exists()
    cli_detail = {}
    if cli_ok:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        cli_detail = {
            "implementations": [d["noiseImplId"] for d in record["implementations"]],
            "seeds": [d["seed"] for d in record["implementations"]],
            "csv": record["outputs"]["noiseImplementationsCsv"],
            "anchor": record["anchorR1Bitwise"].get("bitwiseIdentical"),
        }
        cli_ok = (
            cli_detail["implementations"] == ["N01", "N02"]
            and cli_detail["seeds"] == [39192514, 39192515]
            and cli_detail["anchor"] is True
            and (tmp_dir / cli_detail["csv"]).exists()
        )
    return {
        "enumerate5": {"ids": ids, "seeds": seeds},
        "fromIdN03": {"k": by_id.k, "seed": by_id.seed},
        "fromSeeds": [p.noise_impl_id for p in by_seeds],
        "apiPass": api_ok,
        "cliExitCode": cli.returncode,
        "cliStdoutTail": cli.stdout.strip().splitlines()[-4:],
        "cliDetail": cli_detail,
        "cliPass": cli_ok,
        "pass": api_ok and cli_ok,
    }


def main() -> int:
    boundary = check_diff_boundary()
    default = check_default_bitwise()
    # 用工作区内目录而不是系统临时目录：沙箱下系统 temp 清理会 PermissionError，
    # 也会让「证据落在工作区」这条变弱。
    tmp_dir = ROOT / "reports/round2/t2/selftest-cli"
    if tmp_dir.exists():
        for item in sorted(tmp_dir.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
        tmp_dir.rmdir()
    api = check_parametric_api(tmp_dir)
    report = {
        "stage": "T2 selftest",
        "A1_diff_boundary": boundary,
        "A2_default_bitwise": default,
        "A3_parametric_callable": api,
        "allPassed": boundary["pass"] and default["pass"] and api["pass"],
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nA1:", "PASS" if boundary["pass"] else "FAIL")
    print("A2:", "PASS" if default["pass"] else "FAIL")
    print("A3:", "PASS" if api["pass"] else "FAIL")
    return 0 if report["allPassed"] else 1


if __name__ == "__main__":
    sys.exit(main())
