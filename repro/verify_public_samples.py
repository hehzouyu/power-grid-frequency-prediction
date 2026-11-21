"""只读核对公开仿真样本及冻结档案。无需 MATLAB 或第三方 Python 包。"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "repro/out"
TRANSFER = ROOT / "NewWorkspace/2026-11-21-ECGR滚动接入-v1"
CONDITIONS = {"clean", "white40", "white30", "white20", "ar30"}

def require(ok, message):
    """报告首个失败项目，并以非零退出状态中止。"""
    if not ok:
        raise ValueError(message)

def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)

def verify_lock():
    entries = json.loads((TRANSFER / "calibration-lock.json").read_text(encoding="utf-8"))
    require(len(entries) == 8, "滚动接入冻结清单应有八项")
    for name, expected in entries.items():
        # 冻结文件保留原 Windows 相对路径，用 PureWindowsPath 解析以兼容其他系统。
        path = ROOT.joinpath(*PureWindowsPath(name).parts)
        require(path.is_file(), f"缺少冻结文件：{name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"冻结文件哈希不同：{name}")

def verify_samples():
    scenarios = list(csv_rows(OUTPUT / "revision/grid-index.csv"))
    require({int(row["ID"]) for row in scenarios} == set(range(1, 1201)) and len(scenarios) == 1200,
            "固定截止场景索引缺少或重复编号")
    for relative in ("Project/data/raw/dataset_raw.mat", "repro/out/revision/trajectories.mat",
                     "repro/out/revision/features.mat", "repro/out/rolling/aggregate-trajectories.npz",
                     "repro/out/nadir-pair-tests/independent-v1/pair-predictions.csv"):
        path = ROOT / relative
        require(path.is_file() and path.stat().st_size > 0, f"样本缺失：{relative}")

def verify_rolling():
    counts = defaultdict(Counter)
    keys = set()
    with gzip.open(TRANSFER / "test-event-scores.csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            model = row["Policy"]
            if row["Condition"] not in CONDITIONS or model not in {"gate", "quadratic", "persistence"}:
                continue
            if float(row["TargetFAR"]) != .10:
                continue
            key = (model, row["Event"], row["Condition"])
            require(key not in keys, f"逐事件主工作点存在重复键：{key}")
            keys.add(key)
            crossing, alarm = float(row["Crossing_s"]), float(row["Alarm_s"])
            if math.isfinite(crossing):
                counts[model]["risk"] += 1
                counts[model]["hits"] += alarm < crossing
            else:
                counts[model]["safe"] += 1
                counts[model]["false_alarms"] += math.isfinite(alarm)
    expected = {"gate": (1622, 1076), "quadratic": (1012, 912), "persistence": (1673, 1036)}
    require(len({key[1] for key in keys}) == 2400, "物理事件数不符")
    for model, (hits, false_alarms) in expected.items():
        target = {"risk": 2045, "safe": 9955, "hits": hits, "false_alarms": false_alarms}
        require(counts[model] == target, f"滚动主工作点计数不同：{model}: {counts[model]}")

if __name__ == "__main__":
    verify_lock()
    verify_samples()
    verify_rolling()
    print("PASS: 8 frozen files, 1,200 fixed-cutoff scenarios, and rolling counts over 2,400 physical events.")
