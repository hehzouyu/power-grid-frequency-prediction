#!/usr/bin/env python
"""R2 逐实现记录校验器（结构 + 语义 + 存量覆盖审计）。

用法：
  python repro/validate_r2_record.py <record.json> [更多记录...]
  python repro/validate_r2_record.py --audit-legacy

结构校验用 reports/r2/record-schema.json（JSON Schema 2020-12）；
语义校验覆盖规范 §6/§7 里 JSON Schema 表达不了的约束。
退出码：0=全部通过，1=有失败项。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "reports" / "r2" / "record-schema.json"
FIELD_LIST = ROOT / "reports" / "r2" / "field-list.csv"

QUALITY_SUMMARY_COLUMNS = [
    "Dataset", "Model", "TargetFAR", "Policy", "Condition", "N",
    "PhysicalEvents", "Risk", "Hits", "FalseAlarms", "Late", "Timely",
    "TimelyLow", "TimelyHigh", "FAR", "FARLow", "FARHigh",
    "LeadMedian_ms", "LeadP5_ms",
]
CONDITIONS = ["clean", "white40", "white20", "correlated20", "white30", "noise_jump"]
MODELS = ["persistence", "quadratic", "residual"]

problems: list[str] = []
warnings: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


# ---------- 结构校验 ----------
def structural(record: dict, schema: dict) -> None:
    try:
        import jsonschema
        from jsonschema import Draft202012Validator
    except ImportError:  # pragma: no cover
        warn("jsonschema 未安装，跳过结构校验")
        return
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))
    for err in errors[:20]:
        loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
        fail(f"[结构] {loc}: {err.message}")


# ---------- 语义校验 ----------
def check_single_variable(rec: dict) -> None:
    seeds = rec.get("seeds") or {}
    varied = set(seeds.get("variedByImplementation") or [])
    held = set(seeds.get("heldFixed") or [])
    if varied & held:
        fail(f"[语义] variedByImplementation 与 heldFixed 重叠：{sorted(varied & held)}")
    if rec.get("recordType") == "implementation":
        if varied != {"observationNoise"}:
            fail(f"[语义] 实现间只允许 observationNoise 变，实为 {sorted(varied)}")


def check_rng(rec: dict) -> None:
    rng = rec.get("rng") or {}
    shape = rng.get("shapeContract") or {}
    if not shape:
        fail("[语义] 缺 rng.shapeContract，无法判定种子契约")
        return
    n = shape.get("scenarios")
    samples = shape.get("samplesPerScenario")
    pre = shape.get("preEventSamples")
    tensor = shape.get("noiseTensorShape") or []
    if None not in (n, samples, pre) and tensor:
        if not (isinstance(tensor, list) and len(tensor) == 2):
            fail(f"[语义] noiseTensorShape 应为两元素数组，实为 {tensor}")
        elif tensor[0] != n or tensor[1] != samples + pre:
            fail(
                f"[语义] noiseTensorShape={tensor} 与 [scenarios, samples+pre]="
                f"[{n}, {samples + pre}] 不符"
            )
    order = rng.get("drawOrder") or []
    if len(order) != 4:
        fail(f"[语义] drawOrder 必须 4 步（noise/noise_ar/zi/correlated），实为 {len(order)} 步")
    else:
        names = {str(s.get("name")) for s in order if isinstance(s, dict)}
        for token in ("noise", "noise_ar", "zi", "correlated"):
            if token not in names:
                fail(f"[语义] drawOrder 缺 {token} 步（correlated 步即 lfilter 那一步）")


def check_seeds(rec: dict) -> None:
    seeds = rec.get("seeds") or {}
    obs = seeds.get("observationNoise")
    if not isinstance(obs, int):
        fail("[语义] seeds.observationNoise 必须是整数，便于逐位重放")
    pop = seeds.get("scenarioPopulation") or {}
    if rec.get("dataset", {}).get("scenarioSource") == "simulated" and not pop.get("applicable"):
        fail("[语义] scenarioSource=simulated 时 scenarioPopulation.applicable 必须为 true")
    if "_template" in json.dumps(rec, ensure_ascii=False):
        fail("[语义] 记录里残留 _template 标记")


def check_denominators(rec: dict) -> None:
    d = rec.get("denominators") or {}
    if not d:
        fail("[语义] 缺 denominators")
        return
    planned, failed, scored = d.get("eventsPlanned"), d.get("eventsFailed"), d.get("eventsScored")
    if None not in (planned, failed, scored) and planned - failed != scored:
        fail(f"[语义] eventsScored != eventsPlanned - eventsFailed（{scored} != {planned}-{failed}）")
    risk, nonrisk = d.get("riskUnits"), d.get("nonRiskUnits")
    if None not in (risk, nonrisk, scored) and risk + nonrisk != scored:
        fail(f"[语义] riskUnits + nonRiskUnits != eventsScored（{risk}+{nonrisk}!={scored}）")

    splanned, sfailed, sscored = (
        d.get("scenariosPlanned"), d.get("scenariosFailed"), d.get("scenariosScored"),
    )
    if None not in (splanned, sfailed, sscored) and splanned - sfailed != sscored:
        fail(f"[语义] scenariosScored != scenariosPlanned - scenariosFailed")

    conds = rec.get("dataset", {}).get("conditions") or CONDITIONS
    if isinstance(splanned, int) and conds and planned is not None:
        if planned != splanned * len(conds):
            warn(f"[语义] eventsPlanned={planned} != scenariosPlanned({splanned}) x 条件数({len(conds)})")

    failures = rec.get("failures") or []
    counted = [f for f in failures if not f.get("excludedFromDenominators")]
    if failed is not None and len(counted) != failed:
        fail(f"[语义] eventsFailed={failed} 与 failures[] 中未剔除条目数 {len(counted)} 不一致")
    for i, f in enumerate(failures):
        if f.get("condition") and f["condition"] not in CONDITIONS:
            fail(f"[语义] failures[{i}].condition={f['condition']} 不在条件枚举内")

    done, planned_impl = d.get("implementationsCompleted"), d.get("implementationsPlanned")
    if None not in (done, planned_impl) and done < planned_impl:
        warn(f"[语义] 实现未跑满：{done}/{planned_impl}，跨实现分布只覆盖已完成部分")


def check_null_metrics(rec: dict) -> None:
    for i, h in enumerate(rec.get("headline") or []):
        tag = f"headline[{i}]"
        if h.get("model") and h["model"] not in MODELS:
            fail(f"[语义] {tag}.model={h['model']} 不在三模型枚举内")
        for key in ("hitRate", "FAR"):
            val = h.get(key)
            if val is None:
                if not h.get("metricNullReason"):
                    fail(f"[语义] {tag}.{key}=null 时必须有 metricNullReason（不得把 0 写成 null 或反之）")
            elif not (0.0 <= float(val) <= 1.0):
                fail(f"[语义] {tag}.{key}={val} 越界 [0,1]")


def check_metric_columns(rec: dict) -> None:
    cols = (rec.get("metrics") or {}).get("columns") or []
    if not cols:
        fail("[语义] metrics.columns 缺失，无法校验与 quality-summary 的列对接")
        return
    missing = [c for c in QUALITY_SUMMARY_COLUMNS if c not in cols]
    if missing:
        fail(f"[语义] metrics.columns 缺 quality-summary 列：{missing}")
    if (rec.get("metrics") or {}).get("rowCount") is None:
        warn("[语义] metrics.rowCount 缺失")


def check_threshold_join(rec: dict) -> None:
    for i, t in enumerate(rec.get("thresholds") or []):
        tag = f"thresholds[{i}]"
        for key in ("LowQuality", "CalibrationVariants", "Threshold_pu"):
            if key not in t:
                fail(f"[语义] {tag} 缺 quality-thresholds 同名列 {key}")
        if t.get("model") and t["model"] not in MODELS:
            fail(f"[语义] {tag}.model={t['model']} 不在三模型枚举内")


def check_replay(rec: dict) -> None:
    replay = rec.get("replay") or {}
    level = replay.get("level")
    if level == "bitwise":
        ver = replay.get("verification") or {}
        if ver.get("mode") != "double-run":
            fail("[语义] replay.level=bitwise 必须配 verification.mode=double-run")
        if ver.get("allIdentical") is not True:
            fail("[语义] replay.level=bitwise 必须 allIdentical=true，否则只能降级为 numeric")
        r1, r2 = ver.get("run1OutputsSha256"), ver.get("run2OutputsSha256")
        if not (isinstance(r1, list) and isinstance(r2, list) and r1 and r1 == r2):
            fail("[语义] bitwise 要求双跑输出哈希列表非空且逐项相同")
    elif level == "numeric":
        tol = replay.get("tolerance") or {}
        if not tol.get("value"):
            fail("[语义] replay.level=numeric 必须给 tolerance.value")
    elif level == "parameter-only":
        fail("[语义] replay.level=parameter-only 不是可复现证据（规范 §6 明确排除）")
    else:
        fail(f"[语义] replay.level={level} 不在 bitwise/numeric 内")
    self_check = (rec.get("integrity") or {}).get("selfCheck") or {}
    if level == "bitwise" and self_check.get("bitwiseVerified") is not True:
        warn("[语义] bitwiseVerified 未置 true，记录自述与 replay 声明不一致")


def check_series(rec: dict) -> None:
    if rec.get("recordType") != "series":
        return
    series = rec.get("series") or {}
    impls = series.get("implementations") or []
    if len(impls) < 5:
        fail(f"[语义] R2 要求 >=5 个噪声实现，实为 {len(impls)}")
    if series.get("nImplementations") != len(impls):
        fail(f"[语义] nImplementations={series.get('nImplementations')} 与 implementations 长度 {len(impls)} 不符")
    if series.get("frozenSplitPreserved") is not True:
        fail("[语义] series 必须声明 frozenSplitPreserved=true")


def check_design_link(rec: dict) -> None:
    ds = rec.get("dataset") or {}
    source = ds.get("scenarioSource")
    if source == "design-table":
        if not ds.get("design"):
            fail("[语义] scenarioSource=design-table 时必须有 dataset.design")
        if not ds.get("split"):
            fail("[语义] scenarioSource=design-table 时必须有 dataset.split")
    elif source == "simulated":
        if ds.get("design") is not None:
            fail("[语义] scenarioSource=simulated 时 dataset.design 必须为 null（aggregate 无 design 对应）")
    else:
        fail(f"[语义] dataset.scenarioSource={source} 不在 design-table/simulated 内")


def validate_path(path: Path) -> bool:
    global problems, warnings
    problems, warnings = [], []
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"FAIL {path}: 读取失败 {exc}")
        return False
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    structural(rec, schema)
    for check in (
        check_single_variable, check_rng, check_seeds, check_denominators,
        check_null_metrics, check_metric_columns, check_threshold_join,
        check_replay, check_series, check_design_link,
    ):
        try:
            check(rec)
        except Exception as exc:  # 校验器自身不能被单条坏记录带崩
            fail(f"[内部] {check.__name__} 抛错：{exc}")

    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    if problems:
        print(f"FAIL {rel}  ({len(problems)} 项)")
        for p in problems:
            print(f"   - {p}")
        for w in warnings:
            print(f"   ~ {w}")
        return False
    print(f"PASS {rel}" + (f"  ({len(warnings)} 条提示)" if warnings else ""))
    for w in warnings:
        print(f"   ~ {w}")
    return True


# ---------- 存量覆盖审计 ----------
def flat_keys(obj, prefix="") -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            out.add(key)
            out |= flat_keys(v, key)
    elif isinstance(obj, list):
        for item in obj[:1]:
            out |= flat_keys(item, f"{prefix}[]")
    return out


def audit_legacy() -> int:
    rows = []
    import csv
    with FIELD_LIST.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    required = [r["field"] for r in rows if r["required"] == "是"]

    candidates = sorted(ROOT.glob("repro/out/**/run-record.json"))
    if not candidates:
        print("未找到任何 run-record.json")
        return 1
    print(f"字段清单：{len(rows)} 项（必填 {len(required)}）；存量记录：{len(candidates)} 份\n")
    for path in candidates:
        rec = json.loads(path.read_text(encoding="utf-8"))
        present = flat_keys(rec)
        # 字段清单用点号路径，存量记录用 dict 路径；按末段名 + 全路径双匹配
        hit, miss = [], []
        for field in required:
            normalized = field.replace("[]", "").replace("[].", ".")
            tail = normalized.split(".")[-1]
            ok = any(k == normalized or k.endswith("." + normalized) or k.split(".")[-1] == tail
                     for k in present)
            (hit if ok else miss).append(field)
        ratio = len(hit) / len(required) * 100 if required else 0.0
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"{rel}: 覆盖 {len(hit)}/{len(required)} = {ratio:.1f}%")
        if miss:
            print(f"   缺：{', '.join(miss[:14])}{' …' if len(miss) > 14 else ''}")
    print("\n结论：存量 run-record 均非 R2 格式；上表即 R2 需要补齐的字段。")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    if argv[0] == "--audit-legacy":
        return audit_legacy()
    targets = [Path(a) for a in argv]
    missing = [p for p in targets if not p.exists()]
    for p in missing:
        print(f"FAIL {p}: 文件不存在")
    ok = all([validate_path(p) for p in targets if p.exists()])
    print()
    print("总结：全部通过" if ok and not missing else "总结：存在失败项")
    return 0 if ok and not missing else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
