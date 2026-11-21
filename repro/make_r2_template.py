"""生成 reports/r2/record-template.json（R2 记录的完整可校验样例）。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "r2" / "record-template.json"
ZERO = "0" * 64


def sha(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


split = "repro/out/rolling/ieee39quality-split-v1.csv"
design = "repro/out/rolling/ieee39quality-design.csv"
traj = "repro/out/rolling/ieee39quality-trajectories.npz"

record = {
    "schema": {
        "name": "r2-noise-implementation-record",
        "version": "1.0.0",
        "specPath": "reports/r2/record-field-spec.md",
    },
    "recordType": "implementation",
    "recordId": "r2-ieee39-multi/N01/ieee39_multi/r1",
    "seriesId": "r2-ieee39-multi",
    "artifactVersion": "r2-noise-replay-v1.0.0",
    "implementation": {
        "noiseImplId": "N01",
        "ordinal": 1,
        "definition": "observationNoise=39192514，模型拟合种子 42 与场景种子 39192512/39192513 冻结，仅观测噪声种子作为实现编号",
        "isPairedAcrossConditions": True,
    },
    "replay": {
        "level": "bitwise",
        "unit": "one implementation x one run",
        "bitwiseClaim": "同一环境、同一种子、同一输入形状、同一抽取顺序下两次独立运行，输出文件字节完全相同",
        "tolerance": {
            "metric": "max_abs_diff_pu",
            "value": 1e-9,
            "appliesTo": "仅跨环境（python/numpy/BLAS 不同）时降级为数值复现",
        },
        "verification": {
            "mode": "double-run",
            "allIdentical": True,
            "run1OutputsSha256": [ZERO],
            "run2OutputsSha256": [ZERO],
        },
    },
    "seeds": {
        "observationNoise": 39192514,
        "scenarioPopulation": {"calibration": None, "test": None, "applicable": False},
        "modelFit": 42,
        "bootstrapSummary": 39192515,
        "bootstrapPaired": 39192517,
        "variedByImplementation": ["observationNoise"],
        "heldFixed": ["scenarioPopulation", "modelFit", "bootstrapSummary", "bootstrapPaired"],
    },
    "rng": {
        "library": "numpy.random.Generator",
        "algorithm": "PCG64",
        "libraryVersion": "2.4.6",
        "normalMethod": "Generator.normal（ziggurat，float64）",
        "drawOrder": [
            {"step": 1, "name": "noise", "distribution": "N(0,1)", "size": [72, 1051]},
            {"step": 2, "name": "noise_ar", "distribution": "N(0,1)", "size": [72, 1051]},
            {"step": 3, "name": "zi", "distribution": "N(0,1)", "transform": "*0.8", "size": [72, 1]},
            {"step": 4, "name": "correlated", "transform": "lfilter(b=[0.6], a=[1,-0.8], axis=1, zi=zi)", "size": [72, 1051]},
        ],
        "shapeContract": {
            "scenarios": 72,
            "samplesPerScenario": 1001,
            "preEventSamples": 50,
            "noiseTensorShape": [72, 1051],
        },
        "filters": [
            {"stage": "correlated", "b": [0.6], "a": [1, -0.8], "zi": "0.8 * normal(size=(scenarios,1))", "axis": 1}
        ],
        "conditionModulation": [
            {"condition": "clean", "formula": "deviation = 0（无噪声，仅用于对照）"},
            {"condition": "white40", "formula": "reference/100 * noise"},
            {"condition": "white20", "formula": "reference/10 * noise"},
            {"condition": "correlated20", "formula": "reference/10 * correlated"},
            {"condition": "white30", "formula": "reference/sqrt(1000) * noise"},
            {"condition": "noise_jump", "formula": "concat(reference/100*noise[:,:50], reference/10*noise[:,50:])"},
        ],
        "reference": {
            "noiseRmsReference_Hz": 0.10055295136835443,
            "noiseReference_pu": 0.0020110590273670886,
        },
        "estimator": {
            "statistic": "std(ddof=1)",
            "window": "每场景前 preEventSamples=50 个样本",
            "comparison": ">=",
            "boundary": "reference/sqrt(1000)",
        },
        "shapeSensitivity": "形状属于种子契约：场景数变化会改变流切分，同一种子不再等于同一实现",
    },
    "dataset": {
        "name": "ieee39_multi",
        "scenarioSource": "design-table",
        "scenarioCount": 144,
        "conditions": ["clean", "white40", "white20", "correlated20", "white30", "noise_jump"],
        "split": {
            "file": split,
            "sha256": sha(split),
            "roleRule": "verbatim from repro/analyze_quality_warning.py:137-142, frozen as v1.0.0",
            "calibrationScenarios": 72,
            "testScenarios": 72,
        },
        "design": {
            "file": design,
            "sha256": sha(design),
            "joinKey": "DesignID",
            "statusFilter": "Status == 'ok'",
            "failedRows": 0,
        },
    },
    "inputs": [
        {"role": "trajectories", "file": traj, "sha256": sha(traj), "shape": [144, 1001], "dt_s": 0.02},
        {"role": "design", "file": design, "sha256": sha(design), "shape": [144, 9]},
        {"role": "split", "file": split, "sha256": sha(split), "shape": [144, 11]},
    ],
    "models": [
        {
            "name": "quadratic",
            "version": "t6-quadratic-v1",
            "sourceScript": "reports/fix/t6_fit_quadratic.py",
            "sourceSha256": sha("reports/fix/t6_fit_quadratic.py"),
            "hyperparameters": {"n_estimators": 100, "max_depth": 3, "random_state": 42},
            "fittedParameters": 100,
            "fitSeed": 42,
            "trainSet": "calibration-72",
            "artifact": "repro/out/round2/quadratic-alarm/quadratic-alarm-summary.csv",
            "artifactSha256": sha("repro/out/round2/quadratic-alarm/quadratic-alarm-summary.csv"),
            "testLabelsUsedForTuning": False,
        }
    ],
    "environment": {
        "python": "3.14.6",
        "numpy": "2.4.6",
        "scipy": "1.18.0",
        "pandas": "3.0.3",
        "sklearn": "1.9.0",
        "platform": "Windows-11-10.0.26300-SP0",
    },
    "execution": {
        "command": "python repro/run_round2_noise_impl.py --impl N01 --seed 39192514 --out-dir <out-dir>",
        "startedAt": "2026-11-19T19:45:00+08:00",
        "endedAt": "2026-11-19T19:47:30+08:00",
        "exitCode": 0,
        "attempt": 1,
    },
    "guard": {
        "protectedRoots": ["repro/out/rolling", "repro/out/revision", "out", "artical"],
        "outDir": "repro/out/round2-noise/N01",
        "isNewDirectory": True,
        "protectedBaselineChanged": 0,
    },
    "denominators": {
        "scenariosPlanned": 72,
        "scenariosScored": 72,
        "scenariosFailed": 0,
        "eventsPlanned": 432,
        "eventsScored": 432,
        "eventsFailed": 0,
        "riskUnits": 126,
        "nonRiskUnits": 306,
        "calibrationNegatives": 55,
        "implementationsPlanned": 5,
        "implementationsCompleted": 1,
        "denominatorRule": "eventsScored=eventsPlanned-eventsFailed；riskUnits+nonRiskUnits=eventsScored；hitRate=Hits/riskUnits；FAR=FalseAlarms/nonRiskUnits；零分母写 null+metricNullReason",
    },
    "failures": [],
    "metrics": {
        "file": "repro/out/round2-noise/N01/quality-summary-r2.csv",
        "sha256": ZERO,
        "rowCount": 27,
        "columns": [
            "NoiseImpl", "Dataset", "Model", "TargetFAR", "Policy", "Condition", "N",
            "PhysicalEvents", "Risk", "Hits", "FalseAlarms", "Late", "Timely",
            "TimelyLow", "TimelyHigh", "FAR", "FARLow", "FARHigh",
            "LeadMedian_ms", "LeadP5_ms",
        ],
        "keyColumns": ["Dataset", "Scenario", "Condition", "Model", "TargetFAR", "Policy"],
        "mixedRule": "mixed 行只聚合 4 个主条件（white40/white20/correlated20/white30），N=288、PhysicalEvents=72",
    },
    "headline": [
        {"model": "persistence", "targetFAR": 0.1, "policy": "fixed", "condition": "mixed",
         "MAE_Hz": 0.0, "RMSE_Hz": 0.0, "hitRate": 0.0, "FAR": 0.0, "leadMedian_ms": 0.0},
        {"model": "quadratic", "targetFAR": 0.1, "policy": "fixed", "condition": "mixed",
         "MAE_Hz": 0.0, "RMSE_Hz": 0.0, "hitRate": 0.0, "FAR": 0.0, "leadMedian_ms": 0.0},
        {"model": "residual", "targetFAR": 0.1, "policy": "fixed", "condition": "mixed",
         "MAE_Hz": 0.0, "RMSE_Hz": 0.0, "hitRate": 0.0, "FAR": 0.0, "leadMedian_ms": 0.0},
    ],
    "thresholds": [
        {"model": "persistence", "targetFAR": 0.1, "policy": "fixed", "LowQuality": False,
         "CalibrationVariants": 55, "Threshold_pu": -0.005},
        {"model": "quadratic", "targetFAR": 0.1, "policy": "fixed", "LowQuality": False,
         "CalibrationVariants": 55, "Threshold_pu": -0.005},
        {"model": "residual", "targetFAR": 0.1, "policy": "fixed", "LowQuality": False,
         "CalibrationVariants": 55, "Threshold_pu": -0.005},
    ],
    "qualityAssignments": {
        "file": "repro/out/round2-noise/N01/quality-estimates.csv",
        "sha256": ZERO,
        "estimatedLowCount": 306,
        "estimatedLowHash": ZERO,
    },
    "outputs": [
        {"role": "wide", "file": "repro/out/round2-noise/N01/quality-summary-r2.csv", "sha256": ZERO, "bytes": 0},
        {"role": "long", "file": "repro/out/round2-noise/N01/quality-events.csv", "sha256": ZERO, "bytes": 0},
    ],
    "compatibility": {
        "qualityTables": {
            "datasetValue": "ieee39_multi",
            "joinKeys": ["Dataset", "Scenario", "Condition", "Model", "TargetFAR", "Policy"],
            "addedColumn": "NoiseImpl",
        },
        "designTable": {"file": design, "joinKey": "DesignID"},
    },
    "integrity": {
        "recordSha256": None,
        "selfCheck": {
            "schemaValid": True,
            "denominatorsConsistent": True,
            "bitwiseVerified": True,
        },
    },
}

OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")
