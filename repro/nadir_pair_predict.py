"""Freeze historical regressors before the independent simulation; then infer on X only."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from analyze_revision import OUT, fit_model, loadmat, predict

HERE = Path(__file__).resolve().parent
FEATURE_COLUMNS = {
    "endpoint": ["Endpoint"],
    "initial_rate": ["Endpoint", "InitialRate"],
    "quadratic": ["Endpoint", "Linear", "Curvature"],
    "fitted_segments": ["Endpoint", "Slope1", "Slope2", "Slope3"],
}
SLOTS = [0, 1, 2, 4]
KEYS = ["ReferenceID", "PairID", "Member", "Condition", "Rep"]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sources():
    project = HERE.parent / "Project"
    return [
        HERE / name for name in (
            "nadir_pair_protocol.json", "nadir_pair_test.m", "nadir_pair_predict.py",
            "run_nadir_pair_test.ps1", "analyze_revision.py",
        )
    ] + [
        OUT / name for name in (
            "grid-index.csv", "features.mat", "predictions.csv", "trajectories.mat",
            "continuous-index.csv",
        )
    ] + [OUT.parent / "nadir-intervals" / "scenarios.csv",
         OUT.parent / "nadir-identifiability" / "matched-scan.csv"] + [
        project / name for name in (
            "core/simulate_freq.m", "core/swing_equation.m", "core/pfr_model.m",
            "utils/calc_nadir.m", "config/sim_config.m", "config/sys_config.m",
        )
    ]


def freeze(run):
    if (run / "freeze.json").exists() or (run / "pair-index.csv").exists():
        raise FileExistsError("Freeze is allowed once, before candidate generation.")
    config = json.loads((HERE / "nadir_pair_protocol.json").read_text())
    grid = pd.read_csv(OUT / "grid-index.csv")
    archived = pd.read_csv(OUT / "predictions.csv")
    matlab_anchor = pd.read_csv(run / "anchor-features.csv").set_index("ID")
    slots = loadmat(OUT / "features.mat", simplify_cells=True)["X"]
    tr, va, te = [np.flatnonzero(grid.Split.eq(role)) for role in ("train", "validation", "test")]
    y = grid.Target_Hz.to_numpy()
    assert len(te) == 232 and list(FEATURE_COLUMNS) == config["features"]
    reconstructed, check = {}, []
    for name, slot in zip(config["features"], SLOTS, strict=True):
        x = np.asarray(slots[slot]).reshape(len(grid), -1)
        test_ids = grid.ID.iloc[te]
        extracted = matlab_anchor.loc[test_ids, FEATURE_COLUMNS[name]].to_numpy()
        np.testing.assert_allclose(extracted, x[te], rtol=0, atol=1e-10)
        frozen = fit_model(x, y, tr, va, "tree")
        p = predict(frozen, x[te])
        original = archived.query(
            "Experiment=='conditional' and Model=='tree' and Feature==@name"
        ).set_index("ID").loc[test_ids, "Prediction_Hz"].to_numpy()
        np.testing.assert_allclose(p, original, rtol=0, atol=config["historical_prediction_tolerance_hz"])
        # Compare on the archived feature matrix; dense-tree split thresholds can be sensitive
        # to last-bit differences in a fresh MATLAB least-squares calculation.
        check.append({"Feature": name, "N": len(te), "FeatureMaxDelta_Hz_or_HzPerS":
                      abs(extracted-x[te]).max(), "PredictionMaxDelta_Hz": abs(p-original).max(),
                      "RMSE_Hz": np.sqrt(np.mean((p-y[te])**2))})
        reconstructed[name] = frozen
    pd.DataFrame(check).to_csv(run / "anchor-check.csv", index=False)
    config_path = run / "protocol.json"
    config_path.write_bytes((HERE / "nadir_pair_protocol.json").read_bytes())
    joblib.dump(reconstructed, run / "models.joblib")
    manifest = {
        "protocol_sha256": sha(config_path), "models_sha256": sha(run / "models.joblib"),
        "source_sha256": {str(p.relative_to(HERE.parent)): sha(p) for p in sources()},
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "grid_train": len(tr),
        "grid_validation": len(va), "grid_historical_test": len(te),
    }
    (run / "freeze.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def verify(run):
    manifest = json.loads((run / "freeze.json").read_text())
    assert sha(run / "protocol.json") == manifest["protocol_sha256"], "Protocol changed."
    assert sha(run / "models.joblib") == manifest["models_sha256"], "Model file changed."
    assert {str(p.relative_to(HERE.parent)): sha(p) for p in sources()} == manifest["source_sha256"], (
        "Source or archived inputs changed after freezing."
    )
    assert sklearn.__version__ == manifest["scikit_learn"] and platform.python_version() == manifest["python"]
    assert np.__version__ == manifest["numpy"] and pd.__version__ == manifest["pandas"]


def infer(run):
    verify(run)
    if (run / "predictions.csv").exists():
        raise FileExistsError("Use a new run directory to preserve previous evidence.")
    # This input carries keys and feature columns only. Labels stay in the pair ledger.
    x = pd.read_csv(run / "features.csv")
    assert set(x.columns) == set(KEYS + sum((list(c) for c in FEATURE_COLUMNS.values()), []))
    assert not x.duplicated(KEYS).any()
    assert np.isfinite(x.drop(columns=KEYS).to_numpy()).all()
    models = joblib.load(run / "models.joblib")  # Load only the local artifact frozen above.
    output = []
    for name, cols in FEATURE_COLUMNS.items():
        p = predict(models[name], x[cols].to_numpy()) if len(x) else np.array([])
        assert np.isfinite(p).all(), f"Nonfinite predictions for {name}"
        output.append(x[KEYS].assign(Feature=name, Prediction_Hz=p))
    pd.concat(output, ignore_index=True).to_csv(run / "predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "verify", "infer"])
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "verify": verify, "infer": infer}[args.stage](args.run.resolve())
