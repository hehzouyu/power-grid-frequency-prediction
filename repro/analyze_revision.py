"""Fit the revised models and build manuscript tables from MATLAB exports."""
from pathlib import Path
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import loadmat
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "revision"
ART = HERE.parent / "artical"
GENERATED = ART / "revision-assets"
FEATURES = ("endpoint", "initial_rate", "quadratic", "segmented", "fitted_segments", "vmd")
LABELS = {
    "persistence": "持续性",
    "endpoint": "端点",
    "initial_rate": "端点＋初始变化率",
    "quadratic": "端点＋二次拟合",
    "segmented": "端点＋分段差分",
    "fitted_segments": "端点＋分段拟合",
    "vmd": "端点＋变分模态分解",
}
TREES = FEATURES


def fit_model(matrix, target, train, valid, model):
    scaler = StandardScaler().fit(matrix[train])
    z = scaler.transform(matrix)
    if model == "ridge":
        scores = []
        for alpha in np.logspace(-4, 2, 13):
            candidates = Ridge(alpha=alpha).fit(z[train], target[train])
            scores.append((np.mean((candidates.predict(z[valid]) - target[valid]) ** 2), alpha))
        fitted = np.r_[train, valid]
        estimator = Ridge(alpha=min(scores)[1]).fit(z[fitted], target[fitted])
    else:
        estimator = GradientBoostingRegressor(
            loss="squared_error", n_estimators=100, learning_rate=.1,
            max_depth=3, min_samples_leaf=1, random_state=42,
        ).fit(z[train], target[train])
    return estimator, scaler


def predict(fit, matrix):
    estimator, scaler = fit
    return estimator.predict(scaler.transform(matrix))


def metrics(y, predicted):
    err = predicted - y
    sst = np.sum((y - y.mean()) ** 2)
    return {
        "n": len(y), "rmse": np.sqrt(np.mean(err**2)), "mae": np.mean(abs(err)),
        "p95": np.percentile(abs(err), 95),
        "r2": 1 - np.sum(err**2) / sst if sst > 0 else np.nan,
    }


def features(items):
    return dict(zip(FEATURES, (np.asarray(x).reshape(len(x), -1) for x in items), strict=True))


def record(rows, experiment, feat, model, index, ids, predictions, level="", rep=0):
    part = index.iloc[ids][["ID", "Event", "Lead_s", "Target_Hz"]].copy()
    part.insert(0, "Experiment", experiment)
    part.insert(1, "Feature", feat)
    part.insert(2, "Model", model)
    part["Prediction_Hz"] = predictions
    part["Level"] = level
    part["Rep"] = rep
    rows.append(part)


def analyze(grid, continuous, cascade, inputs, threshold):
    X = features(inputs["X"])
    Xc = features(inputs["Xc"])
    Xs = features(inputs["Xs"])
    target = grid.Target_Hz.to_numpy()
    train = np.flatnonzero(grid.Split == "train")
    valid = np.flatnonzero(grid.Split == "validation")
    test = np.asarray(inputs["te"], dtype=int) - 1
    full = np.flatnonzero((grid.Split == "test") | (grid.Event != "after"))
    eligible = np.flatnonzero(grid.Event == "after")
    ce = np.arange(len(continuous))
    assert np.array_equal(np.sort(test), np.flatnonzero(grid.Split == "test"))
    assert not np.intersect1d(full, np.r_[train, valid]).size
    assert len(grid) == 1200 and len(continuous) == len(cascade) == 500
    assert continuous[["ID","H","D","R","Tg","d"]].equals(cascade[["ID","H","D","R","Tg","d"]])
    rows = []
    for experiment, ids in (("conditional", test), ("full", full)):
        record(rows, experiment, "persistence", "baseline", grid, ids,
               50 + X["endpoint"][ids, 0])
    for feat, matrix in X.items():
        for model in ("ridge", "tree"):
            fitted = fit_model(matrix, target, train, valid, model)
            for experiment, ids in (("conditional", test), ("full", full)):
                record(rows, experiment, feat, model, grid, ids, predict(fitted, matrix[ids]))
            record(rows, "continuous", feat, model, continuous, ce, predict(fitted, Xc[feat]))
            record(rows, "cascade", feat, model, cascade, ce, predict(fitted, Xs[feat]))
            if model == "tree" and feat in TREES:
                dev_h = np.flatnonzero((grid.H < 6) & (grid.Event == "after"))
                test_h = np.flatnonzero((grid.H == 6) & (grid.Event == "after"))
                model_h = fit_model(matrix, target, dev_h, [], "tree")
                record(rows, "H6", feat, model, grid, test_h, predict(model_h, matrix[test_h]))
                j = FEATURES.index(feat)
                for rep, noise_set in enumerate(inputs["noiseFeatures"], 1):
                    for snr, sample in zip(inputs["snrs"], noise_set, strict=True):
                        noisy = np.asarray(sample[j]).reshape(len(test), -1)
                        record(rows, "noise", feat, model, grid, test, predict(fitted, noisy),
                               int(snr), rep)
    for width, col in zip(inputs["widths"], inputs["window"].T, strict=True):
        matrix = col[:, None]
        fitted = fit_model(matrix, target, train, valid, "tree")
        record(rows, "window", "endpoint", "tree", grid, test, predict(fitted, matrix[test]), width)
    for seed in range(1, 11):
        perm = np.random.default_rng(seed).permutation(eligible)
        tr, va, te = np.split(perm, [len(train), len(train) + len(valid)])
        for feat in TREES:
            matrix = X[feat]
            fitted = fit_model(matrix, target, tr, va, "tree")
            record(rows, "repeat", feat, "tree", grid, te, predict(fitted, matrix[te]), "", seed)
    predictions = pd.concat(rows, ignore_index=True)
    assert np.isfinite(predictions.Prediction_Hz).all()
    predictions["ActualRisk"] = predictions.Target_Hz < threshold
    predictions["PredictedRisk"] = predictions.Prediction_Hz < threshold
    return predictions


def summarize(predictions):
    rows = []
    keys = ["Experiment", "Feature", "Model", "Level", "Rep"]
    for key, data in predictions.groupby(keys, sort=False):
        groups = [("all", data)]
        if key[0] in ("full", "continuous", "cascade"):
            groups += list(data.groupby("Event"))
        for event, part in groups:
            row = dict(zip(keys, key, strict=True)) | {"Event": event}
            row |= metrics(part.Target_Hz.to_numpy(), part.Prediction_Hz.to_numpy())
            actual, alert = part.ActualRisk, part.PredictedRisk
            row |= {
                "tp": int((actual & alert).sum()), "fn": int((actual & ~alert).sum()),
                "fp": int((~actual & alert).sum()), "tn": int((~actual & ~alert).sum()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def table_rows(name, rows):
    text = "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) + "\n"
    (GENERATED / name).write_text(text, encoding="utf-8")


def build_assets(stats, grid, continuous, cascade):
    def result(exp, feat, model="tree", event="all"):
        selected = stats[(stats.Experiment == exp) & (stats.Feature == feat)
                         & (stats.Model == model) & (stats.Event == event)]
        return selected.iloc[0]

    def values(row):
        return [f"{row.rmse:.4f}", f"{row.mae:.4f}", f"{row.p95:.4f}", f"{row.r2:.3f}"]

    table_rows("main.tex", [
        [LABELS[feat], "持续性基线" if model == "baseline" else
         ("岭回归" if model == "ridge" else "梯度提升树"),
         *values(result("conditional", feat, model))]
        for feat, model in [("persistence", "baseline")] + [
            (feat, model) for feat in FEATURES for model in ("ridge", "tree")]
    ])
    table_rows("groups.tex", [
        [LABELS[feat], {"before":"截止前","at":"截止时","after":"截止后","all":"合计"}[event],
         row.n, f"{row.rmse:.4f}", f"{row.p95:.4f}", row.fn, row.fp]
        for feat, model in [("persistence","baseline"),("fitted_segments","tree")]
        for event in ["before", "at", "after", "all"]
        if not (row := result("full", feat, model, event)).empty
    ])
    table_rows("validation.tex", [
        [LABELS[feat], *[
            number for exp in ("continuous","H6","cascade")
            for number in (f"{result(exp,feat).rmse:.4f}",f"{result(exp,feat).r2:.3f}")]]
        for feat in TREES
    ])
    noise = stats.loc[stats.Experiment == "noise"].copy()
    table_rows("noise.tex", [
        [LABELS[feat], *[
            f"{block.rmse.mean():.4f} / {block.p95.mean():.4f}"
            for snr in (40, 30, 20)
            if not (block := noise[(noise.Feature == feat) & (noise.Level == snr)]).empty]]
        for feat in TREES
    ])
    table_rows("windows.tex", [
        [f"{float(row.Level):.2f}", *values(row)]
        for row in stats[stats.Experiment == "window"].itertuples()
    ])
    repeats = stats[stats.Experiment == "repeat"]
    table_rows("repeat.tex", [
        [LABELS[feat], f"${block.rmse.mean():.4f}\\pm{block.rmse.std():.4f}$",
         f"${block.r2.mean():.3f}\\pm{block.r2.std():.3f}$"]
        for feat in TREES if not (block := repeats[repeats.Feature == feat]).empty
    ])

    parts = [("Grid", grid), ("Continuous", continuous), ("Cascade", cascade)]
    macros = {}
    for prefix, data in parts:
        for group in ("before", "at", "after"):
            macros[prefix + group.capitalize()] = int((data.Event == group).sum())
    for name in ("train", "validation", "test"):
        macros[name.capitalize() + "N"] = int((grid.Split == name).sum())
    macros["FullN"] = result("full","fitted_segments").n
    full = result("full","fitted_segments")
    macros["FullAlerts"] = full.tp + full.fn
    macros["FullMisses"] = full.fn
    macros["FullFalseAlerts"] = full.fp
    macros["FullBaselineMisses"] = result("full","persistence","baseline").fn
    lead = grid.loc[grid.Event == "after", "Lead_s"]
    for name, val in zip(("LeadMin", "LeadPfive", "LeadMedian"), np.percentile(lead,[0,5,50])):
        macros[name] = f"{val:.2f}"
    for prefix, exp, feat in (
        ("MainEndpoint","conditional","endpoint"), ("MainSegment","conditional","segmented"),
        ("MainFit","conditional","fitted_segments"), ("MainVMD","conditional","vmd"),
        ("MainQuad","conditional","quadratic"), ("CascadeFit","cascade","fitted_segments"),
        ("CascadeQuad","cascade","quadratic"),
        ("ConfirmFit","continuous","fitted_segments"), ("HFit","H6","fitted_segments"),
        ("ConfirmSegment","continuous","segmented"), ("HSegment","H6","segmented"),
    ):
        row = result(exp, feat)
        macros[prefix+"RMSE"] = f"{row.rmse:.4f}"
        macros[prefix+"Rsq"] = f"{row.r2:.3f}"
    for prefix, feat in (("Seg","segmented"),("Fit","fitted_segments"),
                         ("End","endpoint"),("Quad","quadratic")):
        block = noise[(noise.Feature == feat) & (noise.Level == 20)]
        macros[prefix+"NoiseRMSE"] = f"{block.rmse.mean():.4f}"
        macros[prefix+"NoiseP"] = f"{block.p95.mean():.4f}"
    (GENERATED / "metrics.tex").write_text(
        "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key,value in macros.items()), encoding="utf-8")

    raw = loadmat(OUT / "trajectories.mat", simplify_cells=True)
    time = raw["t"] - 1
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.6), sharey=False, layout="constrained")
    fig.supxlabel("扰动后时间 / s", fontsize=9)
    fig.supylabel("频率 / Hz", fontsize=9)
    for ax, group, tag in zip(axes, ("before", "at", "after"), ("(a)", "(b)", "(c)"), strict=True):
        members = np.flatnonzero(grid.Event == group)
        chosen = members[np.argsort(grid.Lead_s.iloc[members].to_numpy())[len(members)//2]]
        trace = raw["grid"]["F"][chosen]
        ax.axvspan(0,.5,color=".90")
        ax.axvline(.5,color=".35",linestyle="--",linewidth=.9)
        ax.plot(time,trace,color="#155366",linewidth=1.3)
        mask = np.flatnonzero(time > .5)
        index = mask[np.argmin(trace[mask])]
        glob = np.argmin(trace)
        ax.plot(time[glob],trace[glob],"ko",markersize=4)
        ax.plot(time[index],trace[index],marker="x",color="#832721",markersize=6)
        ax.set(xlim=(-.10,1.50), title=tag)
        ax.tick_params(labelsize=8)
    fig.savefig(GENERATED / "timeline.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7,3.1), layout="constrained")
    for (feat, label), marker, line in zip(
        zip(TREES, ("端点","初始变化率","二次拟合","分段差分","分段拟合","变分模态分解"), strict=True),
        ("o","v","P","s","^","D"), ("-","--","-.",":","--","-."), strict=True,
    ):
        block = noise[noise.Feature == feat].groupby("Level").rmse.agg(["mean","std"])
        ax.errorbar(block.index,block["mean"],yerr=block["std"],label=label,
                    marker=marker,linestyle=line,capsize=3)
    ax.set(xlabel="信噪比 / dB", ylabel="均方根误差 / Hz", xticks=[20,30,40])
    ax.legend(frameon=False,fontsize=8)
    fig.savefig(GENERATED / "noise.png",dpi=300)
    plt.close(fig)


def main():
    grid = pd.read_csv(OUT / "grid-index.csv")
    continuous = pd.read_csv(OUT / "continuous-index.csv")
    cascade = pd.read_csv(OUT / "cascade-index.csv")
    inputs = loadmat(OUT / "features.mat", simplify_cells=True)
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    predictions = analyze(grid, continuous, cascade, inputs, config["threshold_Hz"])
    stats = summarize(predictions)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    stats.to_csv(OUT / "summary.csv", index=False)
    software = {"python":platform.python_version(), "numpy":np.__version__,
                "scipy":scipy.__version__, "scikit-learn":sklearn.__version__}
    (OUT / "python-config.json").write_text(json.dumps(software, indent=2) + "\n", encoding="utf-8")
    GENERATED.mkdir(parents=True, exist_ok=True)
    build_assets(stats, grid, continuous, cascade)
    print(f"Verified {len(grid)} grid samples, {len(continuous)} continuous samples and {len(predictions)} predictions.")


if __name__ == "__main__":
    main()
