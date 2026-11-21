"""Reconcile submission counts and replay event timing from frozen records."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from run_rolling_warning import DT, ROOT, THRESHOLD, extract, policy_specs, predictions, simulate, train

OUT = ROOT / "out"
ROLL = OUT / "rolling"
PRIMARY = ("clean", "white40", "white20", "correlated20")
SHIFTS = dict(early20ms=-1, early40ms=-2, late20ms=1, late40ms=2, noise20_late40_latency20=2)


def milliseconds(values):
    return np.rint(np.asarray(values) * 1000)


def crossing(trace):
    below = trace < THRESHOLD
    return np.where(below.any(axis=1), below.argmax(axis=1) * 20., np.inf)


def check_events(events, truth, alarm, first_possible):
    observed = milliseconds(events.Alarm_s)
    finite_truth, finite_alarm = np.isfinite(truth), np.isfinite(alarm)
    expected = np.select([finite_truth & (alarm < truth), finite_truth & finite_alarm,
                          ~finite_truth & finite_alarm, finite_truth],
                         ["hit", "late", "false_alarm", "no_alarm"], default="true_negative")
    np.testing.assert_array_equal(milliseconds(events.Crossing_s), truth)
    np.testing.assert_array_equal(observed, alarm)
    np.testing.assert_array_equal(events.Outcome.to_numpy(), expected)
    early = events["Early" if "Early" in events else "WindowMissed"]
    np.testing.assert_array_equal(early.to_numpy(), finite_truth & (truth <= first_possible))
    hits = expected == "hit"
    np.testing.assert_allclose(events.Lead_ms.to_numpy()[hits], truth[hits] - alarm[hits], atol=1e-8)
    assert events.loc[~hits, "Lead_ms"].isna().all()


def aggregate_audit():
    log = pd.read_csv(ROLL / "events.csv")
    obs = pd.read_csv(ROLL / "observations.csv").query("Dataset=='aggregate'")
    index = pd.read_csv(ROLL / "scenarios.csv").query("Role=='test'")
    ids = index.Scenario.to_numpy()
    with np.load(ROLL / "aggregate-trajectories.npz") as trace:
        archived = trace["frequency_pu"][ids] - 1
        np.testing.assert_allclose(trace["dt"], DT)
        # Replay the unrounded frequency deviations before threshold comparison.
        truth = simulate(trace["params"][ids])
        calibration = simulate(trace["params"][:600])
        np.testing.assert_allclose(truth, archived, atol=1e-14)
    development = loadmat(ROOT / "out/revision/trajectories.mat", simplify_cells=True)["grid"]
    dev = (development["F"][:, 50:] - 50) / 50
    models = train(dev)
    protocol = json.loads((ROLL / "protocol-results.json").read_text(encoding="utf-8"))
    reference = protocol["noise_rms_reference_Hz"] / 50
    x, _, labels, _ = extract(calibration)
    dev_x, _, dev_y, _ = extract(dev)
    scales = np.maximum(np.mean(np.abs(predictions(models, dev_x)["quadratic"] - dev_y), axis=0), 1e-9)
    score = ((predictions(models, x)["quadratic"] - labels) / scales).max(axis=1)
    correction = np.sort(score)[int(np.ceil((len(score) + 1) * .90)) - 1] * scales
    np.testing.assert_allclose(correction * 50, protocol["joint_interval_correction_Hz"])
    checked = 0
    clean_cond = log[log.Dataset.eq("aggregate")]
    for condition, samples in obs.groupby("Condition"):
        shift = SHIFTS.get(condition, 0)
        latency = 20 if condition == "noise20_late40_latency20" else 0
        level = 20 if latency else int(condition[:-2]) if condition.endswith("dB") else None
        noise = reference / 10**(level / 20) if level else 0
        window, emitted, _, bounds = extract(truth, shift, noise, 20261119, latency / 1000)
        predicted = predictions(models, window)
        for method, values in predicted.items():
            np.testing.assert_allclose(samples[f"{method}_Hz_at_50"], 50 * (1 + values.ravel()), atol=1e-10)
        expected_end = np.tile(np.array(bounds)[:, 1], len(ids))
        np.testing.assert_array_equal(samples.Scenario, np.repeat(ids, 4))
        np.testing.assert_array_equal(samples.End_sample, expected_end)
        np.testing.assert_array_equal(samples.Start_sample, shift)
        np.testing.assert_array_equal(milliseconds(samples.Emitted_s), 20 * expected_end + latency)
        times = milliseconds(emitted)
        for (policy, target), frame in clean_cond[clean_cond.Condition.eq(condition)].groupby(["Policy", "TargetFAR"]):
            values, cols = policy_specs(predicted, correction)[policy]
            trigger = values[:, cols] < protocol["cutoffs_pu"][f"{policy}_{target}"]
            alarms = np.where(trigger.any(axis=1), times[cols][trigger.argmax(axis=1)], np.inf)
            frame = frame.set_index("Scenario").loc[ids]
            check_events(frame, crossing(truth), alarms, times[cols[0]])
            checked += len(frame)
    return checked, len(obs)


def validate_grid(events, truth):
    alarm = milliseconds(events.Alarm_s)
    assert np.isin(alarm, [200, 300, 400, 500, np.inf]).all()
    check_events(events, truth, alarm, 200)


def audit_other():
    quality = pd.read_csv(ROLL / "quality-events.csv")
    quality_index = pd.read_csv(ROLL / "quality-aggregate-index.csv")
    agg_trace = simulate(quality_index[["H", "D", "R", "Tg", "d"]].to_numpy())
    assert set(quality.query("Dataset=='aggregate'").Scenario) == set(quality_index.query("Role=='test'").Scenario)
    dynamic_index = pd.read_csv(ROLL / "ieee39quality-design.csv").query("Status=='ok'")
    with np.load(ROLL / "ieee39quality-trajectories.npz") as trace:
        dynamic_truth = pd.Series(crossing(trace["coi"]), index=dynamic_index.DesignID)
    for dataset, truth in [("aggregate", pd.Series(crossing(agg_trace), index=quality_index.Scenario)),
                           ("ieee39_multi", dynamic_truth)]:
        rows = quality[quality.Dataset.eq(dataset)]
        validate_grid(rows, truth.loc[rows.Scenario].to_numpy())
    records = {"quality": len(quality)}
    for round_name in ("round4", "round5"):
        folder = OUT / round_name
        event = pd.read_csv(folder / "events.csv")
        series = []
        for shard in ("0", "1"):
            with np.load(folder / shard / "trajectories.npz") as trace:
                series.append(pd.Series(crossing(trace["coi"]), index=trace["ids"]))
        validate_grid(event, pd.concat(series).loc[event.DesignID].to_numpy())
        protocol = json.loads((folder / "score-metadata.json").read_text(encoding="utf-8"))
        for name, expected in protocol["sources_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        records[round_name] = len(event)
    return records


def reconcile():
    datasets = [("rolling", ROLL / "events.csv", ROLL / "warning-summary.csv", ["Dataset", "Condition", "Policy", "TargetFAR"]),
                ("quality", ROLL / "quality-events.csv", ROLL / "quality-summary.csv", ["Dataset", "Model", "TargetFAR", "Policy", "Condition"]),
                *((name, OUT / name / "events.csv", OUT / name / "summary.csv", ["Model", "TargetFAR", "Condition"])
                  for name in ("round4", "round5"))]
    records = []
    for name, event_path, summary_path, keys in datasets:
        events = pd.read_csv(event_path)
        reported = pd.read_csv(summary_path).set_index(keys)
        if name != "rolling":
            events = pd.concat([events, events[events.Condition.isin(PRIMARY)].assign(Condition="mixed")])
        for key, block in events.groupby(keys):
            row = reported.loc[key]
            risk = int(np.isfinite(block.Crossing_s).sum())
            hits = int(block.Outcome.eq("hit").sum())
            false = int(block.Outcome.eq("false_alarm").sum())
            nonrisk = len(block) - risk
            median = block.loc[block.Outcome.eq("hit"), "Lead_ms"].median() if hits else np.nan
            assert risk == row.get("Risk", row.get("RiskVariants")) and hits == row.Hits
            assert false == row.get("False", row.get("FalseAlarms"))
            np.testing.assert_allclose(row.FAR, false / nonrisk if nonrisk else np.nan)
            np.testing.assert_allclose(row.LeadMedian_ms, median)
            if name != "rolling":
                np.testing.assert_allclose(row.Timely, hits / risk if risk else np.nan)
            records.append(dict(Batch=name, **dict(zip(keys, key, strict=True)),
                                N=len(block), Risk=risk, Nonrisk=nonrisk, Hits=hits, FalseAlarms=false,
                                FAR=row.FAR, LeadMedian_ms=median, Source=summary_path.relative_to(ROOT).as_posix()))
    return pd.DataFrame(records)


if __name__ == "__main__":
    destination = OUT / "submission-audit"
    destination.mkdir(parents=True, exist_ok=True)
    timed, observations = aggregate_audit()
    scope = dict(aggregate_event_replay=timed, aggregate_observation_replay=observations, **audit_other())
    metrics = reconcile()
    metrics.to_csv(destination / "metrics.csv", index=False)
    (destination / "scope.json").write_text(json.dumps(scope | {"summary_rows": len(metrics)}, indent=2) + "\n")
    print(scope, len(metrics))
