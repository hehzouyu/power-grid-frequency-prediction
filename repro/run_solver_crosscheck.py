"""Compare three ANDES integrators on the preselected R4 and R5 cases."""
import argparse
import json
import logging
from pathlib import Path

import andes
import numpy as np
import pandas as pd

from run_andes_warning import TIME, coi_frequency, simulate
from run_round3_scenarios import PATTERNS
from run_round5_scenarios import NEW_PATTERNS

OUT = Path(__file__).resolve().parent / "out" / "solver-crosscheck"
METHODS = ("trapezoid", "backeuler", "qndf")
CASES = {"round4": ([1053, 1054, 1077, 1080, 1109, 1110], PATTERNS),
         "round5/transition": ([2101, 2102, 2113, 2114, 2121, 2122], NEW_PATTERNS)}


def collect(ss):
    ts = ss.dae.ts
    grid = 1 + TIME
    grid = grid[grid <= ts.t[-1] + 1e-9]
    result = {"LastAccepted_s": float(ts.t[-1]), "PrefixSamples": len(grid)}
    if not len(grid):
        return result
    samples = np.searchsorted(ts.t, grid + 1e-9, side="right") - 1
    coi = coi_frequency(ss)[samples] - 1
    voltage = ts.y[samples][:, ss.Bus.v.a]
    row, bus = np.unravel_index(voltage.argmin(), voltage.shape)
    crossing = np.flatnonzero(coi < -.005)
    result.update(PrefixNadirDeviation_pu=float(coi.min()),
                  ObservedCrossingAfterStep_s=float(grid[crossing[0]] - 1) if crossing.size else np.nan,
                  MinVoltage_pu=float(voltage[row, bus]), MinVoltageBus=int(ss.Bus.idx.v[bus]),
                  MinVoltageTime_s=float(grid[row]))
    return result


def execute(method, step=.01, selected=None):
    andes.config_logger(stream_level=logging.ERROR)
    dest = OUT / method if selected is None else OUT / f"{method}-{step:g}"
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "config.json").write_text(json.dumps({"andes": andes.__version__, "method": method,
                                                  "configured_step_s": step}, indent=2) + "\n")
    rows = []
    for batch, (ids, patterns) in CASES.items():
        if selected is not None:
            ids = [design_id for design_id in ids if design_id in selected]
        data = pd.read_csv(OUT.parent / batch / "design-results.csv").set_index("DesignID").loc[ids]
        for design_id, row in data.iterrows():
            diagnostics = {}

            def inspect(ss):
                diagnostics.update(collect(ss))

            try:
                simulate("ieee39", row.LoadScale, 1, row.StepFraction, True, patterns[row.Pattern],
                         integration_method=method, integration_step=step, inspect_trace=inspect)
                status, error = "ok", ""
            except RuntimeError as cause:
                status, error = "failed", str(cause)
            rows.append(dict(DesignID=design_id, Batch=batch, PriorStatus=row.Status, Method=method,
                             IntegrationStep_s=step, LoadScale=row.LoadScale, Pattern=row.Pattern,
                             StepFraction=row.StepFraction,
                             Status=status, Error=error, **diagnostics))
            pd.DataFrame(rows).to_csv(dest / "cases.csv", index=False)
            print(f"{method} {design_id}: {status} {error}", flush=True)


def load_main_results():
    batches = []
    expected = {design_id for ids, _ in CASES.values() for design_id in ids}
    for method in METHODS:
        data = pd.read_csv(OUT / method / "cases.csv")
        if len(data) != len(expected) or set(data.DesignID) != expected or not data.Method.eq(method).all():
            raise ValueError(f"Incomplete crosscheck: {method}")
        batches.append(data.assign(IntegrationStep_s=.01))
    return pd.concat(batches, ignore_index=True)


def refinement_ids(results):
    completion = results.pivot(index="DesignID", columns="Method", values="Status")
    disagreement = completion.backeuler.eq("ok") & completion[["trapezoid", "qndf"]].ne("ok").any(axis=1)
    return set(completion.index[disagreement])


def summarize():
    main = load_main_results()
    selected = refinement_ids(main)
    frames = [main]
    refinement = {}
    for step in (.005, .0025):
        if selected:
            rerun = pd.read_csv(OUT / f"backeuler-{step:g}" / "cases.csv")
            if (len(rerun) != len(selected) or set(rerun.DesignID) != selected
                    or not rerun.IntegrationStep_s.eq(step).all() or not rerun.Method.eq("backeuler").all()):
                raise ValueError(f"Incomplete backward Euler refinement at {step:g} s")
            frames.append(rerun)
            refinement[f"{1000 * step:g}"] = sorted(rerun.loc[rerun.Status.eq("ok"), "DesignID"].tolist())
    cases = pd.concat(frames, ignore_index=True)
    cases.to_csv(OUT / "cases.csv", index=False)
    findings = {"cases": main.DesignID.nunique(), "prior_completion": {},
                "only_backeuler_at_10ms": sorted(selected), "backeuler_completion_by_ms": refinement}
    for batch, group in main.groupby("Batch"):
        failed = group[group.PriorStatus.eq("failed")]
        prior_controls = group[group.PriorStatus.eq("ok")]
        findings["prior_completion"][batch] = {
            "failed_cases": failed.DesignID.nunique(),
            "controls_all_complete": int(prior_controls.groupby("DesignID").Status.apply(
                lambda status: status.eq("ok").all()).sum()),
            "all_methods_observed_crossing": int(failed.groupby("DesignID").ObservedCrossingAfterStep_s.apply(
                lambda crossing: crossing.notna().all()).sum()),
            "any_method_observed_crossing": int(failed.groupby("DesignID").ObservedCrossingAfterStep_s.apply(
                lambda crossing: crossing.notna().any()).sum()),
            "all_methods_stopped": int(failed.groupby("DesignID").Status.apply(
                lambda status: status.eq("failed").all()).sum()),
        }
    report = json.dumps(findings, indent=2) + "\n"
    (OUT / "summary.json").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=(*METHODS, "summarize"))
    parser.add_argument("--refine", type=float, choices=(.005, .0025))
    args = parser.parse_args()
    if args.method == "summarize" and args.refine is None:
        summarize()
    elif args.refine is None:
        execute(args.method)
    else:
        if args.method != "backeuler":
            parser.error("--refine is reserved for backward Euler")
        selected = refinement_ids(load_main_results())
        if selected:
            execute(args.method, args.refine, selected)
