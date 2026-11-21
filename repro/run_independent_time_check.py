"""Run a separate BDF2/SuperLU integration of the existing ANDES equations."""
import argparse
import json
import logging
import warnings
from pathlib import Path

import andes
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from run_andes_warning import TIME, coi_frequency, prepare_scenario, simulate
from run_round5_scenarios import NEW_PATTERNS

OUT = Path(__file__).resolve().parent / "out" / "independent-time-check"
CONTROLS = (2101, 2113, 2121)
FAILED = (2102, 2114, 2122)
STEPS = (.005, .0025)
TOL = 1e-8


def sparse(matrix):
    return csc_matrix((np.asarray(matrix.V).ravel(),
                       (np.asarray(matrix.I).ravel(), np.asarray(matrix.J).ravel())), shape=matrix.size)


def integrate(system, step):
    tds, dae = system.TDS, system.dae
    models = system.exist.pflow_tds
    tds.init()
    if not tds.initialized or not np.all(np.isfinite(dae.xy)):
        raise RuntimeError("TDS initialization failed")
    x_addr, weights = [], []
    for generator in system.SynGen.models.values():
        if not generator.n:
            continue
        online = np.asarray(generator.u.v) > 0
        x_addr.extend(generator.omega.a[online])
        weights.extend(generator.M.v[online])

    def step_residual(trial, x_reference, alpha):
        dae.x[:], dae.y[:] = trial[:dae.n], trial[dae.n:]
        system.vars_to_models()
        tds.fg_update(models)
        residual = np.r_[dae.Tf * (alpha * dae.x - x_reference) - step * dae.f, dae.g]
        pegged = []
        for limiter in system.antiwindups:
            for indices, target, _ in limiter.x_set:
                residual[indices] = trial[indices] - target
                pegged.extend(indices)
        return residual, pegged

    def record(at):
        volts = dae.y[system.Bus.v.a]
        lowest = int(np.argmin(volts))
        active = sorted({dae.x_name[int(i)] for limiter in system.antiwindups
                         for indices, _, _ in limiter.x_set for i in indices})
        samples.append((at, np.average(dae.x[x_addr], weights=weights) - 1,
                        volts[lowest], int(system.Bus.idx.v[lowest]), ";".join(active), *volts))

    samples = []
    record(0.)
    last, prior = dae.x.copy(), None
    last_accepted, status, error = 0., "ok", ""
    read_every = round(.02 / step)
    for index in range(1, round(21 / step) + 1):
        t = index * step
        dae.set_t(t)
        alpha = 1 if prior is None else 1.5
        x_reference = last if prior is None else 2 * last - .5 * prior
        trial = dae.xy.copy()
        tds.mis = [1.]
        accepted = False
        for iteration in range(21):
            tds.niter = iteration
            residual, pegged = step_residual(trial, x_reference, alpha)
            norm = float(np.linalg.norm(residual, ord=np.inf))
            if norm <= TOL:
                accepted = True
                break
            if not np.isfinite(norm):
                break
            system.j_update(models)
            jac = bmat([[diags(alpha * dae.Tf) - step * sparse(dae.fx), -step * sparse(dae.fy)],
                        [sparse(dae.gx), sparse(dae.gy)]], format="csc")
            if pegged:
                jac = jac.tolil()
                for position in set(pegged):
                    jac.rows[position], jac.data[position] = [position], [1.]
                jac = jac.tocsc()
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    change = spsolve(jac, residual)
                except MatrixRankWarning:
                    break
            if not np.all(np.isfinite(change)):
                break
            trial -= change
            tds.mis = [norm]
        if not accepted:
            status, error = "failed", f"BDF2 nonlinear solve stopped at {t:.6f} s (residual {norm:.4g})"
            break
        last_accepted = t
        prior, last = last, dae.x.copy()
        if index % read_every == 0:
            record(t)
        if not tds.check_criteria():
            status, error = "failed", f"Rotor angle criterion failed at {t:.6f} s"
            break
        if index == round(1 / step):
            if 1. not in system.switch_dict:
                raise RuntimeError("Missing load event at 1 s")
            system.switch_action(system.switch_dict[1.])
            system.vars_to_models()
            prior = None
    timeline = pd.DataFrame(samples, columns=["Time_s", "COIDeviation_pu", "MinVoltage_pu",
        "MinVoltageBus", "PeggedStates", *(f"VoltageBus{bus}_pu" for bus in system.Bus.idx.v)])
    return timeline, status, last_accepted, error


def run_design(step, group):
    andes.config_logger(stream_level=logging.ERROR)
    design = pd.read_csv(OUT.parent / "round5" / "transition" / "design-results.csv").set_index("DesignID")
    reference = pd.read_csv(OUT.parent / "solver-crosscheck" / "qndf" / "cases.csv").set_index("DesignID")
    control_archive = OUT.parent / "round5" / "transition" / "0" / "trajectories.npz"
    dest = OUT / f"{step:g}" / group
    dest.mkdir(parents=True, exist_ok=False)
    rows = []
    with np.load(control_archive) as archive:
        original = dict(zip(archive["ids"], archive["coi"], strict=True))
    for design_id in (CONTROLS if group == "controls" else FAILED):
        scene = design.loc[design_id]
        system, _, _ = prepare_scenario("ieee39", scene.LoadScale, 1, scene.StepFraction,
                                        True, NEW_PATTERNS[scene.Pattern])
        trace, status, stop, error = integrate(system, step)
        trace.to_csv(dest / f"trace-{design_id}.csv", index=False)
        post = trace[trace.Time_s >= 1.]
        observed = post[post.COIDeviation_pu < -.005]
        crossing = observed.Time_s.iloc[0] - 1. if not observed.empty else np.nan
        min_v = float(post.MinVoltage_pu.min())
        control_error = voltage_error = np.nan
        if group == "controls" and status == "ok":
            control_error = float(np.max(np.abs(post.COIDeviation_pu - original[design_id])))
            voltage_error = abs(min_v - float(reference.loc[design_id, "MinVoltage_pu"]))
        passed = bool(status == "ok" and control_error <= .00005 and voltage_error <= .015)
        category = "observed_crossing" if len(observed) else "complete_no_crossing" if status == "ok" else "truncated_unknown"
        rows.append(dict(DesignID=design_id, Step_s=step, Group=group, Status=status, LastAccepted_s=stop,
                         Outcome=category, CrossingAfterEvent_s=crossing, MinVoltage_pu=min_v,
                         COIMaxError_pu=control_error, MinVoltageError_pu=voltage_error,
                         ControlPassed=passed, Error=error))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(dest / "summary.csv", index=False)


def check_sampling():
    """Compare interpolation and previous-sample hold of the original integrator."""
    andes.config_logger(stream_level=logging.ERROR)
    design = pd.read_csv(OUT.parent / "round5" / "transition" / "design-results.csv").set_index("DesignID")
    indexed_grid = 1 + TIME
    rows = []
    for design_id in CONTROLS:
        scene = design.loc[design_id]
        reference = {}

        def inspect(system):
            reference["time"] = system.dae.ts.t.copy()
            reference["coi"] = coi_frequency(system) - 1

        simulate("ieee39", scene.LoadScale, 1, scene.StepFraction, True, NEW_PATTERNS[scene.Pattern],
                 integration_step=.02, inspect_trace=inspect)
        interpolated = np.interp(indexed_grid, reference["time"], reference["coi"])
        held = reference["coi"][np.searchsorted(reference["time"], indexed_grid + 1e-9, side="right") - 1]
        for step in STEPS:
            trace = pd.read_csv(OUT / f"{step:g}" / "controls" / f"trace-{design_id}.csv")
            post = trace[trace.Time_s >= 1]
            end = len(post)
            arr = post.COIDeviation_pu.to_numpy()
            rows.append(dict(DesignID=design_id, Step_s=step,
                             InterpolatedCOIMaxError_pu=np.max(np.abs(arr - interpolated[:end])),
                             HeldCOIMaxError_pu=np.max(np.abs(arr - held[:end])),
                             ReferenceInterpolationVsHoldMaxError_pu=np.max(np.abs(interpolated - held))))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(OUT / "sampling-diagnostic.csv", index=False)
    print(diagnostics.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("controls", "failures", "summarize", "sampling"))
    parser.add_argument("--step", type=float, choices=STEPS)
    args = parser.parse_args()
    if args.group == "sampling":
        check_sampling()
    elif args.group == "summarize":
        groups = [pd.read_csv(OUT / f"{step:g}" / group / "summary.csv") for step in STEPS
                  for group in ("controls", "failures")]
        result = pd.concat(groups, ignore_index=True)
        expected = {(design_id, step) for design_id in CONTROLS + FAILED for step in STEPS}
        if len(result) != len(expected) or set(zip(result.DesignID, result.Step_s, strict=True)) != expected:
            raise ValueError("Incomplete six-case check")
        result.to_csv(OUT / "summary.csv", index=False)
        controls = result[result.Group.eq("controls")]
        failures = result[result.Group.eq("failures")]
        data = dict(andes=andes.__version__, scipy=scipy.__version__, control_runs=len(controls),
                    controls_accepted=int(controls.ControlPassed.sum()),
                    failure_runs=len(failures), failures_completed=int(failures.Status.eq("ok").sum()),
                    failures_observed_crossing=int(failures.Outcome.eq("observed_crossing").sum()),
                    failures_truncated_unknown=int(failures.Outcome.eq("truncated_unknown").sum()))
        (OUT / "summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        if args.step is None:
            parser.error("--step is required for integration")
        run_design(args.step, args.group)


if __name__ == "__main__":
    main()
