"""Check static AC feasibility on the dynamic crosscheck scenarios with SciPy."""
import hashlib
import json
import logging
from pathlib import Path

import andes
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import root

from run_andes_warning import CASES as MODEL_FILES, load
from run_solver_crosscheck import CASES as SCENARIOS, OUT as DYNAMIC_RESULTS

OUT = Path(__file__).resolve().parent / "out" / "static-flow-check"
TOLERANCE = 1e-8


def admittance(system):
    line = system.Line
    uids = system.Bus.idx2uid
    f, t = np.array(uids(line.bus1.v)), np.array(uids(line.bus2.v))
    series = line.u.v / (line.r.v + 1j * line.x.v)
    charging = line.u.v * (line.g.v + 1j * line.b.v) / 2
    end1 = line.u.v * (line.g1.v + 1j * line.b1.v) + charging
    end2 = line.u.v * (line.g2.v + 1j * line.b2.v) + charging
    tap = line.tap.v * np.exp(1j * line.phi.v)
    matrix = np.zeros((system.Bus.n, system.Bus.n), dtype=complex)
    np.add.at(matrix, (f, f), (series + end1) / np.abs(tap)**2)
    np.add.at(matrix, (t, t), series + end2)
    np.add.at(matrix, (f, t), -series / tap.conj())
    np.add.at(matrix, (t, f), -series / tap)
    if system.Shunt.n:
        shunt = system.Shunt
        idx = np.array(uids(shunt.bus.v))
        np.add.at(matrix, (idx, idx), shunt.u.v * (shunt.g.v + 1j * shunt.b.v))
    return matrix


def solve_case(scale, fraction, ranks):
    system = load("ieee39")
    pq = system.PQ
    active = np.array(pq.p0.v) * scale
    reactive = np.array(pq.q0.v) * scale
    selected = np.argsort(active)[::-1][list(ranks)]
    delta = np.zeros(pq.n, dtype=complex)
    delta[selected] = active.sum() * fraction * (active[selected] + 1j * reactive[selected]) / active[selected].sum()
    for i, device in enumerate(pq.idx.v):
        pq.set("p0", device, active[i])
        pq.set("q0", device, reactive[i])
    if not system.setup() or not system.PFlow.run():
        raise RuntimeError("ANDES operating point unavailable for comparison")
    uids = system.Bus.idx2uid
    buses = np.array(uids(pq.bus.v))
    pv = np.array(uids(system.PV.bus.v))[system.PV.u.v > 0]
    slack = np.array(uids(system.Slack.bus.v))[system.Slack.u.v > 0]
    angles = np.setdiff1d(np.arange(system.Bus.n), slack)
    magnitudes = np.setdiff1d(angles, pv)
    base_a, base_v = system.Bus.a.v.copy(), system.Bus.v.v.copy()
    loads = active + 1j * reactive
    if (np.any(base_v[buses] < pq.vmin.v) or np.any(base_v[buses] > pq.vmax.v)):
        raise RuntimeError("Initial PQ impedance substitution would invalidate comparison")
    generation = np.zeros(system.Bus.n)
    np.add.at(generation, np.array(uids(system.PV.bus.v)), system.PV.p0.v * system.PV.u.v)
    ybus = admittance(system)

    def voltage(state):
        a, v = base_a.copy(), base_v.copy()
        a[angles], v[magnitudes] = state[:len(angles)], state[len(angles):]
        return v * np.exp(1j * a)

    def balance(state, step):
        phasor = voltage(state)
        mismatch = phasor * (ybus @ phasor).conj() - generation
        np.add.at(mismatch, buses, (loads + step * delta) * pq.u.v)
        return np.r_[mismatch[angles].real, mismatch[magnitudes].imag]

    initial = np.r_[base_a[angles], base_v[magnitudes]]
    reference_residual = float(np.linalg.norm(balance(initial, 0), ord=np.inf))
    if reference_residual > 1e-4:
        raise RuntimeError(f"AC equations and ANDES initialization differ: {reference_residual:g} p.u.")
    state = initial
    results = []
    reference_voltage_error = np.nan
    for step in np.linspace(0, 1, 21):
        fit = root(balance, state, args=(step,), method="hybr", options={"xtol": 1e-10})
        residual = float(np.linalg.norm(balance(fit.x, step), ord=np.inf))
        v = np.abs(voltage(fit.x))
        accepted = residual <= TOLERANCE and np.all(fit.x[len(angles):] > 0)
        if step == 0:
            reference_voltage_error = float(np.max(np.abs(v - base_v)))
            if not accepted or reference_voltage_error > 1e-5:
                raise RuntimeError(f"Base voltage check failed: {reference_voltage_error:g} p.u.")
        min_bus = int(v.argmin())
        results.append(dict(StepShare=step, PFStatus="ok" if accepted else "failed",
                            ResidualInf_pu=residual, StaticMinVoltage_pu=float(v[min_bus]) if accepted else np.nan,
                            StaticMinVoltageBus=int(system.Bus.idx.v[min_bus]) if accepted else None,
                            ReferenceResidualInf_pu=reference_residual,
                            BaseVoltageMaxError_pu=reference_voltage_error))
        if not accepted:
            break
        state = fit.x
    return results


def run():
    andes.config_logger(stream_level=logging.ERROR)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch, (ids, patterns) in SCENARIOS.items():
        design = pd.read_csv(DYNAMIC_RESULTS.parent / batch / "design-results.csv").set_index("DesignID")
        for design_id, case in design.loc[ids].iterrows():
            result = solve_case(case.LoadScale, case.StepFraction, patterns[case.Pattern])
            rows.extend(dict(DesignID=design_id, Batch=batch, PriorDynamicStatus=case.Status, **r) for r in result)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "continuation.csv", index=False)
    summary = table.groupby("DesignID", sort=False).tail(1)
    summary.to_csv(OUT / "summary.csv", index=False)
    case_path = Path(andes.get_case(MODEL_FILES["ieee39"][0]))
    metadata = dict(andes=andes.__version__, scipy=scipy.__version__, case=case_path.name,
                    case_sha256=hashlib.sha256(case_path.read_bytes()).hexdigest(), tolerance_pu=TOLERANCE,
                    step_count=20, type="constant PQ, fixed PV setpoints, no generator reactive limits")
    (OUT / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
