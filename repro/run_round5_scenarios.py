"""Run the frozen R5 spatial holdout or the separate R4 solver audit."""
import argparse
import logging
from pathlib import Path

import andes
import numpy as np
import pandas as pd

from run_andes_warning import simulate
from run_round3_scenarios import PATTERNS, simulate_design

OUT = Path(__file__).resolve().parent / "out" / "round5"
NEW_PATTERNS = {"new_a": (1, 3, 8, 12), "new_b": (2, 6, 9, 13)}
LOAD_LEVELS = (1.052, 1.058, 1.067, 1.075)
DESIGN_COLUMNS = ["DesignID", "Block", "LoadScale", "Pattern", "StepFraction"]


def build_design():
    rng = np.random.default_rng(39195019)
    rows = []
    for level in LOAD_LEVELS:
        for pattern in NEW_PATTERNS:
            for step in np.r_[rng.uniform(.092, .111, 2), rng.uniform(.130, .141, 4)]:
                rows.append([2001 + len(rows), "spatial_holdout", level, pattern, step])
    return pd.DataFrame(rows, columns=DESIGN_COLUMNS)


def transition_design():
    rows = [(level, pattern, fraction) for level in LOAD_LEVELS for pattern in NEW_PATTERNS
            for fraction in (.116, .121, .125, .128)]
    return pd.DataFrame([[2101 + i, "exploratory_transition", *row] for i, row in enumerate(rows)],
                        columns=DESIGN_COLUMNS)


def audit():
    andes.config_logger(stream_level=logging.ERROR)
    r4 = pd.read_csv(OUT.parent / "round4" / "design-results.csv")
    failed = r4[r4.Status == "failed"]
    passed = r4[r4.Status == "ok"]
    controls = set()
    for row in failed.itertuples():
        same = passed[(passed.LoadScale == row.LoadScale) & (passed.Pattern == row.Pattern)]
        controls.add(int(same.loc[(same.StepFraction - row.StepFraction).abs().idxmin(), "DesignID"]))
    dest = OUT / "r4-audit"
    dest.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in r4[r4.DesignID.isin(set(failed.DesignID) | controls)].itertuples():
        for step in (.020, .010, .005):
            try:
                coi, _, _, _, _ = simulate("ieee39", row.LoadScale, 1, row.StepFraction, True,
                                            PATTERNS[row.Pattern], integration_step=step)
                nadir, reason = float(coi.min()), ""
            except Exception as error:
                nadir, reason = np.nan, str(error)
            rows.append([row.DesignID, row.Status, row.LoadScale, row.Pattern, row.StepFraction, step,
                         "failed" if reason else "ok", nadir, reason])
            print(row.DesignID, step, reason or "ok", flush=True)
            pd.DataFrame(rows, columns=["DesignID", "R4Status", "LoadScale", "Pattern", "StepFraction",
                                       "IntegrationStep_s", "Status", "NadirDeviation_pu", "Error"]).to_csv(
                dest / "solver-check.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=["pilot", "0", "1", "audit", "transition-0", "transition-1"])
    part = parser.parse_args().part
    if part == "audit":
        audit()
    elif part.startswith("transition-"):
        shard = part[-1]
        simulate_design(transition_design().iloc[int(shard)::2], OUT / "transition" / shard,
                        NEW_PATTERNS, allow_empty=True)
    else:
        design = build_design()
        rows = design.iloc[[0, 2, 6, 8]] if part == "pilot" else design.iloc[int(part)::2]
        simulate_design(rows, OUT / part, NEW_PATTERNS, allow_empty=True)
