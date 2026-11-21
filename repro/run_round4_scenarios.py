"""Run disjoint shards of the predetermined R4 high-load stress design."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_round3_scenarios import simulate_design

OUT=Path(__file__).resolve().parent/"out"/"round4"


def design():
    rng=np.random.default_rng(39194019)
    rows=[]
    for level in (1.052,1.055,1.058,1.061,1.067,1.070,1.073,1.075):
        for pattern in ("known_a","known_b"):
            for step in np.r_[rng.uniform(.092,.111,3),rng.uniform(.130,.141,5)]:
                rows.append([1001+len(rows),"high_load",level,pattern,float(step)])
    return pd.DataFrame(rows,columns=["DesignID","Block","LoadScale","Pattern","StepFraction"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part",choices=["pilot","0","1"])
    args=parser.parse_args()
    data=design()
    rows=data.iloc[[0,3,8,11]] if args.part=="pilot" else data.iloc[int(args.part)::2]
    simulate_design(rows,OUT/args.part)
