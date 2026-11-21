"""Generate the frozen IEEE 39 R3 design in a separate output directory."""
import argparse
import logging
from pathlib import Path

import andes
import numpy as np
import pandas as pd

from run_andes_warning import simulate

OUT=Path(__file__).resolve().parent/"out"/"round3"
PATTERNS={"known_a":(1,4,7,10),"known_b":(2,5,8,11),"held":(1,5,7,11)}
BLOCKS={
    "calibration":((.956,.982,1.018,1.044),("known_a","known_b")),
    "load_holdout":((.923,.936,1.064,1.077),("known_a","known_b")),
    "pattern_holdout":((.969,1.027),("held",)),
}


def build_design():
    rng=np.random.default_rng(39193019)
    rows=[]
    for block,(loads,patterns) in BLOCKS.items():
        for load in loads:
            for pattern in patterns:
                for step in np.r_[rng.uniform(.085,.122,5),rng.uniform(.128,.143,3)]:
                    rows.append([len(rows)+1,block,load,pattern,float(step)])
    return pd.DataFrame(rows,columns=["DesignID","Block","LoadScale","Pattern","StepFraction"])


def run(smoke):
    dest=OUT/("smoke" if smoke else "sealed")
    design=build_design()
    if smoke:
        design=design.iloc[[0,5,26,30]]
    simulate_design(design,dest)


def simulate_design(design,dest,patterns=PATTERNS,*,allow_empty=False):
    andes.config_logger(stream_level=logging.ERROR)
    dest.mkdir(parents=True,exist_ok=False)
    design.to_csv(dest/"frozen-design.csv",index=False)
    traces=[]
    rows=[]
    for row in design.itertuples(index=False):
        try:
            coi,single,multi,buses,measure=simulate("ieee39",row.LoadScale,1,row.StepFraction,
                                                 True,patterns[row.Pattern])
            traces.append((row.DesignID,coi,single,multi))
            result=("ok","",buses,";".join(map(str,measure)),float(coi.min()))
        except Exception as error:
            result=("failed",str(error),"","",np.nan)
        rows.append([*row,*result])
        if len(rows)%12==0 or len(design)<=4:
            print(f"{len(rows)}/{len(design)}, successful {len(traces)}",flush=True)
    pd.DataFrame(rows,columns=[*design.columns,"Status","Error","LoadBuses","MeasurementBuses",
                               "NadirDeviation_pu"]).to_csv(dest/"design-results.csv",index=False)
    if not traces:
        if allow_empty:
            return
        raise RuntimeError("No successful scenario")
    np.savez_compressed(dest/"trajectories.npz",ids=np.array([x[0] for x in traces]),
        **{name:np.stack([x[i] for x in traces]) for i,name in enumerate(("coi","single","multi"),1)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    run(parser.parse_args().smoke)
