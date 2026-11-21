"""Estimate within-implementation event design effects from frozen R2 outputs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/"repro/out/round2/multi-noise/events-long.csv"
OUT=ROOT/"reports/round3"


def main():
    frame=pd.read_csv(SOURCE).query("TargetFAR==0.1 and Risk and Model=='persistence'")
    matrix=frame.assign(hit=frame.Outcome.eq("hit")).pivot(
        index="Seed",columns=["DesignID","Condition"],values="hit")
    ids=matrix.columns.get_level_values(0).unique()
    assert len(matrix)==40 and len(ids)==21 and matrix.shape==(40,126)
    cube=np.stack([matrix.xs(i,axis=1,level=0).to_numpy(float) for i in ids],axis=1)
    rng=np.random.default_rng(39193018)
    rows=[]
    for title,chunk,required in (("MAIN",cube[:20],1.4286),("PREC",cube,1.3563)):
        cluster=chunk.mean(axis=2)
        p=cluster.mean(axis=1)
        binomial=p*(1-p)/126
        # Ratio of average within-implementation cluster variance to IID variance.
        cluster_var=cluster.var(axis=1,ddof=1)/len(ids)
        estimate=cluster_var.mean()/binomial.mean()
        draws=rng.integers(0,len(ids),(5000,len(ids)))
        replicated=chunk[:,draws,:]
        cluster_boot=replicated.mean(axis=-1)
        pb=cluster_boot.mean(axis=-1)
        ratios=(cluster_boot.var(axis=-1,ddof=1)/len(ids)).mean(axis=0)/(
            pb*(1-pb)/126).mean(axis=0)
        lo,hi=np.quantile(ratios,[.025,.975])
        rows.append({"set":title,"implementations":len(chunk),"physical_risk_events":len(ids),
                     "DEFF":estimate,"low_95":lo,"high_95":hi,"R2_pass_threshold":required,
                     "point_estimate_pass":bool(estimate>=required)})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"R2-cluster-audit.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":
    main()
