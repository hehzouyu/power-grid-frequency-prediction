"""Apply frozen R3 thresholds to the independent R4 high-load scenarios."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import GradientBoostingRegressor

from analyze_quality_warning import CONDITIONS, STRESS, observe
from run_rolling_warning import ROOT, evaluate, extract, predictions, train, truth_times
from score_round3 import METHODS, SEALED, paired_comparisons, report

OUT=Path(__file__).resolve().parent/"out"/"round4"
METRICS=["PhysicalEvents","PhysicalRisk","RiskVariants","NonriskVariants","Hits","FalseAlarms",
         "Timely","FAR","TimelyLow","TimelyHigh","FARLow","FARHigh","Late","LeadMedian_ms","LeadP5_ms"]


def load_results(out=OUT, expected_ids=range(1001,1129)):
    batches=[]
    arrays=[]
    for shard in ("0","1"):
        directory=out/shard
        plan=pd.read_csv(directory/"frozen-design.csv")
        results=pd.read_csv(directory/"design-results.csv")
        assert plan.equals(results[plan.columns])
        batches.append(results)
        accepted = results.query("Status=='ok'").DesignID
        archive = directory / "trajectories.npz"
        if accepted.empty and not archive.exists():
            continue
        with np.load(archive) as file:
            assert np.array_equal(file["ids"], accepted)
            arrays.append((file["ids"],file["multi"],file["coi"]))
    combined=pd.concat(batches,ignore_index=True).sort_values("DesignID")
    assert combined.DesignID.tolist()==list(expected_ids)
    assert combined.LoadScale.min()>1.044
    combined.to_csv(out/"design-results.csv",index=False)
    if not arrays:
        raise ValueError(f"No scorable scenarios in {out}")
    ids,signal,truth=(np.concatenate(column) for column in zip(*arrays,strict=True))
    order=np.argsort(ids)
    design=combined.query("Status=='ok'").reset_index(drop=True)
    assert np.array_equal(ids[order],design.DesignID)
    return design,signal[order],truth[order]


def plot(summary,out):
    fig,ax=plt.subplots(figsize=(6.0,3.3),layout="constrained")
    subset=summary[summary.Condition=="mixed"]
    for method,marker in zip(METHODS,("o","^","s"),strict=True):
        rows=subset[subset.Model==method].sort_values("TargetFAR")
        ax.plot(100*rows.FAR,100*rows.Timely,marker=marker,label=method)
    ax.set(xlabel="Observed false-alarm rate (%)",ylabel="Timely hit rate (%)")
    ax.legend(frameon=False)
    fig.savefig(out/"tradeoff.pdf")
    plt.close(fig)


def run(out=OUT, expected_ids=range(1001,1129), noise_seed=39194020, *, report_patterns=False):
    design,signal,truth=load_results(out,expected_ids)
    threshold_source=pd.read_csv(SEALED/"thresholds.csv").set_index(["Model","TargetFAR","LowQuality"])
    assert threshold_source.index.is_unique
    archived=loadmat(ROOT/"out/revision/trajectories.mat",simplify_cells=True)["grid"]
    dev=(archived["F"][:,50:]-50)/50
    features,_,labels,_=extract(dev)
    residual=[GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=42).fit(
        x[1],np.maximum(x[0]-labels[:,i],0)) for i,x in enumerate(features)]
    models=train(dev)
    signals,_,quality=observe(signal,np.sqrt(np.mean(dev[:,:26]**2)),seed=noise_seed)
    crossing,nadir=truth_times(truth)
    frames=[]
    for j,condition in enumerate(CONDITIONS+STRESS):
        x,times,_,_=extract(signals[j])
        pred=predictions(models,x)
        pred["residual"]=np.column_stack([k[0]-np.maximum(fit.predict(k[1]),0)
                                          for k,fit in zip(x,residual,strict=True)])
        for method in METHODS:
            for far in (.05,.10,.20):
                cuts=np.array([threshold_source.loc[(method,far,level),"Cut_pu"] for level in (False,True)])
                applied=cuts[quality[:,j].astype(int)]
                alarm,lead,outcome,early=evaluate(pred[method]-applied[:,None],np.arange(4),0,times,crossing)
                frames.append(design[["DesignID","Block","LoadScale","Pattern"]].assign(Model=method,
                    TargetFAR=far,Condition=condition,Crossing_s=crossing,Nadir_s=nadir,
                    Alarm_s=alarm,Lead_ms=lead,Outcome=outcome,WindowMissed=early))
    events=pd.concat(frames,ignore_index=True)
    assert not events.duplicated(["DesignID","Condition","TargetFAR","Model"]).any()
    hit=events.Outcome.eq("hit")
    assert (events.loc[hit,"Alarm_s"]<events.loc[hit,"Crossing_s"]).all()
    rows=[]
    for key,frame in events.groupby(["Model","TargetFAR"]):
        for name,block in [("mixed",frame[frame.Condition.isin(CONDITIONS)]),*frame.groupby("Condition")]:
            rows.append([*key,name,*report(block)])
    summary=pd.DataFrame(rows,columns=["Model","TargetFAR","Condition",*METRICS])
    grouped=events[events.Condition.isin(CONDITIONS) & events.TargetFAR.eq(.10)]
    by_load=[]
    for (load,model),frame in grouped.groupby(["LoadScale","Model"]):
        by_load.append([load,model,*report(frame)])
    events.to_csv(out/"events.csv",index=False)
    summary.to_csv(out/"summary.csv",index=False)
    paired_comparisons(events).to_csv(out/"paired.csv",index=False)
    pd.DataFrame(by_load,columns=["LoadScale","Model",*METRICS]).to_csv(out/"by-load.csv",index=False)
    if report_patterns:
        by_pattern = [[*key, *report(frame)] for key, frame in grouped.groupby(["Pattern", "Model"])]
        pd.DataFrame(by_pattern, columns=["Pattern", "Model", *METRICS]).to_csv(out/"by-pattern.csv",index=False)
    sources=[SEALED/"thresholds.csv",ROOT/"out/revision/trajectories.mat",
             out/"0"/"frozen-design.csv",out/"1"/"frozen-design.csv"]
    metadata={"noise_seed":noise_seed,"physical_events":len(design),
              "physical_risk_events":int(np.isfinite(crossing).sum()),
              "sources_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources}}
    (out/"score-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    plot(summary,out)
    print(summary.query("Condition=='mixed' and TargetFAR==.10").to_string(index=False))


if __name__=="__main__":
    run()
