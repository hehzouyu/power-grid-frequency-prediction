"""Score the frozen R3 scenarios without changing the simulation design."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import GradientBoostingRegressor

from analyze_quality_warning import CONDITIONS, STRESS, observe
from run_rolling_warning import ROOT, evaluate, extract, predictions, threshold, train, truth_times

SEALED=Path(__file__).resolve().parent/"out"/"round3"/"sealed"
METHODS=("persistence","quadratic","residual")


def report(frame):
    totals=frame.assign(risk=np.isfinite(frame.Crossing_s),hit=frame.Outcome.eq("hit"),
                        false=frame.Outcome.eq("false_alarm"))
    totals["negative"]=~totals.risk
    grouped=totals.groupby("DesignID")[["risk","negative","hit","false"]].sum().to_numpy()
    raw=grouped.sum(axis=0)
    draws=np.random.default_rng(39193021).integers(0,len(grouped),(3000,len(grouped)))
    reps=grouped[draws].sum(axis=1)
    ci=[]
    for numerator,denom in ((2,0),(3,1)):
        samples=reps[reps[:,denom]>0]
        ci.extend(np.percentile(samples[:,numerator]/samples[:,denom],[2.5,97.5])
                  if len(samples) else [np.nan,np.nan])
    return [len(grouped),int((grouped[:,0]>0).sum()),*raw,
        raw[2]/raw[0] if raw[0] else np.nan,raw[3]/raw[1] if raw[1] else np.nan,
        *ci,int(totals.Outcome.eq("late").sum()),totals.Lead_ms.median(),totals.Lead_ms.quantile(.05)]


def paired_comparisons(events):
    primary=events[events.Condition.isin(CONDITIONS) & events.TargetFAR.eq(.10)]
    rows=[]
    for block,frame in primary.groupby("Block"):
        index=["DesignID","Condition"]
        outcomes=frame.pivot(index=index,columns="Model",values="Outcome")
        actual=frame.drop_duplicates(index).set_index(index).Crossing_s.reindex(outcomes.index)
        risk=np.isfinite(actual.to_numpy())
        for model in ("quadratic","residual"):
            a,b=outcomes[model],outcomes.persistence
            values=pd.DataFrame(dict(hit=(a.eq("hit").astype(int)-b.eq("hit").astype(int)),
                false=(a.eq("false_alarm").astype(int)-b.eq("false_alarm").astype(int)),
                risk=risk,nonrisk=~risk)).groupby(level="DesignID").sum().to_numpy(float)
            draws=np.random.default_rng(39193022).integers(0,len(values),(3000,len(values)))
            raw=values.sum(axis=0)
            boot=values[draws].sum(axis=1)
            for name,i in (("Timely",0),("FAR",1)):
                available=boot[:,i+2]>0
                lo,hi=np.percentile(boot[available,i]/boot[available,i+2],[2.5,97.5]) if available.any() else (np.nan,np.nan)
                rows.append([block,model,name,raw[i]/raw[i+2] if raw[i+2] else np.nan,lo,hi])
    return pd.DataFrame(rows,columns=["Block","VersusPersistence","Metric","Difference","Low95","High95"])


def run():
    planned=pd.read_csv(SEALED/"frozen-design.csv")
    outcomes=pd.read_csv(SEALED/"design-results.csv")
    assert planned.equals(outcomes[planned.columns])
    design=outcomes.query("Status=='ok'").reset_index(drop=True)
    with np.load(SEALED/"trajectories.npz") as npz:
        signal,truth=npz["multi"],npz["coi"]
        assert np.array_equal(npz["ids"],design.DesignID)
    archived=loadmat(ROOT/"out/revision/trajectories.mat",simplify_cells=True)["grid"]
    dev=(archived["F"][:,50:]-50)/50
    features,_,labels,_=extract(dev)
    residual=[GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=42)
              .fit(x[1],np.maximum(x[0]-labels[:,i],0)) for i,x in enumerate(features)]
    models=train(dev)
    signals,_,quality=observe(signal,np.sqrt(np.mean(dev[:,:26]**2)),seed=39193020)
    crossing,nadir=truth_times(truth)
    cal=design.Block.eq("calibration").to_numpy()
    negatives=cal & np.isinf(crossing)
    pred=[]
    for v in signals:
        x,times,_,_=extract(v)
        values=predictions(models,x)
        values["residual"]=np.column_stack([k[0]-np.maximum(fit.predict(k[1]),0)
                                           for k,fit in zip(x,residual,strict=True)])
        pred.append(values)
    thresholds=[]
    frames=[]
    for method in METHODS:
        minima=np.column_stack([v[method].min(axis=1) for v in pred[:4]])
        for far in (.05,.10,.20):
            cuts=[]
            for tier in (False,True):
                sample=minima[(quality[:,:4]==tier)&negatives[:,None]]
                if not sample.size:
                    raise ValueError(f"Missing negative calibration data for tier {tier}")
                cuts.append(threshold(sample,far))
                thresholds.append([method,far,tier,len(sample),cuts[-1]])
            for j,condition in enumerate(CONDITIONS+STRESS):
                rows=design.loc[~cal,["DesignID","Block","LoadScale","Pattern"]].copy()
                cut=np.asarray(cuts)[quality[~cal,j].astype(int)]
                first,lead,outcome,early=evaluate(pred[j][method][~cal]-cut[:,None],np.arange(4),
                                                   0,times,crossing[~cal])
                rows=rows.assign(Model=method,TargetFAR=far,Condition=condition,Crossing_s=crossing[~cal],
                    Nadir_s=nadir[~cal],Alarm_s=first,Lead_ms=lead,Outcome=outcome,WindowMissed=early)
                frames.append(rows)
    events=pd.concat(frames,ignore_index=True)
    keys=["Block","Model","TargetFAR"]
    assert not events.duplicated(keys+["DesignID","Condition"]).any()
    assert (events.loc[events.Outcome.eq("hit"),"Alarm_s"]<events.loc[events.Outcome.eq("hit"),"Crossing_s"]).all()
    rows=[]
    for key,frame in events.groupby(keys):
        for name,g in [("mixed",frame[frame.Condition.isin(CONDITIONS)]),*frame.groupby("Condition")]:
            rows.append([*key,name,*report(g)])
    out=pd.DataFrame(rows,columns=keys+["Condition","PhysicalEvents","PhysicalRisk","RiskVariants",
        "NonriskVariants","Hits","FalseAlarms","Timely","FAR","TimelyLow","TimelyHigh","FARLow",
        "FARHigh","Late","LeadMedian_ms","LeadP5_ms"])
    out.to_csv(SEALED/"summary.csv",index=False)
    events.to_csv(SEALED/"events.csv",index=False)
    paired_comparisons(events).to_csv(SEALED/"paired.csv",index=False)
    pd.DataFrame(thresholds,columns=["Model","TargetFAR","LowQuality","NegativeVariants","Cut_pu"]).to_csv(
        SEALED/"thresholds.csv",index=False)
    print(out.query("Condition=='mixed' and TargetFAR==0.10").to_string(index=False))


if __name__=="__main__":
    run()
