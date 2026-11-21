"""Exploratory comparison on the already inspected IEEE 39-bus benchmark."""
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import GradientBoostingRegressor

from analyze_quality_warning import CONDITIONS, observe
from run_rolling_warning import (
    ROOT, OUT, evaluate, extract, make_row, predictions, train, truth_times, threshold,
)


def run():
    old=loadmat(ROOT/"out/revision/trajectories.mat",simplify_cells=True)["grid"]
    development=(old["F"][:,50:]-50)/50
    reference=np.sqrt(np.mean(development[:,:26]**2))
    models=train(development)
    features,_,labels,_=extract(development)
    residual=[GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=42)
              .fit(x[1],np.maximum(x[0]-labels[:,i],0)) for i,x in enumerate(features)]
    design=pd.read_csv(OUT/"ieee39quality-design.csv").query("Status=='ok'")
    with np.load(OUT/"ieee39quality-trajectories.npz") as data:
        signal,truth=data["multi"],data["coi"]
    assert len(design)==len(signal)==len(truth)
    cal=design.LoadScale.isin([.950,.975,1.010,1.040]).to_numpy()
    ids=design.DesignID.to_numpy()
    crossing,nadir=truth_times(truth)
    signals,_,quality=observe(signal,reference)
    conditions=CONDITIONS+("white30","noise_jump")
    scores={}; predicted={}
    diagnostics=[]
    for j,(label,observed) in enumerate(zip(conditions,signals,strict=True)):
        data,times,_,bounds=extract(observed)
        result=predictions(models,data)
        result["residual"]=np.column_stack([x[0]-np.maximum(fit.predict(x[1]),0)
                                           for x,fit in zip(data,residual,strict=True)])
        actual=np.column_stack([truth[:,end+1:].min(axis=1) for _,end in bounds])
        for method in ("persistence","quadratic","residual"):
            record=result[method]
            diagnostics.append(pd.DataFrame({"Scenario":np.repeat(ids[~cal],4),"Condition":label,
                 "Model":method,"Decision_s":np.tile(times,(~cal).sum()),
                 "Error_Hz_equivalent":50*(record[~cal]-actual[~cal]).ravel(),
                 "Risk":np.repeat(np.isfinite(crossing[~cal]),4)}))
            predicted[method,j]=record
            scores[method,j]=record.min(axis=1)
    events=[]; rows=[]
    neg=cal & ~np.isfinite(crossing)
    for method in ("persistence","quadratic","residual"):
        for far in (.05,.10,.20):
            calibration=np.column_stack([scores[method,j] for j in range(4)])
            cuts=[threshold(calibration[neg[:,None] & (quality[:,:4]==level)],far)
                  for level in (False,True)]
            for j,label in enumerate(conditions):
                cut=np.asarray(cuts)[quality[~cal,j].astype(int)]
                alarms,lead,outcome,early=evaluate(predicted[method,j][~cal]-cut[:,None],
                                                   np.arange(4),0,times,crossing[~cal])
                frame=pd.DataFrame({"Scenario":ids[~cal],"Crossing_s":crossing[~cal],
                    "Nadir_s":nadir[~cal],"Alarm_s":alarms,"Lead_ms":lead,"Outcome":outcome,
                    "WindowMissed":early,"Condition":label,"Model":method,"TargetFAR":far})
                events.append(frame)
    all_events=pd.concat(events,ignore_index=True)
    for (method,far),group in all_events.groupby(["Model","TargetFAR"]):
        for label,part in [("mixed",group[group.Condition.isin(CONDITIONS)]),*group.groupby("Condition")]:
            rows.append([far,*make_row(part,"ieee39_multi",label,method)])
    columns=["TargetFAR","Dataset","Condition","Model","N","Risk","Hits","False","Late","Early",
             "FAR","FARLow","FARHigh","Miss","MissLow","MissHigh","LeadP5_ms","LeadMedian_ms"]
    summary=pd.DataFrame(rows,columns=columns)
    # Mixed variants share their underlying events; discard the IID intervals.
    summary.loc[summary.Condition=="mixed",["FARLow","FARHigh","MissLow","MissHigh"]]=np.nan
    all_events.to_csv(OUT/"residual-pilot-events.csv",index=False)
    summary.to_csv(OUT/"residual-pilot-summary.csv",index=False)
    pd.concat(diagnostics,ignore_index=True).to_csv(OUT/"residual-pilot-errors.csv",index=False)
    hits=all_events.query("TargetFAR==0.10 and Condition in @CONDITIONS").pivot(
        index=["Scenario","Condition"],columns="Model",values="Outcome")
    risk=all_events.query("TargetFAR==0.10").drop_duplicates(["Scenario","Condition"]).set_index(
        ["Scenario","Condition"]).Crossing_s
    hits=hits[np.isfinite(risk.reindex(hits.index))]
    print(pd.crosstab(hits.persistence=="hit",hits.quadratic=="hit"))
    print(summary.query("TargetFAR==0.10 and Condition in ['mixed','noise_jump']")[
        ["Condition","Model","Hits","Risk","False","FAR","LeadMedian_ms"]].to_string(index=False))


if __name__=="__main__":
    run()
