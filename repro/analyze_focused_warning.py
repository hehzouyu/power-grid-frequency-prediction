"""Score the focused IEEE 39 case with frozen and noise-matched calibration."""
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat

from run_rolling_warning import (
    ROOT, OUT, evaluate, extract, make_row, predictions, threshold, train, truth_times,
)


def run():
    d=pd.read_csv(OUT/"ieee39focus-design.csv").query("Status=='ok'").reset_index(drop=True)
    protocol=json.loads((OUT/"protocol-results.json").read_text(encoding="utf-8"))
    archived=loadmat(ROOT/"out/revision/trajectories.mat",simplify_cells=True)["grid"]
    models=train((archived["F"][:,50:]-50)/50)
    test=d.LoadScale.isin([.965,1.015,1.055]).to_numpy()
    rows=[]
    events=[]
    with np.load(OUT/"ieee39focus-trajectories.npz") as data:
        crossing,nadir=truth_times(data["coi"])
        negatives=~test & ~np.isfinite(crossing)
        assert len(d)==len(crossing) and negatives.any()
        assert np.isfinite(crossing[test]).any() and (~np.isfinite(crossing[test])).any()
        for channel in ("coi","single","multi"):
            by_condition={}
            for condition,noise in (
                ("clean",0),("20dB",protocol["noise_rms_reference_Hz"]/500),
            ):
                x,t,_,_=extract(data[channel],noise=noise,seed=39191126)
                by_condition[condition]=predictions(models,x)
            for condition,score in by_condition.items():
                for model in ("persistence","quadratic"):
                    for far in (.05,.10,.20):
                        clean=threshold(by_condition["clean"][model][negatives].min(axis=1),far)
                        matched=threshold(score[model][negatives].min(axis=1),far)
                        cutoffs={"clean_local":clean}
                        if condition!="clean":
                            cutoffs["matched_local"]=matched
                        for mode,cutoff in cutoffs.items():
                            first,lead,outcome,early=evaluate(score[model][test],np.arange(4),cutoff,t,crossing[test])
                            result=pd.DataFrame({"Scenario":d.DesignID.to_numpy()[test],
                                "Crossing_s":crossing[test],"Nadir_s":nadir[test],"Alarm_s":first,
                                "Lead_ms":lead,"Outcome":outcome,"WindowMissed":early})
                            policy=model+"_"+mode
                            events.append(result.assign(Channel=channel,Condition=condition,Policy=policy,TargetFAR=far))
                            rows.append([far,*make_row(result,channel,condition,policy),int(negatives.sum()),cutoff])
    pd.DataFrame(rows,columns=["TargetFAR","Channel","Condition","Policy","N","Risk","Hits",
        "False","Late","Early","FAR","FAR_low","FAR_high","Miss","Miss_low","Miss_high",
        "LeadP5_ms","LeadMedian_ms","CalibrationNegatives","Cutoff_pu"]).to_csv(
            OUT/"focused-summary.csv",index=False)
    pd.concat(events,ignore_index=True).to_csv(OUT/"focused-events.csv",index=False)
    d.assign(Role=np.where(test,"test","calibration"),Risk=np.isfinite(crossing)).to_csv(
        OUT/"focused-scoring-index.csv",index=False)


if __name__=="__main__":
    run()
