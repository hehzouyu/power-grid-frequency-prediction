"""Evaluate frozen transfer and local threshold calibration on distributed steps."""
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat

from run_rolling_warning import (
    ROOT, OUT, evaluate, extract, make_row, predictions, threshold, train, truth_times,
)


def run():
    protocol=json.loads((OUT/"protocol-results.json").read_text(encoding="utf-8"))
    old=loadmat(ROOT/"out/revision/trajectories.mat",simplify_cells=True)["grid"]
    models=train((old["F"][:,50:]-50)/50)
    rows=[]
    events=[]
    for case in ("ieee14dist","ieee39dist"):
        design=pd.read_csv(OUT/f"{case}-design.csv").query("Status=='ok'").reset_index(drop=True)
        with np.load(OUT/f"{case}-trajectories.npz") as data:
            cross,nadir=truth_times(data["coi"])
            calibration=np.isclose(design.LoadScale,1)
            test=~calibration
            negatives=calibration & ~np.isfinite(cross)
            assert negatives.sum()>0 and test.sum()>0
            for channel in ("coi","single","multi"):
                obs,t,_,_=extract(data[channel])
                clean=predictions(models,obs)
                noisy,_,_,_=extract(data[channel],noise=protocol["noise_rms_reference_Hz"]/500,
                                    seed=20261119)
                by_condition={"clean":clean,"20dB":predictions(models,noisy)}
                for feat in ("persistence","quadratic"):
                    global_cut=protocol["cutoffs_pu"][f"{feat}_rolling_0.05"]
                    local_cut=threshold(clean[feat][negatives].min(axis=1),.05)
                    for condition,by_feature in by_condition.items():
                        scores=by_feature[feat]
                        for mode,cut in (("aggregate",global_cut),("local",local_cut)):
                            first,lead,outcome,early=evaluate(scores[test],np.arange(4),cut,t,cross[test])
                            frame=pd.DataFrame({"Scenario":design.DesignID.to_numpy()[test],
                                "Crossing_s":cross[test],"Nadir_s":nadir[test],"Alarm_s":first,
                                "Lead_ms":lead,"Outcome":outcome,"WindowMissed":early})
                            dataset=case+"_"+channel
                            policy=feat+"_"+mode
                            rows.append([*make_row(frame,dataset,condition,policy),int(negatives.sum()),cut])
                            events.append(frame.assign(Dataset=dataset,Condition=condition,Policy=policy))
    pd.DataFrame(rows,columns=["Dataset","Condition","Policy","N","Risk","Hits","False","Late",
        "Early","FAR","FAR_low","FAR_high","Miss","Miss_low","Miss_high","LeadP5_ms",
        "LeadMedian_ms","CalibrationNegatives","Cutoff_pu"]).to_csv(OUT/"distributed-summary.csv",index=False)
    pd.concat(events,ignore_index=True).to_csv(OUT/"distributed-events.csv",index=False)


if __name__=="__main__":
    run()
