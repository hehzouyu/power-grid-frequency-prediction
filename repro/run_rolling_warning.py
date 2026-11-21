"""Reproduce rolling warning on aggregate trajectories and optional ANDES cases."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "rolling"
WINDOWS = np.array([10, 15, 20, 25])
DT = .02
THRESHOLD = -.005
METHODS = ("persistence", "endpoint", "quadratic", "segmented")


def simulate(params):
    h, d, r, tg, loss = params.T
    state = np.zeros((2, len(params)))
    result = np.zeros((len(params), 1001))
    def rhs(z):
        x, p = z
        return np.array([(-loss-d*x+p)/(2*h), (np.clip(-x/r,0,.1)-p)/tg])
    for j in range(1, 4001):
        a = rhs(state)
        b = rhs(state+.0025*a)
        c = rhs(state+.0025*b)
        e = rhs(state+.005*c)
        state += (.005/6)*(a+2*b+2*c+e)
        if j % 4 == 0:
            result[:,j//4] = state[0]
    return result


def extract(f, offset=0, noise=0, seed=0, latency=0):
    # Two prefix samples accommodate an early detected origin without wrapping indices.
    padded = np.pad(f, ((0, 0), (2, 0)), mode="edge")
    if noise:
        padded += np.random.default_rng(seed).normal(0,noise,padded.shape)
    output, emissions, labels, indices = [], [], [], []
    for width in WINDOWS:
        start = offset
        end = offset+width
        idx = np.arange(start,end+1)
        v = padded[:,idx+2]
        x = np.linspace(0,1,len(idx))
        poly = np.linalg.lstsq(np.column_stack([np.ones(len(x)),x,x*x]),v.T,rcond=None)[0]
        edges = np.rint(np.linspace(0,width,4)).astype(int)
        slopes = np.diff(v[:,edges],axis=1)/(DT*np.diff(edges))
        output.append((v[:,-1],np.column_stack([v[:,-1],poly[1:].T]),
                       np.column_stack([v[:,-1],slopes])))
        emissions.append(end*DT+latency)
        labels.append(f[:,end+1:].min(axis=1))
        indices.append((start,end))
    return output,np.array(emissions),np.column_stack(labels),indices


def train(f, truth=None):
    x, _, y, _ = extract(f)
    if truth is not None:
        y=np.column_stack([truth[:,end+1:].min(axis=1) for end in WINDOWS])
    return [
        [GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=42)
         .fit(np.asarray(v).reshape(len(f),-1),y[:,i]) for v in block]
        for i,block in enumerate(x)
    ]


def predictions(models, data):
    output={name:[] for name in METHODS}
    for model,block in zip(models,data,strict=True):
        output["persistence"].append(block[0])
        for name,fit,v in zip(METHODS[1:],model,block,strict=True):
            output[name].append(fit.predict(np.asarray(v).reshape(len(v),-1)))
    return {name:np.column_stack(v) for name,v in output.items()}


def truth_times(f):
    crossed=f<THRESHOLD
    return np.where(crossed.any(axis=1),crossed.argmax(axis=1)*DT,np.inf),f.argmin(axis=1)*DT


def threshold(scores, level):
    # Choose an event-level threshold so at most floor(level*n) negatives trigger.
    rank=int(np.floor(level*len(scores)))
    return np.sort(scores)[rank]


def finite_quantile(values, coverage=.90):
    k=min(len(values),int(np.ceil((len(values)+1)*coverage)))
    return np.sort(values)[k-1]


def policy_specs(pred, residual):
    return {
        "persistence_rolling": (pred["persistence"],np.arange(4)),
        "endpoint_rolling": (pred["endpoint"],np.arange(4)),
        "quadratic_rolling": (pred["quadratic"],np.arange(4)),
        "segmented_rolling": (pred["segmented"],np.arange(4)),
        "quadratic_fixed": (pred["quadratic"],np.array([3])),
        "quadratic_lower": (pred["quadratic"]-residual,np.arange(4)),
    }


def evaluate(values, cols, cutoff, times, crossing):
    alarms=values[:,cols]<cutoff
    first=np.where(alarms.any(axis=1),times[cols[alarms.argmax(axis=1)]],np.inf)
    risk=np.isfinite(crossing)
    hit=risk & (first<crossing)
    late=risk & np.isfinite(first) & ~hit
    early=risk & (crossing<=times[cols[0]])
    false=~risk & np.isfinite(first)
    label=np.select([hit,late,false,risk],["hit","late","false_alarm","no_alarm"],default="true_negative")
    lead=np.full(len(first),np.nan)
    lead[hit]=(crossing[hit]-first[hit])*1000
    return first,lead,label,early


def intervals(k,n):
    if not n:
        return [np.nan]*3
    ci=binomtest(k,n).proportion_ci(confidence_level=.95,method="wilson")
    return [k/n,ci.low,ci.high]


def make_row(frame,dataset,condition,policy):
    risk=np.isfinite(frame.Crossing_s)
    hit=frame.Outcome=="hit"
    false=frame.Outcome=="false_alarm"
    lo,mid=(np.percentile(frame.Lead_ms.dropna(),[5,50]) if hit.any() else (np.nan,np.nan))
    far,far_l,far_u=intervals(int(false.sum()),int((~risk).sum()))
    miss,miss_l,miss_u=intervals(int(risk.sum()-hit.sum()),int(risk.sum()))
    return [dataset,condition,policy,len(frame),risk.sum(),hit.sum(),false.sum(),
            (frame.Outcome=="late").sum(),frame.WindowMissed.sum(),far,far_l,far_u,
            miss,miss_l,miss_u,lo,mid]


def check_boundaries():
    f=np.array([[0,THRESHOLD,THRESHOLD-.001],[0,0,0]])
    cross,_=truth_times(f)
    assert cross[0]==.04 and np.isinf(cross[1])
    values=np.array([[-2,-2],[-2,-2],[1,1],[1,-2]])
    first,lead,out,early=evaluate(values,np.arange(2),0,np.array([.2,.3]),np.array([.20,np.inf,np.inf,.4]))
    assert first[0]==first[1]==.2 and list(out)==["late","false_alarm","true_negative","hit"]
    assert early[0] and np.isclose(lead[3],100)


def run():
    check_boundaries()
    OUT.mkdir(parents=True,exist_ok=True)
    old=loadmat(ROOT/"out"/"revision"/"trajectories.mat",simplify_cells=True)["grid"]
    observed=(old["F"][:,50:]-50)/50
    assert np.allclose(old["t"][50:]-1,np.arange(951)*DT)
    models=train(observed)
    development, _, dev_labels, _=extract(observed)
    dev_predictions=predictions(models,development)
    window_scale=np.maximum(np.mean(abs(dev_predictions["quadratic"]-dev_labels),axis=0),1e-9)
    # Independent seeds fixed before inspecting model outcomes.
    limits=np.array([[2,6],[.5,2],[.04,.06],[.2,1],[.02,.08]])
    p=np.concatenate([np.random.default_rng(seed).uniform(limits[:,0],limits[:,1],(n,5))
                      for seed,n in [(2026111901,600),(2026111902,600),(2026111903,1600)]])
    f=simulate(p)
    assert np.isfinite(f).all() and (f[:,0]==0).all()
    # Replay the archived grid to check the new integrator on the common time support.
    archived=simulate(np.asarray(old["params"]))
    comparison=np.max(np.abs(archived[:,:observed.shape[1]]-observed))*50
    assert comparison<.001, comparison
    np.savez_compressed(OUT/"aggregate-trajectories.npz",frequency_pu=1+f,params=p,dt=DT)
    iid=np.arange(len(f))
    roles=np.repeat(["interval_calibration","threshold_calibration","test"],[600,600,1600])
    cols=["H","D","R","Tg","deficit_pu"]
    index=pd.DataFrame(p,columns=cols).assign(Scenario=iid,Role=roles)
    index.to_csv(OUT/"scenarios.csv",index=False)
    cross,nadir=truth_times(f)
    noise_reference=np.sqrt(np.mean(observed[:,:26]**2))
    data, times, labels, bounds=extract(f)
    clean=predictions(models,data)
    interval=roles=="interval_calibration"
    cutset=roles=="threshold_calibration"
    test=roles=="test"
    score=(clean["quadratic"][interval]-labels[interval])/window_scale
    q=finite_quantile(np.max(score,axis=1))*window_scale
    specifications=policy_specs(clean,q)
    targets=(.01,.02,.03,.04,.05,.06,.075,.10,.15,.20)
    policy_cutoffs={
        (name,rate):threshold(values[cutset & ~np.isfinite(cross)][:,windows].min(axis=1),rate)
        for name,(values,windows) in specifications.items() for rate in targets
    }
    conditions=[("clean",0,0,0),("20dB",0,20,0),("30dB",0,30,0),("40dB",0,40,0),
                ("early20ms",-1,0,0),("late20ms",1,0,0),("early40ms",-2,0,0),
                ("late40ms",2,0,0),("noise20_late40_latency20",2,20,.02)]
    tables,metrics,point,observation=[],[],[],[]
    data_sets=[("aggregate",f[test],f[test],iid[test],models)]
    for filename in sorted(OUT.glob("ieee??-trajectories.npz")):
        with np.load(filename) as case:
            if len(case["coi"]) >= 2:
                tag=filename.name.split("-")[0]
                design=pd.read_csv(OUT/f"{tag}-design.csv")
                case_ids=design.loc[design.Status=="ok","DesignID"].to_numpy()
                assert len(case_ids)==len(case["coi"])
                for key in ["coi","single","multi"]:
                    data_sets.append((tag+"_"+key,case[key],case["coi"],case_ids,models))
    for dataset, signal, reference, ids, learned in data_sets:
        crossing,min_time=truth_times(reference)
        for condition,shift,snr,latency in conditions:
            if dataset!="aggregate" and condition not in ("clean","20dB"):
                continue
            obs, times, y, bounds=extract(signal,shift,noise_reference/10**(snr/20) if snr else 0,
                                           20261119,latency)
            y=np.column_stack([reference[:,end+1:].min(axis=1) for _,end in bounds])
            pred=predictions(learned,obs)
            obs_rows=pd.DataFrame({
                "Dataset":dataset,"Condition":condition,"Scenario":np.repeat(ids,4),
                "Start_sample":np.tile([x[0] for x in bounds],len(ids)),
                "End_sample":np.tile([x[1] for x in bounds],len(ids)),
                "Emitted_s":np.tile(times,len(ids)),"Target_Hz_at_50":50*(1+y.ravel()),
                **{name+"_Hz_at_50":50*(1+value.ravel()) for name,value in pred.items()}})
            observation.append(obs_rows)
            for method, values in pred.items():
                for i,w in enumerate(WINDOWS):
                    err=(values[:,i]-y[:,i])*50
                    point.append([dataset,condition,method,w*DT,np.sqrt(np.mean(err**2)),
                                  np.mean(abs(err)),np.percentile(abs(err),95)])
            for policy,(values,windows) in policy_specs(pred,q).items():
                for rate in targets if condition=="clean" and dataset=="aggregate" else (.05,):
                    first,lead,outcome,early=evaluate(values,windows,policy_cutoffs[policy,rate],times,crossing)
                    frame=pd.DataFrame({"Dataset":dataset,"Condition":condition,"Policy":policy,
                        "TargetFAR":rate,"Scenario":ids,"Crossing_s":crossing,"Nadir_s":min_time,
                        "Alarm_s":first,"Lead_ms":lead,"Outcome":outcome,"WindowMissed":early})
                    tables.append(frame)
                    metrics.append([rate,*make_row(frame,dataset,condition,policy)])
    events=pd.concat(tables,ignore_index=True)
    events.to_csv(OUT/"events.csv",index=False)
    pd.concat(observation,ignore_index=True).to_csv(OUT/"observations.csv",index=False)
    summary=pd.DataFrame(metrics,columns=["TargetFAR","Dataset","Condition","Policy","N","Risk",
        "Hits","False","Late","Early","FAR","FAR_low","FAR_high","Miss","Miss_low","Miss_high",
        "LeadP5_ms","LeadMedian_ms"])
    summary.to_csv(OUT/"warning-summary.csv",index=False)
    pd.DataFrame(point,columns=["Dataset","Condition","Model","Window_s","RMSE_Hz_at_50",
        "MAE_Hz_at_50","P95_Hz_at_50"]).to_csv(OUT/"point-summary.csv",index=False)
    metadata={"relative_threshold":THRESHOLD,"old_grid_replay_max_abs_error_Hz":comparison,
        "noise_rms_reference_Hz":50*noise_reference,"joint_interval_correction_Hz":(50*q).tolist(),
        "joint_test_coverage":float(np.mean(np.all(labels[test]>=clean["quadratic"][test]-q,axis=1))),
        "cutoffs_pu":{f"{k[0]}_{k[1]}":v for k,v in policy_cutoffs.items()},
        "seeds":[2026111901,2026111902,2026111903],"scenario_draws":[600,600,1600]}
    (OUT/"protocol-results.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    plot(summary,events)
    print(summary.query("Dataset=='aggregate' and Condition=='clean' and TargetFAR==.05").to_string(index=False))


def plot(summary,events):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    clean=summary.query("Dataset=='aggregate' and Condition=='clean'")
    fig,ax=plt.subplots(figsize=(6,3.3),layout="constrained")
    selected=("persistence_rolling","endpoint_rolling","quadratic_rolling","quadratic_fixed","quadratic_lower")
    for name,mark in zip(selected,("o","s","^","P","v"),strict=True):
        sub=clean[clean.Policy==name]
        ax.plot(sub.FAR*100,sub.Miss*100,marker=mark,label=name.replace("_"," "))
    ax.set(xlabel="False alarms / %",ylabel="Missed timely warnings / %")
    ax.legend(fontsize=8,frameon=False)
    fig.savefig(OUT/"tradeoff.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.3),layout="constrained")
    for name in selected:
        sub=events.query("Dataset=='aggregate' and Condition=='clean' and TargetFAR==.05 and Policy==@name")
        lead=sub.Lead_ms.dropna().sort_values()
        ax.step(lead,np.arange(1,len(lead)+1)/len(lead),where="post",label=name.replace("_"," "))
    ax.set(xlabel="Lead time among timely warnings / ms",ylabel="Empirical cumulative probability")
    ax.legend(fontsize=8,frameon=False)
    fig.savefig(OUT/"lead.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.3),layout="constrained")
    for name,mark in zip(selected[:3],("o","s","^"),strict=True):
        sub=summary.query("Dataset=='aggregate' and TargetFAR==.05 and Policy==@name").set_index("Condition")
        sub=sub.loc[["clean","40dB","30dB","20dB","early40ms","late40ms","noise20_late40_latency20"]]
        ax.plot(range(len(sub)),sub.Miss*100,marker=mark,label=name.replace("_"," "))
    ax.set_xticks(range(7),["Clean","40 dB","30 dB","20 dB","−40 ms","+40 ms","Combined"],rotation=18)
    ax.set(ylabel="Missed timely warnings / %")
    ax.legend(fontsize=8,frameon=False)
    fig.savefig(OUT/"stress.pdf"); plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
