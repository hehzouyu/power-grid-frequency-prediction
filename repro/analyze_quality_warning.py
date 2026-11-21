"""Benchmark event-level warning calibrated from a pre-event noise estimate."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import lfilter

from run_rolling_warning import ROOT, OUT, evaluate, extract, predictions, simulate, threshold, train, truth_times

CONDITIONS=("clean","white40","white20","correlated20")
STRESS=("white30","noise_jump")
METHODS=("persistence","quadratic")


NOISE_IMPLEMENTATION_BASE_SEED=39192514
"""R2 多噪声实现种子基值：实现 k 的观测噪声种子 = 基值 + k，k=0 即 R1 位级锚点。"""


def observe(f,reference,seed=NOISE_IMPLEMENTATION_BASE_SEED):
    """Use the same event and noise seed across methods, with observed pre-event samples.

    ``seed`` 只选择观测噪声的实现（R2 协议 §3.1：``seed_k = 基值 + k``）；
    默认值等于 R1 使用的固定种子，故不传参时输出与改动前逐位一致。
    """
    rng=np.random.default_rng(seed)
    noise=rng.normal(size=(len(f),f.shape[1]+50))
    noise_ar=rng.normal(size=noise.shape)
    correlated,_=lfilter([.6],[1,-.8],noise_ar,axis=1,zi=rng.normal(size=(len(f),1))*.8)
    jump=np.concatenate([reference/100*noise[:,:50],reference/10*noise[:,50:]],axis=1)
    deviations=(np.zeros_like(noise),reference/100*noise,reference/10*noise,reference/10*correlated,
                reference/np.sqrt(1000)*noise,jump)
    boundary=reference/np.sqrt(1000)
    estimates=np.stack([v[:,:50].std(axis=1,ddof=1) for v in deviations],axis=1)
    quality=estimates>=boundary
    signals=[f+v[:,50:] for v in deviations]
    return signals,estimates,quality


def summary(frame,draws):
    risk=np.isfinite(frame.Crossing_s.to_numpy())
    hits=frame.Outcome.to_numpy()=="hit"
    false=frame.Outcome.to_numpy()=="false_alarm"
    lead=frame.Lead_ms.dropna()
    matrix=np.stack([risk,hits,~risk,false],axis=1)
    bins=pd.DataFrame(matrix,columns=["risk","hit","nonrisk","false"]).assign(ID=frame.Scenario.to_numpy())
    cluster=bins.groupby("ID",sort=True).sum().to_numpy(dtype=float)
    sampled=cluster[draws].sum(axis=1)
    rates=np.divide(sampled[:,[1,3]],sampled[:,[0,2]],
                    out=np.full((len(sampled),2),np.nan),where=sampled[:,[0,2]]>0)
    ci=np.nanpercentile(rates,[2.5,97.5],axis=0)
    return dict(N=len(frame),PhysicalEvents=len(cluster),Risk=int(risk.sum()),Hits=int(hits.sum()),
        FalseAlarms=int(false.sum()),Late=int((frame.Outcome=="late").sum()),
        Timely=hits.sum()/risk.sum(),TimelyLow=ci[0,0],TimelyHigh=ci[1,0],
        FAR=false.sum()/(~risk).sum(),FARLow=ci[0,1],FARHigh=ci[1,1],
        LeadMedian_ms=lead.median(),LeadP5_ms=lead.quantile(.05))


def compare_paired(events):
    rows=[]
    base=events[events.Policy=="fixed"].set_index(["Dataset","Model","TargetFAR","Scenario","Condition"])
    graded=events[events.Policy=="estimated"].set_index(base.index.names)
    assert base.index.is_unique and graded.index.is_unique
    base=base.reindex(graded.index)
    for scope,group in graded.groupby(level=["Dataset","Model","TargetFAR"]):
        baseline=base.loc[group.index]
        risk=np.isfinite(group.Crossing_s.to_numpy())
        change=(group.Outcome=="hit").astype(int)-(baseline.Outcome=="hit").astype(int)
        false=(group.Outcome=="false_alarm").astype(int)-(baseline.Outcome=="false_alarm").astype(int)
        agg=pd.DataFrame({"hit":change.to_numpy(),"false":false.to_numpy(),
                          "risk":risk,"nonrisk":~risk,"id":group.index.get_level_values("Scenario")})
        values=agg.groupby("id").sum().to_numpy(float)
        rng=np.random.default_rng(39192517)
        boot=values[rng.integers(0,len(values),(3000,len(values)))].sum(axis=1)
        delta=np.divide(boot[:,:2],boot[:,2:],out=np.full_like(boot[:,:2],np.nan),where=boot[:,2:]>0)
        lo,hi=np.nanpercentile(delta,[2.5,97.5],axis=0)
        estimate=values.sum(axis=0)
        rows.append([*scope,estimate[0]/estimate[2],lo[0],hi[0],estimate[1]/estimate[3],lo[1],hi[1]])
    return pd.DataFrame(rows,columns=["Dataset","Model","TargetFAR","DeltaTimely","TimelyLow","TimelyHigh",
                                        "DeltaFAR","FARLow","FARHigh"])


def score_dataset(name,signal,truth,ids,cal,models,reference):
    crossing,nadir=truth_times(truth)
    negatives=cal & ~np.isfinite(crossing)
    assert negatives.any() and np.isfinite(crossing[~cal]).any() and (~cal & ~np.isfinite(crossing)).any()
    signals,estimates,estimated=observe(signal,reference)
    oracle=np.broadcast_to([False,False,True,True,True,True],estimated.shape)
    results={method:[] for method in METHODS}
    for s in signals:
        inputs,times,_,_=extract(s)
        values=predictions(models,inputs)
        for method in METHODS:
            results[method].append(values[method])
    conditions=CONDITIONS+STRESS
    annotation=pd.DataFrame({"Dataset":name,"Scenario":np.repeat(ids,len(conditions)),
        "Role":np.repeat(np.where(cal,"calibration","test"),len(conditions)),
        "Condition":np.tile(conditions,len(ids)), "SigmaEstimate_pu":estimates.ravel(),
        "EstimatedLow":estimated.ravel(),"TrueLow":oracle.ravel()})
    frames=[]
    cuts=[]
    for method in METHODS:
        values=np.stack(results[method],axis=1)
        minima=values.min(axis=2)
        for far in (.05,.10,.20):
            for policy,labels in (("fixed",np.zeros_like(estimated)),("estimated",estimated),("oracle",oracle)):
                found=np.unique(labels)
                cutoff=np.full_like(minima,np.nan)
                for level in found:
                    region=labels==level
                    primary=np.arange(len(conditions))<len(CONDITIONS)
                    sample=minima[region & negatives[:,None] & primary]
                    assert len(sample)>0,(name,policy,level)
                    bound=threshold(sample,far)
                    cutoff[region]=bound
                    cuts.append([name,method,far,policy,bool(level),int(len(sample)),bound])
                for j,cond in enumerate(conditions):
                    use=~cal
                    first,lead,outcome,early=evaluate(values[use,j,:]-cutoff[use,j,None],np.arange(4),
                                                      0,times,crossing[use])
                    frames.append(pd.DataFrame({"Dataset":name,"Scenario":ids[use],"Condition":cond,
                        "Model":method,"TargetFAR":far,"Policy":policy,"Crossing_s":crossing[use],
                        "Nadir_s":nadir[use],"Alarm_s":first,"Lead_ms":lead,"Outcome":outcome,"Early":early}))
    return pd.concat(frames,ignore_index=True),annotation,cuts


def run(aggregate_only):
    protocol=json.loads((OUT/"protocol-results.json").read_text(encoding="utf-8"))
    old=loadmat(ROOT/"out/revision/trajectories.mat",simplify_cells=True)["grid"]
    models=train((old["F"][:,50:]-50)/50)
    limits=np.array([[2,6],[.5,2],[.04,.06],[.2,1],[.02,.08]])
    params=np.vstack([np.random.default_rng(seed).uniform(limits[:,0],limits[:,1],(n,5))
                      for seed,n in [(39192512,600),(39192513,1800)]])
    f=simulate(params)
    ids=np.arange(len(params))
    calibration=ids<600
    pd.DataFrame(params,columns=["H","D","R","Tg","d"]).assign(
        Scenario=ids,Role=np.where(calibration,"calibration","test")).to_csv(OUT/"quality-aggregate-index.csv",index=False)
    samples=[("aggregate",f,f,ids,calibration)]
    if not aggregate_only:
        design=pd.read_csv(OUT/"ieee39quality-design.csv").query("Status=='ok'")
        with np.load(OUT/"ieee39quality-trajectories.npz") as case:
            ids=design.DesignID.to_numpy()
            cal=design.LoadScale.isin([.950,.975,1.010,1.040]).to_numpy()
            assert len(ids)==len(case["coi"])
            samples.append(("ieee39_multi",case["multi"],case["coi"],ids,cal))
    packets=[score_dataset(*s,models,protocol["noise_rms_reference_Hz"]/50) for s in samples]
    ev=pd.concat([x[0] for x in packets],ignore_index=True)
    estimates=pd.concat([x[1] for x in packets],ignore_index=True)
    cuts=[r for x in packets for r in x[2]]
    rows=[]
    keys=["Dataset","Model","TargetFAR","Policy"]
    for key,group in ev.groupby(keys,sort=False):
        n=group.Scenario.nunique()
        draws=np.random.default_rng(39192515).integers(0,n,(2500,n))
        for condition,block in [("mixed",group[group.Condition.isin(CONDITIONS)]),
                                *group.groupby("Condition",sort=False)]:
            rows.append(dict(zip(keys,key,strict=True))|{"Condition":condition}|summary(block,draws))
    results=pd.DataFrame(rows)
    ev.to_csv(OUT/"quality-events.csv",index=False)
    estimates.to_csv(OUT/"quality-estimates.csv",index=False)
    results.to_csv(OUT/"quality-summary.csv",index=False)
    compare_paired(ev[ev.Condition.isin(CONDITIONS)]).to_csv(OUT/"quality-paired.csv",index=False)
    pd.DataFrame(cuts,columns=[*keys,"LowQuality","CalibrationVariants","Threshold_pu"]).to_csv(
        OUT/"quality-thresholds.csv",index=False)
    graph(results)


def graph(results):
    datasets=list(results.Dataset.unique())
    fig,axs=plt.subplots(1,len(datasets),figsize=(4.15*len(datasets),3.35),squeeze=False,layout="constrained")
    for name,ax in zip(datasets,axs[0],strict=True):
        sub=results.query("Dataset==@name and Condition=='mixed'")
        for (model,policy),part in sub.groupby(["Model","Policy"]):
            if policy=="oracle":
                continue
            part=part.sort_values("TargetFAR")
            ax.plot(100*part.FAR,100*part.Timely,marker="o" if policy=="fixed" else "^",
                    linestyle="--" if model=="persistence" else "-",label=f"{model} / {policy}")
        ax.set(title=name.replace("_"," "),xlabel="Observed false-alarm rate (%)",
               ylabel="Timely hit rate (%)")
        ax.legend(fontsize=7,frameon=False)
    fig.savefig(OUT/"quality-tradeoff.pdf")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only",action="store_true")
    run(parser.parse_args().aggregate_only)
