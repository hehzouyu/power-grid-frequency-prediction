"""Independently reconstruct event labels and compare policies by event cluster."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D=Path(__file__).resolve().parent
C=json.loads((D/"protocol.json").read_text(encoding="utf-8-sig"))
T=C["primary_far_target"]
N=C["test_events"]
COND=C["conditions"]
SETS={"mixed":C["calibration_conditions"]}|{cond:[cond] for cond in COND}
TIME=np.asarray(C["windows"])*.02
POLICIES={
    "persistence":"persistence", "endpoint":"endpoint","quadratic":"quadratic",
    "segmented":"segmented","ecgr":"ecgr","gate":"gate","gate_final_only":"gate_final_only",
    "quadratic_fixed":"quadratic","quadratic_stratified":"quadratic","gate_stratified":"gate",
}
COMPARE=[("gate","quadratic"),("gate","endpoint"),("gate","ecgr"),("gate","gate_final_only"),
         ("gate_stratified","quadratic_stratified")]

def read(name):
    return pd.read_csv(D/name,float_precision="round_trip")

def masks(row):
    risk=row.Crossing_s.notna()&np.isfinite(row.Crossing_s)
    alerted=np.isfinite(row.Alarm_s)
    hit=risk&(row.Alarm_s<row.Crossing_s)
    false=~risk&alerted
    return risk,alerted,hit,false

def check_scores(scores,pred):
    cuts=read("cutoffs.csv").set_index(["Policy","TargetFAR","Group"])
    # Replay the confirmation parser representation as used in production.
    # Use the round-trip version above for exact independent calibration ranks.
    operational=pd.read_csv(D/"cutoffs.csv").set_index(["Policy","TargetFAR","Group"])
    scenes=read("test-scenarios.csv").sort_values("Event")
    cal=read("calibration-window-predictions.csv.gz")
    eligible=cal[np.isinf(cal.Crossing_s)&cal.Condition.isin(C["calibration_conditions"])]
    assert len(scores)==N*len(COND)*len(POLICIES)*len(C["far_targets"])
    assert not scores.duplicated(["Event","Condition","TargetFAR","Policy"]).any()
    for (policy,target,cond),row in scores.groupby(["Policy","TargetFAR","Condition"]):
        row=row.sort_values("Event")
        w=pred[pred.Condition.eq(cond)].sort_values(["Event","Window_s"])
        p=w.pivot(index="Event",columns="Window_s",values=POLICIES[policy]).to_numpy()
        first=w[w.Window_s.eq(TIME[0])].sort_values("Event")
        np.testing.assert_array_equal(row.Event,first.Event)
        np.testing.assert_array_equal(row.Crossing_s,scenes.Crossing_s)
        cols=np.array([3]) if policy=="quadratic_fixed" else np.arange(4)
        tab=operational.loc[policy,target]
        expected=np.where(first.LowPreQuality,tab.loc["low_quality","Cutoff_Hz"],tab.loc["high_quality","Cutoff_Hz"]) \
            if policy.endswith("stratified") else np.full(N,tab.loc["all","Cutoff_Hz"])
        np.testing.assert_array_equal(row.Cutoff_Hz,expected)
        times=np.where(p[:,cols]<expected[:,None],TIME[cols],np.inf)
        np.testing.assert_allclose(row.Alarm_s,times.min(axis=1),rtol=0,atol=1e-14)
        risk,alert,hit,false=masks(row)
        actual=np.select([hit,risk&alert,false,risk],["hit","late","false_alarm","no_alarm"],default="true_negative")
        np.testing.assert_array_equal(row.Outcome,actual)
        np.testing.assert_allclose(row.Lead_ms[hit],(row.Crossing_s[hit]-row.Alarm_s[hit])*1000,atol=1e-9)
        assert row.Lead_ms[~hit].isna().all()
    for (policy,target,group),row in cuts.iterrows():
        cal_block=eligible
        if group!="all":
            cal_block=cal_block[cal_block.LowPreQuality.eq(group=="low_quality")]
        if policy=="quadratic_fixed":
            cal_block=cal_block[cal_block.Window_s.eq(.50)]
        per_event=cal_block.groupby(["Condition","Event"])[POLICIES[policy]].min()
        assert len(per_event)==int(row.NegativeVariants)
        assert (per_event<row.Cutoff_Hz).sum()==int(row.Triggered)
        np.testing.assert_allclose(np.sort(per_event)[int(np.floor(target*len(per_event)))],row.Cutoff_Hz,atol=1e-12)

def event_vectors(scores):
    for scope,conds in SETS.items():
        for (target,policy),r in scores[scores.Condition.isin(conds)].groupby(["TargetFAR","Policy"]):
            risk,alert,hit,false=masks(r)
            lead=r.loc[hit,"Lead_ms"]
            per_event=pd.DataFrame(dict(Event=r.Event,risk=risk,hit=hit,neg=~risk,false=false)).groupby("Event").sum()
            v=per_event.reindex(np.arange(N)).to_numpy(int)
            assert np.isfinite(v).all()
            yield (scope,float(target),policy),v,{"EventCount":N,"RiskVariants":int(risk.sum()),
                "NonriskVariants":int((~risk).sum()),"Hits":int(hit.sum()),"FalseAlarms":int(false.sum()),
                "Late":int(r.Outcome.eq("late").sum()),"NoAlarmRisk":int(r.Outcome.eq("no_alarm").sum()),
                "LeadP5_ms":float(lead.quantile(.05)),"LeadMedian_ms":float(lead.median()),
                "HitRate":float(hit.sum()/risk.sum()),"FAR":float(false.sum()/(~risk).sum())}

def contrasts(vectors):
    rng=np.random.default_rng(C["bootstrap_seed"])
    draws=rng.integers(0,N,(C["bootstrap_draws"],N),dtype=np.int32)
    results=[]
    for scope in SETS:
        for target in C["far_targets"]:
            for a,b in COMPARE:
                x,y=vectors[scope,target,a],vectors[scope,target,b]
                assert np.array_equal(x[:,[0,2]],y[:,[0,2]])
                d=np.column_stack((x[:,0],x[:,1]-y[:,1],x[:,2],x[:,3]-y[:,3]))
                def rates(m): return m[..., [1,3]]/m[..., [0,2]]
                h,f=rates(d.sum(axis=0))
                boot=np.column_stack([d[draws,j].sum(axis=1) for j in range(4)])
                lo,hi=np.quantile(rates(boot),[.025,.975],axis=0)
                results.append(dict(Scope=scope,TargetFAR=target,Policy=a,Comparator=b,
                     DeltaHit_pp=h*100,HitLow95_pp=lo[0]*100,HitHigh95_pp=hi[0]*100,
                     DeltaFAR_pp=f*100,FARLow95_pp=lo[1]*100,FARHigh95_pp=hi[1]*100))
    return pd.DataFrame(results)

def benchmark(d):
    cmp=C["primary_comparison"]
    matching=d[d.TargetFAR.eq(T) & d.Policy.eq(cmp[0]) & d.Comparator.eq(cmp[1])].set_index("Scope")
    main,jump=matching.loc["mixed"],matching.loc["jump"]
    a=C["acceptance"]
    return pd.DataFrame(
        [("mixed point hit gain",main.DeltaHit_pp,"≥",a["min_hit_rate_gain"]*100,main.DeltaHit_pp>=a["min_hit_rate_gain"]*100),
         ("mixed hit gain lower CI",main.HitLow95_pp,">",0,main.HitLow95_pp>0),
         ("mixed FAR increase upper CI",main.FARHigh95_pp,"≤",a["max_far_increase_ci_upper"]*100,main.FARHigh95_pp<=a["max_far_increase_ci_upper"]*100),
         ("jump FAR increase upper CI",jump.FARHigh95_pp,"≤",a["max_jump_far_increase_ci_upper"]*100,jump.FARHigh95_pp<=a["max_jump_far_increase_ci_upper"]*100)],
        columns=["Criterion","Value_pp","Operator","Threshold_pp","Pass"])

def chart(summary, effects, choices):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"pdf.fonttype":42})
    names={"gate":"ECGR门控","quadratic":"二次拟合","endpoint":"端点","ecgr":"固定ECGR",
           "gate_final_only":"仅末窗门控","mixed":"五条件等权","clean":"干净",
           "white40":"白噪声40 dB","white30":"白噪声30 dB","white20":"白噪声20 dB","ar30":"AR(1) 30 dB","jump":"噪声突变"}
    palette=["#126878","#9b4d30","#70824a","#72528b","#806743"]
    view=summary[summary.TargetFAR.eq(T)]
    fig,ax=plt.subplots(figsize=(7,4.5),layout="constrained")
    for (method,color) in zip(("gate","quadratic","endpoint","ecgr","gate_final_only"),palette):
        b=view[view.Policy.eq(method)].set_index("Scope")
        ax.scatter(b.loc["mixed","FAR"]*100,b.loc["mixed","HitRate"]*100,s=50,c=color,label=names[method])
    ax.axvline(T*100,color=".6",ls="--",lw=.9)
    ax.set(xlabel="实际虚警率 / %",ylabel="及时命中率 / %",title="同一校准目标下的实测工作点")
    ax.legend(frameon=False,fontsize=9)
    fig.savefig(D/"operating-points.pdf")
    plt.close(fig)
    b=effects.query("Policy=='gate' and Comparator=='quadratic' and TargetFAR==@T").set_index("Scope").loc[list(SETS)]
    fig,ax=plt.subplots(1,2,figsize=(9,4.9),sharey=True,layout="constrained")
    for panel,title,field,ab in zip(ax,["及时命中率变化","实际虚警率变化"],["DeltaHit_pp","DeltaFAR_pp"],["Hit","FAR"]):
        yy=np.arange(len(b))
        value=b[field].to_numpy()
        panel.axvline(0,c=".6",ls="--",lw=.9)
        panel.errorbar(value,yy,xerr=[value-b[f"{ab}Low95_pp"],b[f"{ab}High95_pp"]-value],fmt="o",c=palette[0],capsize=3)
        panel.set(yticks=yy,yticklabels=[names[k] for k in b.index],xlabel="门控减二次拟合 / 百分点",title=title)
    ax[0].invert_yaxis()
    fig.savefig(D/"paired-event-effects.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9,5.4),sharex=True,sharey=True,layout="constrained")
    for panel,cond in zip(axes.ravel(),COND):
        b=choices[choices.Condition.eq(cond)].pivot(index="Window_s",columns="Selected",values="Fraction").fillna(0)
        columns=[c for c in ("ecgr","quadratic","endpoint") if c in b]
        panel.stackplot(b.index,*[b[c]*100 for c in columns],labels=[names[c] for c in columns],
                        colors=[palette[("ecgr","quadratic","endpoint").index(c)] for c in columns])
        panel.set(title=names[cond],xticks=TIME)
    handles=[]
    from matplotlib.patches import Patch
    for i,n in enumerate(("ecgr","quadratic","endpoint")):
        handles.append(Patch(color=palette[i],label=names[n]))
    fig.legend(handles=handles,loc="outside upper center",ncol=3,frameon=False)
    fig.supxlabel("决策窗 / s")
    fig.supylabel("选择比例 / %")
    fig.savefig(D/"selection-by-window.pdf")
    plt.close(fig)

def main():
    p=read("test-window-predictions.csv.gz")
    scores=read("test-event-scores.csv.gz")
    check_scores(scores,p)
    print("PASS independent event replay, labels, first-trigger timing, calibration thresholds and all denominators")
    records=list(event_vectors(scores))
    summary=pd.DataFrame([dict(Scope=k[0],TargetFAR=k[1],Policy=k[2])|r for k,_,r in records])
    effects=contrasts({k:v for k,v,_ in records})
    choice=(p.groupby(["Condition","Window_s","Selected"]).size().rename("Count").reset_index())
    choice["Fraction"]=choice.Count/N
    px=[]
    for name in ("quadratic","ecgr","gate","endpoint"):
        for (condition,window),frame in p.groupby(["Condition","Window_s"]):
            e=np.abs(frame[name]-frame.Target_Hz)*1000
            px.append(dict(Policy=name,Condition=condition,Window_s=window,MAE_mHz=e.mean(),P95_mHz=e.quantile(.95)))
    pd.DataFrame(px).to_csv(D/"point-prediction-metrics.csv",index=False)
    summary.to_csv(D/"event-metrics.csv",index=False)
    effects.to_csv(D/"paired-effects.csv",index=False)
    choice.to_csv(D/"selection-by-window.csv",index=False)
    accept=benchmark(effects)
    accept.to_csv(D/"acceptance.csv",index=False)
    chart(summary,effects,choice)
    print(accept.to_string(index=False))

if __name__=="__main__":
    main()
