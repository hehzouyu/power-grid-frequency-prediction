"""Independent read-back of calibration, bounds, decisions and event-bootstrap CIs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nadir_risk import DEST, FIELDS, MODELS, c, verify

plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"pdf.fonttype":42,
                     "savefig.dpi":205,"font.size":9})
LABELS={"point":"点预测","symmetric90":"双侧区间下端","uniform":"统一单侧","scaled":"分尺度",
        "grouped":"分组","guarded_scale":"噪声包络"}
CN={"mixed":"同分布","noise_jump":"窗口内突变","ar20":"AR(1) 20 dB","inertia_high":"惯量外推",
    "cascade_fresh":"新两级调频","archive_single":"历史单级","archive_cascade":"历史两级"}
COLORS=["#888888","#8782a4","#316485","#6a884d","#d49444","#b33f54"]


def eq(x,y,eps=5e-10):
    np.testing.assert_allclose(x,y,rtol=0,atol=eps,equal_nan=True)


def compare_calibration(lock,p):
    dev=pd.read_csv(DEST/"scored-development.csv")
    eq(dev.Optimism_Hz,dev.Point_Hz-dev.Target_Hz,1e-12)
    design=dev[(dev.Role=="design")&(dev.Scenario=="mixed")]
    q=np.quantile(design.Quality_Hz,p["quality_cut_quantiles"])
    eq(q,lock["edges_Hz"])
    groups=np.searchsorted(q,dev.Quality_Hz,side="right")
    assert np.array_equal(groups,dev.Group)
    for i,s in enumerate(lock["scales_Hz"]):
        d=design[design.Group.eq(i)]
        eq(s,max(p["scale_floor_hz"],np.quantile(d.Optimism_Hz.clip(lower=0),p["scale_quantile"])))
    cal=dev[(dev.Role=="cal")&(dev.Scenario=="mixed")]
    def kth(s):
        n=len(s)
        k=int(np.ceil((n+1)*(1-p["alpha"])))-1
        return np.partition(s,k)[k]
    eq(kth(abs(cal.Optimism_Hz)),lock["symmetric90_radius_Hz"])
    eq(max(kth(cal.Optimism_Hz),0),lock["uniform_radius_Hz"])
    eq(max(kth(cal.Optimism_Hz/cal.Scale_Hz),0),lock["scaled_q"])
    eq([max(kth(x.Optimism_Hz),0) for _,x in cal.groupby("Group")],lock["grouped_Hz"])
    m=dev[(dev.Role=="cal")&dev.Scenario.isin(p["guard_calibration_scenarios"])]
    assert m.groupby("EventID").size().eq(3).all()
    event_score=(m.Optimism_Hz/m.Scale_Hz).groupby(m.EventID).max()
    eq(max(kth(event_score),0),lock["guarded_q"])


def audit(scored,summary,lock,p):
    k=["Scenario","EventID"]
    truth=pd.read_csv(DEST/"confirmation.csv").set_index(k)
    assert not scored.duplicated(k+["Method"]).any()
    joined=truth.loc[pd.MultiIndex.from_frame(scored[k])]
    eq(joined.Target_Hz,scored.Target_Hz,1e-12)
    eq(joined.Quality_Hz,scored.Quality_Hz,1e-12)
    assert scored.groupby(k).Method.nunique().eq(len(MODELS)).all()
    grp=scored.QualityGroup.to_numpy()
    assert np.array_equal(grp,np.searchsorted(lock["edges_Hz"],scored.Quality_Hz,side="right"))
    s=np.asarray(lock["scales_Hz"])[grp]
    u=np.select([scored.Method.eq(m) for m in MODELS],
        [0,lock["symmetric90_radius_Hz"],lock["uniform_radius_Hz"],s*lock["scaled_q"],
         np.asarray(lock["grouped_Hz"])[grp],s*lock["guarded_q"]])
    eq(u,scored.Margin_Hz)
    eq(scored.Point_Hz-u,scored.Lower_Hz,1e-12)
    y,t=scored.Target_Hz.to_numpy(),scored.Lower_Hz.to_numpy()
    threshold=p["illustrative_safety_threshold_hz"]
    for name,x in {"LowerMiss":y<t,"TrueSafe":y>=threshold,"DeclareSafe":t>=threshold,
                   "FalseSafe":(y<threshold)&(t>=threshold),"WithholdSafe":(y>=threshold)&(t<threshold)}.items():
        assert np.array_equal(scored[name],x)
    zeros=[]
    for (scenario,method),table in scored.groupby(["Scenario","Method"]):
        table=table.sort_values("EventID")
        row=summary.loc[(scenario,method)]
        safe=table.TrueSafe; d=table.DeclareSafe
        unsafe_count=int((~safe).sum())
        if int(table.FalseSafe.sum())==0 and unsafe_count:
            zeros.append(dict(Scenario=scenario,Method=method,UnsafeN=unsafe_count,
                IndependentEventUpper95=1-.05**(1/unsafe_count)))
        eq([len(table),unsafe_count,safe.sum(),d.sum(),table.LowerMiss.sum(),table.FalseSafe.sum(),table.WithholdSafe.sum()],
            row[["N","TrueUnsafe","TrulySafe","DeclareSafe","LowerMissCount","FalseSafeCount","WithholdSafeCount"]])
        def ratio(a,b):
            return a/b if b else np.nan
        vals=[table.LowerMiss.mean(),table.Margin_Hz.mean()*1000,table.Margin_Hz.quantile(.95)*1000,
            ratio(table.FalseSafe.sum(),unsafe_count),ratio(table.FalseSafe.sum(),d.sum()),
            ratio(table.WithholdSafe.sum(),safe.sum()),d.mean()]
        eq(vals,row[FIELDS])
        # Independently recompute stored intervals by indexing the saved event flags.
        base=scored[(scored.Scenario==scenario)&(scored.Method=="uniform")].sort_values("EventID")
        assert np.array_equal(base.EventID,table.EventID)
        n=len(table)
        ids=np.random.default_rng(p["bootstrap_seed"]+int(table.EventID.iloc[0])).integers(0,n,(p["bootstrap_n"],n))
        miss=table.LowerMiss.to_numpy()[ids].mean(axis=1)
        false=table.FalseSafe.to_numpy()[ids].sum(axis=1)
        withhold=table.WithholdSafe.to_numpy()[ids].sum(axis=1)
        truesafe=table.TrueSafe.to_numpy()[ids].sum(axis=1)
        declaration=table.DeclareSafe.to_numpy()[ids].sum(axis=1)
        with np.errstate(divide="ignore",invalid="ignore"):
            stat={"LowerMiss":miss,"MeanMargin_mHz":table.Margin_Hz.to_numpy()[ids].mean(axis=1)*1000,
                  "FalseSafeGivenUnsafe":false/(n-truesafe),"FalseSafeGivenDeclared":false/declaration,
                  "WithholdGivenSafe":withhold/truesafe,
                  "PairedChange":miss-base.LowerMiss.to_numpy()[ids].mean(axis=1)}
        for field,x in stat.items():
            good=x[np.isfinite(x)]
            eq(np.quantile(good,[.025,.975]) if len(good) else [np.nan,np.nan],
               row[[field+"_Low95",field+"_High95"]])
    pd.DataFrame(zeros).to_csv(DEST/"zero-false-safe-upper95.csv",index=False)
    gate=pd.read_csv(DEST/"acceptance.csv")
    methods=summary
    candidate=p["candidate"]
    chosen=lambda scen: methods.loc[(scen,candidate)]
    baseline=lambda scen: methods.loc[(scen,"uniform")]
    for r in gate.itertuples():
        if r.Scenario=="all_fresh":
            s=p["confirm_scenarios"]
            result=max(baseline(k).LowerMiss for k in s)-max(chosen(k).LowerMiss for k in s)
            eq(result,r.Value)
            assert r.Pass==(result>=p["min_worst_miss_reduction"])
            continue
        a,b=chosen(r.Scenario),baseline(r.Scenario)
        mappings={
            "empirical one-sided miss":(a.LowerMiss,p["max_empirical_miss"],True),
            "false-safe given unsafe":(a.FalseSafeGivenUnsafe,p["max_false_safe_among_unsafe"],a.TrueUnsafe>=p["minimum_class_count"]),
            "mean margin relative to uniform":(a.MeanMargin_mHz/b.MeanMargin_mHz,p["max_margin_ratio_vs_uniform"],True),
            "safe-withholding increase vs uniform":(a.WithholdGivenSafe-b.WithholdGivenSafe,
               p["max_safe_withholding_increase"],a.TrulySafe>=p["minimum_class_count"])}
        value,limit,enough=mappings[r.Criterion]
        eq([r.Value,r.Limit],[value,limit])
        assert r.EnoughCases==enough and r.Pass==(enough and value<=limit)
    joint=[]
    for method,b in scored[scored.Scenario.isin(p["guard_calibration_scenarios"])].groupby("Method"):
        wide=b.pivot(index="EventID",columns="Scenario",values="LowerMiss")
        assert wide.notna().all().all()
        hit=wide.any(axis=1).to_numpy()
        ids=np.random.default_rng(p["bootstrap_seed"]+1).integers(0,len(hit),(p["bootstrap_n"],len(hit)))
        q=np.quantile(hit[ids].mean(axis=1),[.025,.975])
        joint.append(dict(Method=method,N=len(hit),AnyMiss=hit.mean(),Low95=q[0],High95=q[1]))
    pd.DataFrame(joint).to_csv(DEST/"joint-noise-stress.csv",index=False)
    print("PASS independent score order, row joins, margins, threshold signs, event CIs and frozen decisions.")


def save(fig,name):
    for ext in ("pdf","png"):
        fig.savefig(DEST/f"{name}.{ext}")
    plt.close(fig)


def charts(summary,p):
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    plotted=MODELS[1:]
    loc=np.arange(len(CN))
    for i,method in enumerate(plotted,1):
        q=summary.xs(method,level="Method").loc[list(CN)]
        pos=loc+(i-3)*.11
        risk=q.LowerMiss.to_numpy()
        axes[0].errorbar(risk*100,pos,xerr=np.stack([risk-q.LowerMiss_Low95,q.LowerMiss_High95-risk])*100,
                         fmt="o",color=COLORS[i],capsize=2,label=LABELS[method])
        axes[1].scatter(q.MeanMargin_mHz,pos,color=COLORS[i],s=17)
    for ax,x in zip(axes,["下侧漏覆盖率 / %","平均安全裕度 / mHz"],strict=True):
        ax.set(yticks=loc,yticklabels=CN.values(),xlabel=x)
        ax.invert_yaxis()
        ax.grid(axis="x",alpha=.22)
    axes[0].axvline(p["alpha"]*100,color="#666",ls="--",lw=1)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,ncol=5,loc="outside upper center",frameon=False)
    save(fig,"risk-coverage-margin")
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout="constrained")
    for i,method in enumerate(("point","uniform","scaled","grouped","guarded_scale")):
        q=summary.xs(method,level="Method").loc[list(CN)]
        pos=loc+(i-2)*.14
        j=MODELS.index(method)
        for ax,field in zip(axes,("FalseSafeGivenUnsafe","WithholdGivenSafe"),strict=True):
            ax.barh(pos,q[field]*100,height=.12,label=LABELS[method],color=COLORS[j])
    for ax,desc in zip(axes,("真实低频场景中错误判安全 / %","真实安全场景中未认证 / %"),strict=True):
        ax.set(yticks=loc,yticklabels=CN.values(),xlabel=desc)
        ax.invert_yaxis()
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,ncol=5,loc="outside upper center",frameon=False)
    save(fig,"risk-certification-cost")


if __name__=="__main__":
    locked=verify()
    p=c()
    compare_calibration(locked,p)
    rows=pd.read_csv(DEST/"lower-bounds.csv")
    summary=pd.read_csv(DEST/"summary.csv").set_index(["Scenario","Method"])
    audit(rows,summary,locked,p)
    charts(summary,p)
    print("PASS figures and same-event simultaneous noise-stress diagnostic.")
