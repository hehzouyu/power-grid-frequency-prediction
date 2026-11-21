"""Independently audit the frozen quality-gate run and draw evidence figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from nadir_quality import ROOT, NAMES, configuration, verify

OUT=ROOT/"confirm"
CONDS=["clean","white40","white30","white20","ar30","jump"]
CN=["干净","白噪声 40 dB","白噪声 30 dB","白噪声 20 dB","AR(1) 30 dB","扰动处突变"]
LABEL=dict(endpoint="端点",initial_rate="初始变化率",quadratic="二次拟合",
           fitted_segments="分段拟合",ecgr="ECGR",gate="质量门控")
COLOURS=dict(endpoint="#455972",initial_rate="#a1a9b4",quadratic="#3180a8",
             fitted_segments="#65845d",ecgr="#df9246",gate="#9a3f54")
plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"savefig.dpi":190,
                     "pdf.fonttype":42,"font.size":9})


def checked(a,b,atol=4e-9):
    np.testing.assert_allclose(a,b,rtol=0,atol=atol)


def ci(t,cluster,y,best,c):
    """Replicate the stratified bootstrap using sampled cluster sums."""
    ids=np.sort(t[cluster].unique())
    local=t.assign(e0=abs(t[best]-t[y])*1000,e1=abs(t.gate-t[y])*1000)
    by=local.groupby([cluster,"Condition"])
    idx=pd.MultiIndex.from_product([ids,CONDS],names=[cluster,"Condition"])
    total=by[["e0","e1"]].sum().reindex(idx,fill_value=0).to_numpy().reshape(len(ids),len(CONDS),2)
    n=by.size().reindex(idx,fill_value=0).to_numpy().reshape(len(ids),len(CONDS))
    rng=np.random.default_rng(c["bootstrap_seed"])
    groups=t[[cluster,"RBand"]].drop_duplicates()
    samples=np.concatenate([rng.choice(np.searchsorted(ids,b[cluster]),size=(c["bootstrap_draws"],len(b)))
                            for _,b in groups.groupby("RBand",sort=True)],axis=1)
    counts=n[samples].sum(axis=1)
    ratios=(total[samples].sum(axis=1)/counts[:,:,None]).mean(axis=1)
    return np.quantile(ratios[:,1]/ratios[:,0]-1,[.025,.975])


def audit(lock,c):
    points=pd.read_csv(OUT/"predictions.csv")
    pair=pd.read_csv(OUT/"pair-predictions.csv")
    results=pd.read_csv(OUT/"metrics.csv").set_index(["Kind","Condition","Feature"])
    controls=pd.read_csv(OUT/"main-contrasts.csv").set_index("Kind")
    oracle=pd.read_csv(OUT/"mis-selection.csv").set_index(["Kind","Condition"])
    observed=pd.read_csv(OUT/"visible-features.csv")
    references=pd.read_csv(OUT/"pair-index.csv").set_index("PairID")
    assert not pair.duplicated(["PairID","Condition","Rep"]).any()
    assert not points.duplicated(["Entity","Condition","Rep"]).any()
    checked(np.maximum(observed.PreQuality,observed.WindowQuality), observed.HybridQuality,1e-12)
    rule=lock["rule"]
    q=points[{"pre":"PreQuality","hybrid":"HybridQuality"}[rule["proxy"]]].to_numpy()
    expected=np.where(np.isfinite(q)&(q>=0)&(q<=rule["clean"]),"ecgr",
                    np.where(np.isfinite(q)&(q>=0)&(q<=rule["fallback"]),rule["middle"],"endpoint"))
    assert np.array_equal(points.Selected,expected)
    keyed=points.set_index(["Entity","Condition","Rep"])
    cols=NAMES+["gate"]
    a=keyed.loc[pd.MultiIndex.from_frame(pair[["A","Condition","Rep"]]),cols].to_numpy()
    b=keyed.loc[pd.MultiIndex.from_frame(pair[["B","Condition","Rep"]]),cols].to_numpy()
    checked(pair[cols],b-a,1e-12)
    checked(pair.DeltaTarget_Hz,references.loc[pair.PairID,"DeltaTarget_Hz"],1e-12)
    chosen=points[cols].to_numpy()[np.arange(len(points)),pd.Index(cols).get_indexer(expected)]
    checked(chosen,points.gate,1e-12)
    pool=[cols.index(n) for n in ["endpoint",rule["middle"],"ecgr"]]
    pair_y=pair.DeltaTarget_Hz.to_numpy()
    ol=abs(b[:,None,pool]-a[:,pool,None]-pair_y[:,None,None]).min(axis=(1,2))
    checked((abs(pair.gate-pair_y)-ol)*1000,pair.OracleLoss_mHz)
    py=points.Target_Hz.to_numpy()
    ql=abs(points[cols].to_numpy()[:,pool]-py[:,None]).min(axis=1)
    checked((abs(points.gate-py)-ql)*1000,points.OracleLoss_mHz)
    primary=pair[pair.DeltaTarget_Hz.abs().ge(.002)]
    assert set(primary.index)==set(pair.index[pair.Status.eq("primary_eligible")])
    single=points[points.Kind.eq("single")]
    best=lock["best_fixed"]
    for kind,frame,cl,y in [("single",single,"Entity","Target_Hz"),("pair",primary,"ReferenceID","DeltaTarget_Hz")]:
        for cond,p in frame.groupby("Condition"):
            allp=pair[pair.Condition.eq(cond)]
            for name in cols:
                er=abs(p[name]-p[y])*1000
                row=results.loc[(kind,cond,name)]
                checked([len(er),er.mean(),np.linalg.norm(er)/np.sqrt(len(er)),er.quantile(.95)],
                        row[["N","MAE_mHz","RMSE_mHz","P95_mHz"]])
                if kind=="pair":
                    d=np.sign(allp[name])*allp[name].abs().gt(.0005)
                    truth=np.sign(allp[y])*allp[y].abs().gt(.0005)
                    checked([len(allp),(d==truth).sum(),(d==0).sum(),(truth==0).sum()],
                        row[["DirectionN","DirectionHits","PredictedTies","TrueTies"]])
            z=p.OracleLoss_mHz
            checked([(z>c["oracle_tolerance_mhz"]).mean(),z.mean(),z.quantile(.95)],
                oracle.loc[(kind,cond),["MisselectRate","MeanOracleLoss_mHz","P95OracleLoss_mHz"]])
        lo,hi=ci(frame,cl,y,best,c)
        checked([lo,hi],controls.loc[kind,["Low95","High95"]])
        rev=results.loc[kind].MAE_mHz.groupby("Feature").mean()
        checked([rev[best],rev["gate"],rev["gate"]/rev[best]-1],
            controls.loc[kind,["BaselineMAE_mHz","GateMAE_mHz","RelativeChange"]])
    gates=pd.read_csv(OUT/"acceptance.csv")
    checks=[]
    for kind in ["single","pair"]:
        ar=controls.loc[kind]
        checks += [ar.RelativeChange<=-.02,ar.High95<0]
        for cond in CONDS:
            cmp={"clean":"quadratic","white30":"endpoint"}.get(cond,best)
            checks.append(results.loc[(kind,cond,"gate"),"P95_mHz"]<=1.05*results.loc[(kind,cond,cmp),"P95_mHz"])
        checks.append(results.loc[(kind,"white30","gate"),"MAE_mHz"]<=1.05*results.loc[(kind,"white30","endpoint"),"MAE_mHz"])
        anchor=results.loc[(kind,"clean","quadratic"),"MAE_mHz"]
        gain=max(anchor-results.loc[(kind,"clean","ecgr"),"MAE_mHz"],0)
        checks.append(results.loc[(kind,"clean","gate"),"MAE_mHz"]<=anchor-.8*gain+1e-10)
    assert gates.Pass.to_list()==checks
    print("PASS five fixed strategies, selected predictor, unique keys, tie denominators, oracle losses, cluster intervals and all frozen gates.")
    return single,pair,results,oracle


def save(fig,name):
    for ext in ["pdf","png"]:
        fig.savefig(OUT/f"{name}.{ext}")
    plt.close(fig)


def charts(lock,single,pair,metrics,oracle):
    fig,axes=plt.subplots(2,3,figsize=(11,5.5),layout="constrained")
    for i,kind in enumerate(["single","pair"]):
        for j,stat in enumerate(["MAE_mHz","RMSE_mHz","P95_mHz"]):
            ax=axes[i,j]
            for name in NAMES+["gate"]:
                x=metrics.loc[(kind,slice(None),name),stat].droplevel(["Kind","Feature"]).reindex(CONDS)
                ax.plot(range(6),x,marker="o" if name=="gate" else ".",color=COLOURS[name],
                        lw=2 if name=="gate" else 1,ls="-" if name=="gate" else "--",label=LABEL[name])
            ax.set(xticks=range(6),xticklabels=CN, ylabel=f"{'单场景' if kind=='single' else '配对差值'} / mHz",
                   title=stat.replace("_mHz",""))
            ax.tick_params(axis="x",rotation=26)
            ax.grid(axis="y",alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=6,loc="outside upper center",frameon=False)
    save(fig,"quality-strategy-errors")

    pool=["ecgr",lock["rule"]["middle"],"endpoint"]
    rule=lock["rule"]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    share=single.groupby(["Condition","Selected"]).size().unstack(fill_value=0).reindex(index=CONDS,columns=pool,fill_value=0)
    share=share.div(share.sum(axis=1),axis=0)
    bottom=np.zeros(len(share))
    for name in pool:
        axes[0].bar(CN,share[name],bottom=bottom,label=LABEL[name],color=COLOURS[name])
        bottom+=share[name].to_numpy()
    axes[0].set(ylabel="单场景选择占比",ylim=(0,1))
    axes[0].tick_params(axis="x",rotation=30)
    axes[0].legend(frameon=False,fontsize=8,loc="upper right")
    y=abs(single.gate-single.Target_Hz)*1000
    for name in pool:
        p=single.Selected.eq(name)
        axes[1].scatter(np.maximum(single.loc[p,"Quality"],1e-5),y[p],
                        s=7,color=COLOURS[name],alpha=.3,rasterized=True)
    for threshold in [rule["clean"],rule["fallback"]]:
        axes[1].axvline(threshold,color="#606060",ls=":",lw=1)
    axes[1].set(xlabel="可观测质量代理（对数轴）",ylabel="单场景绝对误差 / mHz",xscale="log")
    save(fig,"quality-selection-performance")

    fig,axs=plt.subplots(2,3,figsize=(9,5.5),layout="constrained")
    for cond,title,ax in zip(CONDS,CN,axs.ravel(),strict=True):
        x=single[single.Condition.eq(cond)]
        cm=pd.crosstab(x.Selected,x.Oracle).reindex(index=pool,columns=pool,fill_value=0)
        ax.imshow(cm.to_numpy()/len(x),vmin=0,vmax=1,cmap="Blues")
        for i,j in np.ndindex(cm.shape):
            n=cm.iloc[i,j]
            ax.text(j,i,n,ha="center",va="center",color="white" if n/len(x)>.48 else "#273b55")
        ax.set(xticks=range(3),xticklabels=[LABEL[n] for n in pool],yticks=range(3),
               yticklabels=[LABEL[n] for n in pool],xlabel="事后幅值最优",ylabel="门控选择",title=title)
    save(fig,"quality-misselection-matrix")

    fig,axs=plt.subplots(1,2,figsize=(9.5,3.2),layout="constrained")
    for ax,(field,title) in zip(axs,[("MisselectRate","误选比例"),("MeanOracleLoss_mHz","平均事后遗憾 / mHz")],strict=True):
        for i,kind in enumerate(["single","pair"]):
            values=oracle.loc[kind].reindex(CONDS)[field]
            ax.bar(np.arange(6)+(i-.5)*.32,values,width=.32,label="单场景" if not i else "主幅值配对",
                   color=["#3180a8","#9a3f54"][i])
        ax.set(xticks=np.arange(6),xticklabels=CN,ylabel=title)
        ax.tick_params(axis="x",rotation=30)
        ax.legend(frameon=False)
    save(fig,"quality-misselection-loss")


if __name__=="__main__":
    lock=verify()
    c=configuration()
    charts(lock,*audit(lock,c))
    print("PASS rendered four figures from confirmed outputs.")
