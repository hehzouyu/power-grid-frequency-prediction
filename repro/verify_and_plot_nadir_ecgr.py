"""Recheck fixed outputs and produce ECGR publication-quality diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nadir_ecgr import verify

ORDER = ["endpoint","initial_rate","quadratic","fitted_segments","ecgr"]
LABEL = ["端点","端点＋初始变化率","端点＋二次拟合","端点＋分段拟合","ECGR"]
COLORS = ["#657681","#7b8593","#28629a","#62736a","#bc502a"]
plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,
                     "pdf.fonttype":42,"savefig.dpi":200,"font.size":9})


def close(a,b,tol=1e-9):
    np.testing.assert_allclose(a,b,atol=tol,rtol=0)


def check_metrics(groups, table, error):
    for key, block in groups.groupby(["Condition","Feature"]):
        values=np.abs(block[error].to_numpy())*1000
        close(table.loc[key,["N","MAE_mHz","RMSE_mHz","P95_mHz"]],
              [len(values),np.mean(values),np.linalg.norm(values)/np.sqrt(len(values)),np.quantile(values,.95)])


def audit(out,p,s,e,g):
    c=json.loads((out/"method-protocol.json").read_text())
    q=c["confirmation"]
    pair=pd.read_csv(out/"pair-predictions.csv")
    raw=pd.read_csv(out/"member-predictions.csv")
    ledger=pd.read_csv(out/"pair-index.csv")
    assert not pair.duplicated(["PairID","Condition","Feature","Rep"]).any()
    values=raw.pivot(index=["PairID","Condition","Feature","Rep"],columns="Member",values="Prediction_Hz")
    idx=pd.MultiIndex.from_frame(pair[["PairID","Condition","Feature","Rep"]])
    close(values.loc[idx,2]-values.loc[idx,1],pair.DeltaPrediction_Hz,1e-12)
    target=ledger.set_index("PairID").loc[pair.PairID,"DeltaTarget_Hz"].to_numpy()
    close(target,pair.DeltaTarget_Hz,1e-12)
    pair["err_check"]=pair.DeltaPrediction_Hz-target
    close(pair.err_check,pair.DeltaError_Hz,1e-12)
    clean_main=pair[pair.DeltaTarget_Hz.abs()>=q["minimum_target_gap_hz"]]
    assert len(clean_main)==pair.Status.eq("primary_eligible").sum()
    check_metrics(clean_main,p,"err_check")
    for key,b in pair.groupby(["Condition","Feature"]):
        thresholds=[-q["tie_hz"],q["tie_hz"]]
        y=np.searchsorted(thresholds,b.DeltaTarget_Hz,side="right")
        yhat=np.searchsorted(thresholds,b.DeltaPrediction_Hz,side="right")
        # The upper boundary itself is still a tie.
        y[b.DeltaTarget_Hz==thresholds[1]]=1
        yhat[b.DeltaPrediction_Hz==thresholds[1]]=1
        close(p.loc[key,["DirectionTrials","CorrectDirections","PredictionTies","TrueTies"]],
              [len(b),np.count_nonzero(y==yhat),np.count_nonzero(yhat==1),np.count_nonzero(y==1)])
    raw["entity"]=np.where(raw.Member.eq(1),-raw.ReferenceID,raw.PairID)
    raw["err_check"]=raw.Prediction_Hz-raw.Target_Hz
    check_metrics(raw.drop_duplicates(["Condition","Feature","Rep","entity"]),s,"err_check")
    refs=pd.read_csv(out/"references.csv")
    rng=np.random.default_rng(q["bootstrap_seed"])
    draws=q["bootstrap_draws"]
    samples=np.concatenate([rng.choice(b.ReferenceID.to_numpy()-1,size=(draws,len(b)))
                            for _,b in refs.groupby("RBand")],axis=1)
    for (condition,base),effect in e.iterrows():
        b=clean_main[clean_main.Condition.eq(condition)].copy()
        b["abs_check"]=b.err_check.abs()*1000
        v=b.pivot(index=["ReferenceID","PairID","Rep"],columns="Feature",values="abs_check")
        d=(v.ecgr-v[base]).groupby(level=["ReferenceID","PairID"]).mean()
        dc=d.groupby(level=0)
        sums=dc.sum().reindex(refs.ReferenceID,fill_value=0).to_numpy()
        counts=dc.size().reindex(refs.ReferenceID,fill_value=0).to_numpy()
        den=counts[samples].sum(axis=1)
        boot=sums[samples].sum(axis=1)[den>0]/den[den>0]
        close(effect[["ChangeMAE_mHz","Low95_mHz","High95_mHz","Draws","ReferenceClusters"]],
              [d.mean(),*np.quantile(boot,[.025,.975]),len(boot),np.count_nonzero(counts)])
    clean_quad=e.loc[("clean","quadratic"),"High95_mHz"]
    checks=[clean_quad,p.loc[("clean","ecgr"),"MAE_mHz"]]
    cutoffs=[0,p.loc[("clean",ORDER[:-1]),"MAE_mHz"].min()]
    for table,condition,field,base in [(p,"clean","P95_mHz","quadratic"),
            (p,"white30","MAE_mHz","endpoint"),(p,"white30","P95_mHz","endpoint"),
            (s,"clean","P95_mHz","quadratic"),(s,"white30","MAE_mHz","endpoint"),
            (s,"white30","P95_mHz","endpoint")]:
        checks.append(table.loc[(condition,"ecgr"),field])
        cutoffs.append(1.05*table.loc[(condition,base),field])
    close(g.Value,checks)
    close(g.Threshold,cutoffs)
    assert g.Pass.to_list()==[a<b if i<2 else a<=b for i,(a,b) in enumerate(zip(checks,cutoffs,strict=True))]
    print("PASS read-back from 5-model member rows: point and pair metrics, directions, clustered bootstrap, and frozen gates.")


def figures(out,p,e):
    fig, axes=plt.subplots(2,2,figsize=(8.8,5.8),layout="constrained")
    for i,(key,name) in enumerate([("clean","干净"),("white30","30 dB 模拟白噪声")]):
        for j,(field,metric) in enumerate([("MAE_mHz","MAE"),("P95_mHz","P95")]):
            ax=axes[i,j]
            vals=p.loc[key].loc[ORDER,field]
            ax.barh(np.arange(5),vals,color=COLORS,height=.59)
            ax.set(yticks=np.arange(5),yticklabels=LABEL,xlabel=f"配对差值 {metric} / mHz",
                   title=name,ylim=(-.6,4.6))
            ax.set_xlim(0,vals.max()*1.2)
            ax.invert_yaxis()
            for k,v in enumerate(vals):
                ax.text(v+.015*vals.max(),k,f"{v:.2f}",va="center",fontsize=8)
            if i==j==1:
                ax.axvline(1.05*vals.iloc[0],color="#222",ls="--",lw=1,label="预设保护线")
                ax.legend(loc="lower right",frameon=False,fontsize=8)
    fig.savefig(out/"ecgr-pair-errors.pdf")
    fig.savefig(out/"ecgr-pair-errors.png")
    plt.close(fig)
    fig, ax=plt.subplots(figsize=(7.4,3.1),layout="constrained")
    for i,(cond,name,col) in enumerate([("clean","干净","#28629a"),("white30","30 dB 模拟白噪声","#bc502a")]):
        v=e.loc[cond].loc[ORDER[:-1]]
        d=v.ChangeMAE_mHz.to_numpy()
        ax.errorbar(d,np.arange(4)+(.13 if i else -.13),xerr=[d-v.Low95_mHz,v.High95_mHz-d],
                    fmt="o",capsize=3,color=col,label=name)
    ax.axvline(0,color="#555",ls="--",lw=1)
    ax.set(yticks=np.arange(4),yticklabels=LABEL[:-1],xlabel="ECGR 减对照的配对差值 MAE / mHz")
    ax.invert_yaxis()
    ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1.17),ncol=2)
    fig.savefig(out/"ecgr-paired-contrasts.pdf")
    fig.savefig(out/"ecgr-paired-contrasts.png")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    out=parser.parse_args().run.resolve()
    verify(out)
    p=pd.read_csv(out/"pair-results.csv",index_col=["Condition","Feature"])
    s=pd.read_csv(out/"single-results.csv",index_col=["Condition","Feature"])
    e=pd.read_csv(out/"paired-contrasts.csv",index_col=["Condition","Comparator"])
    gates=pd.read_csv(out/"acceptance.csv")
    audit(out,p,s,e,gates)
    figures(out,p,e)
    print("PASS rendered two result figures.")
