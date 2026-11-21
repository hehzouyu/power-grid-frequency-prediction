"""Freeze one-sided conformal lower bounds and evaluate fixed-cutoff risk."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from analyze_revision import OUT, fit_model, loadmat, predict
from nadir_ecgr import verify as verify_ecgr

HERE=Path(__file__).resolve().parent
DEST=OUT.parent/"nadir-risk"
OLD_ECGR=OUT.parent/"nadir-ecgr"/"confirmation-v1"
COLUMNS=["Endpoint","Linear","Curvature"]
PARAM=["H","D","R","Tg","d"]
MODELS=["point","symmetric90","uniform","scaled","grouped","guarded_scale"]


def c():
    return json.loads((DEST/"protocol.json").read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_hashes():
    paths=[HERE/name for name in ("nadir_risk.py","nadir_risk.m","nadir_risk_protocol.json","analyze_revision.py",
                                  "run_nadir_risk.ps1")]
    paths += [DEST/name for name in ("protocol.json","development.csv")]
    paths += [OUT/name for name in ("features.mat","grid-index.csv","predictions.csv","continuous-index.csv",
                                   "cascade-index.csv","trajectories.mat")]
    paths += [HERE.parent/"Project"/name for name in ("core/simulate_freq.m","core/swing_equation.m","core/pfr_model.m",
                                     "config/sys_config.m","config/sim_config.m","utils/calc_nadir.m")]
    return {str(p.relative_to(HERE.parent)):digest(p) for p in paths}


def fixed_model():
    verify_ecgr(OLD_ECGR)
    idx=pd.read_csv(OUT/"grid-index.csv")
    x=loadmat(OUT/"features.mat",simplify_cells=True)["X"][2]
    train,valid,test=[np.flatnonzero(idx.Split.eq(k)) for k in ("train","validation","test")]
    m=fit_model(x,idx.Target_Hz.to_numpy(),train,valid,"tree")
    old=pd.read_csv(OUT/"predictions.csv").query("Experiment=='conditional' and Model=='tree' and Feature=='quadratic'")
    ref=old.set_index("ID").loc[idx.ID.iloc[test],"Prediction_Hz"]
    np.testing.assert_allclose(predict(m,x[test]),ref,rtol=0,atol=1e-10)
    return m


def order_quantile(values,alpha):
    """Exact split-conformal order statistic; never interpolate."""
    s=np.sort(np.asarray(values,dtype=float))
    assert len(s)>0 and np.isfinite(s).all()
    rank=math.ceil((len(s)+1)*(1-alpha))
    if rank>len(s):
        raise ValueError("No finite quantile: calibration count does not support nominal level.")
    return float(s[rank-1])


def group(x,edges):
    return np.searchsorted(edges,np.asarray(x),side="right")


def score_data(d,m):
    assert not d.duplicated(["Role","Scenario","EventID"]).any()
    assert d.SuffixSamples.eq(925).all() and np.isfinite(d[COLUMNS+["Quality_Hz","Target_Hz"]]).all().all()
    assert d.Quality_Hz.ge(0).all()
    d=d.assign(Point_Hz=predict(m,d[COLUMNS].to_numpy()))
    return d.assign(Optimism_Hz=d.Point_Hz-d.Target_Hz)


def confirm_integrity(d,dev,p):
    confirm=d[d.Role.eq("final")]
    archive=d[d.Role.eq("archive")]
    expected={name:p["final_n"] for name in p["confirm_scenarios"]}
    expected["inertia_high"]=p["extrapolation_n"]
    assert confirm.groupby("Scenario").size().to_dict()==expected
    assert archive.groupby("Scenario").size().to_dict()=={"archive_cascade":500,"archive_single":500}
    assert confirm.query("Scenario=='inertia_high'").H.between(6.25,7.5).all()
    first=confirm[confirm.Scenario.ne("inertia_high")]
    assert first.groupby("EventID")[PARAM].nunique().eq(1).all().all()
    joins=first.pivot(index="EventID",columns="Scenario",values="Target_Hz")
    assert joins[["mixed","noise_jump","ar20"]].nunique(axis=1).eq(1).all()
    assert archive.groupby("EventID")[PARAM].nunique().eq(1).all().all()
    old=[dev[PARAM].to_numpy(),pd.read_csv(OUT/"grid-index.csv")[PARAM].to_numpy(),
         pd.read_csv(OUT/"continuous-index.csv")[PARAM].to_numpy(),
         pd.read_csv(OUT.parent/"nadir-intervals"/"scenarios.csv")[["H","D","R","Tg","Deficit"]].to_numpy()]
    for location in (OLD_ECGR,OUT.parent/"nadir-pair-tests"/"independent-v1"):
        q=pd.read_csv(location/"pair-index.csv")
        old += [q[PARAM].to_numpy(),q[["CandidateH","CandidateD","CandidateR","CandidateTg","CandidateD_pu"]].dropna().to_numpy()]
    for stage in ("dev","confirm"):
        old.append(pd.read_csv(OUT.parent/"nadir-quality"/stage/"entity-index.csv")[PARAM].to_numpy())
    unique=confirm[["EventID",*PARAM]].drop_duplicates()
    assert unique.EventID.is_unique and len(unique)==p["final_n"]+p["extrapolation_n"]
    assert len(unique[PARAM].drop_duplicates())==len(unique)
    assert (cKDTree(np.vstack(old)).query(unique[PARAM].to_numpy(),p=np.inf)[0]>1e-9).all()


def freeze():
    assert not (DEST/"freeze.json").exists() and not (DEST/"confirmation.csv").exists()
    assert digest(DEST/"protocol.json")==digest(HERE/"nadir_risk_protocol.json")
    p=c()
    d=score_data(pd.read_csv(DEST/"development.csv"),fixed_model())
    assert d.groupby("EventID").Role.nunique().eq(1).all()
    assert d.groupby("EventID")[PARAM].nunique().eq(1).all().all()
    assert d[d.Scenario.eq("mixed")][PARAM].drop_duplicates().shape[0]==p["design_n"]+p["calibration_n"]
    design=d.query("Role=='design' and Scenario=='mixed'").copy()
    cal=d.query("Role=='cal' and Scenario=='mixed'").copy()
    assert (len(design),len(cal))==(p["design_n"],p["calibration_n"])
    edges=np.quantile(design.Quality_Hz,p["quality_cut_quantiles"])
    design["Group"]=group(design.Quality_Hz,edges)
    scales=design.groupby("Group").Optimism_Hz.apply(
        lambda x:np.quantile(np.maximum(x,0),p["scale_quantile"])).clip(lower=p["scale_floor_hz"]).to_numpy()
    assert len(scales)==3 and np.all(np.diff(edges)>0)
    d["Group"]=group(d.Quality_Hz,edges)
    d["Scale_Hz"]=scales[d.Group]
    cal=d.query("Role=='cal' and Scenario=='mixed'")
    r=cal.Optimism_Hz
    alpha=p["alpha"]
    stress=d[(d.Role=="cal") & d.Scenario.isin(p["guard_calibration_scenarios"])].copy()
    assert stress.groupby("EventID").Scenario.nunique().eq(len(p["guard_calibration_scenarios"])).all()
    scores=(stress.Optimism_Hz/stress.Scale_Hz).groupby(stress.EventID).max()
    assert len(scores)==len(cal)
    locked=dict(hashes=source_hashes(),edges_Hz=edges.tolist(),scales_Hz=scales.tolist(),
        symmetric90_radius_Hz=order_quantile(abs(r),alpha),uniform_radius_Hz=max(0,order_quantile(r,alpha)),
        scaled_q=max(0,order_quantile(r/cal.Scale_Hz,alpha)),
        grouped_Hz=[max(0,order_quantile(b.Optimism_Hz,alpha)) for _,b in cal.groupby("Group")],
        grouped_sizes=cal.groupby("Group").size().tolist(),
        guarded_q=max(0,order_quantile(scores,alpha)),calibration_events=len(cal))
    assert len(locked["grouped_Hz"])==3
    d.to_csv(DEST/"scored-development.csv",index=False)
    (DEST/"freeze.json").write_text(json.dumps(locked,indent=2)+"\n")
    print("PASS historical predictor and disjoint development/calibration. Saved coefficients before confirmation.")
    print({key:value for key,value in locked.items() if key!="hashes"})


def verify():
    lock=json.loads((DEST/"freeze.json").read_text())
    assert lock["hashes"]==source_hashes()
    return lock


def evaluate_bounds(data,p,lock):
    grp=group(data.Quality_Hz,lock["edges_Hz"])
    scale=np.asarray(lock["scales_Hz"])[grp]
    margins=[0,lock["symmetric90_radius_Hz"],lock["uniform_radius_Hz"],scale*lock["scaled_q"],
             np.asarray(lock["grouped_Hz"])[grp],scale*lock["guarded_q"]]
    bound=[]
    for method,margin in zip(MODELS,margins,strict=True):
        q=data.assign(Method=method,QualityGroup=grp,Margin_Hz=margin,Lower_Hz=data.Point_Hz-margin)
        q["LowerMiss"]=q.Target_Hz<q.Lower_Hz
        q["TrueSafe"]=q.Target_Hz>=p["illustrative_safety_threshold_hz"]
        q["DeclareSafe"]=q.Lower_Hz>=p["illustrative_safety_threshold_hz"]
        q["FalseSafe"]=~q.TrueSafe & q.DeclareSafe
        q["WithholdSafe"]=q.TrueSafe & ~q.DeclareSafe
        # A true-unsafe case is 'safe' only if it also missed the lower bound.
        assert q.loc[q.FalseSafe,"LowerMiss"].all()
        bound.append(q)
    return pd.concat(bound,ignore_index=True)


def event_metrics(a):
    margin,miss,safe,declare=a.T
    fs=(~safe.astype(bool)&declare.astype(bool))
    ws=safe.astype(bool)&~declare.astype(bool)
    with np.errstate(divide="ignore",invalid="ignore"):
        return np.array([miss.mean(),margin.mean()*1000,np.quantile(margin,.95)*1000,
            fs.sum()/np.count_nonzero(~safe.astype(bool)),fs.sum()/np.count_nonzero(declare),
            ws.sum()/np.count_nonzero(safe),declare.mean()])


FIELDS=["LowerMiss","MeanMargin_mHz","P95Margin_mHz","FalseSafeGivenUnsafe","FalseSafeGivenDeclared",
        "WithholdGivenSafe","DeclareRate"]


def summarize(results,p):
    rows=[]
    for scenario,b in results.groupby("Scenario",sort=False):
        method_tables={name:v.sort_values("EventID") for name,v in b.groupby("Method",sort=False)}
        ids=method_tables["uniform"].EventID.to_numpy()
        assert all(np.array_equal(ids,ss.EventID) for ss in method_tables.values())
        # Scenarios sharing physical events use the same resampling index.
        rng=np.random.default_rng(p["bootstrap_seed"]+int(ids[0]))
        samples=rng.integers(0,len(ids),(p["bootstrap_n"],len(ids)))
        matrices={name:v[["Margin_Hz","LowerMiss","TrueSafe","DeclareSafe"]].to_numpy(dtype=float)
                  for name,v in method_tables.items()}
        intervals={}
        for method,matrix in matrices.items():
            v=matrix[samples]
            miss=v[:,:,1].mean(axis=1)
            safety=v[:,:,2].astype(bool)
            declare=v[:,:,3].astype(bool)
            with np.errstate(divide="ignore",invalid="ignore"):
                fs=(~safety & declare).sum(axis=1)
                bs=~safety
                intervals[method]=np.column_stack([miss,v[:,:,0].mean(axis=1)*1000,
                    fs/bs.sum(axis=1),fs/declare.sum(axis=1),
                    (safety & ~declare).sum(axis=1)/safety.sum(axis=1)])
        for method,mt in method_tables.items():
            summary=dict(zip(FIELDS,event_metrics(matrices[method]),strict=True))
            for j,field in enumerate(("LowerMiss","MeanMargin_mHz","FalseSafeGivenUnsafe","FalseSafeGivenDeclared","WithholdGivenSafe")):
                finite=intervals[method][:,j]
                good=finite[np.isfinite(finite)]
                lo,hi=np.quantile(good,[.025,.975]) if len(good) else (np.nan,np.nan)
                summary[field+"_Low95"]=lo; summary[field+"_High95"]=hi
            comparison=intervals[method][:,0]-intervals["uniform"][:,0]
            lo,hi=np.quantile(comparison,[.025,.975])
            base=method_tables["uniform"].LowerMiss.mean()
            summary.update(PairedMissChangeVsUniform=summary["LowerMiss"]-base,
                           PairedChange_Low95=lo,PairedChange_High95=hi)
            rows.append(dict(Scenario=scenario,Method=method,N=len(mt),TrueUnsafe=int((~mt.TrueSafe).sum()),
                TrulySafe=int(mt.TrueSafe.sum()),DeclareSafe=int(mt.DeclareSafe.sum()),
                LowerMissCount=int(mt.LowerMiss.sum()),FalseSafeCount=int(mt.FalseSafe.sum()),
                WithholdSafeCount=int(mt.WithholdSafe.sum()))|summary)
    return pd.DataFrame(rows)


def acceptance(summary,p):
    summary=summary.set_index(["Scenario","Method"])
    rows=[]
    candidate=p["candidate"]
    for scenario in p["confirm_scenarios"]:
        x=summary.loc[(scenario,candidate)]
        u=summary.loc[(scenario,"uniform")]
        rules=[
            ("empirical one-sided miss",x.LowerMiss,p["max_empirical_miss"],True),
            ("false-safe given unsafe",x.FalseSafeGivenUnsafe,p["max_false_safe_among_unsafe"],
             x.TrueUnsafe>=p["minimum_class_count"]),
            ("mean margin relative to uniform",x.MeanMargin_mHz/u.MeanMargin_mHz,
             p["max_margin_ratio_vs_uniform"],True),
            ("safe-withholding increase vs uniform",x.WithholdGivenSafe-u.WithholdGivenSafe,
             p["max_safe_withholding_increase"],x.TrulySafe>=p["minimum_class_count"])]
        for name,value,limit,enough in rules:
            rows.append(dict(Scenario=scenario,Criterion=name,Value=value,Limit=limit,
                             EnoughCases=bool(enough),Pass=bool(enough and value<=limit)))
    w=pd.IndexSlice
    guard=summary.loc[w[p["confirm_scenarios"],candidate],"LowerMiss"].max()
    uniform=summary.loc[w[p["confirm_scenarios"],"uniform"],"LowerMiss"].max()
    req=p["min_worst_miss_reduction"]
    rows.append(dict(Scenario="all_fresh",Criterion="worst-case miss reduction",Value=uniform-guard,
                     Limit=req,EnoughCases=True,Pass=bool(uniform-guard>=req)))
    return pd.DataFrame(rows)


def confirm():
    lock=verify()
    assert not (DEST/"lower-bounds.csv").exists(),"Preserve earlier assessment."
    p=c(); data=pd.read_csv(DEST/"confirmation.csv"); dev=pd.read_csv(DEST/"development.csv")
    confirm_integrity(data,dev,p)
    scored=score_data(data,fixed_model())
    results=evaluate_bounds(scored,p,lock)
    summary=summarize(results,p)
    gate=acceptance(summary,p)
    results.to_csv(DEST/"lower-bounds.csv",index=False)
    summary.to_csv(DEST/"summary.csv",index=False)
    gate.to_csv(DEST/"acceptance.csv",index=False)
    bad=results[results.Method.eq(p["candidate"]) & (results.FalseSafe | results.LowerMiss | results.WithholdSafe)].copy()
    bad["OverOptimism_mHz"]=(bad.Lower_Hz-bad.Target_Hz)*1000
    bad.sort_values(["Scenario","OverOptimism_mHz"],ascending=[True,False]).to_csv(DEST/"case-ledger.csv",index=False)
    print("PASS historical predictor, distinct formal events, fixed cutoff and paired archives.")
    print("Gate:", "PASS" if gate.Pass.all() else "FAIL", int(gate.Pass.sum()),"of",len(gate))
    print(gate.loc[~gate.Pass].to_string(index=False))


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("stage",choices=["freeze","verify","confirm"])
    task=a.parse_args().stage
    {"freeze":freeze,"verify":verify,"confirm":confirm}[task]()
