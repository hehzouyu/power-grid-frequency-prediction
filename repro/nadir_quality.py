"""Develop and freeze a causal, threshold-based quality selector."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from analyze_revision import predict
from nadir_ecgr import (COLS, OUT, SLOTS, baseline_models, checkpoint, load_dev,
                        new_model, verify as verify_ecgr)

HERE = Path(__file__).resolve().parent
ROOT = OUT.parent/"nadir-quality"
ECGR = OUT.parent/"nadir-ecgr"/"confirmation-v1"
NAMES = list(SLOTS)+["ecgr"]
KEY = ["Entity","Condition","Rep"]
PARAM = ["H","D","R","Tg","d"]
CPARAM = ["CandidateH","CandidateD","CandidateR","CandidateTg","CandidateD_pu"]
SOURCES = ("nadir_quality.py","nadir_quality.m","prepare_nadir_quality.py","nadir_quality_protocol.json",
           "nadir_pair_test.m","nadir_pair_protocol.json","nadir_ecgr.py","nadir_ecgr.m","analyze_revision.py")


def configuration():
    return json.loads((HERE/"nadir_quality_protocol.json").read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def hashes():
    paths = [HERE/name for name in SOURCES]+[ECGR/"freeze.json", ECGR/"development-features.csv",
        OUT/"grid-index.csv",OUT/"features.mat",OUT/"trajectories.mat",OUT/"predictions.csv",
        OUT/"continuous-index.csv",OUT.parent/"nadir-intervals"/"scenarios.csv",
        OUT.parent/"nadir-identifiability"/"matched-scan.csv"]
    paths += [OUT.parent/"nadir-pair-tests"/"independent-v1"/"pair-index.csv",ECGR/"pair-index.csv"]
    paths += [ROOT/"dev"/name for name in ("stage.json","protocol.json","entity-index.csv","pair-keys.csv",
                                         "visible-features.csv","single-index.csv","pair-index.csv")]
    project=HERE.parent/"Project"
    paths += [project/name for name in ("core/simulate_freq.m","core/swing_equation.m","core/pfr_model.m",
                                        "config/sim_config.m","config/sys_config.m","utils/calc_nadir.m")]
    return {str(path.relative_to(HERE.parent)):sha(path) for path in paths}


def assert_stage(out, stage, c):
    expected = json.loads((HERE/"nadir_pair_protocol.json").read_text())
    expected.update(version="quality-"+stage,reference_seed=c[stage+"_reference_seed"],
                    noise_seed=c[stage+"_noise_seed"],references_per_r_band=c[stage+"_pair_refs_per_band"],
                    noise_repeats=1)
    assert expected==json.loads((out/"protocol.json").read_text())
    assert json.loads((out/"stage.json").read_text())==dict(single_per_band=c[stage+"_single_per_band"],
        noise_seed=c[stage+"_noise_seed"],repeats=c[stage+"_noise_repeats"])


def assert_isolated(entity, stage):
    old_pair=OUT.parent/"nadir-pair-tests"/"independent-v1"/"pair-index.csv"
    prev = [pd.read_csv(OUT/name)[PARAM].to_numpy() for name in ("grid-index.csv","continuous-index.csv")]
    prev.append(pd.read_csv(OUT.parent/"nadir-intervals"/"scenarios.csv")[PARAM[:-1]+["Deficit"]].to_numpy())
    for source in (old_pair,ECGR/"pair-index.csv"):
        led=pd.read_csv(source)
        prev += [led[PARAM].to_numpy(),led[CPARAM].dropna().to_numpy()]
    devmatch=pd.read_csv(OUT.parent/"nadir-identifiability"/"matched-scan.csv")
    ref=np.column_stack([np.tile([4,1,.05,.5],(len(devmatch),1)),devmatch.ReferenceDeficit])
    candidate=ref.copy()
    for i,row in enumerate(devmatch.itertuples()):
        candidate[i,PARAM.index(row.Factor)]=row.FactorValue
    candidate[:,4]=devmatch.MatchedDeficit
    prev += [ref,candidate[np.isfinite(candidate).all(axis=1)]]
    if stage=="confirm":
        prev.append(pd.read_csv(ROOT/"dev"/"entity-index.csv")[PARAM].to_numpy())
    x=entity[PARAM].to_numpy()
    assert np.isfinite(x).all() and len(np.unique(x,axis=0))==len(x)
    assert (cKDTree(np.vstack(prev)).query(x,p=np.inf)[0]>1e-9).all()


def models():
    lock = verify_ecgr(ECGR)
    dev = load_dev(ECGR)
    baseline = baseline_models(dev,True)
    baseline["ecgr"],which = new_model(dev,lock["index"])
    x,n,*_=dev
    cols = dict(endpoint=[2],initial_rate=[2,3],quadratic=[2,4,5],fitted_segments=[2,6,7,8],ecgr=which)
    assert {name:checkpoint(model,x[:n,cols[name]]) for name,model in baseline.items()}==lock["anchor_prediction_sha256"]
    return baseline


def read_stage(stage,c):
    out = ROOT/stage
    assert_stage(out,stage,c)
    d = pd.read_csv(out/"visible-features.csv")
    entity = pd.read_csv(out/"entity-index.csv")
    assert not d.duplicated(KEY).any() and entity.Entity.is_unique
    assert_isolated(entity,stage)
    table = d.merge(entity[["Entity","Kind","RBand","Target_Hz"]],on="Entity",validate="many_to_one")
    table["Condition"] = table.Condition.map(dict(enumerate(c["conditions"],1)))
    assert table.Condition.notna().all()
    repeats = c[stage+"_noise_repeats"]
    assert len(table)==len(entity)*(1+(len(c["conditions"])-1)*repeats)
    assert set(table.loc[table.Condition.eq("clean"),"Rep"])=={0}
    assert set(table.loc[~table.Condition.eq("clean"),"Rep"])==set(range(1,repeats+1))
    assert len(table)==len(table.drop_duplicates(KEY))
    assert np.max(abs(table.Endpoint-table.ECGREndpoint))<1e-12
    for name,model in models().items():
        table[name]=predict(model,table[COLS[name]].to_numpy())
    assert np.isfinite(table[NAMES]).all().all()
    pairs = pd.read_csv(out/"pair-keys.csv")
    assert pairs.PairID.is_unique
    check = table.loc[table.Kind.ne("single"),["Entity","Condition","Rep"]].reset_index().rename(columns={"index":"Row"})
    a = pairs.merge(check,left_on="A",right_on="Entity",validate="many_to_many").rename(columns={"Row":"RowA"})
    b = a.merge(check,left_on=["B","Condition","Rep"],right_on=["Entity","Condition","Rep"],validate="many_to_one")
    b = b.rename(columns={"Row":"RowB"}).drop(columns=["Entity_x","Entity_y"])
    assert len(b)==len(pairs)*(1+(len(c["conditions"])-1)*repeats)
    assert not b.duplicated(["PairID","Condition","Rep"]).any()
    diff = table.Target_Hz.to_numpy()[b.RowB]-table.Target_Hz.to_numpy()[b.RowA]
    np.testing.assert_allclose(diff,b.DeltaTarget_Hz,rtol=0,atol=1e-12)
    return table,b


def choose(observed_q, rule):
    """Online decision: feature quality only, with endpoint as default."""
    q=np.asarray(observed_q)
    result=np.full(len(q),NAMES.index("endpoint"))
    valid=np.isfinite(q)&(q>=0)
    result[valid&(q<=rule["fallback"])]=NAMES.index(rule["middle"])
    result[valid&(q<=rule["clean"])]=NAMES.index("ecgr")
    return result


def inputs(table,pairs,c):
    primary=pairs.Status.eq("primary_eligible").to_numpy()
    point=table.Kind.eq("single").to_numpy()
    first,second=pairs.RowA.to_numpy(),pairs.RowB.to_numpy()
    return dict(p=table[NAMES].to_numpy(),y=table.Target_Hz.to_numpy(),
                a=first,b=second,d=pairs.DeltaTarget_Hz.to_numpy(),
                keep_pair=np.asarray([primary & pairs.Condition.eq(k).to_numpy() for k in c["conditions"]]),
                keep_single=np.asarray([point & table.Condition.eq(k).to_numpy() for k in c["conditions"]]))


def errs(matrix,prediction):
    return (abs(prediction-matrix["y"])*1000,
            abs(prediction[matrix["b"]]-prediction[matrix["a"]]-matrix["d"])*1000)


def stratified(matrix,error):
    return np.array([[np.mean(e[k]),np.quantile(e[k],.95)] for e,key in
                     zip(error,("keep_single","keep_pair"),strict=True) for k in matrix[key]]).reshape(2,-1,2)


def develop(c):
    assert not (ROOT/"confirm").exists(), "Development cannot revisit confirmation."
    assert not (ROOT/"frozen-selector.json").exists()
    table,pairs=read_stage("dev",c)
    matrix=inputs(table,pairs,c)
    fixed=np.array([stratified(matrix,errs(matrix,matrix["p"][:,j])) for j in range(len(NAMES))])
    # Best fixed configuration is chosen from all five by the same equal-weight
    # normalized single/pair MAE used to rank the gates, without confirmation data.
    norm=fixed[0,:,:,0]
    scores=(fixed[:,:,:,0]/norm).mean(axis=(1,2))
    best=int(np.argmin(scores))
    # Tail comparator: preserve the previous clean/30 dB reference, and use
    # the selected fixed configuration for all other conditions.
    guards=fixed[best,:,:,1].copy()
    guards[:,0]=fixed[NAMES.index("quadratic"),:,0,1]
    guards[:,2]=fixed[NAMES.index("endpoint"),:,2,1]
    guards*=c["tail_max_ratio"]
    rows=[]
    for proxy,lo,hi,middle in itertools.product(c["proxy_candidates"],c["clean_thresholds"],
                                               c["fallback_thresholds"],c["middle_experts"]):
        if lo>=hi:
            continue
        rule=dict(proxy=proxy,clean=lo,fallback=hi,middle=middle)
        q=table[{"pre":"PreQuality","hybrid":"HybridQuality"}[proxy]]
        choice=choose(q,rule)
        outcome=stratified(matrix,errs(matrix,matrix["p"][np.arange(len(q)),choice]))
        objective=float((outcome[:,:,0]/norm).mean())
        # Fixed penalty priorities in development; never reselect on the final set.
        tail=np.maximum(outcome[:,:,1]/guards-1,0).sum()
        clean_anchor=fixed[NAMES.index("quadratic"),:,0,0]
        gain=np.maximum(clean_anchor-fixed[NAMES.index("ecgr"),:,0,0],0)
        retention=np.maximum((outcome[:,0,0]-clean_anchor+c["clean_gain_retention"]*gain)/norm[:,0],0).sum()
        noise_guard=np.maximum(outcome[:,2,0]/(c["white30_mae_max_ratio_vs_endpoint"]*fixed[0,:,2,0])-1,0).sum()
        rows.append(rule|dict(objective=objective,tail_violation=tail,clean_violation=retention,
                              noise_mae_violation=noise_guard,selection_score=objective+2*(tail+retention+noise_guard)))
    candidates=pd.DataFrame(rows).sort_values(["selection_score","objective","proxy","clean","fallback","middle"])
    dev=ROOT/"dev"
    candidates.to_csv(dev/"gate-selection.csv",index=False)
    pd.DataFrame({"Feature":NAMES,"NormalizedMAE":scores}).to_csv(dev/"fixed-selection.csv",index=False)
    rule=candidates.iloc[0][["proxy","clean","fallback","middle"]].to_dict()
    # Check fallback and threshold behavior before allowing confirmation generation.
    expected=[NAMES.index("endpoint"),NAMES.index("ecgr"),NAMES.index(rule["middle"]),NAMES.index("endpoint")]
    assert choose([np.nan,rule["clean"],rule["fallback"],rule["fallback"]+.01],rule).tolist()==expected
    lock=dict(rule=rule,best_fixed=NAMES[best],hashes=hashes(),
        acceptance=dict(mae="each equal-condition relative change <= -2% and upper 95% limit < 0",
            tail="each P95 <= 105% of the frozen comparator",
            tail_comparators="clean quadratic; white30 endpoint; other conditions development best fixed",
            noise_mae="white30 <= 105% of endpoint for both tasks",
            clean="both tasks preserve 80% of the ECGR improvement against quadratic, or at least match quadratic if ECGR does not improve",
            jump="report separately; the per-condition P95 gate also applies"))
    (ROOT/"frozen-selector.json").write_text(json.dumps(lock,indent=2))
    print("LOCKED",rule,"best fixed",NAMES[best])


def verify():
    lock=json.loads((ROOT/"frozen-selector.json").read_text())
    assert lock["hashes"]==hashes()
    return lock

def metrics(values):
    mae=np.mean(abs(values))*1000
    return dict(MAE_mHz=mae,RMSE_mHz=np.sqrt(np.mean(values**2))*1000,P95_mHz=np.quantile(abs(values),.95)*1000)


def bootstrap(reference, bands, groups, conditions, base, gate, c):
    """Equal condition means. Every replicate of a cluster moves together."""
    unique=np.unique(reference)
    u=pd.DataFrame({"id":reference,"band":bands}).drop_duplicates()
    assert u.id.is_unique
    mapped=pd.Index(unique).get_indexer(reference)
    sizes=np.zeros((len(unique),len(conditions)))
    totals=np.zeros((2,len(unique),len(conditions)))
    for j,cond in enumerate(conditions):
        use=groups==cond
        np.add.at(sizes[:,j],mapped[use],1)
        for i,x in enumerate((base,gate)):
            np.add.at(totals[i,:,j],mapped[use],x[use])
    rng=np.random.default_rng(c["bootstrap_seed"])
    draws=c["bootstrap_draws"]
    w=np.zeros((draws,len(unique)),dtype=int)
    for _,band in u.groupby("band",sort=True):
        positions=pd.Index(unique).get_indexer(band.id)
        sampled=rng.choice(positions,size=(draws,len(positions)),replace=True)
        np.add.at(w,(np.repeat(np.arange(draws),len(positions)),sampled.ravel()),1)
    den=w@sizes
    valid=(den>0).all(axis=1)
    b0=(totals[0].sum(axis=0)/sizes.sum(axis=0)).mean()
    b1=(totals[1].sum(axis=0)/sizes.sum(axis=0)).mean()
    t0=((w@totals[0])[valid]/den[valid]).mean(axis=1)
    t1=((w@totals[1])[valid]/den[valid]).mean(axis=1)
    assert len(t0)>0 and b0>0
    interval=np.quantile(t1/t0-1,[.025,.975])
    return dict(BaselineMAE_mHz=b0,GateMAE_mHz=b1,RelativeChange=b1/b0-1,
                Low95=interval[0],High95=interval[1],Clusters=len(unique),ValidDraws=len(t0))


def confirm(c):
    lock=verify()
    out=ROOT/"confirm"
    assert not (out/"predictions.csv").exists()
    rule=lock["rule"]
    best=lock["best_fixed"]
    table,pairs=read_stage("confirm",c)
    match=pd.read_csv(out/"pair-index.csv")
    singles=pd.read_csv(out/"single-index.csv")
    status=pd.concat([match.groupby(["RBand","Status"]).size().rename("Count").reset_index().assign(Task="pair"),
        singles.groupby(["RBand","Status"]).size().rename("Count").reset_index().assign(Task="single")])
    status.to_csv(out/"status-counts.csv",index=False)
    quality_col={"pre":"PreQuality","hybrid":"HybridQuality"}[rule["proxy"]]
    quality=table[quality_col].to_numpy()
    selected=choose(quality,rule)
    table=table.assign(Selected=np.array(NAMES)[selected],Quality=quality,
                       gate=table[NAMES].to_numpy()[np.arange(len(table)),selected])
    pool=["endpoint",rule["middle"],"ecgr"]
    point_errors=abs(table[pool].to_numpy()-table.Target_Hz.to_numpy()[:,None])*1000
    chosen_point=abs(table.gate-table.Target_Hz)*1000
    best_point=np.min(point_errors,axis=1)
    tol=c["oracle_tolerance_mhz"]
    table=table.assign(Oracle=np.array(pool)[np.argmin(point_errors,axis=1)],OracleLoss_mHz=chosen_point-best_point,
                       Misselect=(chosen_point-best_point)>tol)
    a,b=pairs.RowA.to_numpy(),pairs.RowB.to_numpy()
    pairs=pairs.assign(SelectA=table.Selected.to_numpy()[a],SelectB=table.Selected.to_numpy()[b],
                       QualityA=quality[a],QualityB=quality[b])
    for k in NAMES+["gate"]:
        pairs[k]=table[k].to_numpy()[b]-table[k].to_numpy()[a]
    left,right=table[pool].to_numpy()[a],table[pool].to_numpy()[b]
    oracle=abs(right[:,None,:]-left[:,:,None]-pairs.DeltaTarget_Hz.to_numpy()[:,None,None])*1000
    index=oracle.reshape(len(pairs),-1).argmin(axis=1)
    minimum=oracle.reshape(len(pairs),-1).min(axis=1)
    gate_loss=abs(pairs.gate-pairs.DeltaTarget_Hz)*1000-minimum
    pairs=pairs.assign(OracleA=np.array(pool)[index//len(pool)],OracleB=np.array(pool)[index%len(pool)],
                       OracleLoss_mHz=gate_loss,Misselect=gate_loss>tol)
    table.to_csv(out/"predictions.csv",index=False)
    pairs.drop(columns=["RowA","RowB"]).to_csv(out/"pair-predictions.csv",index=False)
    point=table[table.Kind.eq("single")].copy()
    primary=pairs[pairs.Status.eq("primary_eligible")].copy()
    rows=[]
    mis=[]
    contrasts=[]
    for kind,p,t_col,ref_col in [("single",point,"Target_Hz","Entity"),("pair",primary,"DeltaTarget_Hz","ReferenceID")]:
        comparable=p if kind=="single" else pairs
        for cond,x in p.groupby("Condition"):
            all_x=comparable[comparable.Condition.eq(cond)]
            for name in NAMES+["gate"]:
                d0=metrics(x[name]-x[t_col])
                direction=dict(DirectionN=0,DirectionHits=np.nan,PredictedTies=np.nan,TrueTies=np.nan)
                if kind=="pair":
                    v=all_x[t_col].to_numpy()
                    w=all_x[name].to_numpy()
                    y=np.where(v>.0005,1,np.where(v<-.0005,-1,0))
                    d=np.where(w>.0005,1,np.where(w<-.0005,-1,0))
                    direction=dict(DirectionN=len(y),DirectionHits=np.sum(y==d),PredictedTies=np.sum(d==0),TrueTies=np.sum(y==0))
                rows.append(dict(Kind=kind,Condition=cond,Feature=name,N=len(x))|d0|direction)
            mis.append(dict(Kind=kind,Condition=cond,N=len(x),MisselectRate=x.Misselect.mean(),
                            MeanOracleLoss_mHz=x.OracleLoss_mHz.mean(),P95OracleLoss_mHz=x.OracleLoss_mHz.quantile(.95)))
            worst=x.assign(ExcessVsFixed_mHz=1000*(abs(x.gate-x[t_col])-abs(x[best]-x[t_col])))
            worst.nlargest(8,"ExcessVsFixed_mHz").to_csv(out/f"worst-{kind}-{cond}.csv",index=False)
        contrasts.append(dict(Kind=kind,Comparator=best)|bootstrap(p[ref_col].to_numpy(),p.RBand.to_numpy(),
            p.Condition.to_numpy(),c["conditions"],abs(p[best]-p[t_col]).to_numpy()*1000,
            abs(p.gate-p[t_col]).to_numpy()*1000,c))
    summary=pd.DataFrame(rows)
    summary.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(mis).to_csv(out/"mis-selection.csv",index=False)
    contrasts=pd.DataFrame(contrasts)
    contrasts.to_csv(out/"main-contrasts.csv",index=False)
    counts=point.groupby(["Condition","Selected","Oracle"]).size().rename("N").reset_index()
    counts.to_csv(out/"selection-confusion.csv",index=False)
    pairconf=primary.groupby(["Condition","SelectA","SelectB","OracleA","OracleB"]).size().rename("N").reset_index()
    pairconf.to_csv(out/"pair-selection-confusion.csv",index=False)
    apply_gates(out,c,lock,summary,contrasts)


def apply_gates(out,c,lock,summaries,contrasts):
    summary=summaries.set_index(["Kind","Condition","Feature"])
    checks=[]
    def record(kind,description,value,limit,comparison="le"):
        checks.append(dict(Kind=kind,Criterion=description,Value=value,Limit=limit,
                           Pass=bool(value<=limit if comparison=="le" else value<limit)))
    best=lock["best_fixed"]
    for row in contrasts.itertuples():
        k=row.Kind
        record(k,"equal-condition MAE relative improvement",row.RelativeChange,-c["relative_mae_gain_minimum"])
        record(k,"upper 95% limit of relative MAE change",row.High95,0.,"lt")
        for cond in c["conditions"]:
            guard={"clean":"quadratic","white30":"endpoint"}.get(cond,best)
            record(k,f"{cond} P95 vs {guard}",summary.loc[(k,cond,"gate"),"P95_mHz"],
                   c["tail_max_ratio"]*summary.loc[(k,cond,guard),"P95_mHz"])
        record(k,"white30 MAE vs endpoint",summary.loc[(k,"white30","gate"),"MAE_mHz"],
               c["white30_mae_max_ratio_vs_endpoint"]*summary.loc[(k,"white30","endpoint"),"MAE_mHz"])
        clean_quad=summary.loc[(k,"clean","quadratic"),"MAE_mHz"]
        g0=max(clean_quad-summary.loc[(k,"clean","ecgr"),"MAE_mHz"],0)
        record(k,"clean ECGR MAE gain retention",summary.loc[(k,"clean","gate"),"MAE_mHz"],
               clean_quad-c["clean_gain_retention"]*g0+1e-10)
    result=pd.DataFrame(checks)
    result.to_csv(out/"acceptance.csv",index=False)
    print("OVERALL:", "PASS" if result.Pass.all() else "FAIL")
    print(result.to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["develop","verify","confirm"])
    s=p.parse_args().stage
    if s=="verify":
        verify()
        print("PASS frozen selector and unchanged prior estimator")
    else:
        {"develop":develop,"confirm":confirm}[s](configuration())
