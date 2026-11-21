"""Freeze ECGR on historical training/validation and score the new pairs once."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.io import loadmat

from analyze_revision import OUT, fit_model, predict

HERE = Path(__file__).resolve().parent
METHOD = "ecgr"
KEY = ["ReferenceID","PairID","Member","Condition","Rep"]
SLOTS = {"endpoint":0,"initial_rate":1,"quadratic":2,"fitted_segments":4}
COLS = {"endpoint":["Endpoint"],"initial_rate":["Endpoint","InitialRate"],
        "quadratic":["Endpoint","Linear","Curvature"],
        "fitted_segments":["Endpoint","Slope1","Slope2","Slope3"],
        METHOD:["Endpoint","Symmetric","Antisymmetric"]}
HISTORY = OUT.parent/"nadir-pair-tests"/"independent-v1"
LOCKED = ["nadir_ecgr.py","nadir_ecgr.m","nadir_ecgr_protocol.json","nadir_pair_test.m",
          "analyze_revision.py","run_nadir_ecgr.ps1"]
PARAMS = ["H","D","R","Tg","d"]
CANDIDATE = ["CandidateH","CandidateD","CandidateR","CandidateTg","CandidateD_pu"]


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def files(out):
    return ([HERE/name for name in LOCKED]+[out/name for name in
             ("method-protocol.json","protocol.json","development-features.csv","training-cubic-moment.csv")]
            +[OUT/name for name in ("grid-index.csv","features.mat","trajectories.mat","predictions.csv")]
            +[HISTORY/"pair-index.csv"]+
            [HERE.parent/"Project"/name for name in ("config/sim_config.m","config/sys_config.m",
                "core/simulate_freq.m","core/pfr_model.m","core/swing_equation.m","utils/calc_nadir.m")])


def environment():
    return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                scikit_learn=sklearn.__version__)


def load_dev(out):
    grid = pd.read_csv(OUT/"grid-index.csv")
    x = np.loadtxt(out/"development-features.csv",delimiter=",",ndmin=2)
    n = len(grid)
    assert np.array_equal(x[:n,0],grid.ID) and (x[:n,1]==0).all()
    train,valid,test = [np.flatnonzero(grid.Split.eq(k)) for k in ("train","validation","test")]
    y = grid.Target_Hz.to_numpy()
    noisy_ids = x[n:,0].astype(int)-1
    assert set(noisy_ids)==set(valid)
    assert len(set(map(tuple,x[n:,:2]))) == len(noisy_ids)
    return x,n,y,train,valid,test,noisy_ids


def baseline_models(grid_data, check_archive=False):
    x,n,y,train,valid,test,_ = grid_data
    saved = loadmat(OUT/"features.mat",simplify_cells=True)["X"]
    models = {}
    archived = pd.read_csv(OUT/"predictions.csv") if check_archive else None
    for name,idx in SLOTS.items():
        a = np.asarray(saved[idx]).reshape(n,-1)
        models[name] = fit_model(a,y,train,valid,"tree")
        if check_archive:
            original = archived.query("Experiment=='conditional' and Model=='tree' and Feature==@name").sort_values("ID")
            np.testing.assert_allclose(predict(models[name],a[test]),original.Prediction_Hz,atol=1e-10,rtol=0)
    return models


def new_model(dev,index):
    x,n,y,train,valid,_,_ = dev
    columns = slice(9+3*index,12+3*index)
    return fit_model(x[:n,columns],y,train,valid,"tree"), columns


def checkpoint(model,x):
    # Pin the freshly fitted deterministic model by its historical prediction vector.
    return hashlib.sha256(predict(model,x).astype("<f8").tobytes()).hexdigest()


def freeze(out,c):
    assert not (out/"freeze.json").exists() and not (out/"pair-index.csv").exists()
    assert digest(out/"method-protocol.json")==digest(HERE/"nadir_ecgr_protocol.json")
    (out/"protocol.json").write_text(json.dumps(c["confirmation"],indent=2),encoding="utf-8")
    dev = load_dev(out)
    x,n,y,_,val,_,noisy_ids = dev
    baselines = baseline_models(dev,True)
    baseline = baselines["quadratic"]
    clean_baseline = np.mean(abs(predict(baseline,x[val][:,[2,4,5]])-y[val]))
    noise_baseline = np.mean(abs(predict(baseline,x[n:][:,[2,4,5]])-y[noisy_ids]))
    results = []
    for i,lam in enumerate(c["lambda_candidates"]):
        model,cols = new_model(dev,i)
        clean = np.mean(abs(predict(model,x[val,cols])-y[val]))
        white = np.mean(abs(predict(model,x[n:,cols])-y[noisy_ids]))
        results.append(dict(index=i,lambda_value=lam,validation_clean_MAE_Hz=clean,
            validation_white_MAE_Hz=white,quadratic_clean_MAE_Hz=clean_baseline,
            quadratic_white_MAE_Hz=noise_baseline,score=.5*(clean/clean_baseline+white/noise_baseline)))
    pd.DataFrame(results).to_csv(out/"validation-selection.csv",index=False)
    winner = min(results,key=lambda row:(row["score"],row["lambda_value"]))
    x_held = {"endpoint":[2],"initial_rate":[2,3],"quadratic":[2,4,5],"fitted_segments":[2,6,7,8]}
    anchor_hashes = {key:checkpoint(model,x[:n,columns]) for key,model in baselines.items()
                     for columns in [x_held[key]]}
    selected,columns = new_model(dev,winner["index"])
    anchor_hashes[METHOD] = checkpoint(selected,x[:n,columns])
    lock = dict(environment=environment(),index=winner["index"],**{"lambda":winner["lambda_value"]},
                training_cubic_second_moment_hz2=float((out/"training-cubic-moment.csv").read_text()),
                anchor_prediction_sha256=anchor_hashes)
    lock["hashes"] = {str(p.relative_to(HERE.parent)):digest(p) for p in files(out)}
    (out/"freeze.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print("Historical anchor verified; selected on historical validation:",winner)


def verify(out):
    lock = json.loads((out/"freeze.json").read_text())
    assert lock["environment"] == environment()
    assert lock["hashes"] == {str(p.relative_to(HERE.parent)):digest(p) for p in files(out)}
    return lock


def signed(x,tie):
    return np.where(x>tie,1,np.where(x<-tie,-1,0))


def collision(ledger):
    old = pd.read_csv(HISTORY/"pair-index.csv")
    previous = np.unique(np.vstack([old[PARAMS],old[CANDIDATE]]),axis=0)
    current = np.vstack([ledger[PARAMS].drop_duplicates(),ledger[CANDIDATE].dropna()])
    assert np.isfinite(current).all()
    assert len(np.unique(current,axis=0))==len(current)
    assert all(np.min(np.max(abs(previous-x),axis=1))>1e-9 for x in current)


def summary(df,error):
    e = df[error].abs().to_numpy()*1000
    return dict(N=len(e),MAE_mHz=e.mean(),RMSE_mHz=np.sqrt(np.mean(e**2)),P95_mHz=np.quantile(e,.95))


def weights(refs,q):
    rng = np.random.default_rng(q["bootstrap_seed"])
    n = q["bootstrap_draws"]
    w = np.zeros((n,len(refs)),int)
    for _,b in refs.groupby("RBand"):
        ids = b.ReferenceID.to_numpy()-1
        sample = rng.choice(ids,size=(n,len(ids)),replace=True)
        np.add.at(w,(np.repeat(np.arange(n),len(ids)),sample.ravel()),1)
    return w


def infer(out,c):
    lock = verify(out)
    assert not (out/"pair-predictions.csv").exists()
    q = c["confirmation"]
    ledger = pd.read_csv(out/"pair-index.csv")
    refs = pd.read_csv(out/"references.csv")
    collision(ledger)
    traces = loadmat(out/"trajectories.mat",simplify_cells=True)
    vis,post = traces["vis"]-1,traces["post"]-1
    t = traces["t"]
    np.testing.assert_allclose(t[[vis[0],vis[-1],post[0],post[-1]]],[1,1.5,1.52,20],atol=1e-12,rtol=0)
    assert len(vis)==26 and len(post)==925
    assert len(refs)==2*q["references_per_r_band"] and len(ledger)==3*len(refs)
    assert not ledger.PairID.duplicated().any()
    included = ledger.Status.isin(["primary_eligible","true_tie","below_minimum_gap"])
    eligible = ledger[included]
    a = traces["traceA"][eligible.ReferenceID.to_numpy()-1]
    b = traces["traceB"][eligible.PairID.to_numpy()-1]
    for trace, side in [(a,"A"),(b,"B")]:
        np.testing.assert_allclose(trace[:,post].min(axis=1),eligible[f"Target{side}_Hz"],atol=1e-10,rtol=0)
        np.testing.assert_allclose(trace[:,vis[-1]],eligible[f"Endpoint{side}_Hz"],atol=1e-10,rtol=0)
    assert (eligible.EndpointB_Hz-eligible.EndpointA_Hz).abs().le(q["endpoint_tolerance_hz"]).all()
    np.testing.assert_allclose(eligible.DeltaTarget_Hz,eligible.TargetB_Hz-eligible.TargetA_Hz,atol=1e-12,rtol=0)
    status = np.select([eligible.DeltaTarget_Hz.abs()<=q["tie_hz"],
                        eligible.DeltaTarget_Hz.abs()<q["minimum_target_gap_hz"]],
                       ["true_tie","below_minimum_gap"],default="primary_eligible")
    assert np.array_equal(eligible.Status,status)
    member_base = pd.read_csv(out/"features.csv").set_index(KEY,verify_integrity=True)
    member_new = pd.read_csv(out/"ecgr-features.csv").set_index(KEY,verify_integrity=True)
    assert member_base.index.equals(member_new.index)
    assert len(member_base)==len(eligible)*2*(1+q["noise_repeats"])
    dev = load_dev(out)
    models = baseline_models(dev)
    models[METHOD],cols = new_model(dev,lock["index"])
    x,n,*_ = dev
    cols_by_name = {"endpoint":[2],"initial_rate":[2,3],"quadratic":[2,4,5],
                    "fitted_segments":[2,6,7,8],METHOD:cols}
    assert {key:checkpoint(model,x[:n,cols_by_name[key]]) for key,model in models.items()} == lock["anchor_prediction_sha256"]
    print("PASS archived hypotheses and deterministic model rebuild, disjoint historical pairs, exact cutoff labels, common members")

    outputs = []
    for name,model in models.items():
        source = member_new if name==METHOD else member_base
        p = predict(model,source[COLS[name]].to_numpy())
        assert np.isfinite(p).all()
        outputs.append(pd.DataFrame(dict(Feature=name,Prediction_Hz=p),index=source.index).reset_index())
    member = pd.concat(outputs,ignore_index=True).merge(eligible[["PairID","TargetA_Hz","TargetB_Hz","RBand"]],
        on="PairID",validate="many_to_one")
    member["Target_Hz"] = np.where(member.Member.eq(1),member.TargetA_Hz,member.TargetB_Hz)
    member["Error_Hz"] = member.Prediction_Hz-member.Target_Hz
    member.to_csv(out/"member-predictions.csv",index=False)
    # A is shared by several candidates. Give each physical reference one
    # vote in single-scenario metrics, while retaining each candidate once.
    a = member[member.Member.eq(1)]
    assert a.groupby(["ReferenceID","Condition","Rep","Feature"]).Prediction_Hz.nunique().eq(1).all()
    unique_a = a.drop_duplicates(["ReferenceID","Condition","Rep","Feature"])
    unique = pd.concat([unique_a,member[member.Member.eq(2)]],ignore_index=True)
    single = pd.DataFrame([dict(Condition=cond,Feature=feat,**summary(b,"Error_Hz"))
                           for (cond,feat),b in unique.groupby(["Condition","Feature"])])
    single.to_csv(out/"single-results.csv",index=False)

    p = member.pivot(index=["ReferenceID","PairID","Condition","Rep","Feature"],columns="Member",values="Prediction_Hz")
    assert p.notna().all().all()
    p = p.rename(columns={1:"PredictionA_Hz",2:"PredictionB_Hz"}).reset_index()
    p = p.merge(eligible[["PairID","Status","DeltaTarget_Hz","ChangedFactor","RBand"]],on="PairID",validate="many_to_one")
    p["DeltaPrediction_Hz"] = p.PredictionB_Hz-p.PredictionA_Hz
    p["DeltaError_Hz"] = p.DeltaPrediction_Hz-p.DeltaTarget_Hz
    p["TrueDirection"] = signed(p.DeltaTarget_Hz,q["tie_hz"])
    p["PredictedDirection"] = signed(p.DeltaPrediction_Hz,q["tie_hz"])
    p["Correct"] = p.TrueDirection.eq(p.PredictedDirection)
    p.to_csv(out/"pair-predictions.csv",index=False)
    main = p[p.Status.eq("primary_eligible")]
    metric_rows=[]
    for (condition,name), groups in p.groupby(["Condition","Feature"]):
        b = groups[groups.Status.eq("primary_eligible")]
        metric_rows.append(dict(Condition=condition,Feature=name,**summary(b,"DeltaError_Hz"),
            PhysicalPairs=b.PairID.nunique(),CorrectDirections=groups.Correct.sum(),
            DirectionTrials=len(groups),DirectionAccuracy=groups.Correct.mean(),
            PredictionTies=groups.PredictedDirection.eq(0).sum(),TrueTies=groups.TrueDirection.eq(0).sum()))
    metrics = pd.DataFrame(metric_rows).set_index(["Condition","Feature"])
    metrics.to_csv(out/"pair-results.csv")
    m = weights(refs,q)
    contrasts = []
    for condition,b in main.groupby("Condition"):
        pivot = b.assign(Abs_mHz=b.DeltaError_Hz.abs()*1000).pivot(
            index=["ReferenceID","PairID","Rep"],columns="Feature",values="Abs_mHz")
        for base in SLOTS:
            d = (pivot[METHOD]-pivot[base]).groupby(level=["ReferenceID","PairID"]).mean()
            count = d.groupby(level="ReferenceID").size().reindex(refs.ReferenceID,fill_value=0).to_numpy()
            sums = d.groupby(level="ReferenceID").sum().reindex(refs.ReferenceID,fill_value=0).to_numpy()
            denom = m@count
            values = (m@sums)[denom>0]/denom[denom>0]
            lo,hi = np.quantile(values,[c["confidence_alpha"]/2,1-c["confidence_alpha"]/2])
            contrasts.append(dict(Condition=condition,Comparator=base,ChangeMAE_mHz=d.mean(),
                Low95_mHz=lo,High95_mHz=hi,Draws=len(values),ReferenceClusters=np.count_nonzero(count)))
    effects = pd.DataFrame(contrasts).set_index(["Condition","Comparator"])
    effects.to_csv(out/"paired-contrasts.csv")
    count = ledger.groupby(["RBand","ChangedFactor","Status"]).size().rename("Count")
    count.to_csv(out/"pair-status-counts.csv")
    gate(out,c,metrics,single,effects)


def gate(out,c,p,s,effects):
    margin = 1+c["guard_relative_margin"]
    clean,noise = c["clean_best_reference"],c["noise_guard_reference"]
    e = effects.loc[("clean",clean)]
    pcheck = [
        ("clean paired MAE vs locked best: upper two-sided 95% bound < 0",float(e.High95_mHz),0.),
        ("clean paired MAE below every baseline",float(p.loc[("clean",METHOD),"MAE_mHz"]),
         float(p.loc[("clean",list(SLOTS)),"MAE_mHz"].min())),
    ]
    s = s.set_index(["Condition","Feature"])
    for set_name,t,column,cond,base in [
        ("pair",p,"P95_mHz","clean",clean),("pair",p,"MAE_mHz","white30",noise),
        ("pair",p,"P95_mHz","white30",noise),("single",s,"P95_mHz","clean",clean),
        ("single",s,"MAE_mHz","white30",noise),("single",s,"P95_mHz","white30",noise)]:
        pcheck.append((f"{set_name} {cond} {column}",float(t.loc[(cond,METHOD),column]),
                       float(margin*t.loc[(cond,base),column])))
    results = pd.DataFrame(pcheck,columns=["Criterion","Value","Threshold"])
    results["Pass"] = [row.Value<row.Threshold if i<2 else row.Value<=row.Threshold
                       for i,row in enumerate(results.itertuples())]
    results.to_csv(out/"acceptance.csv",index=False)
    print(results.to_string(index=False))
    print("OVERALL:", "PASS" if results.Pass.all() else "FAIL")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["freeze","verify","infer"])
    p.add_argument("directory",type=Path)
    a = p.parse_args()
    out = a.directory.resolve()
    c = json.loads((out/"method-protocol.json").read_text())
    if a.stage=="verify":
        verify(out)
    else:
        {"freeze":freeze,"infer":infer}[a.stage](out,c)
