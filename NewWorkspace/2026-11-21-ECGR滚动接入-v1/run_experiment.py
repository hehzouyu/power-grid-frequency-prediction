"""Evaluate transferring fixed-cutoff ECGR representation selection to rolling alarms."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz, solve_triangular
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingRegressor
import joblib

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
REPRO = ROOT / "repro"
sys.path.insert(0, str(REPRO))
from run_rolling_warning import DT, simulate, truth_times, threshold, evaluate

ARCH = REPRO/"out"
EC = ARCH/"nadir-ecgr/confirmation-v1"
QD = ARCH/"nadir-quality"
PARAMS = ["H","D","R","Tg","d"]
RANGES = np.array([[2,6],[.5,2],[.04,.06],[.2,1],[.02,.08]])
POLICIES = {
    "persistence": ("persistence", False, False),
    "endpoint": ("endpoint", False, False),
    "quadratic": ("quadratic", False, False),
    "segmented": ("segmented", False, False),
    "ecgr": ("ecgr", False, False),
    "gate": ("gate", False, False),
    "gate_final_only": ("gate_final_only", False, False),
    "quadratic_fixed": ("quadratic", False, True),
    "quadratic_stratified": ("quadratic", True, False),
    "gate_stratified": ("gate", True, False),
}
PROTOCOL = DEST/"protocol.json"
MODEL = DEST/"window-models.joblib"

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def ecgr_parameters():
    lock=read(EC/"freeze.json")
    method=read(EC/"method-protocol.json")
    return (method["working_ar1_rho"],method["working_ar1_weight"],
            lock["lambda"],lock["training_cubic_second_moment_hz2"])

def ecgr(f, params):
    """Use the locked ECGR equations. Resize the fixed basis and covariance to n."""
    rho,weight,penalty,prior=params
    n=f.shape[1]
    u=np.linspace(0.,1.,n)[1:-1]
    L=np.column_stack((u-1,np.eye(n-2),-u))
    B=np.column_stack((u*(1-u),u*(1-u)*(2*u-1)))
    S=(1-weight)*np.eye(n)+weight*toeplitz(rho**np.arange(n))
    C=np.linalg.cholesky(L@S@L.T)
    W=solve_triangular(C,L,lower=True)
    Z=solve_triangular(C,B,lower=True)
    G=Z.T@Z
    ls=np.linalg.solve(G,Z.T@W)
    rough=W-Z@ls
    # Center around the nominal value to avoid subtracting 50 Hz repeatedly
    # inside the linear operators. L annihilates the constant and the chord.
    y=f-50
    z=(y@ls.T)@G
    s2=np.sum((y@rough.T)**2,axis=1)/(n-4)
    g00,g01,g11=G[0,0],G[0,1],G[1,1]+penalty*s2/prior
    det=g00*g11-g01*g01
    return np.column_stack((y[:,-1],(z[:,0]*g11-z[:,1]*g01)/det,
                            (z[:,1]*g00-z[:,0]*g01)/det))

def features(window, e):
    w=window.shape[1]-1
    x=np.linspace(0,1,w+1)
    deviations=window-50
    slopes=np.rint(np.linspace(0,w,4)).astype(int)
    quad=np.linalg.lstsq(np.column_stack((np.ones(w+1),x,x*x)),deviations.T,rcond=None)[0]
    return {
        "endpoint":deviations[:,-1:],
        "quadratic":np.column_stack((deviations[:,-1],quad[1:].T)),
        "segmented":np.column_stack((deviations[:,-1],np.diff(deviations[:,slopes],axis=1)/(DT*np.diff(slopes)))),
        "ecgr":ecgr(window,e),
    }

def quality(pre, y, floor):
    center=pre.mean(axis=1)
    scale=np.maximum(np.sqrt(np.mean((y-center[:,None])**2,axis=1)),floor)
    qpre=np.sqrt(np.mean((pre-center[:,None])**2,axis=1))/scale
    qwin=np.sqrt(np.mean(np.diff(y,n=3,axis=1)**2,axis=1))/np.sqrt(20)/scale
    return np.maximum(qpre,qwin)

def select(q,rule):
    valid=np.isfinite(q)&(q>=0)
    result=np.full(q.shape,"endpoint",dtype="<U9")
    result[valid&(q<=rule["fallback"])]=rule["middle"]
    result[valid&(q<=rule["clean"])]= "ecgr"
    return result

def historical_fit(config,params):
    grid=loadmat(ARCH/"revision/trajectories.mat",simplify_cells=True)["grid"]
    mask=np.asarray(grid["t"])>=1
    f=grid["F"][:,mask]
    np.testing.assert_allclose(grid["t"][mask]-1,np.arange(f.shape[1])*DT,rtol=0,atol=1e-10)
    old=np.loadtxt(EC/"development-features.csv",delimiter=",")[:len(f)]
    index=read(EC/"freeze.json")["index"]
    check=ecgr(f[:,:26],params)
    np.testing.assert_allclose(check,old[:,9+3*index:12+3*index],rtol=0,atol=3e-12)
    pool={}
    for w in config["windows"]:
        target=f[:,w+1:].min(axis=1)
        pool[w]={name: GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=42).fit(x,target)
                 for name,x in features(f[:,:w+1],params).items()}
    print("Historical MATLAB ECGR feature check, 1,200 26-point windows: max discrepancy",
          np.max(np.abs(check-old[:,9+3*index:12+3*index])))
    return pool

def scene_parameters(config,stage):
    return np.random.default_rng(config[f"{stage}_seed"]).uniform(RANGES[:,0],RANGES[:,1],(config[f"{stage}_events"],5))

def assert_disjoint(a,b):
    assert (cKDTree(a).query(b,p=np.inf)[0]>1e-9).all()
    previous=[pd.read_csv(ARCH/"revision/grid-index.csv")[PARAMS].to_numpy(),
              pd.read_csv(QD/"dev/entity-index.csv")[PARAMS].to_numpy(),
              pd.read_csv(QD/"confirm/entity-index.csv")[PARAMS].to_numpy()]
    with np.load(ARCH/"rolling/aggregate-trajectories.npz") as data:
        previous.append(data["params"])
    assert (cKDTree(np.row_stack(previous) if hasattr(np,"row_stack") else np.vstack(previous)).query(np.vstack([a,b]),p=np.inf)[0]>1e-9).all()

def overlays(d,config,stage):
    """Noise uses clean simulated amplitude only here. Selection sees measured samples."""
    rng=np.random.default_rng(config[f"{stage}_noise_seed"])
    pre_samples=50
    points=max(config["windows"])+1
    clean=np.column_stack((np.full((len(d),pre_samples),50.),d[:,:points]))
    rms=np.sqrt(np.mean((d[:,:points]-50)**2,axis=1))[:,None]
    white=rng.normal(size=clean.shape)
    innovation=rng.normal(size=clean.shape)
    ar,_=lfilter([.6],[1,-.8],innovation,axis=1,zi=.8*rng.normal(size=(len(clean),1)))
    deviations={
        "clean":np.zeros_like(clean),
        "white40":rms*.01*white, "white30":rms*10**(-1.5)*white,
        "white20":rms*.1*white, "ar30":rms*10**(-1.5)*ar,
        "jump":rms*np.column_stack((.01*white[:,:pre_samples],.1*ar[:,pre_samples:])),
    }
    for condition in config["conditions"]:
        obs=clean+deviations[condition]
        yield condition,obs[:,:pre_samples],obs[:,pre_samples:]

def scoring_inputs(config,stage,pool,params):
    x=scene_parameters(config,stage)
    f=simulate(x)
    crossing,_=truth_times(f)
    f=50*(1+f)
    quality_ref=read(ARCH/"rolling/protocol-results.json")["noise_rms_reference_Hz"]*10**(-1.5)
    rule=read(QD/"frozen-selector.json")["rule"]
    floor=read(REPRO/"nadir_quality_protocol.json")["signal_floor_hz"]
    table=dict(scene=pd.DataFrame(x,columns=PARAMS).assign(Event=np.arange(len(x)),Crossing_s=crossing))
    trace=[]
    for cond,pre,post in overlays(f,config,stage):
        stack=[]
        flag=pre.std(axis=1,ddof=1)>=quality_ref
        for w in config["windows"]:
            obs=post[:,:w+1]
            z=features(obs,params)
            q=quality(pre,obs,floor)
            chosen=select(q,rule)
            pred={k:pool[w][k].predict(v) for k,v in z.items()}
            pred["persistence"]=obs[:,-1]
            pred["gate"]=np.choose(np.select([chosen=="ecgr",chosen=="quadratic"],[2,1],default=0),
                                   [pred["endpoint"],pred["quadratic"],pred["ecgr"]])
            pred["gate_final_only"]=pred["gate"] if w==25 else pred["quadratic"]
            block=pd.DataFrame(pred).assign(Event=np.arange(len(x)),Condition=cond,Window_s=w*DT,
                         Crossing_s=crossing,Target_Hz=f[:,w+1:].min(axis=1),Q=q,Selected=chosen,LowPreQuality=flag)
            stack.append(block)
        trace.extend(stack)
    table["trace"]=pd.concat(trace,ignore_index=True)
    return table

def ratios(a):
    return np.stack((a[...,1]/a[...,0],a[...,3]/a[...,2]),axis=-1)

def candidates(trace,config):
    id_cols=["Condition","Event","Window_s"]
    trace=trace.sort_values(id_cols).reset_index(drop=True)
    assert not trace.duplicated(id_cols).any()
    n=trace.Event.nunique()
    for cond in config["conditions"]:
        block=trace[trace.Condition.eq(cond)]
        dims=(n,len(config["windows"]))
        flags=block.LowPreQuality.to_numpy().reshape(dims)
        crossing=block.Crossing_s.to_numpy().reshape(dims)
        assert (flags==flags[:,0,None]).all() and (crossing==crossing[:,0,None]).all()
        yield cond,block,flags[:,0],crossing[:,0]

def calibration(trace,config):
    by=list(candidates(trace,config))
    result=[]
    for target in config["far_targets"]:
        for policy,(source,strat,last) in POLICIES.items():
            low,high=[],[]
            for cond,b,flag,cross in by:
                if cond not in config["calibration_conditions"]:
                    continue
                score=b[source].to_numpy().reshape(len(flag),len(config["windows"]))
                value=score[:,-1] if last else score.min(axis=1)
                eligible=np.isinf(cross)
                low.extend(value[eligible&~flag])
                high.extend(value[eligible&flag])
            all_values=np.r_[low,high]
            pooled=threshold(all_values,target)
            for group,values in (("all",all_values),("high_quality",low),("low_quality",high)):
                if not strat and group!="all": continue
                values=np.asarray(values)
                cut=pooled if group=="all" or len(values)==0 else threshold(values,target)
                result.append(dict(TargetFAR=target,Policy=policy,Group=group,Cutoff_Hz=cut,
                                   NegativeVariants=len(values),Triggered=int((values<cut).sum())))
    return pd.DataFrame(result)

def event_scores(trace,lock,config):
    times=np.asarray(config["windows"])*DT
    results=[]
    for cond,b,flag,cross in candidates(trace,config):
        for target in config["far_targets"]:
            for policy,(source,strat,last) in POLICIES.items():
                t=lock[(lock.TargetFAR==target)&lock.Policy.eq(policy)].set_index("Group").Cutoff_Hz
                cut=np.where(flag,t["low_quality"],t["high_quality"]) if strat else np.full(len(flag),t["all"])
                assert len(t)==(3 if strat else 1)
                p=b[source].to_numpy().reshape(len(flag),len(times))
                idx=np.array([len(times)-1]) if last else np.arange(len(times))
                a,lead,outcome,early=evaluate(p,idx,cut[:,None],times,cross)
                results.append(pd.DataFrame(dict(Event=np.arange(len(flag)),Condition=cond,Policy=policy,
                               TargetFAR=target,Crossing_s=cross,Alarm_s=a,Lead_ms=lead,Outcome=outcome,
                               BeforeFirstWindow=early,Cutoff_Hz=cut,
                               LowPreQuality=flag)))
    return pd.concat(results,ignore_index=True)

def fit_calibration(config):
    if (DEST/"calibration-lock.json").exists(): raise FileExistsError("Calibration is already frozen")
    params=ecgr_parameters()
    a,b=(scene_parameters(config,x) for x in ("calibration","test"))
    assert_disjoint(a,b)
    pool=historical_fit(config,params)
    joblib.dump(pool,MODEL)
    scores=scoring_inputs(config,"calibration",pool,params)
    scores["scene"].to_csv(DEST/"calibration-scenarios.csv",index=False)
    scores["trace"].to_csv(DEST/"calibration-window-predictions.csv.gz",index=False)
    table=calibration(scores["trace"],config)
    table.to_csv(DEST/"cutoffs.csv",index=False)
    files=[PROTOCOL,Path(__file__),MODEL,DEST/"cutoffs.csv",QD/"frozen-selector.json",
           EC/"freeze.json",EC/"method-protocol.json",REPRO/"run_rolling_warning.py"]
    dump(DEST/"calibration-lock.json",{str(x.relative_to(ROOT)):sha(x) for x in files})
    print("Calibration frozen. Negatives by group:",table.NegativeVariants.unique())

def confirm(config):
    if (DEST/"confirmation-started.json").exists(): raise FileExistsError("Confirmation already started")
    for name,digest in read(DEST/"calibration-lock.json").items():
        if sha(ROOT/name)!=digest: raise ValueError(name+" changed after calibration")
    pool=joblib.load(MODEL)
    # Commit the confirmation attempt before loading holdout observations.
    dump(DEST/"confirmation-started.json",{"calibration_lock_sha256":sha(DEST/"calibration-lock.json")})
    data=scoring_inputs(config,"test",pool,ecgr_parameters())
    data["scene"].to_csv(DEST/"test-scenarios.csv",index=False)
    data["trace"].to_csv(DEST/"test-window-predictions.csv.gz",index=False)
    scores=event_scores(data["trace"],pd.read_csv(DEST/"cutoffs.csv"),config)
    scores.to_csv(DEST/"test-event-scores.csv.gz",index=False)
    print(f"Confirmed {len(scores)} event/condition/policy rows.")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("calibrate","confirm"))
    stage=parser.parse_args().stage
    config=read(PROTOCOL)
    {"calibrate":fit_calibration,"confirm":confirm}[stage](config)
