"""Generate IEEE 14/39 synthetic dynamic frequency traces in ANDES."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import andes
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent/"out"/"rolling"
CASES={"ieee14":("ieee14/ieee14.raw","ieee14/ieee14.dyr"),
       "ieee39":("ieee39/ieee39_full.xlsx",None)}
TIME=np.arange(1001)*.02


def load(case):
    name,extra=CASES[case]
    return andes.load(andes.get_case(name),addfile=andes.get_case(extra) if extra else None,
                      setup=False,default_config=True,no_output=True,pycode_path=str(OUT/"andes-cache"))


def coi_frequency(ss):
    speeds=[]; weights=[]
    for model in ss.SynGen.models.values():
        if model.n:
            active=np.asarray(model.u.v)>0
            speeds.append(ss.dae.ts.x[:,model.omega.a][:,active])
            weights.extend(np.asarray(model.M.v)[active])
    return np.average(np.column_stack(speeds),axis=1,weights=weights)


def prepare_scenario(case,scale,site,fraction,distributed=False,load_ranks=None):
    ss=load(case)
    for model in (ss.Toggle,ss.Alter):
        for idx in model.idx.v:
            model.set("u",idx,0)
    loads=np.asarray(ss.PQ.p0.v,dtype=float)
    reactive=np.asarray(ss.PQ.q0.v,dtype=float)
    positions=np.argsort(loads)[::-1]
    selected=positions[[0,len(positions)//3,2*len(positions)//3]][site]
    readout=[int(ss.Bus.idx.v[i]) for i in [0,ss.Bus.n//2,ss.Bus.n-1]]
    for j,bus in enumerate(readout):
        ss.add("BusFreq",idx=f"StudyFreq{j}",bus=bus,fn=60.)
    demand=loads.sum()*scale
    for i,k in enumerate(ss.PQ.idx.v):
        ss.PQ.set("p0",k,loads[i]*scale)
        ss.PQ.set("q0",k,reactive[i]*scale)
    for key,value in dict(p2p=1.,p2i=0.,p2z=0.,q2q=1.,q2i=0.,q2z=0.).items():
        setattr(ss.PQ.config,key,value)
    locations=(positions[list(load_ranks)] if load_ranks is not None else
               positions[site::3][:4] if distributed else [selected])
    total=loads[locations].sum()
    for loc in locations:
        device=ss.PQ.idx.v[loc]
        ss.add("Alter",model="PQ",dev=device,src="Ppf",method="+",
               amount=demand*fraction*loads[loc]/total,t=1.)
        ss.add("Alter",model="PQ",dev=device,src="Qpf",method="+",
               amount=demand*fraction*reactive[loc]/total,t=1.)
    if not ss.setup() or not ss.PFlow.run():
        raise RuntimeError("power flow failed")
    buses=";".join(str(int(ss.PQ.bus.v[i])) for i in locations)
    return ss,buses,readout


def simulate(case,scale,site,fraction,distributed=False,load_ranks=None,*,integration_step=.02,
             integration_method=None,inspect_trace=None):
    ss,buses,readout=prepare_scenario(case,scale,site,fraction,distributed,load_ranks)
    ss.TDS.config.tf=21.
    ss.TDS.config.tstep=integration_step
    if integration_method is not None:
        ss.TDS.set_method(integration_method)
    completed=ss.TDS.run()
    if inspect_trace is not None:
        inspect_trace(ss)
    if not completed:
        raise RuntimeError(f"time-domain simulation failed at t={float(ss.dae.t):.6f} s: {ss.TDS.err_msg}")
    ts=ss.dae.ts
    if ts.t[-1] <20.999:
        raise RuntimeError(f"simulation ended at {ts.t[-1]}")
    coi=coi_frequency(ss)
    read=ts.y[:,ss.BusFreq.f.a[-3:]]
    # Previous-sample hold prevents look-ahead when a solver time is off grid.
    sample=np.searchsorted(ts.t,1.+TIME+1e-9,side="right")-1
    if np.max(abs(coi[ts.t<.99]-1))>1e-4:
        raise RuntimeError("pre-event COI did not initialize near nominal")
    min_freq=coi[sample].min()
    if not (0.8<min_freq<1.0):
        raise RuntimeError(f"unexpected COI nadir {min_freq}")
    return coi[sample]-1, read[sample,0]-1, read[sample].mean(axis=1)-1, buses,readout


def run(case,pilot,distributed,focused=False,quality=False):
    andes.config_logger(stream_level=logging.ERROR)
    OUT.mkdir(parents=True,exist_ok=True)
    fractions=(.085,.095,.105,.115,.125,.135,.145,.155,.165,.175) if distributed else (
        .015,.025,.035,.045,.055,.065,.075,.085,.10,.12)
    params=[(1.,0,.145 if distributed else .055)] if pilot else [(load_level,site,fraction)
        for load_level in (.94,1.,1.06) for site in range(3)
        for fraction in fractions]
    if focused:
        rng=np.random.default_rng(39191126)
        params=[(scale,site,float(fraction))
                for scale in (.945,.965,.985,1.015,1.035,1.055) for site in (1,2)
                for fraction in np.r_[rng.uniform(.093,.121,4),rng.uniform(.128,.147,4)]]
    if quality:
        rng=np.random.default_rng(39192511)
        params=[(scale,site,float(fraction))
                for scale in (.950,.960,.975,.990,1.010,1.025,1.040,1.050)
                for site in (1,2)
                for fraction in np.r_[rng.uniform(.085,.123,6),rng.uniform(.128,.143,3)]]
    results=[]; rows=[]
    for n,(scale,site,fraction) in enumerate(params,1):
        try:
            coi,single,multi,bus,readout=simulate(case,scale,site,fraction,distributed)
            rows.append([n,scale,site,fraction,bus,";".join(map(str,readout)),coi.min(),"ok",""])
            results.append((coi,single,multi))
        except Exception as error:
            rows.append([n,scale,site,fraction,-1,"",np.nan,"failed",str(error)])
            print(f"{case} scenario {n}: {error}",flush=True)
        if n % 10==0:
            print(f"{case} {n}/{len(params)}, {len(results)} successful",flush=True)
    name=case+("quality" if quality else "focus" if focused else "dist" if distributed else "")+("-pilot" if pilot else "")
    pd.DataFrame(rows,columns=["DesignID","LoadScale","SiteIndex","StepFraction","LoadBus",
        "MeasurementBuses","NadirDeviation_pu","Status","Error"]).to_csv(OUT/f"{name}-design.csv",index=False)
    if results:
        coi,single,multi=np.array(results).transpose(1,0,2)
        np.savez_compressed(OUT/f"{name}-trajectories.npz",coi=coi,single=single,multi=multi)
        paths=[x for x in CASES[case] if x]
        details={"version":andes.__version__,"nominal_Hz":60,"case_files":{
            x:hashlib.sha256(Path(andes.get_case(x)).read_bytes()).hexdigest() for x in paths},
            "events_planned":len(params),"events_completed":len(results),"decision_sample_s":.02,
            "distributed_load":distributed,"focused_design_seed":39191126 if focused else None}
        (OUT/f"{name}-config.json").write_text(json.dumps(details,indent=2)+"\n")
        print(case,"risk events",np.sum(coi.min(axis=1)<-.005),"of",len(coi),flush=True)
    else:
        raise SystemExit(f"{case}: no usable cases")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",choices=CASES)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--distributed",action="store_true")
    parser.add_argument("--focused",action="store_true")
    parser.add_argument("--quality",action="store_true")
    args=parser.parse_args()
    if (args.focused or args.quality) and (args.case!="ieee39" or args.pilot):
        parser.error("--focused and --quality require the full IEEE 39 case")
    run(args.case,args.pilot,args.distributed or args.focused or args.quality,args.focused,args.quality)

