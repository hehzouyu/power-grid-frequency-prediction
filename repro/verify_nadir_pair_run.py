"""Independently verify and report the frozen near-endpoint experiment.

MATLAB summaries are cross-checks only. The reported metrics are rebuilt from
the pair-level rows and verified against member predictions and archived traces.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from analyze_revision import predict
from nadir_pair_predict import FEATURE_COLUMNS, verify

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "out" / "revision"
MATCHED = {"primary_eligible", "true_tie", "below_minimum_gap"}
MEMBER_KEYS = ["PairID", "Condition", "Feature", "Rep"]
NAMES = {"endpoint": "端点", "initial_rate": "端点＋初始变化率", "quadratic": "端点＋二次拟合",
         "fitted_segments": "端点＋分段拟合"}
COLORS = ["#557080", "#168272", "#275897", "#ae512b"]
TITLES = {"clean": "干净", "white30": "30 dB 模拟白噪声"}
matplotlib.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                             "pdf.fonttype": 42, "font.size": 9, "savefig.dpi": 220})


def check(a, b, tolerance=2e-9):
    np.testing.assert_allclose(a, b, rtol=0, atol=tolerance, equal_nan=True)


def direction(values, threshold):
    values = np.asarray(values)
    return (values > threshold).astype(int) - (values < -threshold).astype(int)


def verify_inputs(run, config, ledger, refs, pairs):
    grid = pd.read_csv(ARCHIVE / "grid-index.csv")
    cont = pd.read_csv(ARCHIVE / "continuous-index.csv")
    intervals = pd.read_csv(ARCHIVE.parent / "nadir-intervals" / "scenarios.csv")
    old_match = pd.read_csv(ARCHIVE.parent / "nadir-identifiability" / "matched-scan.csv")
    historical = ["H", "D", "R", "Tg", "d"]
    archive_refs = np.tile([4., 1., .05, .5, 0.], (len(old_match), 1))
    archive_refs[:, 4] = old_match.ReferenceDeficit
    archive_candidates = archive_refs.copy()
    for i, row in enumerate(old_match.itertuples()):
        archive_candidates[i, historical.index(row.Factor)] = row.FactorValue
    archive_candidates[:, 4] = old_match.MatchedDeficit
    old = np.vstack([grid[historical], cont[historical],
                     intervals[historical[:-1] + ["Deficit"]], archive_refs,
                     archive_candidates[np.isfinite(archive_candidates).all(axis=1)]])
    archive_r = grid.loc[grid.Split.isin(["train", "validation", "test"]), "R"].unique()
    assert set(archive_r) == {.04, .05, .06}
    assert refs.ReferenceID.is_unique and ledger.PairID.is_unique
    nref = config["references_per_r_band"] * len(config["r_bands"])
    factors = config["candidate_factors"]
    assert refs.ReferenceID.tolist() == list(range(1, nref+1))
    assert ledger.PairID.tolist() == list(range(1, nref*len(factors)+1))
    assert ledger.ChangedFactor.tolist() == factors*nref
    assert ledger.ReferenceID.tolist() == np.repeat(refs.ReferenceID, len(factors)).tolist()
    reference_rows = refs.set_index("ReferenceID").loc[ledger.ReferenceID]
    check(ledger[historical], reference_rows[historical])
    check(ledger.RBand, reference_rows.RBand)
    for band, (lo, hi) in enumerate(config["r_bands"],1):
        r = refs.loc[refs.RBand.eq(band), "R"]
        assert len(r)==config["references_per_r_band"] and r.between(lo, hi).all()
        assert not np.any((archive_r >= lo) & (archive_r <= hi))
    for col,(low,high) in zip(["H","D","Tg","d"],config["reference_ranges_H_D_Tg_d"],strict=True):
        assert refs[col].between(low, high).all()
    candidates = ["CandidateH", "CandidateD", "CandidateR", "CandidateTg", "CandidateD_pu"]
    for factor,mult in zip(factors,config["candidate_multipliers"],strict=True):
        section = ledger[ledger.ChangedFactor.eq(factor)]
        expected = section[historical[:-1]].to_numpy().copy()
        expected[:,historical.index(factor)] *= mult
        check(section[candidates[:-1]], expected)
    new = np.vstack([refs[historical], ledger.loc[ledger.Status.isin(MATCHED), candidates]])
    assert len(np.unique(new, axis=0)) == len(new)
    for row in new:
        assert (np.max(np.abs(old-row),axis=1)>1e-9).all()
    a = loadmat(run/"trajectories.mat",simplify_cells=True)
    av = a["traceA"][ledger.ReferenceID.to_numpy()-1]
    bv = a["traceB"][ledger.PairID.to_numpy()-1]
    vis, post = a["vis"]-1, a["post"]-1
    t = a["t"]
    assert len(vis)==26 and len(post)==925
    check(t[[vis[0],vis[-1],post[0],post[-1]]],[1,1.5,1.52,20])
    has_b = np.isfinite(bv).all(axis=1)
    has_a = np.isfinite(av).all(axis=1)
    check(ledger.loc[has_a,"EndpointA_Hz"],av[has_a][:,vis[-1]])
    check(ledger.loc[has_b,"EndpointB_Hz"],bv[has_b][:,vis[-1]])
    check(ledger.loc[has_a,"TargetA_Hz"],av[has_a][:,post].min(axis=1))
    check(ledger.loc[has_b,"TargetB_Hz"],bv[has_b][:,post].min(axis=1))
    check(ledger.loc[has_b,"DeltaTarget_Hz"],bv[has_b][:,post].min(axis=1)-av[has_b][:,post].min(axis=1))
    for matrix, present, name in ((av,has_a,"EventA"),(bv,has_b,"EventB")):
        minimum = matrix[present][:,vis[0]:].argmin(axis=1)
        expected = np.select([minimum < 25,minimum == 25],["before","at"],default="after")
        assert np.array_equal(expected,ledger.loc[present,name])
    accepted = ledger.Status.isin(MATCHED)
    endpoint_delta = (ledger.EndpointB_Hz-ledger.EndpointA_Hz).abs()
    assert accepted.equals(ledger.PairID.isin(pairs.PairID))
    assert (endpoint_delta[accepted]<=config["endpoint_tolerance_hz"]).all()
    assert ledger.loc[accepted,"CandidateD_pu"].between(*config["deficit_bracket"]).all()
    abs_gap = ledger.loc[accepted,"DeltaTarget_Hz"].abs()
    label = np.select([abs_gap <= config["tie_hz"], abs_gap < config["minimum_target_gap_hz"]],
                      ["true_tie","below_minimum_gap"],default="primary_eligible")
    assert np.array_equal(label,ledger.loc[accepted,"Status"])
    no_bracket = ledger[ledger.Status.eq("no_bracket")]
    assert (no_bracket.ResidualLow_Hz*no_bracket.ResidualHigh_Hz>0).all()
    assert (endpoint_delta[ledger.Status.eq("outside_endpoint_tolerance")]>config["endpoint_tolerance_hz"]).all()
    rejected = pd.read_csv(run/"excluded-and-failed.csv")
    pd.testing.assert_frame_equal(rejected.reset_index(drop=True),
        ledger[~ledger.Status.eq("primary_eligible")].reset_index(drop=True), check_dtype=False)

    member = pd.read_csv(run/"predictions.csv")
    features = pd.read_csv(run/"features.csv")
    ids = ["ReferenceID","PairID","Condition","Rep","Member"]
    assert not member.duplicated(ids+["Feature"]).any() and not features.duplicated(ids).any()
    assert np.isfinite(member.Prediction_Hz).all()
    assert set(member[ids].itertuples(index=False,name=None))==set(features[ids].itertuples(index=False,name=None))
    assert np.array_equal(member.groupby(ids).Feature.nunique().unique(),[len(config["features"])])
    wide = member.pivot(index=MEMBER_KEYS,columns="Member",values="Prediction_Hz").sort_index()
    p = pairs.set_index(MEMBER_KEYS).sort_index()
    pd.testing.assert_index_equal(wide.index,p.index)
    check(wide[1],p.PredictionA_Hz,1e-12)
    check(wide[2],p.PredictionB_Hz,1e-12)
    check(p.DeltaPrediction_Hz,wide[2]-wide[1],1e-12)
    source = ledger.set_index("PairID").loc[p.index.get_level_values("PairID")]
    for col in ["ReferenceID","RBand","ChangedFactor","Status"]:
        assert np.array_equal(source[col],p[col])
    check(source.DeltaTarget_Hz,p.DeltaTarget_Hz,1e-12)

    frozen = joblib.load(run/"models.joblib")
    keyed_features = features.set_index(ids)
    for name, cols in FEATURE_COLUMNS.items():
        archived = member[member.Feature.eq(name)].set_index(ids)
        replay = predict(frozen[name],keyed_features.loc[archived.index,cols].to_numpy())
        check(archived.Prediction_Hz,replay,1e-10)

    # Rebuild the clean inputs directly from trajectories; verify reference reuse
    # for both conditions, including the noise replicate keys.
    clean = features[features.Condition.eq("clean")]
    trajectory = np.where(clean.Member.to_numpy()[:,None]==1,
        a["traceA"][clean.ReferenceID.to_numpy()-1][:,vis],
        a["traceB"][clean.PairID.to_numpy()-1][:,vis]) - config["nominal_hz"]
    time = np.arange(26)*config["step_s"]
    u = time/time[-1]
    beta = np.linalg.lstsq(np.column_stack([np.ones(26),u,u*u]),trajectory.T,rcond=None)[0]
    slopes = []
    for lo,hi in [(0,8),(8,16),(16,24)]:
        tc = time[lo:hi+1]-time[lo:hi+1].mean()
        slopes.append(trajectory[:,lo:hi+1]@tc/(tc@tc))
    feature_columns = ["Endpoint","InitialRate","Linear","Curvature","Slope1","Slope2","Slope3"]
    calc = np.column_stack([trajectory[:,-1],(trajectory[:,5]-trajectory[:,0])/.1,beta[1:].T,*slopes])
    check(clean[feature_columns],calc,1e-10)
    reference_input_range = features[features.Member.eq(1)].groupby(["ReferenceID","Condition","Rep"])[feature_columns]
    check(reference_input_range.max()-reference_input_range.min(),0,1e-12)


def recalc_pairs(run, config, ledger):
    rows = pd.read_csv(run/"pair-predictions.csv")
    expected = [(cond,feature,r) for cond,reps in (("clean",[0]),("white30",range(1,config["noise_repeats"]+1)))
                for feature in config["features"] for r in reps]
    assert set(rows[["Condition","Feature","Rep"]].itertuples(index=False,name=None))==set(expected)
    n = ledger.Status.isin(MATCHED).sum()
    assert len(rows)==n*len(expected) and not rows.duplicated(MEMBER_KEYS).any()
    true = rows.DeltaTarget_Hz.to_numpy()
    predicted = (rows.PredictionB_Hz-rows.PredictionA_Hz).to_numpy()
    error = np.abs(predicted-true)
    d_true = direction(true,config["tie_hz"])
    d_hat = direction(predicted,config["tie_hz"])
    for old,new in [(rows.DeltaPrediction_Hz,predicted),(rows.AbsDeltaError_Hz,error),
                    (rows.TrueDirection,d_true),(rows.PredictedDirection,d_hat),(rows.DirectionCorrect,d_true==d_hat)]:
        check(old,new,1e-12)
    return rows.assign(Error_mHz=1000*error, TrueDirection=d_true, PredictedDirection=d_hat,
                       Correct=d_true==d_hat, Primary=rows.Status.eq("primary_eligible"))


def metrics(block):
    main = block[block.Primary]
    e = main.Error_mHz.to_numpy()
    return dict(ScorablePairs=block.PairID.nunique(),PrimaryPairs=main.PairID.nunique(),
                DirectionTrials=len(block),ErrorTrials=len(e),DeltaMAE_mHz=e.mean(),
                DeltaRMSE_mHz=np.sqrt(np.mean(e**2)),DeltaP95_mHz=np.quantile(e,.95),
                CorrectDirections=block.Correct.sum(),DirectionAccuracy=block.Correct.mean(),
                PredictedTieTrials=block.PredictedDirection.eq(0).sum(),TrueTieTrials=block.TrueDirection.eq(0).sum(),
                TrueTiePairs=block.loc[block.TrueDirection.eq(0),"PairID"].nunique(),
                PrimaryDirectionAccuracy=main.Correct.mean())


def bootstrap_weights(ref, settings):
    # Matlab rng(seed,'twister') uses the same uniform MT19937 stream for this
    # nonzero seed. MATLAB fills the randi matrix in column-major order.
    draws = int(settings["bootstrap_draws"])
    mt = np.random.RandomState(settings["bootstrap_seed"])
    result = np.zeros((draws,len(ref)),dtype=int)
    for _, band in ref.groupby("RBand",sort=True):
        i = band.ReferenceID.to_numpy()-1
        positions = np.floor(mt.random_sample((draws,len(i)))*len(i)).astype(int)
        sampled = i[positions]
        np.add.at(result,(np.repeat(np.arange(draws),len(i)),sampled.ravel()),1)
    assert (result.sum(axis=1)==len(ref)).all()
    return result


def contrasts(rows, refs, config):
    draws = bootstrap_weights(refs,config)
    results = []
    for cond, block in rows[rows.Primary].groupby("Condition",sort=False):
        spread = block.pivot(index=["ReferenceID","PairID","Rep"],columns="Feature",values="Error_mHz")
        for name in list(NAMES)[1:]:
            for comp in (["endpoint"] if name=="initial_rate" else ["endpoint","initial_rate"]):
                gaps = (spread[name]-spread[comp]).groupby(level=["ReferenceID","PairID"]).mean()
                counts = gaps.groupby(level="ReferenceID").size().reindex(refs.ReferenceID,fill_value=0).to_numpy()
                sums = gaps.groupby(level="ReferenceID").sum().reindex(refs.ReferenceID,fill_value=0).to_numpy()
                denominator = draws@counts
                valid = denominator>0
                values = (draws@sums)[valid]/denominator[valid]
                lo,hi = np.quantile(values,[.025,.975])
                results.append(dict(Condition=cond,Feature=name,Comparator=comp,PrimaryPairs=len(gaps),
                    ClustersWithPrimaryPairs=np.count_nonzero(counts),ChangeMAE_mHz=gaps.mean(),
                    Low95_mHz=lo,High95_mHz=hi,ValidBootstrapDraws=valid.sum()))
    return pd.DataFrame(results)


def report_tables(run, config, ledger, refs, rows):
    summary = pd.DataFrame([dict(Condition=cond,Feature=feat,**metrics(block))
                            for (cond,feat),block in rows.groupby(["Condition","Feature"],sort=False)])
    expected = pd.read_csv(run/"summary.csv")
    keys = ["Condition","Feature"]
    check(summary.set_index(keys).loc[expected.set_index(keys).index,expected.columns[2:]],expected.iloc[:,2:])
    effects = contrasts(rows,refs,config)
    cols = keys+["Comparator"]
    for filename in ["contrast-vs-endpoint.csv","contrast-vs-initial-rate.csv"]:
        expected = pd.read_csv(run/filename).set_index(cols)
        check(effects.set_index(cols).loc[expected.index,expected.columns],expected)
    band_results = pd.DataFrame([dict(RBand=band,Condition=cond,Feature=feat,**metrics(block))
                                 for (band,cond,feat),block in rows.groupby(["RBand","Condition","Feature"],sort=False)])
    # Per-band contrasts are descriptive summaries of the same frozen pairs.
    baseline = band_results[band_results.Feature.eq("endpoint")].set_index(["RBand","Condition"]).DeltaMAE_mHz
    band_results["ChangeVsEndpoint_mHz"] = band_results.DeltaMAE_mHz-baseline.loc[
        pd.MultiIndex.from_frame(band_results[["RBand","Condition"]])].to_numpy()
    flow_rows, failure_rows = [], []
    for axis in ([],["RBand"],["ChangedFactor"],["RBand","ChangedFactor"]):
        blocks = [((),ledger)] if not axis else ledger.groupby(axis,sort=True)
        for index, group in blocks:
            index = index if isinstance(index,tuple) else (index,)
            group_key = {"RBand":"all","ChangedFactor":"all"} | dict(zip(axis,index,strict=True))
            ct = group.Status.value_counts()
            scorable = group.Status.isin(MATCHED).sum()
            geometric = scorable+ct.get("feature_failure",0)
            flow_rows.append(group_key|dict(Attempted=len(group),GeometricMatches=int(geometric),
                MatchFailures=int(len(group)-geometric),MatchFailureRate=(len(group)-geometric)/len(group),
                ScorableMatches=int(scorable),PrimaryPairs=ct.get("primary_eligible",0),
                TrueTies=ct.get("true_tie",0),SmallNonTies=ct.get("below_minimum_gap",0),
                FeatureFailures=ct.get("feature_failure",0),NoBracket=ct.get("no_bracket",0),
                OutsideTolerance=ct.get("outside_endpoint_tolerance",0)))
            statuses = ("reference_simulation_failure","bracket_simulation_failure","no_bracket",
                        "root_failure","historical_collision","candidate_simulation_failure",
                        "outside_endpoint_tolerance","feature_failure","true_tie","below_minimum_gap")
            assert set(ct.index) <= set(statuses) | {"primary_eligible"}
            failure_rows.extend(group_key|dict(Status=status,Count=ct.get(status,0),Attempted=len(group))
                                for status in statuses)
    flow = pd.DataFrame(flow_rows)
    archived = pd.read_csv(run/"pair-flow.csv").set_index("ChangedFactor")
    base_flow = flow[flow.RBand.eq("all")].set_index("ChangedFactor")
    check(base_flow.loc[archived.index,archived.columns],archived)
    archive_status = pd.read_csv(run/"status-counts.csv").set_index(["RBand","ChangedFactor","Status"])
    check(ledger.groupby(["RBand","ChangedFactor","Status"]).size().loc[archive_status.index],
          archive_status.GroupCount)
    summary.to_csv(run/"main-results-verified.csv",index=False)
    effects.to_csv(run/"paired-contrasts-verified.csv",index=False)
    band_results.to_csv(run/"results-by-r-band.csv",index=False)
    flow.to_csv(run/"sample-flow-verified.csv",index=False)
    pd.DataFrame(failure_rows).to_csv(run/"failure-reasons.csv",index=False)
    return effects


def draw(run, rows, effects):
    fig, axes = plt.subplots(2,4,figsize=(11.8,5.8),sharex=True,sharey=True,layout="constrained")
    lim = 1.04*max(rows.DeltaTarget_Hz.abs().max(),rows.DeltaPrediction_Hz.abs().max())*1000
    for i,(cond,title) in enumerate(TITLES.items()):
        for j,(feature,name) in enumerate(NAMES.items()):
            ax = axes[i,j]
            block = rows[(rows.Condition==cond)&(rows.Feature==feature)]
            sub = block[block.Primary]
            small = block[~block.Primary]
            ax.axvspan(-.5,.5,color="#d5e1e8",zorder=0)
            ax.axhspan(-.5,.5,color="#d5e1e8",zorder=0)
            ax.plot([-lim,lim],[-lim,lim],":",color="#777777",lw=.8)
            ax.scatter(small.DeltaTarget_Hz*1000,small.DeltaPrediction_Hz*1000,
                       s=7,marker=".",color="#81868c",alpha=.29,rasterized=True)
            ax.scatter(sub.DeltaTarget_Hz*1000,sub.DeltaPrediction_Hz*1000,
                       s=7,marker=".",color=COLORS[j],alpha=.48 if i==0 else .21,rasterized=True)
            ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect="equal")
            if i==0:
                ax.set_title(name)
            if j==0:
                ax.set_ylabel(title+"\n预测差 / mHz")
            if i==1:
                ax.set_xlabel("真实差 / mHz")
    fig.savefig(run/"fig-pair-deltas.pdf")
    fig.savefig(run/"fig-pair-deltas.png")
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(9.3,3.3),sharex=True,layout="constrained")
    for ax, (comparator,title,names) in zip(axes,[
        ("endpoint","相对端点",list(NAMES)[1:]),
        ("initial_rate","相对端点＋初始变化率",list(NAMES)[2:]),
    ],strict=True):
        for i,(condition,condition_title) in enumerate(TITLES.items()):
            ss = effects[(effects.Comparator==comparator)&(effects.Condition==condition)].set_index("Feature").loc[names]
            x = ss.ChangeMAE_mHz.to_numpy()
            errs = np.stack([x-ss.Low95_mHz.to_numpy(),ss.High95_mHz.to_numpy()-x])
            ax.errorbar(x,np.arange(len(names)) + (-.13 if i==0 else .13),xerr=errs,fmt="o",
                        color=("#275897","#ae512b")[i],capsize=3,label=condition_title)
        ax.axvline(0,color="#697078",ls=":",lw=1)
        ax.set(title=title,yticks=np.arange(len(names)),yticklabels=[NAMES[k] for k in names],
               xlabel="配对差值 MAE 变化 / mHz")
        ax.invert_yaxis()
        ax.grid(axis="x",alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc="outside upper center",ncol=2,fontsize=8)
    fig.savefig(run/"fig-paired-contrasts.pdf")
    fig.savefig(run/"fig-paired-contrasts.png")
    plt.close(fig)


def main(run):
    verify(run)
    config_path = run/"protocol.json"
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert config_digest == "7f9cb24f3da243ee5915ff0412a4c670e93c0f788e8a25536d7a521a6f048b44"
    config = json.loads(config_path.read_text())
    ledger = pd.read_csv(run/"pair-index.csv")
    refs = pd.read_csv(run/"references.csv")
    rows = recalc_pairs(run,config,ledger)
    verify_inputs(run,config,ledger,refs,rows)
    effects = report_tables(run,config,ledger,refs,rows)
    draw(run,rows,effects)
    print("PASS protocol SHA-256 and frozen code, inputs, environment and model checks")
    print("PASS eligible pair keys, historical parameter collision checks and family holdouts")
    print("PASS ODE archive indexing, clean features, frozen predictions, target labels, event timing, and failure ledger")
    print("PASS member-to-pair join, tie definitions, all direction denominators and all error metrics")
    print("PASS exact seeded reference-cluster bootstrap (5,000 replicates) against MATLAB")
    print("PASS formal tables and figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    main(parser.parse_args().run.resolve())
