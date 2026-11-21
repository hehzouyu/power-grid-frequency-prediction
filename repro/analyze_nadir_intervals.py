"""Calibrate and evaluate fixed-cutoff frequency prediction intervals."""
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from analyze_nadir_shape import paired_interval
from analyze_revision import OUT, fit_model, loadmat, plt, predict

DEST = OUT.parent / 'nadir-intervals'
FEATURES = ['Endpoint', 'Linear', 'Curvature']
LEVEL = .90


def conformal_quantile(scores):
    ordered = np.sort(np.asarray(scores))
    rank = math.ceil((len(ordered)+1)*LEVEL)
    if rank > len(ordered):
        raise ValueError('Too few calibration events for the requested coverage')
    return float(ordered[rank-1])


def predictions():
    data = pd.read_csv(DEST/'scenarios.csv')
    assert data.groupby('ID').Split.nunique().eq(1).all()
    assert not data.duplicated(['ID','Scenario']).any()
    assert data.query("Scenario=='mixed'").groupby('Split').size().to_dict() == dict(
        scale=600, calibration=600, test=1200)
    params = ['H','D','R','Tg','Deficit']
    assert not data.query("Scenario=='mixed'").duplicated(params).any()
    test = data.query("Split=='test'").pivot(index='ID',columns='Scenario',values='Target_Hz')
    assert test.notna().all().all() and len(test) == 1200
    np.testing.assert_array_equal(test.mixed, test.jump)
    grid = pd.read_csv(OUT/'grid-index.csv')
    x = loadmat(OUT/'features.mat',simplify_cells=True)['X'][2]
    train, valid, historical_test = [np.flatnonzero(grid.Split.eq(name)) for name in ('train','validation','test')]
    model = fit_model(x, grid.Target_Hz.to_numpy(), train, valid, 'tree')
    prior = pd.read_csv(OUT/'predictions.csv').query("Experiment=='conditional' and Feature=='quadratic' and Model=='tree'")
    np.testing.assert_allclose(predict(model,x[historical_test]),
        prior.set_index('ID').loc[grid.ID.iloc[historical_test], 'Prediction_Hz'], rtol=0, atol=1e-10)
    return data.assign(Point_Hz=predict(model,data[FEATURES].to_numpy()))


def calibrate(data):
    data['AbsResidual_Hz'] = abs(data.Target_Hz-data.Point_Hz)
    develop = data.query("Split=='scale'")
    edges = develop.Roughness_Hz.quantile([1/3,2/3]).to_numpy()
    data['ScaleGroup'] = np.searchsorted(edges,data.Roughness_Hz,side='right')
    scale = data.query("Split=='scale'").groupby('ScaleGroup').AbsResidual_Hz.median().clip(lower=.001)
    data['Scale_Hz'] = data.ScaleGroup.map(scale)
    cal = data.query("Split=='calibration'")
    radius = conformal_quantile(cal.AbsResidual_Hz)
    factor = conformal_quantile(cal.AbsResidual_Hz/cal.Scale_Hz)
    configuration = dict(Nominal=LEVEL, CalibrationN=len(cal), Rank=math.ceil((len(cal)+1)*LEVEL),
                         UniformRadius_Hz=radius, ScaleFactor=factor, Cutpoints_Hz=edges.tolist(),
                         Scales_Hz=scale.to_dict(), Seeds=[47101,47102,47103])
    (DEST/'calibration.json').write_text(json.dumps(configuration,indent=2)+'\n',encoding='utf-8')
    result = data.query("Split=='test'").copy()
    frames = []
    for method, halfwidth in [('constant',radius), ('scaled',factor*result.Scale_Hz)]:
        frame = result.assign(Method=method, Lower_Hz=result.Point_Hz-halfwidth,
                              Upper_Hz=result.Point_Hz+halfwidth, Width_Hz=2*halfwidth)
        frame['LowerMiss'] = frame.Target_Hz < frame.Lower_Hz
        frame['UpperMiss'] = frame.Target_Hz > frame.Upper_Hz
        frame['Covered'] = ~(frame.LowerMiss | frame.UpperMiss)
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def report(output):
    rows, deltas = [], []
    groups = [(name,block) for name,block in output.groupby('Scenario')]
    groups.extend((f'mixed-{int(kind)}',block) for kind,block in output.query("Scenario=='mixed'").groupby('Condition'))
    for scenario, block in groups:
        for method, model in block.groupby('Method'):
            ci = binomtest(int(model.Covered.sum()),len(model)).proportion_ci(method='wilson')
            rows.append(dict(Scenario=scenario, Method=method, N=len(model), Coverage=model.Covered.mean(),
                             Lower95=ci.low, Upper95=ci.high, Width_mHz=model.Width_Hz.mean()*1000,
                             LowerMiss=model.LowerMiss.mean(), UpperMiss=model.UpperMiss.mean()))
        pairs = block.pivot(index='ID',columns='Method',values=['Covered','Width_Hz','LowerMiss'])
        for key, unit in [('Covered',100),('Width_Hz',1000),('LowerMiss',100)]:
            values = pairs[key].astype(float)
            change = (values.scaled-values.constant).to_numpy()*unit
            lo, hi = paired_interval(change)
            deltas.append(dict(Scenario=scenario,Metric=key,ScaledMinusConstant=change.mean(),Low95=lo,High95=hi))
    result = pd.DataFrame(rows)
    result.to_csv(DEST/'summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(DEST/'paired.csv',index=False)
    print(result.to_string(index=False))
    plot(result)


def plot(summary):
    scenarios = ['mixed','mixed-1','mixed-2','mixed-3','jump','cascade']
    labels = ['同分布混合','干净量测','白噪声','相关噪声','噪声突变','两级调频']
    fig,axes = plt.subplots(1,2,figsize=(8.4,3.4),layout='constrained')
    y = np.arange(len(scenarios))
    for name, label, offset in [('constant','统一残差',-.12),('scaled','分尺度残差',.12)]:
        row = summary[summary.Method.eq(name)].set_index('Scenario').loc[scenarios]
        errors = np.vstack([row.Coverage-row.Lower95,row.Upper95-row.Coverage])*100
        axes[0].errorbar(row.Coverage*100,y+offset,xerr=errors,fmt='o',capsize=2,label=label)
        axes[1].scatter(row.Width_mHz,y+offset,label=label)
    axes[0].axvline(90,linestyle='--',color='gray',linewidth=1)
    for ax, xlabel in zip(axes,('经验覆盖率 / %','平均区间宽度 / mHz'),strict=True):
        ax.set(yticks=y,yticklabels=labels,xlabel=xlabel,ylim=(len(y)-.55,-.55))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,frameon=False,loc='outside upper center',ncol=2)
    fig.savefig(DEST/'coverage-width.pdf')
    plt.close(fig)


if __name__ == '__main__':
    output = calibrate(predictions())
    output.to_csv(DEST/'intervals.csv',index=False)
    report(output)
