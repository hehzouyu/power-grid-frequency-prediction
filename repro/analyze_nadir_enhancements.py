"""Score MATLAB-generated stress inputs with the original frozen regressors."""
from time import perf_counter_ns

import numpy as np
import pandas as pd

from analyze_nadir_shape import FEATURES, paired_interval
from analyze_revision import LABELS, OUT, fit_model, loadmat, metrics, plt, predict

DEST = OUT.parent / 'nadir-enhancements'
CONDITIONS = ('white', 'ar1', 'jump')


def score():
    grid = pd.read_csv(OUT/'grid-index.csv')
    packet = loadmat(DEST/'noise-features.mat', simplify_cells=True)
    saved = loadmat(OUT/'features.mat', simplify_cells=True)['X']
    train, valid, test = [np.flatnonzero(grid.Split.eq(split)) for split in ('train','validation','test')]
    np.testing.assert_array_equal(packet['ids'], test + 1)
    target = grid.Target_Hz.to_numpy()
    timings = pd.read_csv(DEST/'feature-timing-ms.csv', header=None).to_numpy()
    rows, scores, timing, paired = [], [], [], []
    original = pd.read_csv(OUT/'predictions.csv').query("Experiment=='conditional' and Model=='tree'")
    for k, (name, slot) in enumerate(zip(FEATURES, (0,2,3,4), strict=True)):
        feature = np.asarray(saved[slot]).reshape(len(grid), -1)
        estimator = fit_model(feature, target, train, valid, 'tree')
        np.testing.assert_allclose(predict(estimator,feature[test]), original[original.Feature.eq(name)]
                                   .set_index('ID').loc[grid.ID.iloc[test], 'Prediction_Hz'], atol=1e-10, rtol=0)
        point = feature[test[:1]]
        for _ in range(10):
            predict(estimator, point)
        prediction_times = []
        for _ in range(200):
            started = perf_counter_ns()
            predict(estimator, point)
            prediction_times.append((perf_counter_ns()-started)/1e6)
        for step, values in [('feature',timings[:,k]),('prediction',prediction_times)]:
            timing.extend(dict(Feature=name, Stage=step, Run=i, Time_ms=value)
                          for i,value in enumerate(values,1))
        for j, condition in enumerate(CONDITIONS):
            for rep in range(10):
                x = np.asarray(packet['blocks'][rep,j,k]).reshape(len(test),-1)
                y = predict(estimator,x)
                rows.append(pd.DataFrame(dict(Feature=name,Condition=condition,Rep=rep+1,
                    ID=grid.ID.iloc[test],Target_Hz=target[test],Prediction_Hz=y,AbsoluteError=abs(y-target[test]))))
                scores.append(dict(Feature=name, Condition=condition, Rep=rep+1, **metrics(target[test],y)))
    events = pd.concat(rows,ignore_index=True)
    assert len(events)==232*10*3*4
    events.to_csv(DEST/'noise-predictions.csv',index=False)
    pd.DataFrame(scores).groupby(['Condition','Feature'],as_index=False).mean(numeric_only=True).drop(
        columns='Rep').to_csv(DEST/'noise-summary.csv',index=False)
    wide = events.pivot(index=['Condition','ID','Rep'],columns='Feature',values='AbsoluteError')
    for condition, block in wide.groupby(level='Condition'):
        for base, model in [('segmented','fitted_segments'),('endpoint','fitted_segments'),('endpoint','quadratic')]:
            gain = (block[base]-block[model]).groupby(level='ID').mean().to_numpy()
            low,high = paired_interval(gain)
            paired.append(dict(Condition=condition,Baseline=base,Feature=model,N=len(gain),
                               Gain_mHz=gain.mean()*1e3,Low95_mHz=low*1e3,High95_mHz=high*1e3))
    pd.DataFrame(paired).to_csv(DEST/'noise-paired.csv',index=False)
    calls = pd.DataFrame(timing)
    calls.to_csv(DEST/'timing-calls.csv',index=False)
    calls.groupby(['Feature','Stage']).Time_ms.agg(Median_ms='median',P95_ms=lambda x:x.quantile(.95)).to_csv(
        DEST/'timing-summary.csv')
    pd.DataFrame(packet['variance'],columns=CONDITIONS).to_csv(DEST/'noise-power-check.csv',index=False)


def plots():
    scan = pd.read_csv(DEST/'endpoint-scan.csv')
    success = scan.dropna(subset='MatchedDeficit_pu')
    assert success.EndpointError_Hz.abs().max() < 1e-6
    fig, ax = plt.subplots(figsize=(5.8,2.8),layout='constrained')
    for d, part in success.groupby('BaseDeficit_pu'):
        ax.plot(part.Tg_s,(part.MatchedTarget_Hz-part.BaseTarget_Hz)*1000,marker='o',label=f'$d_0$={d:g} p.u.')
    ax.set(xlabel='匹配轨迹调频时间常数 / s',ylabel='截止后最低频率变化 / mHz')
    ax.legend(frameon=False)
    fig.savefig(DEST/'endpoint-scan.pdf')
    plt.close(fig)

    data = pd.read_csv(DEST/'structure-states.csv').query('Time_s<=4')
    chosen = pd.read_csv(DEST/'structure-selected.csv')
    fig,axes = plt.subplots(2,3,figsize=(9,4),sharex=True,layout='constrained')
    for case,column in zip(chosen.ID,axes.T,strict=True):
        for cascade,block in data[data.ID.eq(case)].groupby('Cascade'):
            label = '两级' if cascade else '单级'
            line, = column[0].plot(block.Time_s,block.Frequency_Hz,label=label)
            column[1].plot(block.Time_s,block.GovernorPower_pu,color=line.get_color())
            column[1].plot(block.Time_s,block.Command_pu,color=line.get_color(),linestyle=':',alpha=.65)
        column[0].set_title(f'场景 {case}')
        column[1].set_xlabel('扰动后时间 / s')
        for ax in column:
            ax.axvline(.5,color='gray',linestyle='--',linewidth=.7)
    axes[0,0].set_ylabel('频率 / Hz')
    axes[1,0].set_ylabel('指令与增功 / p.u.')
    axes[0,0].legend(frameon=False)
    fig.savefig(DEST/'structure-states.pdf')
    plt.close(fig)

    scores = pd.read_csv(DEST/'noise-summary.csv')
    fig,axes = plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for feature,part in scores.groupby('Feature'):
        part = part.set_index('Condition').loc[list(CONDITIONS)]
        for axis,column in zip(axes,('mae','p95'),strict=True):
            axis.plot(range(3),part[column]*1000,marker='o',label=LABELS[feature])
    for axis,label in zip(axes,('MAE / mHz','绝对误差 P95 / mHz'),strict=True):
        axis.set(xticks=range(3),xticklabels=['白噪声','相关噪声','突变噪声'],ylabel=label)
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(DEST/'noise-comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    score()
    plots()
