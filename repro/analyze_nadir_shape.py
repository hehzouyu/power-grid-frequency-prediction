"""Evaluate archived short-window features and pair the existing sensitivity results."""
import argparse

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from analyze_revision import OUT, fit_model, loadmat, metrics, plt, predict

DEST = OUT.parent / 'nadir-shape'
FEATURES = ('endpoint', 'quadratic', 'segmented', 'fitted_segments')


def paired_interval(gain):
    indices = np.random.default_rng(613).integers(0, len(gain), (3000, len(gain)))
    return np.quantile(gain[indices].mean(axis=1), [.025, .975])


def compare_window_gains():
    data = pd.read_csv(DEST / 'window-predictions.csv')
    assert data.ID.nunique() == 232
    assert data.groupby('ID').Target_Hz.nunique().eq(1).all()
    errors = data.assign(error=abs(data.Prediction_Hz - data.Target_Hz)).pivot(
        index=['ID', 'Window_s'], columns='Feature', values='error')
    assert len(errors) == 232 * 3 and not errors.isna().any().any()
    gains = errors[errors.columns.drop('endpoint')].rsub(errors.endpoint, axis=0)
    rows = []
    for short, long in ((.10, .50), (.10, .24), (.24, .50)):
        short_gains, long_gains = (gains.xs(width, level='Window_s') for width in (short, long))
        assert short_gains.index.equals(long_gains.index)
        change = long_gains - short_gains
        for feature in change:
            gain = change[feature].to_numpy()
            low, high = paired_interval(gain)
            rows.append(dict(Feature=feature, ShortWindow_s=short, LongWindow_s=long, N=len(gain),
                             GainChange_mHz=1000*gain.mean(), Low95_mHz=1000*low, High95_mHz=1000*high))
    pd.DataFrame(rows).to_csv(DEST / 'cross-window-gain.csv', index=False)


def features_from_archive(t, frequency):
    """Evaluate the same least-squares and segment equations as the MATLAB export."""
    blocks = []
    for last, edges in [(5, (0, 2, 4, 5)), (12, (0, 4, 8, 12)), (25, (0, 8, 16, 24))]:
        x = frequency[:, :last+1] - 50
        u = t[:last+1] / t[last]
        coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(u)), u, u*u]), x.T, rcond=None)[0]
        difference, fitted = [], []
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            segment = t[start:end+1]
            centered = segment - segment.mean()
            difference.append((x[:, end] - x[:, start]) / (t[end]-t[start]))
            fitted.append(x[:, start:end+1] @ centered / (centered @ centered))
        endpoint = x[:, -1]
        blocks.append([endpoint, np.column_stack([endpoint, coefficients[1:].T]),
                       np.column_stack([endpoint, *difference]), np.column_stack([endpoint, *fitted])])
    return blocks


def window_check(archive, grid, predictions):
    t, f = archive['t'], archive['F']
    start, cut = np.flatnonzero(np.isclose(t, 1)).item(), np.flatnonzero(np.isclose(t, 1.5)).item()
    target = grid.Target_Hz.to_numpy()
    np.testing.assert_allclose(f[:, cut+1:].min(axis=1), target, rtol=0, atol=1e-12)
    nadir = f[:, start:].argmin(axis=1) + start
    events = np.select([nadir < cut, nadir == cut], ['before', 'at'], default='after')
    np.testing.assert_array_equal(grid.Event, events)
    train, valid, test = [np.flatnonzero(grid.Split.eq(split)) for split in ('train', 'validation', 'test')]
    assert len(test) == 232 and cut-start+1 == 26
    blocks = features_from_archive(t[start:] - t[start], f[:, start:])
    matlab = DEST / 'windows-matlab.mat'
    if matlab.exists():
        native = loadmat(matlab, simplify_cells=True)['blocks']
        for row, checked in zip(native, blocks, strict=True):
            for a, b in zip(row, checked, strict=True):
                np.testing.assert_allclose(a, b, atol=1e-11, rtol=0)
        blocks = native
    saved = loadmat(OUT / 'features.mat', simplify_cells=True)['X']
    for block, slot in zip(blocks[-1], (0, 2, 3, 4), strict=True):
        np.testing.assert_allclose(block, saved[slot], atol=1e-11, rtol=0)

    rows, records = [], []
    for width, window in zip((.10, .24, .50), blocks, strict=True):
        for name, values in zip(FEATURES, window, strict=True):
            values = np.asarray(values).reshape(len(grid), -1)
            result = predict(fit_model(values, target, train, valid, 'tree'), values[test])
            if width == .50:
                old = predictions.query("Experiment=='conditional' and Model=='tree' and Feature==@name")
                old = old.set_index('ID').loc[grid.ID.iloc[test], 'Prediction_Hz'].to_numpy()
                np.testing.assert_allclose(metrics(target[test], result)['rmse'], metrics(target[test], old)['rmse'],
                                           atol=5e-5, rtol=0)
                records.append(dict(Feature=name, max_prediction_delta_Hz=abs(result-old).max()))
            err = abs(result - target[test])
            if name == 'endpoint':
                baseline = err
            gain = baseline - err
            low, high = paired_interval(gain)
            rows.append(dict(Window_s=width, Feature=name, **metrics(target[test], result),
                             Gain_MAE_Hz=gain.mean(), Low95_Hz=low, High95_Hz=high))
            yield pd.DataFrame(dict(Window_s=width, Feature=name, ID=grid.ID.iloc[test],
                                    Target_Hz=target[test], Prediction_Hz=result))
    scores = pd.DataFrame(rows)
    scores.to_csv(DEST / 'window-summary.csv', index=False)
    pd.DataFrame(records).to_csv(DEST / 'anchor-check.csv', index=False)
    matrix = scores.pivot(index='Window_s', columns='Feature', values='Gain_MAE_Hz')[list(FEATURES[1:])] * 1000
    fig, ax = plt.subplots(figsize=(6.2, 2.6), layout='constrained')
    image = ax.imshow(matrix, cmap='Blues', aspect='auto')
    for row, col in np.ndindex(matrix.shape):
        value = matrix.iloc[row, col]
        ax.text(col, row, f'{value:.1f}', ha='center', va='center', color='white' if value > 12 else 'black')
    ax.set(xticks=range(3), xticklabels=['二次拟合', '分段差分', '分段拟合'],
           yticks=range(3), yticklabels=['0.10', '0.24', '0.50'], ylabel='实际末样本时刻 / s')
    fig.colorbar(image, ax=ax, label='相对端点的 MAE 降幅 / mHz')
    fig.savefig(DEST / 'window-gain.pdf')
    plt.close(fig)


def existing_evidence(data, archive, grid, continuous):
    wide = data.query("Experiment=='noise' and Model=='tree'").pivot(index=['Level','ID','Rep'],
                                                                      columns='Feature', values='AbsoluteError')
    rows = []
    for level, group in wide.groupby(level='Level'):
        for base, model in [('endpoint','quadratic'), ('endpoint','fitted_segments'),
                            ('segmented','fitted_segments'), ('endpoint','vmd')]:
            gains = (group[base]-group[model]).groupby(level='ID').mean().to_numpy()
            lo, hi = paired_interval(gains)
            rows.append(dict(Level=level, Baseline=base, Feature=model, N=len(gains),
                             Gain_MAE_Hz=gains.mean(), Low95_Hz=lo, High95_Hz=hi))
    pd.DataFrame(rows).to_csv(DEST / 'paired-white-noise.csv', index=False)
    global_min = archive['grid']['F'][:, archive['grid']['t'] >= 1].min(axis=1)
    boundary = data.query("Experiment=='full' and (Model=='tree' or Model=='baseline')").copy()
    boundary['GlobalMinimum_Hz'] = pd.Series(global_min, index=grid.ID).loc[boundary.ID].to_numpy()
    boundary['TargetGap_Hz'] = boundary.Target_Hz - boundary.GlobalMinimum_Hz
    boundary['RegressionError_Hz'] = boundary.Prediction_Hz - boundary.Target_Hz
    boundary.to_csv(DEST / 'boundary-decomposition.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.5), sharey=True, layout='constrained')
    fitted = boundary.query("Feature=='fitted_segments' and Model=='tree'")
    for event, title, ax in zip(('before','at','after'), ('截止前','截止时','截止后'), axes, strict=True):
        row = fitted[fitted.Event.eq(event)].sort_values('AbsoluteError').iloc[-1]
        trace = archive['grid']['F'][int(row.ID)-1]
        time = archive['grid']['t']-1
        show = (time >= 0) & (time <= 2)
        ax.plot(time[show], trace[show], linewidth=1.15)
        ax.axvspan(0,.5,color='#e8eef2')
        ax.scatter([.5], [row.Prediction_Hz], marker='x', color='#963a2d')
        ax.axhline(row.Target_Hz, color='#326a50', linewidth=.8, linestyle='--')
        ax.set(title=f'{title}，场景 {row.ID}', xlabel='扰动后时间 / s')
    axes[0].set_ylabel('频率 / Hz')
    fig.savefig(DEST/'boundary-cases.pdf')
    plt.close(fig)
    cascade = pd.read_csv(OUT / 'cascade-index.csv')
    keys = ['ID','H','D','R','Tg','d']
    assert continuous[keys].equals(cascade[keys])
    structure = data.query("Experiment in ['continuous','cascade'] and Model=='tree'").pivot(
        index=['ID','Feature'], columns='Experiment', values=['AbsoluteError','Target_Hz'])
    change = pd.DataFrame({name: structure[name]['cascade'] - structure[name]['continuous']
                           for name in ('AbsoluteError','Target_Hz')}).reset_index()
    change = change.merge(continuous[keys], on='ID', validate='many_to_one')
    time = archive['t']
    visible = (time >= 1) & (time <= 1.5)
    delta = archive['Fshift'][:, visible] - archive['F'][:, visible]
    chord = (time[visible]-1)/.5
    descriptors = pd.DataFrame(dict(ID=continuous.ID, EndpointDelta_Hz=delta[:,-1],
                                    ChordResidualRMS_Hz=np.sqrt(np.mean((delta-delta[:,[-1]]*chord)**2,axis=1))))
    change = change.merge(descriptors, on='ID', validate='many_to_one')
    change.to_csv(DEST / 'paired-structure.csv', index=False)


def mechanism():
    t = np.arange(951)*.02
    def simulate(tg, deficit, time):
        def rhs(_, y):
            x, p = y
            return [(-deficit-x+p)/8, (-p+np.clip(-x/.05, 0, .1))/tg]
        return solve_ivp(rhs, (0, time[-1]), [0, 0], t_eval=time, rtol=1e-11, atol=1e-13).y
    first = simulate(.2, .035, t)
    other = simulate(.7, .035, t)
    deficit = .035 * first[0,25]/other[0,25]
    second = simulate(.7, deficit, t)
    assert .02 <= deficit <= .08 and abs(first[0,25]-second[0,25])*50 < 1e-6
    rows, check = [], []
    for tg, d, trace in ((.2,.035,first), (.7,deficit,second)):
        rows.append(pd.DataFrame(dict(Time_s=t, Tg_s=tg, Deficit_pu=d, Frequency_Hz=50*(1+trace[0]),
                                      GovernorPower_pu=trace[1], Limited=-trace[0]/.05 >= .1)))
        expected = np.array([-d/8, d/64, -d/512+d/(64*.05*tg)])
        for h in (.005, .0025, .00125):
            early = simulate(tg, d, np.arange(4)*h)[0]
            values = [np.diff(early, n=k)[0]/h**k for k in (1,2,3)]
            check.extend(dict(Tg_s=tg, Step_s=h, Order=k, FiniteDifference=v, Analytical=q)
                         for k, (v, q) in enumerate(zip(values, expected, strict=True), 1))
    matched = pd.concat(rows)
    matched.to_csv(DEST/'matched-endpoint.csv', index=False)
    pd.DataFrame(check).to_csv(DEST/'derivative-check.csv', index=False)
    fig, (frequency, power) = plt.subplots(2, 1, figsize=(6.5, 4.1), sharex=True, layout='constrained')
    for (tg, d), block in matched[matched.Time_s <= 3].groupby(['Tg_s','Deficit_pu']):
        line = frequency.plot(block.Time_s, block.Frequency_Hz, label=f'$T_g$={tg:g} s, $d$={d:.5f}')[0]
        power.plot(block.Time_s, block.GovernorPower_pu, color=line.get_color())
    for ax in (frequency, power):
        ax.axvline(.5, color='gray', linestyle='--', linewidth=1)
    frequency.set(ylabel='频率 / Hz')
    frequency.legend(frameon=False)
    power.set(xlabel='扰动后时间 / s', ylabel='调频增功 / p.u.')
    fig.savefig(DEST/'matched-endpoint.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-contrast-only', action='store_true')
    if parser.parse_args().window_contrast_only:
        compare_window_gains()
        raise SystemExit(0)
    DEST.mkdir(parents=True, exist_ok=True)
    archive = loadmat(OUT/'trajectories.mat', simplify_cells=True)
    grid = pd.read_csv(OUT/'grid-index.csv')
    data = pd.read_csv(OUT/'predictions.csv')
    data['AbsoluteError'] = abs(data.Prediction_Hz-data.Target_Hz)
    pd.concat(window_check(archive['grid'], grid, data)).to_csv(DEST/'window-predictions.csv', index=False)
    compare_window_gains()
    existing_evidence(data, archive, grid, pd.read_csv(OUT/'continuous-index.csv'))
    mechanism()
