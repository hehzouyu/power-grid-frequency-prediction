"""Recompute the evidence reported before the independent near-endpoint study."""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_nadir_intervals as intervals
from analyze_nadir_identifiability import representations
from analyze_nadir_shape import FEATURES, paired_interval
from analyze_revision import OUT, fit_model, loadmat, metrics, predict

root = OUT.parent
revision = pd.read_csv(OUT / 'predictions.csv')
grid = pd.read_csv(OUT / 'grid-index.csv')
shape = root / 'nadir-shape'
identity = root / 'nadir-identifiability'
enhance = root / 'nadir-enhancements'


def compare(actual, expected, tol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=0)


scan = pd.read_csv(identity / 'matched-scan.csv')
valid = scan.MatchedDeficit.notna() & (scan.EndpointError_Hz.abs() < 1e-6)
assert len(scan) == 36 and valid.sum() == 35 and (scan.MatchedDeficit.notna() == valid).all()
gaps = scan[valid].assign(gap=scan.TargetDifference_Hz.abs() * 1000).groupby('Factor').gap.agg(['size','min','max'])
print('MATCHES', int(valid.sum()), len(scan), 'max endpoint Hz', scan.loc[valid,'EndpointError_Hz'].abs().max())
print(gaps.to_string())

w = pd.read_csv(shape / 'window-predictions.csv')
assert w.ID.nunique() == 232 and w.groupby('ID').Target_Hz.nunique().eq(1).all()
p = w.assign(e=(w.Prediction_Hz-w.Target_Hz).abs()).pivot(index=['ID','Window_s'], columns='Feature', values='e')
g = p[['quadratic','segmented','fitted_segments']].rsub(p.endpoint, axis=0)
assert len(p) == 696 and p.notna().all().all()
diff = (g.xs(.50, level='Window_s')-g.xs(.10, level='Window_s')).quadratic.to_numpy()
actual = 1000*np.r_[diff.mean(),paired_interval(diff)]
saved = pd.read_csv(shape/'cross-window-gain.csv').query("Feature=='quadratic' and ShortWindow_s==.10 and LongWindow_s==.50")
compare(actual, saved[['GainChange_mHz','Low95_mHz','High95_mHz']].to_numpy()[0])
for width, block in w.groupby('Window_s'):
    for feature, sub in block.groupby('Feature'):
        old = pd.read_csv(shape/'window-summary.csv').query('Window_s==@width and Feature==@feature').iloc[0]
        current = metrics(sub.Target_Hz.to_numpy(),sub.Prediction_Hz.to_numpy())
        compare([current[k] for k in ('rmse','mae','p95','r2')], old[['rmse','mae','p95','r2']].to_numpy(dtype=float))
print('WINDOWS', actual, 'MAE gains mHz', (1000*g.groupby('Window_s').mean()).to_dict())

# Repeat inference on all historical test cases and recalibrate from development/calibration rows.
old_intervals = pd.read_csv(intervals.DEST/'intervals.csv').sort_values(['Scenario','ID','Method'])
with tempfile.TemporaryDirectory(dir=root) as scratch:
    predicted = intervals.predictions()
    intervals.DEST = Path(scratch)
    fresh = intervals.calibrate(predicted).sort_values(['Scenario','ID','Method'])
    assert fresh[['Scenario','ID','Method']].reset_index(drop=True).equals(
        old_intervals[['Scenario','ID','Method']].reset_index(drop=True))
    for col in ('Point_Hz','Lower_Hz','Upper_Hz'):
        compare(fresh[col], old_intervals[col])
for condition, block in fresh.groupby('Scenario'):
    for method, model in block.groupby('Method'):
        covered = model.Target_Hz.between(model.Lower_Hz,model.Upper_Hz)
        assert np.array_equal(covered, model.Covered)
        print('INTERVAL',condition, method, len(model), int(covered.sum()),
              (model.Upper_Hz-model.Lower_Hz).mean()*1000,
              (model.Target_Hz < model.Lower_Hz).mean(), (model.Target_Hz > model.Upper_Hz).mean())

packet = loadmat(identity/'sensitivities.mat',simplify_cells=True)
use = ~packet['limited'].astype(bool)
J = packet['jacobian'][use]
G = packet['target_jacobian'][use]
noise = np.sqrt(np.mean((packet['observed'][use]-50)**2,axis=1))
for snr in (30,40):
    for name, L in representations(26,(0,8,16,24)).items():
        _, s, v = np.linalg.svd(L,full_matrices=False)
        _, singular, directions = np.linalg.svd(v[s>s[0]*1e-12]@J[:,:,:,1],full_matrices=False)
        keep = .1*singular/(noise[:,None]*10**(-snr/20)) >= 1
        share = ((np.einsum('ijk,ik->ij',directions,G[:,:,1])/np.linalg.norm(G[:,:,1],axis=1)[:,None])**2*keep).sum(axis=1)
        print('DIRECTIONS',snr,name,int(use.sum()),int((keep.sum(axis=1)>=2).sum()),np.median(share))
print('SCALING',np.isfinite(packet['scaling_error']).sum(),np.nanmax(packet['scaling_error']))
print('FD',np.median(np.linalg.norm(J[:,:,:,1]-J[:,:,:,0],axis=(1,2))/np.linalg.norm(J[:,:,:,1],axis=(1,2))),
      np.median(np.linalg.norm(G[:,:,1]-G[:,:,0],axis=1)/np.linalg.norm(G[:,:,1],axis=1)))

n = pd.read_csv(enhance/'noise-predictions.csv')
assert len(n)==232*10*3*4
p = n.assign(e=(n.Prediction_Hz-n.Target_Hz).abs()).pivot(index=['Condition','ID','Rep'],columns='Feature',values='e')
for condition, b in p.groupby(level='Condition'):
    gain = (b.segmented-b.fitted_segments).groupby(level='ID').mean().to_numpy()*1000
    old = pd.read_csv(enhance/'noise-paired.csv').query("Condition==@condition and Baseline=='segmented' and Feature=='fitted_segments'")
    actual = np.r_[gain.mean(),paired_interval(gain)]
    compare(actual, old[['Gain_mHz','Low95_mHz','High95_mHz']].to_numpy()[0])
    print('NOISE FIT VS DIFF', condition, actual, 'MAE by feature', (1000*b.mean()).to_dict())

base = loadmat(OUT/'trajectories.mat',simplify_cells=True)['grid']
start = np.flatnonzero(np.isclose(base['t'],1)).item()
cut = np.flatnonzero(np.isclose(base['t'],1.5)).item()
f = base['F']
compare(f[:,cut+1:].min(axis=1),grid.Target_Hz,1e-12)
nadir = f[:,start:].argmin(axis=1)+start
event = np.select([nadir<cut,nadir==cut],['before','at'],default='after')
assert np.array_equal(event,grid.Event)
boundary = pd.read_csv(shape/'boundary-decomposition.csv')
for name, part in boundary.query("Feature=='fitted_segments' or Feature=='persistence'").groupby(['Feature','Event']):
    original = revision.query("Experiment=='full'").set_index(['Feature','Model','ID'])
    keys = pd.MultiIndex.from_frame(part[['Feature','Model','ID']])
    compare(part.Prediction_Hz,original.loc[keys,'Prediction_Hz'])
    low = f[part.ID.to_numpy()-1,start:].min(axis=1)
    gap = part.Target_Hz-low
    err = part.Prediction_Hz-part.Target_Hz
    compare(gap,part.TargetGap_Hz)
    compare(err,part.RegressionError_Hz)
    print('BOUNDARY',name,len(part),gap.mean()*1000,np.sqrt(np.mean(err**2))*1000)

hist = np.flatnonzero(grid.Split.eq('test'))
train, valid = (np.flatnonzero(grid.Split.eq(c)) for c in ('train','validation'))
x = loadmat(OUT/'features.mat',simplify_cells=True)['X']
for name, slot in zip(('endpoint','initial_rate','quadratic','fitted_segments'),(0,1,2,4),strict=True):
    a = np.asarray(x[slot]).reshape(len(grid),-1)
    pred = predict(fit_model(a,grid.Target_Hz.to_numpy(),train,valid,'tree'), a[hist])
    saved = revision.query("Experiment=='conditional' and Model=='tree' and Feature==@name").set_index('ID')
    compare(pred,saved.loc[grid.ID.iloc[hist],'Prediction_Hz'])
    print('ANCHOR',name,len(hist),np.max(abs(pred-saved.loc[grid.ID.iloc[hist],'Prediction_Hz'])),
          np.sqrt(np.mean((pred-grid.Target_Hz.iloc[hist])**2)))

# Recreate the equal-power stress design from the stored noise packet and
# independently recompute white-noise and structure contrasts from raw predictions.
noise_packet = loadmat(enhance/'noise-features.mat',simplify_cells=True)
compare(noise_packet['variance'], pd.read_csv(enhance/'noise-power-check.csv').to_numpy())
weights = np.ones(26)
weights[13:] = 3
weights /= np.sqrt(np.mean(weights**2))
assert abs(np.mean(weights**2)-1)<1e-12 and abs(.8**2+.6**2-1)<1e-12
print('NOISE realized power factors', noise_packet['variance'].mean(axis=0))

with_noise = revision.query("Experiment=='noise' and Model=='tree'").copy()
with_noise['e'] = (with_noise.Prediction_Hz-with_noise.Target_Hz).abs()
p = with_noise.pivot(index=['Level','ID','Rep'],columns='Feature',values='e')
old = pd.read_csv(shape/'paired-white-noise.csv')
for level, b in p.groupby(level='Level'):
    for base_name, name in [('segmented','fitted_segments'),('endpoint','quadratic'),('endpoint','fitted_segments')]:
        contrast = (b[base_name]-b[name]).groupby(level='ID').mean().to_numpy()
        archived = old.query('Level==@level and Baseline==@base_name and Feature==@name')
        compare(np.r_[contrast.mean(),paired_interval(contrast)],
                archived[['Gain_MAE_Hz','Low95_Hz','High95_Hz']].to_numpy()[0])
        if float(level)==20:
            print('ORIGINAL WHITE 20',base_name,name,1000*np.r_[contrast.mean(),paired_interval(contrast)])

structure = revision.query("Experiment in ['continuous','cascade'] and Feature=='fitted_segments' and Model=='tree'").copy()
structure['e'] = (structure.Prediction_Hz-structure.Target_Hz).abs()
p = structure.pivot(index='ID',columns='Experiment',values=['Target_Hz','e'])
assert len(p)==500 and p.notna().all().all()
q = pd.read_csv(shape/'paired-structure.csv').query("Feature=='fitted_segments'").set_index('ID')
target_delta = p.Target_Hz.cascade-p.Target_Hz.continuous
error_delta = p.e.cascade-p.e.continuous
compare(target_delta,q.loc[p.index,'Target_Hz'])
compare(error_delta,q.loc[p.index,'AbsoluteError'])
print('STRUCTURE 500 target and absolute-error changes mHz',1000*target_delta.mean(),1000*error_delta.mean())
