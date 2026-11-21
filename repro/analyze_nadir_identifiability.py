"""Summarize matched endpoints and noise-whitened local parameter sensitivities."""
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_revision import loadmat, plt

DEST = Path(__file__).resolve().parent/'out'/'nadir-identifiability'
WINDOWS = ((.10,6,(0,2,4,5)),(.24,13,(0,4,8,12)),(.50,26,(0,8,16,24)))
NAMES = {'endpoint':'端点','quadratic':'端点＋二次拟合','segment':'端点＋分段拟合','full':'完整窗口'}


def representations(n, edges):
    endpoint = np.zeros(n)
    endpoint[-1] = 1
    u = np.linspace(0,1,n)
    coefficients = np.linalg.pinv(np.column_stack([np.ones(n),u,u*u]))[1:]
    slopes = np.zeros((3,n))
    for j,(start,stop) in enumerate(zip(edges[:-1],edges[1:],strict=True)):
        t = .02*np.arange(start,stop+1)
        centered = t-t.mean()
        slopes[j,start:stop+1] = centered/(centered@centered)
    return dict(endpoint=endpoint[None,:],quadratic=np.vstack([endpoint,coefficients]),
                segment=np.vstack([endpoint,slopes]),full=np.eye(n))


def analyze():
    source = loadmat(DEST/'sensitivities.mat',simplify_cells=True)
    ids = source['ids']
    jac = source['jacobian']
    assert jac.shape == (60,26,5,2)
    assert np.isfinite(jac).all() and len(np.unique(ids)) == 60
    primary = ~source['limited'].astype(bool)
    noise_ref = np.sqrt(np.mean((source['observed']-50)**2,axis=1))
    step_error = np.linalg.norm(jac[:,:,:,1]-jac[:,:,:,0],axis=(1,2))/np.linalg.norm(jac[:,:,:,1],axis=(1,2))
    target = source['target_jacobian']
    target_norm = np.linalg.norm(target[:,:,1],axis=1)
    target_step_error = np.linalg.norm(target[:,:,1]-target[:,:,0],axis=1)/target_norm
    null = np.linalg.norm(jac[:,:,:,1]@np.array([1,1,-1,0,1]),axis=1)/np.linalg.norm(jac[:,:,:,1],axis=(1,2))
    pd.DataFrame(dict(ID=ids,Unsaturated=primary,StepRelError=step_error,TargetStepRelError=target_step_error,
                      ScaleDirectionRelResidual=null,
                      ScaleReplayMax_Hz=source['scaling_error'])).to_csv(DEST/'numerical-checks.csv',index=False)
    rows = []
    for width,n,edges in WINDOWS:
        for name,L in representations(n,edges).items():
            _, values, vectors = np.linalg.svd(L,full_matrices=False)
            basis = vectors[values > values[0]*1e-12]
            _, singular, directions = np.linalg.svd(basis@jac[:,:n,:,1],full_matrices=False)
            target_projection = np.einsum('ijk,ik->ij',directions,target[:,:,1])/target_norm[:,None]
            for snr in (20,30,40):
                scaled = .1*singular/(noise_ref[:,None]*10**(-snr/20))
                rank = (scaled>=1).sum(axis=1)
                visible_share = (target_projection**2*(scaled>=1)).sum(axis=1)
                rows.extend(dict(ID=case,Window_s=width,Representation=name,SNR_dB=snr,Directions=k,
                                 Unsaturated=accepted,TargetGradientShare=share,
                                 **{f'S{j}':v for j,v in enumerate(spectrum,1)})
                            for case,k,accepted,spectrum,share in zip(ids,rank,primary,scaled,visible_share,strict=True))
    table = pd.DataFrame(rows)
    table.to_csv(DEST/'local-spectra.csv',index=False)
    summary = table[table.Unsaturated].groupby(['Window_s','SNR_dB','Representation']).Directions.agg(
        N='size',Median='median',Mean='mean',AtLeastTwo=lambda x:(x>=2).mean()).reset_index()
    share = table[table.Unsaturated].groupby(['Window_s','SNR_dB','Representation']).TargetGradientShare.median()
    summary = summary.merge(share.rename('MedianTargetGradientShare').reset_index(),
                            on=['Window_s','SNR_dB','Representation'],validate='one_to_one')
    summary.to_csv(DEST/'direction-summary.csv',index=False)
    scan = pd.read_csv(DEST/'matched-scan.csv')
    matches = scan.dropna(subset='MatchedDeficit')
    assert len(scan)==36 and matches.EndpointError_Hz.abs().max()<1e-6
    grouped = matches.assign(AbsTarget_mHz=abs(matches.TargetDifference_Hz)*1000).groupby('Factor').agg(
        Matched=('MatchedDeficit','size'),MinTargetDifference_mHz=('AbsTarget_mHz','min'),
        MaxTargetDifference_mHz=('AbsTarget_mHz','max'),Limited=('EitherLimited','sum'))
    grouped.to_csv(DEST/'matched-summary.csv')
    draw(summary,matches)
    print(grouped.to_string())
    print(summary.query('Window_s==.50').to_string(index=False))
    print('Unsaturated:',int(primary.sum()),'Scaling replays:',np.isfinite(source['scaling_error']).sum(),
          'Largest scaling error:',np.nanmax(source['scaling_error']),'Median FD error:',np.median(step_error[primary]))


def draw(summary,matches):
    fig,axes = plt.subplots(1,3,figsize=(13.0,3.7),layout='constrained')
    for factor,rows in matches.groupby('Factor'):
        axes[0].scatter(rows.ShapeRMS_Hz*1000,abs(rows.TargetDifference_Hz)*1000,label=f'变化参数 {factor}')
    axes[0].set(xlabel='等端点窗口频率差 RMS / mHz',ylabel='截止后最低频率绝对差 / mHz')
    axes[0].legend(frameon=False)
    for name,series in summary.query('SNR_dB==30').groupby('Representation'):
        axes[1].plot(series.Window_s,series.AtLeastTwo*100,marker='o',label=NAMES[name])
    axes[1].set(xlabel='实际末样本时刻 / s',ylabel='至少两个可测方向的场景比例 / %',ylim=(-4,104),
                title='30 dB 模拟白噪声')
    axes[1].legend(frameon=False,fontsize=8)
    for name,series in summary.query('Window_s==0.50').groupby('Representation'):
        axes[2].plot(series.SNR_dB,series.MedianTargetGradientShare*100,marker='o',label=NAMES[name])
    axes[2].set(xlabel='模拟白噪声信噪比 / dB',ylabel='目标梯度投影比例中位数 / %',xticks=[20,30,40],
                ylim=(60,104),title='0.50 s 窗口')
    fig.savefig(DEST/'matched-and-sensitivity.pdf')
    plt.close(fig)


if __name__ == '__main__':
    analyze()
