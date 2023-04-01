import os
import logging

import json as jsonlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy.linalg import det
from scipy.ndimage import zoom, gaussian_filter1d

from nems0 import xforms
import nems0.plots.api as nplt
from nems0.plots.api import ax_remove_box, spectrogram, fig2BytesIO
from nems0.plots.heatmap import _get_wc_coefficients, _get_fir_coefficients
from nems0.uri import NumpyEncoder
from nems0.utils import get_setting, smooth
from nems0.modules.fir import per_channel
from nems0.xforms import load_analysis

log = logging.getLogger(__name__)


def compute_dstrf(modelspec, rec, index_range=None, sample_count=100, out_channel=[0], memory=10,
                  norm_mean=True, method='jacobian', **kwargs):

    # remove static nonlinearities from end of modelspec chain
    modelspec = modelspec.copy()
    """
    if ('double_exponential' in modelspec[-1]['fn']):
        log.info('removing dexp from tail')
        modelspec.pop_module()
    if ('relu' in modelspec[-1]['fn']):
        log.info('removing relu from tail')
        modelspec.pop_module()
    if ('levelshift' in modelspec[-1]['fn']):
        log.info('removing lvl from tail')
        modelspec.pop_module()
    """
    modelspec.rec = rec
    stimchans = rec['stim'].shape[0]
    bincount = rec['pred'].shape[1]
    stim_mean = np.mean(rec['stim'].as_continuous(), axis=1, keepdims=True)
    if index_range is None:
        index_range = np.arange(bincount)
        if sample_count is not None:
            np.random.shuffle(index_range)
            index_range = index_range[:sample_count]
    sample_count = len(index_range)
    dstrf = np.zeros((sample_count, stimchans, memory, len(out_channel)))
    for i, index in enumerate(index_range):
        if i % 50 == 0:
            log.info(f"dSTRF: {i}/{len(index_range)} idx={index}")
        if len(out_channel)==1:
            dstrf[i,:,:,0] = modelspec.get_dstrf(rec, index, memory, out_channel=out_channel, method=method)
        else:
            dstrf[i,:,:,:] = modelspec.get_dstrf(rec, index, memory, out_channel=out_channel, method=method)

    if norm_mean:
        dstrf *= stim_mean[np.newaxis, ..., np.newaxis]

    return dstrf

def force_signal_silence(rec, signal='stim'):
    rec = rec.copy()
    for e in ["PreStimSilence", "PostStimSilence"]:
        s = rec[signal].extract_epoch(e)
        s[:,:,:]=0
        rec[signal]=rec[signal].replace_epoch(e, s)
    return rec

def dstrf_details(modelspec, rec,cellid,rr,dindex, dstrf=None, dpcs=None, memory=20, stepbins=3, maxbins=1500, n_pc=3):
    cellids = rec['resp'].chans
    match=[c==cellid for c in cellids]
    c = np.where(match)[0][0]

    # analyze all output channels
    out_channel = [c]
    channel_count=len(out_channel)

    if dstrf is not None:
        stimmag = dstrf.shape[0]

    rec = force_signal_silence(rec)
    stim_mag = rec['stim'].as_continuous()[:,:maxbins].sum(axis=0)
    index_range = np.arange(0, len(stim_mag), 1)
    if dstrf is None:
        log.info('Calculating dstrf for %d channels, %d timepoints (%d steps), memory=%d',
                 channel_count, len(index_range), stepbins, memory)
        dstrf = compute_dstrf(modelspec, rec.copy(), out_channel=out_channel,
                              memory=memory, index_range=index_range)

    if dpcs is None:
        # don't skip silent bins

        stim_big = stim_mag > np.max(stim_mag) / 1000
        driven_index_range = index_range[(index_range > memory) & stim_big[index_range]]
        driven_index_range = driven_index_range[(driven_index_range >= memory)]

        # limit number of bins to speed up analysis
        driven_index_range = driven_index_range[:maxbins]
        #index_range = np.arange(525,725)

        pcs, pc_mag = compute_dpcs(dstrf[driven_index_range,:,:,:], pc_count=n_pc)

    c_=0

    ii = np.arange(rr[0],rr[1])
    rr_orig = ii

    print(pcs.shape, dstrf.shape)
    u = np.reshape(pcs[:,:,:,c_],[n_pc, -1])
    d = np.reshape(dstrf[ii,:,:,c_],[len(ii),-1])
    pc_proj = d @ u.T

    stim = rec['stim'].as_continuous()
    pred=np.zeros((pcs.shape[0],stim.shape[1]))
    for i in range(pcs.shape[0]):
        pred[i,:] = per_channel(stim, np.fliplr(pcs[i,:,:,0]))

    n_strf=len(dindex)

    f = plt.figure(constrained_layout=True, figsize=(18,8))
    gs = f.add_gridspec(5, n_strf)
    ax0 = f.add_subplot(gs[0, :])
    ax1 = f.add_subplot(gs[1, :])
    ax2 = f.add_subplot(gs[2, :])
    ax = np.array([f.add_subplot(gs[3, i]) for i in range(n_strf)])
    ax3 = np.array([f.add_subplot(gs[4, i]) for i in range(n_strf)])

    ax0.imshow(rec['stim'].as_continuous()[:, rr_orig], aspect='auto', origin='lower', cmap="gray_r")
    xl=ax0.get_xlim()

    ax1.plot(rec['resp'].as_continuous()[c, rr_orig], color='gray');
    ax1.plot(rec['pred'].as_continuous()[c, rr_orig], color='purple');
    ax1.legend(('actual','pred'), frameon=False)
    ax1.set_xlim(xl)
    yl1=ax1.get_ylim()

    #ax2.plot(pc_proj);
    ax2.plot(pred[:,rr_orig].T);
    ax2.set_xlim(xl)
    ax2.set_ylabel('pc projection')
    ax2.legend(('PC1','PC2','PC3'), frameon=False)
    yl2=ax2.get_ylim()

    dindex = np.array(dindex)
    mmd=np.max(np.abs(dstrf[np.array(dindex)+rr[0],:,:,c_]))
    stim = rec['stim'].as_continuous()[:,rr_orig] ** 2
    mms = np.max(stim)
    stim /= mms

    for i,d in enumerate(dindex):
        ax1.plot([d,d],yl1,'--', color='darkgray')
        ax2.plot([d,d],yl2,'--', color='darkgray')
        _dstrf = dstrf[d+rr[0],:,:,c_]
        if True:
            #_dstrf = np.concatenate((_dstrf,stim[:,(d-_dstrf.shape[1]):d]*mmd), axis=0)
            _dstrf = np.concatenate((_dstrf,_dstrf*stim[:,(d-_dstrf.shape[1]):d]), axis=0)

            #_dstrf *= stim[:,(d-_dstrf.shape[1]):d]
        ds = np.fliplr(_dstrf)
        ds=zoom(ds, [2,2])
        ax[i].imshow(ds, aspect='auto', origin='lower', clim=[-mmd, mmd], cmap=get_setting('WEIGHTS_CMAP'))
        #plot_heatmap(ds, aspect='auto', ax=ax[i], interpolation=2, clim=[-mmd, mmd], show_cbar=False, xlabel=None, ylabel=None)

        ax[i].set_title(f"Frame {d}", fontsize=8)
        if i<n_pc:
            ds=np.fliplr(pcs[i,:,:,c_])
            ds=zoom(ds, [2,2])
            mmp = np.max(np.abs(ds))
            #ax3[i].imshow(ds, aspect='auto', origin='lower', clim=[-mmp, mmp])
            ax3[i].imshow(ds, aspect='auto', origin='lower', clim=[-mmp, mmp], cmap=get_setting('WEIGHTS_CMAP'))
        else:
            ax3[i].set_axis_off()

    ax[0].set_ylabel('example frames')
    ax3[0].set_ylabel('PCs')

    return f


def compute_dpcs(dstrf, pc_count=3):

    #from sklearn.decomposition import PCA
    channel_count = dstrf.shape[3]
    s = list(dstrf.shape)
    s[0]=pc_count
    pcs = np.zeros(s)
    pc_mag = np.zeros((pc_count, channel_count))

    for c in range(channel_count):
        d = np.reshape(dstrf[:, :, :, c], (dstrf.shape[0], s[1]*s[2]))
        #d -= d.mean(axis=0, keepdims=0)

        _u, _s, _v = np.linalg.svd(d.T @ d)
        _s = np.sqrt(_s)
        _s /= np.sum(_s[:pc_count])
        pcs[:, :, :, c] = np.reshape(_v[:pc_count, :],[pc_count, s[1], s[2]])
        pc_mag[:, c] = _s[:pc_count]

    return pcs, pc_mag


def dstrf_pca(modelspec, rec, pc_count=3, out_channel=[0], memory=10, return_dstrf=False,
              pca_index_range=None, **kwargs):

    dstrf = compute_dstrf(modelspec, rec.copy(), out_channel=out_channel,
                          memory=memory, **kwargs)

    if pca_index_range is None:
        pcs, pc_mag = compute_dpcs(dstrf, pc_count)
    else:
        pcs, pc_mag = compute_dpcs(dstrf[pca_index_range, :, :], pc_count)

    if return_dstrf:
       return pcs, pc_mag, dstrf
    else:
       return pcs, pc_mag


def dstrf_sample(ctx=None, cellid='TAR010c-18-2', savepath=None):

    if ctx is None:
        xf, ctx = load_analysis(savepath)

    rec = ctx['val']
    modelspec = ctx['modelspec']
    cellids = rec['resp'].chans
    match = [c == cellid for c in cellids]
    c = np.where(match)[0][0]
    maxbins = 1000
    stepbins = 3
    memory = 15

    # analyze all output channels
    out_channel = [c]
    channel_count = len(out_channel)

    stim_mag = rec['stim'].as_continuous()[:, :maxbins].sum(axis=0)
    index_range = np.arange(0, len(stim_mag), 1)
    log.info('Calculating dstrf for %d channels, %d timepoints (%d steps), memory=%d',
             channel_count, len(index_range), stepbins, memory)
    dstrf = compute_dstrf(modelspec, rec.copy(), out_channel=out_channel,
                          memory=memory, index_range=index_range)

    rr = (150, 550)
    dindex = [52, 54, 56, 65, 133, 165, 215, 220, 233, 240, 250]

    f = dstrf_details(modelspec, rec, cellid, rr, dindex, dstrf=dstrf, dpcs=None, maxbins=maxbins)
    f.suptitle(f"{cellid} r_test={modelspec.meta['r_test'][c][0]:.3f}");

    return f, dstrf