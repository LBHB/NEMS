"""
nems_lbhb.analysis.dstrf - tools for generating/analyzing dstrfs

dstrf_pca -
subspace_model_fit -

"""

import logging
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from scipy.ndimage import zoom, gaussian_filter

from nems.tools import dstrf as dtools
from nems.layers import filter
from nems.metrics import correlation

#import nems0.db as nd
#from nems0.utils import shrinkage, smooth
#from nems0.plots.file import fig2BytesIO
#from nems0.plots.file import fig2BytesIO
#from nems0 import xforms, db, xform_helper
#from nems0.initializers import init_nl_lite
#from nems_lbhb.plots import histscatter2d, histmean2d, scatter_comp, scatter_bin_lin, CB_color_cycle
from nems.models import LN
#from nems_lbhb.analysis import depth
from nems.preprocessing.filters import smooth

log = logging.getLogger(__name__)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def compute_extract_dpc(modelspec_list, stim, D=15, timestep=3, timeoffset=0, pc_count=10, max_frames=3000,
              out_channels=None, channel_chunksize=None, t_indexes=None, mask_channels=None, leave_channel_out=False,
              snr_threshold=5, method='pca', mask_inputs=None,
              first_lin=False, compute_dpc_all=True, remove_nl=False, sigrat=0.75, target_layer=None, use_multi_dstrf=None):
    """
    Compute and extract dynamic principal components from dSTRF data.

    Additional Parameters for Multi-Input Models:
    --------------------------------------------
    target_layer : str, optional
        Name of intermediate layer to compute dSTRF for (e.g., 'concatenate' for HRTF+DLC)
        Only used when use_multi_dstrf=True
    use_multi_dstrf : bool, optional
        Force use of multi-input dSTRF method. If None, auto-detects based on stim format.
        Set to True for multi-input models (HRTF, etc.), False for standard models.
    """

    if 'input' not in stim.keys():
        try:
            in_stim = stim['stim']
        except:
            RaisedError = ValueError("No input named input or stim in input dict")
    else:
        in_stim = stim['input']

    if t_indexes is None:
        smag = in_stim.sum(axis=1)
        t_indexes = np.arange(timestep+timeoffset, in_stim.shape[0], timestep)
        t_indexes = t_indexes[t_indexes>D]
        log.info(f'initial t_index len: {len(t_indexes)}')
        t_indexes = t_indexes[(smag[t_indexes]>0) & (smag[t_indexes-D+1]>0)]
        log.info(f't_index len after removing zero stim: {len(t_indexes)}')
        if len(t_indexes)>max_frames:
            log.info(f'Reducing t_indexes length to {max_frames}')
            t_indexes = t_indexes[np.linspace(0,len(t_indexes)-1,max_frames).astype(int)]

    # Auto-detect if we should use multi-input dSTRF
    if use_multi_dstrf is None:
        # Check if we have multiple input signals (excluding 'input' which is often concatenated)
        non_input_keys = [k for k in stim.keys() if k != 'input']
        use_multi_dstrf = len(non_input_keys) > 1 or (len(stim) > 1 and 'input' not in stim)
        if use_multi_dstrf:
            log.info("Detected multi-input model, using dstrf_multi method")

    # extra loop to save memory (but slow things down substantially)
    # seems to be necessary if max_frames is big, which for now is only for control analyses
    if channel_chunksize is None:
        channel_chunksize = len(out_channels)
    chunk_offsets = np.arange(0,len(out_channels),channel_chunksize)
    mean_dstrf_list = []
    for chunk, chunkoffset in enumerate(chunk_offsets):
        dstrfs = []
        for mi, m in enumerate(modelspec_list):
            log.info(f"Chunk {chunk+1}/{len(chunk_offsets)}: Computing dSTRF {mi+1}/{len(modelspec_list)} at {len(t_indexes)} points (timestep={timestep})")
            oc = out_channels[chunkoffset:(chunkoffset+channel_chunksize)]
            if use_multi_dstrf and hasattr(m, 'dstrf_multi'):
                log.info(f"Using dstrf_multi for multi-input model (target_layer={target_layer})")
                # stim is already in [time, channels] format which is what dstrf_multi expects
                d_ = m.dstrf_multi(stim, D=D, out_channels=oc, t_indexes=t_indexes,
                                 reset_backend=False, target_layer=target_layer)
            else:
                # Use the standard single-input method
                if use_multi_dstrf:
                    log.warning("Model doesn't support dstrf_multi, falling back to standard method")
                d_ = m.dstrf(stim, D=D, out_channels=oc, t_indexes=t_indexes, reset_backend=False)

            dpc_edges=[v.shape[2] for k, v in d_.items()]
            d_ = np.concatenate([v for k, v in d_.items()], axis=2)
            dstrfs.append(d_)

        dstrf = np.stack(dstrfs, axis=1)

        if mask_channels is not None:
            # hack, zero-out top channel
            log.info(f'Removing spectral channel(s) {mask_channels} (axis 3 of {dstrf.shape}) -- bias issue?')
            dstrf[:, :, :, mask_channels, :] = 0

        s = np.std(dstrf, axis=(2, 3, 4), keepdims=True)
        if (s==0).sum()>0:
            log.warning(f"{(s==0).sum()} dSTRFs have zero value across all time")
            s[s==0]=1
        if (np.isnan(s)).sum()>0:
            log.warning(f"{(np.isnan(s)).sum()} (some) dSTRF stds have nan value")
            s[np.isnan(s)]=1

        dstrf /= s
        dstrf /= np.nanmax(np.abs(dstrf)) * 0.9
        F = dstrf.shape[3]
        U = dstrf.shape[4]

        if (sigrat == 0) | (dstrf.shape[1] == 1):
            log.info("Averaging across jackknifes")
            mdstrf = np.nanmean(dstrf, axis=1, keepdims=True)
            mdstrf /= np.max(np.abs(mdstrf), axis=(2,3,4), keepdims=True) * 0.9
            mean_dstrf = mdstrf

        else:
            log.info(f"Shrinkage across jackknifes (sigrat={sigrat})")
            mdstrf = np.nanmean(dstrf, axis=1, keepdims=True)
            sdstrf = dstrf.std(axis=1, keepdims=True)
            sdstrf[sdstrf == 0] = 1
            mzdstrf = shrinkage(mdstrf, sdstrf, sigrat=0.75)
            del sdstrf
            mzdstrf /= np.max(np.abs(mdstrf), axis=(2,3,4), keepdims=True) * 0.9
            if leave_channel_out == False:
                del mdstrf
            mean_dstrf = mzdstrf

        if mask_inputs is not None:
            log.info(f"Masking out channels {mask_inputs} prior to dpc calculation")
            mean_dstrf[:,:,:,mask_inputs,:]=0

        if leave_channel_out:
            ccnt = len(out_channels)
            cellcount = len(out_channels)
            newlen = (cellcount-1)*int(mdstrf.shape[2]/ccnt)
            dstrf2 = np.zeros([cellcount,1,newlen]+list(mean_dstrf.shape[3:]))
            log.info(f'odd-one-out dstrf {cellcount} out channels, mean_dstrf shape: {mean_dstrf.shape} dstrf2 shape: {dstrf2.shape} ccnt {ccnt}')
            for ci in range(cellcount):
                #log.info(f"{[np.arange(ci), np.arange(ci+1,cellcount)]}")
                aa = np.concatenate([np.arange(ci), np.arange(ci+1,cellcount)])
                dstrf2[ci] = np.reshape(mdstrf[aa,:,(ccnt-1)::ccnt], [-1, F, U])
            d_ = dtools.compute_dpcs(dstrf2[:, 0], pc_count=pc_count, first_lin=first_lin,
                                    method=method, snr_threshold=None, as_dict=True)
        else:
            d_ = dtools.compute_dpcs(mean_dstrf[:, 0], pc_count=pc_count, first_lin=first_lin,
                                     method=method, snr_threshold=snr_threshold, as_dict=True)
        if chunk == 0:
            d = d_.copy()
        else:
            for k1 in d.keys():
                d[k1]['pcs'] = np.concatenate([d[k1]['pcs'],d_[k1]['pcs']],axis=0)
                d[k1]['pc_mag'] = np.concatenate([d[k1]['pc_mag'],d_[k1]['pc_mag']],axis=1)
                d[k1]['mean'] = np.concatenate([d[k1]['mean'],d_[k1]['mean']],axis=0)
                d[k1]['projection'] = None
        mean_dstrf_list.append(mean_dstrf)

    mean_dstrf=np.concatenate(mean_dstrf_list, axis=0)
    del mean_dstrf_list

    if compute_dpc_all:
        # dpcs for all cells in site
        cellcount = len(out_channels)
        F = mean_dstrf.shape[3]
        U = mean_dstrf.shape[4]
        #T = mean_dstrf[:,:,::10].shape[-1]
        dstrf_all = np.reshape(mean_dstrf[:,:,::cellcount], [1, -1, F, U])
        log.info(f"Computing site-wide dPCs: shape={mean_dstrf.shape}")
        dall = dtools.compute_dpcs(dstrf_all, pc_count=pc_count, snr_threshold=None,
                                   first_lin=first_lin, method='pca', as_dict=True)
        d['input']['dpc_all'] = dall['input']['pcs']
        d['input']['dpc_mag_all'] = dall['input']['pc_mag']

        # compute variance of each unit's dSTRF in each dpc_all dimension
        # dstrf shape:  = (192, 1, 2954, 24, 40)
        dd = np.tensordot(d['input']['dpc_all'][0], mean_dstrf[:,0], axes=((1,2),(2,3)))
        ddsim = dd.var(axis=2)
        ddsim = ddsim/ddsim.sum(axis=0, keepdims=True)
        d['input']['dpc_all_var'] = ddsim
        log.info(f"dpc_all_var.shape={ddsim.shape}")
    else:
        d['input']['dpc_all'] = None
        d['input']['dpc_mag_all'] = None
        d['input']['dpc_all_var'] = None

    if 'projection' not in d['input'].keys():
        d['input']['projection'] = None
    for oi, oc in enumerate(out_channels):
        for di in range(pc_count):
            # flip the sign of filters that are majority negative
            if d['input']['pcs'][oi, di].sum()<0:
                d['input']['pcs'][oi, di] = -d['input']['pcs'][oi, di]
                if d['input']['projection'] is not None:
                    d['input']['projection'][oi,:,di] = -d['input']['projection'][oi,:,di]

    d['dpc_edges'] = dpc_edges
    return d, mean_dstrf
    
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def dstrf_pca(est=None, modelspec=None, val=None, sig='input', modelspec_list=None,
              D=15, timestep=3, timeoffset=0, pc_count=10, max_frames=3000,
              out_channels=None, noise_floor_reps=10, compute_dpc_all=True,
              figures=None, fit_ss_model=False, ss_pccount=5, ss_dpc_var=0.95,
              first_lin=False, remove_nl=False, snr_threshold=5, sigrat=1, verbose=False,
              method='pca', fa_compress=None, cellid=None, mask_channels=None, leave_channel_out=False,
              leave_channels=None, channel_chunksize=None, IsReload=False,
              target_layer=None, use_multi_dstrf=None, **ctx):
    """
    xforms function
    use modelspec or modelspec_list to compute dSTRFs from est recording (using nems.tools.dstrf)
    then perform PCA on the collection of dSTRFs, save the top pc_count
    if fit_ss_model, fit a DNN using the projection into the subspace for each neurons
    save results in modelspec.meta
    :param est:
    :param modelspec:
    :param val:
    :param sig:
    :param modelspec_list:
    :param D:
    :param timestep:
    :param pc_count:
    :param max_frames:
    :param out_channels:
    :param noise_floor_reps:
    :param figures:
    :param fit_ss_model:
    :param ss_pccount:
    :param ss_dpc_var:
    :param first_lin:
    :param snr_threshold:
    :param sigrat:
    :param verbose:
    :param method:
    :param fa_compress: if int>0, optionally run the dpcs through Factor Analysis to further reduce dimensionality
    :param IsReload:
    :param ctx:
    :return:
    """

    if IsReload:
        # load dstrf data saved in modelpath.. or don't if not needed?
        return {}
    if ((est is None) and ((modelspec is not None) | (modelspec_list is not None))):
        raise ValueError("est and modelspec parameters required")
    r = est

    if modelspec_list is None:
        modelspec_list = [modelspec]
    if modelspec is None:
        modelspec = modelspec_list[0]
    cellids = r['resp'].chans
    if out_channels is None:
        out_channels = np.arange(len(cellids))
    cellcount=len(out_channels)

    if sig=='input':
        stim = {'input': r['stim'].as_continuous().T}
        input_names = modelspec.get_io_names()[0]
    elif sig=='model':
        stim = {}
        input_names = modelspec.get_io_names()[0]

        # Automatically detect and include all required input signals
        # Exclude 'input' as it's typically a derived/concatenated signal
        for input_name in input_names:
            if input_name not in ['input'] and input_name in r.signals:
                stim[input_name] = r[input_name].as_continuous().T
                log.info(f"Added input signal: {input_name}, shape: {stim[input_name].shape}")

        # Only add 'input' if no other signals were found (fallback)
        if len(stim) == 0:
            stim['input'] = r['stim'].as_continuous().T
            log.info(f"Using stimulus as input signal")

        # dstrf appears to concat all inputs - order matters
        sig = 'input'
    d, mean_dstrf = compute_extract_dpc(modelspec_list, stim, out_channels=out_channels, D=D, timestep=timestep, timeoffset=timeoffset,
                                        pc_count=pc_count, max_frames=max_frames, sigrat=sigrat, snr_threshold=snr_threshold,
                                        mask_channels=mask_channels, channel_chunksize=channel_chunksize, compute_dpc_all=compute_dpc_all,
                                        leave_channel_out=leave_channel_out, target_layer=target_layer, use_multi_dstrf=use_multi_dstrf)
    # if compute_dpc_all:
    #     # dpcs for all cells in site
    #     F = mean_dstrf.shape[3]
    #     U = mean_dstrf.shape[4]
    #     #T = mean_dstrf[:,:,::10].shape[-1]
    #     dstrf_all = np.reshape(mean_dstrf[:,:,::cellcount], [1, -1, F, U])
    #     log.info("Computing site-wide dPCs")
    #     dall = dtools.compute_dpcs(dstrf_all, pc_count=pc_count, snr_threshold=None,
    #                                first_lin=first_lin, method='pca', as_dict=True)
    # else:
    #     dall = {'input': {'pcs': None, 'pc_mag': None}}

    if noise_floor_reps > 0:
        log.info(f"Computing dPC noise floor for each unit (N={noise_floor_reps})")
        # compute noise floor by measuring PCs with shuffled spectro-temporal parameters
        sh_mags = []
        for i in range(noise_floor_reps):
            m_ = shuffle_along_axis(mean_dstrf, axis=2)
            d_ = dtools.compute_dpcs(m_[:, 0], pc_count=pc_count, first_lin=first_lin,
                                     snr_threshold=None, method=method, as_dict=True,
                                     flip_sign=False)
            sh_mags.append(d_[sig]['pc_mag'])
        sh_mags = np.stack(sh_mags, axis=2)
        msh = sh_mags.mean(axis=2)
        esh = sh_mags.std(axis=2)/(noise_floor_reps**0.5)
    else:
        log.info(f"Skipping noise floor calc.")
        msh = None
        esh = None
    
    # save the dPCs computed after dSTRF shrinkage!!
    dpc = d[sig]['pcs']
    dpc_mag = d[sig]['pc_mag']

    if fa_compress is not None:
        # second stage. Project stimulus into dPC space, run FA to rotate
        # to a space where 95% variance requires fewer dims
        mspec = modelspec.copy()
        mspec.meta['dpc'] = dpc
        mspec.meta['dpc_mag'] = dpc_mag
        X_est = {'input': est['stim'].as_continuous().T}
        Y_est = est['resp'].as_continuous().T 
        #X_est, Y_est = xforms.lite_input_dict(mspec, est, epoch_name="")
        for o in range(dpc.shape[0]):
            X = project_to_subspace(modelspec=mspec, X=X_est['input'], out_channels=[o], pc_count=dpc.shape[1],
                                    poly_expand=1, use_dpc_all=False, verbose=True)[0].T
            d_ = dpc[o]
            d_mag = dpc_mag[:,o]
            log.info(f"d_.shape: {d_.shape} d_mag.shape: {d_mag.shape} X.shape: {X.shape}")
            dpc_new, dpc_mag_new = dpc_compress(d_, d_mag, X, verbose=True, fa_compress=fa_compress)
            log.info(f"done. dpc_new.shape={dpc_new.shape}")
            dpc[0] = dpc_new
            dpc_mag[:, o] = dpc_mag_new

    #dproj = dz[sig]['projection']
    #log.info(f"dproj.shape={dproj.shape}")

    if msh is not None:
        dpc_nsig = np.zeros(len(out_channels))
        for oi, oc in enumerate(out_channels):
            dp = dpc_mag[:, oi]/dpc_mag[:, oi].sum()

            de = esh[:, oi] / esh[:, oi].sum() * np.sqrt(10)
            dsh = msh[:, oi] / msh[:, oi].sum()
            #for pp in range(1, pc_count):
            #    de[pp:] = de[pp:] / dsh[pp:].sum() * (1 - dp[:pp].sum())
            #    dsh[pp:] = dsh[pp:] / dsh[pp:].sum() * (1 - dp[:pp].sum())
            mag_rat = dp / dsh
            dpc_nsig[oi]=np.max(np.where(mag_rat >= 1)[0]) + 1
    else:
        dpc_nsig = np.ones(len(out_channels)) * pc_count

    modelspec = modelspec.copy()
    modelspec.meta = copy.deepcopy(modelspec.meta)
    modelspec.meta['dpc'] = dpc
    modelspec.meta['dpc_mag'] = dpc_mag
    modelspec.meta['dpc_mag_sh'] = msh
    modelspec.meta['dpc_mag_e'] = esh
    modelspec.meta['dpc_nsig'] = dpc_nsig
    modelspec.meta['dpc_edges'] = d['dpc_edges']
    # modelspec.meta['dpc_all'] = dall['input']['pcs']
    # modelspec.meta['dpc_mag_all'] = dall['input']['pc_mag']
    modelspec.meta['dpc_all'] = d['input']['dpc_all']
    modelspec.meta['dpc_mag_all'] = d['input']['dpc_mag_all']
    modelspec.meta['dpc_all_var'] = d['input']['dpc_all_var']

    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower', 'interpolation': 'none'}

    if verbose:
        f = plot_dpcs(modelspec=modelspec, out_channels=out_channels[:10], pc_count=None, D=None, z=2,
                      include_avg=True, **ctx)

        # rowcount = np.min([10, len(out_channels)])
        # colcount = np.min([10, pc_count])
        # f, ax = plt.subplots(rowcount, colcount + 1, figsize=(colcount + 1, rowcount * 0.75), sharex='col',
        #                      sharey='col')
        # f.subplots_adjust(top=0.98, bottom=0.02)
        # if rowcount == 1:
        #     ax = ax[np.newaxis, ...]
        # for oi, oc in enumerate(out_channels[:rowcount]):
        #     for di in range(colcount):
        #         d = dpc[oi, di]
        #         d = d / np.max(np.abs(d)) / np.max(dpc_mag[:, oi]) * dpc_mag[di, oi]
        #         ax[oi, di].imshow(np.fliplr(d), **imopts)
        #         ax[oi, di + 1].set_yticklabels([])
        #     if msh is not None:
        #         ax[oi, -1].plot(msh[:, oi] / msh[:, oi].sum(), lw=0.5, color='gray')
        #     ax[oi, -1].plot(dpc_mag[:, oi] / dpc_mag[:, oi].sum())
        #     yl = ax[oi, -1].get_ylim()
        #     ax[oi, -1].text(0, yl[1], modelspec.meta['cellids'][oi], fontsize=6, va='top')

        # populate modelspec.meta for saving important results
        if figures is None:
            figures = []
        figures.append(fig2BytesIO(f))

    if fit_ss_model:
        d = subspace_model_fit(est, val, modelspec, out_channels=out_channels,
                               pc_count=ss_pccount, dpc_var=ss_dpc_var)
        modelspec = d['modelspec']
    
    log.info("Removing backends from modelspec to avoid potential memory problem")
    modelspec.backend = None
    modelspec.dstrf_backend = None
    
    return {'modelspec': modelspec, 'figures': figures}


def dpc_compress(dpc, dpc_mag, X, verbose=False, n_components=None):
    """
    pass results of dSTRF PCA through FactorAnalysis to further reduce dimensionality
    :param dpc:
    :param dpc_mag:
    :param X:
    :param verbose:
    :return:
    """
    dm = dpc_mag**2 / (dpc_mag**2).sum()
    if n_components is None:
        n_components = len(dm)
    X = X[:170000:17] * dm[np.newaxis, :]
    log.info(f"X.shape: {X.shape} len(dm)={len(dm)} n_components={n_components}")
    p = FactorAnalysis(n_components=n_components)
    p = p.fit(X)

    py = p.transform(X)
    log.info(f"py.shape: {py.shape}")

    # evar = p.explained_variance_ratio_
    pcoefs = p.components_
    evar = (pcoefs**2).mean(axis=1)
    evar = 1/(evar + (evar==0)*10000)
    evar = evar / evar.sum()

    ii = np.argsort(-evar)
    evar = evar[ii]
    pcoefs = p.components_[ii]
    log.info(f"sorted evar: {evar}")

    s = dpc.shape
    dpc_new = np.zeros_like(dpc)
    dpc_mag_new = np.zeros_like(dpc_mag)

    dpc_new[:n_components] = np.reshape(pcoefs @ np.reshape(dpc, [s[0], -1]), [-1, s[1], s[2]])
    dpc_mag_new[:n_components] = evar**0.5

    if verbose:
        pcc = np.min([9,n_components])
        f, ax = plt.subplots(2, pcc+1, figsize=(12, 3), sharex='col', sharey='col')
        imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower', 'interpolation': 'none'}

        for di in range(pcc):
            d = dpc[di]
            d = d / np.max(np.abs(d))  # / np.max(dpc_mag[:, oc]) * dpc_mag[di, oc]
            ax[0, di+1].imshow(np.fliplr(d), **imopts)
            d = dpc_new[di]
            d = d / np.max(np.abs(d))  # / np.max(dpc_mag[:, oc]) * dpc_mag[di, oc]
            ax[1, di+1].imshow(np.fliplr(d), **imopts)
        ax[0, 0].plot(np.cumsum(evar), label='pca')
        ax[0, 0].plot(np.cumsum(dm), label='dpcmag')
        ax[0, 0].axhline(0.95)
        ax[0, 0].legend()
        ax[0, 0].set_ylabel('dPC+FA')
        ax[1, 0].set_ylabel('dPC')
    return dpc_new, dpc_mag_new


def subspace_model_fit(est, val, modelspec, pc_count=5, dpc_var=0.8, out_channels=None, cell_list=None,
                       use_dpc_all=False, single_fit=True, poly_model=None, fa_compress=None,
                       return_all=False, shape=None, units_per_layer=15, cost_function='nmse',
                       IsReload=False, **ctx):
    """
    :param est:
    :param val:
    :param modelspec:
    :param pc_count:  use pc_count dimensional subspace, unless None...  (default 5)
    :param dpc_var:   if pc_count is None, use dpc_var fraction of subspace (default 0.8)
    :param out_channels:   which channels to fit (default all)
    :param use_dpc_all:    if True, use dpc_all to define subspace (default False)
    :param single_fit:     if True, fit a single model, require use_dpc_all==True (default True)
    :param poly_model: {None, int} if True, fit polynomial model degree poly_model
    :param IsReload:
    :param return_all:     return smodels list
    :param units_per_layer:
    :param shape: default (units_per_layer,units_per_layer) -- number of units in each layer
    :param ctx:
    :return:
    """
    if IsReload:
        # load dstrf data saved in modelpath
        return {'modelspec': modelspec}

    batch_size = None  # X_est.shape[0]  # or None or bigger?
    if shape is None:
        shape = (units_per_layer, units_per_layer)
        
    X_est = {'input': est['stim'].as_continuous().T}
    Y_est = est['resp'].as_continuous().T 
    #X_est, Y_est = xforms.lite_input_dict(modelspec, est, epoch_name="")
    X_val = {'input': val['stim'].as_continuous().T}
    Y_val = val['resp'].as_continuous().T 
    #X_val, Y_val = xforms.lite_input_dict(modelspec, val, epoch_name="")

    X_est['input'] = X_est['input'][np.newaxis]
    Y_est = Y_est[np.newaxis]
    X_val['input'] = X_val['input'][np.newaxis]
    Y_val = Y_val[np.newaxis]

    r = est
    cellids = r['resp'].chans
    if cell_list is not None:
        out_channels = np.array([i for i, c in enumerate(cellids) if c in ns_cellids])
    if out_channels is None:
        out_channels = np.arange(len(cellids))
    # always generate cell_list to make sure it matches the order of out_channels
    cell_list = [cellids[i] for i in out_channels]

    ssmodels = []
    sspredxc = np.zeros(len(out_channels))
    sspc_count = np.zeros(len(out_channels))
    ss0predxc = np.zeros(len(out_channels))

    if use_dpc_all & single_fit:
        out_channels=[out_channels]
    else:
        single_fit=False
    for oi, o in enumerate(out_channels):

        if fa_compress is not None:
            log.info(f"fa_compress={fa_compress}")
            if use_dpc_all:
                dpc = modelspec.meta['dpc_all'][0]
                dpc_mag = modelspec.meta['dpc_mag_all'][:,0]
            else:
                dpc = modelspec.meta['dpc'][o]
                dpc_mag = modelspec.meta['dpc_mag'][:, o]
            X = np.concatenate([
                project_to_subspace(modelspec=modelspec, X=x, out_channels=[o], pc_count=dpc.shape[0],
                                    poly_expand=1, use_dpc_all=use_dpc_all, verbose=True)[0].T
                for x in X_est['input']], axis=0)

            dpc_new, dpc_mag_new = dpc_compress(dpc, dpc_mag, X, verbose=True, n_components=fa_compress)
            modelspec = modelspec.copy()
            modelspec.meta['dpc'][o] = dpc_new
            modelspec.meta['dpc_mag'][:, o] = dpc_mag_new

        if single_fit:
            R=len(o)
            y_select = o
            pcc = pc_count
            oc = [0]
        else:
            R=1
            y_select = [o]
            if pc_count is None:
                if use_dpc_all:
                    dpc_mag = modelspec.meta['dpc_mag_all'][:, 0] ** 2
                else:
                    dpc_mag = modelspec.meta['dpc_mag'][:, o] ** 2
                dpc_mag = dpc_mag / dpc_mag.sum()
                dsum = np.cumsum(dpc_mag)
                pcc = int(np.min(np.where(dsum > dpc_var)[0]) + 1)
                log.info(f'dpc_var={dpc_var}: pc_count={pcc}')
            else:
                pcc = pc_count
            oc=[o]
        if poly_model == 2:
            poly_expand = 2
        else:
            poly_expand = 1
        
        X = np.stack([
            project_to_subspace(modelspec=modelspec, X=x, out_channels=oc, pc_count=pcc,
                                poly_expand=poly_expand, use_dpc_all=use_dpc_all, verbose=True)[0].T
            for x in X_est['input']], axis=0)
        Xv = project_to_subspace(modelspec=modelspec, X=X_val['input'][0], out_channels=oc, pc_count=pcc,
                                poly_expand=poly_expand, use_dpc_all=use_dpc_all, verbose=True)[0].T

        Y = Y_est[:, :, y_select]
        Yv = np.reshape(Y_val[:, :, y_select], [-1, R])

        if poly_model is not None:
            N = X.shape[2]
            log.info(f"** SS poly order {poly_model} N={N} {val['resp'].chans[o]} ({oi+1}/{len(out_channels)}):")
            keywordstring = f"wc.{N}x{R}-dexp.1"

        elif single_fit:
            log.info(f"** Fitting SS model for {R} cells:")
            s=shape
            keywordstring = f'wc.{pc_count}x{s[0]}-relu.{s[0]}.s-wc.{s[0]}x45-relu.45.s-wc.45x{R}-dexp.{R}'
        else:
            log.info(f"** Fitting SS model for cell {val['resp'].chans[o]} ({oi+1}/{len(out_channels)}):")
            s=shape
            if len(s)==2:
                keywordstring = f'wc.{pcc}x{s[0]}-relu.{s[0]}.s-wc.{s[0]}x{s[1]}-relu.{s[1]}.s-wc.{s[1]}x1-dexp.1'
            elif len(s)==3:
                keywordstring = f'wc.{pcc}x{s[0]}-relu.{s[0]}.s-wc.{s[0]}x{s[1]}-relu.{s[1]}.s-'+\
                                f'wc.{s[1]}x{s[2]}-relu.{s[2]}.s-wc.{s[2]}x1-dexp.1'

        lmodel0 = xforms.init_nems_keywords(keywordstring, meta=modelspec.meta)['modelspec']
        lmodel0 = lmodel0.sample_from_priors()
        fitter_options = {'cost_function': cost_function, 'early_stopping_delay': 100,
                          'early_stopping_patience': 150,
                          'early_stopping_tolerance': 1e-3,
                          'learning_rate': 1e-3, 'epochs': 10000,
                          }
        fit_opts2 = fitter_options.copy()
        #fit_opts2['early_stopping_tolerance'] = 5e-4
        #fit_opts2['learning_rate'] = 1e-4
        fit_opts2['early_stopping_tolerance'] = 1e-4
        fit_opts2['learning_rate'] = 5e-4

        lmodel0.layers[-1].skip_nonlinearity()
        lmodel = lmodel0.fit(input=X, target=Y, backend='tf',
                             batch_size=batch_size, fitter_options=fitter_options)
        lmodel = init_nl_lite(lmodel, X, Y)
        lmodel.layers[-1].unskip_nonlinearity()
        lmodel = lmodel.fit(input=X, target=Y, backend='tf',
                            batch_size=batch_size, fitter_options=fit_opts2)

        p0 = lmodel0.predict(Xv)
        p = lmodel.predict(Xv)
        if type(p) is dict:
            p=p['output']
            p0 = p0['output']
        if single_fit:
            for ii in range(R):
                sspredxc[ii] = correlation(p[:, ii], Yv[:, ii])
                ss0predxc[ii] = correlation(p0[:, ii], Yv[:, ii])
                sspc_count[ii] = pcc
        else:
            sspredxc[oi] = correlation(p, Yv)
            ss0predxc[oi] = correlation(p0, Yv)
            sspc_count[oi] = pcc
        lmodel.backend=None
        # TODO -- save disk space??
        #modelspec2 = ctx2['modelspec_list'][0].copy()
        del lmodel.meta['dpc']
        del lmodel.meta['dpc_all']
        del lmodel.meta['dpc_mag']
        del lmodel.meta['dpc_mag_e']
        
        ssmodels.append(lmodel)
        
    if single_fit:
        out_channels = out_channels[0]

    newmodelspec=modelspec.copy()
    newmodelspec.meta['sspredxc'] = sspredxc
    newmodelspec.meta['sspc_count'] = sspc_count
    if 'r_test' in newmodelspec.meta.keys():
        log.info("Cellid        Orig  Subspace")
        for oi, o in enumerate(out_channels):
            log.info(f"{newmodelspec.meta['cellids'][o]}" + \
                     f" {newmodelspec.meta['r_test'][o, 0]:.3f}" + \
                     f" {newmodelspec.meta['sspredxc'][oi]:.3f}")
    if return_all:
        return {'modelspec': newmodelspec, 'ssmodels': ssmodels}
    else:        
        return {'modelspec': newmodelspec}


def dpc_load_subspace_fit(modelspec=None, meta=None,
                          refit_dpc=False, refit_pc_count=25, refit_method='pca',
                          D=30, sigrat=1, pc_count=5, dpc_var=0.8, out_channels=None,
                          units_per_layer=15, IsReload=False, **ctx):
    """
    :param modelspec:
    :param meta:
    :param refit_dpc:
    :param pc_count: overrides dpc_var. Leave as None to enable dpc_var
    :param dpc_var:
    :param out_channels:
    :param units_per_layer:
    :param ctx: passed through to subspace_model_fit: use_dpc_all, single_fit
    :return:
    """
    if IsReload:
        return {}
        
    log.info("ssfitting: "+meta['modelname'])
    batch=meta['batch']
    saved_cellid = meta['cellids'][0]
    siteid = saved_cellid.split("-")[0]
    loader = meta['loader']
    modelspecname = meta['modelspecname']
    fitters_to_try = [
        'lite.tf.init.lr1e3.t3.es20.jk8.rb4-lite.tf.lr1e4.t5e4-dstrf.d30.t47.p15.ss95.nl',
        'lite.tf.init.lr1e3.t3.es20.jk8.rb4-lite.tf.lr1e4.t5e4-dstrf.d30.t47.p25.ss95.nl',
        'lite.tf.init.lr1e3.t3.es20.jk8.rb4-lite.tf.lr1e4.t5e4-dstrf.d30.t47.p25.ss95.nl.sr0',
        'lite.tf.init.lr1e3.t3.es20.jk8.rb4-lite.tf.lr1e4.t5e4-dstrf.d25.t47.p15.ss95.nl',
        'lite.tf.init.lr1e3.t3.es20.jk8.rb4-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.ss95.nl',
        'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.ss.nl',
        'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.nl',
        'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15',
        'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4',
        'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.ss95.nl'
        ]
    test_modelnames = [f"{loader}_{modelspecname}_{f}" for f in fitters_to_try]
    for i,m in enumerate(test_modelnames):
        log.info(f'trying {saved_cellid}, {m}')
        try:
            x_,c_ = xform_helper.load_model_xform(saved_cellid, batch=batch, modelname=m,eval_model=False,verbose=True)
            log.info(f'success on model {i}!')
            log.info(m)
            break
        except:
            log.info(f"failed to load model {i}")
    save_meta = modelspec.meta.copy()
    modelspec = c_['modelspec']
    modelspec.meta.update(save_meta)
    modelspec.name = f"{siteid}/{batch}/{meta['modelname']}"

    ctx['modelspec'] = modelspec
    if 'modelspec_list' in c_.keys():
        ctx['modelspec_list'] = c_['modelspec_list']

    if refit_dpc:
        ctx['verbose']=False
        c_ = dstrf_pca(D=D, sigrat=sigrat, timestep=47, pc_count=refit_pc_count,
                       first_lin=False, noise_floor_reps=0, fit_ss_model=False,
                       method=refit_method, fa_compress=None, **ctx)
        ctx['modelspec'] = c_['modelspec']

    sspredxc = []
    sspc_count = []
    ssunits = []
    ssdpc_var = []
    #out_channels = None   # np.arange(13)
    #dpc_var_range = [0.8, 0.9, 0.95]
    #units_per_layer_range = [5, 10, 15]
    dpc_var_range = [dpc_var]
    pc_count_range = [pc_count]
    units_per_layer_range = [units_per_layer]
    for dpc_var,pc_count in zip(dpc_var_range,pc_count_range):
        for units_per_layer in units_per_layer_range:
            if pc_count is None:
                log.info(f"Fitting SS with dpc_var={dpc_var}, unit_per_layer={units_per_layer}")
            else:
                log.info(f"Fitting SS with pc_count={pc_count}, unit_per_layer={units_per_layer}")
            res_ = subspace_model_fit(pc_count=pc_count, dpc_var=dpc_var, out_channels=out_channels,
                                      figures=None, return_all=True, fa_compress=None,
                                      units_per_layer=units_per_layer, **ctx)
            sspredxc.append(res_['modelspec'].meta['sspredxc'])
            ssmodels = res_['ssmodels']
            sspc_count.append(res_['modelspec'].meta['sspc_count'])
            ssunits.append(units_per_layer)
            ssdpc_var.append(dpc_var)

    modelspec = res_['modelspec']
    modelspec.meta['r_test_orig'] = modelspec.meta['r_test'].copy()
    modelspec.meta['r_test'] = np.stack(sspredxc, axis=1)
    #modelspec.meta['sspredxc'] = np.stack(sspredxc, axis=1)
    modelspec.meta['sspc_count'] = np.stack(sspc_count, axis=1)
    modelspec.meta['ssunits'] = np.array(ssunits)
    modelspec.meta['ssdpc_var'] = np.array(ssdpc_var)

    plt.figure()
    labels=[f"ss {p},{u}" for (p,u) in zip(ssdpc_var, ssunits)]
    if len(ssdpc_var) == 1:
        labels = labels[0]
    plt.plot(modelspec.meta['r_test'][:,0], label=labels)
    plt.plot(modelspec.meta['r_test_orig'][:,0], label='r_test')
    plt.legend()

    return {'modelspec': modelspec, 'modelspec_list': ssmodels}


### dPC plots ###

def project_to_subspace(modelspec=None, X=None, dpc0=None, out_channels=None, rec=None, est=None, val=None,
                        input_name='stim', pc_count=None, dpc_var=0.95, use_dpc_all=False, ss_name='subspace',
                        poly_expand=1, norm_std=False, verbose=True, **ctx):

    cellids = modelspec.meta['cellids']
    if out_channels is None:
        if use_dpc_all:
            out_channels=[0]
        else:
            out_channels = np.arange(len(cellids))
    if X is None:
        recs = [(n,r) for n,r in zip(['rec', 'est','val'],[rec, est, val]) if r is not None]
    else:
        recs = [('raw', X)]
    if X is None and (len(recs)==0):
        raise ValueError("must provide either X input matrix or valid NEMS recording")
    #log.info(f"{out_channels}")
    res = {}
    for name, rec in recs:
        if verbose:
            log.info(f"** Recording {name}:")

        if type(rec) is not np.ndarray:
            inp = rec[input_name].as_continuous().T
        else:
            inp = rec

        outs = []
        res[name]=rec.copy()
        outcells=[c for i,c in enumerate(modelspec.meta['cellids']) if i in out_channels]
        for oi, o in enumerate(out_channels):
            if dpc0 is not None:
                if verbose:
                    log.info('   Passing through dpc0')
                dpc = np.moveaxis(dpc0, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, o]
                if dpc.shape[2] == 1:
                    dpc=dpc[:, :, 0]
            elif use_dpc_all:
                if 'dpc_all' not in modelspec.meta:
                    raise ValueError("modelspec missing site-wide dSTRF pcs, run nems_lbhb.analysis.dstrf.dstrf_pca first")
                dpc = modelspec.meta['dpc_all']
                dpc_mag = modelspec.meta['dpc_mag_all'][:,0] ** 2

                dpc = np.moveaxis(dpc, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, 0]
            else:
                if 'dpc' not in modelspec.meta:
                    raise ValueError("modelspec missing dSTRF pcs, run nems_lbhb.analysis.dstrf.dstrf_pca first")
                dpc = modelspec.meta['dpc']
                dpc_mag = modelspec.meta['dpc_mag'][:,o] ** 2
                dpc = np.moveaxis(dpc, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, o]

            if pc_count is None:
                dsum = np.cumsum(dpc_mag/dpc_mag.sum())
                pcc = np.min(np.where(dsum > dpc_var)) + 1
                log.info(f'keeping {pcc}/{len(dsum)} dims for dpc_var={dpc_var}')
            else:
                pcc = pc_count

            if verbose:
                log.info(f"   SS proj oi={oi} o={o} {cellids[o]} pc_count={pcc}:")

            dpc = dpc[:, :, :pcc]
            fir = filter.FIR(shape=dpc.shape)
            fir['coefficients'] = np.flip(dpc, axis=0)

            ss = fir.evaluate(inp)
            outs.append(ss.T)

        ssout = np.stack(outs, axis=0)
        if poly_expand == 2:
            s2 = [ssout[:, [i], :] * ssout[:, i:, :] for i in range(pc_count)]
            ssout = np.concatenate([ssout] + s2, axis=1)
            if verbose:
                log.info(f"   poly_expand=2: {pc_count} to {ssout.shape[1]}")
            
        elif poly_expand > 2:
            raise ValueError('poly_expand>2 not supported')

        if norm_std:
            sstd = ssout.std(axis=2, keepdims=True)
            sstd[sstd==0]=1
            if verbose:
                log.info(f'norming std ssout shape={ssout.shape} sstd shape={sstd.shape}')
            ssout = ssout / sstd
        
        if name == 'raw':
            return ssout

        sig = res[name][input_name]._modified_copy(data=ssout, name=ss_name, chans=outcells)
        res[name].signals[ss_name] = sig

    return res

def project_model_to_ss(modelspec, X=None, rec=None, input_name='stim',
                        cellid=None, invert=False):

    if X is None:
        inp = rec[input_name].as_continuous().T
    else:
        inp = X
    if cellid is None:
        oi = 0
    else:
        oi = [i for i,c in enumerate(modelspec.meta['cellids']) if c==cellid][0]
    dpcz = modelspec.meta['dpc']
    dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]
    fir = filter.FIR(shape=dpcz.shape)
    fir['coefficients'] = np.flip(dpcz, axis=0)
    print(oi,cellid)
    ss = fir.evaluate(inp)

    return ss


### dPC plots ###

def nl_marginal(xx, yy, smoothwin=7, ex_pct=0.05, bins=25):
    ab = np.percentile(xx, [ex_pct, 100 - ex_pct])
    bb = np.linspace(ab[0], ab[1], bins + 1)
    result = np.zeros((2, bins))
    resulte = np.zeros((2, bins))
    x = np.stack([xx, yy], axis=1)

    for i in range(bins):
        b_ = (x[:, 0] >= bb[i]) & (x[:, 0] < bb[i + 1]) & np.isfinite(x[:, 1])
        if b_.sum() > 0:
            result[:, i] = np.nanmean(x[b_, :], axis=0)
            resulte[:, i] = np.nanstd(x[b_, :], axis=0) / np.sqrt(np.sum(b_))
    x_ = result[0]
    y_ = result[1]

    y_ = smooth(y_, window_len=7)
    dy = np.diff(y_)
    dx = (x_[:-1] + x_[1:]) / 2
    asym = 1 - np.abs(np.sum(dy)) / np.sum(np.abs(dy))
    if (dy > 0).sum() == 0:
        asym = -asym
    elif (dy < 0).sum() == 0:
        asym = asym
    elif (np.mean(dx[dy > 0]) - np.mean(dx[dy < 0])) < 0:
        asym = -asym
    return x_, y_, asym


def plot_dpcs(modelspec=None, out_channels=None, cell_list=None,
              pc_count=None, D=None, dfinfo=None, z=2, stride=None,
              figsize=None, dpc_var=0.95, fs=100, include_avg=False, show_mwf=True, dpc_edges=None,
              est=None, spec_norm=False, rec=None, **ctx):
    """
    plot the dpcs, one neuron per row
    :param modelspec:
    :param out_channels:
    :param pc_count:
    :param ctx:
    :return:
    """
    
    cellids = modelspec.meta['cellids'].copy()
    if 'r_test' in modelspec.meta.keys():
        r_test = modelspec.meta['r_test'][:,0]
    else:
        r_test = np.zeros(len(cellids))
    if (out_channels is None) and (cell_list is not None):
        out_channels = [[i for i, c in enumerate(modelspec.meta['cellids']) if c==cell_][0] for cell_ in cell_list]
    elif out_channels is None:
        out_channels=list(np.arange(len(cellids)))
    else:
        out_channels = list(out_channels)
    if (rec is not None) & (stride is None):
        stride = int(rec['stim'].fs/rec['resp'].fs)
    elif stride is None:
        stride = 1

    dpc = modelspec.meta['dpc']
    dpc_mag = modelspec.meta['dpc_mag'] ** 2
    msh = modelspec.meta.get('dpc_mag_sh', None)
    #esh=modelspec.meta.get('dpc_mag_e', None)
    
    if include_avg:
        dpc=np.concatenate((dpc,modelspec.meta['dpc_all']),axis=0)
        dpc_mag=np.concatenate((dpc_mag,modelspec.meta['dpc_mag_all']),axis=1)
        out_channels.append(dpc.shape[0]-1)
        cellids.append('-ALL')

    if pc_count is None:
        pc_count = dpc.shape[1]

    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower', 'interpolation': 'none'}
    if show_mwf & (dfinfo is not None):
        extracount=2
        oset=1
    else:
        extracount=1
        oset=0
        show_mwf=False
    if figsize is None:
        figsize = (pc_count+extracount, len(out_channels) * 0.75)
    f, ax = plt.subplots(len(out_channels), pc_count + extracount, sharex='col',
                         sharey='col', figsize=figsize)
    f.subplots_adjust(top=0.98, bottom=0.02)
    if len(out_channels) == 1:
        ax = ax[np.newaxis, ...]
    
    if dpc_edges is None:
        dpc_edges = np.cumsum(modelspec.meta.get('dpc_edges',np.ones(1)))
        dpc_edges = dpc_edges[:-1]
    
    if spec_norm & (est is not None):
        meanspect = est['stim'].as_continuous().mean(axis=1, keepdims=True)
        meanspect /= meanspect.max()
    else:
        spec_norm=False

    for oi, oc in enumerate(out_channels):
        cellid = cellids[oc]
        if show_mwf & (oset == 1):
            if (cellid in dfinfo.index):
                mwf = dfinfo.loc[cellid, 'mwf']
                if dfinfo.loc[cellid, 'narrow']:
                    ax[oi, 0].plot(mwf, 'r', lw=1)
                else:
                    ax[oi, 0].plot(mwf, 'gray', lw=0.5)
                ax[oi, 0].set_xticklabels([])
                ax[oi, 0].set_yticklabels([])
                #if j==1:
                #    ax[j, 0].set_title(title)
        ax[oi, 0].set_ylabel(f"{'-'.join(cellid.split('-')[1:])}", fontsize=6)

        if msh is not None:
            if oi<msh.shape[1]-1:
                ax[oi, -1].plot(msh[:,oi]/msh[:,oi].sum(), lw=0.5, color='gray')
        dpc_mag1 = dpc_mag[:, oc] / dpc_mag[:, oc].sum()
        #print(dpc_mag1)
        dsum1 = np.cumsum(dpc_mag1, axis=0)
        pcpc1 = int(np.min(np.where(dsum1 > dpc_var)[0]) + 1)
        for di in range(pc_count):
            d = dpc[oc, di]
            if stride>1:
                d = np.pad(d,[[0,0],[0,int(stride/2)]])
            if spec_norm:
                d *= meanspect
            if d[:,-4*stride:].mean()<0:
                # flip sign to make positive onset
                d = -d
            if D is None:
                D = d.shape[1]
            extent=[-0.5/fs/stride, (D+0.5)/fs/stride, -0.5, d.shape[0]+0.5]
            d = d / np.max(np.abs(d)) / np.max(dpc_mag[:, oc])**0.5 * dpc_mag[di, oc]**0.5
            ax[oi, di + oset].imshow(zoom(np.fliplr(d)[:,:D], z), extent=extent, **imopts)
            for ee in dpc_edges:
                ax[oi, di+oset].axhline(ee, color='k', lw=0.5, linestyle='--')
            ax[oi, di + oset + 1].set_yticklabels([])
        #yl = ax[oi, -1].get_ylim()

        ax[oi, -1].plot(dpc_mag[:, oc] / dpc_mag[:, oc].sum())
        ax[oi, -1].set_ylim([0, 0.5])
        str_extra = ''

        if (oset==1):
            if (cellid in dfinfo.index):
                str_extra = f"\nsw={dfinfo.loc[cellid,'sw']:.2f} d={dfinfo.loc[cellid,'depth']}"
        if cellid!='-ALL':
            ax[oi, -1].text(0, 0.5, f"{cellids[oc]}\nr={r_test[oc]:.3f}\npcc={pcpc1}{str_extra}",
                            fontsize=6, va='top')
    plt.tight_layout()
    return f


def plot_dpc_space(modelspec=None, cell_list=None, val=None, est=None, modelspec2=None, show_preds=True, plot_stim=True,
                   use_val=False, print_figs=False, **ctx):
    if cell_list is None:
        cell_list = ctx['cellids']
    elif type(cell_list) is str:
        cell_list = [cell_list]
    if use_val:
        rec=val
    else:
        rec=est

    orange = [i for i,c in enumerate(rec['resp'].chans) if c in cell_list]
    spont = np.zeros(len(rec['resp'].chans))

    if modelspec2 is not None:
        lnstrf=LN.LNpop_get_strf(modelspec2)
    else:
        lnstrf=None
    for oi, cellid in zip(orange,cell_list):
        print(oi, cellid)

        X = rec['stim'].as_continuous().T
        Y = project_to_subspace(modelspec=modelspec, X=X, out_channels=[oi])[0].T

        dpc = modelspec.meta['dpc']
        dpc = np.moveaxis(dpc, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]
        dpc_mag = modelspec.meta['dpc_mag']
        pc_count = dpc.shape[1]

        pred = rec['pred'].as_continuous().T[:, oi]
        r = rec['resp'].as_continuous().T[:, oi]

        pcp = 3
        imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
                  'interpolation': 'none'}
        if show_preds:
            f, ax = plt.subplots(pcp, 4, figsize=(6, 4.5))
        else:
            f, ax = plt.subplots(pcp, 3, figsize=(3, 4.5))
        for i in range(pcp):
            d = modelspec.meta['dpc'][oi, i]
            d = d / np.max(np.abs(d))  # / dpc_magz[0, oi] * dpc_magz[i, oi]

            if i == 0:
                ax[i, 1].imshow(np.fliplr(d), **imopts)
                ax[i, 1].set_ylabel(f'Dim {i + 1}')
            else:
                ax[i, 0].imshow(np.fliplr(d), **imopts)
                ax[i, 0].set_ylabel(f'Dim {i + 1}')

        ax[0, 0].plot(np.arange(1, dpc_mag.shape[0] + 1), dpc_mag[:, oi] / dpc_mag[:, oi].sum(), 'o-', markersize=3)
        ax[0, 0].set_xlabel('PC dimension')
        ax[0, 0].set_ylabel('Var. explained')
        ax[0, 0].set_xticks(np.arange(1, dpc_mag.shape[0] + 1))
        for j in range(1, pcp):
            Zresp = None
            Zpred = None
            ac,bc,Zresp,N=histmean2d(Y[:,0],Y[:,j],r, bins=20, ax=ax[j,1], spont=spont[oi], ex_pct=ex_pct, Z=Zresp)
            ax[j, 1].set_xlabel('Dim 1')
            ax[j, 1].set_ylabel(f"Dim {j + 1}")
            if show_preds:
                ac,bc,Zpred,N=histmean2d(Y[:,0],Y[:,j],pred, bins=20, ax=ax[j,2], spont=spont[oi], ex_pct=ex_pct, Z=Zpred)
                ax[j, 2].set_xlabel('Dim 1')
                ax[j, 2].set_ylabel(f"Dim {j + 1}")
        if show_preds & (lnstrf is not None):
            l = lnstrf[:,:,oi]
            l /= np.abs(l).max()
            ax[0, 2].imshow(lnstrf[:,:15,oi], **imopts)
            ax[0, 2].set_title(f"LN: {modelspec2.meta['r_test'][oi, 0]:.3f}")

        ax[0, 0].set_title(cellid)
        if 'sspredxc' in modelspec.meta.keys():
            ax[0, 1].set_title(f"CNN: {modelspec.meta['r_test'][oi, 0]:.3f} SS: {modelspec.meta['sspredxc'][oi]:.3f}")
        else:
            ax[0, 1].set_title(f"CNN: {modelspec.meta['r_test'][oi, 0]:.3f}")

        ymin, ymax = 1,0
        for pci in range(3):
            y = Y[:, pci]
            b = np.linspace(y.min(), y.max(), 11)
            mb = (b[:-1] + b[1:]) / 2
            mr = [np.mean(r[(y >= b[i]) & (y < b[i + 1])]) for i in range(10)]
            me = [np.std(r[(y >= b[i]) & (y < b[i + 1])]) / np.sqrt(np.sum((y >= b[i]) & (y < b[i + 1]))) for i in range(10)]
            ax[pci,3].errorbar(mb, mr, me)
            ax[pci,3].set_xlabel(f'Dim {pci+1} proj')
            ax[pci,3].set_ylabel('Mean prediction')
            yl=ax[pci,3].get_ylim()
            ymin = np.min([yl[0],ymin])
            ymax = np.max([yl[1], ymax])
        for pci in range(3):
            ax[pci, 3].set_ylim((ymin,ymax))
        ax[0,2].set_axis_off()
        plt.tight_layout()

    return f

def _plot_with_contour(ax, minN=5, level=0.7, **kwargs):
    if type(level) is not list:
        level = [level]

    ac, bc, Z, N = histmean2d(ax=ax, minN=minN, **kwargs)
    allbins = np.isfinite(Z).sum()
    Znan = np.isnan(Z)
    Z = Z - np.nanmin(Z)
    Z[Znan] = 0
    Z = smooth(Z, window_len=5, axis=(0, 1))
    if 'Z' not in kwargs.keys():
        Z[N < minN] = 0
    Z = Z / np.max(Z)

    ii=0
    colors = ['k','k']
    #ax.contourf(ac, bc, Z, levels=level + [1], colors=colors, alpha=0.2)
    ax.contour(ac, bc, Z, levels=level + [1], colors=colors, alpha=1)

    # count
    highbins = (Z>level[0]).sum()
    return highbins,allbins

def plot_dpc_rows(modelspec=None, cell_list=None, modelspecln=None, use_val=False,
                  est=None, val=None, maxrows=15, df=None, w=None, use_dpc_all=False,
                  z=2, title=None, emax=100000, overlay_traces=True, stim_banks=1,
                  dpc_timecourse=False, 
                  T1=270, T2=470, o1=20, o2=60, cmap='Greens', plot_contours=False,
                  ex_pct=0.025, minN=2, show_valid_edges=False, plot_pred=True, stride=None,
                  **ctx):

    """ top 3 dPCs for cells in cell_list """
    if cell_list is None:
        cell_list = modelspec.meta['cellids'][:4]
    siteid = cell_list[0].split("-")[0]

    orange = [est['resp'].chans.index(c) for c in cell_list]
    log.info(f"{cell_list}: {orange}")
    if 'spont_mean' in modelspec.meta.keys():
        spont = modelspec.meta['spont_mean']
    else:
        spont = np.zeros(est['resp'].shape[0])

    if modelspecln is not None:
        from nems.models import LN
        lnstrf = LN.LNpop_get_strf(modelspecln)
    else:
        lnstrf = None
    fs=est['stim'].fs

    if stride is None:
        stride = int(val['stim'].fs/val['resp'].fs)
        
    X = np.concatenate([val['stim'].as_continuous().T,
                        est['stim'].as_continuous().T[:emax*stride,:]], axis=0)
    Y = project_to_subspace(modelspec=modelspec, X=X, out_channels=orange,
                            pc_count=3, use_dpc_all=use_dpc_all, verbose=True)
    cellcount = modelspec.meta['dpc'].shape[0]

    if stride>0:
        log.info(f"stim len {est['stim'].shape[-1]} ds to {est['stim'].shape[-1]/stride} to match resp len {est['resp'].shape[-1]}")
        log.info(f"{X.shape} {Y.shape}")
        X=X[::stride]
        Y=Y[:,:,::stride]
        log.info(f"{X.shape} {Y.shape}")

    if use_dpc_all:
        dpc = np.concatenate([modelspec.meta['dpc_all']]*cellcount,axis=0)
        dpc_mag = np.concatenate([modelspec.meta['dpc_mag_all']]*cellcount,axis=1)
    else:
        dpc = modelspec.meta['dpc'].copy()
        dpc_mag = modelspec.meta['dpc_mag'].copy()
    #dpc_mag_sh = modelspec.meta['dpc_mag_sh']
    #dpc_mag_e = modelspec.meta['dpc_mag_e']
    pc_count = dpc.shape[1]

    rows = np.min([len(orange), maxrows])
    f = plt.figure(figsize=(13*0.9, (rows + 1) * 0.9))
    gs = f.add_gridspec(rows + 1, 13)
    ax = np.zeros((rows + 1, 9), dtype='O')
    for r in range(rows + 1):
        for c in range(9):
            if (c < 8) & (r > 0):
                ax[r, c] = f.add_subplot(gs[r, c])
            elif (c >= 8):
                ax[r, c] = f.add_subplot(gs[r, c:])

    # top row, just plot stim in one panel
    ss = val['stim'].as_continuous()[:, (T1*stride):(T2*stride)]
    ax[0, -1].imshow(ss, aspect='auto', origin='lower', cmap='gray_r')
    ax[0, -1].set_yticklabels([])
    ax[0, -1].set_xticklabels([])

    for j_, (oi, cellid) in enumerate(zip(orange[:rows], cell_list[:rows])):
        j = j_ + 1
        log.info(f"{j}/{rows} cid={oi} cellid={cellid}")

        if plot_pred:
            pred = np.concatenate([val['pred'].as_continuous()[oi,:],
                                   est['pred'].as_continuous()[oi, :emax]])
        else:
            pred = np.concatenate([val['resp'].as_continuous()[oi,:],
                                   est['resp'].as_continuous()[oi, :emax]])
        log.info(f"{X.shape} {Y.shape} {pred.shape}")
        pcp = 3
        for i in range(pcp):
            #cc = np.corrcoef(Y[j_, i], r)[0, 1]
            cc = dpc[oi, i, :, -5*stride:].mean()
            if cc < 0:
                Y[j_, i] = -Y[j_, i]
                dpc[oi, i] = -dpc[oi, i]
        dp = dpc_mag[:, oi] / dpc_mag[:, oi].sum()

        imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
                  'interpolation': 'none'}
        ymin, ymax = 1, 0
        for i in range(pcp):
            d = dpc[oi, i]
            #if stride>1:
            #    d = np.pad(d,[[0,0],[0,stride-1]])
            d = d / np.max(np.abs(d)) * 1.25  # / dpc_magz[0, oi] * dpc_magz[i, oi]
            prat = dp[i] / dp[0]
            d *= prat
            if stim_banks==1:
                ax[j, i].imshow(zoom(np.fliplr(d)[:,1:w], z), **imopts)
            else:
                if w is None:
                    h, w = d.shape
                else:
                    h = d.shape[0]
                m = int(h/stim_banks)
                ds = np.concatenate([d[:m,-w:], np.zeros((2, w)), d[m:,-w:]], axis=0)
                ax[j, i].imshow(np.fliplr(ds), **imopts)
                ax[j, i].set_yticks([])
                ax[j, i].spines[['left']].set_visible(False)
                ax[j, i].axhline(m+2, lw=0.5, color='k')
                x0 = ax[j, i].get_xlim()[0]
                y0,y1 = ax[j, i].get_ylim()
                ax[j, i].plot([x0, x0], [y0, y0+m], lw=0.5, color='k')
                ax[j, i].plot([x0, x0], [y1-m, y1], lw=0.5, color='k')
            ax[j, i].set_xticklabels([])
            ax[j, i].set_yticklabels([])
        #ax[j,0].set_ylabel('Contra - Ipsi')
        # PC0 vs. PC1 heatmap
        Z = [None, None]
        for p2 in range(1,pcp):
            if (j==rows) & (p2==pcp-1):
                #add_colorbar=True
                add_colorbar=False
            else:
                add_colorbar=False
            if plot_contours:
                ac,bc,Z[p2-1],_ = histmean2d(Y[j_, 0, :], Y[j_, p2, :], pred, bins=20,
                            cmap=cmap, spont=spont[oi], ex_pct=ex_pct, minN=minN, show_plot=False)

                ha,alla = _plot_with_contour(a=Y[j_, 0, :], b=Y[j_, p2, :], d=pred, Z=Z[p2-1], bins=20, ax=ax[j, p2+2],
                            cmap=cmap, spont=spont[oi], ex_pct=ex_pct, minN=minN, add_colorbar=add_colorbar)
                overlay_traces=False
            else:
                ac,bc,Z[p2-1],N = histmean2d(Y[j_, 0, :], Y[j_, p2, :], pred, bins=20, ax=ax[j, p2+2],
                            cmap=cmap, spont=spont[oi], ex_pct=ex_pct, minN=minN, add_colorbar=add_colorbar)
                if show_valid_edges:
                    image = (N >= minN).astype(int)

                    frm = lambda x, y: image[int(y), int(x)]
                    grm = np.vectorize(frm)
                    dx=(ac[1]-ac[0])
                    dy=(bc[1]-bc[0])
                    xx = np.linspace(0, image.shape[1], image.shape[1] * 100)
                    yy = np.linspace(0, image.shape[0], image.shape[0] * 100)
                    XX, YY = np.meshgrid(xx[:-1], yy[:-1])
                    ZZ = grm(XX[:-1], YY[:-1])

                    ax[j, p2+2].contour(ZZ[::-1], [0.5], colors='darkgreen', linewidths=[0.5],
                                        extent=[ac[0]-dx, ac[-1]+dx, bc[0]-dy, bc[-1]+dy],
                                        origin='upper')

        zz = np.stack(Z).flatten()
        zz = zz[np.isfinite(zz)]
        vmin, vmax = np.percentile(zz, [5, 95])

        for p2 in range(1, pcp):
            # if plot_contours:
            #     ha, alla = _plot_with_contour(a=Y[j_, 0, :], b=Y[j_, p2, :], d=pred, Z=Z[p2-1], bins=20,
            #                                   ax=ax[j, p2 + 2],
            #                                   cmap='summer', spont=spont[oi], ex_pct=ex_pct, minN=2,
            #                                   add_colorbar=add_colorbar)
            #     overlay_traces = False
            # else:
            #     histmean2d(Y[j_, 0, :], Y[j_, p2, :], pred, bins=20, ax=ax[j, p2 + 2],
            #                cmap=cmap, spont=spont[oi], ex_pct=ex_pct, minN=minN,
            #                vmin=vmin, vmax=vmax, Z=Z[p2-1])
            if overlay_traces:
                ax[j, p2+2].plot(Y[j_, 0, (T1+o1):(T1+o2)], Y[j_, p2, (T1+o1):(T1+o2)], lw=0.5, color='k')
            ax[j, p2+2].set_yticklabels([])
            ax[j, p2+2].set_xticklabels([])
        #log.info(f"{cellid} p={p2} {ax[j, p2+2].get_xlim()} {Y[j_, 0].min()}, {Y[j_, 0].max()} {Y[j_, 0, (T1+o1):(T1+o2)].min()}, {Y[j_, 0, (T1+o1):(T1+o2)].max()}")

        asym=np.zeros(pcp)
        for p1 in range(pcp):
            stimmag = np.sum(X,axis=1)
            minx = np.min(stimmag)
            xx = Y[j_, p1, stimmag > minx]
            yy = pred[stimmag > minx]
            bins = 25

            x_, y_, asym[p1] = nl_marginal(xx, yy, smoothwin=7, ex_pct=ex_pct, bins=bins)
            bd = 0.33
            if asym[p1]<-bd:
                c_ = 0
            elif asym[p1]<bd:
                c_ = 1
            else:
                c_ = 2
            #ax[j, -2].plot(x_[:-1], dy, color=CB_color_cycle[p1])
            ax[j, p1+5].plot(x_, y_, color=CB_color_cycle[c_])
            ax[j, p1+5].axvline(0, color='gray', lw=0.5, linestyle='--')
            ax[j, p1+5].text(ax[j, p1+5].get_xlim()[0], ax[j, p1+5].get_ylim()[1],
                           f"{asym[p1]:.2f}", fontsize=6, va='top')
            ax[j, p1+5].set_xticklabels([])
            ax[j, p1+5].set_yticklabels([])
            #x0 = np.argmin(np.abs(x_))
            #y_ -= y_[x0]
            #asym[p1] = np.abs((y_[x_ > 0].mean() - y_[x_ < 0].mean())) / (np.abs(y_[x_ > 0].mean()) + np.abs(y_[x_ < 0].mean()))
            #net[p1] = y_.mean() / np.abs(y_).mean()
        #tt = "\n".join([f"{a:.1f}/{n:.1f}" for a,n in zip(asym,net)])
        #ax[j, -2].text(ax[j,-2].get_xlim()[0],ax[j,-2].get_ylim()[1], tt,
        #               fontsize=6, va='top')

        # snippet of resp/pred PSTH
        rr = val['resp'].as_continuous()[oi, T1:T2]
        pp = val['pred'].as_continuous()[oi, T1:T2]
        pp = pp - pp.mean()
        r0 = rr.mean()
        pp = pp / pp.std() * (rr - r0).std()
        pp = pp + r0
        ax[j, -1].plot(pp, color='gray', lw=0.5)
        ax[j, -1].plot(rr, color='black', lw=0.5)
        if dpc_timecourse:
            y01 = Y[j_, 0:2, T1:T2]
            y01 -= y01.min()
            y01 = y01/y01.max() * pp.max()
            ax[j, -1].plot(y01[0], lw=0.5, color=CB_color_cycle[0])
            ax[j, -1].plot(y01[1], lw=0.5, color=CB_color_cycle[1])

        ax[j, -1].set_yticklabels([])
        ax[j, -1].set_xticklabels([])
        ax[j, -1].set_xlim([0, T2 - T1])
        ax[j, -1].axvline(o1, lw=0.5, color='red')
        ax[j, -1].axvline(o2, lw=0.5, color='red')
        yl = ax[j, -1].get_ylim()
        try:
            if df is None:
                df = depth.get_depth_details([siteid], verbose=False)
            dep = df.loc[cellid, 'depth']
            sw = df.loc[cellid, 'sw']
        except:
            dep, sw = 0, 0
        if 'sspredxc' in modelspec.meta.keys():
            ax[j, -1].text(0, yl[1], f"r={modelspec.meta['r_test'][oi, 0]:.3f} ss={modelspec.meta['sspredxc'][oi]:.3f} dp={dep} sw={sw:.2f}")
        else:
            ax[j, -1].text(0, yl[1], f"r={modelspec.meta['r_test'][oi, 0]:.3f}")

        ax[j, 0].text(0,ax[j,0].get_ylim()[-1],f"{cellid}", fontsize=6, va='center')

    ax[-1,0].set_xticks([0,10],[0,100])
    ax[-1,0].set_xlabel('Time (ms)')
    ax[-1,-1].set_xticks(np.arange(0,T2-T1,50)/fs)
    ax[-1,-1].set_xlabel('Time (s)')

    for pci in range(pcp):
        ax[1, pci].set_title(f'Dim {pci + 1}')
        ax[1, pci+5].set_title(f'Dim {pci + 1}')
        if pci>0:
           ax[1, pci + 2].set_title(f'Dim 1 v {pci+1}')
    if title is not None:
        f.suptitle(title)
    plt.tight_layout()
    return f

def compute_subspace_density(Y, d, bins=16, ex_pct=1):
    a = Y[:, 0]
    b = Y[:, 1]
    c = Y[:, 2]

    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    ab = np.percentile(a[keep], [ex_pct, 100 - ex_pct])
    bb = np.percentile(b[keep], [ex_pct, 100 - ex_pct])
    cb = np.percentile(c[keep], [ex_pct, 100 - ex_pct])
    av = np.linspace(ab[0], ab[1], bins + 1)
    bv = np.linspace(bb[0], bb[1], bins + 1)
    cv = np.linspace(cb[0], cb[1], bins + 1)

    x, y, z = np.mgrid[ab[0]:ab[1]:(ab[1] - ab[0]) / (bins + 1),
              bb[0]:bb[1]:(bb[1] - bb[0]) / (bins + 1),
              cb[0]:cb[1]:(cb[1] - cb[0]) / (bins + 1)]

    mmv = np.zeros((bins, bins, bins))
    N = np.zeros_like(mmv, dtype=int)
    for i_, (a1,a2) in enumerate(zip(av[:-1], av[1:])):
        for j_, (b1,b2) in enumerate(zip(bv[:-1], bv[1:])):
            for k_, (c1,c2) in enumerate(zip(cv[:-1], cv[1:])):
                v_ = (a >= a1) & (a < a2) & (b >= b1) & (b < b2) & \
                     (c >= c1) & (c < c2) & np.isfinite(d)
                if (v_.sum() > 0):
                    mmv[k_, j_, i_] = np.nanmean(d[v_])
                    N[k_, j_, i_] = v_.sum()
    #mmv=np.swapaxes(mmv, 0, 2)
    #N=np.swapaxes(N, 0, 2)
    
    return mmv, N

def plot_subspace_density(mmv, level=0.5, ax=None, ci=0, color=None, alpha=0.4):

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure
    from scipy.ndimage import gaussian_filter

    if ax is None:
        ax = plt.figure().add_subplot(1, 1, 1, projection='3d')

    ds = gaussian_filter(mmv, 0.75)
    ds = ds / ds.max()
    ds = np.pad(ds, 1, constant_values=0)

    verts, faces, normals, values = measure.marching_cubes(ds, level)

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    if color is None:
        color = CB_color_cycle[ci]
    mesh.set(edgecolor=None, facecolor=color, linewidth=0.1, alpha=alpha)
    ax.add_collection3d(mesh)
    s = ds.shape
    ax.set_xlim(0, s[0]-2)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, s[1]-2)  # b = 10
    ax.set_zlim(0, s[2]-2)  # c = 16
    ax.view_init(elev=20, azim=-30, roll=0)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    return ax

def plot_dpc_space_3d(modelspec=None, cell_list=None, val=None, est=None, modelspec2=None, show_preds=True, plot_stim=True,
                      use_val=False, emax=100000, level=0.5, use_dpc_all=True, ex_pct=1, cmap='Greens', **ctx):

    import matplotlib as mpl
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure
    from scipy.ndimage import gaussian_filter

    if cell_list is None:
        cell_list = [ctx['cellids'][0]]
    elif type(cell_list) is str:
        cell_list = [cell_list]
    if use_val:
        rec = val
    else:
        rec = est

    orange = [i for i,c in enumerate(rec['resp'].chans) if c in cell_list]

    if modelspec2 is not None:
        lnstrf = LN.LNpop_get_strf(modelspec2)
    else:
        lnstrf = None

    pcp = 3
    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
              'interpolation': 'none'}
    i_lookup = [4, 1, 2]

    f = plt.figure()
    fontsize=6
    ax = [f.add_subplot(3, 3, i_lookup[i]) for i in range(pcp)]
    ax1 = f.add_subplot(3,3,9)
    ax2 = f.add_subplot(3,3,6)
    ax3 = f.add_subplot(3,3,8)
    axsp = f.add_subplot(3,3,5, projection='3d')
    for ii, (oi, cellid) in enumerate(zip(orange, cell_list)):
        log.info(f"{oi} {cellid} dpc all={use_dpc_all}")
        if use_dpc_all:
            dpcz = modelspec.meta['dpc_all']
            dpc_magz = modelspec.meta['dpc_mag_all']
            dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, 0]
        else:
            dpcz = modelspec.meta['dpc']
            dpc_magz = modelspec.meta['dpc_mag']
            dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]

        pred = rec['pred'].as_continuous().T[:emax, oi]
        r = rec['resp'].as_continuous().T[:emax, oi]

        if (use_dpc_all==False) | (ii==0):
            X = rec['stim'].as_continuous().T[:emax, :]
            Y = project_to_subspace(modelspec=modelspec, X=X, out_channels=[oi], pc_count=pcp, use_dpc_all=use_dpc_all)[0].T

            for i in range(Y.shape[1]):
                cc=np.corrcoef(Y[:,i],r)[0,1]
                if cc < 0:
                    Y[:,i]=-Y[:,i]
                    dpcz[:,:,i] = -dpcz[:,:,i]

        #log.info(f"Calling compute_subspace_density...")
        mmv, N = compute_subspace_density(Y, pred, ex_pct=ex_pct)

        keep = np.isfinite(Y[:, 0]) & np.isfinite(Y[:, 1]) & np.isfinite(Y[:, 2])
        ab = np.percentile(Y[keep,0], [ex_pct, 100 - ex_pct])
        bb = np.percentile(Y[keep,1], [ex_pct, 100 - ex_pct])
        cb = np.percentile(Y[keep,2], [ex_pct, 100 - ex_pct])
        bins = mmv.shape[0]
        av = np.linspace(ab[0], ab[1], bins)
        bv = np.linspace(bb[0], bb[1], bins)
        cv = np.linspace(cb[0], cb[1], bins)

        log.info(f'Computing 3d subspace projection, level={level}')
        Z = mmv.copy()
        Z[N < 3] = np.nan

        for i in range(pcp):
            d = dpcz[:,:,i].T
            d = d / np.max(np.abs(d))  # / dpc_magz[0, oi] * dpc_magz[i, oi]
            ax[i].imshow(np.fliplr(d), **imopts)
            ax[i].set_title(f'Dim {i + 1}')
        Zz = np.nanmean(Z,axis=0)
        Zy = np.nanmean(Z,axis=1)
        Zx = np.nanmean(Z,axis=2)
        Zz[np.isnan(Zz)]=0
        Zy[np.isnan(Zy)]=0
        Zx[np.isnan(Zx)]=0

        Zz = smooth(Zz, window_len=5, axis=(0,1))
        Zy = smooth(Zy, window_len=5, axis=(0,1))
        Zx = smooth(Zx, window_len=5, axis=(0,1))

        if ii==0:
            Zzcount = np.zeros_like(Zz, dtype=int)
            Zycount = np.zeros_like(Zy, dtype=int)
            Zxcount = np.zeros_like(Zx, dtype=int)

        Zzcount = Zzcount + (Zz/np.nanmax(Zz)>=level[0]).astype(int)
        Zycount = Zycount + (Zy/np.nanmax(Zy)>=level[0]).astype(int)
        Zxcount = Zxcount + (Zx/np.nanmax(Zx)>=level[0]).astype(int)

        print(ii, Zzcount.sum(), Zycount.sum(), Zxcount.sum())
        vmm = np.min([np.max(Zz), np.max(Zy), np.max(Zx)])*1.4

        cmap3d = mpl.colormaps[cmap]
        if use_dpc_all:
            colors = [CB_color_cycle[ii % len(CB_color_cycle)]] * (len(level)+1)
            ax1.contourf(cv, bv, Zz/np.nanmax(Zz), levels=level+[1], colors=colors, alpha=0.2)
            ax2.contourf(cv, av, Zy/np.nanmax(Zy), levels=level+[1], colors=colors, alpha=0.2)
            ax3.contourf(bv, av, Zx/np.nanmax(Zx), levels=level+[1], colors=colors, alpha=0.2)

        else:
            colors = cmap3d(np.linspace(0, 1, len(level)+1))
            ax1.imshow(Zz, extent=[cb[0],cb[1],bb[0],bb[1]], vmax=vmm, cmap=cmap, aspect='equal', origin='lower')
            ax1.contour(cv, bv, Zz/np.nanmax(Zz), levels=level, colors=colors)

            ax2.imshow(Zy, extent=[cb[0],cb[1],ab[0],ab[1]], vmax=vmm, cmap=cmap, aspect='equal', origin='lower')
            ax2.contour(cv, av, Zy/np.nanmax(Zy), levels=level, colors=colors)

            ax3.imshow(Zx, extent=[bb[0],bb[1],ab[0],ab[1]], vmax=vmm, cmap=cmap, aspect='equal', origin='lower')
            ax3.contour(bv, av, Zx/np.nanmax(Zx), levels=level, colors=colors)

        ax1.set_xlabel('Dim 1', fontsize=fontsize)
        ax1.set_ylabel('Dim 2', fontsize=fontsize)
        ax2.set_xlabel('Dim 1', fontsize=fontsize)
        ax2.set_ylabel('Dim 3', fontsize=fontsize)
        ax3.set_xlabel('Dim 2', fontsize=fontsize)
        ax3.set_ylabel('Dim 3', fontsize=fontsize)

        if type(level) is not list:
            level=[level]

        if use_dpc_all:
            colors = [CB_color_cycle[ii % len(CB_color_cycle)]] * (len(level)+1)
            alphas=np.linspace(0.2, 1, len(level)+1)
        else:
            # Take colors at regular intervals spanning the colormap.
            colors = cmap3d(np.linspace(0, 1, len(level)+1))
            #colors=colors[:,:-1]
            alphas=np.linspace(0.3, 1, len(level))
            colors[1:,-1]=alphas

        for i,l in enumerate(level):
            log.info(f"level {l} color={colors[i+1]}, alpha={alphas[i]}")
            plot_subspace_density(np.swapaxes(mmv, 0, 2), level=l, ax=axsp, color=colors[i+1], alpha=alphas[i])
        nn = np.arange(Zz.shape[0])
        xt = np.array([0,int(bins/2)-1,int(bins/2)*2-1])
        axsp.set_xticks(xt,np.round(av[xt],1))
        axsp.set_yticks(xt,np.round(bv[xt],1))
        axsp.set_zticks(xt,np.round(cv[xt],1))

        # Find contours at a constant value of 0.8

        if use_dpc_all:
            colors = [CB_color_cycle[ii % len(CB_color_cycle)]] * (len(level)+1)
            alphas=np.linspace(0.3, 1, len(level)+1)
        else:
            colors = cmap3d(np.linspace(0, 1, len(level)+1))
        zf=3
        for i,l in enumerate(level):
            z = zoom(Zz,zf)/vmm
            contours_all = measure.find_contours(z, l)
            for contours in contours_all:
                #log.info(f"level {l} color={colors[i + 1]}")
                axsp.plot(contours[:,1]/zf, contours[:,0]/zf,
                        color=colors[i+1], lw=0.5)

            z = zoom(Zy,zf)/vmm
            contours_all = measure.find_contours(z, l)
            for contours in contours_all:
                axsp.plot(contours[:,1]/zf, np.zeros_like(contours[:,0])+bins-1,
                        contours[:,0]/zf, color=colors[i+1], lw=0.5)

            z = zoom(Zx,zf)/vmm
            contours_all = measure.find_contours(z, l)
            for contours in contours_all:
                axsp.plot(np.zeros_like(contours[:,0]), contours[:,1]/zf,
                        contours[:,0]/zf, color=colors[i+1], lw=0.5)

    if use_dpc_all:
        axcount = f.add_subplot(3, 3, 3)
        im = axcount.imshow(Zzcount, extent=[cb[0], cb[1], bb[0], bb[1]], cmap=cmap, aspect='equal', origin='lower')
        plt.colorbar(im, ax=axcount)
        axcount2 = f.add_subplot(3, 3, 7)
        im = axcount2.imshow(Zycount, extent=[cb[0], cb[1], ab[0], ab[1]], cmap=cmap, aspect='equal', origin='lower')
        plt.colorbar(im, ax=axcount2)

    plt.tight_layout()
    return f


def plot_common_dpc_space_2d(modelspec=None, cell_list=None, val=None, est=None, modelspec2=None,
                             show_preds=True, plot_stim=True, use_val=False, emax=100000, level=0.5,
                             use_dpc_all=True, pc_count=3, ex_pct=1, cmap='Greens',
                             return_stats=False, Nmin=5,
                             fill_colors=None, outline_colors=None, celltypes=None, **ctx):

    if cell_list is None:
        cell_list = [ctx['cellids'][0]]
    elif type(cell_list) is str:
        cell_list = [cell_list]
    if use_val:
        rec = val
    else:
        rec = est

    orange = [i for i, c in enumerate(rec['resp'].chans) if c in cell_list]

    if modelspec2 is not None:
        lnstrf = LN.LNpop_get_strf(modelspec2)
    else:
        lnstrf = None

    pcp = pc_count
    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
              'interpolation': 'none'}

    if type(level) is not list:
        level = [level]

    pairs = [[0, 1], [0, 2], [1, 2]]
    Zlist = []
    Zshlist = []
    Nlist = []
    Nbins = np.zeros((len(cell_list),len(pairs)), dtype=int)
    aclist = []
    bclist = []

    bins=20
    Zall = np.zeros((len(cell_list), len(pairs), bins, bins))

    f, ax = plt.subplots(pc_count, 4)
    fontsize = 6
    for ii, (oi, cellid) in enumerate(zip(orange, cell_list)):
        #log.info(f"{oi} {cellid} dpc all={use_dpc_all}")

        pred = rec['pred'].as_continuous().T[:emax, oi]
        r = rec['resp'].as_continuous().T[:emax, oi]
        X = rec['stim'].as_continuous().T[:emax, :]

        if use_dpc_all:
            dpcz = modelspec.meta['dpc_all']
            dpc_magz = modelspec.meta['dpc_mag_all']
            dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, 0]
        else:
            dpcz = modelspec.meta['dpc']
            dpc_magz = modelspec.meta['dpc_mag']
            dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]

        if (use_dpc_all == False) | (ii == 0):
            Y = project_to_subspace(modelspec=modelspec, X=X, out_channels=[oi],
                                    pc_count=pc_count, use_dpc_all=use_dpc_all)[0].T

            for i in range(pc_count):
                #cc = np.corrcoef(Y[:, i], r)[0, 1]
                cc = dpcz[-5:, :, i].mean()

                if cc < 0:
                    Y[:, i] = -Y[:, i]
                    dpcz[:, :, i] = -dpcz[:, :, i]

                d = dpcz[:, :, i].T
                d = d / np.max(np.abs(d))
                ax[i,0].imshow(zoom(np.fliplr(d)[:,:11],2), **imopts)
                ax[i,0].set_title(f'Dim {i + 1}')

        for pairidx in range(len(pairs)):
            p1,p2 = pairs[pairidx]
            ac, bc, Z, N = histmean2d(Y[:,p1], Y[:,p2], pred, bins=bins, cmap=cmap, ex_pct=ex_pct,
                                     show_plot=False)
            Znan = np.isnan(Z)
            Z = Z-np.nanmin(Z)
            Z[Znan]=0
            Z = smooth(Z, window_len=5, axis=(0, 1))
            Z[N<Nmin] = 0
            Z = Z / np.max(Z)
            Zall[ii,pairidx] = Z
            #Zsh = np.random.permutation(Z.flatten()).reshape(Z.shape)

            aclist.append(ac)
            bclist.append(bc)

            if ii == 0:
                Zlist.append(np.zeros_like(Z, dtype=int))
                randcount = 10
                Zshlist.append(np.zeros(list(Z.shape)+[randcount], dtype=int))
                Nlist.append(N)
            Nbins[ii,pairidx] = (Z >= level[0]).sum()
            if Nbins[ii,pairidx] in [0,360]:
                print(f"weird Nbins value?")
            Zlist[pairidx] = Zlist[pairidx] + (Z >= level[0]).astype(int)
            for rr in range(randcount):
                #Zsh = np.random.permutation(Z.flatten()).reshape(Z.shape)
                Zsh = np.roll(Z,np.random.randint(20, size=2), axis=(0, 1))
                Zshlist[pairidx][:, :, rr] = Zshlist[pairidx][:,:,rr] + (Zsh >= level[0]).astype(int)

            if fill_colors is None:
                colors = [CB_color_cycle[ii % len(CB_color_cycle)]] * (len(level) + 1)
            else:
                colors = fill_colors[ii]
            ax[pairidx,1].contourf(ac, bc, Z, levels=level + [1], colors=colors, alpha=0.2)

            if outline_colors is not None:
                ax[pairidx, 1].contour(ac, bc, Z, levels=level + [1], colors=[outline_colors[ii]], linewidths=0.5)

        if (ii+1)%10==0:
            log.info(f"{ii+1} max olap {[z.max() for z in Zlist]} shuff {[z.max() for z in Zshlist]}")

        # ax1.set_xlabel('Dim 1', fontsize=fontsize)
        # ax1.set_ylabel('Dim 2', fontsize=fontsize)
        # ax2.set_xlabel('Dim 1', fontsize=fontsize)
        # ax2.set_ylabel('Dim 3', fontsize=fontsize)
        # ax3.set_xlabel('Dim 2', fontsize=fontsize)
        # ax3.set_ylabel('Dim 3', fontsize=fontsize)

    if use_dpc_all:
        for pairidx in range(len(pairs)):
            extent = [aclist[pairidx][0], aclist[pairidx][-1], bclist[pairidx][0], bclist[pairidx][-1]]
            im = ax[pairidx, 2].imshow(Zlist[pairidx], extent=extent, cmap=cmap, origin='lower') # aspect='equal',
            mask = (Nlist[pairidx]>=Nmin).astype(float)
            mask = np.repeat(mask, 100, axis=0)
            mask = np.repeat(mask, 100, axis=1)
            ax[pairidx, 2].contour(mask, levels=[0.5],
                                   linewidths=[0.5], colors=['k'],
                                   extent=extent)
            # extent=[extent[0]-astep, extent[1]+astep, extent[2]-bstep, extent[3]+bstep]
            plt.colorbar(im, ax=ax[pairidx, 2])
            im = ax[pairidx, 3].imshow(Zshlist[pairidx][:,:,0], extent=extent, cmap=cmap, origin='lower') # aspect='equal',
            plt.colorbar(im, ax=ax[pairidx, 3])

    plt.tight_layout()

    if return_stats:
        return f, Zlist, Zshlist, Nlist, Nbins, Zall
    else:
        return f


def plot_dpc_all_3d(modelspec=None, cell_list=None, val=None, est=None, modelspec2=None, show_preds=True, plot_stim=True,
                      use_val=False, print_figs=False, level=0.5, use_dpc_all=True,
                    title=None, **ctx):

    if cell_list is None:
        cell_list = [ctx['cellids'][0]]
    elif type(cell_list) is str:
        cell_list = [cell_list]
    if use_val:
        rec = val
    else:
        rec = est
    if title is None:
        title = ",".join(cell_list)

    orange = [i for i,c in enumerate(rec['resp'].chans) if c in cell_list]

    X = rec['stim'].as_continuous().T
    
    if use_dpc_all:
        dpcz = modelspec.meta['dpc_all']
        dpc_magz = modelspec.meta['dpc_mag_all']
        dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, 0]
        orange=[0]
        Y = project_to_subspace(modelspec, X, use_dpc_all=use_dpc_all)[0].T
    else:
        dpcz = modelspec.meta['dpc']
        dpc_magz = modelspec.meta['dpc_mag']
        dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, orange[0]]
        Y = project_to_subspace(modelspec, X, out_channels=orange, use_dpc_all=use_dpc_all)[0].T

    f = plt.figure()

    pcp = 3
    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
              'interpolation': 'none'}
    i_lookup = [3, 1, 2]
    for i in range(pcp):
        d = dpcz[:, :, i].T
        d = d / np.max(np.abs(d))  # / dpc_magz[0, oi] * dpc_magz[i, oi]

        ax = f.add_subplot(2, 2, i_lookup[i])
        ax.imshow(np.fliplr(d), **imopts)
        ax.set_title(f'{title} - Dim {i + 1}')

    ax = f.add_subplot(2,2,4, projection='3d')

    for ci, (oi, cellid) in enumerate(zip(orange, cell_list)):
        print(f"{oi} {cellid} dpc all={use_dpc_all}")

        pred = rec['pred'].as_continuous().T[:, oi]
        r = rec['resp'].as_continuous().T[:, oi]

        mmv, N = compute_subspace_density(Y, pred)

        log.info(f'computing 3d subspace projection, level={level}')
        plot_subspace_density(mmv, level=level, ax=ax, ci=ci)

    plt.tight_layout()

    return f


def plot_dpc_proj(modelspec=None, cell_list=None, val=None, est=None, modelspec2=None, show_preds=True, plot_stim=True,
                  use_val=False, print_figs=False, T1=50, T2=250, D=None, emax=100000, ex_pct=0.01, t_highlight=[], 
                  cmap='Greens', **ctx):

    if cell_list is None:
        cell_list = ctx['cellids']
    elif type(cell_list) is str:
        cell_list = [cell_list]
    if use_val:
        rec=val
    else:
        rec=est

    X_est = {'input': rec['stim'].as_continuous().T}
    Y_est = rec['resp'].as_continuous().T 
    #X_est, Y_est = xforms.lite_input_dict(modelspec, rec, epoch_name="")
    fs=rec['resp'].fs

    orange = [i for i,c in enumerate(rec['resp'].chans) if c in cell_list]
    spont = np.zeros(len(rec['resp'].chans))

    cnnPred = modelspec.predict(X_est)
    if modelspec2 is not None:
        lnstrf=LN.LNpop_get_strf(modelspec2)
    else:
        lnstrf=None

    for oi, cellid in zip(orange,cell_list):
        log.info(f"subspace pred/project for {oi}, {cellid}")
        pcp = 3
        rows = pcp+4
        f = plt.figure(figsize=(9, rows))
        colcount=11
        gs = f.add_gridspec(rows, colcount)
        ax = np.zeros((rows-2, 4), dtype='O')
        for r in range(rows-2):
            for c in range(3):
                if (c <= 1):
                    ax[r, c] = f.add_subplot(gs[r, c])
                elif (c > 1):
                    ax[r, c] = f.add_subplot(gs[r, c:-1])
            ax[r, 3] = f.add_subplot(gs[r, -1])
        ax2 = np.zeros(colcount, dtype='O')
        ax3 = np.zeros(colcount, dtype='O')
        for c in range(colcount):
            ax2[c] = f.add_subplot(gs[-2, c])
            ax3[c] = f.add_subplot(gs[-1, c])

        dpcz = modelspec.meta['dpc']
        pc_count = dpcz.shape[1]
        dpc_magz = modelspec.meta['dpc_mag']
        dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]
        fir = filter.FIR(shape=dpcz.shape)
        fir['coefficients'] = np.flip(dpcz, axis=0)

        pred = rec['pred'].as_continuous().T[:, oi]
        X = rec['stim'].as_continuous().T
        r = rec['resp'].as_continuous().T[:, oi]
        if use_val:
            pred2 = est['pred'].as_continuous().T[:emax, oi]
            pred = np.concatenate([pred,pred2])
            X2 = est['stim'].as_continuous().T[:emax, :]
            X = np.concatenate([X,X2], axis=0)
            r2 = est['resp'].as_continuous().T[:emax, oi]
            r = np.concatenate([r, r2])
            
        lnpred = modelspec2.predict(X)[:, oi]

        Y = fir.evaluate(X)
        for i in range(pc_count):
            cc=np.corrcoef(Y[:,i],r)[0,1]
            if cc < 0:
                Y[:,i]=-Y[:,i]
                modelspec.meta['dpc'][oi, i] = -modelspec.meta['dpc'][oi, i]

        imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
                  'interpolation': 'none'}
        trange=np.arange(T1,T2)/fs
        if t_highlight is not None:
            t_highlight=np.array(t_highlight)
        markersize = 5
        statecolor = 'red'

        #ax[0, 0].plot(np.arange(1, dpc_magz.shape[0] + 1), dpc_magz[:, oi] / dpc_magz[:, oi].sum(), 'o-', markersize=3)
        #ax[0, 0].set_xlabel('PC dimension')
        #ax[0, 0].set_ylabel('Var. explained')
        #ax[0, 0].set_xticks(np.arange(1, dpc_magz.shape[0] + 1))
        for i in range(pcp):
            d = modelspec.meta['dpc'][oi, i]
            if D is None:
                D = d.shape[1]
            d = d / np.max(np.abs(d))  # / dpc_magz[0, oi] * dpc_magz[i, oi]

            #ax[i, 0].imshow(np.fliplr(d)[:,:D], **imopts)
            #ax[i, 0].set_ylabel(f'Dim {i + 1}')
            if i>0:
                histmean2d(Y[:, 0], Y[:, i], pred, bins=20, ax=ax[i, 0], spont=spont[oi], cmap=cmap, ex_pct=ex_pct)
                #ax[i, 0].plot(Y[T1:T2, 0], Y[T1:T2, i], linestyle='--', color=statecolor, lw=0.5)
                for ii,ti in enumerate(t_highlight):
                    ax[i, 0].plot(Y[ti, 0], Y[ti, i], '.', color='k', markersize=markersize)
                    ax[i, 0].text(Y[ti, 0]*1.3, Y[ti, i]*1.1,f"{ii+1}", color='k', fontsize=6, ha='center', va='center')
                ax[i,0].set_ylabel(f"Dim {i+1}")
            ax[i, 1].imshow(d[:,-D:], **imopts)

            mm = np.max(np.abs(Y[T1:T2,i]))*1.02
            ax[i, 2].imshow(X[T1:T2, :].T, origin='lower', cmap='gray_r',
                            alpha=0.5, extent=[trange[0], trange[-1], -mm, mm])
            ax[i, 2].plot(trange,Y[T1:T2,i], color=CB_color_cycle[0])
            for ii, ti in enumerate(t_highlight):
                ax[i, 2].axvline(trange[ti-T1], color='k', linestyle='--', lw=0.5)
                ax[i, 2].plot(trange[ti-T1], Y[ti, i], '.', color='k', markersize=markersize)
                ax[i, 2].text(trange[ti-T1], mm, f"{ii + 1}", color='k', fontsize=6, ha='center')

            ax[i, 2].axhline(0, color='k', linestyle='--', lw=0.5)
            ax[i, 2].set_xlim(trange[0],trange[-1])

            stimmag = np.sum(X,axis=1)
            minx = np.min(stimmag)
            xx = Y[stimmag > minx, i]
            yy = pred[stimmag > minx]
            bins = 25

            x_, y_, asym = nl_marginal(xx, yy, smoothwin=7, ex_pct=ex_pct, bins=bins)
            bd = 0.33
            if asym<-bd:
                c_ = 0
            elif asym<bd:
                c_ = 1
            else:
                c_ = 2
            #ax[j, -2].plot(x_[:-1], dy, color=CB_color_cycle[p1])
            ax[i, 3].plot(x_, y_, color=CB_color_cycle[c_])
            ax[i, 3].axvline(0, color='gray', linestyle='--', lw=0.5)
            ax[i, 3].text(ax[i, 3].get_xlim()[0], ax[i, 3].get_ylim()[1],
                           f"{asym:.2f}", fontsize=6, va='top')
            ax[i, 3].set_xticklabels([])
            ax[i, 3].set_yticklabels([])


        mm = Y_est[T1:T2, oi].max()
        ax[-2, 2].plot(trange, smooth(Y_est[T1:T2, oi].T,7), 'k', lw=0.5)
        ax[-2, 2].plot(trange, cnnPred[T1:T2, oi].T, color='purple')
        mm = ax[-2, 2].get_ylim()[1]
        for ii, ti in enumerate(t_highlight):
            ax[-2, 2].axvline(trange[ti - T1], color='k', linestyle='--', lw=0.5)
            ax[-2, 2].plot(trange[t_highlight-T1], cnnPred[t_highlight, oi], '.', color='k', markersize=markersize)
            ax[-2, 2].text(trange[ti - T1], mm, f"{ii + 1}", color='k', fontsize=6, ha='center')
        ax[-2, 2].set_xlim(trange[0], trange[-1])
        ax[-2, 2].text(trange[-1], mm, f"{cellid} CNN: {modelspec.meta['r_test'][oi, 0]:.3f}", ha='right')

        if (lnstrf is not None):
            l = lnstrf[:, :, oi]
            l /= np.abs(l).max()
            #ax[-1, 0].imshow(lnstrf[:,:D,oi], **imopts)
            histmean2d(Y[:, 0], Y[:, 1], lnpred, bins=20, ax=ax[-1, 0], spont=spont[oi], cmap=cmap, ex_pct=ex_pct)
            #ax[-1, 0].plot(Y[T1:T2, 0], Y[T1:T2, 1], linestyle='--', color=statecolor, lw=0.5)
            for ii, ti in enumerate(t_highlight):
                ax[-1, 0].plot(Y[ti, 0], Y[ti, 1], '.', color='k', markersize=markersize)

            ax[-1, 1].imshow(np.fliplr(lnstrf[:,:D,oi]), **imopts)
            ax[-1, 0].set_ylabel(f'LN STRF')

            ax[-1, 2].plot(trange, smooth(Y_est[T1:T2, oi].T,7), 'k', lw=0.5)
            ax[-1, 2].plot(trange, lnpred[T1:T2].T, color='orange')
            mm = ax[-1,2].get_ylim()[1]
            for ii, ti in enumerate(t_highlight):
                ax[-1, 2].axvline(trange[ti - T1], color='k', linestyle='--', lw=0.5)
                ax[-1, 2].plot(trange[ti - T1], lnpred[ti], '.', color='k', markersize=markersize)
                ax[-1, 2].text(trange[ti - T1], mm, f"{ii + 1}", color='k', fontsize=6, ha='center')
            ax[-1, 2].text(trange[-1], mm, f"LN: {modelspec2.meta['r_test'][oi, 0]:.3f}", ha='right')
            ax[-1, 2].set_xlim(trange[0], trange[-1])

        #ac, bc, Zresp, N=histmean2d(Y[:,0], Y[:,1], pred, bins=20, ax=ax[0, 1], spont=spont[oi], cmap=cmap, ex_pct=ex_pct)
        T_set = np.arange(T1+20, T1+325, 25)
        Zresp=None
        for c, a in enumerate(ax2):
            t1, t2 = T_set[c], T_set[c+1]
            ac, bc, Zresp, N = histmean2d(Y[:, 0], Y[:, 1], pred, bins=20, ax=a, spont=spont[oi], cmap=cmap, ex_pct=ex_pct, Z=Zresp)
            a.plot(Y[T1:t2, 0], Y[T1:t2, 1], color=statecolor, lw=0.5)
            a.plot(Y[t1:t2, 0], Y[t1:t2, 1], color=statecolor)

            a.text(a.get_xlim()[0], a.get_ylim()[1], f"{t1/fs:.2f}-{t2/fs:.2f}", fontsize=7)
            for rr in range(1, 5):
                ax[rr, 2].axvline(t1/fs, lw=0.5, linestyle='--', color='lightgray')


        for c,a in enumerate(ax.flatten()):
            if c < len(ax.flatten())-1:
                a.set_xticklabels([])
            #else:
            #    x = a.get_xticklabels()
            #    a.set_xticklabels(x, fontsize=8)

            a.set_yticklabels([])

        yl = ax2[0].get_ylim()
        xl = ax2[0].get_xlim()
        for a in ax2.flatten():
            a.set_xticklabels([])
            a.set_yticklabels([])
            #a.set_axis_off()
            a.set_ylim(yl)
            a.set_xlim(xl)

        #a=Y[:,0]
        #b=Y[:,1]
        #keep = np.isfinite(a) & np.isfinite(b)
        #ab = np.percentile(a[keep], [ex_pct, 100 - ex_pct])
        #bb = np.percentile(b[keep], [ex_pct, 100 - ex_pct])
        #print('ex_pct', ex_pct, 'ab',ab,'bb',bb)
        
        histmean2d(Y[:, 0], Y[:, 1], pred, bins=20, ax=ax3[0], spont=spont[oi], cmap=cmap, ex_pct=ex_pct, Z=Zresp)
        histmean2d(Y[:, 0], Y[:, 2], pred, bins=20, ax=ax3[1], spont=spont[oi], cmap=cmap, ex_pct=ex_pct)
        ax3[0].plot(Y[T1:T2, 0], Y[T1:T2, 1], color=statecolor, lw=0.5)
        ax3[1].plot(Y[T1:T2, 0], Y[T1:T2, 2], color=statecolor, lw=0.5)

        histmean2d(Y[:, 0], Y[:, 1], lnpred, bins=20, ax=ax3[3], spont=spont[oi], cmap=cmap, ex_pct=ex_pct)
        histmean2d(Y[:, 0], Y[:, 2], lnpred, bins=20, ax=ax3[4], spont=spont[oi], cmap=cmap, ex_pct=ex_pct)
        ax3[3].plot(Y[T1:T2, 0], Y[T1:T2, 1], color=statecolor, lw=0.5)
        ax3[4].plot(Y[T1:T2, 0], Y[T1:T2, 2], color=statecolor, lw=0.5)

        for i in range(3):
            d = modelspec.meta['dpc'][oi, i]
            d = d / np.max(np.abs(d))  # / dpc_magz[0, oi] * dpc_magz[i, oi]

            ax3[i+7].imshow(np.fliplr(d)[:,:D], **imopts)
            ax3[i+7].text(D-2,0,f'D{i + 1}', ha='right')

        for i in [2,5,6]:
            ax3[i].set_axis_off()
        ax[pcp,-1].set_axis_off()
        ax[pcp+1,-1].set_axis_off()
        ax3[-1].set_axis_off()
        for i in [0,1,3,4,7,8,9]:
            ax3[i].set_xticklabels([])
            ax3[i].set_yticklabels([])
        f.suptitle(cellid)
        plt.tight_layout()

    return f


def plot_dstrf_example(modelspec=None, modelspec_list=None, D=20, dstrf=None, sigrat=1, first_lin=False, pc_count=25, cell_list=None, val=None, t_indexes=None, timestep=100, **ctx):

    if cell_list is None:
        cell_list = ctx['cellids'][0]
    elif type(cell_list) is str:
        cell_list = [cell_list]

    if t_indexes is None:
        t_indexes = np.arange(timestep, val['stim'].shape[1], timestep)
    t_indexes = t_indexes[t_indexes > D]
    tcount = len(t_indexes)
    if modelspec_list is None:
        modelspec_list = [modelspec]

    out_channels = [[i for i,c in enumerate(modelspec.meta['cellids']) if c==c0][0] for c0 in cell_list]
    cellcount = len(out_channels)

    stim = {'input': val['stim'].as_continuous().T}
    if 'dlc' in val.signals.keys():
        stim['dlc']=val['dlc'].as_continuous().T
    sig = 'input'
    if dstrf is None:
        dstrfs = []
        for mi, m in enumerate(modelspec_list):
            log.info(f"Computing dSTRF {mi + 1}/{len(modelspec_list)} at {tcount} points (timestep={timestep})")

            d = m.dstrf(stim, D=D, out_channels=out_channels, t_indexes=t_indexes, reset_backend=False)
            dstrfs.append(d[sig].astype(np.float32))  # to save memory


        dstrf = np.stack(dstrfs, axis=1)
        del dstrfs
        s = np.std(dstrf, axis=(2, 3, 4), keepdims=True)
        dstrf /= s
        dstrf /= np.max(np.abs(dstrf), axis=(1,2,3,4), keepdims=True)

        log.info("Averaging across jackknifes")
        mdstrf = dstrf.mean(axis=1, keepdims=True)
        mdstrf /= np.max(np.abs(mdstrf)) * 0.9
        d = dtools.compute_dpcs(mdstrf[:, 0], pc_count=pc_count, first_lin=first_lin, as_dict=True)

        if len(modelspec_list) > 1:
            log.info("Shrinking across jackknifes")
            sdstrf = dstrf.std(axis=1, keepdims=True)
            sdstrf[sdstrf == 0] = 1

            log.info("Calling shrinkage()")
            mzdstrf = shrinkage(mdstrf, sdstrf, sigrat=sigrat)
            mzdstrf /= np.max(np.abs(mzdstrf)) * 0.9
            mean_dstrf = mzdstrf
            del sdstrf

            log.info("Calling compute_dpcs()")
            dz = dtools.compute_dpcs(mzdstrf[:, 0], pc_count=pc_count, first_lin=first_lin, as_dict=True)

        else:
            dz = d
            mean_dstrf = mdstrf
    else:
        mean_dstrf=dstrf
        mean_dstrf /= np.max(np.abs(mean_dstrf)) * 0.9
        d = dtools.compute_dpcs(mean_dstrf[:, 0], pc_count=pc_count, first_lin=first_lin, as_dict=True)
        dz = d

    return mean_dstrf, dz

    mm = np.max(np.abs(mean_dstrf)) * 0.75
    f,ax = plt.subplots(8,8,figsize=(8,8), sharex=True, sharey=True)
    ax=ax.flatten()
    for i,a in enumerate(ax):
        d_ = np.fliplr(mean_dstrf[0,0,i])
        #mm = np.max(np.abs(d_))
        a.imshow(d_, origin='lower', cmap='bwr', vmin=-mm, vmax=mm)


    del dstrf

    log.info("Computing site-wide dPCs")
    # dpcs for all cells in site
    T = len(t_indexes)
    F = mean_dstrf.shape[3]
    U = mean_dstrf.shape[4]
    dstrf_all = np.reshape(mean_dstrf, [1, cellcount * T, F, U])
    dall = dtools.compute_dpcs(dstrf_all, pc_count=pc_count, snr_threshold=None,
                               first_lin=False, as_dict=True)

    N = 10
    log.info(f"Computing dPC noise floor for each unit (N={N})")
    # compute noise floor by measuring PCs with shuffled spectro-temporal parameters
    sh_mags = []
    for i in range(N):
        m_ = shuffle_along_axis(mean_dstrf, axis=2)
        d_ = dtools.compute_dpcs(m_[:, 0], pc_count=pc_count, first_lin=first_lin,
                                 snr_threshold=None, as_dict=True, flip_sign=False)
        sh_mags.append(d_[sig]['pc_mag'])
        sh_mags.append(d_[sig]['pc_mag'])
    sh_mags = np.stack(sh_mags, axis=2)
    msh = sh_mags.mean(axis=2)
    esh = sh_mags.std(axis=2) / (N ** 0.5)

    dpc = d[sig]['pcs']
    dpc_mag = d[sig]['pc_mag']
    dpcz = dz[sig]['pcs']
    dpc_magz = dz[sig]['pc_mag']
    dproj = dz[sig]['projection']
    log.info(f"dproj.shape={dproj.shape}")

    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower', 'interpolation': 'none'}

    f, ax = plt.subplots(len(out_channels), pc_count + 1, figsize=(pc_count, len(out_channels) * 0.75), sharex='col',
                         sharey='col')
    f.subplots_adjust(top=0.98, bottom=0.02)
    if len(out_channels) == 1:
        ax = ax[np.newaxis, ...]
    for oi, oc in enumerate(out_channels):
        ax[oi, -1].plot(msh[:, oi] / msh[:, oi].sum(), lw=0.5, color='gray')
        ax[oi, -1].plot(dpc_magz[:, oi] / dpc_mag[:, oi].sum())
        for di in range(pc_count):
            d = dpcz[oi, di]
            d = d / np.max(np.abs(d)) / np.max(dpc_magz[:, oi]) * dpc_magz[di, oi]
            ax[oi, di].imshow(np.fliplr(d), **imopts)
            ax[oi, di + 1].set_yticklabels([])
        yl = ax[oi, -1].get_ylim()
        ax[oi, -1].text(0, yl[1], modelspec.meta['cellids'][oi], fontsize=6, va='top')

    if figures is None:
        figures = []
    figures.append(fig2BytesIO(f))
    modelspec.meta['dpc'] = dpcz
    modelspec.meta['dpc_mag'] = dpc_magz
    modelspec.meta['dpc_mag_sh'] = msh
    modelspec.meta['dpc_mag_e'] = esh
    modelspec.meta['dpc_all'] = dall['input']['pcs']
    modelspec.meta['dpc_mag_all'] = dall['input']['pc_mag']

    if fit_ss_model:
        d = subspace_model_fit(est, val, modelspec, out_channels=out_channels,
                               pc_count=ss_pccount, dpc_var=ss_dpc_var)
        modelspec = d['modelspec']

    log.info("removing backends from modelspec")
    modelspec.backend = None
    modelspec.dstrf_backend = None
    return {'modelspec': modelspec, 'figures': figures}


def histmean2d(a,b,d, bins=10, ax=None, spont=None, ex_pct=0.05,
               cmap='viridis', vmin=None, vmax=None, zerolines=True,
               minN=1, flipZ=True, Z=None, add_colorbar=False, av=None, bv=None,
               show_plot=True, overlap_fraction=1.0):
    keep = np.isfinite(a) & np.isfinite(b)
    if av is None:
        ab = np.percentile(a[keep], [ex_pct, 100 - ex_pct])
        av = np.linspace(ab[0], ab[1], bins + 1)
    if bv is None:
        bb = np.percentile(b[keep], [ex_pct, 100 - ex_pct])
        bv = np.linspace(bb[0], bb[1], bins + 1)
    ac = (av[:-1]+av[1:])/2
    bc = (bv[:-1]+bv[1:])/2

    mmv = np.zeros((bins, bins)) * np.nan
    N = np.zeros((bins, bins))
    if Z is None:
        # Calculate bin widths for overlap calculation
        a_width = av[1] - av[0]
        b_width = bv[1] - bv[0]

        # Calculate overlap amounts
        a_overlap = a_width * overlap_fraction
        b_overlap = b_width * overlap_fraction

        for i_, a_center in enumerate(ac):
            for j_, b_center in enumerate(bc):
                if overlap_fraction > 0:
                    # Use overlapping windows centered on bin centers
                    a_half_window = (a_width + a_overlap) / 2
                    b_half_window = (b_width + b_overlap) / 2

                    a_min = a_center - a_half_window
                    a_max = a_center + a_half_window
                    b_min = b_center - b_half_window
                    b_max = b_center + b_half_window

                    v_ = (a >= a_min) & (a < a_max) & (b >= b_min) & (b < b_max) & np.isfinite(d)
                else:
                    # Use original non-overlapping bins
                    a_min = av[i_]
                    a_max = av[i_ + 1]
                    b_min = bv[j_]
                    b_max = bv[j_ + 1]

                    v_ = (a >= a_min) & (a < a_max) & (b >= b_min) & (b < b_max) & np.isfinite(d)

                if (v_.sum() > 0):
                    mmv[j_, i_] = np.nanmean(d[v_])
                    N[j_,i_] = v_.sum()

        if spont is None:
            # find the zero bin:
            zi = np.argmin(np.abs(ac))
            zj = np.argmin(np.abs(bc))
            spont = mmv[zj, zi]
        mmv -= spont
        mmv[N<minN]=np.nan
        #mmv[np.isnan(mmv)] = 0

        # option to interpolate (not used)
        # x = (llv[:-1] + llv[1:]) / 2
        # y = (ttv[:-1] + ttv[1:]) / 2
        # X, Y = np.meshgrid(x, y)  # 2D grid for interpolation
        #
        # valididx = np.isfinite(mm)
        # interp = LinearNDInterpolator(list(zip(X[valididx], Y[valididx])),
        #                               mm[valididx], fill_value=np.nanmean(mm))

        # plot heatmaps
        Z = mmv
        # Z = interp(X, Y)
        # zsm = 0.5
        # Zz = (Z == 0)
        # Z = gaussian_filter(Z, [zsm, zsm])
        # Z[Zz] = 0

    if show_plot:
        if ax is None:
            f, ax = plt.subplots()

        #cmap='bwr'
        #cmap='viridis'
        if (vmin is None) or (vmax is None):
            vmin, vmax = np.percentile(Z[np.isfinite(Z)], [2, 98])
            #vmin = -vmax
            #print(vmin,vmax)

        if flipZ:
            Z_ = np.flipud(Z)
        else:
            Z_ = Z
        astep=(av[1]-av[0])/2
        bstep=(bv[1]-bv[0])/2
        im = ax.imshow(Z_, extent=[av[0]-astep, av[-1]+astep,
                                   bv[0]-bstep, bv[-1]+bstep],
                       interpolation='none', aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax)
        if add_colorbar:
            plt.colorbar(im, ax=ax)
        #ax.contour(ac, bc, N, [minN-0.5], linewidths=0.5)
        if zerolines:
            ax.axhline(0, ls='--', lw=0.5, color='gray')
            ax.axvline(0, ls='--', lw=0.5, color='gray')

    return ac, bc, Z, N


def histscatter2d(a, b, d, N=1000, ax=None, spont=None, ex_pct=0.05,
                  vmin=None, vmax=None, zerolines=True):
    keep = np.isfinite(a) & np.isfinite(b)
    ab = np.percentile(a[keep], [ex_pct, 100 - ex_pct])
    bb = np.percentile(b[keep], [ex_pct, 100 - ex_pct])
    keep = (a >= ab[0]) & (a <= ab[1]) & (b >= bb[0]) & (b <= bb[1])
    a_ = a[keep]
    b_ = b[keep]
    d_ = d[keep]

    if N < len(d_):
        ii = np.round(np.linspace(0, len(d_) - 1, N)).astype('int')
    else:
        ii = np.arange(len(d_), dtype='int')

    if ax is None:
        f, ax = plt.subplots()

    cmap = 'bwr'
    cmap = 'viridis'
    m = np.mean(d)
    s = np.std(d)
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = m + s * 2
    if vmax<=vmin:
        vmin = np.min(d)
        vmax = np.max(d)
    #print(vmin, vmax, len(ii))
    #s0 = np.argsort(d_[ii])
    #ii = ii[s0]
    im = ax.scatter(a_[ii], b_[ii], c=d_[ii], s=1, cmap=cmap, vmin=vmin, vmax=vmax)
    # ax.contour(ac, bc, N, [0.5], linewidths=0.5)
    if zerolines:
        ax.axhline(0, ls='--', lw=0.75, color='black')
        ax.axvline(0, ls='--', lw=0.75, color='black')
    return N
    

def scatter_comp(x=None, y=None, n1='model1', n2='model2', hist_bins=20,
                 hist_range=[-1, 1], title="", s=None,
                 highlight=None, scatter_only=True, ax=None,
                 data=None, color='k', good_threshold=None, stat_test=None):
    """
    beta1, beta2 are T x 1 vectors
    scatter plot comparing beta1 vs. beta2
    histograms of marginals
    """
    if ((x is None) or (y is None)) & (data is None):
        raise ValueError("x/y or data parameters required")

    if data is None:
        beta1 = np.array(x)
        beta2 = np.array(y)
    else:
        if x is None:
            beta1 = data.iloc[:, 0]
        else:
            beta1 = data[x]
        if y is None:
            beta2 = data.iloc[:, 1]
        else:
            beta2 = data[y]
        if n1=='model1':
            n1=beta1.name
        if n2=='model2':
            n2=beta2.name

    gg = np.isfinite(beta1) & np.isfinite(beta2)
    beta1 = beta1[gg]
    beta2 = beta2[gg]

    if hist_range==[-1, 1]:
        print('adjusting histrange')
        hist_range = [np.min([beta1.min(),beta2.min()]),
                      np.max([beta1.max(),beta2.max()])]
    # exclude cells without prepassive
    outcells = ((beta1 > hist_range[1]) | (beta1 < hist_range[0]) |
                (beta2 > hist_range[1]) | (beta2 < hist_range[0]))
    goodcells = (np.abs(beta1) > 0) | (np.abs(beta2) > 0)

    beta1[beta1 > hist_range[1]] = hist_range[1]
    beta1[beta1 < hist_range[0]] = hist_range[0]
    beta2[beta2 > hist_range[1]] = hist_range[1]
    beta2[beta2 < hist_range[0]] = hist_range[0]

    if good_threshold is not None:
        if type(good_threshold)==str:
            set1 = data[good_threshold]
            set2 = np.logical_not(set1)
        else:
            set1 = (beta1>good_threshold) & (beta2>good_threshold)
            set2 = np.logical_not(set1)

    elif highlight is None:
        set1 = goodcells
        set2 = []
    else:
        highlight = np.array(highlight)
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))

    if ax is not None:
        scatter_only=True
        fh = ax.figure
    elif scatter_only:
        fh, ax = plt.subplots()
    else:
        fh = plt.figure()
        ax = plt.subplot(2, 2, 3)

    ax.scatter(beta1[outcells], beta2[outcells], s=s, marker='.', color='red')
    ax.scatter(beta1[set2], beta2[set2], s=s, marker='.', color='lightgray')
    ax.scatter(beta1[set1], beta2[set1], s=s, marker='.', color=color)
    #ax.plot(np.array(hist_range), np.array([0, 0]), 'k--', lw=0.5)
    #ax.plot(np.array([0, 0]), np.array(hist_range), 'k--', lw=0.5)
    ax.plot(np.array(hist_range), np.array(hist_range), 'k--', lw=0.5)
    ax.axis('equal')
    ax.axis('tight')

    if stat_test=='wilcoxon':
        res = st.wilcoxon(beta1[set1], beta2[set1])
        title+=f"W={res.statistic:.3f} p={res.pvalue:.2e}"
    elif stat_test=='pearson':
        res = st.pearsonr(beta1[set1], beta2[set1])
        title+=f"r={res.statistic:.3f} p={res.pvalue:.2e}"
    ax.set_xlabel(f"{n1} {np.mean(beta1[set1]):.3f}")
    ax.set_ylabel(f"{n2} {np.mean(beta2[set1]):.3f}")
    ax.set_title(title)

    if scatter_only:
        return fh

    plt.subplot(2, 2, 1)
    plt.hist([beta1[set1], beta1[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True,
             color=['black', 'lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta1[goodcells]),
                     np.mean(np.abs(beta1[goodcells]))))
    plt.xlabel(n1)

    ax = plt.subplot(2, 2, 4)
    plt.hist([beta2[set1], beta2[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True, orientation="horizontal",
             color=['black', 'lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel(n2)

    ax = plt.subplot(2, 2, 2)
    plt.hist([(beta2[set1]-beta1[set1]) * np.sign(beta2[set1]),
              beta2[set2]-beta1[set2] * np.sign(beta2[set2])],
             bins=hist_bins-1, range=[-hist_range[1]/2, hist_range[1]/2],
             histtype='bar', stacked=True,
             color=['black', 'lightgray'])
    plt.title('mean={:.3f} sterr={:.3f}'.
              format(np.mean(beta2[goodcells]-beta1[goodcells]),
                     np.std(beta2[goodcells]-beta1[goodcells])/np.sqrt(np.sum(goodcells))))
    plt.xlabel('difference')
    plt.tight_layout()

    return fh
