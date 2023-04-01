import copy
import logging
import time
import typing
from functools import partial
import re

import numpy as np
from sklearn.decomposition import PCA

from nems0.analysis.cost_functions import basic_cost
from nems0.fitters.api import scipy_minimize
import nems0.priors
import nems0.fitters.mappers
import nems0.modelspec as ms
import nems0.metrics.api as metrics
import nems0.segmentors
import nems0.utils
import nems0.recording as recording
from nems0.initializers import modelspec_freeze_layers, modelspec_unfreeze_layers

log = logging.getLogger(__name__)


def cc_shrink(a,jackcount=20,sigrat=0.5):
    if jackcount>a.shape[1]:
        jackcount=a.shape[1]
    
    jcov = []
    jackstep = a.shape[1]/jackcount
    for j in range(jackcount):
        eidx = np.concatenate((np.arange(0,int(np.round(j*jackstep))),
                               np.arange(int(np.round((j+1)*jackstep)),a.shape[1])
                              ))
        jcov.append(np.cov(a[:,eidx]))
    jcov = np.stack(jcov,axis=2)
    mcov = jcov.mean(axis=2)
    scov = jcov.std(axis=2) * np.sqrt(jackcount-1)
    
    smd=np.abs(mcov) / (scov+np.finfo(float).eps * (scov==0)) / sigrat;
    smd= 1 - smd**-2;
    smd[smd<0 | np.isnan(smd)] = 0

    return smd * mcov


def cc_lowrank(a,n_pcs=1):
    cc=np.cov(a)
    u,s,vh = np.linalg.svd(cc)
    s[n_pcs:]=0
    cc1 = u @ np.diag(s) @ vh
    np.fill_diagonal(cc1, np.diag(cc))    
    return cc1


def cc_shared_space(a,U):
    cc=np.cov(a)
    cc1 = np.cov(U @ a)
    np.fill_diagonal(cc1, np.diag(cc))    
    return cc1


def compute_cc_matrices(modelspec, est,
                        shrink_cc: float = 0,
                        noise_pcs: int = 0,
                        shared_pcs: int = 0,
                        force_psth: bool = False,
                        verbose=False,
                        signal_for_cc='resp',
                        **ctx
                        ):

    if ctx.get('IsReload',False):
        log.info('IsReload: skipping re-eval')
    else:
        est = modelspec.evaluate(est, stop=2)
    if ('pred0' in est.signals.keys()) & (not force_psth):
        input_name = 'pred0'
        log.info('Found pred0 for fitting CC')
    else:
        input_name = 'psth'
        log.info('No pred0, using psth for fitting CC')

    conditions = ["_".join(k.split("_")[1:]) for k in est.signals.keys() if k.startswith("mask_")]
    if (len(conditions)>2) and any([c.split("_")[-1]=='lg' for c in conditions]):
        conditions.remove("small")
        conditions.remove("large")
    #conditions = conditions[0:2]
    #conditions = ['large','small']

    group_idx = [est['mask_'+c].as_continuous()[0,:] for c in conditions]
    cg_filtered = [(c, g) for c,g in zip(conditions,group_idx) if g.sum()>0]
    conditions, group_idx = zip(*cg_filtered)

    for c,g in zip(conditions, group_idx):
        log.info(f"cc data for {c} len {g.sum()}")

    resp = est[signal_for_cc].as_continuous()
    pred0 = est[input_name].as_continuous()
    #print((resp).std(axis=1))
    #print(pred0.std(axis=1))
    #import pdb; pdb.set_trace()
    if shrink_cc > 0:
        log.info(f'cc approx: shrink_cc={shrink_cc}')
        group_cc = [cc_shrink(resp[:,idx]-pred0[:,idx], sigrat=shrink_cc) for idx in group_idx]
    elif shared_pcs > 0:
        log.info(f'cc approx: shared_pcs={shared_pcs}')
        rresp = est['resp'].as_continuous()
        cc=np.cov(rresp-pred0)
        u,s,vh = np.linalg.svd(cc)
        U = u[:, :shared_pcs] @ u[:, :shared_pcs].T
        group_cc = [cc_shared_space(resp[:, idx]-pred0[:,idx], U) for idx in group_idx]
        
    elif noise_pcs > 0:
        log.info(f'cc approx: noise_pcs={noise_pcs}')
        group_cc = [cc_lowrank(resp[:,idx]-pred0[:,idx], n_pcs=noise_pcs) for idx in group_idx]
    else:
        group_cc = [np.cov(resp[:,idx]-pred0[:,idx]) for idx in group_idx]
    group_cc_raw = [np.cov(resp[:,idx]-pred0[:,idx]) for idx in group_idx]

    if verbose:
        import matplotlib.pyplot as plt
        cols = 4
        rows = int(np.ceil(len(group_cc)/2))
        f, ax = plt.subplots(rows, cols, figsize=(4, 2*rows))
        ax = ax.flatten()
        i = 0
        for g, g_raw, cond in zip(group_cc, group_cc_raw, conditions):
            mm= np.max(np.abs(g))
            ax[i*2].imshow(g, cmap='bwr', clim=[-mm,mm], origin='lower', interpolation='none')
            ax[i*2+1].imshow(g_raw, cmap='bwr', clim=[-mm,mm], origin='lower', interpolation='none')
            ax[i*2].set_title(cond)
            i += 1

    return group_idx, group_cc, conditions

def fit_ccnorm(modelspec,
        est: recording.Recording,
        metric=None,
        use_modelspec_init: bool = True,
        optimizer: str = 'adam',
        max_iter: int = 10000,
        early_stopping_steps: int = 5,
        tolerance: float = 5e-4,
        learning_rate: float = 1e-4,
        batch_size: typing.Union[None, int] = None,
        seed: int = 0,
        initializer: str = 'random_normal',
        freeze_layers: typing.Union[None, list] = None,
        epoch_name: str = "REFERENCE",
        shrink_cc: float = 0,
        noise_pcs: int = 0,
        shared_pcs: int = 0,
        also_fit_resp: bool = False,
        force_psth: bool = False,
        use_metric: typing.Union[None, str] = None,
        alpha: float = 0.1,
        beta: float = 1,
        exclude_idx=None, exclude_after=None,
        freeze_idx=None, freeze_after=None,
        **context):
    '''
    Required Arguments:
     est          A recording object
     modelspec     A modelspec object

    Optional Arguments:
     <copied from fit_tf for now

    Returns
     dictionary: {'modelspec': updated_modelspec}
    '''

    # Hard-coded
    cost_function = basic_cost
    fitter=scipy_minimize
    segmentor=nems0.segmentors.use_all_data
    mapper=nems0.fitters.mappers.simple_vector
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    start_time = time.time()

    fit_index = modelspec.fit_index
    if (exclude_idx is not None) | (freeze_idx is not None) | \
            (exclude_after is not None) | (freeze_after is not None):
        modelspec0 = modelspec.copy()
        modelspec, include_set = modelspec_freeze_layers(
            modelspec, include_idx=None,
            exclude_idx=exclude_idx, exclude_after=exclude_after,
            freeze_idx=freeze_idx, freeze_after=freeze_after)
        modelspec0.set_fit(fit_index)
        modelspec.set_fit(fit_index)
    else:
        include_set = None

    # Computing PCs before masking out unwanted stimuli in order to
    # preserve match with epochs
    epoch_regex = "^STIM_"
    stims = (est.epochs['name'].value_counts() >= 8)
    stims = [stims.index[i] for i, s in enumerate(stims)
             if bool(re.search(epoch_regex, stims.index[i])) and s == True]

    Rall_u = est.apply_mask()['psth'].as_continuous().T
    # can't simply extract evoked for refs because can be longer/shorted if it came after target
    # and / or if it was the last stim. So, masking prestim / postim doesn't work. Do it manually
    #d = est['resp'].extract_epochs(stims, mask=est['mask'])

    #R = [v.mean(axis=0) for (k, v) in d.items()]
    #R = [np.reshape(np.transpose(v,[1,0,2]),[v.shape[1],-1]) for (k, v) in d.items()]
    #Rall_u = np.hstack(R).T

    pca = PCA(n_components=2)
    pca.fit(Rall_u)
    pc_axes = pca.components_

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    if 'mask_small' in est.signals.keys():
        log.info('reseting mask with mask_small+mask_large subset')
        est['mask'] = est['mask']._modified_copy(data=est['mask_small']._data + est['mask_large']._data)

    if 'mask' in est.signals.keys():
        log.info("Data len pre-mask: %d", est['mask'].shape[1])
        est = est.apply_mask()
        log.info("Data len post-mask: %d", est['mask'].shape[1])

    # if we want to fit to first-order cc error.
    #uncomment this and make sure sdexp is generating a pred0 signal
    est = modelspec.evaluate(est, stop=2)
    if ('pred0' in est.signals.keys()) & (not force_psth):
        input_name = 'pred0'
        log.info('Found pred0 for fitting CC')
    else:
        input_name = 'psth'
        log.info('No pred0, using psth for fitting CC')

    conditions = ["_".join(k.split("_")[1:]) for k in est.signals.keys() if k.startswith("mask_")]
    if (len(conditions)>2) and any([c.split("_")[-1]=='lg' for c in conditions]):
        conditions.remove("small")
        conditions.remove("large")
    #conditions = conditions[0:2]
    #conditions = ['large','small']

    group_idx = [est['mask_'+c].as_continuous()[0,:] for c in conditions]
    cg_filtered = [(c, g) for c,g in zip(conditions,group_idx) if g.sum()>0]
    conditions, group_idx = zip(*cg_filtered)
    
    for c,g in zip(conditions, group_idx):
        log.info(f"cc data for {c} len {g.sum()}")
        
    resp = est['resp'].as_continuous()
    pred0 = est[input_name].as_continuous()
    #import pdb; pdb.set_trace() 
    if shrink_cc > 0:
        log.info(f'cc approx: shrink_cc={shrink_cc}')
        group_cc = [cc_shrink(resp[:,idx]-pred0[:,idx], sigrat=shrink_cc) for idx in group_idx]
    elif shared_pcs > 0:
        log.info(f'cc approx: shared_pcs={shared_pcs}')
        cc=np.cov(resp-pred0)
        u,s,vh = np.linalg.svd(cc)
        U = u[:, :shared_pcs] @ u[:, :shared_pcs].T

        group_cc = [cc_shared_space(resp[:, idx]-pred0[:,idx], U) for idx in group_idx]
    elif noise_pcs > 0:
        log.info(f'cc approx: noise_pcs={noise_pcs}')
        group_cc = [cc_lowrank(resp[:,idx]-pred0[:,idx], n_pcs=noise_pcs) for idx in group_idx]
    else:
        group_cc = [np.cov(resp[:,idx]-pred0[:,idx]) for idx in group_idx]
    group_cc_raw = [np.cov(resp[:,idx]-pred0[:,idx]) for idx in group_idx]

    if 0:
        import matplotlib.pyplot as plt
        cols = 4
        rows = int(np.ceil(len(group_cc)/2))
        f, ax = plt.subplots(rows, cols, figsize=(8, 4*rows))
        ax = ax.flatten()
        i = 0
        for g, g_raw, cond in zip(group_cc, group_cc_raw, conditions):
            mm= np.max(np.abs(g))
            ax[i*2].imshow(g, cmap='bwr', clim=[-mm,mm], origin='lower', interpolation='none')
            ax[i*2+1].imshow(g_raw, cmap='bwr', clim=[-mm,mm], origin='lower', interpolation='none')
            ax[i*2].set_title(cond)
            i += 1

    # variance of projection onto PCs (PCs computed above before masking)
    pcproj0 = (resp-pred0).T.dot(pc_axes.T).T
    pcproj_std = pcproj0.std(axis=1)

    if (use_metric=='cc_err_w'):

        def metric(d, verbose=False):
            return metrics.cc_err_w(d, pred_name='pred', pred0_name=input_name,
                                    group_idx=group_idx, group_cc=group_cc, alpha=alpha,
                                    pcproj_std=None, pc_axes=None, verbose=verbose)
        log.info(f"fit_ccnorm metric: cc_err_w (alpha={alpha})")

    elif (metric is None) and also_fit_resp:
        log.info(f"resp_cc_err: pred0_name: {input_name} beta: {beta}")
        metric = lambda d: metrics.resp_cc_err(d, pred_name='pred', pred0_name=input_name,
                                          group_idx=group_idx, group_cc=group_cc, beta=beta,
                                          pcproj_std=None, pc_axes=None)

    elif (use_metric=='cc_err_md'):
        def metric(d, verbose=False):
            return metrics.cc_err_md(d, pred_name='pred', pred0_name=input_name,
                                    group_idx=group_idx, group_cc=group_cc,
                                    pcproj_std=None, pc_axes=None)
        log.info(f"fit_ccnorm metric: cc_err_md")

    elif (metric is None):
        #def cc_err(result, pred_name='pred_lv', resp_name='resp', pred0_name='pred',
        #   group_idx=None, group_cc=None, pcproj_std=None, pc_axes=None):        
        # current implementation of cc_err
        metric = lambda d: metrics.cc_err(d, pred_name='pred', pred0_name=input_name,
                                          group_idx=group_idx, group_cc=group_cc,
                                          pcproj_std=None, pc_axes=None)
        log.info(f"fit_ccnorm metric: cc_err")

    # turn on "fit mode". currently this serves one purpose, for normalization
    # parameters to be re-fit for the output of each module that uses
    # normalization. does nothing if normalization is not being used.
    ms.fit_mode_on(modelspec, est)

    # Create the mapper functions that translates to and from modelspecs.
    # It has three functions that, when defined as mathematical functions, are:
    #    .pack(modelspec) -> fitspace_point
    #    .unpack(fitspace_point) -> modelspec
    #    .bounds(modelspec) -> fitspace_bounds
    packer, unpacker, pack_bounds = mapper(modelspec)

    # A function to evaluate the modelspec on the data
    evaluator = nems0.modelspec.evaluate

    my_cost_function = cost_function
    my_cost_function.counter = 0

    # Freeze everything but sigma, since that's all the fitter should be
    # updating.
    cost_fn = partial(my_cost_function,
                      unpacker=unpacker, modelspec=modelspec,
                      data=est, segmentor=segmentor, evaluator=evaluator,
                      metric=metric, display_N=1000)

    # get initial sigma value representing some point in the fit space,
    # and corresponding bounds for each value
    sigma = packer(modelspec)
    bounds = pack_bounds(modelspec)

    # Results should be a list of modelspecs
    # (might only be one in list, but still should be packaged as a list)
    improved_sigma = fitter(sigma, cost_fn, bounds=bounds,
                            **fit_kwargs)
    improved_modelspec = unpacker(improved_sigma)
    elapsed_time = (time.time() - start_time)

    start_err = cost_fn(sigma)
    final_err = cost_fn(improved_sigma)
    log.info("Delta error: %.06f - %.06f = %e", start_err, final_err, final_err-start_err)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.fit_mode_off(improved_modelspec)

    if include_set is not None:
        # pull out updated phi values from improved_modelspec, include_set only
        improved_modelspec = \
            modelspec_unfreeze_layers(improved_modelspec, modelspec0, include_set)
        improved_modelspec.set_fit(fit_index)

    log.info(f"Updating improved modelspec with fit_idx={improved_modelspec.fit_index}")
    improved_modelspec.meta['fitter'] = 'ccnorm'
    improved_modelspec.meta['n_parms'] = len(improved_sigma)
    if modelspec.fit_count == 1:
        improved_modelspec.meta['fit_time'] = elapsed_time
        improved_modelspec.meta['loss'] = final_err
    else:
        if modelspec.fit_index == 0:
            improved_modelspec.meta['fit_time'] = np.zeros(improved_modelspec.fit_count)
            improved_modelspec.meta['loss'] = np.zeros(improved_modelspec.fit_count)
        improved_modelspec.meta['fit_time'][fit_index] = elapsed_time
        improved_modelspec.meta['loss'][fit_index] = final_err

    return {'modelspec': improved_modelspec}
    # return {'modelspec': improved_modelspec.copy(), 'save_context': True}


def pc_err(result, pred_name='pred_lv', resp_name='resp', pred0_name='pred',
           group_idx=None, group_pc=None, pc_axes=None, resp_std=None):
    '''
    Given the evaluated data, return the mean squared error for correlation coefficient computed
    separately for each group of the data (eg, passive vs. active or big vs. small pupil)

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    (these other parameters can be hard-coded into a partial function that is then passed onto the fitter:)
    pred_name : string
        Name of prediction in the result recording
    pred0_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording
    group_idx: list of array indexes defining which samples belong in which group
    group_pc: list of PC projection stds, one for each group
    pc_axes: projection vectors for first N pcs--to project prediction and compute difference from
             actual pcproj_std

    Returns
    -------
    E : float
        Mean-squared difference between the CC matrix for each group

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> error = cc_err(result, 'pred', 'resp', <other parameters>)

    '''
    if type(result) is dict:
        pred=result[pred_name]
        pred0=result[pred0_name]
    else:
        pred = result[pred_name].as_continuous()
        pred0 = result[pred0_name].as_continuous()
        
    pred_std = pred.std(axis=1)
    E = np.std(pred_std-resp_std)/np.std(resp_std)
    
    pcproj = (pred-pred0).T.dot(pc_axes.T).T
    
    for idx,pc_act in zip(group_idx, group_pc):
        #group_pc = [pcproj[:,idx].std(axis=1) for idx in group_idx]
        E += np.sum(np.abs(pcproj[:,idx].std(axis=1) - pc_act) / pc_act)

    return E


def fit_pcnorm(modelspec,
        est: recording.Recording,
        metric = None,
        use_modelspec_init: bool = True,
        optimizer: str = 'adam',
        max_iter: int = 10000,
        early_stopping_steps: int = 5,
        tolerance: float = 5e-4,
        learning_rate: float = 1e-4,
        batch_size: typing.Union[None, int] = None,
        seed: int = 0,
        initializer: str = 'random_normal',
        freeze_layers: typing.Union[None, list] = None,
        epoch_name: str = "REFERENCE",
        n_pcs = 2,
        **context):
    '''
    Required Arguments:
     est          A recording object
     modelspec     A modelspec object

    Optional Arguments:
     <copied from fit_tf for now

    Returns
     dictionary: {'modelspec': updated_modelspec}
    '''

    # Hard-coded
    cost_function = basic_cost
    fitter = scipy_minimize
    segmentor = nems0.segmentors.use_all_data
    mapper = nems0.fitters.mappers.simple_vector
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    start_time = time.time()

    modelspec = copy.deepcopy(modelspec)

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    if 'mask' in est.signals.keys():
        log.info("Data len pre-mask: %d", est['mask'].shape[1])
        est = est.apply_mask()
        log.info("Data len post-mask: %d", est['mask'].shape[1])

    conditions = ["_".join(k.split("_")[1:]) for k in est.signals.keys() if k.startswith("mask_")]
    if (len(conditions)>2) and any([c.split("_")[-1]=='lg' for c in conditions]):
        conditions.remove("small")
        conditions.remove("large")
    #conditions = conditions[0:2]
    #conditions = ['large','small']
    
    group_idx = [est['mask_'+c].as_continuous()[0,:] for c in conditions]
    cg_filtered = [(c, g) for c,g in zip(conditions,group_idx) if g.sum()>0]
    conditions, group_idx = zip(*cg_filtered)
    
    for c,g in zip(conditions, group_idx):
        log.info(f"Data subset for {c} len {g.sum()}")
        
    resp = est['resp'].as_continuous()
    pred0 = est['pred0'].as_continuous()
    residual = resp - pred0
    
    pca = PCA(n_components=n_pcs)
    pca.fit(residual.T)
    pc_axes = pca.components_
    
    pcproj = residual.T.dot(pc_axes.T).T
    
    group_pc = [pcproj[:,idx].std(axis=1) for idx in group_idx]
    resp_std = resp.std(axis=1)
    #import pdb; pdb.set_trace()
    
    if metric is None:
        metric = lambda d: pc_err(d, pred_name='pred', pred0_name='pred0',
                                  group_idx=group_idx, group_pc=group_pc, pc_axes=pc_axes, resp_std=resp_std)

    # turn on "fit mode". currently this serves one purpose, for normalization
    # parameters to be re-fit for the output of each module that uses
    # normalization. does nothing if normalization is not being used.
    ms.fit_mode_on(modelspec, est)

    # Create the mapper functions that translates to and from modelspecs.
    # It has three functions that, when defined as mathematical functions, are:
    #    .pack(modelspec) -> fitspace_point
    #    .unpack(fitspace_point) -> modelspec
    #    .bounds(modelspec) -> fitspace_bounds
    packer, unpacker, pack_bounds = mapper(modelspec)

    # A function to evaluate the modelspec on the data
    evaluator = nems0.modelspec.evaluate

    my_cost_function = cost_function
    my_cost_function.counter = 0

    # Freeze everything but sigma, since that's all the fitter should be
    # updating.
    cost_fn = partial(my_cost_function,
                      unpacker=unpacker, modelspec=modelspec,
                      data=est, segmentor=segmentor, evaluator=evaluator,
                      metric=metric, display_N=1000)

    # get initial sigma value representing some point in the fit space,
    # and corresponding bounds for each value
    sigma = packer(modelspec)
    bounds = pack_bounds(modelspec)

    # Results should be a list of modelspecs
    # (might only be one in list, but still should be packaged as a list)
    improved_sigma = fitter(sigma, cost_fn, bounds=bounds, **fit_kwargs)
    improved_modelspec = unpacker(improved_sigma)
    elapsed_time = (time.time() - start_time)

    start_err = cost_fn(sigma)
    final_err = cost_fn(improved_sigma)
    log.info("Delta error: %.06f - %.06f = %e", start_err, final_err, final_err-start_err)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.fit_mode_off(improved_modelspec)
    ms.set_modelspec_metadata(improved_modelspec, 'fitter', 'ccnorm')
    ms.set_modelspec_metadata(improved_modelspec, 'fit_time', elapsed_time)
    ms.set_modelspec_metadata(improved_modelspec, 'n_parms', len(improved_sigma))

    return {'modelspec': improved_modelspec.copy(),
            'save_context': True}

