import logging
import re
import copy
import os

import numpy as np

from nems0.plugins import default_keywords
from nems0.utils import find_module, get_default_savepath
from nems0.analysis.api import fit_basic
from nems0.fitters.api import scipy_minimize
from nems0 import xforms
import nems0.priors as priors
import nems0.modelspec as ms
from nems0.uri import save_resource, load_resource
import nems0.metrics.api as metrics
from nems0 import get_setting

log = logging.getLogger(__name__)
#_kws = KeywordRegistry()
#default_kws.register_module(default_keywords)
#default_kws.register_plugins(get_setting('KEYWORD_PLUGINS'))


def fill_keyword_string_values(keyword_string,
                               rec=None, rec_list=None,
                               input_name='stim', output_name='resp', **ctx):

    if rec_list is None:
        rec_list=[rec]
    cell_count = len(rec_list)

    keywords = keyword_string.split('-')
    new_keywords = []
    _rec=rec_list[0]

    for kw in keywords:

        shared = True
        if (kw.startswith("fir.Nx") or kw.startswith("wc.Nx")) and \
                (_rec is not None):
            N = _rec[input_name].nchans
            kw_old = kw
            kw = kw.replace(".N", ".{}".format(N))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)
        elif kw.startswith("stategain.N") and (_rec is not None):
            N = _rec['state'].nchans
            kw_old = kw
            kw = kw.replace("stategain.N", "stategain.{}".format(N))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)
        elif (kw.endswith(".N")) and (_rec is not None):
            N = _rec[input_name].nchans
            kw_old = kw
            kw = kw.replace(".N", ".{}".format(N))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)
        elif (kw.endswith(".cN")) and (_rec is not None):
            N = _rec[input_name].nchans
            kw_old = kw
            kw = kw.replace(".cN", ".c{}".format(N))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)

        elif (kw.endswith("xN")) and (_rec is not None):
            N = _rec[input_name].nchans
            kw_old = kw
            kw = kw.replace("xN", "x{}".format(N))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)

        elif ("xN" in kw) and (_rec is not None):
            N = _rec[input_name].nchans
            kw_old = kw
            kw = kw.replace("xN", "x{}".format(N))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)

        if (".S" in kw or ".Sx" in kw) and (_rec is not None):
            S = _rec['state'].nchans
            kw_old = kw
            kw = kw.replace(".S", ".{}".format(S))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)

        if ("xS" in kw) and (_rec is not None):
            S = _rec['state'].nchans
            kw_old = kw
            kw = kw.replace("xS", "x{}".format(S))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)

        if (".R" in kw) and (_rec is not None):
            R = _rec[output_name].nchans
            kw_old = kw
            kw = kw.replace(".R", ".{}".format(R))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)
            shared = False  #  "R" means output-dependent
        if ("xR" in kw) and (_rec is not None):
            R = _rec[output_name].nchans
            kw_old = kw
            kw = kw.replace("xR", "x{}".format(R))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)
            shared = False  #  "R" means output-dependent

        other_xr = re.findall(r"[x\.]?[0-9]+[R]", kw)
        if bool(other_xr):
            R = _rec[output_name].nchans
            kw_old = kw
            digits = int(re.findall("[0-9]+", other_xr[0])[0])
            kw = kw.replace("{}R".format(digits), "{}".format(digits*R))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)
            shared = False  #  "R" means output-dependent

        log.info('kw: %s', kw)
        new_keywords.append(kw)

    return "-".join(new_keywords)

def from_keywords(keyword_string, registry=None, rec=None, meta={}, rec_list=None, branch_at=0,
                  init_phi_to_mean_prior=True, input_name='stim', output_name='resp'):
    '''
    Returns a modelspec created by splitting keyword_string on underscores
    and replacing each keyword with what is found in the nems0.keywords.defaults
    registry. You may provide your own keyword registry using the
    registry={...} argument.
    
    rec_list : if not None, should be a list of recordings. This will generate modelspec with multiple cells, 
    branching at branch_at
    '''

    if registry is None:
        from nems0.xforms import keyword_lib
        registry = keyword_lib
    keywords = keyword_string.split('-')

    # Lookup the modelspec fragments in the registry
    if rec_list is None:
        rec_list=[rec]
    cell_count = len(rec_list)

    modelspec = ms.ModelSpec(cell_count=cell_count)
    destination = None
    for cell_index, _rec in enumerate(rec_list):
        modelspec.cell_index = cell_index
        shared_modules = []
        for kw in keywords:
            shared = True
            if (kw.startswith("fir.Nx") or kw.startswith("wc.Nx")) and \
                    (_rec is not None):
                N = _rec[input_name].nchans
                kw_old = kw
                kw = kw.replace(".N", ".{}".format(N))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)
            elif kw.startswith("stategain.N") and (_rec is not None):
                N = _rec['state'].nchans
                kw_old = kw
                kw = kw.replace("stategain.N", "stategain.{}".format(N))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)
            elif (kw.endswith(".N")) and (_rec is not None):
                N = _rec[input_name].nchans
                kw_old = kw
                kw = kw.replace(".N", ".{}".format(N))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)
            elif (kw.endswith(".cN")) and (_rec is not None):
                N = _rec[input_name].nchans
                kw_old = kw
                kw = kw.replace(".cN", ".c{}".format(N))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)

            elif (kw.endswith("xN")) and (_rec is not None):
                N = _rec[input_name].nchans
                kw_old = kw
                kw = kw.replace("xN", "x{}".format(N))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)

            elif ("xN" in kw) and (_rec is not None):
                N = _rec[input_name].nchans
                kw_old = kw
                kw = kw.replace("xN", "x{}".format(N))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)

            if (".S" in kw or ".Sx" in kw) and (_rec is not None):
                S = _rec['state'].nchans
                kw_old = kw
                kw = kw.replace(".S", ".{}".format(S))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)

            if ("xS" in kw) and (_rec is not None):
                S = _rec['state'].nchans
                kw_old = kw
                kw = kw.replace("xS", "x{}".format(S))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)

            if (".R" in kw) and (_rec is not None):
                R = _rec[output_name].nchans
                kw_old = kw
                kw = kw.replace(".R", ".{}".format(R))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)
                shared = False  #  "R" means output-dependent
            if ("xR" in kw) and (_rec is not None):
                R = _rec[output_name].nchans
                kw_old = kw
                kw = kw.replace("xR", "x{}".format(R))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)
                shared = False  #  "R" means output-dependent

            other_xr = re.findall(r"[x\.]?[0-9]+[R]", kw)
            if bool(other_xr):
                R = _rec[output_name].nchans
                kw_old = kw
                digits = int(re.findall("[0-9]+", other_xr[0])[0])
                kw = kw.replace("{}R".format(digits), "{}".format(digits*R))
                log.info("kw: dynamically subbing %s with %s", kw_old, kw)
                shared = False  #  "R" means output-dependent

            log.info('kw: %s', kw)
            shared_modules.append(shared)
            if registry.kw_head(kw) not in registry:
                raise ValueError("unknown keyword: {}".format(kw))

            templates = copy.deepcopy(registry[kw])
            if not isinstance(templates, list):
                templates = [templates]
            for d in templates:
                d['id'] = kw
                if init_phi_to_mean_prior:
                    d = priors.set_mean_phi([d])[0]  # Inits phi for 1 module
                modelspec.append(d)
        shared_modules[-1]=False
        shared_modules=np.array(shared_modules)
        shared_count = np.min(np.where(shared_modules==False)[0])
        modelspec.shared_count = shared_count

        # first module that takes input='pred' should take ctx['input_name']
        # instead. can't hard-code in keywords, since we don't know which
        # keyword will be first. and can't assume that it will be module[0]
        # because those might be state manipulations
        first_input_found = False
        i = 0
        while (not first_input_found) and (i < len(modelspec)):
            if ('i' in modelspec[i]['fn_kwargs'].keys()) and (modelspec[i]['fn_kwargs']['i'] == 'pred'):
                log.info("Setting modelspec[%d] input to %s", i, input_name)
                modelspec[i]['fn_kwargs']['i'] = input_name
                """ OLD
                if input_name != 'stim':
                    modelspec[i]['fn_kwargs']['i'] = input_name
                elif 'state' in modelspec[i]['fn']:
                    modelspec[i]['fn_kwargs']['i'] = 'psth'
                else:
                    modelspec[i]['fn_kwargs']['i'] = input_name
                """

                # 'i' key found
                first_input_found = True
            i += 1

        # insert metadata, if provided
        if _rec is not None:
            if 'cellids' in meta.keys():
                # cellids list already exists. keep it.
                pass
            elif 'cellids' in _rec.meta.keys():
                # cellids list already exists. keep it.
                meta['cellids'] = _rec.meta['cellids'].copy()
            elif ((_rec['resp'].shape[0] > 1) and (type(_rec.meta['cellid']) is list)):
                # guess cellids list from rec.meta
                meta['cellids'] = _rec.meta['cellid']
            elif 'cellid' in meta.keys():
                meta['cellids'] = [meta['cellid']]

        meta['input_name'] = input_name
        meta['output_name'] = output_name

        # for modelspec object, we know that meta must exist, so just update
        modelspec.meta.update(meta)

        if destination is None:
            destination = get_default_savepath(modelspec)
        modelspec.meta['modelpath'] = destination
        modelspec.meta['figurefile'] = os.path.join(destination, 'figure.0000.png')

        #modelspec.meta.update(meta)

        print(destination, modelspec.meta['modelpath'])
        if 'cellids' in meta.keys():
            del meta['cellids']

    modelspec.cell_index = 0
        
    return modelspec


def from_keywords_as_list(keyword_string, registry=None, meta={}):
    '''
    wrapper for from_keywords that returns modelspec as a modelspecs list,
    ie, [modelspec]
    '''
    if registry is None:
        registry = default_kws
    return [from_keywords(keyword_string, registry, meta)]


def rand_phi(modelspec, rand_count=10, IsReload=False, skip_init=False,
             freeze_idx=[],
             rand_seed=1234, **context):
    """ 
    initialize modelspec phi to random values based on priors
    2021-11-08: svd added support for multiple-cell modelspecs
    """

    if IsReload or skip_init:
        return {}
    jack_count = modelspec.jack_count
    modelspec = modelspec.copy(jack_index=0)

    modelspec.tile_fits(rand_count)
    log.info(f'randomizing {rand_count} models, freeze_idx={freeze_idx}')

    # set random seed for reproducible results
    save_state = np.random.get_state()
    np.random.seed(rand_seed)

    for cell_idx in range(modelspec.cell_count):
        modelspec.set_cell(cell_idx)
        save_phi = modelspec.phi.copy()
        for i in range(rand_count):
            modelspec.set_fit(i)
            if i == 0:
                # make first one mean of priors:
                modelspec = priors.set_mean_phi(modelspec)
            else:
                modelspec = priors.set_random_phi(modelspec)
            for j in freeze_idx:
                log.debug(f'fit {i} reverting frozen layer {j}')
                if modelspec.phi[j]:
                    for k in modelspec.phi[j].keys():
                        modelspec.phi[j][k] = save_phi[j][k].copy()

    modelspec.set_cell(0)
    modelspec.set_fit(0)

    # restore random seed
    np.random.set_state(save_state)

    modelspec.tile_jacks(jack_count)

    return {'modelspec': modelspec}

def load_phi(modelspec, prefit_uri="", prefit_modelspec=None, copy_layers=0, **kwargs):
    """
    copy the first copy_layers of a previously fit modelspec into
    the current modelspec. barfs if the phi sizes don't match exactly
    """
    if prefit_modelspec is None:
        _, ctx_old = xforms.load_analysis(prefit_uri, eval_model=False)
        prefit_modelspec = ctx_old['modelspec']
        #prefit_modelspec = ms.ModelSpec([load_resource(prefit_uri)])

    # copy phi from existing modelspec to the other:
    for cellidx in range(modelspec.cell_count):
        modelspec.set_cell(cellidx)
        prefit_modelspec.set_cell(cellidx)

        for i, phi in enumerate(prefit_modelspec.phi[:copy_layers]):
            if phi is not None:
                for k in phi.keys():
                    log.info(f"{i} {k} {phi[k].shape}")
                    #import pdb; pdb.set_trace()
                    if (k not in modelspec.phi[i].keys()):
                        log.info(f'module {i} creating phi entry for {k}')
                        modelspec.phi[i][k] = phi[k]
                    elif phi[k].shape == modelspec.phi[i][k].shape:
                        modelspec.phi[i][k] = phi[k]
                    else:
                        raise ValueError(f"new shape mismatch {modelspec.phi[i][k].shape}")
    modelspec.set_cell(0)
    prefit_modelspec.set_cell(0)

    return {'modelspec': modelspec, 'old_modelspec': prefit_modelspec}


def prefit_LN(rec, modelspec, analysis_function=fit_basic,
              fitter=scipy_minimize, metric=None, norm_fir=False,
              tolerance=10**-5.5, max_iter=1500, nl_kw={}):
    '''
    Initialize modelspecs in a way that avoids getting stuck in
    local minima.

    written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)

    input: a single est recording and a single modelspec

    output: a single modelspec

    TODO -- make sure this works generally or create alternatives

    '''
    log.info('prefit_LN parameters: tol=%.2e max_iter=%d', tolerance, max_iter)
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    # Instead of using FIR prior, initialize to random coefficients then
    # divide by L2 norm to force sum of squares = 1
    if norm_fir:
        modelspec = fir_L2_norm(modelspec)

    # fit without STP module first (if there is one)
    modelspec = prefit_to_target(rec, modelspec, fit_basic,
                                 target_module=['levelshift', 'relu'],
                                 extra_exclude=['stp', 'rdt_gain','state_dc_gain','state_gain'],
                                 fitter=fitter,
                                 metric=metric,
                                 fit_kwargs=fit_kwargs)

    # then initialize the STP module (if there is one)
    for i, m in enumerate(modelspec):
        if 'stp' in m['fn'] and m.get('phi') is None:
            m = priors.set_mean_phi([m])[0]  # Init phi for module
            modelspec[i] = m
            break

    # pre-fit static NL if it exists
    d = init_static_nl(rec, modelspec, tolerance=tolerance, **nl_kw)
    modelspec = d['modelspec']

    return modelspec


def init_static_nl(est=None, modelspec=None, tolerance=10**-4, **nl_kw):

    include_names = []
    if (est is None) or (modelspec is None):
        raise ValueError('est and modelspec paramters required')

    # pre-fit static NL if it exists
    for m in modelspec.modules[-2:]:
        if 'double_exponential' in m['fn']:
            modelspec = init_dexp(est, modelspec, **nl_kw)
            include_names = ['double_exponential']
            modelspec = prefit_subset(est, modelspec,
                                      include_names=include_names,
                                      tolerance=tolerance)
            break

        elif 'relu' in m['fn']:
            m['phi']['offset'][:]=-0.1
            include_names = ['relu']
            log.info('pre-fitting relu from -0.1')
            # in case there are other relus, just fit last module
            modelspec = prefit_subset(est, modelspec,
                                      include_idx=[len(modelspec)-1],
                                      tolerance=10**-4)
            break

        elif 'logistic_sigmoid' in m['fn']:
            log.info("initializing priors and bounds for logsig ...\n")
            modelspec = init_logsig(est, modelspec)
            include_names = ['logistic_sigmoid']
            modelspec = prefit_subset(est, modelspec,
                                      include_names=include_names,
                                      tolerance=tolerance)
            break

        elif 'saturated_rectifier' in m['fn']:
            log.info('initializing priors and bounds for relat ...\n')
            modelspec = init_relsat(est, modelspec)
            include_names = ['saturated_rectifier']
            modelspec = prefit_subset(est, modelspec,
                                      include_names=include_names,
                                      tolerance=tolerance)
            break

    return {'modelspec': modelspec, 'include_names': include_names}


def prefit_to_target(rec, modelspec, analysis_function, target_module,
                     extra_exclude=[],
                     fitter=scipy_minimize, metric=None,
                     fit_kwargs={}):
    """Removes all modules from the modelspec that come after the
    first occurrence of the target module, then performs a
    rough fit on the shortened modelspec, then adds the latter
    modules back on and returns the full modelspec.
    """

    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    # figure out last modelspec module to fit
    target_i = None
    if type(target_module) is not list:
        target_module = [target_module]
    for i, m in enumerate(modelspec.modules):
        tlist = [True for t in target_module if t in m['fn']]

        if len(tlist):
            target_i = i + 1
            # don't break. use last occurrence of target module

    if not target_i:
        log.info("target_module: {} not found in modelspec."
                 .format(target_module))
        return modelspec
    else:
        log.info("target_module: {0} found at modelspec[{1}]."
                 .format(target_module, target_i-1))

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    exclude_idx = []
    freeze_idx = []
    include_idx = []
    for i in range(len(modelspec)):
        m = modelspec[i]

        for fn in extra_exclude:
            if (fn in m['fn']):
                freeze_idx.append(i)

        if ('relu' in m['fn']):
            log.info('found relu')

        elif ('levelshift' in m['fn']):
            #m = priors.set_mean_phi([m])[0]
            output_name = modelspec.meta.get('output_name', 'resp')
            try:
                mean_resp = np.nanmean(rec[output_name].as_continuous(), axis=1, keepdims=True)
            except NotImplementedError:
                # as_continuous only available for RasterizedSignal
                mean_resp = np.nanmean(rec[output_name].rasterize().as_continuous(), axis=1, keepdims=True)
            if len(mean_resp)==len(m['phi']['level'][:]):
                log.info('Mod %d (%s) initializing level to %s mean %.3f',
                         i, m['fn'], output_name, mean_resp[0])
                log.info('Output %s has %d channels',
                         output_name, len(mean_resp))
                m['phi']['level'][:] = mean_resp

        if (i < target_i) or ('merge_channels' in m['fn']):
            include_idx.append(i)
    exclude_idx=np.setdiff1d(np.arange(len(modelspec)), np.union1d(include_idx, freeze_idx))
    modelspec = prefit_subset(rec, modelspec, analysis_function,
                  include_idx=include_idx,
                  exclude_idx=exclude_idx,
                  freeze_idx=freeze_idx,
                  fitter=fitter, metric=metric, **fit_kwargs)

    return modelspec


def _prefit_to_target(rec, modelspec, analysis_function, target_module,
                     extra_exclude=[],
                     fitter=scipy_minimize, metric=None,
                     fit_kwargs={}):
    """Removes all modules from the modelspec that come after the
    first occurrence of the target module, then performs a
    rough fit on the shortened modelspec, then adds the latter
    modules back on and returns the full modelspec.
    """

    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    # figure out last modelspec module to fit
    target_i = None
    if type(target_module) is not list:
        target_module = [target_module]
    for i, m in enumerate(modelspec.modules):
        tlist = [True for t in target_module if t in m['fn']]

        if len(tlist):
            target_i = i + 1
            # don't break. use last occurrence of target module

    if not target_i:
        log.info("target_module: {} not found in modelspec."
                 .format(target_module))
        return modelspec
    else:
        log.info("target_module: {0} found at modelspec[{1}]."
                 .format(target_module, target_i-1))

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    exclude_idx = []
    tmodelspec = ms.ModelSpec()
    for i in range(len(modelspec)):
        m = copy.deepcopy(modelspec[i])

        for fn in extra_exclude:
            if (fn in m['fn']):
                if (m.get('phi') is None):
                    m = priors.set_mean_phi([m])[0]  # Inits phi
                    log.info('Mod %d (%s) fixing phi to prior mean', i, fn)
                else:
                    log.info('Mod %d (%s) fixing phi', i, fn)

                m['fn_kwargs'].update(m['phi'])
                del m['phi']
                del m['prior']
                exclude_idx.append(i)

        if ('relu' in m['fn']):
            log.info('found relu')

        elif ('levelshift' in m['fn']):
            #m = priors.set_mean_phi([m])[0]
            output_name = modelspec.meta.get('output_name', 'resp')
            try:
                mean_resp = np.nanmean(rec[output_name].as_continuous(), axis=1, keepdims=True)
            except NotImplementedError:
                # as_continuous only available for RasterizedSignal
                mean_resp = np.nanmean(rec[output_name].rasterize().as_continuous(), axis=1, keepdims=True)
            log.info('Mod %d (%s) initializing level to %s mean %.3f',
                     i, m['fn'], output_name, mean_resp[0])
            log.info('resp has %d channels', len(mean_resp))
            m['phi']['level'][:] = mean_resp

        if (i < target_i) or ('merge_channels' in m['fn']):
            tmodelspec.append(m)

    # fit the subset of modules
    if metric is None:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       fit_kwargs=fit_kwargs)
    else:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       metric=metric, fit_kwargs=fit_kwargs)
    if type(tmodelspec) is list:
        # backward compatibility
        tmodelspec = tmodelspec[0]

    # reassemble the full modelspec with updated phi values from tmodelspec
    #print(modelspec[0])
    #print(modelspec.phi[2])
    for i in np.setdiff1d(np.arange(target_i), np.array(exclude_idx)).tolist():
        modelspec[int(i)] = tmodelspec[int(i)]

    return modelspec


def prefit_mod_subset(rec, modelspec, analysis_function,
                      fit_set=[],
                      fitter=scipy_minimize, metric=None, fit_kwargs={}):
    """Removes all modules from the modelspec that come after the
    first occurrence of the target module, then performs a
    rough fit on the shortened modelspec, then adds the latter
    modules back on and returns the full modelspec.
    """

    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
#    fit_idx = []
#    tmodelspec = []
#    for i, m in enumerate(modelspec):
#        m = copy.deepcopy(m)
#        for fn in fit_set:
#            if fn in m['fn']:
#                fit_idx.append(i)
#                log.info('Found module %d (%s) for subset prefit', i, fn)
#        tmodelspec.append(m)

    if type(fit_set[0]) is int:
        fit_idx = fit_set
    else:
        fit_idx = []
        for i, m in enumerate(modelspec.modules):
            for fn in fit_set:
                if fn in m['fn']:
                    fit_idx.append(i)
                    log.info('Found module %d (%s) for subset prefit', i, fn)

    tmodelspec = copy.deepcopy(modelspec)

    if len(fit_idx) == 0:
        log.info('No modules matching fit_set for subset prefit')
        return modelspec

    exclude_idx = np.setdiff1d(np.arange(0, len(modelspec)),
                               np.array(fit_idx)).tolist()
    for i in exclude_idx:
        m = tmodelspec[i]
        if not m.get('phi'):
            log.info('Intializing phi for module %d (%s)', i, m['fn'])
            m = priors.set_mean_phi([m])[0]  # Inits phi

        log.info('Freezing phi for module %d (%s)', i, m['fn'])

        m['fn_kwargs'].update(m['phi'])
        m['phi'] = {}
        tmodelspec[i] = m

    # fit the subset of modules
    if metric is None:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       fit_kwargs=fit_kwargs)
    else:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       metric=metric, fit_kwargs=fit_kwargs)

    # reassemble the full modelspec with updated phi values from tmodelspec
    for i in fit_idx:
        modelspec[i] = tmodelspec[i]

    return modelspec


def _calc_neg_modidx(idx_list, ms_len):
    if type(idx_list) is int:
        idx_list=np.array([idx_list], dtype=int)
    else:
        idx_list=np.array(idx_list)
    idx_list[idx_list<0]=ms_len+idx_list[idx_list<0]

    return idx_list

def modelspec_freeze_layers(modelspec,
                            include_idx=None, include_through=None,
                            include_names=None,
                            exclude_idx=None, exclude_after=None,
                            freeze_idx=None, freeze_after=None
                            ):

    mslen = len(modelspec)

    # first, figure out which modules to fit
    include_set = []
    if include_names is not None:
        for i, m in enumerate(modelspec.modules):
            for fn in include_names:
                if fn in m['fn']:
                    include_set.append(i)
                    log.info('Found module %d (%s) for subset prefit', i, fn)
    include_set = np.array(include_set, dtype=int)
    if include_through is not None:
        include_set = np.union1d(include_set, np.arange(_calc_neg_modidx(include_through, mslen), dtype=int))
    if include_idx is not None:
        include_set = np.union1d(include_set, _calc_neg_modidx(include_idx, mslen))
    if len(include_set) == 0:
        include_set = np.arange(mslen, dtype=int)

    # then figure out which ones to freeze, possibly overwriting some of the fits
    freeze_set = np.setdiff1d(np.arange(mslen), include_set)
    if freeze_idx is not None:
        freeze_set = np.union1d(freeze_set, _calc_neg_modidx(freeze_idx, mslen))
    if freeze_after is not None:
        freeze_set = np.union1d(freeze_set, np.arange(_calc_neg_modidx(freeze_after, mslen)[0], mslen))
    include_set = np.setdiff1d(include_set, freeze_set)

    # then figure out which ones to exclude, again possibly overwriting the include/freeze sets
    exclude_set = []
    if exclude_idx is not None:
        exclude_set = np.union1d(exclude_set, _calc_neg_modidx(exclude_idx, mslen))
    if exclude_after is not None:
        exclude_set = np.union1d(exclude_set, np.arange(_calc_neg_modidx(exclude_after, mslen)[0], mslen, dtype=int))
    include_set = np.setdiff1d(include_set, exclude_set)
    freeze_set = np.setdiff1d(freeze_set, exclude_set)

    log.info("Fit: %s", include_set)
    log.info("Freeze: %s", freeze_set)
    log.info("Exclude: %s", exclude_set)

    if len(include_set) == 0:
        log.info('No modules matching include_set for subset prefit')
        return modelspec

    tmodelspec = copy.deepcopy(modelspec)

    for i in range(mslen):
        m = tmodelspec[i]
        if not m.get('phi'):
            log.info('Initializing phi for module %d (%s)', i, m['fn'])
            m = priors.set_mean_phi([m])[0]  # Inits phi
        if i in freeze_set:
            log.info('Freezing phi for module %d (%s)', i, m['fn'])
            m['fn_kwargs'].update(m['phi'])
            m['phi'] = {}
            tmodelspec[i] = m
        elif i in exclude_set:
            # replace with null module
            log.info('Excluding module %d (%s)', i, m['fn'])
            if type(m) is dict:
                # fn = _lookup_fn_at(m['fn'])
                m['fn'] = 'nems0.modules.scale.null'
            else:
                # it's a NemsModule object, need to divert the .eval method to null
                m.eval = ms._lookup_fn_at('nems0.modules.scale.null')
            m['phi'] = {}
            tmodelspec[i] = m

    return tmodelspec, include_set


def modelspec_unfreeze_layers(modelspec, modelspec0, include_set):
    """
    Generate a modelspec using inlcude_set layers from modelspec and other layers from modelspec0
    :param modelspec: modelspec with layers that were fit (include_set)
    :param modelspec0: pre-fit modelspec with appropriately defined frozen layers (~include_set)
    :param include_set: list of fit (non-frozen) layers
    :return: modelspec
    """
    modelspecnew = modelspec0.copy()
    meta_save = modelspec.meta.copy()
    for i in include_set:
        log.info('saving module %d', i)
        modelspecnew[i] = modelspec[i]

    modelspecnew.meta.update(meta_save)

    return modelspecnew


def prefit_subset(est, modelspec, analysis_function=fit_basic,
                  include_idx=None, include_through=None,
                  include_names=None,
                  exclude_idx=None, exclude_after=None,
                  freeze_idx=None, freeze_after=None,
                  fitter=scipy_minimize, metric=None,
                  tolerance=10**-5.5, max_iter=700,
                  **context):
    """
    Fit subset of modules in a modelspec. Exact behavior depends on how
    included/excluded modules are specified. Default is to fit all modules.
    Non-negative indexes specify modules from the start, negative from the
    end
    :param est:
    :param modelspec:
    :param analysis_function:
    :param include_idx: list of modules to include, if not specified, fit
                        all modules. By default, freeze all others.
    :param include_through: fit all modules up to and including this index
    :param freeze_idx: list of modules to freeze (move phi entries into
                       fn_kwargs)
    :param freeze_after: freeze all modules from this index to the end
    :param exclude_idx: modules to remove before fitting, should run faster
                        than freezing, but user must ensure this doesn't
                        break the model
    :param exclude_after: exclude all modules from this index to the end
    :param fitter:
    :param metric:
    :param fit_kwargs:
    :return: updated modelspec
    """

    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    tmodelspec, include_set = modelspec_freeze_layers(modelspec,
        include_idx=include_idx, include_through=include_through,
        include_names=include_names,
        exclude_idx=exclude_idx, exclude_after=exclude_after,
        freeze_idx=freeze_idx, freeze_after=freeze_after)

    """
    # preserve input modelspec
    tmodelspec = copy.deepcopy(modelspec)
    mslen=len(modelspec)

    #first, figure out which modules to fit
    include_set = []
    if include_names is not None:
        for i, m in enumerate(modelspec.modules):
            for fn in include_names:
                if fn in m['fn']:
                    include_set.append(i)
                    log.info('Found module %d (%s) for subset prefit', i, fn)
    include_set = np.array(include_set, dtype=int)
    if include_through is not None:
        include_set=np.union1d(include_set, np.arange(_calc_neg_modidx(include_through, mslen), dtype=int))
    if include_idx is not None:
        include_set=np.union1d(include_set, _calc_neg_modidx(include_idx, mslen))
    if len(include_set)==0:
        include_set = np.arange(mslen, dtype=int)

    # then figure out which ones to freeze, possibly overwriting some of the fits
    freeze_set = np.setdiff1d(np.arange(mslen), include_set)
    if freeze_idx is not None:
        freeze_set=np.union1d(freeze_set,_calc_neg_modidx(freeze_idx, mslen))
    if freeze_after is not None:
        freeze_set = np.union1d(freeze_set, np.arange(_calc_neg_modidx(freeze_after, mslen)[0], mslen))
    include_set = np.setdiff1d(include_set, freeze_set)

    # then figure out which ones to exclude, again possibly overwriting the include/freeze sets
    exclude_set = []
    if exclude_idx is not None:
        exclude_set=np.union1d(exclude_set, _calc_neg_modidx(exclude_idx,mslen))
    if exclude_after is not None:
        exclude_set = np.union1d(exclude_set, np.arange(_calc_neg_modidx(exclude_after, mslen)[0], mslen, dtype=int))
    include_set = np.setdiff1d(include_set, exclude_set)
    freeze_set = np.setdiff1d(freeze_set, exclude_set)

    log.info("Fit: %s", include_set)
    log.info("Freeze: %s", freeze_set)
    log.info("Exclude: %s", exclude_set)

    if len(include_set) == 0:
        log.info('No modules matching include_set for subset prefit')
        return modelspec

    tmodelspec = copy.deepcopy(modelspec)
    for i in range(mslen):
        m = tmodelspec[i]
        if not m.get('phi'):
            log.info('Intializing phi for module %d (%s)', i, m['fn'])
            m = priors.set_mean_phi([m])[0]  # Inits phi
        if i in freeze_set:
            log.info('Freezing phi for module %d (%s)', i, m['fn'])
            m['fn_kwargs'].update(m['phi'])
            m['phi'] = {}
            tmodelspec[i] = m
        elif i in exclude_set:
            # replace with null module
            log.info('Excluding module %d (%s)', i, m['fn'])
            if type(m) is dict:
                #fn = _lookup_fn_at(m['fn'])
                m['fn'] = 'nems0.modules.scale.null'
            else:
                #it's a NemsModule object, need to divert the .eval method to null
                m.eval = ms._lookup_fn_at('nems0.modules.scale.null')
            m['phi'] = {}
            tmodelspec[i] = m
    """

    # fit the subset of modules
    tmodelspec = analysis_function(est, tmodelspec, fitter=fitter,
                                   metric=metric, fit_kwargs=fit_kwargs)

    # pull out updated phi values from tmodelspec, include_set only
    modelspec = modelspec_unfreeze_layers(tmodelspec, modelspec, include_set)

    return modelspec


def modelspec_remove_input_layers(modelspec, rec, remove_count=0):
    """
    remove the first remove_count layers from a modelspec, evaluate rec so that the "stim" is the output of layer remove_count
    return the modified rec as well.
    """
    input_name = modelspec.meta['input_name']
    if remove_count == 0:
        log.info('modelspec_remove_input_layers doing nothing')
        rec.signals['saved_input'] = rec[input_name].copy()

        return modelspec.copy(), rec
    
    # generate recording starting at layer remove_count
    dropcount = len(modelspec)-remove_count
    modelspec_dropped = modelspec.copy()
    for i in range(dropcount):
        modelspec_dropped.drop_module(in_place=True)

    rec_trunc = modelspec_dropped.evaluate(rec.copy(), stop=remove_count)
    rec_trunc.signals['saved_input'] = rec_trunc[input_name].copy()
    rec_trunc.signals[input_name] = rec_trunc["pred"]._modified_copy(data=rec_trunc["pred"]._data.copy())

    raw_trunc = copy.deepcopy(modelspec.raw)
    for i_ in range(raw_trunc.shape[0]):
        for j_ in range(raw_trunc.shape[1]):
            for k_ in range(raw_trunc.shape[2]):
                raw_trunc[i_,j_,k_] = modelspec.raw[i_,j_,k_][remove_count:]
                raw_trunc[i_,j_,k_][0]['fn_kwargs']['i'] = "stim"

        raw_trunc[i_,0,0][0]['meta'] = copy.deepcopy(modelspec.raw[i_,0,0][0]['meta'])
    
    modelspec_trunc = ms.ModelSpec(raw=raw_trunc, cell_count=modelspec.cell_count, cell_index=modelspec.cell_index)
    modelspec_trunc.fit_index = modelspec.fit_index
    modelspec_trunc.jack_index = modelspec.jack_index
    
    return modelspec_trunc, rec_trunc


def modelspec_restore_input_layers(modelspec_trunc, rec_trunc, modelspec_original):
    """
    put a newly fit truncated modelspec back on top of the frozen layers from modelspec_original
    """
    restore_stop = len(modelspec_original)
    restore_start = restore_stop - len(modelspec_trunc)
    
    log.info(f"Updating modules {restore_start} to {restore_stop} in modelspec_original")
    
    modelspec_restored = modelspec_original.copy()
    for i_ in range(modelspec_restored.raw.shape[0]):
        for j_ in range(modelspec_restored.raw.shape[1]):
            for k_ in range(modelspec_restored.raw.shape[2]):
                for r in range(restore_stop-restore_start):
                    modelspec_restored.raw[i_,j_,k_][r+restore_start] = modelspec_trunc.raw[i_,j_,k_][r]
                modelspec_restored.raw[i_,j_,k_][restore_start]['fn_kwargs']['i'] = "pred"

    # return recording to normal. maybe unnecessary, since original rec should have been saved?
    input_name = modelspec_trunc.meta['input_name']
    rec_restored = rec_trunc.copy()
    rec_restored.signals[input_name] = rec_restored.signals['saved_input']
    
    rec_restored = modelspec_restored.evaluate(rec_restored)
    
    return modelspec_restored, rec_restored

def modelspec_slice_output_layers(modelspec, rec, slice_channels, slice_at=-1):
    """
    Tweak the model to predict a subset of the response, for multi-dataset fitting
    return the modified rec as well.

    slice_channels : keep these output channels (numerical index!)
    slice_at : start slicing at layer slice_at (default -1 means last layer)
    """
    input_name = modelspec.meta['input_name']
    output_name = modelspec.meta['output_name']
    output_count_old = rec[output_name].shape[0]
    output_count_new = len(slice_channels)

    if output_count_new == output_count_old:
        log.info('modelspec_slice_output_layers doing nothing')
        rec.signals['saved_input'] = rec[output_name].copy()

        return modelspec.copy(), rec
    
    if slice_at <= 0:
        slice_at = len(modelspec)+slice_at

    log.info(f"Slicing modules {slice_at} to {len(modelspec)} from modelspec")

    # generate recording with sliced outputs
    rec_sliced = rec.copy()
    # don't need this, assumption is that the user is saving the original
    #rec_sliced.signals['saved_output'] = rec_sliced[output_name].copy()
    rec_sliced[output_name] = rec_sliced[output_name].extract_channels(
        chan_idx=slice_channels)

    raw_sliced = copy.deepcopy(modelspec.raw)
    for i_ in range(raw_sliced.shape[0]):
        for j_ in range(raw_sliced.shape[1]):
            for k_ in range(raw_sliced.shape[2]):
                for l_ in range(slice_at,len(modelspec)):
                    for parm in raw_sliced[i_,j_,k_][l_]['phi'].keys():
                        v = raw_sliced[i_,j_,k_][l_]['phi'][parm]
                        s_old=v.shape
                        if v.shape[0] == output_count_old:
                            v_new = v[slice_channels]
                            s_new=v_new.shape
                        elif v.shape[1] == output_count_old:
                            v_new = v[:, slice_channels]
                            s_new=v_new.shape

                        log.debug(f"raw[{i_},{j_},{k_}][{l_}]['phi'][{parm}]: {s_old} -> {s_new}")
                        raw_sliced[i_,j_,k_][l_]['phi'][parm] = v_new
    modelspec_sliced = ms.ModelSpec(raw=raw_sliced, cell_count=modelspec.cell_count, cell_index=modelspec.cell_index)
    modelspec_sliced.fit_index = modelspec.fit_index
    modelspec_sliced.jack_index = modelspec.jack_index

    for l_ in range(slice_at, len(modelspec_sliced)):
        if 'chans' in modelspec_sliced[l_]['fn_kwargs']:
            modelspec_sliced[l_]['fn_kwargs']['chans'] = len(slice_channels)

    return modelspec_sliced, rec_sliced

def modelspec_restore_sliced_output(modelspec_sliced, rec_sliced, slice_channels, slice_at, modelspec_original, rec_original):
    """
    put a newly fit output slice from a modelspec back into modelspec_original
    """
    output_name = modelspec_sliced.meta['output_name']
    output_count_old = rec_original[output_name].shape[0]
    if slice_at <= 0:
        slice_at = len(modelspec_original)+slice_at

    log.info(f"Inserting sliced modules {slice_at} to {len(modelspec_original)} into modelspec_original")

    modelspec_restored = modelspec_original.copy()
    raw_restored = modelspec_restored.raw
    raw_sliced = modelspec_sliced.raw

    for i_ in range(modelspec_restored.raw.shape[0]):
        for j_ in range(modelspec_restored.raw.shape[1]):
            for k_ in range(modelspec_restored.raw.shape[2]):
                for l_ in range(slice_at, len(modelspec_restored)):

                    for parm in raw_sliced[i_,j_,k_][l_]['phi'].keys():
                        v = raw_restored[i_,j_,k_][l_]['phi'][parm]
                        s_old=v.shape
                        if v.shape[0] == output_count_old:
                            v[slice_channels] = raw_sliced[i_,j_,k_][l_]['phi'][parm]
                            s_new = v[slice_channels].shape
                        elif v.shape[1] == output_count_old:
                            v[:, slice_channels] = raw_sliced[i_,j_,k_][l_]['phi'][parm]
                            s_new = v[:, slice_channels].shape

                        log.debug(f"raw[{i_},{j_},{k_}][{l_}]['phi'][{parm}]: {s_new} -> {s_old}")
                        raw_restored[i_,j_,k_][l_]['phi'][parm] = v

    return modelspec_restored



def init_dexp(rec, modelspec, nl_mode=2, override_target_i=None):
    """
    choose initial values for dexp applied after preceeding fir is
    initialized
    nl_mode must be in {1,2} (default is 2),
            pre 11/29/18 models were fit with v1
            1: amp = np.nanstd(resp) * 3
               kappa = np.log(2 / (np.max(pred) - np.min(pred) + 1))
            2:
               amp = resp[pred>np.percentile(pred,90)].mean()
               kappa = np.log(2 / (np.std(pred)*3))

    override_target_i should be an integer index into the modelspec.
    This replaces the normal behavior of the function which would look up
    the index of the 'double_exponential' module. Use this if you want
    to use dexp's initialization procedure for a similar nonlinearity module.

    """
    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    if override_target_i is None:
        target_i = find_module('double_exponential', modelspec)
        if target_i is None:
            log.warning("No dexp module was found, can't initialize.")
            return modelspec
    else:
        target_i = override_target_i

    if target_i == len(modelspec):
        fit_portion = modelspec.modules
    else:
        fit_portion = modelspec.modules[:target_i]

    # ensures all previous modules have their phi initialized
    # choose prior mean if not found
    for i, m in enumerate(fit_portion):
        if ('phi' not in m.keys()) and ('prior' in m.keys()):
            log.debug('Phi not found for module, using mean of prior: %s',
                      m)
            m = priors.set_mean_phi([m])[0]  # Inits phi for 1 module
            fit_portion[i] = m

    # generate prediction from module preceeding dexp
    #rec = ms.evaluate(rec, ms.ModelSpec(fit_portion))
    rec = ms.ModelSpec(fit_portion).evaluate(rec)

    in_signal = modelspec[target_i]['fn_kwargs']['i']
    pchans = rec[in_signal].shape[0]
    amp = np.zeros([pchans, 1])
    base = np.zeros([pchans, 1])
    kappa = np.zeros([pchans, 1])
    shift = np.zeros([pchans, 1])
    out_signal = modelspec.meta.get('output_name','resp')
    for i in range(pchans):
        resp = rec[out_signal].as_continuous()
        pred = rec[in_signal].as_continuous()[i:(i+1), :]
        if resp.shape[0] == pchans:
            resp = resp[i:(i+1), :]

        keepidx = np.isfinite(resp) * np.isfinite(pred)
        resp = resp[keepidx]
        pred = pred[keepidx]

        # choose phi s.t. dexp starts as almost a straight line
        # phi=[max_out min_out slope mean_in]
        # meanr = np.nanmean(resp)
        stdr = np.nanstd(resp)

        # base = np.max(np.array([meanr - stdr * 4, 0]))
        base[i, 0] = np.min(resp)

        # amp = np.max(resp) - np.min(resp)
        if nl_mode == 1:
            amp[i, 0] = stdr * 3
            predrange = 2 / (np.max(pred) - np.min(pred) + 1)
            shift[i, 0] = np.mean(pred)
            kappa[i, 0] = np.log(predrange)
        elif (nl_mode == 2) & (np.std(pred)==0):
            log.warning('Init dexp: channel %d prediction has zero std, reverting to nl_mode==1', i)
            amp[i, 0] = stdr * 3
            predrange = 2 / (np.max(pred) - np.min(pred) + 1)
            shift[i, 0] = np.mean(pred)
            kappa[i, 0] = np.log(predrange)
        elif nl_mode == 2:
            mask = np.zeros_like(pred, dtype=bool)
            pct = 91
            while (sum(mask) < .01*pred.shape[0]) and (pct > 1):
                pct -= 1
                mask = pred > np.percentile(pred, pct)
            if np.sum(mask) == 0:
                mask = np.ones_like(pred, dtype=bool)

            if pct !=90:
                log.warning('Init dexp: Default for init mode 2 is to find mean '
                         'of responses for times where pred>pctile(pred,90). '
                         '\nNo times were found so this was lowered to '
                         'pred>pctile(pred,%d).', pct)
            amp[i, 0] = resp[mask].mean()
            predrange = 2 / (np.std(pred)*3)
            if not np.isfinite(predrange):
                predrange = 1
            shift[i, 0] = np.mean(pred)
            kappa[i, 0] = np.log(predrange)
        elif nl_mode == 3:
            base[i, 0] = np.min(resp)-stdr
            amp[i, 0] = stdr * 4
            predrange = 1 / (np.std(pred)*3)

            shift[i, 0] = np.mean(pred)
            kappa[i, 0] = np.log(predrange)
        elif nl_mode ==4:
            base[i, 0] = np.mean(resp) - stdr * 1
            amp[i, 0] = stdr * 4
            predrange = 2 / (np.std(pred)*3)
            if not np.isfinite(predrange):
                predrange = 1
            kappa[i, 0] = 0
            shift[i, 0] = 0
        else:
            raise ValueError('nl mode = {} not valid'.format(nl_mode))

    modelspec[target_i]['phi'] = {'amplitude': amp, 'base': base,
                                  'kappa': kappa, 'shift': shift}
    if len(amp)<2:
       log.info("Init dexp: %s", modelspec[target_i]['phi'])
    else:
       log.info("Init dexp completed for %d channels ", len(modelspec[target_i]['phi']['amplitude']))

    return modelspec


def init_logsig(rec, modelspec):
    '''
    Initialization of priors for logistic_sigmoid,
    based on process described in methods of Rabinowitz et al. 2014.
    '''
    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    target_i = find_module('logistic_sigmoid', modelspec)
    if target_i is None:
        log.warning("No logsig module was found, can't initialize.")
        return modelspec

    if target_i == len(modelspec):
        fit_portion = modelspec.modules
    else:
        fit_portion = modelspec.modules[:target_i]

    # generate prediction from module preceeding dexp
    #rec = ms.evaluate(rec, ms.ModelSpec(fit_portion))
    rec = ms.ModelSpec(fit_portion).evaluate(rec)

    pred = rec['pred'].as_continuous()
    resp = rec['resp'].as_continuous()

    mean_pred = np.nanmean(pred)
    min_pred = np.nanmean(pred) - np.nanstd(pred)*3
    max_pred = np.nanmean(pred) + np.nanstd(pred)*3
    if min_pred < 0:
        min_pred = 0
        mean_pred = (min_pred+max_pred)/2

    pred_range = max_pred - min_pred
    min_resp = max(np.nanmean(resp)-np.nanstd(resp)*3, 0)  # must be >= 0
    max_resp = np.nanmean(resp)+np.nanstd(resp)*3
    resp_range = max_resp - min_resp

    # Rather than setting a hard value for initial phi,
    # set the prior distributions and let the fitter/analysis
    # decide how to use it.
    base0 = min_resp + 0.05*(resp_range)
    amplitude0 = resp_range
    shift0 = mean_pred
    kappa0 = pred_range
    log.info("Initial   base,amplitude,shift,kappa=({}, {}, {}, {})"
             .format(base0, amplitude0, shift0, kappa0))

    base = ('Exponential', {'beta': base0})
    amplitude = ('Exponential', {'beta': amplitude0})
    shift = ('Normal', {'mean': shift0, 'sd': pred_range})
    kappa = ('Exponential', {'beta': kappa0})

    if 'phi' in modelspec[target_i]:
        modelspec[target_i]['phi'].update({
                'base': base0, 'amplitude': amplitude0, 'shift': shift0,
                'kappa': kappa0})

    modelspec[target_i]['prior'].update({
            'base': base, 'amplitude': amplitude, 'shift': shift,
            'kappa': kappa})

    modelspec[target_i]['bounds'] = {
            'base': (1e-15, None),
            'amplitude': (1e-15, None),
            'shift': (None, None),
            'kappa': (1e-15, None)
            }

    return modelspec


def init_relsat(rec, modelspec):
    modelspec = copy.deepcopy(modelspec)

    target_i = find_module('saturated_rectifier', modelspec)
    if target_i is None:
        log.warning("No relsat module was found, can't initialize.")
        return modelspec

    if target_i == len(modelspec):
        fit_portion = modelspec.modules
    else:
        fit_portion = modelspec.modules[:target_i]

    # generate prediction from module preceeding dexp
    #rec = ms.evaluate(rec, ms.ModelSpec(fit_portion)).apply_mask()
    rec = ms.ModelSpec(fit_portion).evaluate(rec).apply_mask()

    pred = rec['pred'].as_continuous().flatten()
    resp = rec['resp'].as_continuous().flatten()
    stdr = np.nanstd(resp)

    base = np.min(resp)
    amplitude = min(np.mean(resp)+stdr*3, np.max(resp))
    shift = np.mean(pred) - 1.5*np.nanstd(pred)
    kappa = 1

    base_prior = ('Exponential', {'beta': base})
    amplitude_prior = ('Exponential', {'beta': amplitude})
    shift_prior = ('Normal', {'mean': shift, 'sd': shift})
    kappa_prior = ('Exponential', {'beta': kappa})

    modelspec['prior'] = {'base': base_prior, 'amplitude': amplitude_prior,
                          'shift': shift_prior, 'kappa': kappa_prior}

    return modelspec


def fir_L2_norm(modelspec):
    modelspec = copy.deepcopy(modelspec)
    fir_idx = find_module('fir', modelspec)
    prior = priors._tuples_to_distributions(modelspec[fir_idx]['prior'])
    random_coeffs = np.random.rand(*prior['coefficients'].mean().shape)
    normed = random_coeffs / np.linalg.norm(random_coeffs)
    # Assumes fir phi hasn't been initialized yet and that coefficients
    # is the only parameter to set. MAY NOT BE TRUE FOR SOME MODELS.
    modelspec[fir_idx]['phi'] = {'coefficients': normed}

    return modelspec
