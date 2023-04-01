#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:37:31 2018

@author: svd
"""

import copy
import logging
import time

import numpy as np

import nems0.analysis.api as analysis
import nems0.fitters.mappers
import nems0.initializers as init
import nems0.metrics.api
import nems0.metrics.api as metrics
import nems0.modelspec as ms
import nems0.priors as priors
from nems0.analysis.cost_functions import basic_cost
from nems0.fitters.api import scipy_minimize, coordinate_descent
from nems0.utils import find_module

log = logging.getLogger(__name__)


def init_pop_pca(est, modelspec, flip_pcs=False, IsReload=False,
                 pc_signal='pca', tolerance=1e-4, **context):
    """
    fit up through the fir module of a population model using the pca signal
    :param est: recording object with fit data
    :param modelspec: modelspec to be fit
    :param flip_pcs: use each pc to fit two channels, one with sign flipped
    :param IsReload: don't fit if IsReload=True
    :param pc_signal: string, name of signal to use for fitting population filters
       (by default PCs in 'pca')
    :param tolerance: tolerance for fitting each subspace filter and
       subsequent neural filter weights
    :param context: dictionary of other context variables
    :return: initialized modelspec
 """
    if IsReload:
        return {}

    # preserve input modelspec. necessary?
    modelspec = copy.deepcopy(modelspec)

    ifir = find_module('filter_bank', modelspec)
    try:
        dim_count = modelspec[ifir]['fn_kwargs'].get('bank_count', 1)
    except:
        dim_count = 1

    rec = est.copy()
    respcount = est['resp'].shape[0]
    fit_set_all, fit_set_slice = _figure_out_mod_split(modelspec)

    log.info('Initializing %d subspace channels with signal %s', dim_count, pc_signal)

    """
    iwc=find_module('weight_channels', modelspec)
    chan_per_bank = int(modelspec[iwc]['prior']['mean'][1]['mean'].shape[0]/dim_count)
    kw=[m['id'] for m in modelspec[:iwc]]

    wc = modelspec[iwc]['id'].split(".")
    wcs = wc[1].split("x")
    wcs[1]=str(chan_per_bank)
    wc[1]="x".join(wcs)
    wc=".".join(wc)

    fir = modelspec[ifir]['id'].split(".")
    fircore = fir[1].split("x")
    fir[1]="x".join(fircore[:-1])
    fir = ".".join(fir)

    kw.append(wc)
    kw.append(fir)
    kw.append("lvl.1")
    keywordstring="-".join(kw)
    keyword_lib = KeywordRegistry()
    keyword_lib.register_module(default_keywords)
    keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))
    """

    if flip_pcs:
        pc_fit_count = int(np.ceil(dim_count/2))
    else:
        pc_fit_count = dim_count
    for pc_idx in range(pc_fit_count):
        log.info('Initializing filter %d', pc_idx)
        if pc_idx < rec[pc_signal].shape[0]:
            r = rec[pc_signal].extract_channels([rec[pc_signal].chans[pc_idx]])
        else:
            log.info("Out of PCs, generating random signal")
            r = _random_resp_combos(rec['resp'], dim_count=1, whiten=True)

        m = np.nanmean(r.as_continuous())
        d = np.nanstd(r.as_continuous())
        rec['resp'] = r._modified_copy((r._data-m) / d)

        if flip_pcs:
            d = pc_idx * 2
        else:
            d = pc_idx

        tmodelspec = _extract_pop_channel(modelspec, d, fit_set_all)

        iwcg = find_module('weight_channels.gaussian', modelspec)
        iwc = find_module('weight_channels', modelspec, find_all_matches=True)
        relumod = find_module('relu', modelspec)
        stpidx = find_module('stp', tmodelspec)

        if iwcg is not None:
            n_outputs = tmodelspec.phi[iwcg]['mean'].shape[0]
            mean = np.arange(n_outputs+1)/(n_outputs*2+2) + 0.25
            tmodelspec.phi[iwcg]['mean'] = mean[1:]

        if relumod is not None:
            log.info('Temporarily converting relu to lvl')
            m_save = copy.deepcopy(tmodelspec[relumod])
            tmodelspec[relumod]['fn'] = 'nems0.modules.levelshift.levelshift'
            tmodelspec[relumod]['phi'] = {'level': np.array([[0]])}

            tmodelspec = init.prefit_LN(rec, tmodelspec,
                                        tolerance=tolerance, max_iter=700)
            m_save['phi']['offset'] = -tmodelspec[relumod]['phi']['level']
            tmodelspec[relumod] = m_save
        else:
            # import pdb; pdb.set_trace()
            tmodelspec = init.prefit_LN(rec, tmodelspec,
                                        tolerance=tolerance, max_iter=700)

        # now fit STP if it's in the model:
        if stpidx is not None:
            sp_kwargs = {'tolerance': tolerance, 'max_iter': 100}
            tmodelspec = analysis.fit_basic(rec, tmodelspec, fitter=scipy_minimize,
                                            fit_kwargs=sp_kwargs)

        """
        #tmodelspec = init.from_keywords(keyword_string=keywordstring,
        #                               meta={}, registry=keyword_lib, rec=rec)
        #if pc_idx > 0:
        #    # fix parameters that are shared across subspace filters
        #    # (ie, log compression)
        #    for tm, mm in zip(tmodelspec[:iwc], modelspec[:iwc]):
        #        for k, v in mm['phi'].items():
        #            log.info('fixing module %s key %s=%s', mm['fn'], k, v)
        #            tm['fn_kwargs'][k] = v
        #        del tm['phi']
        #        del tm['prior']

        #tmodelspec = init.prefit_LN(rec, tmodelspec,
        #                            tolerance=tolerance, max_iter=700)
        """

        # save subspace model back to main modelspec. If flipping PC dim fits,
        # save back to two channels in the main model, once flipped.
        if flip_pcs:
            d = pc_idx * 2
            modelspec = _update_pop_channel(tmodelspec, modelspec, d, fit_set_all)
            if (pc_idx * 2 < dim_count - 1):
                itfir = find_module('fir', tmodelspec)
                tmodelspec.phi[itfir]['coefficients'] *= -1
                if 'offset' in tmodelspec.phi[itfir+1].keys():
                    tmodelspec.phi[itfir+1]['offset'] *= -1
                elif 'level' in tmodelspec.phi[itfir+1].keys():
                    tmodelspec.phi[itfir + 1]['level'] *= -1

                d = pc_idx * 2 + 1
                modelspec = _update_pop_channel(tmodelspec, modelspec, d, fit_set_all)
        else:
            d = pc_idx
            modelspec = _update_pop_channel(tmodelspec, modelspec, d, fit_set_all)

        """
        # save results back into main modelspec
        #itfir=find_module('fir', tmodelspec)
        #itwc=find_module('weight_channels', tmodelspec)

        #if pc_idx==0:
        #    for tm, m in zip(tmodelspec[:(iwc+1)], modelspec[:(iwc+1)]):
        #        m['phi']=tm['phi'].copy()
        #    modelspec[ifir]['phi']=tmodelspec[itfir]['phi'].copy()
        #else:
        #    for k, v in tmodelspec[iwc]['phi'].items():
        #        modelspec[iwc]['phi'][k]=np.concatenate((modelspec[iwc]['phi'][k],v))
        #    for k, v in tmodelspec[itfir]['phi'].items():
        #        #if k=='coefficients':
        #        #    v/=100 # kludge
        #        modelspec[ifir]['phi'][k]=np.concatenate((modelspec[ifir]['phi'][k],v))

        #if flip_pcs and (pc_idx*2 < dim_count-1):
        #    # add negative flipped version of fit
        #    for k, v in tmodelspec[iwc]['phi'].items():
        #        modelspec[iwc]['phi'][k]=np.concatenate((modelspec[iwc]['phi'][k],v))
        #    for k, v in tmodelspec[itfir]['phi'].items():
        #        #if k=='coefficients':
        #        #    v/=100 # kludge
        #        modelspec[ifir]['phi'][k]=np.concatenate((-modelspec[ifir]['phi'][k],v))
        """

    # now fit weights for each neuron separately, using the initial subspace
    # slice_fitter = scipy_minimize
    sp_kwargs = {'tolerance': tolerance, 'max_iter': 20}

    # slice_fitter = coordinate_descent
    # cd_kwargs = {'tolerance': tolerance, 'max_iter': 20,
    #             'step_size': 0.1}
    if len(iwc) < 3:
        for s in range(respcount):
            log.info('First fit per cell slice %d', s)
            # TODO : don't fit static NL ? just weight channels... plus?
            modelspec = fit_population_slice(
                    est, modelspec, slice=s,
                    fit_set=fit_set_slice,
                    analysis_function=analysis.fit_basic,
                    metric=metrics.nmse,
                    fitter=scipy_minimize,
                    fit_kwargs=sp_kwargs)

    return {'modelspec': modelspec}


def _random_resp_combos(resp, dim_count=1, whiten=True):

    resp_count = resp.shape[0]
    log.info('Initializing %d x %d random weight matrix', dim_count, resp_count)
    weights = np.random.randn(dim_count, resp_count)
    d = resp.as_continuous().copy()
    if whiten:
        d -= np.mean(d, axis=1, keepdims=True)
        d /= np.std(d, axis=1, keepdims=True)
        d -= np.min(d, axis=1, keepdims=True)

    d_rand = np.matmul(weights, d)
    log.info('d shape: (%d, %d)', d.shape[0], d.shape[1])
    log.info('d_rand shape: (%d, %d)', d_rand.shape[0], d_rand.shape[1])
    rand_resp = resp._modified_copy(data=d_rand)
    rand_resp.chans = ['n'+str(i) for i in range(dim_count)]

    return rand_resp


def init_pop_rand(est, modelspec, IsReload=False, start_count=1,
                  pc_signal='rand_resp', whiten=True, **context):
    """
    initialize population model with random combinations of responses.
    generates random response combinations and passes them through to
    init_pop_pca()
    :param est: recording object with fit data
    :param modelspec: un-fit modelspec
    :param IsReload: don't fit if IsReload=True
    :param pc_signal: name of signal to generate with random combinations of responses
    :param context: dictionary of other context variables
    :return: initialized modelspec
    """
    if IsReload:
        return {}

    # guess at number of subspace dimensions
    fit_set_all, fit_set_slice = _figure_out_mod_split(modelspec)
    dim_count = modelspec[fit_set_slice[0]]['phi']['coefficients'].shape[1]

    mset = []
    E = np.ones(start_count)
    for i in range(start_count):
        log.info('Rand init: %d/%d', i, start_count)
        rec = est.copy()
        rec[pc_signal] = _random_resp_combos(
            rec['resp'], dim_count=dim_count, whiten=whiten)

        log.info('rec signal: %s (%d x %d)', pc_signal,
                 rec[pc_signal].shape[0], rec[pc_signal].shape[1])

        mset.append(init_pop_pca(rec, modelspec, pc_signal=pc_signal, **context))
        rec = mset[-1]['modelspec'].evaluate(rec)
        E[i] = metrics.nmse(rec)

    imax = np.nanargmin(E)
    for i in range(start_count):
        ss = "**" if (i == imax) else ""
        log.info('i=%d E=%.3e %s', i, E[i], ss)

    return mset[imax]


def _figure_out_mod_split(modelspec):
    """
    determine where to split modelspec for pop vs. slice fit
    :param modelspec:
    :return:
    """
    bank_mod = find_module('filter_bank', modelspec, find_all_matches=True)
    wc_mod = find_module('weight_channels', modelspec, find_all_matches=True)

    if len(wc_mod) >= 2:
        fit_set_all = list(range(wc_mod[1]))
        fit_set_slice = list(range(wc_mod[1], len(modelspec)))
    elif len(bank_mod) == 1:
        fit_set_all = list(range(bank_mod[0]))
        fit_set_slice = list(range(bank_mod[0], len(modelspec)))
    else:
        raise ValueError("Can't figure out how to split all and slices")

    return fit_set_all, fit_set_slice


def _invert_slice(rec, modelspec, fit_set_slice, population_channel=None):
    """
    Try to invert actual response into population subspace
    :param rec:
    :param modelspec:
    :return:
    """
    rec = rec.copy()

    # get actual response
    data = rec['resp']._data.copy()

    # go backwards through slice modules and apply inverse for each
    for s in reversed(fit_set_slice):
        m = modelspec[s]
        if 'levelshift' in m['fn']:
            data -= m['phi']['level']
        elif 'relu' in m['fn']:
            data += m['phi']['offset']
        elif 'weight_channels' in m['fn']:
            C = m['phi']['coefficients']
            # first way to invert slice - matrix inverse
            # Cinv = np.linalg.pinv(C)
            # data = np.matmul(Cinv, data)

            # second way, just inverse of the one channel
            dim_count = C.shape[1]
            data2 = np.zeros((dim_count, data.shape[1]))
            for ch in range(dim_count):
                Cinv = np.linalg.pinv(C[:, [ch]])
                data2[ch, :] = np.matmul(Cinv, data)
            data = data2

    # extract only the channel of interest
    if population_channel is not None:
        data = data[[population_channel], :]

    # return data to new "response" signal
    rec['resp'] = rec['resp']._modified_copy(data=data)
    return rec


def _extract_pop_channel(modelspec, d, fit_set_all, freeze_idx=[]):
    """
    extract mini model from modelspec, just for channel d
    over the modules indexed by fit_set_all
    :param modelspec:
    :param fit_set_all:
    :param d:
    :param freeze_idx: list of module indices to freeze (move phi to fn_kwargs)
    :return: tmodelspec - subspace model
    """
    # create modelspec with single population subspace filter
    dim_count = modelspec[fit_set_all[-1]+1]['phi']['coefficients'].shape[1]
    tmodelspec = ms.ModelSpec()
    for i in fit_set_all:
        m = copy.deepcopy(modelspec[i])
        print(m['fn'])
        for k, v in m['phi'].items():
            x = v.shape[0]
            if x >= dim_count:
                x1 = int(x/dim_count) * d
                x2 = int(x/dim_count) * (d+1)

                m['phi'][k] = v[x1:x2]
                if 'bounds' in m.keys():
                    m['bounds'][k] = (m['bounds'][k][0][x1:x2],
                                      m['bounds'][k][1][x1:x2])
                if 'bank_count' in m['fn_kwargs'].keys():
                    m['fn_kwargs']['bank_count'] = 1

            else:
                # single model-wide parameter, only fit for d==0
                if d == 0:
                    m['phi'][k] = v
                else:
                    m['fn_kwargs'][k] = v  # keep fixed for d>0
                    del m['phi']
                    del m['prior']
        tmodelspec.append(m)

    return tmodelspec


def _update_pop_channel(tmodelspec, modelspec, d, fit_set_all):
    """
    paste subspace model back into full model
    :param tmodelspec:
    :param modelspec:
    :param d:
    :param fit_set_all:
    :return:
    """
    dim_count = modelspec[fit_set_all[-1]+1]['phi']['coefficients'].shape[1]
    for i in fit_set_all:
        for k, v in tmodelspec[i]['phi'].items():
            x = modelspec[i]['phi'][k].shape[0]
            if x >= dim_count:
                x1 = int(x / dim_count) * d
                x2 = int(x / dim_count) * (d + 1)

                modelspec[i]['phi'][k][x1:x2] = v
            else:
                modelspec[i]['phi'][k] = v

    # print([modelspec.phi[f] for f in fit_set_all])
    return modelspec


def fit_population_channel(rec, modelspec,
                           fit_set_all, fit_set_slice,
                           analysis_function=analysis.fit_basic,
                           metric=metrics.nmse,
                           fitter=scipy_minimize, fit_kwargs={}):
    """
    DEPRECATED?
    Fit all the population channels, but only trying to predict the
    responses inverted through the weight_channels of layer 2
    :param rec:
    :param modelspec:
    :param fit_set_all:
    :param fit_set_slice:
    :param analysis_function:
    :param metric:
    :param fitter:
    :param fit_kwargs:
    :return:
    """
    # guess at number of subspace dimensions
    # dim_count = modelspec[fit_set_slice[0]]['phi']['coefficients'].shape[1]

    # invert cell-specific modules
    trec = _invert_slice(rec, modelspec, fit_set_slice)

    tmodelspec = ms.ModelSpec()

    for i in fit_set_all:
        m = modelspec[i].copy()
        tmodelspec.append(m)

    tmodelspec = analysis_function(trec, tmodelspec, fitter=fitter,
                                   metric=metric, fit_kwargs=fit_kwargs)

    for i in fit_set_all:
        for k, v in tmodelspec[i]['phi'].items():
            modelspec[i]['phi'][k] = v

    return modelspec


def fit_population_channel_fast2(rec, modelspec,
                                 fit_set_all, fit_set_slice,
                                 analysis_function=analysis.fit_basic,
                                 metric=metrics.nmse,
                                 fitter=scipy_minimize, fit_kwargs={}):

    # guess at number of subspace dimensions
    dim_count = modelspec[fit_set_slice[0]]['phi']['coefficients'].shape[1]
    wi = [i for i in fit_set_slice if 'weight_channels' in modelspec[i]['fn']]
    wi = wi[0]
    li = [i for i in fit_set_slice if 'levelshift' in modelspec[i]['fn']]
    li = li[0]

    for d in range(dim_count):
        # fit each dim separately
        log.info('Updating dim %d/%d', d+1, dim_count)

        # create modelspec with single population subspace filter
        tmodelspec = _extract_pop_channel(modelspec, d, fit_set_all)

        # temp append full-population layer as non-free parameters
        tmodelspec2 = copy.deepcopy(tmodelspec)
        for i in fit_set_slice:
            m = copy.deepcopy(modelspec[i])
            for k, v in m['phi'].items():
                # just applies to wc module?
                if v.shape[1] >= dim_count:
                    m['phi'][k] = v[:, [d]]
                else:
                    m['phi'][k] = v
            tmodelspec2.append(m)

        # compute residual from prediction by the rest of the pop model
        trec = rec.copy()
        trec = ms.evaluate(trec, modelspec)
        r = trec['resp'].as_continuous()
        p = trec['pred'].as_continuous().copy()
        respstd = np.nanstd(r)  # std of actual response

        trec = ms.evaluate(trec, tmodelspec2)
        p2 = trec['pred'].as_continuous()

        trec = ms.evaluate(trec, tmodelspec)

        # residual we're trying to predict with tmodelspec
        r = r - p + p2

        # calculate streamlined nMSE function for single pop channel model
        # by inverting neuron-specific gains and level shifts
        a = modelspec[wi]['phi']['coefficients'][:, [d]]
        b = modelspec[li]['phi']['level']

        r -= b  # subtract level shift from residual
        A1 = np.sum(a ** 2)
        A2 = np.sum(2 * a * r, axis=0, keepdims=True)
        A3 = np.sum(r**2, axis=0, keepdims=True)

        def my_nmse(result):
            '''
            hacked from nems0.metrics.mse.nmse. optimized nMSE for situation when a single
            population channel is predicting responses with fixed per-neuron gains and levelshifts
            A1, A2, A3, respstd defined outside of function
            result :  recording object updated by fitter, prediction response of single pop channel
            '''
            X1 = result['pred'].as_continuous()

            squared_errors = A1 * (X1**2) - A2 * X1 + A3
            mean_sq_err = np.sum(squared_errors) / (r.shape[0]*r.shape[1])

            mse = np.sqrt(mean_sq_err)
            return mse / respstd

        # import pdb
        # pdb.set_trace()

        tmodelspec = analysis_function(trec, tmodelspec, fitter=fitter,
                                       metric=my_nmse, fit_kwargs=fit_kwargs)

        modelspec = _update_pop_channel(tmodelspec, modelspec, d, fit_set_all)

    return modelspec


def fit_population_channel_fast(rec, modelspec,
                                fit_set_all, fit_set_slice,
                                analysis_function=analysis.fit_basic,
                                metric=metrics.nmse,
                                fitter=scipy_minimize, fit_kwargs={}):

    # guess at number of subspace dimensions
    dim_count = modelspec[fit_set_slice[0]]['phi']['coefficients'].shape[1]

    for d in range(dim_count):
        # fit each dim separately
        log.info('Updating dim %d/%d', d+1, dim_count)

        # create modelspec with single population subspace filter
        tmodelspec = ms.ModelSpec()
        for i in fit_set_all:
            m = copy.deepcopy(modelspec[i])
            for k, v in m['phi'].items():
                x = v.shape[0]
                if x >= dim_count:
                    x1 = int(x/dim_count) * d
                    x2 = int(x/dim_count) * (d+1)

                    m['phi'][k] = v[x1:x2]
                    if 'bank_count' in m['fn_kwargs'].keys():
                        m['fn_kwargs']['bank_count'] = 1
                else:
                    # single model-wide parameter, only fit for d==0
                    if d == 0:
                        m['phi'][k] = v
                    else:
                        m['fn_kwargs'][k] = v  # keep fixed for d>0
                        del m['phi']
                        del m['prior']
            tmodelspec.append(m)

        # append full-population layer as non-free parameters
        for i in fit_set_slice:
            m = copy.deepcopy(modelspec[i])
            for k, v in m['phi'].items():
                # just applies to wc module?
                if v.shape[1] >= dim_count:
                    m['fn_kwargs'][k] = v[:, [d]]
                else:
                    m['fn_kwargs'][k] = v
                # print(k)
                # print(m['fn_kwargs'][k])
            del m['phi']
            del m['prior']
            tmodelspec.append(m)
            # print(tmodelspec[-1]['fn_kwargs'])

        # compute residual from prediction by the rest of the pop model
        trec = rec.copy()
        trec = ms.evaluate(trec, modelspec)
        r = trec['resp'].as_continuous()
        p = trec['pred'].as_continuous()

        trec = ms.evaluate(trec, tmodelspec)
        p2 = trec['pred'].as_continuous()
        trec['resp'] = trec['resp']._modified_copy(data=r-p+p2)

        # import pdb;
        # pdb.set_trace()
        tmodelspec = analysis_function(trec, tmodelspec, fitter=fitter,
                                       metric=metric, fit_kwargs=fit_kwargs)

        for i in fit_set_all:
            for k, v in tmodelspec[i]['phi'].items():
                x = modelspec[i]['phi'][k].shape[0]
                if x >= dim_count:
                    x1 = int(x/dim_count) * d
                    x2 = int(x/dim_count) * (d+1)

                    modelspec[i]['phi'][k][x1:x2] = v
                else:
                    modelspec[i]['phi'][k] = v

        # for i in fit_set_slice:
        #    for k, v in tmodelspec[i]['phi'].items():
        #        if modelspec[i]['phi'][k].shape[0] >= dim_count:
        #            modelspec[i]['phi'][k][d,:] = v

        # print([modelspec.phi[f] for f in fit_set_all])

    return modelspec


def fit_population_channel_fast_old(rec, modelspec,
                                    fit_set_all, fit_set_slice,
                                    analysis_function=analysis.fit_basic,
                                    metric=metrics.nmse,
                                    fitter=scipy_minimize, fit_kwargs={}):

    # guess at number of subspace dimensions
    dim_count = modelspec[fit_set_slice[0]]['phi']['coefficients'].shape[1]

    for d in range(dim_count):
        # fit each dim separately
        log.info('Updating dim %d/%d', d+1, dim_count)

        # invert cell-specific modules
        trec = _invert_slice(rec, modelspec, fit_set_slice,
                             population_channel=d)

        tmodelspec = ms.ModelSpec()
        for i in fit_set_all:
            m = copy.deepcopy(modelspec[i])
            for k, v in m['phi'].items():
                x = v.shape[0]
                if x >= dim_count:
                    x1 = int(x / dim_count) * d
                    x2 = int(x / dim_count) * (d + 1)

                    m['phi'][k] = v[x1:x2]
                    if 'bank_count' in m['fn_kwargs'].keys():
                        m['fn_kwargs']['bank_count'] = 1
                else:
                    # single model-wide parameter, only fit for d == 0
                    if d == 0:
                        m['phi'][k] = v
                    else:
                        m['fn_kwargs'][k] = v  # keep fixed for d>0

            tmodelspec.append(m)

        tmodelspec = analysis_function(trec, tmodelspec, fitter=fitter,
                                       metric=metric, fit_kwargs=fit_kwargs)

        for i in fit_set_all:
            for k, v in tmodelspec[i]['phi'].items():
                x = modelspec[i]['phi'][k].shape[0]
                if x >= dim_count:
                    x1 = int(x/dim_count) * d
                    x2 = int(x/dim_count) * (d+1)

                    modelspec[i]['phi'][k][x1:x2] = v
                else:
                    modelspec[i]['phi'][k] = v
        # print([modelspec.phi[f] for f in fit_set_all])

    return modelspec


def fit_population_slice(rec, modelspec, slice=0, fit_set=None,
                         analysis_function=analysis.fit_basic,
                         metric=metrics.nmse,
                         fitter=scipy_minimize, fit_kwargs={}):

    """
    fits a slice of a population model. modified from prefit_mod_subset

    slice: int
        response channel to fit
    fit_set: list
        list of mod names to fit

    """

    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    slice_count = rec['resp'].shape[0]
    if slice > slice_count:
        raise ValueError("Slice %d > slice_count %d", slice, slice_count)

    if fit_set is None:
        raise ValueError("fit_set list of module indices must be specified")

    if type(fit_set[0]) is int:
        fit_idx = fit_set
    else:
        fit_idx = []
        for i, m in enumerate(modelspec):
            for fn in fit_set:
                if fn in m['fn']:
                    fit_idx.append(i)

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    tmodelspec = ms.ModelSpec()
    sliceinfo = []
    for i, m in enumerate(modelspec):
        m = copy.deepcopy(m)
        # need to have phi in place
        if not m.get('phi'):
            log.info('Initializing phi for module %d (%s)', i, m['fn'])
            m = priors.set_mean_phi([m])[0]  # Inits phi

        if i in fit_idx:
            s = {}
            for key, value in m['phi'].items():
                log.debug('Slicing %d (%s) key %s chan %d for fit',
                          i, m['fn'], key, slice)

                # keep only sliced channel(s)
                if 'bank_count' in m['fn_kwargs'].keys():
                    bank_count = m['fn_kwargs']['bank_count']
                    filters_per_bank = int(value.shape[0] / bank_count)
                    slices = np.arange(slice*filters_per_bank,
                                       (slice+1)*filters_per_bank)
                    m['phi'][key] = value[slices, ...]
                    s[key] = slices
                    m['fn_kwargs']['bank_count'] = 1
                elif value.shape[0] == slice_count:
                    m['phi'][key] = value[[slice], ...]
                    s[key] = [slice]
                else:
                    raise ValueError("Not sure how to slice %s %s",
                                     m['fn'], key)

            # record info about how sliced this module parameter
            sliceinfo.append(s)

        tmodelspec.append(m)

    if len(fit_idx) == 0:
        log.info('No modules matching fit_set for slice fit')
        return modelspec

    exclude_idx = np.setdiff1d(np.arange(0, len(modelspec)),
                               np.array(fit_idx)).tolist()
    for i in exclude_idx:
        m = tmodelspec[i]
        log.debug('Freezing phi for module %d (%s)', i, m['fn'])

        m['fn_kwargs'].update(m['phi'])
        m['phi'] = {}
        tmodelspec[i] = m

    # generate temp recording with only resposnes of interest
    temp_rec = rec.copy()
    slice_chans = [temp_rec['resp'].chans[slice]]
    temp_rec['resp'] = temp_rec['resp'].extract_channels(slice_chans)

    # remove initial modules
    first_idx = fit_idx[0]
    if first_idx > 0:
        # print('firstidx {}'.format(first_idx))
        temp_rec = ms.evaluate(temp_rec, tmodelspec, stop=first_idx)
        # temp_rec['stim'] = temp_rec['pred'].copy()
        # tmodelspec = tmodelspec.copy(lb=first_idx)
        # tmodelspec[0]['fn_kwargs']['i'] = 'stim'
        tmodelspec = tmodelspec.copy()
        tmodelspec.fast_eval_on(rec=temp_rec, subset=fit_idx)
        first_idx = 0
        # print(tmodelspec)
    # print(temp_rec.signals.keys())

    # IS this mask necessary? Does it work?
    # if 'mask' in temp_rec.signals.keys():
    #    print("Data len pre-mask: %d" % (temp_rec['mask'].shape[1]))
    #    temp_rec = temp_rec.apply_mask()
    #    print("Data len post-mask: %d" % (temp_rec['mask'].shape[1]))

    # fit the subset of modules
    temp_rec = ms.evaluate(temp_rec, tmodelspec)
    error_before = metric(temp_rec)
    tmodelspec = analysis_function(temp_rec, tmodelspec, fitter=fitter,
                                   metric=metric, fit_kwargs=fit_kwargs)
    tmodelspec.fast_eval_off()
    temp_rec = ms.evaluate(temp_rec, tmodelspec)
    error_after = metric(temp_rec)
    dError = error_before - error_after
    if dError < 0:
        log.info("dError (%.6f - %.6f) = %.6f worse. not updating modelspec"
                 % (error_before, error_after, dError))
    else:
        log.info("dError (%.6f - %.6f) = %.6f better. updating modelspec"
                 % (error_before, error_after, dError))

        # reassemble the full modelspec with updated phi values from tmodelspec
        for i, mod_idx in enumerate(fit_idx):
            m = copy.deepcopy(modelspec[mod_idx])
            # need to have phi in place
            if not m.get('phi'):
                log.info('Intializing phi for module %d (%s)', mod_idx, m['fn'])
                m = priors.set_mean_phi([m])[0]  # Inits phi
            for key, value in tmodelspec[mod_idx - first_idx]['phi'].items():
                # print(key)
                # print(m['phi'][key].shape)
                # print(sliceinfo[i][key])
                # print(value)
                m['phi'][key][sliceinfo[i][key], :] = value

            modelspec[mod_idx] = m

    return modelspec


def fit_population_iteratively(
        est, modelspec,
        cost_function=basic_cost,
        fitter=coordinate_descent, evaluator=ms.evaluate,
        segmentor=nems0.segmentors.use_all_data,
        mapper=nems0.fitters.mappers.simple_vector,
        metric=lambda data: nems0.metrics.api.nmse(data, 'pred', 'resp'),
        metaname='fit_basic', fit_kwargs={},
        module_sets=None, invert=False, tolerances=None, tol_iter=50,
        fit_iter=10, IsReload=False, **context
        ):
    '''
    Required Arguments:
     est          A recording object
     modelspec     A modelspec object

    Optional Arguments:
    TODO: need to deal with the fact that you can't pass functions in an xforms-frieldly fucntion
     fitter        (CURRENTLY NOT USED?)
                   A function of (sigma, costfn) that tests various points,
                   in fitspace (i.e. sigmas) using the cost function costfn,
                   and hopefully returns a better sigma after some time.
     mapper        (CURRENTLY NOT USED?)
                   A class that has two methods, pack and unpack, which define
                   the mapping between modelspecs and a fitter's fitspace.
     segmentor     (CURRENTLY NOT USED?)
                   An function that selects a subset of the data during the
                   fitting process. This is NOT the same as est/val data splits
     metric        A function of a Recording that returns an error value
                   that is to be minimized.

     module_sets   (CURRENTLY NOT USED?)
                   A nested list specifying which model indices should be fit.
                   Overall iteration will occurr len(module_sets) many times.
                   ex: [[0], [1, 3], [0, 1, 2, 3]]

     invert        (CURRENTLY NOT USED?)
                   Boolean. Causes module_sets to specify the model indices
                   that should *not* be fit.


    Returns
    A list containing a single modelspec, which has the best parameters found
    by this fitter.
    '''

    if IsReload:
        return {}

    modelspec = copy.deepcopy(modelspec)
    data = est.copy()

    fit_set_all, fit_set_slice = _figure_out_mod_split(modelspec)

    if tolerances is None:
        tolerances = [1e-4, 1e-5]

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    # then delete the mask signal so that it's not reapplied on each fit
    if 'mask' in data.signals.keys():
        log.info("Data len pre-mask: %d", data['mask'].shape[1])
        data = data.apply_mask()
        log.info("Data len post-mask: %d", data['mask'].shape[1])
        del data.signals['mask']

    start_time = time.time()
    ms.fit_mode_on(modelspec, data)

    # modelspec = init_pop_pca(data, modelspec)
    # print(modelspec)

    # Ensure that phi exists for all modules; choose prior mean if not found
    # for i, m in enumerate(modelspec):
    #    if ('phi' not in m.keys()) and ('prior' in m.keys()):
    #        m = nems0.priors.set_mean_phi([m])[0]  # Inits phi for 1 module
    #        log.debug('Phi not found for module, using mean of prior: {}'
    #                  .format(m))
    #        modelspec[i] = m

    error = np.inf

    slice_count = data['resp'].shape[0]
    step_size = 0.1
    if 'nonlinearity' in modelspec[-1]['fn']:
        skip_nl_first = True
        tolerances = [tolerances[0]] + tolerances
    else:
        skip_nl_first = False

    for toli, tol in enumerate(tolerances):

        log.info("Fitting subsets with tol: %.2E fit_iter %d tol_iter %d",
                 tol, fit_iter, tol_iter)
        cd_kwargs = fit_kwargs.copy()
        cd_kwargs.update({'tolerance': tol, 'max_iter': fit_iter,
                          'step_size': step_size})
        sp_kwargs = fit_kwargs.copy()
        sp_kwargs.update({'tolerance': tol, 'max_iter': fit_iter})

        if (toli == 0) and skip_nl_first:
            log.info('skipping nl on first tolerance loop')
            saved_modelspec = copy.deepcopy(modelspec)
            saved_fit_set_slice = fit_set_slice.copy()
            # import pdb;
            # pdb.set_trace()
            modelspec.pop_module()
            fit_set_slice = fit_set_slice[:-1]

        inner_i = 0
        error_reduction = np.inf
        # big_slice = 0
        # big_n = data['resp'].ntimes
        # big_step = int(big_n/10)
        # big_slice_size = int(big_n/2)
        while (error_reduction >= tol) and (inner_i < tol_iter):

            log.info("(%d) Tol %.2e: Loop %d/%d (max)",
                     toli, tol, inner_i, tol_iter)
            improved_modelspec = copy.deepcopy(modelspec)
            cc = 0
            slist = list(range(slice_count))
            # random.shuffle(slist)

            for i, m in enumerate(modelspec):
                if i in fit_set_all:
                    log.info(m['fn'] + ": fitting")
                else:
                    log.info(m['fn'] + ": frozen")

            # partially implemented: select temporal subset of data for fitting
            # on current loop.
            # data2 = data.copy()
            # big_slice += 1
            # sl = np.zeros(big_n, dtype=bool)
            # sl[:big_slice_size]=True
            # sl = np.roll(sl, big_step*big_slice)
            # log.info('Sampling temporal subset %d (size=%d/%d)', big_step, big_slice_size, big_n)
            # for s in data2.signals.values():
            #    e = s._modified_copy(s._data[:,sl])
            #    data2[e.name] = e

            # improved_modelspec = init.prefit_mod_subset(
            #        data, improved_modelspec, analysis.fit_basic,
            #        metric=metric,
            #        fit_set=fit_set_all,
            #        fit_kwargs=sp_kwargs)
            improved_modelspec = fit_population_channel_fast2(
                data, improved_modelspec, fit_set_all, fit_set_slice,
                analysis_function=analysis.fit_basic,
                metric=metric,
                fitter=scipy_minimize, fit_kwargs=sp_kwargs)

            for s in slist:
                log.info('Slice %d set %s' % (s, fit_set_slice))
                improved_modelspec = fit_population_slice(
                        data, improved_modelspec, slice=s,
                        fit_set=fit_set_slice,
                        analysis_function=analysis.fit_basic,
                        metric=metric,
                        fitter=scipy_minimize,
                        fit_kwargs=sp_kwargs)
                # fitter = coordinate_descent,
                # fit_kwargs = cd_kwargs)

                cc += 1
                # if (cc % 8 == 0) or (cc == slice_count):

            data = ms.evaluate(data, improved_modelspec)
            new_error = metric(data)
            error_reduction = error - new_error
            error = new_error
            log.info("tol=%.2E, iter=%d/%d: deltaE=%.6E",
                     tol, inner_i, tol_iter, error_reduction)
            inner_i += 1
            if error_reduction > 0:
                modelspec = improved_modelspec

        log.info("Done with tol %.2E (i=%d, max_error_reduction %.7f)",
                 tol, inner_i, error_reduction)

        if (toli == 0) and skip_nl_first:
            log.info('Restoring NL module after first tol loop')
            modelspec.append(saved_modelspec[-1])

            fit_set_slice = saved_fit_set_slice
            if 'double_exponential' in saved_modelspec[-1]['fn']:
                modelspec = init.init_dexp(data, modelspec)
            elif 'logistic_sigmoid' in saved_modelspec[-1]['fn']:
                modelspec = init.init_logsig(data, modelspec)
            elif 'relu' in saved_modelspec[-1]['fn']:
                # just keep initialized to zero
                pass
            else:
                raise ValueError("Output NL %s not supported",
                                 saved_modelspec[-1]['fn'])
            # just fit the NL
            improved_modelspec = copy.deepcopy(modelspec)

            kwa = cd_kwargs.copy()
            kwa['max_iter'] *= 2
            for s in range(slice_count):
                log.info('Slice %d set %s' % (s, [fit_set_slice[-1]]))
                improved_modelspec = fit_population_slice(
                        data, improved_modelspec, slice=s,
                        fit_set=fit_set_slice,
                        analysis_function=analysis.fit_basic,
                        metric=metric,
                        fitter=scipy_minimize,
                        fit_kwargs=sp_kwargs)
                # fitter = coordinate_descent,
                # fit_kwargs = cd_kwargs)
            data = ms.evaluate(data, modelspec)
            old_error = metric(data)
            data = ms.evaluate(data, improved_modelspec)
            new_error = metric(data)
            log.info('Init NL fit error change %.5f-%.5f = %.5f',
                     old_error, new_error, old_error-new_error)
            modelspec = improved_modelspec

        else:
            step_size *= 0.25

    elapsed_time = (time.time() - start_time)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.fit_mode_off(improved_modelspec)
    ms.set_modelspec_metadata(improved_modelspec, 'fitter', metaname)
    ms.set_modelspec_metadata(improved_modelspec, 'fit_time', elapsed_time)

    return {'modelspec': improved_modelspec.copy()}
