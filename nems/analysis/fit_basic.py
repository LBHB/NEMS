import copy
import logging
import time
from functools import partial
import numpy as np

from nems0.analysis.cost_functions import basic_cost
from nems0.fitters.api import scipy_minimize
import nems0.priors
import nems0.fitters.mappers
import nems0.modelspec as ms
import nems0.metrics.api as metrics
import nems0.segmentors
import nems0.utils

log = logging.getLogger(__name__)


def fit_basic(data, modelspec,
              fitter=scipy_minimize, cost_function=None,
              segmentor=nems0.segmentors.use_all_data,
              mapper=nems0.fitters.mappers.simple_vector,
              metric=None,
              metaname='fit_basic', fit_kwargs={}, require_phi=True):
    '''
    Required Arguments:
     data          A recording object
     modelspec     A modelspec object

    Optional Arguments:
     fitter        A function of (sigma, costfn) that tests various points,
                   in fitspace (i.e. sigmas) using the cost function costfn,
                   and hopefully returns a better sigma after some time.
     mapper        A class that has two methods, pack and unpack, which define
                   the mapping between modelspecs and a fitter's fitspace.
     segmentor     An function that selects a subset of the data during the
                   fitting process. This is NOT the same as est/val data splits
     metric        A function of a Recording that returns an error value
                   that is to be minimized.

    Returns
    A list containing a single modelspec, which has the best parameters found
    by this fitter.
    '''
    start_time = time.time()

    modelspec = copy.deepcopy(modelspec)
    output_name = modelspec.meta.get('output_name', 'resp')

    if metric is None:
        metric = lambda data: metrics.nmse(data, 'pred', output_name)

    if cost_function is None:
        # Use the cost function defined in this module by default
        cost_function = basic_cost

    if require_phi:
        # Ensure that phi exists for all modules;
        # choose prior mean if not found
        for i, m in enumerate(modelspec.modules):
            if ('phi' not in m.keys()) and ('prior' in m.keys()):
                log.debug('Phi not found for module, using mean of prior: %s', m)
                m = nems0.priors.set_mean_phi([m])[0]  # Inits phi for 1 module
                modelspec[i] = m

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    if 'mask' in data.signals.keys():
        log.info("Data len pre-mask: %d", data['mask'].shape[1])
        data = data.apply_mask()
        log.info("Data len post-mask: %d", data['mask'].shape[1])

    # turn on "fit mode". currently this serves one purpose, for normalization
    # parameters to be re-fit for the output of each module that uses
    # normalization. does nothing if normalization is not being used.
    ms.fit_mode_on(modelspec, data)

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
                      data=data, segmentor=segmentor, evaluator=evaluator,
                      metric=metric)

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
    ms.set_modelspec_metadata(improved_modelspec, 'fitter', metaname)
    ms.set_modelspec_metadata(improved_modelspec, 'n_parms',
                              len(improved_sigma))
    if modelspec.fit_count == 1:
        improved_modelspec.meta['fit_time'] = elapsed_time
        improved_modelspec.meta['loss'] = final_err
    else:
        fit_index = modelspec.fit_index
        if fit_index == 0:
            improved_modelspec.meta['fit_time'] = np.zeros(improved_modelspec.fit_count)
            improved_modelspec.meta['loss'] = np.zeros(improved_modelspec.fit_count)
        improved_modelspec.meta['fit_time'][fit_index] = elapsed_time
        improved_modelspec.meta['loss'][fit_index] = final_err

    if type(improved_modelspec) is list:
        return [copy.deepcopy(improved_modelspec)]
    else:
        return improved_modelspec.copy()


def fit_random_subsets(data, modelspec, nsplits=1, rebuild_every=10000):
    """
    Randomly picks a small fraction of the data to fit on.
    Intended to speed up initial converge on fitting large data sets.
    To improve efficiency, you may generally good to use the same subset
    for a bunch of cost function evaluations in a row.
    """
    maker = nems0.segmentors.random_jackknife_maker
    segmentor = maker(nsplits=nsplits, rebuild_every=rebuild_every,
                      invert=True, excise=True)
    return fit_basic(data, modelspec,
                     segmentor=segmentor,
                     metaname='fit_random_subsets')


def fit_state_nfold(data_list, modelspecs, generate_psth=False,
                    fitter=scipy_minimize, metric=None,
                    fit_kwargs={}):
    '''
    Generic state-dependent-stream model fitter
    Takes njacks jackknifes, where each jackknife has some small
    fraction of data NaN'd out, and fits modelspec to them.

    DEPRECATED? REPLACED BY STANDARD nfold?
    '''
    nfolds = len(data_list)

    models = []
    if not metric:
        def metric(d):
            metrics.nmse(d, 'pred', 'resp')

    for i in range(nfolds):
        log.info("Fitting fold {}/{}".format(i+1, nfolds))
        tms = nems0.initializers.prefit_to_target(
                data_list[i], copy.deepcopy(modelspecs[0]),
                nems0.analysis.api.fit_basic, 'merge_channels',
                fitter=scipy_minimize,
                fit_kwargs={'options': {'tolerance': 1e-4, 'max_iter': 500}})

        models += fit_basic(data_list[i], tms,
                            fitter=fitter,
                            metric=metric,
                            metaname='fit_nfold',
                            fit_kwargs=fit_kwargs)

    return models


def fit_jackknifes(data, modelspec, njacks=10):
    '''
    Takes njacks jackknifes, where each jackknife has some small
    fraction of data NaN'd out, and fits modelspec to them.

    TODO : check if deprecated, replaced by fit_nfold?
    '''
    models = []
    for i in range(njacks):
        log.info("Fitting jackknife {}/{}".format(i+1, njacks))
        jk = data.jackknife_by_time(njacks, i)
        models += fit_basic(jk, modelspec, fitter=scipy_minimize,
                            metaname='fit_jackknifes')

    return models


def fit_subsets(data, modelspec, nsplits=10):
    '''
    Divides the data evenly into nsplits pieces, and fits a model
    to each of the pieces.

    TODO : Test, add more parameters
    '''
    models = []
    for i in range(nsplits):
        # TODO: Minor glitch: when fitting, print output from fitter
        #       comes back *after* log from next iteration
        #       (i.e. "fitting 1/3"
        #             "fitting 2/3"
        #             "final error <for 1/3>: 0.111")
        log.info("Fitting subset {}/{}".format(i+1, nsplits))
        split = data.jackknife_by_time(nsplits, i, invert=True, excise=True)
        models += fit_basic(split, modelspec, fitter=scipy_minimize,
                            metaname='fit_subset')

    return models
