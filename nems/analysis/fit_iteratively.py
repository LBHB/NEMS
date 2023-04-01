from functools import partial
import time
import logging
import copy

import numpy as np

from nems0.fitters.api import coordinate_descent
from nems0.analysis.cost_functions import basic_cost
import nems0.fitters.mappers
import nems0.metrics.api
import nems0.modelspec as ms

log = logging.getLogger(__name__)


def fit_module_sets(
        data, modelspec,
        cost_function=basic_cost, evaluator=ms.evaluate,
        segmentor=nems0.segmentors.use_all_data,
        mapper=nems0.fitters.mappers.simple_vector,
        metric=lambda data: nems0.metrics.api.nmse(data, 'pred', 'resp'),
        fitter=coordinate_descent, fit_kwargs={}, metaname='fit_module_sets',
        module_sets=None, invert=False, tolerance=1e-4, max_iter=1000
        ):
    '''
    Required Arguments:
     data          A recording object
     modelspec     A modelspec object

    Optional Arguments:
     fitter        A function of (sigma, costfn) that tests various points,
                   in fitspace (i.e. sigmas) using the cost function costfn,
                   and hopefully returns a better sigma after some time.

     module_sets   A nested list specifying which model indices should be fit.
                   Overall iteration will occurr len(module_sets) many times.
                   ex: [[0], [1, 3], [0, 1, 2, 3]]

     invert        Boolean. Causes module_sets to specify the model indices
                   that should *not* be fit.


    Returns
    A list containing a single modelspec, which has the best parameters found
    by this fitter.
    '''
    if module_sets is None:
        module_sets = [[i] for i in range(len(modelspec))]
    fit_kwargs.update({'tolerance': tolerance, 'max_iter': max_iter})

    # Ensure that phi exists for all modules; choose prior mean if not found
    for i, m in enumerate(modelspec):
        if ('phi' not in m.keys()) and ('prior' in m.keys()):
            m = nems0.priors.set_mean_phi([m])[0]  # Inits phi for 1 module
            log.debug('Phi not found for module, using mean of prior: {}'
                      .format(m))
            modelspec[i] = m

    if invert:
        module_sets = _invert_subsets(modelspec, module_sets)

    ms.fit_mode_on(modelspec, data)
    start_time = time.time()

    log.info("Fitting all subsets with tolerance: %.2E", tolerance)
    for subset in module_sets:
        improved_modelspec = _module_set_loop(
                subset, data, modelspec, cost_function, fitter,
                mapper, segmentor, evaluator, metric, fit_kwargs
                )

    elapsed_time = (time.time() - start_time)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.fit_mode_off(improved_modelspec)
    ms.set_modelspec_metadata(improved_modelspec, 'fitter', metaname)
    ms.set_modelspec_metadata(improved_modelspec, 'fit_time', elapsed_time)
    results = [copy.deepcopy(improved_modelspec)]

    return results


def _invert_subsets(modelspec, module_sets):
    inverted = []
    for subset in module_sets:
        subset_inverted = [
                None if i in subset else i
                for i, in enumerate(modelspec)
                ]
        inverted.append([i for i in subset_inverted if i is not None])
        log.debug("Inverted subset: %s\n", subset)

    return inverted


def _module_set_loop(subset, data, modelspec, cost_function, fitter,
                     mapper, segmentor, evaluator, metric, fit_kwargs):
    mods = [m['fn'] for i, m in enumerate(modelspec) if i in subset]
    log.info("Fitting subset(s): %s %s", subset, mods)
    ms.fit_mode_on(modelspec, data, subset)

    packer, unpacker, pack_bounds = mapper(modelspec, subset=subset)

    # cost_function.counter = 0
    cost_function.error = None
    cost_fn = partial(cost_function,
                      unpacker=unpacker, modelspec=modelspec,
                      data=data, segmentor=segmentor, evaluator=evaluator,
                      metric=metric)
    sigma = packer(modelspec)
    bounds = pack_bounds(modelspec)

    improved_sigma = fitter(sigma, cost_fn, bounds=bounds, **fit_kwargs)
    improved_modelspec = unpacker(improved_sigma)

    return improved_modelspec


def fit_iteratively(
        data, modelspec,
        cost_function=basic_cost,
        fitter=coordinate_descent, evaluator=ms.evaluate,
        segmentor=nems0.segmentors.use_all_data,
        mapper=nems0.fitters.mappers.simple_vector,
        metric=lambda data: nems0.metrics.api.nmse(data, 'pred', 'resp'),
        metaname='fit_basic', fit_kwargs={},
        module_sets=None, invert=False, tolerances=None, tol_iter=50,
        fit_iter=10,
        ):
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

     module_sets   A nested list specifying which model indices should be fit.
                   Overall iteration will occurr len(module_sets) many times.
                   ex: [[0], [1, 3], [0, 1, 2, 3]]

     invert        Boolean. Causes module_sets to specify the model indices
                   that should *not* be fit.


    Returns
    A list containing a single modelspec, which has the best parameters found
    by this fitter.
    '''
    if module_sets is None:
        module_sets = []
        for i, m in enumerate(modelspec):
            if 'prior' in m.keys():
                if 'levelshift' in m['fn'] and 'fir' in modelspec[i-1]['fn']:
                    # group levelshift with preceding fir filter by default
                    module_sets[-1].append(i)
                elif m.get('phi', {}):
                    # otherwise just fit each module separately
                    module_sets.append([i])
        log.info('Fit sets: %s', module_sets)

    elif invert:
        # use module_sets to specify modules NOT to fit. Not sure if anyone uses this
        exclude_mods = []
        for i in module_sets:
            exclude_mods.extend(i)
        module_sets = [[i] for i in range(len(modelspec)) if i not in exclude_mods]

    if tolerances is None:
        tolerances = [1e-6]

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    if 'mask' in data.signals.keys():
        log.info("Data len pre-mask: %d", data['mask'].shape[1])
        data = data.apply_mask()
        log.info("Data len post-mask: %d", data['mask'].shape[1])

    start_time = time.time()
    # Ensure that phi exists for all modules; choose prior mean if not found
    for i, m in enumerate(modelspec):
        if ('phi' not in m.keys()) and ('prior' in m.keys()):
            m = nems0.priors.set_mean_phi([m])[0]  # Inits phi for 1 module
            log.debug('Phi not found for module, using mean of prior: {}'
                      .format(m))
            modelspec[i] = m

    error = np.inf
    for tol in tolerances:
        log.info("\nFitting subsets with tol: %.2E fit_iter %d tol_iter %d",
                 tol, fit_iter, tol_iter)
        fit_kwargs.update({'tolerance': tol, 'max_iter': fit_iter})
        max_error_reduction = np.inf
        i = 0

        while (max_error_reduction >= tol) and (i < tol_iter):
            max_error_reduction = 0
            j = 0
            for subset in module_sets:
                improved_modelspec = _module_set_loop(
                        subset, data, modelspec, cost_function, fitter,
                        mapper, segmentor, evaluator, metric, fit_kwargs
                        )
                new_error = cost_function.error
                error_reduction = error-new_error
                error = new_error
                j += 1
                if error_reduction > max_error_reduction:
                    max_error_reduction = error_reduction
            log.info("tol=%.2E, iter=%d/%d: max deltaE=%.6E",
                     tol, i, tol_iter, max_error_reduction)
            i += 1
        log.info("Done with tol %.2E (i=%d, max_error_reduction %.7f)",
                 tol, i, error_reduction)

    elapsed_time = (time.time() - start_time)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.fit_mode_off(improved_modelspec)
    ms.set_modelspec_metadata(improved_modelspec, 'fitter', metaname)
    ms.set_modelspec_metadata(improved_modelspec, 'fit_time', elapsed_time)
    results = copy.deepcopy(improved_modelspec)

    return results
