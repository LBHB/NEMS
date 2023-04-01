import logging
import inspect

import nems0.utils

log = logging.getLogger(__name__)


def basic_cost(sigma, unpacker, modelspec, data, segmentor,
               evaluator, metric, display_N=100):
    '''Standard cost function for use by fit_basic and other analyses.'''
    updated_spec = unpacker(sigma)
    # The segmentor takes a subset of the data for fitting each step
    # Intended use is for CV or random selection of chunks of the data
    # For fit_basic the 'segmentor' just passes it all through.
    data_subset = segmentor(data)
    updated_data_subset = evaluator(data_subset, updated_spec)
    error = metric(updated_data_subset)
    #log.debug("inside cost function, current error: %.06f", error)
    #log.debug("current sigma: %s", sigma)

    if hasattr(basic_cost, 'counter'):
        basic_cost.counter += 1
        if basic_cost.counter % display_N == 0:
            a=inspect.getfullargspec(metric)
            if 'verbose' in a.args:
                e2 = metric(updated_data_subset, verbose=True)
                
            log.info('Eval #%d. E=%.06f', basic_cost.counter, error)
            # log.debug("current sigma: %s", sigma)
            nems0.utils.progress_fun()

    if hasattr(basic_cost, 'error'):
        basic_cost.error = error

    return error


def basic_with_copy(sigma, unpacker, modelspec, data, segmentor,
                    evaluator, metric, copy_phi=None):
    '''Same as basic_cost, but allows copying of parameters between modules.

    Note: This cost_function will run slower than basic_cost, so it should only
    be used if the copy_phi keyword argument is actually needed. Otherwise,
    always use basic_cost.

    Parameters:
    -----------
    copy_phi : list of 2-tuples of ints
        For each 2-tuple, the first entry specifies the index of the module
        to copy from, and the second entry is the module to copy to.
        Ex: copy_phi=[(1, 3)] would copy phi from the module at idx=1
            to the fn_kwargs of the module at idx=3

    '''
    updated_spec = unpacker(sigma)

    if copy_phi is not None:
        for t in copy_phi:
            m = updated_spec.get_module(mod_index=t[0])
            p = m['phi'].copy()
            updated_spec[t[1]]['fn_kwargs'].update(p)

    data_subset = segmentor(data)
    updated_data_subset = evaluator(data_subset, updated_spec)
    error = metric(updated_data_subset)
    log.debug("inside cost function, current error: %.06f", error)
    log.debug("current sigma: %s", sigma)

    if hasattr(basic_cost, 'counter'):
        basic_cost.counter += 1
        if basic_cost.counter % 100 == 0:
            log.info('Eval #%d. E=%.06f', basic_cost.counter, error)
            # log.debug("current sigma: %s", sigma)
            nems0.utils.progress_fun()

    if hasattr(basic_cost, 'error'):
        basic_cost.error = error

    return error
