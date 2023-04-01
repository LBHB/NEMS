import logging
import copy

from nems0.fitters.fitter import scipy_minimize
import nems0.metrics.api as metrics
from .fit_basic import fit_basic
from .fit_iteratively import fit_iteratively

log = logging.getLogger(__name__)


def fit_nfold(data_list, modelspecs, generate_psth=False,
              fitter=scipy_minimize, analysis='fit_basic',
              metric=None, tolerances=None, module_sets=None,
              tol_iter=100, fit_iter=20, fit_kwargs={}):
    '''
    Takes njacks jackknifes, where each jackknife has some small
    fraction of data NaN'd out, and fits modelspec to them.

    TESTING:
    if input len(modelspecs) == len(data_list) then use each
      modelspec as initial condition for corresponding data_list fold
    if len(modelspecs) == 1, then use the same initial conditions for
      each fold

    '''
    if type(data_list) is list:
        nfolds = len(data_list)
    else:
        # backward compatibility
        data_list = data_list.views()
        nfolds = len(data_list)

    models = []
    if metric is None:
        def metric(d):
            metrics.nmse(d, 'pred', 'resp')

    for i in range(nfolds):
        if len(modelspecs) > 1:
            msidx = i
        else:
            msidx = 0

        log.info("Fitting fold %d/%d, modelspec %d", i+1, nfolds, msidx)

        if analysis == 'fit_basic':
            models += fit_basic(data_list[i], copy.deepcopy(modelspecs[msidx]),
                                fitter=fitter,
                                metric=metric,
                                metaname='fit_nfold',
                                fit_kwargs=fit_kwargs)
        elif analysis == 'fit_iteratively':
            models += fit_iteratively(
                        data_list[i], copy.deepcopy(modelspecs[msidx]),
                        fitter=fitter, metric=metric,
                        metaname='fit_nfold', fit_kwargs=fit_kwargs,
                        module_sets=module_sets, invert=False,
                        tolerances=tolerances, tol_iter=tol_iter,
                        fit_iter=fit_iter,
                        )

        else:
            # Unknown analysis
            # TODO: use getattr / import to make this more general for
            #       use with any analysis function?
            #       Maybe too much of a pain.
            raise NotImplementedError

    return models
