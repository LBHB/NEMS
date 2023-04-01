import logging
import copy

import nems0.priors
from nems0.analysis.fit_basic import fit_basic
from nems0.analysis.fit_iteratively import fit_iteratively

log = logging.getLogger(__name__)


def fit_from_priors(data, modelspec, ntimes=10, analysis='fit_basic',
                    subset=None, basic_kwargs={}, iter_kwargs={}):
    '''
    Fit ntimes times, starting from random points sampled from the prior.

    TODO : Test, add more parameters
    '''
    if subset is None:
        subset = [i for i in range(len(modelspec))]

    models = []
    for i in range(ntimes):
        log.info("Fitting from random start: {}/{}".format(i+1, ntimes))
        # Only randomize phi for modules specified in subset
        cp = copy.deepcopy(modelspec)
        sub = [m for i, m in enumerate(cp) if i in subset]
        rand = nems0.priors.set_random_phi(sub)
        merged_ms = [m if i not in subset else rand.pop(0)
                     for m in cp]

        if analysis == 'fit_basic':
            models.append(fit_basic(data, merged_ms,
                                    metaname='fit_from_priors',
                                    **basic_kwargs)[0])

        elif analysis == 'fit_iteratively':
            models.append(fit_iteratively(data, merged_ms,
                                          metaname='fit_from_priors',
                                          **iter_kwargs)[0])
        else:
            raise NotImplementedError("No support for analysis: %s" % analysis)

    return models
