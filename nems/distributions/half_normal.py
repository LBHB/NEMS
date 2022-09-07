from scipy import stats
import numpy as np

from .base import Distribution


class HalfNormal(Distribution):
    """HalfNormal prior.

    Parameters
    ----------
    sigma : scalar or ndarray
        Standard deviation of distribution
    
    """

    def __init__(self, sd):
        self._sd = np.asarray(sd)
        self.distribution = stats.halfnorm(scale=self._sd)

    def __repr__(self):
        sd = self.value_to_string(self._sd)
        return 'HalfNormal(σ={})'.format(sd)
