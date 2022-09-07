from scipy import stats
import numpy as np

from .base import Distribution


class Exponential(Distribution):
    """Exponential Prior.

    Parameters
    ----------
    beta : scalar or ndarray
        Scale of distribution.

    """

    def __init__(self, beta):
        self._beta = np.asarray(beta)
        self.distribution = stats.expon(scale=beta)

    def __repr__(self):
        return 'Exponential(β={})'.format(self._beta)
