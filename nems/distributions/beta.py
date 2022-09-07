from scipy import stats
import numpy as np

from .base import Distribution


class Beta(Distribution):
    """Beta prior.

    Parameters
    ----------
    alpha : scalar or ndarray
        Parameter of distribution
    beta : scalar or ndarray
        Parameter of distribution

    """

    def __init__(self, alpha, beta):
        self._alpha = np.asarray(alpha)
        self._beta = np.asarray(beta)

        # The beauty of using the scipy stats module is it correctly handles
        # scalar and multidimensional arrays of distributions.
        self.distribution = stats.beta(self._alpha, self._beta)

    def __repr__(self):
        alpha = self.value_to_string(self._alpha)
        beta = self.value_to_string(self._beta)
        return 'Beta(α={}, β={})'.format(alpha, beta)
        