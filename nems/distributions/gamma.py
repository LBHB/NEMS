from scipy import stats
import numpy as np

from .base import Distribution


class Gamma(Distribution):
    """Gamma prior.

    Parameters
    ----------
    alpha : scalar or ndarray
        Parameter of distribution
    beta : scalar or ndarray
        Parameter of distribution

    Examples
    --------
    To create a Gamma prior using the mean and standard deviation, use the
    classmethod, `from_moments`. This will solve for `alpha` and `beta` given
    your desired mean and standard deviation.
    >>> prior = Gamma.from_moments(mu=10, sd=100)

    """

    @classmethod
    def from_moments(cls, mu, sd):
        # E[X] = α/β
        # Var[X] = α/β^2
        var = sd**2
        alpha = (mu**2)/var
        beta = alpha/mu
        return cls(alpha, beta)

    def __init__(self, alpha, beta):
        self._alpha = np.asarray(alpha)
        self._beta = np.asarray(beta)
        self.distribution = stats.gamma(a=self._alpha, scale=1/self._beta)

    def __repr__(self):
        alpha = self.value_to_string(self._alpha)
        beta = self.value_to_string(self._beta)
        return 'Gamma(α={}, β={})'.format(alpha, beta)
