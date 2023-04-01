import numpy as np
#import matplotlib.pyplot as plt


class Distribution:
    '''
    Base class for a Distribution
    '''

    @classmethod
    def value_to_string(cls, value):
        if value.ndim == 0:
            return 'scalar'
        else:
            shape = ', '.join(str(v) for v in value.shape)
            return 'array({})'.format(shape)

    def mean(self):
        '''
        Return the expected value of the distribution
        '''
        return self.distribution.mean()

    def percentile(self, percentile):
        '''
        Calculate the percentile

        Parameters
        ----------
        percentile : float [0, 1]
            Probability at which the result is calculated. Should be specified as
            a fraction in the range 0 ... 1 rather than a percent.

        Returns
        -------
        value : float
            Value of random variable at given percentile

        For some distributions (e.g., Normal), the bounds will be +/- infinity.
        In those situations, you can request that you get the bounds for the 99%
        interval to get a slightly more reasonable constraint that can be passed
        to the fitter.

        >>> from nems0.distributions.api import Normal
        >>> prior = Normal(mu=0, sd=1)
        >>> lower = prior.percentile(0.005)
        >>> upper = prior.percentile(0.995)
        '''
        return self.distribution.ppf(percentile)

    @property
    def shape(self):
        return self.mean().shape

    def sample(self, n=None):
        if n is None:
            return self.distribution.rvs().reshape(self.shape)
        size = [n] + list(self.shape)
        return self.distribution.rvs(size=size)

    def tolist(self):
        d = self.__dict__
        if 'distribution' in d:
            del d['distribution']
        name = type(self).__name__
        l = [name, d]
        return l

    ## TODO: Move to plots.py
    #def plot(self, ax=None, **plot_kw):
    #    # Get mi and max percentiles across the full set of priors.
    #    prior_min = self.percentile(0.01)
    #    prior_max = self.percentile(0.99)
    #    x_min = np.min(prior_min)
    #    x_max = np.max(prior_max)

    #    # Create x and reshape so that it broadcasts across all dimensions of
    #    # the set of priors.
    #    x = np.linspace(x_min, x_max, 1000)
    #    x.shape = [x.size] + [1]*len(self.shape)
    #    y = self.distribution.pdf(x)

    #    # Reshape x and y to facilitate plotting
    #    x = x.ravel()
    #    y.shape = (x.size, -1)

    #    if ax is None:
    #        figure, ax = plt.subplots(1, 1)

    #    ax.plot(x, y, **plot_kw)
    #    ax.set_xlabel('Value')
    #    ax.set_ylabel('PDF')

    #    return ax
