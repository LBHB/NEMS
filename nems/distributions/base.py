import numpy as np


class Distribution:
    """Base class for a Distribution."""

    # Any subclass of Distribution will be registered here, for use by
    # `Distribution.from_json`
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def value_to_string(cls, value):
        if value.ndim == 0:
            return 'scalar'
        else:
            shape = ', '.join(str(v) for v in value.shape)
            return 'array({})'.format(shape)

    def mean(self):
        """Return the expected value of the distribution."""
        return self.distribution.mean()

    def percentile(self, percentile):
        """Calculate the percentile.

        Parameters
        ----------
        percentile : float [0, 1]
            Probability at which the result is calculated. Should be specified as
            a fraction in the range 0 ... 1 rather than a percent.

        Returns
        -------
        value : float
            Value of random variable at given percentile

        Examples
        --------
        For some distributions (e.g., Normal), the bounds will be +/- infinity.
        In those situations, you can request that you get the bounds for the 99%
        interval to get a slightly more reasonable constraint that can be passed
        to the fitter.
        >>> from nems.distributions.api import Normal
        >>> prior = Normal(mu=0, sd=1)
        >>> lower = prior.percentile(0.005)
        >>> upper = prior.percentile(0.995)

        """
        return self.distribution.ppf(percentile)

    @property
    def shape(self):
        return self.mean().shape

    def sample(self, n=1, bounds=None):
        """Draw random sample(s) from a (truncated) distribution.
        
        Parameters
        ----------
        n : int
            Number of random samples to get.
        bounds : 2-tuple or None
            If not None, samples with at least one value less than `bounds[0]`
            or greater than `bounds[1]` will be rejected and replaced with
            a new sample.

        Returns
        -------
        good_sample : ndarray
        
        """
        size = [n] + list(self.shape)
        good_sample = np.full(shape=size, fill_value=np.nan)

        while np.sum(np.isnan(good_sample)) > 0:
            sample = self.distribution.rvs(size=size)
            if bounds is not None:
                lower, upper = bounds
                keep = (sample >= lower) & (sample <= upper)
                good_sample[keep] = sample[keep]
            else:
                good_sample = sample
                break

        # Drop first dimension if n = 1
        if n == 1:
            good_sample = np.squeeze(good_sample, axis=0)

        return good_sample

    def tolist(self):
        """Represent distribution as a list.
        
        See also
        --------
        Distribution.to_json
        Distribution.from_json

        """
        name = type(self).__name__
        d = self.__dict__.copy()
        for k in list(d.keys()):
            # Remove `self.distribution`
            # and any attributes with two leading underscores
            if (k == 'distribution') or (k.startswith(f'_{name}__')):
                _ = d.pop(k)

        l = [name, d]
        return l

    def to_json(self):
        """Encode a distribution instance as a dictionary.

        See also
        --------
        `nems.tools.json`.
        
        """
        return {'data': self.tolist()}

    @classmethod
    def from_json(cls, json):
        """Decode a distribution from a dictionary.
        
        Warnings
        --------
        Distribution subclasses should avoid assigning instance attributes that
        are not used in `__init__`, as this will break compatibility with
        `from_json`. If such attributes are absolutely needed, use two leading
        underscores (i.e. `self.__my_attr = attr`), as those will not be encoded
        by `to_json`.

        See also
        --------
        `nems.tools.json`.

        """
        class_name, kwargs = json['data']
        # remove first leading underscore kwargs keys, if any
        kwargs = {'_'.join(k.split('_')[1:]): v for k, v in kwargs.items()}
        if class_name == 'Distribution':
            class_obj = Distribution
        else:
            class_obj = cls.subclasses[class_name]
        return class_obj(**kwargs)

    def __eq__(self, other):
        if isinstance(other, Distribution):
            list1 = self.tolist()
            list2 = other.tolist()
            conditions = [
                (list1[0] == list2[0]),  # same subclass
                all([(k1 == k2 and np.allclose(v1, v2)) for (k1,v1), (k2,v2) in
                     zip(list1[1].items(), list2[1].items())])
            ]
            return all(conditions)
        else:
            return NotImplemented
