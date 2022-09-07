import numpy as np

from nems.distributions import Normal


class Parameter:
    """Stores and manages updates to values for one parameter of one Layer."""

    def __init__(self, name, shape=(), prior=None, bounds=None,
                 default_bounds='infinite', zero_to_epsilon=False,
                 initial_value='mean'):
        """Stores and manages updates to values for one parameter of one Layer.

        Parameters are intended to exist as components of a parent Phi instance,
        by invoking `Phi.add_parameter`. Without establishing this relationship,
        most Parameter methods will not work.

        As with Phi, Parameters should generally not be interacted with
        directly unless implementing new Layer subclasses or other core
        functionality. Wherever possible, users should interact with fittable
        parameters using Model- or Layer-level methods.

        Parameters
        ----------
        name : str
            This will also be the Parameter's key in `Phi._dict`.
        shape : N-tuple, default=()
            Shape of `Parameter.values`.
        prior : nems.distributions.Distribution or None; optional
            Prior distribution for this parameter, with matching shape.
        bounds : 2-tuple or None; optional
            (minimum, maximum) values for the entries of `Parameter.values`.
        default_bounds : str, default='infinite'
            Determines behavior when `bounds=None`.
            If `'infinite'`  : set bounds to (-np.inf, np.inf)
            If `'percentile'`: set bounds to tails of `Parameter.prior`.
                (prior.percentile(0.0001), prior.percentile(0.9999))
        zero_to_epsilon : False
            If True, change 0-entries of bounds to machine epsilon for float32,
            to avoid division by 0. Essentially this makes bounds an open
            (or half-open) interval if it includes zero values. This flag should
            be set any time that `Parameter.values` would be used in division,
            log, etc.
        initial_value : str, scalar, or ndarray, default='mean'
            Determines initial entries of `Parameter.values`.
            If `'mean'`   : set values to `Parameter.prior.mean()`.
            If `'sample'` : set values to `Parameter.prior.sample()`.
            If scalar     : set values to `np.full(Parameter.shape, scalar)`.
            If ndarray    : set values to array (must match `Parameter.shape`).

        See also
        --------
        nems.models.base.Model
        nems.layers.base.Layer
        nems.layers.base.Phi

        """
        self.name = name
        self.shape = tuple(shape)
        self.size = 1
        for axis in shape:
            self.size *= axis

        # default to multivariate normal
        if prior is None:
            zero = np.zeros(shape=self.shape)
            one = np.ones(shape=self.shape)
            prior = Normal(mean=zero, sd=one)  
        self.prior = prior
        if prior.shape != self.shape:
            raise ValueError(
                "Parameter.shape != Parameter.prior.shape for...\n"
                f"Parameter:       {self.name}\n"
                f"Parameter.shape: {self.shape}\n"
                f"prior.shape:     {prior.shape}"
                )

        if bounds is None:
            # Set bounds based on `default_bounds`
            if default_bounds == 'percentile':
                bounds = (prior.percentile(0.0001), prior.percentile(0.9999))
            elif default_bounds == 'infinite':
                bounds = (-np.inf, np.inf)
            else:
                raise ValueError(
                    "Unrecognized default_bounds for...\n"
                    f"Parameter:      {self.name}\n"
                    f"default_bounds: {default_bounds}\n"
                    "Accepted values are 'percentile' or 'infinite'."
                    )
        else:
            bounds = tuple(bounds)

        if zero_to_epsilon:
            # Swap 0 with machine epsilon for float32
            eps = np.finfo(np.float32).eps
            lower, upper = bounds
            lower = eps if lower == 0 else lower
            upper = eps if upper == 0 else upper
            bounds = (lower, upper)

        self.bounds = bounds

        if isinstance(initial_value, str) and (initial_value == 'mean'):
            value = self.prior.mean()
        elif isinstance(initial_value, str) and (initial_value) == 'sample':
            value = self.prior.sample(bounds=self.bounds)
        elif np.isscalar(initial_value):
            value = np.full(self.shape, initial_value)
        elif isinstance(initial_value, np.ndarray):
            value = initial_value
        else:
            raise ValueError(
                "Unrecognized initial_value for...\n"
                f"Parameter:     {self.name}\n"
                f"initial_value: {initial_value}\n"
                "Accepted values are 'mean', 'sample', scalar, or ndarray."
                )

        self.initial_value = np.ravel(value)

        # Must be set by `Phi.add_parameter` for `Parameter.values` to work.
        self.phi = None  # Pointer to parent Phi instance.
        self.first_index = None  # Location of data within Phi.get_vector()
        self.last_index = None

        self.is_frozen = False
        self.is_permanent = False
    
    # TODO: any other tracking/upkeep that needs to happen with
    #       freezing/unfreezing, or is the flag sufficient?
    def freeze(self):
        """Use parameter values for evaluation only, do not optimize."""
        self.is_frozen = True

    def unfreeze(self):
        """Make parameter values optimizable."""
        if not self.is_permanent:
            self.is_frozen = False

    def make_permanent(self):
        """Set parameter to a fixed value. Do not unfreeze or resample."""
        self.freeze()
        self.is_permanent = True

    @property
    def is_fittable(self):
        """Alias property for negation of `Parameter.is_frozen`."""
        return not self.is_frozen

    @property
    def values(self):
        """Get corresponding parameter values stored by parent Phi."""
        values = self.phi._get_parameter_vector(self)
        return np.reshape(values, self.shape)

    def sample(self, n=1, inplace=False):
        """Get or set new values by sampling from `Parameter.prior`.

        Re-sampling is used to ensure that the final sample falls within
        `Parameter.bounds`.
        
        Parameters
        ----------
        n : int
            Number of samples to get.
        inplace : bool
            If True, sampled values will be used to update `Parameter.values`
            inplace. Otherwise, the sampled values will be returned without
            changing current values.

        See also
        --------
        nems.layers.base.Phi.sample
        nems.distributions.base.Distribution.sample

        Return
        ------
        sample : ndarray

        """

        sample = self.prior.sample(n=n, bounds=self.bounds)
        if inplace:
            if self.is_permanent:
                # TODO: switch this to log.debug? or do something else maybe,
                #       instead of silently skipping.
                pass
            else:
                self.update(sample)
        else:
            return sample

    def mean(self, inplace=False):
        """Get, or set parameter values to, mean of `Parameter.prior`.
        
        Note that `Parameter.mean()` may return a value outside of
        `Parameter.bounds`. In that event, either priors or bounds should be
        changed.

        Parameters
        ----------
        inplace : bool, default=False
            If True, mean value will be used to update `Parameter.values`
            inplace Otherwise, mean will be returned without changing current
            values.

        Return
        ------
        mean : ndarray

        """
        mean = self.prior.mean()
        if inplace:
            if self.is_permanent:
                # TODO: switch this to log.debug? or do something else maybe,
                #       instead of silently skipping.
                pass
            else:
                self.update(mean)
        else:
            return mean

    def update(self, value, ignore_bounds=False):
        """Set `Parameters.values` to `value` by updating `Phi._array`.
        
        Parameters
        ----------
        value : scalar or array-like
            New value for `Parameter.values`. Must match `Parameter.shape`.
        ignore_bounds : bool
            If True, ignore `Parameter.bounds` when updating. Otherwise,
            new values will be rejected if they are less than `bounds[0]` or
            greater than `bounds[1]`.
        
        """
        value = np.asarray(value)
        if not ignore_bounds:
            lower, upper = self.bounds
            if np.any(value < lower) or np.any(value > upper):
                raise ValueError(
                    f"value out-of-bounds for...\n"
                    f"Parameter: {self.name}\n"
                    f"Bounds:    {self.bounds}\n"
                    f"Value:     {value}"
                )

        if np.shape(value) != self.shape:
            raise ValueError(
                f"Parameter {self.name} requires shape {self.shape}, but "
                f"{value} has shape {np.shape(value)}"
            )
        else:
            flat_value = np.ravel(value)
            mask = self.phi._get_parameter_mask(self)
            self.phi._array[mask] = flat_value

    def get_bounds_vector(self, none_for_inf=True):
        """Get a list with one bounds copy per entry in `Parameter.values`.
        
        Parameters
        ----------
        none_for_inf : bool, default=True
            If True, replace (+/-)`np.inf` with None for compatibility with
            `scipy.optimize.minimize`.
        
        Returns
        -------
        bounds : list of 2-tuples
        
        """
        bounds = self.bounds
        if none_for_inf:
            lower, upper = bounds
            if np.isinf(lower):
                lower = None
            if np.isinf(upper):
                upper = None
            bounds = (lower, upper)
        return [bounds] * self.size

    # Add compatibility for saving to .json    
    def to_json(self):
        """Encode Parameter object as json. See `nems.tools.json`."""
        data = {
            'kwargs': {
                'name': self.name,
                'shape': self.shape,
                'prior': self.prior,
                'bounds': self.bounds
            },
            'is_frozen': self.is_frozen,
            'is_permanent': self.is_permanent
        }
        return data

    @classmethod
    def from_json(cls, json):
        """Decode Parameter object from json. See `nems.tools.json`."""
        p = cls(**json['kwargs'])
        if json['is_permanent']:
            p.make_permanent()
        elif json['is_frozen']:
            p.freeze()
        return p

    def __str__(self):
        dash_break = "-"*16 + "\n"
        string = f"Parameter(name={self.name}, shape={self.shape})\n"
        string += dash_break
        string += f".prior:     {self.prior}\n"
        string += f".bounds:    {self.bounds}\n"
        if self.is_permanent:
            string += ".is_permanent: True\n"
        else:
            string += f".is_frozen: {self.is_frozen}\n"
        string += ".values:\n"
        string += f"{self.values}\n"
        string += dash_break
        return string

    def __repr__(self):
        string = f"Parameter(name={self.name}, shape={self.shape})\n"
        return string

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.to_json() == other.to_json()
        else:
            return NotImplemented


    # TODO: Maybe it would be better to leave this functionality out?
    #       I keep running into cases where this fails and end up using
    #       .values anyway.

    # Add compatibility with numpy ufuncs, len(), and other methods that
    # should point to `Parameter.values` instead of `Parameter`.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Propagate numpy ufunc operations to `Parameter.values`.
        
        Convenience method to eliminate excessive references to
        `Parameter.values`.
        
        Notes
        -----
        Works with `@` but not `np.dot`.

        See also
        --------
        https://numpy.org/doc/stable/reference/ufuncs.html
        
        """
        f = getattr(ufunc, method)
        # replace Parameter objects with Parameter.values
        subbed_inputs = [
            x.values if isinstance(x, Parameter) else x
            for x in inputs
            ]
        output = f(*subbed_inputs, **kwargs)
        return output

    @property
    def T(self):
        # np.transpose() works without this, but have to add .T separately.
        return self.values.T

    # Pass through iteration and math operators to `Parameter.values`.
    def __len__(self):
        return self.values.__len__()
    def __iter__(self):
        return self.values.__iter__()
    def __add__(self, other):
        return self.values.__add__(other)
    def __sub__(self, other):
        return self.values.__sub__(other)
    def __mul__(self, other):
        return self.values.__mul__(other)
    def __matmul__(self, other):
        return self.values.__matmul__(other)
    def __truediv__(self, other):
        return self.values.__truediv__(other)
    def __floordiv__(self, other):
        return self.values.__floordiv__(other)
    def __mod__(self, other):
        return self.values.__mod__(other)
    def __divmod__(self, other):
        return self.values.__divmod__(other)
