import numpy as np
import numexpr as ne

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter, require_shape


class StaticNonlinearity(Layer):
    """Apply a nonlinear transformation to input(s).
    
    TODO: Test if current implementations will work with higher dim data.

    Parameters
    ----------
    shape : N-tuple of int (usually N=1)
        Determines the shape of each Parameter in `.parameters`.
        First dimension should match the last dimension of the input.
        Note that higher-dimesional shapes are also allowed, but overall
        Layer design is intended for 1-dimensional shapes.

    Attributes
    ----------
    skip_nonlinearity : bool
        If True, don't apply `nonlinearity` during `evaluate`. Still apply
        `input += shift`, if `'shift' in StaticNonlinearity.parameters`.
    
    """

    def __init__(self, **kwargs):
        require_shape(self, kwargs, minimum_ndim=1)
        super().__init__(**kwargs)
        self._skip_nonlinearity = False
        self._unfrozen_parameters = []

    def skip_nonlinearity(self):
        """Don't use `nonlinearity`, freeze nonlinear parameters."""
        self._unfrozen_parameters = [p.name for p in self.parameters
                                     if not p.is_frozen]
        self.freeze_parameters()
        self.unfreeze_parameters('shift')
        self._skip_nonlinearity = True

    def unskip_nonlinearity(self):
        """Use `nonlinearity`, unfreeze nonlinear parameters."""
        # Only unfreeze parameters that were previously unfrozen, but then
        # frozen by `skip_nonlinearity`.
        self.unfreeze_parameters(*self._unfrozen_parameters)
        self._unfrozen_parameters = []
        self._skip_nonlinearity = False

    def evaluate(self, input):
        """Apply `nonlinearity` to input(s). This should not be overwriten."""
        if not self._skip_nonlinearity:
            output = self.nonlinearity(input)
        else:
            # TODO: This works for time on 0-axis and 1-dim parameters,
            #       but need to add option to make this more general.
            # If there's a `shift` parameter for the subclassed nonlinearity,
            # still apply that. Otherwise, pass through inputs.
            output = input + self.parameters.get('shift', 0)
        return output

    def nonlinearity(self, input):
        """Pass through input(s). Subclasses should overwrite this."""
        return input

    def as_tensorflow_layer(self, **kwargs):
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        class StaticNonlinearityTF(NemsKerasLayer):
            def call(self, inputs):
                # TODO: why identity?
                return tf.identity(inputs + self.shift)

        return StaticNonlinearityTF(self, **kwargs)


class LevelShift(StaticNonlinearity):
    """Applies a scalar shift to each input channels.
    
    Notes
    -----
    While `LevelShift.evaluate` is linear, this Layer is grouped with
    `StaticNonlinearity` because of its close relation to these other layers.
    In short, we have found in the past that it is often helpful to "turn off"
    a nonlinearity during fitting, but still shift the input.
    
    """

    def initial_parameters(self):
        """Get initial values for `StaticNonlinearity.parameters`.
        
        Layer parameters
        ----------------
        shift : scalar or ndarray
            Value(s) that are added to input(s) prior to rectification. Shape
            (N,) must match N channels per input.
            Prior:  TODO
        
        """
        # TODO: explain choice of priors.
        prior = Normal(
            np.zeros(shape=self.shape), 
            np.ones(shape=self.shape)/100
            )

        shift = Parameter('shift', shape=self.shape, prior=prior)
        return Phi(shift)

    def nonlinearity(self, input):
        """constant shift

        Notes
        -----
        Simply add a constant shift to the signal

        """
        shift, = self.get_parameter_values()
        return input + shift

    @layer('lvl')
    def from_keyword(keyword):
        """Construct LevelShift from a keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape; required.

        Returns
        -------
        LevelShift

        See also
        --------
        Layer.from_keyword

        """
        options = keyword.split('.')
        shape = None
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])

        return LevelShift(shape=shape)


class DoubleExponential(StaticNonlinearity):
    """TODO: doc here? maybe just copy .evaluate?"""

    def initial_parameters(self):
        """Get initial values for `DoubleExponential.parameters`.
        
        Layer parameters
        ----------------
        base : scalar or ndarray
            Y-axis height of the center of the sigmoid.
            Shape (N,) must match N input channels (same for other parameters),
            such that one sigmoid transformation is applied to each channel.
            Prior:  Normal(mean=0, sd=1)
            Bounds: TODO
        amplitude : scalar or ndarray
            Y-axis distance from ymax asymptote to ymin asymptote
            Prior:  Normal(mean=5, sd=1.5)
            Bounds: TODO
        shift : scalar or ndarray
            Centerpoint of the sigmoid along x axis
            Prior:  Normal(mean=0, sd=1)
            Bounds: TODO
        kappa : scalar or ndarray
            Sigmoid curvature. Larger numbers mean steeper slop.
            Prior:  Normal(mean=1, sd=10)
            Bounds: TODO

        Returns
        -------
        nems.layers.base.Phi
        
        """
        # TODO: explain choices for priors.
        zero = np.zeros(shape=self.shape)
        one = np.ones(shape=self.shape)
        phi = Phi(
            Parameter('base', shape=self.shape, prior=Normal(-one, one/5)),
            Parameter('amplitude', shape=self.shape, prior=Normal(2*one, one/5)),
            Parameter('shift', shape=self.shape, prior=Normal(zero, one/5)),
            Parameter('kappa', shape=self.shape, prior=Normal(one, one/5))
            )
        return phi

    def nonlinearity(self, input):
        """Apply sigmoid transform to input x: $b+a*exp[-exp(-exp(k)(x-s)]$.
        
        See Thorson, Liénard, David (2015).
        
        """
        base, amplitude, shift, kappa = self.get_parameter_values()

        if (input.shape[-1] < base.shape[-1]) or (not self._inplace_ok):
            # First condition means output will be larger than input, so we
            # can't store it in the same array.
            out = None
        else:
            out = input

        output = ne.evaluate(
            "base + amplitude*exp(-exp(-exp(kappa)*(input+shift)))",
            out=out
            )

        return output

    @layer('dexp')
    def from_keyword(keyword):
        """Construct DoubleExponential from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape; required.

        Returns
        -------
        DoubleExponential

        See also
        --------
        Layer.from_keyword
        
        """
        shape = None
        options = keyword.split('.')
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])
        
        return DoubleExponential(shape=shape)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        if self._skip_nonlinearity:
            return super().as_tensorflow_layer(**kwargs)
        else:
            class DoubleExponentialTF(NemsKerasLayer):
                def call(self, inputs):
                    exp = tf.math.exp(-tf.math.exp(
                        -tf.math.exp(self.kappa) * (inputs + self.shift)
                        ))
                    return self.base + self.amplitude * exp

            return DoubleExponentialTF(self, **kwargs)


class RectifiedLinear(StaticNonlinearity):
    """TODO: doc here? maybe just copy .evaluate?"""
    def __init__(self, no_shift=True, no_offset=True, no_gain=True, **kwargs):
        super().__init__(**kwargs)
        fixed_parameters = {}
        shift, offset, gain = self.get_parameter_values()
        if no_shift: fixed_parameters['shift'] = np.full_like(shift, 0)
        if no_offset: fixed_parameters['offset'] = np.full_like(offset, 0)
        if no_gain: fixed_parameters['gain'] = np.full_like(gain, 1)
        self.set_permanent_values(**fixed_parameters)

    def initial_parameters(self):
        """Get initial values for `RectifiedLinear.parameters`.
        
        Layer parameters
        ----------------
        shift : scalar or ndarray
            Value(s) that are added to input prior to rectification. Shape
            (N,) must match N channels per input.
            Prior:  Normal(mean=-0.1, sd=1/sqrt(N))
        offset : scalar or ndarray
            Value(s) that are added to input after rectification.
            Prior:  TODO
        gain : scalar or ndarray
            Rectified input(s) will be multiplied by this (i.e. slope of the
            linear portion for each output).
            Prior:  TODO
        
        """
        # TODO: explain choice of prior.
        zero = np.zeros(shape=self.shape)
        one = np.ones(shape=self.shape)
        shift_prior = {'mean': zero+0.05, 'sd': one/100}
        offset_prior = {'mean': zero-0.05, 'sd': one/100}
        gain_prior = {'mean': one, 'sd': one/100}
        phi = Phi(
            Parameter('shift', shape=self.shape, prior=Normal(**shift_prior)),
            Parameter('offset', shape=self.shape, prior=Normal(**offset_prior)),
            Parameter('gain', shape=self.shape, prior=Normal(**gain_prior))
            )

        return phi

    def nonlinearity(self, input):
        """Implements `y = offset + gain * rectify(x - shift)`.
        
        By default, `offset=0, shift=0, gain=1` and this is equivalent to
        standard linear rectification: `y = 0 if x < 0, else x`. 

        Notes
        -----
        The negative of `shift` is used so that its interpretation in
        `StaticNonlinearity.evaluate` is the same as for other subclasses.
        
        """

        shift, offset, gain = self.get_parameter_values()

        if (input.shape[-1] < shift.shape[-1]) or (not self._inplace_ok):
            # First condition means output will be larger than input, so we
            # can't store it in the same array.
            out = None
        else:
            out = input

        output = ne.evaluate('offset + gain*((input + shift)*(input > -shift))')
        return output

    @layer('relu')
    def from_keyword(keyword):
        """Construct RectifiedLinear from a keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape; required.

        Returns
        -------
        RectifiedLinear

        See also
        --------
        Layer.from_keyword

        """
        options = keyword.split('.')
        no_shift = True
        no_offset = True
        no_gain = True
        shape=None

        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])
            elif op == 's':
                no_shift = False
            elif op == 'o':
                no_offset = False
            elif op == 'g':
                no_gain = False

        relu = RectifiedLinear(
            shape=shape, no_shift=no_shift, no_offset=no_offset,
            no_gain=no_gain
            )

        return relu

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        if self._skip_nonlinearity:
            return super().as_tensorflow_layer(**kwargs)
        else:
            class RectifiedLinearTF(NemsKerasLayer):
                def call(self, inputs):
                    rectified  = tf.nn.relu(inputs + self.shift)
                    return self.offset + self.gain * rectified

        return RectifiedLinearTF(self, **kwargs)

# Optional alias
class ReLU(RectifiedLinear):
    pass
