import numpy as np
import numexpr as ne

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter
from .tools import require_shape, pop_shape


class StateGain(Layer):
    
    state_arg = 'state'  # see Layer docs for details

    def __init__(self, **kwargs):
        """Docs TODO.
        
        Parameters
        ----------
        shape : 2-tuple of int.
            (size of last dimension of state, size of last dimension of input)

        Examples
        --------
        TODO
        
        """
        require_shape(self, kwargs, minimum_ndim=2)
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Docs TODO
        
        Layer parameters
        ----------------
        gain : TODO
            prior:
            bounds:
        offset : TODO
            prior:
            bounds:
        
        """
        zero = np.zeros(shape=self.shape)
        one = np.ones(shape=self.shape)

        gain_mean = zero.copy()
        gain_mean[0,:] = 1  # TODO: Explain purpose of this?
        gain_sd = one/20
        gain_prior = Normal(gain_mean, gain_sd)
        gain = Parameter('gain', shape=self.shape, prior=gain_prior)

        offset_mean = zero
        offset_sd = one/20
        offset_prior = Normal(offset_mean, offset_sd)
        offset = Parameter('offset', shape=self.shape, prior=offset_prior)
        
        return Phi(gain, offset)

    def evaluate(self, input, state):
        """Multiply and shift input(s) by weighted sums of state channels.
        
        Parameters
        ----------
        input : ndarray
            Data to be modulated by state, typically the output of a previous
            Layer.
        state : ndarray
            State data to modulate input with.

        """

        gain, offset = self.get_parameter_values()
        return np.matmul(state, gain) * input + np.matmul(state, offset)

    @layer('stategain')
    def from_keyword(keyword):
        """Construct StateGain from keyword.
        
        Keyword options
        ---------------
        {digit}x{digit} : specifies shape, (n state channels, n stim channels)
            n stim channels can also be 1, in which case the same weighted
            channel will be broadcast to all stim channels (if there is more
            than 1).
        
        See also
        --------
        Layer.from_keyword

        """
        # TODO: other options from old NEMS
        options = keyword.split('.')
        shape = pop_shape(options)

        return StateGain(shape=shape)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class StateGainTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors, with state second.
                # TODO: Use tensor names to not require this arbitrary order.
                input = inputs[0]
                state = inputs[1]

                with_gain = tf.multiply(tf.matmul(state, self.gain), input)
                with_offset = with_gain + tf.matmul(state, self.offset)
                
                return with_offset

        return StateGainTF(self, **kwargs)


# Author: SVD, 2022-09-15.
class StateDexp(Layer):
    """State-dependent modulation through double exponential sigmoid."""

    state_arg = 'state'  # see Layer docs for details

    def __init__(self, per_channel=False, **kwargs):
        """Docs TODO.

        Parameters
        ----------
        shape : 2-tuple of int. (state_chans, input_chans), ie,
            (size of last dimension of state, size of last dimension of input)

        Examples
        --------
        TODO

        """
        require_shape(self, kwargs, minimum_ndim=2)
        self.per_channel = per_channel
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Docs TODO

        Layer parameters
        ----------------
        base_g, amp_g, kappa_g, offset_g : TODO
            prior:
            bounds:
        base_d, amp_d, kappa_d, offset_d : TODO
            prior:
            bounds:

        """
        zeros = np.zeros(shape=self.shape)
        ones = np.ones(shape=self.shape)

        # init gain params
        base_mean_g = zeros.copy()
        base_sd_g = ones.copy()
        amp_mean_g = zeros.copy() + 0
        amp_mean_g[:, 0] = 1
        amp_sd_g = ones.copy() * 0.1
        kappa_mean_g = zeros.copy()
        kappa_sd_g = ones.copy() * 0.1
        offset_mean_g = zeros.copy()
        offset_sd_g = ones.copy() * 0.1

        # init dc params
        base_mean_d = zeros.copy()
        base_sd_d = ones.copy()
        amp_mean_d = zeros.copy() + 0
        amp_sd_d = ones.copy() * 0.1
        kappa_mean_d = zeros.copy()
        kappa_sd_d = ones.copy() * 0.1
        offset_mean_d = zeros.copy()
        offset_sd_d = ones.copy() * 0.1

        return Phi(
            Parameter('base_g', shape=self.shape,
                      prior=Normal(base_mean_g, base_sd_g), bounds=(0,10)),
            Parameter('amp_g', shape=self.shape,
                      prior=Normal(amp_mean_g, amp_sd_g), bounds=(0,10)),
            Parameter('kappa_g', shape=self.shape,
                      prior=Normal(kappa_mean_g, kappa_sd_g),
                      bounds=(-np.inf,np.inf)),
            Parameter('offset_g', shape=self.shape,
                      prior=Normal(offset_mean_g, offset_sd_g),
                      bounds=(-np.inf,np.inf)),
            Parameter('base_d', shape=self.shape,
                      prior=Normal(base_mean_d, base_sd_d), bounds=(-10,10)),
            Parameter('amp_d', shape=self.shape,
                      prior=Normal(amp_mean_d, amp_sd_d), bounds=(-10,10)),
            Parameter('kappa_d', shape=self.shape,
                      prior=Normal(kappa_mean_d, kappa_sd_d),
                      bounds=(-np.inf,np.inf)),
            Parameter('offset_d', shape=self.shape,
                      prior=Normal(offset_mean_d, offset_sd_d),
                      bounds=(-np.inf,np.inf))
        )

    def evaluate(self, input, state):
        """Modulate input using double exponential transformation of state.

        TODO: Give some more context here. What's the reasoning for this
              implementation? Relevant publications?
              (Same goes for other Layers).

        Parameters
        ----------
        input : ndarray
            Data to be modulated by state, typically the output of a previous
            Layer.
        state : ndarray
            State data to modulate input with.

        Returns
        -------
        np.ndarray

        See also
        --------
        DoubleExponential.nonlinearity

        """

        base_g, amp_g, kappa_g, offset_g, base_d, amp_d, kappa_d, offset_d = \
            self.get_parameter_values()
        _g = [base_g, amp_g, kappa_g, offset_g]
        _d = [base_d, amp_d, kappa_d, offset_d]
        dexp = "b + a*exp(-exp(-exp(k)*(s-o)))"

        n_states = base_g.shape[0]
        n_inputs = base_g.shape[1]
        state = state[..., :n_states]

        if self.per_channel:
            if input.shape[-1] > 1:
                raise ValueError(
                    "StateDexp per-channel option only supports 1-channel input."
                )
            expression = dexp
            sg = np.empty_like(state)
            sd = np.empty_like(state)

            for i in range(n_states):
                s = state[..., i]
                for parameter_set, array in [(_g, sg), (_d, sd)]:
                    b, a, o, k = [p[i, [0]] for p in parameter_set]
                    s_dexp = ne.evaluate(expression, out=array[..., i])

        else:
            expression = f"sum({dexp}, 1)"
            sg = np.empty_like(input)
            sd = np.empty_like(input)

            for i in range(n_inputs):
                s = state
                for parameter_set, array in [(_g, sg), (_d, sd)]:
                    b, a, o, k = [p[:, i] for p in parameter_set]
                    s_dexp = ne.evaluate(expression, out=array[..., i])

        return sg * input + sd


    @layer('sdexp')
    def from_keyword(keyword):
        """Construct sdexp from keyword.

        Keyword options
        ---------------
        {digit}x{digit} : specifies shape, (n state channels, n stim channels)
            n stim channels can also be 1, in which case the same weighted
            channel will be broadcast to all stim channels (if there is more
            than 1).

        See also
        --------
        Layer.from_keyword

        """

        # TODO: other options from old NEMS
        options = keyword.split('.')
        shape = None
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])

        return StateDexp(shape=shape)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class SDexpTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors, with state second.
                # TODO: Use tensor names to not require this arbitrary order.
                input = inputs[0]
                state = inputs[1]

                bg = tf.expand_dims(tf.expand_dims(self.base_g, 0) ,0)
                ag = tf.expand_dims(tf.expand_dims(self.amp_g, 0) ,0)
                kg = tf.expand_dims(tf.expand_dims(self.kappa_g, 0) ,0)
                og = tf.expand_dims(tf.expand_dims(self.offset_g, 0) ,0)
                state_g = tf.reduce_sum(bg + ag * tf.math.exp(-tf.math.exp(
                    -tf.math.exp(kg) * (tf.expand_dims(state, -1) - og)
                    )), axis=2)
                bd = tf.expand_dims(tf.expand_dims(self.base_d, 0) ,0)
                ad = tf.expand_dims(tf.expand_dims(self.amp_d, 0) ,0)
                kd = tf.expand_dims(tf.expand_dims(self.kappa_d, 0) ,0)
                od = tf.expand_dims(tf.expand_dims(self.offset_d, 0) ,0)
                state_d = tf.reduce_sum(bd + ad * tf.math.exp(-tf.math.exp(
                    -tf.math.exp(kd) * (tf.expand_dims(state, -1) - od)
                    )), axis=2)

                return state_d + state_g * input

        return SDexpTF(self, **kwargs)
