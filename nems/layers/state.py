import numpy as np
import numexpr as ne
import logging

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter
from .tools import require_shape, pop_shape

log = logging.getLogger(__name__)


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
        #log.info(f"{state.shape}, {input.shape}, {gain.shape}")
        if gain.shape[0]==state.shape[1]:
            with_gain = np.matmul(state, gain) * input
        else:
            with_gain = (np.matmul(state, gain[1:,]) + gain [0,:]) * input
        if offset.shape[0]==state.shape[1]:
            with_offset = with_gain + np.matmul(state, offset)
        else:
            with_offset = with_gain + offset[[0],:] + np.matmul(state, offset[1:,])
            
        return with_offset

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
        kwargs = {}
        kwargs['shape'] = pop_shape(options)

        for op in options:
            if op.startswith('l2'):
                kwargs['regularizer'] = op

        return StateGain(**kwargs)

    def as_tensorflow_layer(self, input_shape=None, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer
        if input_shape is None:
            raise ValueError(f"input_shape=[input.shape, staet.shape] required")
        stim_shape=input_shape[0]
        state_shape=input_shape[1]
        gain_len=self['gain'].shape[0]
        offset_len=self['offset'].shape[0]
        
        class StateGainTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors, with state second.
                # TODO: Use tensor names to not require this arbitrary order.
                input = inputs[0]
                state = inputs[1]
                if gain_len==state_shape[1]:
                    with_gain = tf.multiply(tf.matmul(state, self.gain), input)
                else:
                    with_gain = tf.multiply(tf.slice(self.gain,[0,0],[1,-1]) + tf.matmul(state, tf.slice(self.gain,[1,0],[-1,-1])), input)
                if offset_len==state_shape[1]:
                    with_offset = with_gain + tf.matmul(state, self.offset)
                else:
                    with_offset = with_gain + tf.slice(self.offset,[0,0],[1,-1]) + tf.matmul(state, tf.slice(self.offset,[1,0],[-1,-1]))

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

    
class StateHinge(Layer):
    
    state_arg = 'state'  # see Layer docs for details
    def __init__(self, match_sign=False, **kwargs):
        """Docs TODO.
        
        Offset / gain state effect with a hinge. Ie, locally linear in two parts. 
        
        Parameters
        ----------
        shape : 2-tuple of int.
            (size of last dimension of state, size of last dimension of input)
        match_sign : if True, force sign of gain/offset to be same on both sides of hinge point
        
        Examples
        --------
        TODO
        
        """
        require_shape(self, kwargs, minimum_ndim=2)
        self.match_sign = match_sign
        
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Docs TODO
        
        Layer parameters
        ----------------
        gain1, gain2 : TODO
            prior:
            bounds:
        offset1, offset2 : TODO
            prior:
            bounds:
        x0 : TODO
        
        """
        zero = np.zeros(shape=self.shape)
        one = np.ones(shape=self.shape)

        gain_mean = zero.copy()
        gain_mean[0,:] = 1  # Why - gain for first dim is hard-coded, assuming that first state channel is a constant
        gain_sd = one/20
        gain_prior = Normal(gain_mean, gain_sd)
        gain1 = Parameter('gain1', shape=self.shape, prior=gain_prior)
        gain_prior2 = Normal(gain_mean, gain_sd)
        gain2 = Parameter('gain2', shape=self.shape, prior=gain_prior2)

        offset_mean = zero
        offset_sd = one
        offset1_prior = Normal(offset_mean, offset_sd)
        offset1 = Parameter('offset1', shape=self.shape, prior=offset1_prior)
        offset2_prior = Normal(offset_mean, offset_sd)
        offset2 = Parameter('offset2', shape=self.shape, prior=offset2_prior)
        
        x0_mean = zero
        x0_sd = one
        x0_prior = Normal(x0_mean, x0_sd)
        x0 = Parameter('x0', shape=self.shape, prior=x0_prior)
        
        return Phi(gain1, gain2, offset1, offset2, x0)

    def get_parameter_values(self, *parameter_keys, as_dict=False):
        
        d = super().get_parameter_values(*parameter_keys, as_dict=True)
        if parameter_keys == ():
            parameter_keys = self.parameters.keys()
        
        if self.match_sign:
            if 'gain2' in parameter_keys:
                s1 = np.sign(d['gain1'])
                d['gain2'] = np.abs(d['gain2']) * s1
            if 'offset2' in parameter_keys:
                s1 = np.sign(d['offset1'])
                d['offset2'] = np.abs(d['offset2']) * s1
        
        if as_dict:
            return d
        else:
            return tuple([d[k] for k in parameter_keys])

    def evaluate(self, input, state, return_intermediates=False):
        """Multiply and shift input(s) by weighted sums of state channels.
        
        Parameters
        ----------
        input : ndarray
            T x N x ... matrix. Data to be modulated by state, typically the output of a previous
            Layer.
        state : ndarray
            T x S matrix. State data to modulate input with.
        return_intermediates : bool
            if True, returns (out, d, g) tuple instead of just out
        """

        gain1, gain2, offset1, offset2, x0 = self.get_parameter_values()
        
        #if self.match_sign:
        #    s1 = np.sign(gain1)
        #    gain2 = np.abs(gain2) * s1
        #    s1 = np.sign(offset1)
        #    offset2 = np.abs(offset2) * s1
        
        # gain applied point-wise to each state channel, split above and below x0
        s_ = state[..., np.newaxis]-x0[np.newaxis, ...]
        sp = s_ * (s_>0)
        sn = s_ * (s_<0)
        
        g = (np.tensordot(sn, gain1, (1, 0)) + np.tensordot(sp, gain2, (1, 0)))[:,0,:]
        d = (np.tensordot(sn, offset1, (1, 0)) + np.tensordot(sp, offset2, (1, 0)))[:,0,:]
        out = g * input + d
        
        if return_intermediates:
            return out, d, g
        else:
            return out
            
    @layer('statehinge')
    def from_keyword(keyword):
        """Construct StateHinge from keyword.
        
        Keyword options
        ---------------
        {digit}x{digit} : specifies shape, (n state channels, n stim channels)
            n stim channels can also be 1, in which case the same weighted
            channel will be broadcast to all stim channels (if there is more
            than 1).
        'm' : match sign of gain and offset on both sides of hinge point.
        
        See also
        --------
        Layer.from_keyword

        """
        # TODO: other options from old NEMS?
        options = keyword.split('.')
        opts = {}
        
        for op in options:
            if op == 'm':
                opts['matched_sign']=True
            else:
                shape = pop_shape(options)

        return StateHinge(shape=shape, **opts)

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


class HRTF(Layer):

    # TODO fix this.. currenty defaulting to aerd breaks pytest
    state_arg = 'state'  # azimuth, elevation, tilt, distance
    #state_arg = 'aerd'  # azimuth, elevation, tilt, distance

    def __init__(self, speaker_count=2, **kwargs):
        """Docs TODO.

        Parameters
        ----------
        shape : 2-tuple of int.
            (size of last dimension of aerd, number of spectral channels per source)
            eg, 6 x 18 means 6 aerd channels, 18-channel spectrogram
        speaker_count : number of sound sources (default 2)

        Examples
        --------
        TODO

        """
        require_shape(self, kwargs, minimum_ndim=1)

        from nems_lbhb.projects.freemoving.free_tools import load_hrtf

        self.speaker_count=speaker_count
        self.num_freqs = kwargs['shape'][1]
        self.L, self.R, self.c, self.A, self.E = load_hrtf(
            format='az_el', fmin=200, fmax=20000,
            num_freqs=self.num_freqs)

        super().__init__(**kwargs)

    #def initial_parameters(self):
    #    pass

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
        pass
        return input
        #gain, offset = self.get_parameter_values()
        #return np.matmul(state, gain) * input + np.matmul(state, offset)

    @layer('hrtf')
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
        kwargs = {}
        for op in options[1:]:
            if op[0]=='c':
                kwargs['speaker_count']=int(op[1:])

        return HRTF(shape=shape, **kwargs)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class hrtfTF(NemsKerasLayer):

            def call(self, inputs):

                geometry = inputs[0]

                # stub from ChatGPT

                def generate_tensor_and_select_values(numpy_array_3d, index_matrix_3_by_T):
                    # Convert NumPy array to TensorFlow constant tensor
                    tf_tensor = tf.constant(numpy_array_3d, dtype=tf.float32)

                    # Get the dimensions of the input arrays
                    depth, height, width = numpy_array_3d.shape
                    _, T = index_matrix_3_by_T.shape

                    # Round values to the nearest integer and clip to valid index range
                    rounded_indices = tf.clip_by_value(tf.round(index_matrix_3_by_T), 0,
                                                       tf.constant([depth - 1, height - 1, width - 1],
                                                                   dtype=tf.float32))

                    # Convert to integer indices
                    flattened_indices = tf.cast(tf.reshape(rounded_indices, [-1]), dtype=tf.int32)

                    # Generate indices for tf.gather_nd
                    indices = tf.transpose(tf.reshape(flattened_indices, [T, -1]))

                    # Use tf.gather_nd to select values from the 3D tensor
                    selected_values = tf.gather_nd(tf_tensor, indices)

                    return selected_values

                # Example usage:
                # Create a random 3D NumPy array
                numpy_array_3d = np.random.rand(2, 3, 4)

                # Create a random 3-by-T index matrix with continuous values
                index_matrix_3_by_T = np.array([[0.3, 1.6], [1.1, 2.8], [0.9, 2.4]])

                # Call the function
                result = generate_tensor_and_select_values(numpy_array_3d, index_matrix_3_by_T)

                # Start a TensorFlow session and evaluate the result
                with tf.Session() as sess:
                    result_value = sess.run(result)

                print("Original 3D Tensor:")
                print(numpy_array_3d)
                print("\nIndex Matrix (continuous values):")
                print(index_matrix_3_by_T)
                print("\nSelected Values:")
                print(result_value)


                # Assume inputs is a list of two tensors, with state second.
                # TODO: Use tensor names to not require this arbitrary order.
                input = inputs[0]
                state = inputs[1]

                with_gain = tf.multiply(tf.matmul(state, self.gain), input)
                with_offset = with_gain + tf.matmul(state, self.offset)

                return with_offset

        return hrtfTF(self, **kwargs)
