import numpy as np
import numexpr as ne
import scipy.signal
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

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        # shape[0] = n_state_channels; if gain/offset has an extra row it's a bias term
        n_state = self.shape[0]
        has_gain_bias = (self['gain'].shape[0] != n_state)
        has_offset_bias = (self['offset'].shape[0] != n_state)

        class StateGainTF(NemsKerasLayer):

            def call(self, inputs):
                # inputs is [input, state], state second (state_arg = 'state')
                inp = inputs[0]
                state = inputs[1]
                if not has_gain_bias:
                    with_gain = tf.matmul(state, self.gain) * inp
                else:
                    with_gain = (tf.slice(self.gain, [0, 0], [1, -1]) +
                                 tf.matmul(state, tf.slice(self.gain, [1, 0], [-1, -1]))) * inp
                if not has_offset_bias:
                    with_offset = with_gain + tf.matmul(state, self.offset)
                else:
                    with_offset = (with_gain +
                                   tf.slice(self.offset, [0, 0], [1, -1]) +
                                   tf.matmul(state, tf.slice(self.offset, [1, 0], [-1, -1])))
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
        options = keyword.split('.')
        shape = pop_shape(options)
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
        options = keyword.split('.')
        shape = pop_shape(options)
        opts = {}
        for op in options:
            if op == 'm':
                opts['match_sign'] = True
        return StateHinge(shape=shape, **opts)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        match_sign = self.match_sign

        class StateHingeTF(NemsKerasLayer):

            def call(self, inputs):
                # inputs is [input, state], state second (state_arg = 'state')
                inp = inputs[0]
                state = inputs[1]

                gain1 = tf.convert_to_tensor(self.gain1, dtype=tf.float32)
                gain2 = tf.convert_to_tensor(self.gain2, dtype=tf.float32)
                offset1 = tf.convert_to_tensor(self.offset1, dtype=tf.float32)
                offset2 = tf.convert_to_tensor(self.offset2, dtype=tf.float32)
                x0 = tf.convert_to_tensor(self.x0, dtype=tf.float32)

                if match_sign:
                    gain2 = tf.abs(gain2) * tf.sign(gain1)
                    offset2 = tf.abs(offset2) * tf.sign(offset1)

                # state: (..., T, S), x0: (S, N)
                # s_: (..., T, S, N) — deviation from hinge point per state/output
                s_ = tf.expand_dims(state, -1) - x0
                sp = s_ * tf.cast(s_ > 0, tf.float32)
                sn = s_ * tf.cast(s_ < 0, tf.float32)

                # Contract S dim (axis -2 of sp/sn, axis 0 of gain).
                # Mirrors numpy: tensordot(sn, gain1, (1,0))[:,0,:]
                # Result: (..., T, N_sn, N_gain) → take index 0 of N_sn → (..., T, N_gain)
                g = (tf.einsum('...tsn,sm->...tnm', sn, gain1)[..., 0, :] +
                     tf.einsum('...tsn,sm->...tnm', sp, gain2)[..., 0, :])
                d = (tf.einsum('...tsn,sm->...tnm', sn, offset1)[..., 0, :] +
                     tf.einsum('...tsn,sm->...tnm', sp, offset2)[..., 0, :])

                return g * inp + d

        return StateHingeTF(self, **kwargs)


class HRTF(Layer):

    # TODO fix this.. currenty defaulting to aerd breaks pytest
    state_arg = 'state'  # azimuth, elevation, tilt, distance
    #state_arg = 'aerd'  # azimuth, elevation, tilt, distance

    def __init__(self, shape, channel0=0, apply_gain=True, **kwargs):
        """
        Docs TODO.
        :param shape : tuple of int
            if 1dim -- number of stimulus channels (assume 1 lag, 2 banks)
            if 2dim -- time lags X number of total stimulus channels (assume 2 banks)
            if 3dim -- time lags X stim channel count X number of banks
            channel per bank = # channels/ # banks (should be int)
        :param channel0: index of first channel of first source (permits additional
                         non-auditory channels either above or below acoustic channels)
        :param apply_gain: if False, additive (since stim could be in log space)
        :param kwargs:

        Examples
        --------
        stim = np.zeros((100,36))
        stim[25,3] = 1
        stim[50,25] = 2
        h = HRTF(shape=(1,36,2))

        TODO

        """
        require_shape(self, {'shape': shape}, minimum_ndim=1)

        if len(shape) == 1:
            shape = (1, shape[0])
        if len(shape) == 2:
            shape = (shape[0], shape[1], 2)
        self.speaker_count = shape[2]
        self.channels = int(shape[1]/shape[2])
        self.channel0 = channel0
        self.apply_gain = apply_gain

        # acausal filter
        filter_length = shape[0]
        pre_length = int(np.floor(filter_length/2)) - 1
        post_length = filter_length - pre_length - 1
        self.padding = [[pre_length, post_length]] + [[0, 0]]

        super().__init__(shape=shape, **kwargs)


    def initial_parameters(self):
        """Get initial values for `FIR.parameters`.

        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `FIR.shape`.
            Prior:  Normal(mean=0, sd=1/size)
            Bounds: (-np.inf, np.inf)

        Returns
        -------
        nems.layers.base.Phi

        """
        mean = np.full(shape=self.shape, fill_value=0.0)
        # sd = np.full(shape=self.shape, fill_value=1/np.prod(self.shape))
        sd = np.full(shape=self.shape, fill_value=1 / self.shape[0])
        # TODO: May be more appropriate to make this a hard requirement, but
        #       for now this should stop tiny filter sizes from causing errors.

        mm = int(self.shape[0]/2)
        for bb in range(self.speaker_count):
            mean[mm, (bb*self.channels):((bb+1)*self.channels), bb] = 1
        prior = Normal(mean, sd)
        bounds = (0, 2)
        coefficients = Parameter(name='coefficients', shape=self.shape, prior=prior,
                                 bounds=bounds)
        return Phi(coefficients)

    @property
    def coefficients(self):
        """Filter that will be convolved with input.

        Re-parameterized subclasses should overwrite this so that `evaluate`
        doesn't need to change.

        Returns
        -------
        coefficients : ndarray
            coefficients.shape = WeightChannels.shape

        """
        return self.parameters['coefficients'].values

    def evaluate(self, input):
        """Convolve `FIR.coefficients` with input."""

        # Flip rank, any other dimensions except time & number of outputs.
        coefficients = self._reshape_coefficients()

        # pad with zeros, select acoustic channels
        input_with_padding = np.pad(input[..., self.channel0:(self.channel0+self.shape[1])], self.padding)

        # Convolve each filter with the corresponding input channel.
        outputs = []
        n_filters = coefficients.shape[-1]
        for i in range(n_filters):
            y = scipy.signal.convolve(
                input_with_padding, coefficients[..., i], mode='valid'
            )
            outputs.append(y)

        # Concatenate on n_outputs axis
        output_ = np.stack(outputs, axis=-1)
        # output_ = time X <totalchans> X bankcount
        output_ = np.reshape(output_, [-1, self.channels, self.speaker_count, self.speaker_count])
        output_ = np.mean(output_, axis=2)

        output = input.copy()
        output[..., self.channel0:(self.channel0+self.shape[1])] = output_

        return output

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
        return HRTF(shape=shape)

    def as_tensorflow_layer(self, **kwargs):
        raise NotImplementedError("HRTF TF backend not yet implemented.")


class HRTFGainLayer(Layer):
    """Calculate pure spatial HRTF gains from location coordinates.

    This layer takes location data and produces spatial gains by:
    1. Parsing DLC coordinates into spatial positions for each source
    2. Performing bilinear interpolation on HRTF grid (dB)
    3. Outputting pure spatial gains in dB (distance handled by DistanceAttenuationLayer)

    Parameters
    ----------
    shape : tuple
        (azimuth_bins, elevation_bins, freq_bins, ears) for the HRTF grid
    location_ranges : dict, optional
        Spatial coordinate ranges. Defaults to azimuth: (-180, 180),
        elevation: (-90, 90) in degrees.
    """

    def __init__(self, location_ranges=None, num_sources=None, **kwargs):
        """Initialize HRTF gain calculation layer."""
        require_shape(self, kwargs, minimum_ndim=4)  # Need (az, el, freq, ears)

        # Handle num_sources that might be in kwargs from older saved models
        if 'num_sources' in kwargs:
            num_sources = kwargs.pop('num_sources') if num_sources is None else num_sources

        if location_ranges is None:
            location_ranges = {
                'azimuth': (-180.0, 180.0),
                'elevation': (-90.0, 90.0)
            }

        self.location_ranges = location_ranges
        self.num_sources = num_sources if num_sources is not None else 2  # Fixed for current work

        super().__init__(**kwargs)

    def initial_parameters(self):
        """Initialize HRTF gain parameters."""
        az_bins, el_bins, freq_bins, ears = self.shape
        gain_shape = (az_bins, el_bins, freq_bins, ears)

        # Initialize with 0 dB mean and 2 dB standard deviation
        gain_prior = Normal(mean=np.zeros(gain_shape), sd=2.0 * np.ones(gain_shape))
        gain_bounds = (-100.0, 100.0)  # dB bounds

        hrtf_gains = Parameter('hrtf_gains', shape=gain_shape,
                               prior=gain_prior, bounds=gain_bounds)

        return Phi(hrtf_gains)

    @layer('hrtfgain')
    def from_keyword(keyword):
        """Construct HRTFGainLayer from keyword string.

        Expected format: 'hrtfgain.{az_bins}x{el_bins}x{freq_bins}x{ears}'
        Examples:
        - 'hrtfgain.36x1x18x2' creates a 36x1x18x2 HRTF grid for spatial processing only
        """
        options = keyword.split('.')
        shape = pop_shape(options)

        if len(shape) != 4:
            raise ValueError(
                f"HRTFGainLayer requires 4D shape (az_bins, el_bins, freq_bins, ears), "
                f"got {shape}"
            )

        return HRTFGainLayer(shape=shape)

    def evaluate(self, dlc):
        """Calculate pure spatial HRTF gains from location coordinates.

        Distance values are ignored - use DistanceAttenuationLayer separately.

        Parameters
        ----------
        dlc : np.ndarray
            Shape: (batch, time, 4)
            Location data in [dist1, az1, dist2, az2] format

        Returns
        -------
        np.ndarray
            Pure spatial gains in dB: (batch, time, sources, freq_bins, ears)
        """
        # Get HRTF gains parameter (stored in dB)
        hrtf_gains_db = self.parameters['hrtf_gains'].values

        # Handle both 2D (time, 4) and 3D (batch, time, 4) inputs
        if dlc.ndim == 2:
            dlc = dlc[np.newaxis, ...]  # Add batch dimension: (1, time, 4)

        batch_size, time_steps = dlc.shape[:2]
        freq_bins, ears = hrtf_gains_db.shape[2], hrtf_gains_db.shape[3]

        # 1. PARSE DLC: (batch, time, 4) - ignore distances, extract azimuths only
        locations = np.zeros((batch_size, self.num_sources, time_steps, 3))

        # dlc format: [dist1, az1, dist2, az2]  locations[source][az, el, dist]
        # Source 1
        locations[:, 0, :, 0] = dlc[:, :, 1]  # azimuth1
        locations[:, 0, :, 1] = 0.0  # elevation (assume 0)
        # Skip distance - not used for spatial HRTF

        # Source 2
        locations[:, 1, :, 0] = dlc[:, :, 3]  # azimuth2
        locations[:, 1, :, 1] = 0.0  # elevation (assume 0)
        # Skip distance - not used for spatial HRTF

        # 3. BILINEAR INTERPOLATION ON HRTF GRID (dB)
        # Normalize spatial coordinates to grid indices
        az_min, az_max = self.location_ranges['azimuth']
        el_min, el_max = self.location_ranges['elevation']
        az_bins, el_bins = hrtf_gains_db.shape[0], hrtf_gains_db.shape[1]

        az_coords = (locations[..., 0] - az_min) / (az_max - az_min) * (az_bins - 1)

        # Handle elevation normalization (avoid division by zero for single elevation)
        el_range = el_max - el_min
        if el_range > 0:
            el_coords = (locations[..., 1] - el_min) / el_range * (el_bins - 1)
        else:
            el_coords = np.zeros_like(locations[..., 1])

        # Flatten coordinates for interpolation
        az_flat = az_coords.reshape(-1)
        el_flat = el_coords.reshape(-1)

        # Bilinear interpolation on HRTF grid (still in dB)
        interpolated_hrtf_db = self._bilinear_interpolation_binaural(
            hrtf_gains_db, az_flat, el_flat
        )

        # Reshape to match spatial dimensions
        hrtf_gains_interp_db = interpolated_hrtf_db.reshape(
            batch_size, self.num_sources, time_steps, freq_bins, ears
        )

        # 4. OUTPUT PURE SPATIAL GAINS (no distance attenuation)
        # Transpose to (batch, time, sources, freq_bins, ears) for output
        output = hrtf_gains_interp_db.transpose(0, 2, 1, 3, 4)

        # If input was 2D, remove batch dimension from output
        if output.shape[0] == 1:
            output = output[0]  # (time, sources, freq_bins, ears)

        return output

    def _bilinear_interpolation_binaural(self, grid, az_coords, el_coords):
        """Perform bilinear interpolation for both left and right ear gains."""
        left_gains = self._bilinear_interpolation(grid[..., 0], az_coords, el_coords)
        right_gains = self._bilinear_interpolation(grid[..., 1], az_coords, el_coords)
        return np.stack([left_gains, right_gains], axis=-1)

    def _bilinear_interpolation(self, grid, x_coords, y_coords):
        """Perform bilinear interpolation on a 2D grid."""
        # Clamp coordinates to valid range
        x_coords = np.clip(x_coords, 0, grid.shape[0] - 1)
        y_coords = np.clip(y_coords, 0, grid.shape[1] - 1)

        # Get integer coordinates and fractional parts
        x0 = np.floor(x_coords).astype(int)
        x1 = np.minimum(x0 + 1, grid.shape[0] - 1)
        y0 = np.floor(y_coords).astype(int)
        y1 = np.minimum(y0 + 1, grid.shape[1] - 1)

        xd = x_coords - x0
        yd = y_coords - y0

        # Get corner values
        c00 = grid[x0, y0]
        c01 = grid[x0, y1]
        c10 = grid[x1, y0]
        c11 = grid[x1, y1]

        # Bilinear interpolation
        xd = xd[:, np.newaxis]  # (n_points, 1)
        yd = yd[:, np.newaxis]  # (n_points, 1)

        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd

        return c0 * (1 - xd) + c1 * xd

    def as_tensorflow_layer(self, **kwargs):
        """Build TensorFlow equivalent of HRTFGainLayer (pure spatial gains, no distance)."""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        location_ranges = self.location_ranges
        num_sources = self.num_sources

        class HRTFGainLayerTF(NemsKerasLayer):
            def call(self, inputs):
                dlc = inputs  # (batch, time, 4)

                if len(dlc.shape) == 3:  # extra sample dim present
                    dlc = tf.expand_dims(dlc[0], 0)  # (1, time, 4)

                batch_size = tf.shape(dlc)[0]
                time_steps = tf.shape(dlc)[1]

                hrtf_gains = tf.convert_to_tensor(self.hrtf_gains, dtype=tf.float32)
                freq_bins = hrtf_gains.shape[2]
                ears = hrtf_gains.shape[3]

                # 1. PARSE DLC → azimuth coords, ignore distances (pure spatial)
                # locations: (batch, num_sources, time, 2) — [az, el]
                az1 = dlc[:, tf.newaxis, :, 1]  # (batch, 1, time)
                az2 = dlc[:, tf.newaxis, :, 3]  # (batch, 1, time)
                az = tf.concat([az1, az2], axis=1)  # (batch, 2, time)
                el = tf.zeros_like(az)              # elevation = 0

                # 2. NORMALIZE TO GRID INDICES
                az_min, az_max = location_ranges['azimuth']
                el_min, el_max = location_ranges['elevation']
                az_bins = tf.cast(tf.shape(hrtf_gains)[0], tf.float32)
                el_bins = tf.cast(tf.shape(hrtf_gains)[1], tf.float32)

                az_coords = (az - az_min) / (az_max - az_min) * (az_bins - 1)
                el_range = el_max - el_min
                if el_range > 0:
                    el_coords = (el - el_min) / el_range * (el_bins - 1)
                else:
                    el_coords = tf.zeros_like(el)

                # 3. BILINEAR INTERPOLATION — flatten, interpolate, reshape
                total = batch_size * num_sources * time_steps
                az_flat = tf.reshape(az_coords, [total])
                el_flat = tf.reshape(el_coords, [total])
                coords = tf.stack([az_flat, el_flat], axis=1)

                interpolated = _tf_bilinear_interpolation(hrtf_gains, coords)
                hrtf_interp = tf.reshape(
                    interpolated, [batch_size, num_sources, time_steps, freq_bins, ears]
                )

                # 4. TRANSPOSE to (batch, time, sources, freq_bins, ears)
                return tf.transpose(hrtf_interp, [0, 2, 1, 3, 4])

        def _tf_bilinear_interpolation(grid, coords):
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            coord_dtype = x_coords.dtype
            x_dim = tf.cast(tf.shape(grid)[0], coord_dtype)
            y_dim = tf.cast(tf.shape(grid)[1], coord_dtype)
            x_coords = tf.clip_by_value(x_coords, 0.0, x_dim - 1.0)
            y_coords = tf.clip_by_value(y_coords, 0.0, y_dim - 1.0)
            x0 = tf.cast(tf.floor(x_coords), tf.int32)
            x1 = tf.minimum(x0 + 1, tf.cast(x_dim - 1, tf.int32))
            y0 = tf.cast(tf.floor(y_coords), tf.int32)
            y1 = tf.minimum(y0 + 1, tf.cast(y_dim - 1, tf.int32))
            xd = tf.expand_dims(tf.expand_dims(x_coords - tf.cast(x0, coord_dtype), -1), -1)
            yd = tf.expand_dims(tf.expand_dims(y_coords - tf.cast(y0, coord_dtype), -1), -1)
            g0 = tf.gather(grid, x0, axis=0)
            g1 = tf.gather(grid, x1, axis=0)
            c00 = tf.gather(g0, y0, axis=1, batch_dims=1)
            c01 = tf.gather(g0, y1, axis=1, batch_dims=1)
            c10 = tf.gather(g1, y0, axis=1, batch_dims=1)
            c11 = tf.gather(g1, y1, axis=1, batch_dims=1)
            return (c00 * (1 - yd) + c01 * yd) * (1 - xd) + (c10 * (1 - yd) + c11 * yd) * xd

        return HRTFGainLayerTF(self, **kwargs)


# class HRTFGainLayerSinCos(Layer):
#     """Calculate spatial HRTF gains from sin/cos encoded location coordinates.
#
#     This layer is designed specifically for sin/cos angle representations to avoid
#     discontinuities at the -180/+180 degree boundary. It takes location data in
#     sin/cos format and produces spatial gains by:
#     1. Parsing DLC coordinates with sin/cos angle encoding
#     2. Calculating distance attenuation gains (dB)
#     3. Performing circular bilinear interpolation on HRTF grid (dB)
#     4. Adding distance + HRTF gains in dB space
#     5. Outputting combined gains in dB
#
#     Input format: [dist1, az1_sin, az1_cos, dist2, az2_sin, az2_cos] (6 channels)
#
#     Parameters
#     ----------
#     shape : tuple
#         (azimuth_bins, elevation_bins, freq_bins, ears) for the HRTF grid
#     location_ranges : dict, optional
#         Spatial coordinate ranges. Defaults to azimuth: (-180, 180),
#         elevation: (-90, 90) in degrees.
#     dist_atten : float, optional
#         Distance attenuation in dB. Default is 6 dB following stim_filt_hrtf.
#     """
#
#     def __init__(self, location_ranges=None, dist_atten=6.0, **kwargs):
#         """Initialize HRTF gain calculation layer for sin/cos format."""
#         require_shape(self, kwargs, minimum_ndim=4)  # Need (az, el, freq, ears)
#
#         if location_ranges is None:
#             location_ranges = {
#                 'azimuth': (-180.0, 180.0),
#                 'elevation': (-90.0, 90.0)
#             }
#
#         self.location_ranges = location_ranges
#         self.dist_atten = dist_atten
#         self.num_sources = 2  # Fixed for current work
#
#         super().__init__(**kwargs)
#
#     def initial_parameters(self):
#         """Initialize HRTF gain parameters."""
#         az_bins, el_bins, freq_bins, ears = self.shape
#         gain_shape = (az_bins, el_bins, freq_bins, ears)
#
#         # Initialize with 0 dB mean and 2 dB standard deviation
#         gain_prior = Normal(mean=np.zeros(gain_shape), sd=2.0 * np.ones(gain_shape))
#         gain_bounds = (-100.0, 100.0)  # dB bounds
#
#         hrtf_gains = Parameter('hrtf_gains', shape=gain_shape,
#                               prior=gain_prior, bounds=gain_bounds)
#
#         return Phi(hrtf_gains)
#
#     @layer('hrtfgainsincos')
#     def from_keyword(keyword):
#         """Construct HRTFGainLayerSinCos from keyword string.
#
#         Expected format: 'hrtfgainsincos.{az_bins}x{el_bins}x{freq_bins}x{ears}[.dist{dist_atten}]'
#         Examples:
#         - 'hrtfgainsincos.36x1x18x2' creates a 36x1x18x2 HRTF grid with default 6 dB attenuation
#         - 'hrtfgainsincos.36x1x18x2.dist3' creates same grid with 3 dB attenuation
#         """
#         options = keyword.split('.')
#         shape = pop_shape(options)
#
#         # Parse distance attenuation parameter
#         dist_atten = 6.0  # Default value
#         for option in options:
#             if option.startswith('dist'):
#                 dist_atten = float(option[4:])
#
#         if len(shape) != 4:
#             raise ValueError(
#                 f"HRTFGainLayerSinCos requires 4D shape (az_bins, el_bins, freq_bins, ears), "
#                 f"got {shape}"
#             )
#
#         return HRTFGainLayerSinCos(shape=shape, dist_atten=dist_atten)
#
#     def evaluate(self, dlc):
#         """Calculate HRTF gains from sin/cos location coordinates.
#
#         Parameters
#         ----------
#         dlc : np.ndarray
#             Shape: (batch, time, 6)
#             Location data in [dist1, az1_sin, az1_cos, dist2, az2_sin, az2_cos] format
#
#         Returns
#         -------
#         np.ndarray
#             Gains in dB: (batch, time, sources, freq_bins, ears)
#         """
#         # Get HRTF gains parameter (stored in dB)
#         hrtf_gains_db = self.parameters['hrtf_gains'].values
#
#         # Handle both 2D and 3D inputs
#         if dlc.ndim == 2:
#             dlc = dlc[np.newaxis, ...]  # Add batch dimension: (1, time, 6)
#
#         batch_size, time_steps, num_channels = dlc.shape
#         freq_bins, ears = hrtf_gains_db.shape[2], hrtf_gains_db.shape[3]
#
#         # Verify input format
#         if num_channels != 6:
#             raise ValueError(f"HRTFGainLayerSinCos expects 6 channels, got {num_channels}")
#
#         # 1. PARSE DLC: Sin/cos format [dist1, az1_sin, az1_cos, dist2, az2_sin, az2_cos]
#         sin_cos_data = np.zeros((batch_size, self.num_sources, time_steps, 3))  # [sin, cos, dist]
#
#         # Source 1
#         sin_cos_data[:, 0, :, 0] = dlc[:, :, 1]  # sin(azimuth1)
#         sin_cos_data[:, 0, :, 1] = dlc[:, :, 2]  # cos(azimuth1)
#         sin_cos_data[:, 0, :, 2] = dlc[:, :, 0]  # distance1
#
#         # Source 2
#         sin_cos_data[:, 1, :, 0] = dlc[:, :, 4]  # sin(azimuth2)
#         sin_cos_data[:, 1, :, 1] = dlc[:, :, 5]  # cos(azimuth2)
#         sin_cos_data[:, 1, :, 2] = dlc[:, :, 3]  # distance2
#
#         # 2. CALCULATE DISTANCE ATTENUATION (dB)
#         distance = sin_cos_data[..., 2:3]  # (batch, sources, time, 1)
#         distance_gain_db = -(distance - 1.0) * self.dist_atten
#
#         # 3. CIRCULAR BILINEAR INTERPOLATION ON HRTF GRID (dB)
#         sin_flat = sin_cos_data[..., 0].reshape(-1)  # sin values
#         cos_flat = sin_cos_data[..., 1].reshape(-1)  # cos values
#         el_flat = np.zeros_like(sin_flat)  # elevation = 0 for now
#
#         # Circular bilinear interpolation on HRTF grid (still in dB)
#         interpolated_hrtf_db = self._circular_bilinear_interpolation_binaural(
#             hrtf_gains_db, sin_flat, cos_flat, el_flat
#         )
#
#         # Reshape to match spatial dimensions
#         hrtf_gains_interp_db = interpolated_hrtf_db.reshape(
#             batch_size, self.num_sources, time_steps, freq_bins, ears
#         )
#
#         # 4. ADD DISTANCE + HRTF GAINS (dB space)
#         # Expand distance gains to match frequency and ear dimensions
#         distance_gain_db_expanded = np.broadcast_to(
#             distance_gain_db[..., np.newaxis, :],
#             (batch_size, self.num_sources, time_steps, freq_bins, ears)
#         )
#
#         combined_gains_db = hrtf_gains_interp_db + distance_gain_db_expanded
#
#         # Transpose to (batch, time, sources, freq_bins, ears) for output
#         output = combined_gains_db.transpose(0, 2, 1, 3, 4)
#
#         # If input was 2D, remove batch dimension from output
#         if output.shape[0] == 1:
#             output = output[0]  # (time, sources, freq_bins, ears)
#
#         return output
#
#     def _circular_bilinear_interpolation_binaural(self, grid, sin_coords, cos_coords, el_coords):
#         """Perform circular bilinear interpolation for sin/cos azimuth coordinates.
#
#         Parameters
#         ----------
#         grid : np.ndarray
#             Shape: (az_bins, el_bins, freq_bins, 2)
#         sin_coords : np.ndarray
#             Sin of azimuth coordinates (flattened)
#         cos_coords : np.ndarray
#             Cos of azimuth coordinates (flattened)
#         el_coords : np.ndarray
#             Elevation coordinates (flattened, currently unused - assumed 0)
#
#         Returns
#         -------
#         np.ndarray
#             Interpolated gains for both ears
#         """
#         left_gains = self._circular_bilinear_interpolation(grid[..., 0], sin_coords, cos_coords, el_coords)
#         right_gains = self._circular_bilinear_interpolation(grid[..., 1], sin_coords, cos_coords, el_coords)
#
#         return np.stack([left_gains, right_gains], axis=-1)
#
#     def _circular_bilinear_interpolation(self, grid, sin_coords, cos_coords, el_coords):
#         """Perform circular bilinear interpolation using sin/cos coordinates.
#
#         This method interpolates on the azimuth grid using the circular nature of angles,
#         avoiding discontinuities at the -180/+180 boundary.
#
#         Parameters
#         ----------
#         grid : np.ndarray
#             Shape: (az_bins, el_bins, freq_bins)
#         sin_coords : np.ndarray
#             Sin of azimuth coordinates
#         cos_coords : np.ndarray
#             Cos of azimuth coordinates
#         el_coords : np.ndarray
#             Elevation coordinates (currently assumed 0)
#
#         Returns
#         -------
#         np.ndarray
#             Interpolated values
#         """
#         az_bins, el_bins = grid.shape[0], grid.shape[1]
#
#         # Create azimuth angles for each grid point (assume uniform spacing)
#         az_min, az_max = self.location_ranges['azimuth']
#         grid_angles_deg = np.linspace(az_min, az_max, az_bins, endpoint=False)
#         grid_angles_rad = grid_angles_deg * np.pi / 180
#         grid_sin = np.sin(grid_angles_rad)
#         grid_cos = np.cos(grid_angles_rad)
#
#         interpolated_values = []
#
#         for i in range(len(sin_coords)):
#             sin_target = sin_coords[i]
#             cos_target = cos_coords[i]
#
#             # Calculate circular distances to all grid points on the unit circle
#             # Distance = 2 - 2*cos(angle_difference) = 2 - 2*(cos1*cos2 + sin1*sin2)
#             # But for computational efficiency, use squared Euclidean distance on unit circle
#             distances = (grid_sin - sin_target)**2 + (grid_cos - cos_target)**2
#
#             # Find the closest grid point
#             closest_idx = np.argmin(distances)
#
#             # Find the second closest for interpolation, considering circular nature
#             next_idx = (closest_idx + 1) % az_bins
#             prev_idx = (closest_idx - 1) % az_bins
#
#             next_dist = (grid_sin[next_idx] - sin_target)**2 + (grid_cos[next_idx] - cos_target)**2
#             prev_dist = (grid_sin[prev_idx] - sin_target)**2 + (grid_cos[prev_idx] - cos_target)**2
#
#             if next_dist < prev_dist:
#                 second_idx = next_idx
#                 second_dist = next_dist
#             else:
#                 second_idx = prev_idx
#                 second_dist = prev_dist
#
#             # Calculate interpolation weight based on circular distances
#             closest_dist = distances[closest_idx]
#             total_dist = closest_dist + second_dist
#
#             if total_dist > 1e-12:  # Avoid division by zero
#                 weight = second_dist / total_dist  # Weight for closest point
#             else:
#                 weight = 1.0  # If exactly on grid point, use that point
#
#             # Get values at the two closest azimuth points
#             # For now, assume elevation index 0 (can be extended later)
#             el_idx = int(np.clip(el_coords[i], 0, el_bins - 1))
#
#             val1 = grid[closest_idx, el_idx]  # Shape: (freq_bins,)
#             val2 = grid[second_idx, el_idx]   # Shape: (freq_bins,)
#
#             # Linear interpolation between the two closest points
#             interpolated_val = val1 * weight + val2 * (1 - weight)
#             interpolated_values.append(interpolated_val)
#
#         return np.array(interpolated_values)  # Shape: (n_points, freq_bins)
#
#     def as_tensorflow_layer(self, **kwargs):
#         """Build TensorFlow equivalent of HRTF gain layer for sin/cos format."""
#         import tensorflow as tf
#         from nems.backends.tf import NemsKerasLayer
#
#         location_ranges = self.location_ranges
#         dist_atten = self.dist_atten
#         num_sources = self.num_sources
#
#         class HRTFGainLayerSinCosTF(NemsKerasLayer):
#             def call(self, inputs):
#                 dlc = inputs  # (batch, time, 6)
#
#                 # Handle extra sample dimension during training
#                 if len(dlc.shape) == 3:  # Extra sample dimension present during training
#                     dlc = dlc[0]  # Remove sample dimension
#                     dlc = tf.expand_dims(dlc, 0)  # Make it (1, time, 6)
#
#                 batch_size = tf.shape(dlc)[0]
#                 time_steps = tf.shape(dlc)[1]
#
#                 # Verify input format
#                 static_channels = dlc.shape[-1]
#                 if static_channels != 6:
#                     raise ValueError(f"HRTFGainLayerSinCosTF expects 6 channels, got {static_channels}")
#
#                 # Access HRTF gains parameter - get the actual tensor values
#                 hrtf_gains = None
#                 for weight in self.weights:
#                     if 'hrtf_gains' in weight.name:
#                         hrtf_gains = weight
#                         break
#
#                 if hrtf_gains is None:
#                     raise ValueError("Could not find hrtf_gains weight in TensorFlow layer")
#
#                 freq_bins, ears = hrtf_gains.shape[2], hrtf_gains.shape[3]
#
#                 # 1. PARSE DLC: Sin/cos format [dist1, az1_sin, az1_cos, dist2, az2_sin, az2_cos]
#                 # Extract data directly without complex tensor manipulation
#                 sin1 = dlc[:, :, 1]  # (batch, time) - az1_sin
#                 cos1 = dlc[:, :, 2]  # (batch, time) - az1_cos
#                 dist1 = dlc[:, :, 0]  # (batch, time) - distance1
#
#                 sin2 = dlc[:, :, 4]  # (batch, time) - az2_sin
#                 cos2 = dlc[:, :, 5]  # (batch, time) - az2_cos
#                 dist2 = dlc[:, :, 3]  # (batch, time) - distance2
#
#                 # Stack sources: (batch, sources=2, time, features=3)
#                 sin_cos_data = tf.stack([
#                     tf.stack([sin1, cos1, dist1], axis=-1),  # source 1: [sin, cos, dist]
#                     tf.stack([sin2, cos2, dist2], axis=-1)   # source 2: [sin, cos, dist]
#                 ], axis=1)
#
#                 # 2. CALCULATE DISTANCE ATTENUATION (dB)
#                 distance = sin_cos_data[..., 2:3]  # (batch, sources, time, 1)
#                 distance_gain_db = -(distance - 1.0) * dist_atten
#
#                 # 3. TRUE CIRCULAR INTERPOLATION - work entirely in sin/cos space
#                 sin_flat = tf.reshape(sin_cos_data[..., 0], [-1])  # (batch*sources*time,)
#                 cos_flat = tf.reshape(sin_cos_data[..., 1], [-1])  # (batch*sources*time,)
#                 el_flat = tf.zeros_like(sin_flat)  # elevation = 0
#
#                 # Circular interpolation on HRTF grid (stays in dB)
#                 interpolated_hrtf_db = self._tf_circular_bilinear_interpolation_binaural(
#                     hrtf_gains, sin_flat, cos_flat, el_flat
#                 )
#
#                 # Reshape back to spatial dimensions
#                 hrtf_gains_interp_db = tf.reshape(
#                     interpolated_hrtf_db,
#                     [batch_size, num_sources, time_steps, freq_bins, ears]
#                 )
#
#                 # 4. ADD DISTANCE + HRTF GAINS (dB space)
#                 distance_gain_db_expanded = tf.broadcast_to(
#                     distance_gain_db[..., tf.newaxis, :],
#                     [batch_size, num_sources, time_steps, freq_bins, ears]
#                 )
#
#                 combined_gains_db = hrtf_gains_interp_db + distance_gain_db_expanded
#
#                 # Transpose to (batch, time, sources, freq_bins, ears)
#                 output = tf.transpose(combined_gains_db, [0, 2, 1, 3, 4])
#
#                 return output
#
#             def _tf_circular_bilinear_interpolation_binaural(self, grid, sin_coords, cos_coords, el_coords):
#                 """TensorFlow circular bilinear interpolation for both ears."""
#                 left_gains = self._tf_circular_bilinear_interpolation(grid[..., 0], sin_coords, cos_coords, el_coords)
#                 right_gains = self._tf_circular_bilinear_interpolation(grid[..., 1], sin_coords, cos_coords, el_coords)
#                 return tf.stack([left_gains, right_gains], axis=-1)
#
#             def _tf_circular_bilinear_interpolation(self, grid, sin_coords, cos_coords, el_coords):
#                 """True circular interpolation in TensorFlow using sin/cos coordinates.
#
#                 This method never reconstructs angles, working entirely in sin/cos space to avoid discontinuities.
#                 """
#                 az_bins, el_bins = grid.shape[0], grid.shape[1]
#                 n_points = tf.shape(sin_coords)[0]
#
#                 # Create azimuth grid points in sin/cos space (precomputed on unit circle)
#                 az_min, az_max = location_ranges['azimuth']
#                 grid_angles_deg = tf.linspace(az_min, az_max, az_bins)  # Exclude endpoint for circular
#                 grid_angles_rad = grid_angles_deg * tf.constant(3.14159265359 / 180.0)
#                 grid_sin = tf.sin(grid_angles_rad)  # (az_bins,)
#                 grid_cos = tf.cos(grid_angles_rad)  # (az_bins,)
#
#                 # Expand dimensions for broadcasting: (n_points, 1) vs (1, az_bins)
#                 sin_target = tf.expand_dims(sin_coords, 1)  # (n_points, 1)
#                 cos_target = tf.expand_dims(cos_coords, 1)  # (n_points, 1)
#                 grid_sin_exp = tf.expand_dims(grid_sin, 0)   # (1, az_bins)
#                 grid_cos_exp = tf.expand_dims(grid_cos, 0)   # (1, az_bins)
#
#                 # Calculate circular distances using broadcasting: (n_points, az_bins)
#                 # Distance = ||target - grid_point||² on unit circle
#                 distances = (grid_sin_exp - sin_target)**2 + (grid_cos_exp - cos_target)**2
#
#                 # Find closest azimuth bin for each point
#                 closest_idx = tf.argmin(distances, axis=1, output_type=tf.int32)  # (n_points,)
#
#                 # Find second closest considering circular nature
#                 next_idx = tf.math.mod(closest_idx + 1, az_bins)  # Wrap around
#                 prev_idx = tf.math.mod(closest_idx - 1, az_bins)  # Wrap around
#
#                 # Get distances to next and previous points
#                 point_indices = tf.range(n_points)
#                 next_distances = tf.gather_nd(distances, tf.stack([point_indices, next_idx], axis=1))
#                 prev_distances = tf.gather_nd(distances, tf.stack([point_indices, prev_idx], axis=1))
#
#                 # Choose second closest (next or prev)
#                 use_next = next_distances < prev_distances
#                 second_idx = tf.where(use_next, next_idx, prev_idx)
#                 second_distances = tf.where(use_next, next_distances, prev_distances)
#
#                 # Calculate interpolation weights based on circular distances
#                 closest_distances = tf.gather_nd(distances, tf.stack([point_indices, closest_idx], axis=1))
#                 total_distances = closest_distances + second_distances
#
#                 # Avoid division by zero
#                 safe_total = tf.where(total_distances > 1e-12, total_distances, tf.ones_like(total_distances))
#                 weights = tf.where(
#                     total_distances > 1e-12,
#                     second_distances / safe_total,  # Weight for closest point
#                     tf.ones_like(total_distances)   # If exactly on grid, use closest point fully
#                 )
#
#                 # Get HRTF values at the two closest azimuth points
#                 # For now, use elevation index 0 (can be extended)
#                 el_indices = tf.cast(tf.clip_by_value(el_coords, 0, el_bins - 1), tf.int32)
#
#                 # Gather values from grid: grid[closest_idx, el_idx, :]
#                 closest_indices = tf.stack([closest_idx, el_indices], axis=1)
#                 second_indices = tf.stack([second_idx, el_indices], axis=1)
#
#                 val1 = tf.gather_nd(grid, closest_indices)  # (n_points, freq_bins)
#                 val2 = tf.gather_nd(grid, second_indices)   # (n_points, freq_bins)
#
#                 # Linear interpolation between the two closest points
#                 weights_exp = tf.expand_dims(weights, 1)  # (n_points, 1) for broadcasting
#                 interpolated_vals = val1 * weights_exp + val2 * (1 - weights_exp)
#
#                 return interpolated_vals  # (n_points, freq_bins)
#
#         return HRTFGainLayerSinCosTF(self, **kwargs)


class HRTFGainLayerSinCos(Layer):
    """Calculate pure spatial HRTF gains from sin/cos encoded location coordinates.

    This layer maps sin/cos encoded angles to a learnable grid of spatial gains.
    Distance attenuation is handled by a separate DistanceAttenuationLayer.

    Input format: [dist1, az1_sin, az1_cos, dist2, az2_sin, az2_cos] (6 channels)
    Note: Distance values are ignored by this layer.

    Output format: (time, sources*ears*freq) = [S1_L_freqs, S1_R_freqs, S2_L_freqs, S2_R_freqs]

    Parameters
    ----------
    shape : tuple
        (azimuth_bins, elevation_bins, freq_bins, ears)
    location_ranges : dict, optional
        Spatial coordinate ranges. Defaults to azimuth: (-180, 180).
    num_sources : int
        Number of sound sources (default: 2)
    """

    def __init__(self, location_ranges=None, num_sources=2, **kwargs):
        require_shape(self, kwargs, minimum_ndim=4)  # (az, el, freq, ears)

        if location_ranges is None:
            location_ranges = {
                'azimuth': (-180.0, 180.0),
                'elevation': (-90.0, 90.0)
            }

        self.location_ranges = location_ranges
        self.num_sources = num_sources

        super().__init__(**kwargs)

        # Check if grid aligns with Front (0) and Back (180/-180) and log spacing
        self._check_grid_alignment()

    @property
    def grid_angles_deg(self):
        """Calculate the azimuth angles (in degrees) for the grid bins."""
        az_bins = self.shape[0]
        az_min, az_max = self.location_ranges['azimuth']
        # endpoint=False ensures we cover the circle evenly without duplicate poles
        return np.linspace(az_min, az_max, az_bins, endpoint=False)

    @property
    def grid_angles(self):
        """Calculate the azimuth angles (in radians) for the grid bins."""
        return np.deg2rad(self.grid_angles_deg)

    def _check_grid_alignment(self):
        """Log bin spacing and warn if grid bins are not centered on Front/Back."""
        angles = self.grid_angles_deg
        az_bins = self.shape[0]
        az_min, az_max = self.location_ranges['azimuth']

        # Calculate and log bin spacing
        if az_bins > 0:
            spacing = (az_max - az_min) / az_bins
            log.info(f"HRTFGainLayerSinCos: Grid initialized with {az_bins} bins. "
                     f"Bin spacing: {spacing:.2f} degrees.")

        # Tolerance for float comparison
        tol = 1e-5

        # Check Front (0 degrees)
        has_front = np.any(np.abs(angles - 0.0) < tol)

        # Check Back (180 or -180 degrees)
        # Note: In [-180, 180) with endpoint=False, -180 is included, 180 is not.
        has_back = np.any(np.abs(angles - 180.0) < tol) or \
                   np.any(np.abs(angles - (-180.0)) < tol)

        if not has_front:
            closest_front = angles[np.argmin(np.abs(angles))]
            log.warning(
                f"HRTFGainLayerSinCos: Grid does NOT have a bin centered at 0 degrees (Front). "
                f"Current closest bin: {closest_front:.2f}. "
                f"For range [-180, 180), ensure az_bins is EVEN."
            )

        if not has_back:
            log.warning(
                f"HRTFGainLayerSinCos: Grid does NOT have a bin centered at +/- 180 degrees (Back). "
                f"This may cause issues with circular interpolation at the rear."
            )

    def initial_parameters(self):
        """Initialize HRTF gain parameters."""
        az_bins, el_bins, freq_bins, ears = self.shape
        gain_shape = (az_bins, el_bins, freq_bins, ears)

        # Flat Random Initialization (mean=0, sd=2)
        # Allows model to learn spatial tuning from scratch
        gain_prior = Normal(mean=np.zeros(gain_shape), sd=2.0 * np.ones(gain_shape))
        gain_bounds = (-100.0, 100.0)  # dB bounds

        hrtf_gains = Parameter('hrtf_gains', shape=gain_shape,
                               prior=gain_prior, bounds=gain_bounds)

        return Phi(hrtf_gains)

    @layer('hrtfgainsincos')
    def from_keyword(keyword):
        options = keyword.split('.')
        shape = pop_shape(options)

        return HRTFGainLayerSinCos(shape=shape)

    def evaluate(self, dlc):
        """Calculate pure spatial HRTF gains from sin/cos location coordinates.

        Distance values are ignored - use DistanceAttenuationLayer separately.

        Returns
        -------
        output : ndarray
            (time, sources*ears*freq) = [S1_L_freqs, S1_R_freqs, S2_L_freqs, S2_R_freqs]
        """
        hrtf_gains_db = self.parameters['hrtf_gains'].values

        if dlc.ndim == 2:
            dlc = dlc[np.newaxis, ...]

        batch_size, time_steps, _ = dlc.shape
        freq_bins, ears = hrtf_gains_db.shape[2], hrtf_gains_db.shape[3]

        # 1. Parse DLC: [dist1, sin1, cos1, dist2, sin2, cos2] - ignore distances
        sin1, cos1 = dlc[..., 1], dlc[..., 2]  # Source 1 angles
        sin2, cos2 = dlc[..., 4], dlc[..., 5]  # Source 2 angles

        # Stack: (batch, sources, time, features)
        # features: 0=sin, 1=cos (ignore distance)
        source_data = np.stack([
            np.stack([sin1, cos1], axis=-1),
            np.stack([sin2, cos2], axis=-1)
        ], axis=1)  # -> (B, 2, T, 2)

        # 2. HRTF Interpolation (no distance attenuation)
        # Flatten for vectorized operation
        sin_flat = source_data[..., 0].reshape(-1)
        cos_flat = source_data[..., 1].reshape(-1)
        # Elevation assumed 0 for now
        el_flat = np.zeros_like(sin_flat)

        interpolated_hrtf_db = self._circular_bilinear_interpolation_binaural(
            hrtf_gains_db, sin_flat, cos_flat, el_flat
        )

        # Reshape back
        hrtf_gains_interp_db = interpolated_hrtf_db.reshape(
            batch_size, self.num_sources, time_steps, freq_bins, ears
        )

        # Output: (batch, time, sources, freq, ears) -> flatten to (time, sources*ears*freq)
        output = hrtf_gains_interp_db.transpose(0, 2, 1, 3, 4)

        if output.shape[0] == 1:
            output = output[0]
        if output.ndim == 3:
            output = output[np.newaxis, ...]

        t_steps, sources, f_bins, n_ears = output.shape
        output = output.transpose(0, 1, 3, 2).reshape(t_steps, -1)

        return output

    def _circular_bilinear_interpolation_binaural(self, grid, sin_coords, cos_coords, el_coords):
        left = self._circular_bilinear_interpolation(grid[..., 0], sin_coords, cos_coords, el_coords)
        right = self._circular_bilinear_interpolation(grid[..., 1], sin_coords, cos_coords, el_coords)
        return np.stack([left, right], axis=-1)

    def _circular_bilinear_interpolation(self, grid, sin_coords, cos_coords, el_coords):
        """Vectorized circular interpolation."""
        az_bins, el_bins, freq_bins = grid.shape
        n_points = sin_coords.shape[0]

        # Get Grid in Sin/Cos
        grid_rad = self.grid_angles
        grid_sin = np.sin(grid_rad)
        grid_cos = np.cos(grid_rad)

        # --- Vectorized Distance Calculation ---
        # dist^2 = (sin_t - sin_g)^2 + (cos_t - cos_g)^2
        # Expanding: sin_t^2 + sin_g^2 - 2sin_t*sin_g + ... = 2 - 2(sin_t*sin_g + cos_t*cos_g)

        # Inputs: (N, 1), Grid: (1, M)
        dot_prod = (sin_coords[:, np.newaxis] * grid_sin[np.newaxis, :]) + \
                   (cos_coords[:, np.newaxis] * grid_cos[np.newaxis, :])

        # Distances (squared Euclidean on unit circle)
        distances = 2.0 - 2.0 * dot_prod

        # Find closest indices
        closest_idx = np.argmin(distances, axis=1)  # (N,)

        # Find second closest (neighbors in circular buffer)
        next_idx = (closest_idx + 1) % az_bins
        prev_idx = (closest_idx - 1) % az_bins

        # Gather distances to next and prev
        rows = np.arange(n_points)
        dist_next = distances[rows, next_idx]
        dist_prev = distances[rows, prev_idx]

        # Choose neighbor
        use_next = dist_next < dist_prev
        second_idx = np.where(use_next, next_idx, prev_idx)
        dist_second = np.where(use_next, dist_next, dist_prev)

        # Calculate weights
        total_dist = distances[rows, closest_idx] + dist_second

        # Avoid div by zero (if exactly on point, total_dist approx 0)
        mask_nonzero = total_dist > 1e-12
        weight = np.ones(n_points)  # Default to 1 (takes closest)
        weight[mask_nonzero] = dist_second[mask_nonzero] / total_dist[mask_nonzero]

        # Interpolate
        el_idx = np.clip(el_coords, 0, el_bins - 1).astype(int)

        val1 = grid[closest_idx, el_idx]
        val2 = grid[second_idx, el_idx]

        w = weight[:, np.newaxis]
        return val1 * w + val2 * (1.0 - w)

    def as_tensorflow_layer(self, **kwargs):
        """Build TensorFlow equivalent."""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        location_ranges = self.location_ranges
        num_sources = self.num_sources

        class HRTFGainLayerSinCosTF(NemsKerasLayer):
            def call(self_tf, inputs):
                dlc = inputs
                if len(dlc.shape) == 3:
                    dlc = dlc[0]
                    dlc = tf.expand_dims(dlc, 0)

                batch_size = tf.shape(dlc)[0]
                time_steps = tf.shape(dlc)[1]

                hrtf_gains = None
                for weight in self_tf.weights:
                    if 'hrtf_gains' in weight.name:
                        hrtf_gains = weight
                        break
                if hrtf_gains is None:
                    raise ValueError("Could not find hrtf_gains weight")

                freq_bins, ears = hrtf_gains.shape[2], hrtf_gains.shape[3]

                # Parse DLC: [dist1, sin1, cos1, dist2, sin2, cos2] - ignore distances
                sin1, cos1 = dlc[:, :, 1], dlc[:, :, 2]  # Source 1 angles
                sin2, cos2 = dlc[:, :, 4], dlc[:, :, 5]  # Source 2 angles

                sin_cos_data = tf.stack([
                    tf.stack([sin1, cos1], axis=-1),
                    tf.stack([sin2, cos2], axis=-1)
                ], axis=1)

                sin_flat = tf.reshape(sin_cos_data[..., 0], [-1])
                cos_flat = tf.reshape(sin_cos_data[..., 1], [-1])
                el_flat = tf.zeros_like(sin_flat)

                interpolated_hrtf_db = self_tf._tf_circular_bilinear_interpolation_binaural(
                    hrtf_gains, sin_flat, cos_flat, el_flat
                )

                hrtf_gains_interp_db = tf.reshape(
                    interpolated_hrtf_db,
                    [batch_size, num_sources, time_steps, freq_bins, ears]
                )

                # Output: (batch, time, sources, freq, ears) -> flatten to (batch, time, sources*ears*freq)
                output = tf.transpose(hrtf_gains_interp_db, [0, 2, 1, 3, 4])
                output = tf.transpose(output, [0, 1, 2, 4, 3])
                output = tf.reshape(output, [batch_size, time_steps, -1])

                return output

            def _tf_circular_bilinear_interpolation_binaural(self, grid, sin_coords, cos_coords, el_coords):
                left = self._tf_circular_bilinear_interpolation(grid[..., 0], sin_coords, cos_coords, el_coords)
                right = self._tf_circular_bilinear_interpolation(grid[..., 1], sin_coords, cos_coords, el_coords)
                return tf.stack([left, right], axis=-1)

            def _tf_circular_bilinear_interpolation(self, grid, sin_coords, cos_coords, el_coords):
                az_bins = grid.shape[0]
                az_min, az_max = location_ranges['azimuth']

                # TF equivalent of linspace(..., endpoint=False)
                step = (az_max - az_min) / tf.cast(az_bins, tf.float32)
                grid_angles_deg = tf.linspace(az_min, az_max - step, az_bins)

                grid_rad = grid_angles_deg * (3.14159265359 / 180.0)
                grid_sin = tf.sin(grid_rad)
                grid_cos = tf.cos(grid_rad)

                sin_t = tf.expand_dims(sin_coords, 1)
                cos_t = tf.expand_dims(cos_coords, 1)
                grid_sin_e = tf.expand_dims(grid_sin, 0)
                grid_cos_e = tf.expand_dims(grid_cos, 0)

                distances = (grid_sin_e - sin_t) ** 2 + (grid_cos_e - cos_t) ** 2
                closest_idx = tf.argmin(distances, axis=1, output_type=tf.int32)

                next_idx = tf.math.mod(closest_idx + 1, az_bins)
                prev_idx = tf.math.mod(closest_idx - 1, az_bins)

                point_indices = tf.range(tf.shape(sin_coords)[0])

                # Gather distances using indices
                closest_dist = tf.gather_nd(distances, tf.stack([point_indices, closest_idx], axis=1))
                next_dist = tf.gather_nd(distances, tf.stack([point_indices, next_idx], axis=1))
                prev_dist = tf.gather_nd(distances, tf.stack([point_indices, prev_idx], axis=1))

                use_next = next_dist < prev_dist
                second_idx = tf.where(use_next, next_idx, prev_idx)
                second_dist = tf.where(use_next, next_dist, prev_dist)

                total_dist = closest_dist + second_dist
                safe_total = tf.where(total_dist > 1e-12, total_dist, tf.ones_like(total_dist))
                weights = tf.where(total_dist > 1e-12, second_dist / safe_total, tf.ones_like(total_dist))

                el_indices = tf.cast(tf.clip_by_value(el_coords, 0, tf.cast(grid.shape[1] - 1, tf.float32)), tf.int32)

                idx1 = tf.stack([closest_idx, el_indices], axis=1)
                idx2 = tf.stack([second_idx, el_indices], axis=1)

                val1 = tf.gather_nd(grid, idx1)
                val2 = tf.gather_nd(grid, idx2)

                w = tf.expand_dims(weights, 1)
                return val1 * w + val2 * (1.0 - w)

        return HRTFGainLayerSinCosTF(self, **kwargs)


# Backwards-compatible alias for saved models
class HRTFGainLayerSinCosFlattened(HRTFGainLayerSinCos):
    pass


class HRTFGainLayerMLP(Layer):
    """
    Calculates pure spatial HRTF gains from sin/cos encoded angles using a 3-layer MLP.

    This layer uses a three-layer MLP to learn a direct, non-linear mapping
    from the sin/cos of the azimuth angle to frequency- and ear-specific gains.
    Distance attenuation is handled by a separate DistanceAttenuationLayer.

    Input format: [dist1, az1_sin, az1_cos, dist2, az2_sin, az2_cos] (6 channels)
    Note: Distance values are ignored by this layer.

    Parameters
    ----------
    shape : tuple
        (freq_bins, ears) for the output gains.
    hidden_units : list of int
        Number of units in each hidden layer of the MLP.
    """
    def __init__(self, hidden_units=(32, 16), output_scale=20.0, output_bias=-10.0,
                 num_sources=2, **kwargs):
        require_shape(self, kwargs, minimum_ndim=2)  # (freq_bins, ears)

        # Set attributes before calling super().__init__() since initial_parameters() needs them
        self.hidden_units = hidden_units
        self.output_scale = output_scale  # Scale tanh(-1,1) to dB range
        self.output_bias = output_bias    # Center the dB range
        self.num_sources = num_sources

        super().__init__(**kwargs)

        # CRITICAL FIX: Initialize parameter values from priors
        # NEMS only auto-initializes when priors are passed via kwargs, not when defined in initial_parameters()
        # This ensures parameters have actual values instead of remaining as zero arrays
        self.parameters.sample(inplace=True)  # Sample from priors for random initialization

    @layer('hrtfmlp')
    def from_keyword(keyword):
        """
        Construct HRTFGainLayerMLP from a keyword string.

        Expected format: 'hrtfmlp.{freq_bins}x{ears}.h{h1}[.h{h2}][.dist{dist_atten}]'
        Examples:
        - 'hrtfmlp.18x2.h100' -> 2-layer MLP (100 hidden units)
        - 'hrtfmlp.18x2.h100.h50' -> 3-layer MLP (100 and 50 hidden units)
        """
        options = keyword.split('.')
        shape = pop_shape(options)
        if len(shape) != 2:
            raise ValueError(f"HRTFGainLayerMLP requires 2D shape (freq_bins, ears), got {shape}")

        hidden_units = [int(opt[1:]) for opt in options if opt.startswith('h')]
        if not hidden_units:
            hidden_units = [32, 16]

        output_scale = 20.0
        output_bias = -10.0
        for option in options:
            if option.startswith('scale'):
                output_scale = float(option[5:])
            elif option.startswith('bias'):
                output_bias = float(option[4:])

        return HRTFGainLayerMLP(shape=shape, hidden_units=tuple(hidden_units),
                               output_scale=output_scale, output_bias=output_bias)

    def initial_parameters(self):
        """Initialize MLP weights and biases using WeightChannels pattern for proper sample_from_priors()."""
        input_size = 2  # sin(angle), cos(angle)
        freq_bins, ears = self.shape
        output_size = freq_bins * ears
        layer_sizes = [input_size] + list(self.hidden_units) + [output_size]

        params = []
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]

            # Weight parameters - following WeightChannels pattern exactly
            w_shape = (in_size, out_size)
            w_mean = np.full(shape=w_shape, fill_value=0.01)  # Small positive mean like WeightChannels
            w_sd = np.full(shape=w_shape, fill_value=0.1)     # Same sd as WeightChannels
            w_prior = Normal(w_mean, w_sd)
            w_param = Parameter(name=f'w{i+1}', shape=w_shape, prior=w_prior)
            params.append(w_param)

            # Bias parameters - following WeightChannels pattern exactly
            b_shape = (out_size,)
            b_mean = np.full(shape=b_shape, fill_value=0.01)  # Small positive mean
            b_sd = np.full(shape=b_shape, fill_value=0.1)     # Same sd as WeightChannels
            b_prior = Normal(b_mean, b_sd)
            b_param = Parameter(name=f'b{i+1}', shape=b_shape, prior=b_prior)
            params.append(b_param)

        return Phi(*params)

    def evaluate(self, dlc):
        """Calculate HRTF gains using the MLP in NumPy."""
        if dlc.ndim == 2:
            dlc = dlc[np.newaxis, ...]

        batch_size, time_steps, _ = dlc.shape
        freq_bins, ears = self.shape

        # 1. Parse DLC input
        sin1, cos1, dist1 = dlc[..., 1], dlc[..., 2], dlc[..., 0]
        sin2, cos2, dist2 = dlc[..., 4], dlc[..., 5], dlc[..., 3]

        X1 = np.stack([sin1.ravel(), cos1.ravel()], axis=1)
        X2 = np.stack([sin2.ravel(), cos2.ravel()], axis=1)

        # 2. MLP Forward Pass
        params = self.get_parameter_values()
        num_layers = len(self.hidden_units) + 1

        def forward_pass(X):
            h = X
            for i in range(num_layers):
                # Parameters are stored as [w1, b1, w2, b2, ..., wN, bN]
                w = params[i*2]      # weights: params[0], params[2], params[4], ...
                b = params[i*2 + 1]  # biases: params[1], params[3], params[5], ...

                if i < num_layers - 1:
                    h = np.tanh(h @ w + b)  # Hidden layers with tanh
                else:
                    h = h @ w + b  # Linear output layer (no tanh) to match TensorFlow
            return h

        gains1_flat = forward_pass(X1)  # (batch*time, freq*ears)
        gains2_flat = forward_pass(X2)  # (batch*time, freq*ears)

        # Reshape back to (batch, time, freq*ears)
        gains1 = gains1_flat.reshape(batch_size, time_steps, freq_bins * ears)
        gains2 = gains2_flat.reshape(batch_size, time_steps, freq_bins * ears)

        # No distance attenuation - pure spatial HRTF learning
        # Stack the two sources: (batch, time, 2*freq*ears)
        output = np.concatenate([gains1, gains2], axis=-1)

        return output[0] if output.shape[0] == 1 else output

    def as_tensorflow_layer(self, **kwargs):
        """Build TensorFlow equivalent of the MLP HRTF gain layer."""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        # Capture parent layer attributes as closure variables
        freq_bins, ears = self.shape
        hidden_units = self.hidden_units

        # COMMENTED OUT OLD IMPLEMENTATION - KEEPING FOR REFERENCE
        # class HRTFGainLayerMLPTF_OLD(NemsKerasLayer):
        #     def call(self, inputs):
        #         dlc = inputs
        #         if len(dlc.shape) == 3:
        #             dlc = dlc[0]
        #             dlc = tf.expand_dims(dlc, 0)
        #         batch_size, time_steps = tf.shape(dlc)[0], tf.shape(dlc)[1]
        #         sin1, cos1, dist1 = dlc[:, :, 1], dlc[:, :, 2], dlc[:, :, 0]
        #         sin2, cos2, dist2 = dlc[:, :, 4], dlc[:, :, 5], dlc[:, :, 3]
        #         X1 = tf.stack([tf.reshape(sin1, [-1]), tf.reshape(cos1, [-1])], axis=1)
        #         X2 = tf.stack([tf.reshape(sin2, [-1]), tf.reshape(cos2, [-1])], axis=1)
        #         def forward_pass(X):
        #             h = X
        #             num_layers = len(hidden_units) + 1
        #             for i in range(num_layers):
        #                 w = getattr(self, f'w{i+1}')
        #                 b = getattr(self, f'b{i+1}')
        #                 if i < num_layers - 1:
        #                     h = tf.nn.tanh(tf.matmul(h, w) + b)
        #                 else:
        #                     h = tf.matmul(h, w) + b
        #             return h
        #         gains1_flat = forward_pass(X1)
        #         gains2_flat = forward_pass(X2)
        #         gains1 = tf.reshape(gains1_flat, [batch_size, time_steps, freq_bins * ears])
        #         gains2 = tf.reshape(gains2_flat, [batch_size, time_steps, freq_bins * ears])
        #         flattened = tf.concat([gains1, gains2], axis=-1)
        #         return flattened

        # CLEAN IMPLEMENTATION - Uses proper NemsKerasLayer pattern
        class HRTFGainLayerMLPTF(NemsKerasLayer):
            # No custom build() method needed! NemsKerasLayer handles parameter creation
            # from the NEMS layer's parameters automatically

            def call(self, inputs):
                dlc = inputs
                if len(dlc.shape) == 3:
                    dlc = dlc[0]
                    dlc = tf.expand_dims(dlc, 0)

                batch_size, time_steps = tf.shape(dlc)[0], tf.shape(dlc)[1]

                # Parse DLC input - extract sin/cos for each source
                sin1, cos1 = dlc[:, :, 1], dlc[:, :, 2]  # Skip distance channels
                sin2, cos2 = dlc[:, :, 4], dlc[:, :, 5]

                # Create source inputs: (batch*time, 2)
                source1_input = tf.stack([tf.reshape(sin1, [-1]), tf.reshape(cos1, [-1])], axis=1)
                source2_input = tf.stack([tf.reshape(sin2, [-1]), tf.reshape(cos2, [-1])], axis=1)

                # Sequential MLP processing (NO LOOPS - explicit operations)
                def process_source(source_input):
                    # Layer 1: Input -> Hidden1 (with tanh)
                    h1 = tf.nn.tanh(tf.matmul(source_input, self.w1) + self.b1)

                    # Layer 2: Hidden1 -> Hidden2 (with tanh)
                    h2 = tf.nn.tanh(tf.matmul(h1, self.w2) + self.b2)

                    # Layer 3: Hidden2 -> Hidden3 (with tanh)
                    h3 = tf.nn.tanh(tf.matmul(h2, self.w3) + self.b3)

                    # Output: Hidden3 -> Output (linear, no activation)
                    output = tf.matmul(h3, self.w4) + self.b4

                    return output

                # Process each source through the SAME MLP (sequential processing)
                gains1 = process_source(source1_input)  # (batch*time, freq_bins*ears)
                gains2 = process_source(source2_input)  # (batch*time, freq_bins*ears)

                # Simple concatenation: [S1_all_freqs, S2_all_freqs] - keep this simple format
                flattened = tf.concat([gains1, gains2], axis=1)  # (batch*time, freq_bins*ears*2)

                # Reshape back to NEMS format: (batch, time, freq_bins*4)
                final_output = tf.reshape(flattened, [batch_size, time_steps, freq_bins * 4])

                return final_output

        return HRTFGainLayerMLPTF(self, **kwargs)


class HRTFGainLayerMLPReg(Layer):
    """
    Calculates pure spatial HRTF gains from sin/cos encoded angles using a 3-layer MLP,
    with biologically-motivated regularization constraints.

    This is a copy of HRTFGainLayerMLP with added regularization for:
    - Spectral smoothness (smooth frequency response)
    - Azimuthal smoothness (smooth spatial variation)
    - Bilateral symmetry (HRTF_L(θ) ≈ HRTF_R(-θ))
    - Head shadow constraint (small ILD at low frequencies)
    - Frontal ILD constraint (ILD ≈ 0 at 0° and 180°)
    - Gain range bounds (-20 to +5 dB typical)
    - HRTF prior (penalize divergence from generic/measured HRTF)

    Input format: [dist1, az1_sin, az1_cos, dist2, az2_sin, az2_cos] (6 channels)
    Note: Distance values are ignored by this layer.

    Parameters
    ----------
    shape : tuple
        (freq_bins, ears) for the output gains.
    hidden_units : list of int
        Number of units in each hidden layer of the MLP.
    hrtf_regularization : dict, optional
        Regularization weights. Keys:
        - 'freq_smooth': penalize frequency gradients (default 0.0)
        - 'az_smooth': penalize azimuthal gradients (default 0.0)
        - 'symmetry': penalize L/R asymmetry (default 0.0)
        - 'ild_lowfreq': penalize ILD at low frequencies (default 0.0)
        - 'ild_frontal': penalize ILD at 0°/180° (default 0.0)
        - 'range': penalize gains outside -20 to +5 dB (default 0.0)
        - 'hrtf_prior': penalize divergence from generic HRTF (default 0.0)
    az_samples : int
        Number of azimuth samples for computing regularization (default 36)
    freq_centers : array-like, optional
        Center frequencies in Hz for frequency-weighted regularization
    generic_hrtf_params : dict, optional
        Parameters to load generic ferret HRTF for prior regularization.
        Keys: 'f_min', 'f_max'. If provided and hrtf_prior > 0, the ferret
        HRTF will be loaded and sampled at az_samples azimuths.
    generic_hrtf_table : ndarray, optional
        Pre-computed generic HRTF table with shape (az_samples, freq_bins, 2).
        If provided, this is used directly instead of loading from ferret data.
    """
    def __init__(self, hidden_units=(32, 16), output_scale=20.0, output_bias=-10.0,
                 num_sources=2, hrtf_regularization=None, az_samples=36,
                 freq_centers=None, generic_hrtf_params=None, generic_hrtf_table=None,
                 reg_config=None,  # Ignored - for backwards compatibility with saved models
                 **kwargs):
        require_shape(self, kwargs, minimum_ndim=2)  # (freq_bins, ears)

        # Extract shape before super().__init__() since _load_generic_hrtf needs it
        self.shape = kwargs.get('shape')

        # Set attributes before calling super().__init__() since initial_parameters() needs them
        self.hidden_units = hidden_units
        self.output_scale = output_scale  # Scale tanh(-1,1) to dB range
        self.output_bias = output_bias    # Center the dB range
        self.num_sources = num_sources
        self.az_samples = az_samples
        self.freq_centers = freq_centers
        self.generic_hrtf_params = generic_hrtf_params

        # Regularization config with defaults
        self.hrtf_regularization = hrtf_regularization or {}
        self.reg_config = {
            'freq_smooth': self.hrtf_regularization.get('freq_smooth', 0.0),
            'az_smooth': self.hrtf_regularization.get('az_smooth', 0.0),
            'symmetry': self.hrtf_regularization.get('symmetry', 0.0),
            'ild_lowfreq': self.hrtf_regularization.get('ild_lowfreq', 0.0),
            'ild_frontal': self.hrtf_regularization.get('ild_frontal', 0.0),
            'range': self.hrtf_regularization.get('range', 0.0),
            'hrtf_prior': self.hrtf_regularization.get('hrtf_prior', 0.0),
        }

        # Store or generate generic HRTF table for prior regularization
        self.generic_hrtf_table = None
        if generic_hrtf_table is not None:
            # Use provided table directly
            self.generic_hrtf_table = np.array(generic_hrtf_table, dtype=np.float32)
        elif self.reg_config['hrtf_prior'] > 0 and generic_hrtf_params is not None:
            # Load ferret HRTF and sample at uniform azimuths
            self.generic_hrtf_table = self._load_generic_hrtf(
                generic_hrtf_params.get('f_min', 200),
                generic_hrtf_params.get('f_max', 20000)
            )

        super().__init__(**kwargs)

        # CRITICAL FIX: Initialize parameter values from priors
        self.parameters.sample(inplace=True)

    def _load_generic_hrtf(self, f_min, f_max):
        """
        Load generic ferret HRTF and sample at uniform azimuths.

        Returns
        -------
        hrtf_table : ndarray
            Shape (az_samples, freq_bins, 2) with left/right ear gains in dB
        """
        from scipy import interpolate
        try:
            from nems_lbhb.free_tools import load_hrtf
        except ImportError:
            log.warning("Could not import load_hrtf from nems_lbhb.free_tools. "
                        "HRTF prior regularization will be disabled.")
            return None

        freq_bins = self.shape[0]

        # Load ground truth HRTF
        L0, R0, c, A = load_hrtf(format='az', fmin=f_min, fmax=f_max, num_freqs=freq_bins)
        # L0, R0 shape: (freq_bins, n_azimuths)
        # A: azimuth angles in degrees

        # Create interpolators
        f_left = interpolate.interp1d(A, L0, axis=1, kind='linear', fill_value='extrapolate')
        f_right = interpolate.interp1d(A, R0, axis=1, kind='linear', fill_value='extrapolate')

        # Sample at uniform azimuths (same as used for regularization)
        az_deg = np.linspace(-180, 180, self.az_samples, endpoint=False)

        # Interpolate HRTF at these azimuths
        L_interp = f_left(az_deg)  # (freq_bins, az_samples)
        R_interp = f_right(az_deg)  # (freq_bins, az_samples)

        # Reshape to (az_samples, freq_bins, 2)
        hrtf_table = np.stack([L_interp.T, R_interp.T], axis=-1).astype(np.float32)

        log.info(f"Loaded generic HRTF for prior regularization: shape {hrtf_table.shape}")

        return hrtf_table

    @layer('hrtfmlpreg')
    def from_keyword(keyword):
        """
        Construct HRTFGainLayerMLPReg from a keyword string.

        Expected format: 'hrtfmlpreg.{freq_bins}x{ears}.h{h1}[.h{h2}][.reg{weight}]'
        Examples:
        - 'hrtfmlpreg.18x2.h100' -> 2-layer MLP (100 hidden units)
        - 'hrtfmlpreg.18x2.h100.h50.reg01' -> with regularization weight 0.01
        """
        options = keyword.split('.')
        shape = pop_shape(options)
        if len(shape) != 2:
            raise ValueError(f"HRTFGainLayerMLPReg requires 2D shape (freq_bins, ears), got {shape}")

        hidden_units = [int(opt[1:]) for opt in options if opt.startswith('h')]
        if not hidden_units:
            hidden_units = [32, 16]

        output_scale = 20.0
        output_bias = -10.0
        reg_weight = 0.0
        for option in options:
            if option.startswith('scale'):
                output_scale = float(option[5:])
            elif option.startswith('bias'):
                output_bias = float(option[4:])
            elif option.startswith('reg'):
                reg_weight = float('0.' + option[3:])

        # Apply uniform regularization if specified
        hrtf_regularization = None
        if reg_weight > 0:
            hrtf_regularization = {
                'freq_smooth': reg_weight,
                'az_smooth': reg_weight,
                'symmetry': reg_weight * 0.5,
                'ild_lowfreq': reg_weight,
                'ild_frontal': reg_weight * 0.5,
                'range': reg_weight * 0.1,
            }

        return HRTFGainLayerMLPReg(shape=shape, hidden_units=tuple(hidden_units),
                                    output_scale=output_scale, output_bias=output_bias,
                                    hrtf_regularization=hrtf_regularization)

    def initial_parameters(self):
        """Initialize MLP weights and biases using WeightChannels pattern for proper sample_from_priors()."""
        input_size = 2  # sin(angle), cos(angle)
        freq_bins, ears = self.shape
        output_size = freq_bins * ears
        layer_sizes = [input_size] + list(self.hidden_units) + [output_size]

        params = []
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]

            # Weight parameters - following WeightChannels pattern exactly
            w_shape = (in_size, out_size)
            w_mean = np.full(shape=w_shape, fill_value=0.01)
            w_sd = np.full(shape=w_shape, fill_value=0.1)
            w_prior = Normal(w_mean, w_sd)
            w_param = Parameter(name=f'w{i+1}', shape=w_shape, prior=w_prior)
            params.append(w_param)

            # Bias parameters
            b_shape = (out_size,)
            b_mean = np.full(shape=b_shape, fill_value=0.01)
            b_sd = np.full(shape=b_shape, fill_value=0.1)
            b_prior = Normal(b_mean, b_sd)
            b_param = Parameter(name=f'b{i+1}', shape=b_shape, prior=b_prior)
            params.append(b_param)

        return Phi(*params)

    def evaluate(self, dlc):
        """Calculate HRTF gains using the MLP in NumPy."""
        if dlc.ndim == 2:
            dlc = dlc[np.newaxis, ...]

        batch_size, time_steps, _ = dlc.shape
        freq_bins, ears = self.shape

        # 1. Parse DLC input
        sin1, cos1, dist1 = dlc[..., 1], dlc[..., 2], dlc[..., 0]
        sin2, cos2, dist2 = dlc[..., 4], dlc[..., 5], dlc[..., 3]

        X1 = np.stack([sin1.ravel(), cos1.ravel()], axis=1)
        X2 = np.stack([sin2.ravel(), cos2.ravel()], axis=1)

        # 2. MLP Forward Pass
        params = self.get_parameter_values()
        num_layers = len(self.hidden_units) + 1

        def forward_pass(X):
            h = X
            for i in range(num_layers):
                w = params[i*2]
                b = params[i*2 + 1]

                if i < num_layers - 1:
                    h = np.tanh(h @ w + b)
                else:
                    h = h @ w + b
            return h

        gains1_flat = forward_pass(X1)
        gains2_flat = forward_pass(X2)

        gains1 = gains1_flat.reshape(batch_size, time_steps, freq_bins * ears)
        gains2 = gains2_flat.reshape(batch_size, time_steps, freq_bins * ears)

        output = np.concatenate([gains1, gains2], axis=-1)

        return output[0] if output.shape[0] == 1 else output

    def as_tensorflow_layer(self, **kwargs):
        """Build TensorFlow equivalent of the MLP HRTF gain layer with regularization."""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        # Capture parent layer attributes as closure variables
        freq_bins, ears = self.shape
        hidden_units = self.hidden_units
        reg_config = self.reg_config
        az_samples = self.az_samples
        freq_centers = self.freq_centers
        generic_hrtf_table = self.generic_hrtf_table  # Pre-computed generic HRTF for prior regularization

        class HRTFGainLayerMLPRegTF(NemsKerasLayer):

            def call(self, inputs):
                dlc = inputs
                if len(dlc.shape) == 3:
                    dlc = dlc[0]
                    dlc = tf.expand_dims(dlc, 0)

                batch_size, time_steps = tf.shape(dlc)[0], tf.shape(dlc)[1]

                # Parse DLC input - extract sin/cos for each source
                sin1, cos1 = dlc[:, :, 1], dlc[:, :, 2]
                sin2, cos2 = dlc[:, :, 4], dlc[:, :, 5]

                # Create source inputs: (batch*time, 2)
                source1_input = tf.stack([tf.reshape(sin1, [-1]), tf.reshape(cos1, [-1])], axis=1)
                source2_input = tf.stack([tf.reshape(sin2, [-1]), tf.reshape(cos2, [-1])], axis=1)

                # Sequential MLP processing
                def process_source(source_input):
                    # Layer 1: Input -> Hidden1 (with tanh)
                    h1 = tf.nn.tanh(tf.matmul(source_input, self.w1) + self.b1)
                    # Layer 2: Hidden1 -> Hidden2 (with tanh)
                    h2 = tf.nn.tanh(tf.matmul(h1, self.w2) + self.b2)
                    # Layer 3: Hidden2 -> Hidden3 (with tanh)
                    h3 = tf.nn.tanh(tf.matmul(h2, self.w3) + self.b3)
                    # Output: Hidden3 -> Output (linear, no activation)
                    output = tf.matmul(h3, self.w4) + self.b4
                    return output

                # Process each source through the SAME MLP
                gains1 = process_source(source1_input)
                gains2 = process_source(source2_input)

                # Add regularization losses (computed on sampled HRTF)
                if any(v > 0 for v in reg_config.values()):
                    self._add_hrtf_regularization()

                # Simple concatenation: [S1_all_freqs, S2_all_freqs]
                flattened = tf.concat([gains1, gains2], axis=1)

                # Reshape back to NEMS format: (batch, time, freq_bins*4)
                final_output = tf.reshape(flattened, [batch_size, time_steps, freq_bins * 4])

                return final_output

            def _add_hrtf_regularization(self):
                """
                Add biologically-motivated HRTF regularization losses.

                Since MLP doesn't have an explicit lookup table, we sample the MLP
                at uniform azimuths to compute regularization on the implicit HRTF.
                """
                # Sample MLP at uniform azimuths to get implicit HRTF table
                # az_samples uniform angles from -pi to pi (endpoint=False for circular continuity)
                az_rad = tf.linspace(-3.14159265359, 3.14159265359, az_samples + 1)[:-1]
                sin_az = tf.sin(az_rad)
                cos_az = tf.cos(az_rad)

                # Create input for MLP: (az_samples, 2)
                az_input = tf.stack([sin_az, cos_az], axis=1)

                # Forward pass through MLP to get HRTF at each azimuth
                h1 = tf.nn.tanh(tf.matmul(az_input, self.w1) + self.b1)
                h2 = tf.nn.tanh(tf.matmul(h1, self.w2) + self.b2)
                h3 = tf.nn.tanh(tf.matmul(h2, self.w3) + self.b3)
                hrtf_flat = tf.matmul(h3, self.w4) + self.b4  # (az_samples, freq_bins*ears)

                # Reshape to (az_samples, freq_bins, ears)
                hrtf_table = tf.reshape(hrtf_flat, [az_samples, freq_bins, ears])

                # Precompute front/back indices from sampled angles
                front_idx = tf.argmin(tf.abs(az_rad))
                back_idx_pos = tf.argmin(tf.abs(az_rad - 3.14159265359))
                back_idx_neg = tf.argmin(tf.abs(az_rad + 3.14159265359))
                back_indices = tf.stack([back_idx_pos, back_idx_neg], axis=0)

                # 1. Spectral smoothness: penalize sharp frequency gradients
                if reg_config['freq_smooth'] > 0:
                    freq_grad = hrtf_table[:, 1:, :] - hrtf_table[:, :-1, :]
                    loss_freq_raw = tf.reduce_mean(tf.square(freq_grad))
                    loss_freq = reg_config['freq_smooth'] * loss_freq_raw
                    self.add_loss(loss_freq)
                    self.add_metric(loss_freq_raw, name='hrtf_freq_smooth')

                # 2. Azimuthal smoothness: penalize sharp angular gradients (circular)
                if reg_config['az_smooth'] > 0:
                    az_grad = tf.roll(hrtf_table, -1, axis=0) - hrtf_table
                    loss_az_raw = tf.reduce_mean(tf.square(az_grad))
                    loss_az = reg_config['az_smooth'] * loss_az_raw
                    self.add_loss(loss_az)
                    self.add_metric(loss_az_raw, name='hrtf_az_smooth')

                # 3. Bilateral symmetry: HRTF_L(θ) ≈ HRTF_R(-θ)
                if reg_config['symmetry'] > 0:
                    left_ear = hrtf_table[:, :, 0]   # (az, freq)
                    right_ear = hrtf_table[:, :, 1]  # (az, freq)
                    # Flip azimuth for comparison (θ → -θ)
                    right_flipped = tf.reverse(right_ear, axis=[0])
                    sym_error = left_ear - right_flipped
                    loss_sym_raw = tf.reduce_mean(tf.square(sym_error))
                    loss_sym = reg_config['symmetry'] * loss_sym_raw
                    self.add_loss(loss_sym)
                    self.add_metric(loss_sym_raw, name='hrtf_symmetry')

                # 4. Low-frequency ILD penalty (head shadow physics)
                # Low frequencies diffract around head → ILD should be small
                if reg_config['ild_lowfreq'] > 0:
                    left_ear = hrtf_table[:, :, 0]
                    right_ear = hrtf_table[:, :, 1]
                    ild = left_ear - right_ear  # (az, freq)

                    # Weight: penalize more at low frequencies
                    if freq_centers is not None and len(freq_centers) == freq_bins:
                        freq_centers_tf = tf.constant(freq_centers, dtype=tf.float32)
                        fmin = tf.reduce_min(freq_centers_tf)
                        fmax = tf.reduce_max(freq_centers_tf)
                        freq_weight = (fmax - freq_centers_tf) / tf.maximum(fmax - fmin, 1e-6)
                    else:
                        freq_weight = tf.linspace(1.0, 0.0, freq_bins)
                    weighted_ild = ild * freq_weight[tf.newaxis, :]
                    loss_ild_raw = tf.reduce_mean(tf.square(weighted_ild))
                    loss_ild = reg_config['ild_lowfreq'] * loss_ild_raw
                    self.add_loss(loss_ild)
                    self.add_metric(loss_ild_raw, name='hrtf_ild_lowfreq')

                # 5. Frontal ILD ≈ 0 at 0° and 180°
                if reg_config['ild_frontal'] > 0:
                    left_ear = hrtf_table[:, :, 0]
                    right_ear = hrtf_table[:, :, 1]
                    ild = left_ear - right_ear

                    # Gather frontal ILDs at computed front/back indices
                    frontal_ild = tf.gather(ild, tf.concat([[front_idx], back_indices], axis=0), axis=0)
                    loss_frontal_raw = tf.reduce_mean(tf.square(frontal_ild))
                    loss_frontal = reg_config['ild_frontal'] * loss_frontal_raw
                    self.add_loss(loss_frontal)
                    self.add_metric(loss_frontal_raw, name='hrtf_ild_frontal')

                # 6. Gain range penalty (-20 to +5 dB typical)
                if reg_config['range'] > 0:
                    below_min = tf.nn.relu(-20.0 - hrtf_table)
                    above_max = tf.nn.relu(hrtf_table - 5.0)
                    loss_range_raw = (
                        tf.reduce_mean(tf.square(below_min)) +
                        tf.reduce_mean(tf.square(above_max))
                    )
                    loss_range = reg_config['range'] * loss_range_raw
                    self.add_loss(loss_range)
                    self.add_metric(loss_range_raw, name='hrtf_range')

                # 7. HRTF prior: penalize divergence from generic/measured HRTF
                if reg_config['hrtf_prior'] > 0 and generic_hrtf_table is not None:
                    # generic_hrtf_table shape: (az_samples, freq_bins, 2)
                    # hrtf_table shape: (az_samples, freq_bins, ears)
                    generic_hrtf_tf = tf.constant(generic_hrtf_table, dtype=tf.float32)
                    prior_error = hrtf_table - generic_hrtf_tf
                    loss_prior_raw = tf.reduce_mean(tf.square(prior_error))
                    loss_prior = reg_config['hrtf_prior'] * loss_prior_raw
                    self.add_loss(loss_prior)
                    self.add_metric(loss_prior_raw, name='hrtf_prior')

        return HRTFGainLayerMLPRegTF(self, **kwargs)


class DistanceAttenuationLayer(Layer):
    """Calculate distance attenuation from location coordinates.

    This layer extracts distance information from location coordinates and applies
    distance-based attenuation, independent of spatial HRTF processing.

    Input format: [dist1, sin1, cos1, dist2, sin2, cos2] (6 channels)
    Output format: (batch, time, sources, ears) - distance gains in dB

    Parameters
    ----------
    dist_atten : float, optional
        Distance attenuation in dB per unit distance. Default is 6.0 dB.
    num_sources : int, optional
        Number of audio sources. Default is 2.
    """

    def __init__(self, dist_atten=6.0, num_sources=2, **kwargs):
        self.dist_atten = dist_atten
        self.num_sources = num_sources
        super().__init__(**kwargs)

    @layer('distanceatten')
    def from_keyword(keyword):
        """Construct DistanceAttenuationLayer from keyword.

        Expected format: 'distanceatten[.dist{dist_atten}]'
        Examples:
        - 'distanceatten' creates layer with default 6 dB attenuation
        - 'distanceatten.dist3' creates layer with 3 dB attenuation
        """
        options = keyword.split('.')

        dist_atten = 6.0
        for option in options:
            if option.startswith('dist'):
                dist_atten = float(option[4:])

        return DistanceAttenuationLayer(dist_atten=dist_atten)

    def initial_parameters(self):
        """No learnable parameters for distance attenuation."""
        return Phi()

    def evaluate(self, dlc):
        """Calculate distance attenuation from location coordinates.

        Parameters
        ----------
        dlc : np.ndarray
            Shape: (batch, time, 6)
            Location data in [dist1, sin1, cos1, dist2, sin2, cos2] format

        Returns
        -------
        np.ndarray
            Distance gains in dB: (batch, time, sources, ears)
        """
        if dlc.ndim == 2:
            dlc = dlc[np.newaxis, ...]  # Add batch dimension

        batch_size, time_steps, _ = dlc.shape

        # Extract distances: [dist1, sin1, cos1, dist2, sin2, cos2]
        dist1 = dlc[..., 0]  # (batch, time)
        dist2 = dlc[..., 3]  # (batch, time)

        # Stack distances: (batch, time, sources)
        distances = np.stack([dist1, dist2], axis=2)

        # Calculate distance attenuation for each source
        # Formula: -(distance - 1.0) * dist_atten
        distance_gain_db = -(distances - 1.0) * self.dist_atten

        # Expand for ears dimension (same attenuation for both ears)
        # (batch, time, sources) -> (batch, time, sources, ears)
        distance_gains = np.stack([distance_gain_db, distance_gain_db], axis=3)

        # Remove batch dimension if input was 2D
        if distance_gains.shape[0] == 1:
            distance_gains = distance_gains[0]  # (time, sources, ears)

        return distance_gains

    def as_tensorflow_layer(self, **kwargs):
        """Build TensorFlow equivalent of distance attenuation layer."""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        dist_atten = self.dist_atten
        num_sources = self.num_sources

        class DistanceAttenuationLayerTF(NemsKerasLayer):
            def call(self, inputs):
                dlc = inputs
                if len(dlc.shape) == 3:
                    dlc = dlc[0]
                    dlc = tf.expand_dims(dlc, 0)

                batch_size, time_steps = tf.shape(dlc)[0], tf.shape(dlc)[1]

                # Extract distances: [dist1, sin1, cos1, dist2, sin2, cos2]
                dist1 = dlc[:, :, 0]  # (batch, time)
                dist2 = dlc[:, :, 3]  # (batch, time)

                # Stack distances: (batch, time, sources)
                distances = tf.stack([dist1, dist2], axis=2)

                # Calculate distance attenuation
                distance_gain_db = -(distances - 1.0) * dist_atten

                # Expand for ears dimension (same attenuation for both ears)
                distance_gains = tf.stack([distance_gain_db, distance_gain_db], axis=3)

                return distance_gains

        return DistanceAttenuationLayerTF(self, **kwargs)
