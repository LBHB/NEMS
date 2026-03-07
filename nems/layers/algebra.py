import numpy as np

from nems.registry import layer
from nems.distributions import Normal, HalfNormal
from .base import Layer, Phi, Parameter
from .tools import require_shape, pop_shape
from .weight_channels import WeightChannels

class SwapDims(Layer):
    """Swap two dimensions of an input array.

    See also
    --------
    nems.layers.base.Layer

    Examples
    --------
    >>> sd = SwapDims(dim1=1, dim2=2)
    >>> input = np.random.rand(10000, 18, 2)  # (time, channels, bands)
    >>> out = np.moveaxis(input, self.dim1, self.dim2)      # sd.evaluate(input)
    >>> out.shape
    (10000, 2, 18)

    """
 
    def __init__(self, dim1=1, dim2=2, **kwargs):
        self.dim1 = dim1
        self.dim2 = dim2
        super().__init__(**kwargs)

    def evaluate(self, input):
        """Swap two dimensions of the input."""
        return np.moveaxis(input, [self.dim1, self.dim2], [self.dim2, self.dim1])

    @layer('sd')
    def from_keyword(keyword):
        """Construct SwapDims from keyword.

        Keyword options
        ---------------
        {digit} : First dimension to swap; optional, default=1.
        {digit} : Second dimension to swap; optional, default=2.

        See also
        --------
        Layer.from_keyword
        
        """

        options = keyword.split('.')[1:3]
        kwargs = {f'dim{i+1}': d for i, d in enumerate(options)}
        return SwapDims(**kwargs)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs."""
        
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer
  
        dim1 = self.dim1 + 1
        dim2 = self.dim2 + 1
    
        class SwapDimsTF(NemsKerasLayer):
            @tf.function
            def call(self, inputs):
                out = tf.experimental.numpy.moveaxis(
                    inputs, [dim1, dim2], [dim2, dim1]
                    )
                return out

        return SwapDimsTF(self, **kwargs)

    @property
    def plot_options(self):
        """Add legend at right of plot, with default formatting.

        Notes
        -----
        The legend will grow quite large if there are many output channels,
        but for common use cases (< 10) this should not be an issue. If needed,
        increase figsize to accomodate the labels.

        See also
        --------
        Layer.plot
        
        """
        return {'legend': True}

class ConcatSkip(WeightChannels):

    def __init__(self, axis=1, input=None, **kwargs):

        if input is None:
            raise ValueError('value for input required')
        require_shape(self, kwargs, minimum_ndim=2)

        self.axis = axis
        super().__init__(input=input, **kwargs)

    @layer('skip')
    def from_keyword(keyword):
        """Construct ConcatSkip from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape; required.
        l2{value} : L2 regularizer, e.g. 'l2e-3'.
        {name} : Any other option string is added as an input signal name.

        """

        options = keyword.split('.')

        kwargs = {'input': []}
        kwargs['shape'] = pop_shape(options)
        for op in options[1:]:
            if op.startswith('l2'):
                kwargs['regularizer'] = op
            else:
                kwargs['input'].append(op)

        return ConcatSkip(**kwargs)

    def initial_parameters(self):
        """Get initial values for `ConcatSkip.parameters`.
        
        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `self.shape`.
            Prior:  zero +/ 0.1
            Bounds: TODO

        Returns
        -------
        nems.layers.base.Phi

        """
        mean = np.full(shape=self.shape, fill_value=0.0)
        sd = np.full(shape=self.shape, fill_value=0.1)
        prior = Normal(mean, sd)
        coefficients = Parameter(
            name='coefficients', shape=self.shape, prior=prior
            )
        return Phi(coefficients)
        
    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.
        stepsize = int(self.coefficients.shape[1]/(len(inputs)-1))
        inputs_=[inputs[0]] + [
            np.tensordot(inp, self.coefficients[:inp.shape[1],(i*stepsize):((i+1)*stepsize)], axes=(1, 0))
            for i, inp in enumerate(inputs[1:])
        ]
        return np.concatenate(inputs_, axis=self.axis)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        # TODO: how to deal with batch dimension. Currently, kludged to add 1 to axis
        ax = self.axis+1
        stepsize = int(self.coefficients.shape[1]/(len(self.input)-1))

        class ConcatSkipTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.

                inputs_ = [inputs[0]] + [
                    tf.tensordot(inp, self.coefficients[:inp.shape[2], (i*stepsize):((i+1)*stepsize)], axes=[[2], [0]])
                    for i,inp in enumerate(inputs[1:])]
                return tf.concat(inputs_, ax)

        return ConcatSkipTF(self, **kwargs)


class ConcatSignals(Layer):

    def __init__(self, input1='stim', input2='hrtf', axis=1, input=None, compute_sum=False, **kwargs):
        self.axis = axis
        if input is not None:
            if len(input)>0:
                self.input1 = input[0]
            if len(input)>1:
                self.input2 = input[1]
        else:
            self.input1 = input1
            self.input2 = input2
            input = [input1, input2]
        self.compute_sum = compute_sum
        super().__init__(input=input, **kwargs)

    @layer('cat')
    def from_keyword(keyword):
        """Construct ConcatSignals from keyword.

        Keyword options
        ---------------
        s : Sum non-primary inputs along concat axis before concatenating
            (compute_sum=True).
        {name} : Any other option string is added as an input signal name.

        """
        options = keyword.split('.')
        kwargs={'input': []}
        for op in options[1:]:
            if op == 's':
                kwargs['compute_sum'] = True
            else:
                kwargs['input'].append(op)

        return ConcatSignals(**kwargs)

    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.

        if self.compute_sum:
            inputs_=[inputs[0]] + [i.sum(axis=self.axis, keepdims=True) for i in inputs[1:]]
            return np.concatenate(inputs_, axis=self.axis)
        else:
            return np.concatenate(inputs, axis=self.axis)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        # TODO: how to deal with batch dimension. Currently, kludged to add 1 to axis
        ax = self.axis+1
        compute_sum = self.compute_sum
        class ConcatSignalsTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.
                if compute_sum:
                    inputs_ = [inputs[0]] + [tf.math.reduce_sum(i, axis=ax, keepdims=True) for i in inputs[1:]]
                    return tf.concat(inputs_, ax)
                else:
                    return tf.concat(inputs, ax)

        return ConcatSignalsTF(self, **kwargs)


class MultiplySignals(Layer):

    def __init__(self, input1='hrtf', input2='input', input=None, **kwargs):
        self.input1 = input1
        self.input2 = input2

        super().__init__(input=[input1, input2], **kwargs)

    @layer('mult')
    def from_keyword(keyword):
        """Construct MultiplySignals from keyword.

        Keyword options
        ---------------
        {name1} : First input signal name (default 'hrtf').
        {name2} : Second input signal name (default 'input').
        Output is set to 'hstim'.

        """
        options = keyword.split('.')
        kwargs={}
        if len(options)>1:
            kwargs['input1']=options[1]
        if len(options)>2:
            kwargs['input2']=options[2]
        kwargs['output']='hstim'
        return MultiplySignals(**kwargs)

        
    def evaluate(self, input1, input2):
        # All inputs are treated the same, no fittable parameters.
        return input1*input2

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class MultiplySignalsTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.
                return tf.math.multiply(inputs[0], inputs[1])

        return MultiplySignalsTF(self, **kwargs)


class ApplyHRTF(Layer):

    def __init__(self, input1='hrtf', input2='stim', input=None, **kwargs):
        input = [input1, input2]
        super().__init__(input=input, **kwargs)

    @layer('mult')
    def from_keyword(keyword):
        """Construct ApplyHRTF from keyword.

        Keyword options
        ---------------
        {name1} : First input signal name (default 'hrtf').
        {name2} : Second input signal name (default 'stim').
        Output is set to 'hstim'.

        Note: this shares the 'mult' keyword with MultiplySignals;
        whichever class is imported last will be active in the registry.

        """
        options = keyword.split('.')
        kwargs = {}
        if len(options) > 1:
            kwargs['input1'] = options[1]
        if len(options) > 2:
            kwargs['input2'] = options[2]
        kwargs['output'] = 'hstim'
        return ApplyHRTF(**kwargs)

    def evaluate(self, input1, input2):
        # All inputs are treated the same, no fitable parameters.
        s = list(input1.shape)
        if s[-2]==2:
            #s = s[:-3] + [s[-3]*s[-2], s[-1]]
            #x = np.reshape(input1, s) * input2[..., np.newaxis]
            x = np.concatenate((input1[...,0,:],input1[...,1,:]),axis=-2) * input2[..., np.newaxis]
        else:
            x = input1 * input2[..., np.newaxis]

        m = int(x.shape[-2]/2)
        x = x[..., :m, :] + x[..., m:, :]
        x = np.reshape(np.swapaxes(x, -1, -2), [x.shape[0], x.shape[1]*x.shape[2]])

        return x

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class ApplyHRTFTF(NemsKerasLayer):
            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.
                # inputs = [hrtf, stim] hrtf
                m = int(tf.shape(inputs[1])[-1] / 2)

                # add power
                #x = tf.math.multiply(inputs[0], tf.math.square(tf.expand_dims(inputs[1], -1)))
                #x = tf.math.sqrt(tf.nn.relu(tf.math.add(x[...,:m,:], x[...,m:,:])))

                # apply to input channels separately, then stack
                s = list(inputs[0].shape)
                if s[-2]==2:
                    #s = [-1] + s[1:-3] + [s[-3]*s[-2], s[-1]]
                    #in0 = tf.reshape(inputs[0], s)
                    in0 = tf.concat((inputs[0][...,0,:],inputs[0][...,1,:]), axis=-2)
                    x = tf.math.multiply(in0, tf.expand_dims(inputs[1], -1))
                else:
                    x = tf.math.multiply(inputs[0], tf.expand_dims(inputs[1], -1))
                x = tf.math.add(x[..., :m, :], x[..., m:, :])

                x_shape = tf.shape(x)
                x = tf.reshape(tf.transpose(x, [0, 1, 3, 2]),
                               [-1, x_shape[1], tf.reduce_prod(x_shape[2:])])
                return x

        return ApplyHRTFTF(self, **kwargs)


class MultiplyByExp(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @layer('multexp')
    def from_keyword(keyword):
        """Construct MultiplySignals from keyword."""
        return MultiplyByExp()

    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.
        return inputs[0] * np.exp(inputs[1])

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class MultiplyByExpTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.
                return tf.math.multiply(inputs[0], tf.math.exp(inputs[1]))

        return MultiplyByExpTF(self, **kwargs)


class ApplyHRTFGainLayer(Layer):
    """Apply separate HRTF and distance gains to spectrograms and generate binaural output.

    This layer takes separate spatial HRTF gains and distance attenuation gains plus stimulus data
    to produce binaural audio by:
    1. Combining HRTF and distance gains in dB space
    2. Converting combined gains to linear scale
    3. Reshaping stimulus from stacked format (time, freq*ears*sources) to (time, sources, freq, ears)
    4. Applying linear gains to each source spectrogram
    5. RMS combining across sources for each ear
    6. Concatenating ears as [right, left] output

    Expected stimulus format: [S1_Left_freqs, S1_Right_freqs, S2_Left_freqs, S2_Right_freqs, ...]

    Parameters
    ----------
    freq_bins : int
        Number of frequency channels per ear per source
    num_sources : int, optional
        Number of audio sources. Default is 2.
    """

    def __init__(self, freq_bins=18, num_sources=None, input=None, **kwargs):
        """Initialize HRTF gain application layer."""

        # Handle num_sources that might be in kwargs from older saved models
        if 'num_sources' in kwargs:
            num_sources = kwargs.pop('num_sources') if num_sources is None else num_sources

        self.freq_bins = freq_bins
        self.num_sources = num_sources if num_sources is not None else 2

        # Default to expecting 3 inputs: [hrtf_gains, distance_gains, stim]
        if input is None:
            input = ['hrtf_gains', 'distance_gains', 'stim']

        super().__init__(input=input, **kwargs)

    @layer('applyhrtf')
    def from_keyword(keyword):
        """Construct ApplyHRTFGainLayer from keyword string.

        Expected format: 'applyhrtf[.{freq_bins}]'
        Examples:
        - 'applyhrtf' creates layer with default 18 frequency bins
        - 'applyhrtf.36' creates layer with 36 frequency bins
        """
        options = keyword.split('.')

        # Parse frequency bins (default=18)
        freq_bins = 18
        if len(options) > 1 and options[1].isdigit():
            freq_bins = int(options[1])

        return ApplyHRTFGainLayer(freq_bins=freq_bins)

    def evaluate(self, hrtf_gains, distance_gains, stim):
        """Apply separate HRTF and distance gains to stimulus spectrograms.

        Parameters
        ----------
        hrtf_gains : np.ndarray
            Shape: (batch, time, sources*2*freq_bins)
            Concatenated HRTF gains: [S1_Left_freqs, S1_Right_freqs, S2_Left_freqs, S2_Right_freqs, ...]
        distance_gains : np.ndarray
            Shape: (batch, time, sources, ears)
            Distance attenuation gains in dB
        stim : np.ndarray
            Shape: (batch, time, num_sources * freq_bins)
            Stacked source spectrograms: [S1_freqs, S2_freqs, ...]

        Returns
        -------
        np.ndarray
            Binaural output: (batch, time, freq_bins*2) = [right_freqs, left_freqs]
        """
        # Handle both 2D and 3D inputs
        if stim.ndim == 2:
            stim = stim[np.newaxis, ...]  # Add batch dimension: (1, time, channels)
        if hrtf_gains.ndim == 2:
            hrtf_gains = hrtf_gains[np.newaxis, ...]  # Add batch dimension
        # distance_gains can be 2D (time, sources) or 3D (time, sources, ears)
        if distance_gains.ndim == 2:
            distance_gains = distance_gains[np.newaxis, ...]  # Add batch dimension
        elif distance_gains.ndim == 3:
            # DistanceAttenuationLayer outputs (time, sources, ears), add batch dim
            distance_gains = distance_gains[np.newaxis, ...]  # (1, time, sources, ears)

        batch_size, time_steps, total_channels = stim.shape

        # Validate channels: should be num_sources * freq_bins
        expected_channels = self.num_sources * self.freq_bins
        if total_channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, got {total_channels}")

        # Validate HRTF gains shape
        expected_hrtf_channels = self.num_sources * 2 * self.freq_bins  # 2 ears per source
        if hrtf_gains.shape[-1] != expected_hrtf_channels:
            raise ValueError(f"Expected {expected_hrtf_channels} HRTF channels, got {hrtf_gains.shape[-1]}")

        # 1. RESHAPE HRTF GAINS: (batch, time, sources*2*freq_bins) → (batch, time, sources, freq_bins, ears)
        # Input format: [S1_Left_freqs, S1_Right_freqs, S2_Left_freqs, S2_Right_freqs, ...]
        hrtf_gains_reshaped = hrtf_gains.reshape(batch_size, time_steps, self.num_sources, 2, self.freq_bins)
        # Reorder to (batch, time, sources, freq_bins, ears)
        hrtf_gains_proper = hrtf_gains_reshaped.transpose(0, 1, 2, 4, 3)

        # 2. RESHAPE STIM: (batch, time, sources*freq_bins) → (batch, time, sources, freq_bins)
        spectrograms = stim.reshape(batch_size, time_steps, self.num_sources, self.freq_bins)

        # 3. COMBINE HRTF AND DISTANCE GAINS IN DB SPACE
        # Broadcast distance gains to match HRTF gains shape
        # distance_gains: (batch, time, sources, ears) -> (batch, time, sources, freq_bins, ears)
        distance_gains_expanded = np.expand_dims(distance_gains, axis=3)  # Add freq dimension
        distance_gains_broadcast = np.broadcast_to(
            distance_gains_expanded,
            (batch_size, time_steps, self.num_sources, self.freq_bins, 2)
        )

        # Combine gains in dB space
        total_gains_db = hrtf_gains_proper + distance_gains_broadcast

        # 4. APPLY COMBINED GAINS (following stim_filt_hrtf pattern)
        # spectrograms: (batch, time, sources, freq_bins)
        # total_gains_db: (batch, time, sources, freq_bins, ears)

        # Square spectrograms FIRST (like original: s1**2, s2**2)
        squared_spectrograms = spectrograms ** 2  # (batch, time, sources, freq_bins)

        # Add ear dimension for broadcasting
        squared_spectrograms_expanded = squared_spectrograms[..., np.newaxis]  # (batch, time, sources, freq_bins, 1)

        # Convert combined gains to linear scale
        linear_gains = 10.0 ** (total_gains_db / 10.0)

        # Apply gains to squared spectrograms (like original: (s1**2) * gain1)
        spatialized_power = squared_spectrograms_expanded * linear_gains  # (batch, time, sources, freq_bins, ears)

        # 5. RMS COMBINATION ACROSS SOURCES
        # Sum across sources (already squared), then sqrt
        summed_power = np.sum(spatialized_power, axis=2)  # (batch, time, freq_bins, ears)
        rms_output = np.sqrt(summed_power + 1e-12)  # Small epsilon for numerical stability

        # 6. CONCATENATE EARS: [right, left]
        left_ear = rms_output[..., 0]   # Index 0 = left ear (matches HRTF grid layout)
        right_ear = rms_output[..., 1]  # Index 1 = right ear (matches HRTF grid layout)

        # Concatenate RIGHT ear first, then LEFT ear (matches stim_filt_hrtf ear ordering)
        binaural_output = np.concatenate([right_ear, left_ear], axis=-1)

        # If input was 2D, remove batch dimension from output
        if binaural_output.shape[0] == 1:
            binaural_output = binaural_output[0]  # (time, freq_bins*2)

        return binaural_output

    def as_tensorflow_layer(self, **kwargs):
        """Build TensorFlow equivalent of HRTF gain application layer."""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        freq_bins = self.freq_bins
        num_sources = self.num_sources

        class ApplyHRTFGainLayerTF(NemsKerasLayer):
            def call(self, inputs):
                # Expect 3 inputs: [hrtf_gains, distance_gains, stim]
                hrtf_gains, distance_gains, stim = inputs

                # Handle extra sample dimension during training
                if len(hrtf_gains.shape) == 3:
                    hrtf_gains = hrtf_gains[0]  # Remove sample dimension
                    hrtf_gains = tf.expand_dims(hrtf_gains, 0)
                if len(distance_gains.shape) == 4:
                    distance_gains = distance_gains[0]  # Remove sample dimension
                    distance_gains = tf.expand_dims(distance_gains, 0)
                elif len(distance_gains.shape) == 3:
                    # (time, sources, ears) -> (1, time, sources, ears)
                    distance_gains = tf.expand_dims(distance_gains, 0)
                if len(stim.shape) == 3:
                    stim = stim[0]  # Remove sample dimension
                    stim = tf.expand_dims(stim, 0)

                batch_size = tf.shape(stim)[0]
                time_steps = tf.shape(stim)[1]

                # 1. RESHAPE HRTF GAINS: (batch, time, sources*2*freq_bins) → (batch, time, sources, freq_bins, ears)
                hrtf_gains_reshaped = tf.reshape(hrtf_gains, [batch_size, time_steps, num_sources, 2, freq_bins])
                hrtf_gains_proper = tf.transpose(hrtf_gains_reshaped, [0, 1, 2, 4, 3])

                # 2. RESHAPE STIM: (batch, time, sources*freq_bins) → (batch, time, sources, freq_bins)
                spectrograms = tf.reshape(stim, [batch_size, time_steps, num_sources, freq_bins])

                # 3. COMBINE HRTF AND DISTANCE GAINS IN DB SPACE
                # distance_gains: (batch, time, sources, ears) -> (batch, time, sources, freq_bins, ears)
                distance_gains_expanded = tf.expand_dims(distance_gains, axis=3)
                distance_gains_broadcast = tf.broadcast_to(
                    distance_gains_expanded,
                    [batch_size, time_steps, num_sources, freq_bins, 2]
                )
                total_gains_db = hrtf_gains_proper + distance_gains_broadcast

                # 4. APPLY COMBINED GAINS
                squared_spectrograms = tf.square(spectrograms)
                squared_spectrograms_expanded = tf.expand_dims(squared_spectrograms, -1)

                linear_gains = tf.pow(10.0, total_gains_db / 10.0)
                spatialized_power = squared_spectrograms_expanded * linear_gains

                # 5. RMS COMBINATION ACROSS SOURCES
                summed_power = tf.reduce_sum(spatialized_power, axis=2)
                rms_output = tf.sqrt(summed_power + 1e-12)

                # 6. CONCATENATE EARS: [right, left]
                left_ear = rms_output[..., 0]
                right_ear = rms_output[..., 1]
                binaural_output = tf.concat([right_ear, left_ear], axis=-1)

                return binaural_output

        return ApplyHRTFGainLayerTF(self, **kwargs)