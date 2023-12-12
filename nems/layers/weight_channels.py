import numpy as np

from nems.registry import layer
from nems.distributions import Normal, HalfNormal
from .base import Layer, Phi, Parameter, ShapeError
from .tools import require_shape, pop_shape
from tensorflow.python.keras import regularizers


# TODO: double check all shape references after dealing w/ data order etc,
#       make sure correct dims are lined up.
class WeightChannels(Layer):
    """Compute linear weighting of input channels, akin to a dense layer.

    Parameters
    ----------
    shape : N-tuple (usually N=2)
        Determines the shape of `WeightChannels.coefficients`.
        First dimension should match the spectral dimension of the input,
        second dimension should match the spectral dimension of the output.
        Note that higher-dimesional shapes are also allowed and should work
        as-is for this base class, but overall Layer design is intended for
        2-dimensional data so subclasses might not support other shapes.

    See also
    --------
    nems.layers.base.Layer

    Examples
    --------
    >>> wc = WeightChannels(shape=(18,4))
    >>> spectrogram = np.random.rand(10000, 18)  # (time, channels)
    >>> out = spectrogram @ wc.coefficients      # wc.evaluate(spectrogram)
    >>> out.shape
    (10000, 1)

    """
    def __init__(self, **kwargs):
        require_shape(self, kwargs, minimum_ndim=2)
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Get initial values for `WeightChannels.parameters`.
        
        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `WeightChannels.shape`.
            Prior:  TODO, currently using defaults
            Bounds: TODO

        Returns
        -------
        nems.layers.base.Phi

        """
        # Mean and sd for priors were chosen mostly arbitrarily, to start
        # with most weights near zero (but not exactly at 0).
        mean = np.full(shape=self.shape, fill_value=0.01)
        sd = np.full(shape=self.shape, fill_value=0.05)
        prior = Normal(mean, sd)

        coefficients = Parameter(
            name='coefficients', shape=self.shape, prior=prior
            )
        return Phi(coefficients)

    @property
    def coefficients(self):
        """Weighting matrix that will be applied to input.
        
        Re-parameterized subclasses should overwrite this so that `evaluate`
        doesn't need to change.

        Returns
        -------
        coefficients : ndarray
            coefficients.shape = WeightChannels.shape
        
        """
        return self.parameters['coefficients'].values

    def evaluate(self, input):
        """Multiply input by WeightChannels.coefficients.

        Computes $y = XA$ for input $X$, where $A$ is
        `WeightChannels.coefficients` and $y$ is the output.
        
        Parameters
        ----------
        input : np.ndarray

        Returns
        -------
        np.ndarray
        
        """

        try:
            output = np.tensordot(input, self.coefficients, axes=(1, 0))
        except ValueError as e:
            # Check for dimension swap, to give more informative error message.
            if 'shape-mismatch for sum' in str(e):
                raise ShapeError(self, input=input.shape,
                                 coefficients=self.coefficients.shape)
            else:
                # Otherwise let the original error through.
                raise e

        return output

    @layer('wc')
    def from_keyword(keyword):
        """Construct WeightChannels (or subclass) from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape.
        g : Use gaussian function(s) to determine coefficients.

        See also
        --------
        Layer.from_keyword
        
        """
        wc_class = WeightChannels
        kwargs = {}
        options = keyword.split('.')
        kwargs['shape'] = pop_shape(options)

        for op in options:
            if op == 'g':
                wc_class = WeightChannelsGaussian
            elif op == 'b':
                wc_class = WeightChannelsMulti
            elif op.startswith('l2'):
                kwargs['regularizer'] = op

        wc = wc_class(**kwargs)

        return wc

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs
        
        NOTE: This is currently hard-coded to dot the 2nd dim of the input
        (batch, time, channels, ...) and first dim of coefficients
        (channels, rank, ...).
        
        """
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        class WeightChannelsTF(NemsKerasLayer):
            @tf.function
            def call(self, inputs):
                out = tf.tensordot(inputs, self.coefficients, axes=[[2], [0]])
                return out
        
        return WeightChannelsTF(self, **kwargs)

    @property
    def plot_kwargs(self):
        """Add incremented labels to each output channel for plot legend.
        
        See also
        --------
        Layer.plot
        
        """
        kwargs = {
            'label': [f'Channel {i}' for i in range(self.shape[1])]
        }
        return kwargs

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
        return {'legend': False}


class WeightChannelsMulti(WeightChannels):
    """WeightChannels specialized for multichannel input (eg, binaural).

    Parameters
    ----------
    shape : N-tuple (usually N=3)

    See also
    --------
    nems.layers.base.Layer
    nems.layers.weight_channels.WeightChannels

    Examples
    --------
    >>> wc = WeightChannelsMulti(shape=(18,4,2))
    >>> spectrogram = np.random.rand(1000, 18, 2)  # (time, spectral_channels, ear)
    >>> # wc.evaluate(spectrogram) equivalent...
    >>> out = np.empty((spectrogram.shape[0], wc.shape[1], wc.shape[2]))
    >>> for i in range(wc.shape[-1]):
            out[:,:,i] = spectrogram[:,:,i] @ wc.coefficients[:,:,i]
    >>> out.shape
    (1000, 4, 2)

    """

    def evaluate(self, input):
        """Multiply `input[...,i]` by `WeightChannels.coefficients[...,i]`."""

        try:
            # hack -- reshape spectral axis to allow using the same weights
            # across multiple tiles, the input, reflecting different dimensions
            # (eg, the stimulus into each ear)
            if input.shape[-1] > self.coefficients.shape[0]:
                input = np.swapaxes(np.reshape(input, [input.shape[0], -1, self.coefficients.shape[0]]), -2, -1)

            output = np.moveaxis(np.matmul(np.moveaxis(input, [0,1], [-2,-1]),
                                           np.moveaxis(self.coefficients, [0,1], [-2,-1])),
                                 [-2,-1], [0, 1])
        except ValueError as e:
            # Check for dimension swap, to give more informative error message.
            if 'mismatch in core dimension' in str(e):
                raise ShapeError(self, input=input.shape,
                                 coefficients=self.coefficients.shape)
            else:
                # Otherwise let the original error through.
                raise e

        return output

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        class WeightChannelsMultiTF(NemsKerasLayer):
            @tf.function
            def call(self, inputs):
                # reshape inputs and coefficients so that mult can happen on last
                # two dimensions. Broadcasting seems to work fine this way
                if inputs.shape[-1] > self.coefficients.shape[0]:
                    d = int(inputs.shape[-1]/self.coefficients.shape[0])
                    inputs_ = tf.experimental.numpy.swapaxes(
                        tf.reshape(tf.convert_to_tensor(inputs), [-1, inputs.shape[1], d, self.coefficients.shape[0]]), -2, -1)
                else:
                    inputs_ = inputs

                out = tf.experimental.numpy.moveaxis(
                    tf.matmul(
                        tf.experimental.numpy.moveaxis(inputs_, [1, 2], [-2, -1]),
                        tf.experimental.numpy.moveaxis(self.coefficients, [0, 1], [-2, -1])
                        ),
                    [-2, -1], [1, 2]
                    )
                return out

        return WeightChannelsMultiTF(self, **kwargs)


class WeightChannelsGaussian(WeightChannels):
    """As WeightChannels, but sample coefficients from gaussian functions."""

    def initial_parameters(self):
        """Get initial values for `WeightChannelsGaussian.parameters`.

        Layer parameters
        ----------------
        mean : scalar or ndarray
            Mean of gaussian, shape is `self.shape[1:]`.
            Prior:  TODO  # Currently using defaults
            Bounds: (0, 1)
        sd : scalar or ndarray
            Standard deviation of gaussian, shape is `self.shape[1:]`.
            Prior:  TODO  # Currently using defaults
            Bounds: (0*, np.inf)
            * Actually set to machine epsilon to avoid division by zero.

        Returns
        -------
        nems.layers.base.Phi

        """

        mean_bounds = (-0.1, 1.1)
        sd_bounds = (0.05, np.inf)

        rank = self.shape[1]
        other_dims = self.shape[2:]
        shape = (rank,) + other_dims
        # Pick means so that the centers of the gaussians are spread across the 
        # available frequencies.
        channels = np.arange(rank + 1)[1:]
        tiled_means = channels / (rank*2 + 2) + 0.25
        for dim in other_dims:
            # Repeat tiled gaussian structure for other dimensions.
            tiled_means = tiled_means[...,np.newaxis].repeat(dim, axis=-1)

        mean_prior = Normal(tiled_means, np.full_like(tiled_means, 0.2))
        sd_prior = HalfNormal(np.full_like(tiled_means, 0.4))

        parameters = Phi(
            Parameter(name='mean', shape=shape, bounds=mean_bounds,
                        prior=mean_prior),
            Parameter(name='sd', shape=shape, bounds=sd_bounds,
                        prior=sd_prior, zero_to_epsilon=True)
            )

        return parameters

    @property
    def coefficients(self):
        """Return N discrete gaussians with T bins, where `shape=(T,N)`."""
        mean = self.parameters['mean'].values
        sd = self.parameters['sd'].values
        n_input_channels = self.shape[0]

        x = np.arange(n_input_channels)/n_input_channels
        mean = np.asanyarray(mean)[..., np.newaxis]
        sd = np.asanyarray(sd)[..., np.newaxis]
        coefficients = np.exp(-0.5*((x-mean)/sd)**2)  # (rank, ..., outputs, T)
        reordered = np.moveaxis(coefficients, -1, 0)  # (T, rank, ..., outputs)
        # Normalize by the cumulative sum for each channel
        cumulative_sum = np.sum(reordered, axis=-1, keepdims=True)
        # If all coefficients for a channel are 0, skip normalization
        cumulative_sum[cumulative_sum == 0] = 1
        normalized = reordered/cumulative_sum

        return normalized

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        # TODO: Ask SVD about this kludge in old NEMS code. Is this still needed?
        # If so, explain: I think this was to keep gradient from "blowing up"?
        # Scale up sd bound
        sd, = self.get_parameter_values('sd')
        sd_lower, sd_upper = self.parameters['sd'].bounds
        new_values = {'sd': sd*10}
        new_bounds = {'sd': (sd_lower, sd_upper*10)}

        class WeightChannelsGaussianTF(NemsKerasLayer):
            def call(self, inputs):
                mean = tf.expand_dims(self.mean, -1)
                sd = tf.expand_dims(self.sd/10, -1)
                input_features = tf.cast(tf.shape(inputs)[-1],
                                         dtype=inputs.dtype)
                temp = tf.range(input_features) / input_features

                # roll channel dim from -1 to 0
                sh = np.roll(np.arange(len(mean.shape)),1)
                temp = tf.transpose((temp-mean)/sd, sh)

                temp = tf.math.exp(-0.5 * tf.math.square(temp))
                norm = tf.math.reduce_sum(temp, axis=-1, keepdims=True)
                kernel = temp / norm
                return tf.tensordot(inputs, kernel, axes=[[2], [0]])

            def weights_to_values(self):
                values = self.parameter_values
                # Undo scaling.
                values['sd'] = values['sd'] / 10
                return values

        return WeightChannelsGaussianTF(self, new_values=new_values,
                                        new_bounds=new_bounds, **kwargs)

    
class WeightChannelsMultiGaussian(WeightChannels):
    """WeightChannels specialized for multichannel input (eg, binaural) and with Gaussian tuning.

    TODO : actually make this work -- current pasted together fomr WeightChannelsMulti and WeightChannelsGaussian, but not tested

    Parameters
    ----------
    shape : N-tuple (usually N=3)

    See also
    --------
    nems.layers.base.Layer
    nems.layers.weight_channels.WeightChannels

    Examples
    --------
    >>> wc = WeightChannelsMultiGaussian(shape=(18,4,2))
    >>> spectrogram = np.random.rand(1000, 18, 2)  # (time, spectral_channels, ear)
    >>> # wc.evaluate(spectrogram) equivalent...
    >>> out = np.empty((spectrogram.shape[0], wc.shape[1], wc.shape[2]))
    >>> for i in range(wc.shape[-1]):
            out[:,:,i] = spectrogram[:,:,i] @ wc.coefficients[:,:,i]
    >>> out.shape
    (1000, 4, 2)

    """

    def initial_parameters(self):
        """Get initial values for `WeightChannelsGaussian.parameters`.

        Layer parameters
        ----------------
        mean : scalar or ndarray
            Mean of gaussian, shape is `self.shape[1:]`.
            Prior:  TODO  # Currently using defaults
            Bounds: (0, 1)
        sd : scalar or ndarray
            Standard deviation of gaussian, shape is `self.shape[1:]`.
            Prior:  TODO  # Currently using defaults
            Bounds: (0*, np.inf)
            * Actually set to machine epsilon to avoid division by zero.

        Returns
        -------
        nems.layers.base.Phi

        """

        mean_bounds = (0, 1)
        sd_bounds = (0, np.inf)

        rank = self.shape[1]
        other_dims = self.shape[2:]
        shape = (rank,) + other_dims
        # Pick means so that the centers of the gaussians are spread across the 
        # available frequencies.
        channels = np.arange(rank + 1)[1:]
        tiled_means = channels / (rank*2 + 2) + 0.25
        for dim in other_dims:
            # Repeat tiled gaussian structure for other dimensions.
            tiled_means = tiled_means[...,np.newaxis].repeat(dim, axis=-1)

        mean_prior = Normal(tiled_means, np.full_like(tiled_means, 0.2))
        sd_prior = HalfNormal(np.full_like(tiled_means, 0.4))
            
        parameters = Phi(
            Parameter(name='mean', shape=shape, bounds=mean_bounds,
                        prior=mean_prior),
            Parameter(name='sd', shape=shape, bounds=sd_bounds,
                        prior=sd_prior, zero_to_epsilon=True)
            )

        return parameters

    @property
    def coefficients(self):
        """Return N discrete gaussians with T bins, where `shape=(T,N)`."""
        mean = self.parameters['mean'].values
        sd = self.parameters['sd'].values
        n_input_channels = self.shape[0]

        x = np.arange(n_input_channels)/n_input_channels
        mean = np.asanyarray(mean)[..., np.newaxis]
        sd = np.asanyarray(sd)[..., np.newaxis]
        coefficients = np.exp(-0.5*((x-mean)/sd)**2)  # (rank, ..., outputs, T)
        reordered = np.moveaxis(coefficients, -1, 0)  # (T, rank, ..., outputs)
        # Normalize by the cumulative sum for each channel
        cumulative_sum = np.sum(reordered, axis=-1, keepdims=True)
        # If all coefficients for a channel are 0, skip normalization
        cumulative_sum[cumulative_sum == 0] = 1
        normalized = reordered/cumulative_sum

        return normalized

    def evaluate(self, input):
        """Multiply `input[...,i]` by `WeightChannels.coefficients[...,i]`."""

        try:
            output = np.moveaxis(
                np.matmul(np.rollaxis(input, 2), np.rollaxis(self.coefficients, 2)),
                [0,1,2],[2,0,1])
        except ValueError as e:
            # Check for dimension swap, to give more informative error message.
            if 'mismatch in core dimension' in str(e):
                raise ShapeError(self, input=input.shape,
                                 coefficients=self.coefficients.shape)
            else:
                # Otherwise let the original error through.
                raise e

        return output

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        class WeightChannelsMultiTF(NemsKerasLayer):
            @tf.function
            def call(self, inputs):
                # reshape inputs and coefficients so that mult can happen on last
                # two dimensions. Broadcasting seems to work fine this way
                out = tf.transpose(
                    tf.matmul(
                        tf.transpose(inputs, perm=[0, 3, 1, 2]),
                        tf.transpose(self.coefficients, perm=[2, 0, 1])
                        ),
                    perm=[0, 2, 3, 1]
                    )
                return out

        return WeightChannelsMultiTF(self, **kwargs)


class WeightGaussianExpand(Layer):
    """As WeightChannels, but sample coefficients from gaussian functions."""

    def initial_parameters(self):
        """Get initial values for `WeightChannelsGaussian.parameters`.

        Layer parameters
        ----------------
        mean : scalar or ndarray
            Mean of gaussian, shape is `self.shape[1:]`.
            Prior:  TODO  # Currently using defaults
            Bounds: (0, 1)
        sd : scalar or ndarray
            Standard deviation of gaussian, shape is `self.shape[1:]`.
            Prior:  TODO  # Currently using defaults
            Bounds: (0*, np.inf)
            * Actually set to machine epsilon to avoid division by zero.

        Returns
        -------
        nems.layers.base.Phi

        """

        mean_bounds = (0, 1.0)
        sd_bounds = (0.2, np.inf)
        amp_bounds = (0.05, np.inf)

        input_channels = self.shape[0]
        output_channels = self.shape[1]
        other_dims = self.shape[2:]
        shape = (input_channels,) + other_dims
        # Pick means so that the centers of the gaussians are spread across the
        # available frequencies.
        channels = np.arange(input_channels + 1)[1:]
        tiled_means = channels / (input_channels*2 + 2) + 0.25
        for dim in other_dims:
            # Repeat tiled gaussian structure for other dimensions.
            tiled_means = tiled_means[...,np.newaxis].repeat(dim, axis=-1)

        mean_prior = Normal(tiled_means, np.full_like(tiled_means, 0.2))
        sd_prior = HalfNormal(np.full_like(tiled_means, 0.4))
        amp_prior = HalfNormal(np.full_like(tiled_means, 0.4))

        parameters = Phi(
            Parameter(name='mean', shape=shape, bounds=mean_bounds,
                        prior=mean_prior),
            Parameter(name='sd', shape=shape, bounds=sd_bounds,
                        prior=sd_prior, zero_to_epsilon=True),
            Parameter(name='amp', shape=shape, bounds=amp_bounds,
                        prior=amp_prior, zero_to_epsilon=True)
            )

        return parameters

    @property
    def coefficients(self):
        """Return N discrete gaussians with T bins, where `shape=(T,N)`."""
        mean = self.parameters['mean'].values
        sd = self.parameters['sd'].values
        amp = self.parameters['amp'].values
        output_channels = self.shape[1]

        x = np.arange(output_channels)/output_channels
        mean = np.asanyarray(mean)[..., np.newaxis]
        sd = np.asanyarray(sd)[..., np.newaxis]
        amp = np.asanyarray(amp)[..., np.newaxis]
        coefficients = np.exp(-0.5*((x-mean)/sd)**2)  # (rank, ..., outputs, T)

        # Normalize by the cumulative sum for each channel
        #cumulative_sum = np.sum(coefficients, axis=-1, keepdims=True)
        # If all coefficients for a channel are 0, skip normalization
        #cumulative_sum[cumulative_sum == 0] = 1
        normalized = coefficients*amp # / cumulative_sum

        reordered = np.moveaxis(normalized, -1, 1)  # (T, rank, ..., outputs)
        return reordered

    def evaluate(self, input):
        """Multiply input by WeightChannels.coefficients.

        Computes $y = XA$ for input $X$, where $A$ is
        `WeightChannels.coefficients` and $y$ is the output.

        Parameters
        ----------
        input : np.ndarray

        Returns
        -------
        np.ndarray

        """

        try:
            sci = 1 / (1+np.exp(-input))
            output = np.tensordot(sci, self.coefficients, axes=(1, 0))
            #mm = np.exp(output)
            #nn = np.sum(mm, axis=1, keepdims=True)
            #output = mm/nn
            #output = mm

        except ValueError as e:
            # Check for dimension swap, to give more informative error message.
            if 'shape-mismatch for sum' in str(e):
                raise ShapeError(self, input=input.shape,
                                 coefficients=self.coefficients.shape)
            else:
                # Otherwise let the original error through.
                raise e

        return output

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        # TODO: Ask SVD about this kludge in old NEMS code. Is this still needed?
        # If so, explain: I think this was to keep gradient from "blowing up"?
        # Scale up sd bound
        sd, = self.get_parameter_values('sd')
        sd_lower, sd_upper = self.parameters['sd'].bounds
        new_values = {'sd': sd*10}
        new_bounds = {'sd': (sd_lower, sd_upper*10)}
        output_channels = self.shape[1]
        class WeightGaussianExpandTF(NemsKerasLayer):
            def call(self, inputs):
                mean = tf.expand_dims(self.mean, -1)
                sd = tf.expand_dims(self.sd/10, -1)
                amp = tf.expand_dims(self.amp, -1)
                output_features = tf.cast(output_channels, dtype=inputs.dtype)
                temp = tf.expand_dims(tf.range(output_features) / output_features, 0)
                temp = (temp-mean)/sd
                temp = tf.math.exp(-0.5 * tf.math.square(temp))
                #norm = tf.math.reduce_sum(temp, axis=0, keepdims=True)
                kernel = temp * amp
                sci = 1 / (1+tf.math.exp(-inputs))
                return tf.tensordot(sci, kernel, axes=[[2], [0]])

                #mm = tf.math.exp(sc)
                #nn = tf.reduce_sum(mm, axis=1, keepdims=True)
                #return mm/nn
                #return sc

            def weights_to_values(self):
                values = self.parameter_values
                # Undo scaling.
                values['sd'] = values['sd'] / 10
                return values

        return WeightGaussianExpandTF(self, new_values=new_values,
                                        new_bounds=new_bounds, **kwargs)

    def from_keyword(keyword):
        # TODO
        pass

    @property
    def plot_kwargs(self):
        """Add incremented labels to each output channel for plot legend.

        See also
        --------
        Layer.plot

        """
        kwargs = {
            'label': [f'Channel {i}' for i in range(self.shape[1])]
        }
        return kwargs

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
        return {'legend': False}
