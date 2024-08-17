import numpy as np
import scipy.signal
from scipy import interpolate

from .base import Layer, Phi, Parameter
from .tools import require_shape, pop_shape
from nems.registry import layer
from nems.distributions import Normal, HalfNormal
from nems.tools.arrays import broadcast_axes


class FiniteImpulseResponse(Layer):

    def __init__(self, stride=1, include_anticausal=False, **kwargs):
        """Convolve linear filter(s) with input.

        Parameters
        ----------
        shape : N-tuple
            Determines the shape of `FIR.coefficients`. Axes should be:
            (T time bins, C input channels (rank),  ..., N output channels)
            where only the first two dimensions are required. Aside from the
            time and filter axes (index 0 and -1, respectively), the size of
            each dimension must match the size of the input's dimensions.

            If only two dimensions are present, a singleton dimension will be
            appended to represent a single output. For higher-dimensional data,
            users are responsible for adding this singleton dimension if needed.

        See also
        --------
        nems.layers.base.Layer

        Examples
        --------
        >>> fir = FiniteImpulseResponse(shape=(15,4))   # (time, input channels)
        >>> weighted_input = np.random.rand(10000, 4)   # (time, channels)
        >>> out = fir.evaluate(weighted_input)
        >>> out.shape
        (10000, 1)

        # strf alias
        >>> fir = STRF(shape=(25, 18))                   # full-rank STRF
        >>> spectrogram = np.random.rand(10000,18)
        >>> out = fir.evaluate(spectrogram)
        >>> out.shape
        (10000, 1)

        # FIR alias                                     
        >>> fir = FIR(shape=(25, 4, 100))               # rank 4, 100 filters
        >>> spectrogram = np.random.rand(10000,4)
        >>> out = fir.evaluate(spectrogram)
        >>> out.shape
        (10000, 1, 100)

        """
        require_shape(self, kwargs, minimum_ndim=2)
        self.stride = stride
        self.include_anticausal = include_anticausal

        super().__init__(**kwargs)


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
        #sd = np.full(shape=self.shape, fill_value=1/np.prod(self.shape))
        sd = np.full(shape=self.shape, fill_value=1/self.shape[0])
        # TODO: May be more appropriate to make this a hard requirement, but
        #       for now this should stop tiny filter sizes from causing errors.
        if mean.shape[0] > 2:
            #mean[1, :] = 2/np.prod(self.shape)
            #mean[2, :] = -1/np.prod(self.shape)
            mean[1, :] = 2 / self.shape[0]
            mean[2, :] = -1 / self.shape[0]
        prior = Normal(mean, sd)

        coefficients = Parameter(name='coefficients', shape=self.shape,
                                 prior=prior)
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
        # Match number of outputs in input and coefficients by broadcasting.
        input, coefficients, _ = self._broadcast(input, coefficients)

        # Prepend zeros. (and append, if include_anticausal)
        padding = self._get_filter_padding(input, coefficients)
        input_with_padding = np.pad(input, padding)

        # Convolve each filter with the corresponding input channel.
        outputs = []
        n_filters = coefficients.shape[-1]
        for i in range(n_filters):
            y = scipy.signal.convolve(
                input_with_padding[...,i], coefficients[...,i], mode='valid'
                )
            outputs.append(y)

        # Concatenate on n_outputs axis
        output = np.stack(outputs, axis=-1)
        # Squeeze out rank dimension
        output = np.squeeze(output, axis=1)
        if self.stride > 1:
            output = output[::self.stride, ...]
        return output

    def _reshape_coefficients(self):
        """Get `coefficients` in the format needed for `evaluate`."""
        coefficients = self.coefficients
        if coefficients.ndim == 2:
            # Add a dummy filter/output axis
            coefficients = coefficients[..., np.newaxis]

        # Coefficients are applied "backwards" (convolution) relative to how
        # they are specified (filter), so have to flip all dimensions except
        # time and number of filters/outputs.
        flipped_axes = [1]  # Always flip rank
        other_dims = coefficients.shape[2:-1]
        for i, d in enumerate(other_dims):
            # Also flip any additional dimensions
            flipped_axes.append(i+2)
        coefficients = np.flip(coefficients, axis=flipped_axes)

        return coefficients

    def _broadcast(self, input, coefficients):
        """Internal for `evaluate`."""
        # Add axis for n output channels to input if one doesn't exist.
        # NOTE: This will only catch a missing output dimension for 2D data.
        #       For higher-dimensional data, the output dimension needs to be
        #       specified by users.
        insert_dim = None

        if input.ndim < 3:
            if input.shape[1] == coefficients.shape[2]:
                input = input[:, np.newaxis, :]
                insert_dim = 1
            else:
                input = input[..., np.newaxis]
                insert_dim = -1

        if input.shape[-1] < coefficients.shape[-1]:
            try:
                input = broadcast_axes(input, coefficients, axis=-1)
            except ValueError:
                raise TypeError(
                    "Last dimension of FIR input must match last dimension of "
                    "coefficients, or one must be broadcastable to the other."
                    )
        elif coefficients.shape[-1] < input.shape[-1]:
            try:
                coefficients = broadcast_axes(coefficients, input, axis=-1)
            except ValueError:
                raise TypeError(
                    "Last dimension of FIR input must match last dimension of "
                    "coefficients, or one must be broadcastable to the other."
                    )
        
        return input, coefficients, insert_dim

    def _get_filter_padding(self, input, coefficients):
        """Get zeros of correct shape to prepend to input on time axis."""
        filter_length = coefficients.shape[0]

        if self.include_anticausal:
            pre_length = int(np.floor(filter_length/2)) - 1
            post_length = filter_length - pre_length - 1
            padding = [[pre_length, post_length]] + [[0, 0]]*(input.ndim-1)
        else:
            # Prepend 0s on time axis, no padding on other axes
            padding = [[filter_length-1, 0]] + [[0, 0]]*(input.ndim-1)

        return padding

    @layer('fir')
    def from_keyword(keyword):
        """Construct FIR (or subclass) from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape.
            (time, input channels a.k.a. rank, ..., output channels) 

        See also
        --------
        Layer.from_keyword
        
        """
        kwargs = {}
        fir_class = FiniteImpulseResponse

        options = keyword.split('.')
        kwargs['shape'] = pop_shape(options)
        for op in options:
            if op.startswith('p') and op[1].isdigit():
                # Pole-zero parameterization
                fir_class = PoleZeroFIR
                fs_idx = op.index('fs')
                zeros_idx = op.index('z')
                kwargs['n_poles'] = int(op[1:zeros_idx])
                kwargs['n_zeros'] = int(op[zeros_idx+1:fs_idx])
                kwargs['fs'] = int(op[fs_idx+2:])
            elif op.startswith('s'):
                kwargs['stride'] = int(op[1:])
            elif op.startswith('l2'):
                kwargs['regularizer'] = op
        fir = fir_class(**kwargs)

        return fir
    
    def as_tensorflow_layer(self, input_shape, **kwargs):
        """Convert FiniteImpulseResponse to a TensorFlow Keras Layer.
        
        Parameters
        ----------
        inputs : tf.Tensor or np.ndarray
            Initial input to Layer, supplied by TensorFlowBackend during model
            building.
        
        Returns
        -------
        FiniteImpulseResponseTF
        
        """

        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        old_c = self.parameters['coefficients']
        coefficients = self.coefficients
        if coefficients.ndim == 2:
            # Add a dummy filter/output axis
            coefficients = coefficients[..., np.newaxis]
        new_c = np.flip(coefficients, axis=0)
        filter_width, rank, _ = new_c.shape
        if new_c.ndim > 3:
            raise NotImplementedError(
                "FIR TF implementation currently only works for 2D data."
                )
        new_values = {'coefficients': new_c}  # override Parameter.values

        # Define broadcasting behavior for inputs and coefficients based on
        # input_shape and new_c.shape.
        broadcast_inputs, broadcast_coefficients, n_outputs = \
            self._define_tf_broadcasting(
                tf, input_shape, new_c
                )
        # Define convolution operation, depends on whether a GPU is available.
        convolve = self._define_tf_convolution(
            tf, filter_width, rank, n_outputs
            )

        class FiniteImpulseResponseTF(NemsKerasLayer):
            def weights_to_values(self):
                c = self.parameter_values['coefficients']
                unflipped = np.flip(c, axis=0)  # Undo flip time
                unshaped = np.reshape(unflipped, old_c.shape)

                return {'coefficients': unshaped}

            def call(self, inputs):
                # This will add an extra dim if there is no output dimension.
                input_width = tf.shape(inputs)[1] # tf.shape(inputs)[1] or inputs.shape[1]
                # Broadcast output shape if needed.
                inputs = broadcast_inputs(inputs)
                coefficients = broadcast_coefficients(self.coefficients)
                # Make None shape explicit
                rank_4 = tf.reshape(inputs, [-1, input_width, rank, n_outputs])
                return convolve(rank_4, coefficients)

        return FiniteImpulseResponseTF(self, new_values=new_values, **kwargs)


    def _define_tf_broadcasting(self, tf, input_shape, new_c):
        """Internal for `as_tensorflow_layer`.
        
        Builds `broadcast_inputs` and `broadcast_coefficients` for use inside
        `call` method.

        Parameters
        ----------
        tf : package
            Reference to imported TensorFlow package.
        input_shape : tuple
            Shape of inputs.
        new_c : np.ndarray
            Reshaped coefficients.

        Returns
        -------
        broadcast_inputs : function
        broadcast_coefficients : function
        n_outputs : int
            Number of broadcasted outputs.

        """

        # Fake input to set up correct broadcasting behavior.
        # Only the number of outputs matters, this drops the batch dimension.
        # TODO: This might mess up with multiple batches similar to WC?
        #       Need to check if list?
        fake_inputs = np.empty(shape=input_shape[1:])
        new_inputs, broadcast_c, insert_dim = self._broadcast(fake_inputs, new_c)
        new_coefs_shape = list(new_c.shape[:-1]) + [broadcast_c.shape[-1]]
        new_inputs_shape = list(new_inputs.shape)
        n_outputs = new_coefs_shape[-1]

        if new_inputs_shape[-1] > input_shape[-1]:
            # If output dimension increased, then TF needs to broadcast output
            # dimension of input in call function.
            if new_inputs.ndim > fake_inputs.ndim:
                # A singleton output dimension needs to be appended as well.
                @tf.function
                def expand_inputs(inputs):
                    return tf.expand_dims(inputs, axis=-1)
            else:
                @tf.function
                def expand_inputs(inputs):
                    return inputs

            @tf.function()
            def broadcast_inputs(inputs):
                # Convert None batch shape to int, add singleton output dim
                # if needed. Then broadcast outputs.
                batch_size = tf.keras.backend.shape(inputs)[0]
                shape = [batch_size] + new_inputs_shape
                return tf.broadcast_to(expand_inputs(inputs), shape)

        elif new_inputs.ndim > fake_inputs.ndim:
            #print(new_inputs_shape)
            #print(input_shape)

            @tf.function()
            def broadcast_inputs(inputs):
                # Convert None batch shape to int, add singleton output dim
                # if needed. Then broadcast outputs.
                batch_size = tf.keras.backend.shape(inputs)[0]
                shape = [batch_size] + new_inputs_shape
                # insert_dim is where a dummy dimension was added to the inputs
                # so that the summing occurs (or not) across the appropriate dims
                return tf.broadcast_to(tf.expand_dims(inputs, axis=insert_dim+1), shape)

        else:
            # Otherwise, don't need to do anything to inputs.
            @tf.function
            def broadcast_inputs(inputs):
                # This will still add a singleton output dim if needed.
                #return tf.reshape(inputs, new_inputs_shape)
                return inputs

        if new_coefs_shape[-1] > new_c.shape[-1]:
            # Coefficients outputs increased, need to broadcast coefs in call.
            @tf.function
            def broadcast_coefficients(coefficients):
                return tf.broadcast_to(coefficients, new_coefs_shape)
        else:
            # Otherwise, don't need to do anything to coefficients.
            @tf.function
            def broadcast_coefficients(coefficients): return coefficients
        
        return broadcast_inputs, broadcast_coefficients, n_outputs

    def _define_tf_convolution(self, tf, filter_width, rank, n_outputs):
        """Internal for `as_tensorflow_layer`.
        
        Builds `convolution` function for use in `call` method.

        Parameters
        ----------
        tf : package
            Reference to imported TensorFlow package.
        filter_width, rank, n_outputs : coefficient shape components 

        Returns
        -------
        convolution : function
        
        """

        num_gpus = len(tf.config.list_physical_devices('GPU'))
        stride = self.stride
        if num_gpus == 0:
            # Use CPU-compatible (but slower) version.
            @tf.function
            def convolve(inputs, coefficients):
                # Reorder coefficients to shape (n outputs, time, rank, 1)
                new_coefs = tf.expand_dims(
                    tf.transpose(coefficients, [2, 0, 1]), -1
                    )
                if self.include_anticausal:
                    pre_length = int(np.floor(filter_width / 2)) - 1
                    post_length = filter_width - pre_length - 1
                    padded_input = tf.pad(
                        inputs, [[0, 0], [pre_length, post_length], [0, 0], [0, 0]]
                    )
                else:
                    padded_input = tf.pad(
                        inputs, [[0, 0], [filter_width-1, 0], [0, 0], [0, 0]]
                        )
                # Reorder input to shape (n outputs, batch, time, rank)
                x = tf.transpose(padded_input, [3, 0, 1, 2])
                fn = lambda t: tf.cast(tf.nn.conv1d(  # TODO: don't like forcing dtype here
                    t[0], t[1], stride=stride, padding='VALID'
                    ), tf.float64)
                # Apply convolution for each output
                y = tf.map_fn(
                    fn=fn,
                    elems=(x, new_coefs),
                    fn_output_signature=tf.float64
                    )
                # Reorder output back to (batch, time, n outputs)
                z = tf.transpose(tf.squeeze(y, axis=3), [1, 2, 0])
                return z
        else:
            # Use GPU-only version (grouped convolutions), much faster.
            @tf.function
            def convolve(inputs, coefficients):
                input_width = tf.shape(inputs)[1]
                # Reshape will group by output before rank w/o transpose.
                #print("input shape:", inputs.shape.as_list(), "coef shape:", coefficients.shape.as_list())
                transposed = tf.transpose(inputs, [0, 1, 3, 2])
                # Collapse rank and n_outputs to one dimension.
                # -1 for batch size b/c it can be None.
                reshaped = tf.reshape(
                    transposed, [-1, input_width, rank*n_outputs]
                    )
                if self.include_anticausal:
                    pre_length = int(np.floor(filter_width / 2)) - 1
                    post_length = filter_width - pre_length - 1
                    padded_input = tf.pad(
                        reshaped, [[0, 0], [pre_length, post_length], [0, 0]]
                        )
                else:
                    # Prepend 0's on time axis as initial conditions for filter.
                    padded_input = tf.pad(
                        reshaped, [[0, 0], [filter_width-1, 0], [0, 0]]
                        )
                #print("reshaped shape:", reshaped.shape.as_list(), "padded shape:", padded_input.shape.as_list())
                # Convolve filters with input slices in groups of size `rank`.
                y = tf.nn.conv1d(
                    padded_input, coefficients, stride=stride, padding='VALID'
                    )
                #print("y shape:", y.shape.as_list())
                return y 

        return convolve


# Aliases, STRF specifically for full-rank (but not enforced)
class FIR(FiniteImpulseResponse):
    pass
class STRF(FiniteImpulseResponse):
    pass


class PoleZeroFIR(FiniteImpulseResponse):

    def __init__(self, n_poles, n_zeros, fs, **kwargs):
        """TODO: docs.
        
        TODO: Possible to remove need for sampling rate?
        
        """
        self.n_poles = n_poles
        self.n_zeros = n_zeros
        self.fs = fs
        require_shape(self, kwargs, minimum_ndim=2, maximum_ndim=3)
        super().__init__(**kwargs)

    def initial_parameters(self):
        """TODO: docs."""

        # TODO: explain choice of priors
        rank = self.shape[1]
        if len(self.shape) == 3:
            n_filters = self.shape[2]
        else:
            n_filters = 1
        pole_set = np.array([[[0.8, -0.4, 0.1, 0.0, 0]]])[..., :self.n_poles]
        zero_set = np.array([[[0.1,  0.1, 0.1, 0.1, 0]]])[..., :self.n_zeros]

        poles_prior = Normal(
            mean = pole_set.repeat(rank, 0).repeat(n_filters, 1),
            sd = np.ones((rank, n_filters, self.n_poles))*0.3,
            )
        zeros_prior = Normal(
            mean = zero_set.repeat(rank, 0).repeat(n_filters, 1),
            sd = np.ones((rank, n_filters, self.n_zeros))*0.2,
            )
        delays_prior = HalfNormal(sd = np.ones((rank, n_filters))*0.02)
        gains_prior = Normal(
            mean = np.zeros((rank, n_filters))+0.1,
            sd = np.ones((rank, n_filters))*0.2
            )

        poles = Parameter('poles', shape=(rank, n_filters, self.n_poles),
                          prior=poles_prior, bounds=(-1, 1))
        zeros = Parameter('zeros', shape=(rank, n_filters, self.n_zeros),
                          prior=zeros_prior, bounds=(-1, 1))
        # TODO: what do the delays do exactly?
        delays = Parameter('delays', shape=(rank, n_filters),
                           prior=delays_prior, bounds=(0, np.inf))
        gains = Parameter('gains', shape=(rank, n_filters),
                          prior=gains_prior)

        return Phi(poles, zeros, delays, gains)

    @property
    def coefficients(self):
        """TODO: docs."""
        poles, zeros, delays, gains = self.get_parameter_values()

        n_taps, rank = self.shape[:2]
        if len(self.shape) == 2:
            n_filters = 1
        else:
            n_filters = self.shape[-1]

        coefficients = np.zeros((n_taps, rank, n_filters))

        # TODO: can we do this without fs?
        # TODO: explain why 5*original
        fs2 = 5*self.fs                      

        for i in range(rank):
            for j in range(n_filters):
                # TODO: rename variables, improve documentation.
                #       still don't really know what this is doing.
                t = np.arange(0, n_taps*5 + 1) / fs2
                h = scipy.signal.ZerosPolesGain(
                    zeros[i,j], poles[i,j], gains[i,j], dt=1/fs2
                    )
                tout, ir = scipy.signal.dimpulse(h, t=t)
                f = interpolate.interp1d(tout, ir[0][:,0], bounds_error=False,
                                         fill_value=0)

                tnew = np.arange(0, n_taps)/self.fs - delays[i,j] + 1/self.fs
                coefficients[:, i, j] = f(tnew)

        return coefficients

    # TODO: as_tensorflow_layer
