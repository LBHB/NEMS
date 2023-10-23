from .base import Layer, Phi, Parameter
import tensorflow as tf
import numpy as np
import scipy
from .tools import require_shape, pop_shape
from nems.distributions import Normal, HalfNormal
from nems.registry import layer
from nems.tools.arrays import broadcast_axes


# NOTE: WIP, some things may not work as intended
#       - Output is most likely incorrect atm
class Conv2d(Layer):
    '''
    NEMS compatible convolutional 2-D layer. 

    NOTE: Make sure rank from other layers is or isn't
    required here, if it is, implement via separable_conv2d

    Parameters
    ----------
    shape: N-Tuple
        Determines the shape of the filter used on the
        given input through Conv2D. First 2 dimensions
        determine filter size, last dimension is used
        for amount of filters, batch dimension used in
        conv2d function

        Shape should be of format;
        (Time x Neurons x Batch x Filters) or,
        (Time x Neurons x Filters) or,
        (Time x Neurons)

    See also
    --------
    nems.layers.base.Layer

    Other Info
    ----------
    If batches are needed in filters, full 4-D shape must be provided.

    If shape == 3, we assume filters are provided, and include defaults for batches
    If shape == 2, we provided defaults for filters and batches

    Input format is;
    (Batch x Time x Neurons x Out_Channels) or,
    (Batch x Time x Neurons) or,
    (Time x Neurons)

    '''
    def __init__(self, stride=1, pad_type='zero', pad_axes='both',
                  pool_type='AVG', **kwargs):
        '''
        Initializes layer with given parameters

        Parameters
        ----------
        stride: Tuple-Int
            Provides stride values to be used in convolutions
        pad_type: String
            Type of padding to be used on input data. See
            self.pad for more info
        pad_axes: String
            Axes to apply padding to our input
        pool_type: String
            Type of pooling to be used on filters dimension reduction.
            See self.pool for more info 
        '''
        require_shape(self, kwargs, minimum_ndim=2)
        self.stride      = stride
        self.pad_type    = pad_type
        self.pad_axes    = pad_axes
        self.pool_type   = pool_type
        super().__init__(**kwargs)

    def initial_parameters(self):
        '''
        Saves important paramenters such as input data shape, to be used
        in evaluation.

        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `FIR.shape`.
            Prior:  Normal(mean=0, sd=1/size)
            Bounds: (-np.inf, np.inf)

        Returns
        -------
        nems.layers.base.Phi
        '''
        mean = np.full(shape=self.shape, fill_value=0.0)
        sd = np.full(shape=self.shape, fill_value=1/np.prod(self.shape))
        if mean.shape[0] > 2:
            mean[1, :] = 2/np.prod(self.shape)
            mean[2, :] = -1/np.prod(self.shape)
        prior = Normal(mean, sd)
        coefficients = Parameter(name='coefficients', shape=self.shape,
                                 prior=prior)
        return Phi(coefficients)

    def evaluate(self, input):
        '''
        '''
        filter_array = self.shape_filter(self.coefficients)
        input_array, filter_array = self.shape_input(input, filter_array)
        pad_indices = [0, input_array.shape[1], 0, input_array.shape[2]]

        if self.pad_type != 'None':
            input_array, pad_indices = self.pad(input_array)

            
        # Should return data in format of Time x Neurons with padding removed
        convolved_array = self.convolution(input_array, filter_array)
        convolved_array = convolved_array[0, pad_indices[0]:pad_indices[1], pad_indices[2]:pad_indices[3], 0]
        return convolved_array

    def shape_filter(self, coefficients):
        '''
        Checking existing dimensions and adding new ones if needed
        to create 4-D array filter. 

        if ndim == 2, Add empty in_channel, filter dimensions
        if ndim == 3, Add empty in_channel dimension

        When shaping our array, last dimension of filter/input
        are broadcasted
        '''
        if coefficients.ndim == 2:
            coefficients = coefficients[..., np.newaxis]
            coefficients = coefficients [..., np.newaxis]
        elif coefficients.ndim == 3:
            coefficients = coefficients[:, :, np.newaxis, :]

        return coefficients
    
    def shape_input(self, input_array, coefficients):
        '''
        Shapes our input data to fit convolve2d if non-4D is provided.
        Last dimension of input/filter are also broadcasted.

        if ndim == 2, Add empty batch, and filter dimension
        if ndim == 3, Add empty filter dimension
        '''
        if input_array.ndim == 2:
            input_array = input_array[..., np.newaxis]
            input_array = input_array[np.newaxis, ...]
        elif input_array.ndim == 3:
            input_array = input_array[..., np.newaxis]

        broad_input, broad_coeff = self.broadcast_arrays(input_array, coefficients)

        return broad_input, broad_coeff

    def convolution(self, input_array, filter_array):
        '''
        Applies scipy convolution with given input and filter array data.
        NOTE: Currently batch data is ignored... WIP

        Convolutions are applied filter_array.shape[-1] times and stacked.
        Stacked arrays are sent to pooling function to be reduced

        Parameters
        ----------
        input_array: 4-D ndarray 
            A 4 dimensional array of format (batch*height*width*in_channels)
        filter_array: 4-D ndarray 
            A 4 dimensional array of format (height*width*in_channels*filters)

        '''
        input_convolutions = [scipy.signal.convolve2d(input_array[0, :, :, 0], filter_array[:, :, 0, 0], mode='valid')[np.newaxis,..., np.newaxis] 
                              for idx in range(filter_array.shape[-1])]
        input_convolutions = np.stack(input_convolutions, axis=-1)
        return self.pool(input_convolutions)
    
    def pool(self, input_array):
        '''
        Pooling of total filters created from convolve2d function. Done via
        dimension reduction on the filter dimension via given pool_type.

        Default reduction is MEAN

        NOTE: Other reductions may be added/removed

        Parameters
        ----------
        input_array: 4-D ndarray
            Given array with completed convolutions to pool
        pool_type: String
            String value to determine type of reduction used. Includes:
            MAX - Reduction via average values along array 
            MIN - Reduction via minimum values 
            PROD - Reduction via production of values
            STD - Reduction via standard deviation
            SUM - Reduction via sum values
            
            default is mean reduction
        '''
        pool_type = self.pool_type
        if pool_type == 'MAX':
            pooled_array = np.max(input_array, axis=-1, keepdims=False)
        elif pool_type == 'MIN':
            pooled_array = np.min(input_array, axis=-1, keepdims=False)
        elif pool_type == 'PROD':
            pooled_array = np.prod(input_array, axis=-1, keepdims=False)
        elif pool_type == 'STD':
            pooled_array = np.std(input_array, axis=-1, keepdims=False)
        elif pool_type == 'SUM':
            pooled_array = np.sum(input_array, axis=-1, keepdims=False)
        else:
            pooled_array = np.mean(input_array, axis=-1, keepdims=False)
        return pooled_array
    
    
    def pad(self, input_array):
        '''
        Pads X and Y dimensions of input, for use with convolutions and strides
        Uses total shape/size of filter window to determine amount of padding. 
        Currently no way to provide a specified size.

        NOTE: Implement custom sizing for padding? 

        Parameters
        ----------
        input_array: 4-D ndarray
            Input array that we wish to pad values with
        pad_type: String
            String value to determine type of padding used. Includes:
            reflect - pads with reflection of edge values
            symmetric - 
            max - pads with abs max values
            min - pads with abs min values
            zero - pads with 0 values

            Default is constant of 0's
        '''
        y_pad = int(self.coefficients.shape[0]/2) + 1
        x_pad = int(self.coefficients.shape[1]/2) + 1
        if self.pad_axes == 'x':
            y_pad = 0
        elif self.pad_axes == 'y':
            x_pad = 0

        # Saving pad indices to remove later
        pad_indices = [int(y_pad/2), input_array.shape[1]+int(y_pad/2), int(x_pad/2), input_array.shape[2]+int(x_pad/2)]
        pad_array = [[0,0], [y_pad, y_pad], [x_pad, x_pad], [0,0]]

        if self.pad_type == 'reflect':
            input_array = np.pad(input_array, pad_array, mode='reflect')
        elif self.pad_type == 'symmetric':

            input_array = np.pad(input_array, pad_array, mode='symmetric')
        else:
            pad_constant = 0
            if self.pad_type == 'max':
                pad_constant = np.max(np.abs(input_array))
            elif self.pad_type == 'min':
                pad_constant = np.min(np.abs(input_array))
            input_array = np.pad(input_array, pad_array, mode='constant', constant_values=pad_constant)
        return input_array, pad_indices

    def as_tensorflow_layer(self, input_shape, **kwargs):
        """
        Converts Conv2d to Tensorflow Keras layer. Building 
        inner-class with NemsKerasLayer to make compatible
        with tensorflow backend
        
        Parameters
        ----------
        inputs : tf.Tensor or np.ndarray
            Initial input to Layer, supplied by TensorFlowBackend during model
            building.
        
        Returns
        -------
        Conv2dTF
        
        """
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        # Putting relevant data in scope for class call
        stride      = self.stride
        pad_type    = self.pad_type
        pool_type   = self.pool_type
        pad_axes    = self.pad_axes

        filters     = self.as_tf_shape_filter(self.coefficients)
        shape_input, shape_coeff = self.as_tf_shape_tensor(input_shape, filters)
        convolve    = self.as_tf_convolution()
        pool        = self.as_tf_pool(pool_type)
        _pad_indices, pad = self.as_tf_pad(input_shape, pad_type, pad_axes)

        class Conv2dTF(NemsKerasLayer):
            def weights_to_values(self):
                c = self.parameter_values['coefficients']
                return {'coefficients': c}

            def call(self, inputs):
                input_tensor = shape_input(inputs)
                filter_tensor = shape_coeff(filters)
                pad_indices = [0, input_tensor.shape[1], 0, input_tensor.shape[2]]
                if pad_type != 'None':
                    input_tensor = pad(input_tensor)
                    pad_indices = _pad_indices
                convolved_tensor = convolve(input_tensor, filter_tensor, stride)
                convolved_tensor = pool(convolved_tensor)[:, pad_indices[0]:pad_indices[1], pad_indices[2]:pad_indices[3], 0]
                return convolved_tensor

        return Conv2dTF(self, new_values={'coefficients': self.coefficients}, **kwargs)
    
    def as_tf_convolution(self):
        '''
        Convolutions on input_tensor with given filter_tensor and
        stride. Completed convolutions are sent back to pool to reduce
        dimensions

        Parameters
        ----------
        input_tensor: 4-D Tensor
            Modified tensor must be 4-Dimensional of
            Batch x Height x Width x In_channels
        filter_tensor: 4-D Tensor
            Modified tensor, also 4-Dimensional of
            Height x Width x In_channels x Filters

        NOTE: Currently only implemented with CPU in mind.
        '''
        #Temp CPU only
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        if num_gpus == 0:
                #NOTE: May need to look at this and make sure batch data isn't removed by this set up
            @tf.function
            def convolve(input_tensor, filter_tensor, stride):
                conv_fn = lambda t: tf.cast(
                    tf.nn.conv2d(tf.expand_dims(t[0], axis=0), tf.expand_dims(t[1], axis=0), stride, padding='VALID'),
                    tf.float64)
                input_convolutions = tf.map_fn(
                    fn=conv_fn,
                    elems=[input_tensor, filter_tensor],
                    dtype=tf.float64
                    )
                input_convolutions = tf.squeeze(input_convolutions, 1)
                return input_convolutions
        else:
            @tf.function
            def convolve(input_tensor, filter_tensor, stride):
                input_convolutions = tf.nn.conv2d(input_tensor, filter_tensor, stride, padding='VALID')
                return input_convolutions
        return convolve
    
    def as_tf_shape_filter(self, coefficients):
        '''
        Checking existing dimensions and adding new ones if needed
        to create 4-D array filter. 

        if ndim == 2, Add empty in_channel, filter dimensions
        if ndim == 3, Add empty in_channel dimension

        When shaping our tensor, last dimension of filter/input
        are broadcasted
        '''
        if coefficients.ndim == 2:
            coefficients = coefficients[..., np.newaxis]
            coefficients = coefficients [..., np.newaxis]
        elif coefficients.ndim == 3:
            coefficients = coefficients[:, :, np.newaxis, :]

        return coefficients
    
    def as_tf_shape_tensor(self, input_shape, coefficients):
        '''
        Shapes our input data to fit conv2d if non-4D is provided.
        In channels of our input and filters are broadcasted

        if ndim == 2, Add empty batch, and in_channel dimension
        if ndim == 3, Add empty in_channel dimension
        '''
        # Making batch explicit so np.empty works
        if input_shape[0] is None:
            input_shape = [1] + list(input_shape[1:])

        fake_input = np.empty(shape=input_shape)
        if fake_input.ndim == 2:
            fake_input = fake_input[..., np.newaxis]
            fake_input = fake_input[np.newaxis, ...]
            @tf.function()
            def expand_input(input_tensor):
                expanded_tensor = tf.expand_dims(input_tensor, axis=-1)
                expanded_tensor = tf.expand_dims(input_tensor, axis=0)
                return expanded_tensor
        elif fake_input.ndim == 3:
            fake_input = fake_input[..., np.newaxis]
            @tf.function()
            def expand_input(input_tensor):
                expanded_tensor = tf.expand_dims(input_tensor, axis=-1)
                return expanded_tensor
        else:
            @tf.function()
            def expand_input(input_tensor): return input_tensor
    
        broad_fake_input, broad_coeff = self.broadcast_arrays(fake_input, coefficients)
        fake_input_shape = fake_input.shape
        coeff_shape = coefficients.shape

        if broad_fake_input.shape[-1] > fake_input_shape[-1]:
            @tf.function()
            def shape_input(input_tensor):
                batch_size = tf.keras.backend.shape(input_tensor)[0]
                shape = [batch_size] + list(broad_fake_input.shape[1:])
                return tf.broadcast_to(expand_input(input_tensor), shape)
        else:
            @tf.function()
            def shape_input(input_tensor): return expand_input(input_tensor)
            
        if broad_coeff.shape[-2] > coeff_shape[-2]:
            @tf.function()
            def shape_coeff(coefficients):
                return tf.broadcast_to(coefficients, broad_coeff)
        else:
            @tf.function()
            def shape_coeff(coefficients): return coefficients

        return shape_input, shape_coeff
    
    def as_tf_pool(self, pool_type):
        '''
        Pooling of total filters created from conv2d function. Done via
        dimension reduction on the filter dimension via given pool_type.

        Default reduction is MEAN

        NOTE: Other reductions may be added/removed

        Parameters
        ----------
        input_tensor: 4D Tensor
            Given tensor with completed convolutions to pool
        pool_type: String
            String value to determine type of reduction used. Includes:
            MAX - Reduction via average values along tensor 
            MIN - Reduction via minimum values 
            PROD - Reduction via production of values
            STD - Reduction via standard deviation
            SUM - Reduction via sum values

        '''
        if pool_type == 'MAX':
            @tf.function
            def pool(input_tensor):
                return tf.math.reduce_max(input_tensor, axis=-1, keepdims=True)
        elif pool_type == 'MIN':
            @tf.function
            def pool(input_tensor):
                return tf.math.reduce_min(input_tensor, axis=-1, keepdims=True)
        elif pool_type == 'PROD':
            @tf.function
            def pool(input_tensor):
                return tf.math.reduce_prod(input_tensor, axis=-1, keepdims=True)
        elif pool_type == 'STD':
            @tf.function
            def pool(input_tensor):
                return tf.math.reduce_std(input_tensor, axis=-1, keepdims=True)
        elif pool_type == 'SUM':
            @tf.function
            def pool(input_tensor):
                return tf.math.reduce_sum(input_tensor, axis=-1, keepdims=True)
        else:
            @tf.function
            def pool(input_tensor):
                return tf.math.reduce_mean(input_tensor, axis=-1, keepdims=True)
        return pool
    
    def as_tf_pad(self, input_shape, pad_type, pad_axes):
        '''
        Pads X and Y dimensions of input, for use with convolutions and strides
        Uses total shape/size of filter window to determine amount of padding. 
        Currently no way to provide a specified size.

        NOTE: Implement custom sizing for padding? 

        Parameters
        ----------
        input_tensor: 4D-Tensor
            Input tensor that we wish to pad values with
        pad_type: String
            String value to determine type of padding used. Includes:
            reflect - pads with reflection of edge values
            symmetric - 
            max - pads with abs max values
            min - pads with abs min values
            zero - pads with 0 values

            Default is constant of 0's
        '''
        y_pad = int(self.coefficients.shape[0]/2) + 1
        x_pad = int(self.coefficients.shape[1]/2) + 1
        if pad_axes == 'x':
            y_pad = 0
        elif pad_axes == 'y':
            x_pad = 0

        # Saving pad indices to remove later
        pad_indices = [int(y_pad/2), input_shape[1]+int(y_pad/2), int(x_pad/2), input_shape[2]+int(x_pad/2)]
        pad_array = tf.constant([[0,0], [y_pad, y_pad], [x_pad, x_pad], [0,0]])

        
        if pad_type == 'reflect':
            @tf.function
            def pad(input_tensor):
                return tf.pad(input_tensor, pad_array, mode='REFLECT')
        elif pad_type == 'symmetric':
            @tf.function
            def pad(input_tensor):
                return tf.pad(input_tensor, pad_array, mode='SYMMETRIC')
        else:
            pad_constant = 0
            if pad_type == 'max':
                pad_constant = np.max(np.abs(self.input))
            elif pad_type == 'min':
                pad_constant = np.min(np.abs(self.input))
            @tf.function
            def pad(input_tensor):
                return tf.pad(input_tensor, pad_array, mode='CONSTANT', constant_values=pad_constant)
        return pad_indices, pad
    
    def broadcast_arrays(self, input_array, coefficients):
        '''
        Interal class function for broadcasting array and coefficients 
        based on their input_channels. Compares in_channels and broadcasts 
        one array to the other. Returns broadcasted array or 4-Tuple coefficients

        Parameters
        ----------
        input_array: 4-D ndarray
            Input array to compare and broadcast to/from
        coefficients: 4-tuple
            coefficients shape values for filters to broadcast to/from
        '''
        # Removing batch layer, to set both arrays in_channels to [-2]
        fake_input = input_array[0,..., np.newaxis]

        if input_array.shape[-2] < coefficients.shape[-2]:
            try:
                input_array = broadcast_axes(fake_input, coefficients, axis=-2)
            except ValueError:
                raise TypeError(
                    "Last dimension of FIR input must match last dimension of "
                    "coefficients, or one must be broadcastable to the other."
                    )
        elif coefficients.shape[-2] < input_array.shape[-2]:
            try:
                coefficients = broadcast_axes(coefficients, fake_input, axis=-2)
            except ValueError:
                raise TypeError(
                    "Last dimension of FIR input must match last dimension of "
                    "coefficients, or one must be broadcastable to the other."
                    )
        return input_array, coefficients

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

    @layer('conv2d')
    def from_keyword(keyword):
        '''
        Keyword layer call. 'conv2d' should call this layer
        with default values
        
                Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape.
            (time, input channels a.k.a. rank, ..., output channels) 

        See also
        --------
        Layer.from_keyword
        
        '''
        kwargs = {}
        Conv2d_class = Conv2d

        options = keyword.split('.')
        kwargs['shape'] = pop_shape(options)
        for op in options:
            if op.startswith('s'):
                kwargs['stride'] = int(op[1:])
        conv = Conv2d_class(**kwargs)

        return conv
