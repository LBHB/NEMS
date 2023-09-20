from .base import Layer, Phi, Parameter
import tensorflow as tf
import numpy as np
from .tools import require_shape, pop_shape
from nems.distributions import Normal, HalfNormal
from nems.registry import layer
from nems.tools.arrays import broadcast_axes


# NOTE: WIP, some things may not work as intended
class Conv2d(Layer):
    '''
    NEMS compatible convolutional 2-D layer. Currently
    for tensorflow.

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
        self.pad_indices = [0, 0, 0, 0]
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
        prior = Normal(mean, sd)
        coefficients = Parameter(name='coefficients', shape=self.shape,
                                 prior=prior)
        return Phi(coefficients)

    def evaluate(self, input):
        '''
        '''
        filter_tensor = self.shape_filter(self.coefficients)
        input_tensor, filter_tensor = self.shape_tensor(input, filter_tensor)
        if self.pad_type != 'None':
            input_tensor = self.pad(input_tensor)
            
        # Should return data in format of Time x Neurons with padding removed
        pad_indices = self.pad_indices
        convolved_tensor = self.convolution(input_tensor, filter_tensor)[0, pad_indices[0]:pad_indices[1], pad_indices[2]:pad_indices[3]]
        return convolved_tensor

    def shape_filter(self, coefficients):
        '''
        Checking existing dimensions and adding new ones if needed
        to create 4D array filter. 

        if ndim == 2, Add empty batch, out dimensions
        if ndim == 3, Add empty batch dimension

        When shaping our tensor, last dimension of filter/input
        are broadcasted
        '''
        if coefficients.ndim == 2:
            coefficients = coefficients[..., np.newaxis]
            coefficients = coefficients [..., np.newaxis]
        elif coefficients.ndim == 3:
            coefficients = coefficients[0:1, np.newaxis, :]

        return coefficients
    
    def shape_tensor(self, input_tensor, coefficients):
        '''
        Shapes our input data to fit conv2d if non-4D is provided.
        Last dimension of tensor/filter are also broadcasted.

        if ndim == 2, Add empty batch, and filter dimension
        if ndim == 3, Add empty filter dimension
        '''
        if input_tensor.ndim == 2:
            input_tensor = input_tensor[..., np.newaxis]
            input_tensor = input_tensor[np.newaxis, ...]
        elif input_tensor.ndim == 3:
            input_tensor = input_tensor[..., np.newaxis]
        
        if input_tensor.shape[-1] < coefficients.shape[-1]:
            try:
                input_tensor = broadcast_axes(input_tensor, coefficients, axis=-1)
            except ValueError:
                raise TypeError(
                    "Last dimension of FIR input must match last dimension of "
                    "coefficients, or one must be broadcastable to the other."
                    )
        elif coefficients.shape[-1] < input_tensor.shape[-1]:
            try:
                coefficients = broadcast_axes(coefficients, input_tensor, axis=-1)
            except ValueError:
                raise TypeError(
                    "Last dimension of FIR input must match last dimension of "
                    "coefficients, or one must be broadcastable to the other."
                    )
        
        return input_tensor, coefficients

    def convolution(self, input_tensor, filter_tensor):
        '''
        Convolutions on input_tensor with given filter_tensor and
        stride. Completed convolutions are sent back to pool to reduce
        dimensions
        '''
        input_convolutions = tf.nn.conv2d(input_tensor, filter_tensor, self.stride, padding='VALID') 

        return self.pool(input_convolutions)
    
    def pool(self, input_tensor):
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
            AVG - Reduction via average values along tensor 
            MIN - Reduction via minimum values 
            PROD - Reduction via production of values
            STD - Reduction via standard deviation
            SUM - Reduction via sum values

        '''
        pool_type = self.pool_type
        if pool_type == 'AVG':
            pooled_tensor = tf.math.reduce_max(input_tensor, axis=-1, keepdims=False)
        elif pool_type == 'MIN':
            pooled_tensor = tf.math.reduce_min(input_tensor, axis=-1, keepdims=False)
        elif pool_type == 'PROD':
            pooled_tensor = tf.math.reduce_prod(input_tensor, axis=-1, keepdims=False)
        elif pool_type == 'STD':
            pooled_tensor = tf.math.reduce_std(input_tensor, axis=-1, keepdims=False)
        elif pool_type == 'SUM':
            pooled_tensor = tf.math.reduce_sum(input_tensor, axis=-1, keepdims=False)
        else:
            pooled_tensor = tf.math.reduce_mean(input_tensor, axis=-1, keepdims=False)

        return pooled_tensor
    
    
    def pad(self, input_tensor):
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
        y_pad = int(self.coefficients.shape[1]/4+1)
        x_pad = int(self.coefficients.shape[0]/4+1)

        # Saving pad indices to remove later
        self.pad_indices = [int(x_pad/2)-1, int(input_tensor.shape[1]-(x_pad/2))+1, int(y_pad/2)-1, int(input_tensor.shape[2]-(y_pad/2))+1]

        if self.pad_axes == 'x':
            y_pad = 0
        elif self.pad_axes == 'y':
            x_pad = 0

        pad_array = tf.constant([[0,0], [y_pad,y_pad], [x_pad,x_pad], [0,0]])

        if self.pad_type == 'reflect':
            input_tensor = tf.pad(input_tensor, pad_array, mode='REFLECT')
        elif self.pad_type == 'symmetric':
            input_tensor = tf.pad(input_tensor, pad_array, mode='SYMMETRIC')
        else:
            pad_constant = 0
            if self.pad_type == 'max':
                pad_constant = np.max(np.abs(self.input))
            elif self.pad_type == 'min':
                pad_constant = np.min(np.abs(self.input))
            input_tensor = tf.pad(input_tensor, pad_array, mode='CONSTANT', constant_values=pad_constant)
        return input_tensor

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
