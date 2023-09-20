from .base import Layer, Phi, Parameter
import tensorflow as tf
import numpy as np
from .tools import require_shape, pop_shape
from nems.distributions import Normal, HalfNormal
from nems.registry import layer
from nems.tools.arrays import broadcast_axes


# NOTE: WIP, currently only partially-functional
class Conv2d(Layer):
    '''
    NEMS compatible convolutional 2-D layer. Currently
    for tensorflow.

    Shape should be of format;
    (Batch x Time x Neurons x Filters) or,
    (Time x Neurons x Filters) or,
    (Time x Neurons)

    If batches are needed, full 4-D shape must be provided.

    if shape == 3, we assume filters are provided, and include defaults for batches
    if shape == 2, we provided defaults for filters and batches

    '''
    def __init__(self, stride=1, pad_type='zero', pad_axes='both',
                  pool_type='mean', **kwargs):
        '''
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
        '''
        mean = np.full(shape=self.shape, fill_value=0.0)
        sd = np.full(shape=self.shape, fill_value=1/np.prod(self.shape))
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
        '''
        '''
        filter_tensor = self.shape_filter(self.coefficients)
        input_tensor, filter_tensor = self.shape_tensor(input, filter_tensor)
        #input_tensor = tf.cast(tf.reshape(input, shape=[1, input.shape[0], input.shape[1], 1]), tf.float32)
        #filter_tensor = tf.cast(tf.Variable(tf.constant(0, shape=[self.coefficients.shape[0], self.coefficients.shape[1], 1, 1])), tf.float32)
        if self.pad_type != 'None':
            input_tensor = self.pad(input_tensor)
            
        # Should return data in original format of Batch x Time x Neurons
        return self.convolution(input_tensor, filter_tensor)

    def shape_filter(self, coefficents):
        '''
        Adds filter dims if non are provided.

        NOTE: Is flipping rank needed in conv2d?
        Flips non time/filter dimensions for convolutions
        '''
        coefficients = self.coefficients
        if coefficients.ndim == 2:
            # Add a dummy filter/output axis
            coefficients = coefficients[..., np.newaxis]
            coefficients = coefficients [..., np.newaxis]
        elif coefficients.ndim == 3:
            coefficients = coefficients[0:1, np.newaxis, :]


        '''
        NOTE: Prob delete this
        # Coefficients are applied "backwards" (convolution) relative to how
        # they are specified (filter), so have to flip all dimensions except
        # time and number of filters/outputs.
        flipped_axes = [1]  # Always flip rank
        other_dims = coefficients.shape[2:-1]
        for i, d in enumerate(other_dims):
            # Also flip any additional dimensions
            flipped_axes.append(i+2)
        coefficients = np.flip(coefficients, axis=flipped_axes)
        '''


        return coefficients
    
    def shape_tensor(self, input_tensor, coefficients):
        '''
        Shapes our input data to fit conv2d if non-4D is provided
        
        '''
        if input_tensor.ndim == 2:
            # Add a dummy filter/output axis
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
        Convolutions on input_tensor with given filter_tensor.
        Total convolutions are stacked, and reshaped to the same
        dimension as original input, before being pooled.
        '''
        #input_convolutions = [tf.nn.conv2d(input_tensor, filter_tensor, self.stride, padding='VALID') 
        #                     for idx in range(filter_tensor.shape[-1])]
        input_convolutions = tf.nn.conv2d(input_tensor, filter_tensor, self.stride, padding='VALID') 
        #convolved = tf.stack(input_convolutions, -1)
        #reshape_height = int(convolved.shape[0]*convolved.shape[1]/2)
        #reshape_width = int(convolved.shape[0]*convolved.shape[2]/2)
        #convolved = tf.reshape(convolved, [1, reshape_height, reshape_width, 1])

        return self.pool(input_convolutions)
    
    def pool(self, input_tensor):
        '''
        temp tests
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

        NOTE: Implement custom sizing for padding? 
        '''
        y_pad = int(self.coefficients.shape[1]/4+1)
        x_pad = int(self.coefficients.shape[0]/4+1)
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

    @layer('c2d')
    def from_keyword(keyword):
        '''
        Keyword layer calls. 'c2d' should call this function
        with default values
        '''
        return
    
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