from .base import Layer, Phi, Parameter
import tensorflow as tf
import numpy as np
from .tools import require_shape, pop_shape
from nems.distributions import Normal, HalfNormal
from nems.registry import layer

# NOTE: WIP, currently only partially-functional
class Conv2d(Layer):
    '''
    NEMS compatible convolutional 2-D layer. Currently
    for tensorflow.
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

    def evaluate(self, input):
        '''
        '''
        self.x_window    = input.shape[0]
        self.y_window    = input.shape[1]
        input_tensor = tf.cast(tf.reshape(input, shape=[1, input.shape[0], input.shape[1], 1]), tf.float32)
        filter_tensor = tf.cast(tf.Variable(tf.constant(0, shape=[self.coefficients.shape[0], self.coefficients.shape[1], 1, 1])), tf.float32)
        if self.pad_type != 'None':
            input_tensor = self.pad(input_tensor)
            
        # Should return data in original format of X * Y
        return self.convolution(input_tensor, filter_tensor)[0,:,:,0]

    def convolution(self, input_tensor, filter_tensor):
        '''
        Convolutions on input_tensor with given filter_tensor.
        Total convolutions are stacked, and reshaped to the same
        dimension as original input, before being pooled.
        '''
        input_convolutions = [tf.nn.conv2d(input_tensor, filter_tensor, self.stride, padding='VALID')[0,:,:,:] 
                              for idx in range(self.coefficients.shape[-1])]
        convolved = tf.stack(input_convolutions, 0)
        reshape_height = int(convolved.shape[0]*convolved.shape[1]/2)
        reshape_width = int(convolved.shape[0]*convolved.shape[2]/2)
        convolved = tf.reshape(convolved, [1, reshape_height, reshape_width, 1])

        return self.pooling(convolved)
    
    def pooling(self, input_tensor):
        '''
        Pools given input_tensor by k_size, from original input sizes.
        Reduces size of filters in tensor either by average, or max, 
        depending on user input

        NOTE: k_size was needs a more thoughtfull decision, or
        provided option for users
        '''
        k_size = [self.x_window, self.y_window]
        if self.pool_type == 'AVG':
            pooled_tensor = tf.nn.avg_pool2d(input_tensor, k_size, 1, 'VALID')
        else:
            pooled_tensor = tf.nn.max_pool2d(input_tensor, k_size, 1, 'VALID')
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