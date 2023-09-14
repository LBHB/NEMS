from .base import Layer, Phi, Parameter
import tensorflow as tf
import numpy as np
from .tools import require_shape, pop_shape
from nems.distributions import Normal, HalfNormal
from nems.registry import layer

# NOTE: WIP, currently non-functional
class Conv2d(Layer):
    '''
    NEMS compatible convolutional 2-D layer. Currently
    for tensorflow.
    '''
    def __init__(self, stride=1, pad_type='zero', pad_axes='both',
                  pool_type='mean', **kwargs):
        '''
        NOTE: Any way to make filter simpler? 
        Filter: 4D array format [filter_height, filter_width, in_channels, out_channels]
        '''
        require_shape(self, kwargs, minimum_ndim=2)
        self.stride      = stride
        self.pad_type    = pad_type
        self.pad_axes    = pad_axes
        self.pool_type   = pool_type
        super().__init__(**kwargs)

    def initial_parameters(self):
        '''
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
        return self.convolution(input)

    def convolution(self, input):
        '''
        '''
        #input_tensor = tf.cast(tf.reshape(input, [1, input.shape[0], input.shape[1], 1] ), tf.float32)
        input_tensor = input
        filter_tensor = tf.cast(tf.reshape(self.coefficients, [self.coefficients.shape[0], self.coefficients.shape[1], 1, self.coefficients.shape[-1]]), tf.float32)
        if self.pad_type != 'None':
            self.pad(input_tensor)
        
        convolved = []
        for idx in self.coefficients[-2]:
            input_convolution = tf.nn.conv2d(input_tensor, filter_tensor[...,idx], self.stride, padding='VALID')
            convolved.append(input_convolution)

        return self.pooling(convolved)
    
    def pooling(self, input_tensor):
        '''
        '''
        return tf.nn.pool(input_tensor, self.coefficients[0:1], pooling_type=self.pool_type)
    
    def pad(self, input_tensor):
        '''
        NOTE: Implement custom sizing for padding? 
        '''
        y_axes, x_axes = (1, 1)
        if self.pad_axes == 'x':
            y_axes = 0
        elif self.pad_axes == 'y':
            x_axes = 0

        y_kernel = int(self.coefficients.shape[1]/2+1)*y_axes
        x_kernel = int(self.coefficients.shape[0]/2+1)*x_axes
        pad_amounts = [y_kernel, x_kernel]
        pad_array = tf.constant([[pad_amounts[0],pad_amounts[0] ], [pad_amounts[1], pad_amounts[1]]])

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