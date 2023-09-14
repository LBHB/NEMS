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
    def __init__(self, filters, stride=1, pad_type='mean', pad_axes='both',
                  pool_type='mean', **kwargs):
        '''
        '''
        require_shape(self, kwargs, minimum_ndim=2)
        self.filters     = filters
        self.stride      = stride
        self.pad_type    = pad_type
        self.pad_axes    = pad_axes
        self.pool_type   = pool_type
        super().__init__(**kwargs)

        # NOTE: Confirm best course for getting kernel_size values
        self.kernel_size = self.shape

    
    def initial_parameters(self):
        '''
        '''
        mean = np.full(shape=self.shape, fill_value=0.0)
        sd = np.full(shape=self.shape, fill_value=1/np.prod(self.shape))
        prior = Normal(mean, sd)
        coefficients = Parameter(name='coefficients', shape=self.shape,
                                 prior=prior)
        return Phi(coefficients)

    def evaluate(self):
        '''
        '''
        self.convolution()
        return

    def convolution(self):
        '''
        '''
        input_tensor = tf.constant(self.input)
        filter_tensor = tf.constant()
        self.pad()
        tf.nn.conv2d(input_tensor, filter_tensor, self.stride, padding='VALID')
        self.pooling()
        return
    
    def pooling(self):
        '''
        '''
        if self.pool_type == 'absmax':
            pass
        elif self.pool_type == 'absmin':
            pass
        # Default mean pooling
        else:
            pass
        tf.nn.pool()
        return
    
    def pad(self):
        '''
        '''
        if self.pad_type == 'absmax':
            pass
        elif self.pad_type == 'absmin':
            pass
        # Default mean padding
        else:
            pass
        
        if self.pad_axes == 'x':
            pass
        elif self.pad_axes == 'y':
            pass
        # Default both axes
        else:
            pass

        tf.pad()
        return
    
    def weights_to_phi(self):
        '''
        Returns Phi built via weights of tensor
        '''
        phi = {'kernel': self.kernel.numpy(),
               'bias': self.bias.numpy()}
        return phi

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