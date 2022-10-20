import numpy as np

from nems.registry import layer
from nems.distributions import Normal, HalfNormal
from .base import Layer, Phi, Parameter, ShapeError


# TODO: double check all shape references after dealing w/ data order etc,
#       make sure correct dims are lined up.
class SwapDims(Layer):
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
    >>> sd = SwapDims(dim1=1, dim2=2)
    >>> input = np.random.rand(10000, 18, 2)  # (time, channels, bands)
    >>> out = np.moveaxis(input, self.dim1, self.dim2)      # sd.evaluate(input)
    >>> out.shape
    (10000, 2, 18)

    """
    def __init__(self, dim1=1, dim2=2, **kwargs):
        self.dim1=dim1
        self.dim2=dim2
        super().__init__(**kwargs)

    def initial_parameters(self):
        """No parameters

        """
        return Phi()

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

        return np.moveaxis(input, [self.dim1, self.dim2], [self.dim2, self.dim1])

    @layer('sd')
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
        kwargs = {}
        options = keyword.split('.')
        
        if len(options)>1:
            kwargs['dim1']=int(options[1])
        if len(options)>2:
            kwargs['dim2']=int(options[2])
            
        return SwapDims(**kwargs)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs
        
        NOTE: This is currently hard-coded to dot the 2nd dim of the input
        (batch, time, channels, ...) and first dim of coefficients
        (channels, rank, ...).
        
        """
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer
        dim1 = self.dim1+1
        dim2 = self.dim2+1
        class SwapDimsTF(NemsKerasLayer):
            @tf.function
            def call(self, inputs):
                out = tf.experimental.numpy.moveaxis(inputs, [dim1, dim2], [dim2, dim1])
                return out
        
        return SwapDimsTF(self, **kwargs)

    @property
    def plot_kwargs(self):
        """Add incremented labels to each output channel for plot legend.
        
        See also
        --------
        Layer.plot
        
        """
        kwargs = {}
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
        return {'legend': True}

