import numpy as np

from nems.registry import layer
from .base import Layer, Phi, Parameter


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

class ConcatSignals(Layer):

    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.

        return np.concatenate(inputs, axis=self.axis)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        # TODO: how to deal with batch dimension. Currently, kludged to add 1 to axis
        ax = self.axis+1
        class ConcatSignalsTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.
                return tf.concat(inputs, ax)

        return ConcatSignalsTF(self, **kwargs)
