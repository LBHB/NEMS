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

    def __init__(self, input1='stim', input2='hrtf', axis=1, input=None, **kwargs):
        self.axis = axis
        self.input1 = input1
        self.input2 = input2

        super().__init__(input=[input1, input2], **kwargs)

    @layer('cat')
    def from_keyword(keyword):
        """Construct ConcatSignals from keyword."""
        options = keyword.split('.')
        kwargs={}
        if len(options)>1:
            kwargs['input1']=options[1]
        if len(options)>2:
            kwargs['input2']=options[2]

        return ConcatSignals(**kwargs)

    def evaluate(self, input1, input2):
        # All inputs are treated the same, no fittable parameters.

        return np.concatenate([input1, input2], axis=self.axis)

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
                return tf.concat([inputs[0], inputs[1]], ax)

        return ConcatSignalsTF(self, **kwargs)

class MultiplySignals(Layer):

    def __init__(self, input1='hrtf', input2='input', input=None, **kwargs):
        self.input1 = input1
        self.input2 = input2

        super().__init__(input=[input1, input2], **kwargs)

    @layer('mult')
    def from_keyword(keyword):
        """Construct MultiplySignals from keyword."""
        options = keyword.split('.')
        kwargs={}
        if len(options)>1:
            kwargs['input1']=options[1]
        if len(options)>2:
            kwargs['input2']=options[2]
        kwargs['output']='hstim'
        return MultiplySignals(**kwargs)

        
    def evaluate(self, input1, input2):
        # All inputs are treated the same, no fittable parameters.
        return input1*input2

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class MultiplySignalsTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.
                return tf.math.multiply(inputs[0], inputs[1])

        return MultiplySignalsTF(self, **kwargs)


class MultiplyByExp(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @layer('multexp')
    def from_keyword(keyword):
        """Construct MultiplySignals from keyword."""
        return MultiplyByExp()

    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.
        return inputs[0] * np.exp(inputs[1])

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class MultiplyByExpTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.
                return tf.math.multiply(inputs[0], tf.math.exp(inputs[1]))

        return MultiplyByExpTF(self, **kwargs)

