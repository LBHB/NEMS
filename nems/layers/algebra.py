import numpy as np

from nems.registry import layer
from nems.distributions import Normal, HalfNormal
from .base import Layer, Phi, Parameter
from .tools import require_shape, pop_shape
from .weight_channels import WeightChannels

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

class ConcatSkip(WeightChannels):

    def __init__(self, axis=1, input=None, **kwargs):

        if input is None:
            raise ValueError('value for input required')
        require_shape(self, kwargs, minimum_ndim=2)

        self.axis = axis
        super().__init__(input=input, **kwargs)

    @layer('skip')
    def from_keyword(keyword):
        """Construct ConcatSkip from keyword."""

        options = keyword.split('.')

        kwargs = {'input': []}
        kwargs['shape'] = pop_shape(options)
        for op in options[1:]:
            if op.startswith('l2'):
                kwargs['regularizer'] = op
            else:
                kwargs['input'].append(op)

        return ConcatSkip(**kwargs)

    def initial_parameters(self):
        """Get initial values for `ConcatSkip.parameters`.
        
        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `self.shape`.
            Prior:  zero +/ 0.1
            Bounds: TODO

        Returns
        -------
        nems.layers.base.Phi

        """
        mean = np.full(shape=self.shape, fill_value=0.0)
        sd = np.full(shape=self.shape, fill_value=0.1)
        prior = Normal(mean, sd)
        coefficients = Parameter(
            name='coefficients', shape=self.shape, prior=prior
            )
        return Phi(coefficients)
        
    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.
        stepsize = int(self.coefficients.shape[1]/(len(inputs)-1))
        inputs_=[inputs[0]] + [
            np.tensordot(inp, self.coefficients[:inp.shape[1],(i*stepsize):((i+1)*stepsize)], axes=(1, 0))
            for i, inp in enumerate(inputs[1:])
        ]
        return np.concatenate(inputs_, axis=self.axis)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        # TODO: how to deal with batch dimension. Currently, kludged to add 1 to axis
        ax = self.axis+1
        stepsize = int(self.coefficients.shape[1]/(len(self.input)-1))

        class ConcatSkipTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.

                inputs_ = [inputs[0]] + [
                    tf.tensordot(inp, self.coefficients[:inp.shape[2], (i*stepsize):((i+1)*stepsize)], axes=[[2], [0]])
                    for i,inp in enumerate(inputs[1:])]
                return tf.concat(inputs_, ax)

        return ConcatSkipTF(self, **kwargs)


class ConcatSignals(Layer):

    def __init__(self, input1='stim', input2='hrtf', axis=1, input=None, compute_sum=False, **kwargs):
        self.axis = axis
        if input is not None:
            if len(input)>0:
                self.input1 = input[0]
            if len(input)>1:
                self.input2 = input[1]
        else:
            self.input1 = input1
            self.input2 = input2
            input = [input1, input2]
        self.compute_sum = compute_sum
        super().__init__(input=input, **kwargs)

    @layer('cat')
    def from_keyword(keyword):
        """Construct ConcatSignals from keyword."""
        options = keyword.split('.')
        kwargs={'input': []}
        for op in options[1:]:
            if op == 's':
                kwargs['compute_sum'] = True
            else:
                kwargs['input'].append(op)

        return ConcatSignals(**kwargs)

    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.

        if self.compute_sum:
            inputs_=[inputs[0]] + [i.sum(axis=self.axis, keepdims=True) for i in inputs[1:]]
            return np.concatenate(inputs_, axis=self.axis)
        else:
            return np.concatenate(inputs, axis=self.axis)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        # TODO: how to deal with batch dimension. Currently, kludged to add 1 to axis
        ax = self.axis+1
        compute_sum = self.compute_sum
        class ConcatSignalsTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.
                if compute_sum:
                    inputs_ = [inputs[0]] + [tf.math.reduce_sum(i, axis=ax, keepdims=True) for i in inputs[1:]]
                    return tf.concat(inputs_, ax)
                else:
                    return tf.concat(inputs, ax)

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


class ApplyHRTF(Layer):

    def __init__(self, input1='hrtf', input2='stim', input=None, **kwargs):
        input = [input1, input2]
        super().__init__(input=input, **kwargs)

    @layer('mult')
    def from_keyword(keyword):
        """Construct MultiplySignals from keyword."""
        options = keyword.split('.')
        kwargs = {}
        if len(options) > 1:
            kwargs['input1'] = options[1]
        if len(options) > 2:
            kwargs['input2'] = options[2]
        kwargs['output'] = 'hstim'
        return ApplyHRTF(**kwargs)

    def evaluate(self, input1, input2):
        # All inputs are treated the same, no fitable parameters.
        s = list(input1.shape)
        if s[-2]==2:
            #s = s[:-3] + [s[-3]*s[-2], s[-1]]
            #x = np.reshape(input1, s) * input2[..., np.newaxis]
            x = np.concatenate((input1[...,0,:],input1[...,1,:]),axis=-2) * input2[..., np.newaxis]
        else:
            x = input1 * input2[..., np.newaxis]

        m = int(x.shape[-2]/2)
        x = x[..., :m, :] + x[..., m:, :]
        x = np.reshape(np.swapaxes(x, -1, -2), [x.shape[0], x.shape[1]*x.shape[2]])

        return x

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class ApplyHRTFTF(NemsKerasLayer):
            def call(self, inputs):
                # Assume inputs is a list of two tensors
                # TODO: Use tensor names to not require this arbitrary order.
                m = int(tf.shape(inputs[1])[-1] / 2)

                # add power
                #x = tf.math.multiply(inputs[0], tf.math.square(tf.expand_dims(inputs[1], -1)))
                #x = tf.math.sqrt(tf.nn.relu(tf.math.add(x[...,:m,:], x[...,m:,:])))

                # apply to input channels separately, then stack
                s = list(inputs[0].shape)
                if s[-2]==2:
                    #s = [-1] + s[1:-3] + [s[-3]*s[-2], s[-1]]
                    #in0 = tf.reshape(inputs[0], s)
                    in0 = tf.concat((inputs[0][...,0,:],inputs[0][...,1,:]), axis=-2)
                    x = tf.math.multiply(in0, tf.expand_dims(inputs[1], -1))
                else:
                    x = tf.math.multiply(inputs[0], tf.expand_dims(inputs[1], -1))
                x = tf.math.add(x[..., :m, :], x[..., m:, :])

                x_shape = tf.shape(x)
                x = tf.reshape(tf.transpose(x, [0, 1, 3, 2]),
                               [-1, x_shape[1], tf.reduce_prod(x_shape[2:])])
                return x

        return ApplyHRTFTF(self, **kwargs)


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

