import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter
from .tools import require_shape, pop_shape


# [AGENT EDIT START | agent: claude | user: sbp894 | reason: port ACNet's Conv1DRowLayer (one causal FIR filter per channel, no cross-channel mixing) as a NEMS Layer -- nems.layers.FiniteImpulseResponse sums across channels (STRF-style) and can't express this, confirmed by reading filter.py's evaluate() | date: 2026-09-16]
class DepthwiseFIR(Layer):
    """Convolve one independent causal FIR filter per input channel.

    Unlike `nems.layers.FiniteImpulseResponse` (which sums its filtered
    channels together, STRF-style, into one or more output channels),
    `DepthwiseFIR` keeps every channel independent: output channel `c` is
    only a function of input channel `c`. This matches ACNet's
    `Conv1DRowLayer` (`PT_EncMdl_helpers_v2.py`), a depthwise
    (`groups=num_filters`) `nn.Conv1d` with explicit causal padding
    `(kernel_size-1, 0)`.

    Parameters
    ----------
    shape : 2-tuple of int (kernel_size, channels).

    See also
    --------
    nems.layers.filter.FiniteImpulseResponse
    nems.layers.base.Layer

    Examples
    --------
    >>> dfir = DepthwiseFIR(shape=(7, 80))
    >>> x = np.random.rand(1000, 80)  # (time, channels)
    >>> out = dfir.evaluate(x)
    >>> out.shape
    (1000, 80)

    """

    def __init__(self, **kwargs):
        require_shape(self, kwargs, minimum_ndim=2, maximum_ndim=2)
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Get initial values for `DepthwiseFIR.parameters`.

        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `DepthwiseFIR.shape` (kernel_size, channels).
            Prior: Normal(mean=0, sd=1/kernel_size).

        Returns
        -------
        nems.layers.base.Phi

        """
        kernel_size = self.shape[0]
        mean = np.zeros(shape=self.shape)
        sd = np.full(shape=self.shape, fill_value=1/kernel_size)
        coefficients = Parameter(
            name='coefficients', shape=self.shape, prior=Normal(mean, sd)
            )
        return Phi(coefficients)

    @property
    def coefficients(self):
        """Per-channel filter, `coefficients.shape = DepthwiseFIR.shape`."""
        return self.parameters['coefficients'].values

    def evaluate(self, input):
        """Causally convolve each channel of `input` with its own filter.

        Parameters
        ----------
        input : np.ndarray
            Shape (T, N), N must match `DepthwiseFIR.shape[1]`.

        Returns
        -------
        np.ndarray
            Shape (T, N).

        """
        kernel_size = self.shape[0]
        padded = np.pad(input, ((kernel_size - 1, 0), (0, 0)))
        # windowed: (T, N, kernel_size) -- a zero-copy view.
        windowed = sliding_window_view(padded, kernel_size, axis=0)
        # coef[0] should correspond to lag=0 (most recent sample), matching
        # FiniteImpulseResponse's convention.
        coef_t = np.flip(self.coefficients, axis=0)
        return np.einsum('tnf,fn->tn', windowed, coef_t)

    @layer('dfir')
    def from_keyword(keyword):
        """Construct DepthwiseFIR from keyword.

        Keyword options
        ---------------
        {digit}x{digit} : (kernel_size, channels) shape; required.

        Returns
        -------
        DepthwiseFIR

        See also
        --------
        Layer.from_keyword

        """
        options = keyword.split('.')
        shape = pop_shape(options)
        return DepthwiseFIR(shape=shape)

    def as_tensorflow_layer(self, **kwargs):
        """Return a Keras layer that applies the same depthwise causal FIR."""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        kernel_size = self.shape[0]

        class DepthwiseFIRTF(NemsKerasLayer):
            def call(self, inputs):
                # inputs: (batch, T, N). depthwise_conv2d needs a 4D NHWC
                # tensor and a (filter_height, filter_width, in_channels,
                # channel_multiplier) filter -- treat time as "width" with a
                # singleton "height", channel_multiplier=1 (one filter per
                # input channel, no channel expansion).
                x = tf.pad(inputs, [[0, 0], [kernel_size - 1, 0], [0, 0]])
                x = x[:, tf.newaxis, :, :]

                coef_t = tf.reverse(self.coefficients, axis=[0])
                filt = coef_t[tf.newaxis, :, :, tf.newaxis]

                out = tf.nn.depthwise_conv2d(
                    x, filt, strides=[1, 1, 1, 1], padding='VALID'
                    )
                return out[:, 0, :, :]

        return DepthwiseFIRTF(self, **kwargs)
# [AGENT EDIT END]
