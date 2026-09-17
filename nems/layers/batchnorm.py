import numpy as np

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter
from .tools import require_shape, pop_shape


# [AGENT EDIT START | agent: claude | user: sbp894 | reason: port ACNet's Batchnorm1d_txp as a NEMS Layer with true train-time batch stats / eval-time running EMA stats, for the ACNet-in-NEMS model port | date: 2026-09-16]
class BatchNorm1d(Layer):
    """Normalize input across every axis but the last (channel) axis.

    Ported from ACNet's `Batchnorm1d_txp` (`PT_EncMdl_helpers_v2.py`), which
    applies `nn.BatchNorm1d` to a (batch, time, channels) tensor transposed so
    that channels come first -- i.e. one running mean/var/gamma/beta per
    channel, computed jointly across every (batch, time) element for that
    channel. This Layer reproduces that split exactly:

    - **Training** (`train_mode()`): normalizes using the current batch's own
      mean/(biased) variance, and updates `running_mean`/`running_var` with an
      exponential moving average (`running = (1-momentum)*running +
      momentum*batch`, using the *unbiased* variance estimate for the running
      update -- matching PyTorch's `BatchNorm1d` convention exactly, including
      its quirk of using the biased estimator for normalization but the
      unbiased one for the running average).
    - **Eval** (`eval_mode()`, the default): normalizes using the stored
      `running_mean`/`running_var` instead of the current batch's statistics.

    `running_mean`/`running_var` are permanent parameters (never fit by
    gradient descent, per `Parameter.make_permanent`) but are still updated
    in place during training -- see `evaluate`/`as_tensorflow_layer`.

    Parameters
    ----------
    shape : 1-tuple of int.
        Number of channels.
    momentum : float; default=0.1.
        Weight given to the current batch's statistics when updating the
        running mean/var. Matches PyTorch's `BatchNorm1d` convention (not
        TF/Keras's `BatchNormalization`, which uses the opposite convention
        for its own `momentum` argument).
    eps : float; default=1e-5.

    See also
    --------
    nems.layers.base.Layer

    Examples
    --------
    >>> bn = BatchNorm1d(shape=(80,))
    >>> x = np.random.rand(1000, 80)  # (time, channels)
    >>> out = bn.evaluate(x)          # eval_mode by default: running stats
    >>> out.shape
    (1000, 80)

    """

    def __init__(self, momentum=0.1, eps=1e-5, **kwargs):
        require_shape(self, kwargs, minimum_ndim=1, maximum_ndim=1)
        self.momentum = momentum
        self.eps = eps
        self._training = False
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Get initial values for `BatchNorm1d.parameters`.

        Layer parameters
        ----------------
        gamma : ndarray
            Per-channel scale. Prior: Normal(mean=1, sd=0.1).
        beta : ndarray
            Per-channel shift. Prior: Normal(mean=0, sd=0.1).
        running_mean : ndarray
            Permanent (not gradient-fit); updated via EMA during training.
        running_var : ndarray
            Permanent (not gradient-fit); updated via EMA during training.

        Returns
        -------
        nems.layers.base.Phi

        """
        n = self.shape[0]
        zero, one = np.zeros(n), np.ones(n)

        gamma = Parameter('gamma', shape=(n,), prior=Normal(one, one/10))
        beta = Parameter('beta', shape=(n,), prior=Normal(zero, one/10))

        running_mean = Parameter('running_mean', shape=(n,), prior=Normal(zero, one))
        running_mean.make_permanent()
        running_var = Parameter(
            'running_var', shape=(n,), prior=Normal(one, one), bounds=(1e-6, np.inf)
            )
        running_var.make_permanent()

        return Phi(gamma, beta, running_mean, running_var)

    def train_mode(self):
        """Use (and update) batch statistics during `evaluate`."""
        self._training = True

    def eval_mode(self):
        """Use stored running statistics during `evaluate` (the default)."""
        self._training = False

    def evaluate(self, input):
        """Normalize `input` across every axis but the last.

        Parameters
        ----------
        input : np.ndarray
            Shape (..., N); every axis but the last is reduced over.

        Returns
        -------
        np.ndarray

        """
        gamma, beta, running_mean, running_var = self.get_parameter_values()
        reduce_axes = tuple(range(input.ndim - 1))

        if self._training:
            n = int(np.prod([input.shape[a] for a in reduce_axes]))
            batch_mean = input.mean(axis=reduce_axes)
            batch_var = input.var(axis=reduce_axes)  # biased; used for normalization
            unbiased_var = batch_var * n / max(n - 1, 1)

            self.set_permanent_values(
                running_mean=(1 - self.momentum) * running_mean + self.momentum * batch_mean,
                running_var=(1 - self.momentum) * running_var + self.momentum * unbiased_var,
                )
            mean, var = batch_mean, batch_var
        else:
            mean, var = running_mean, running_var

        return gamma * (input - mean) / np.sqrt(var + self.eps) + beta

    @layer('bn')
    def from_keyword(keyword):
        """Construct BatchNorm1d from keyword.

        Keyword options
        ---------------
        {digit} : number of channels; required.

        Returns
        -------
        BatchNorm1d

        See also
        --------
        Layer.from_keyword

        """
        options = keyword.split('.')
        shape = pop_shape(options)
        return BatchNorm1d(shape=shape)

    def as_tensorflow_layer(self, **kwargs):
        """Return a Keras layer with the same train/eval statistics split.

        Relies on Keras's standard automatic `training` dispatch (the
        TensorFlow backend calls the built model with an explicit
        `training=True`/`training=False`, which Keras propagates to any
        sublayer whose `call` declares a `training` parameter).
        """
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        momentum = self.momentum  # Python float captured in closure
        eps = self.eps

        class BatchNorm1dTF(NemsKerasLayer):
            def call(self, inputs, training=None):
                reduce_axes = list(range(len(inputs.shape) - 1))

                if training:
                    batch_mean, batch_var = tf.nn.moments(inputs, axes=reduce_axes)
                    n = tf.cast(
                        tf.reduce_prod([tf.shape(inputs)[a] for a in reduce_axes]),
                        inputs.dtype
                        )
                    unbiased_var = batch_var * n / tf.maximum(n - 1, 1.0)

                    self.running_mean.assign(
                        (1 - momentum) * self.running_mean + momentum * batch_mean
                        )
                    self.running_var.assign(
                        (1 - momentum) * self.running_var + momentum * unbiased_var
                        )
                    mean, var = batch_mean, batch_var
                else:
                    mean, var = self.running_mean, self.running_var

                return self.gamma * (inputs - mean) / tf.sqrt(var + eps) + self.beta

        return BatchNorm1dTF(self, **kwargs)
# [AGENT EDIT END]
