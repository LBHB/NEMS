import numpy as np

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter
from .tools import require_shape, pop_shape


# [AGENT EDIT START | agent: claude | user: sbp894 | reason: port ACNet's residual-block skip connection (zero-init shortcut projection + learnable scalar gamma) as a NEMS Layer, for the ACNet-in-NEMS model port | date: 2026-09-16]
class ResAdd(Layer):
    """Combine a block's main path with a projected skip connection.

    Computes `gamma * res_scale * main + skip @ shortcut + shortcut_bias`,
    where `shortcut`/`shortcut_bias` are a learnable (Cin, Cout) projection
    and (Cout,) bias **initialized to zero**, and `gamma` is a learnable
    scalar initialized to 1 -- so at construction the skip contributes
    nothing and the block is its main path at full strength. Matches ACNet's
    `ResBlock_CNN1d_W_NL_v1` residual sum exactly (`PT_EncMdl_helpers_v2.py`):
    `shortcut` there is an `nn.Linear` (weight *and* bias) applied to the
    block's original (pre-FIR) input.

    This layer does not apply the block's final ReLU -- chain a plain
    `nems.layers.RectifiedLinear`/`relu` keyword after it, since NEMS already
    has that primitive and there is no reason to duplicate it here.

    Parameters
    ----------
    shape : 2-tuple of int (Cin, Cout).
        Cin must match the skip (block-input) signal's channel count, Cout
        the main-path signal's channel count.
    res_scale : float; default=1.0.
        Fixed (non-fittable) scale applied to the main path before adding the
        projected skip. Matches ACNet's per-config `res_scale` hyperparameter
        (not learned -- only `gamma` is).
    input : 2-list of str.
        `[main_name, skip_name]`: the block's post-BatchNorm main-path signal,
        and the block's original (pre-block) input signal, respectively.

    See also
    --------
    nems.layers.base.Layer

    Examples
    --------
    >>> ra = ResAdd(shape=(80, 100), res_scale=0.1, input=['b1_main', 'b1_in'])
    >>> main = np.random.rand(1000, 100)
    >>> skip = np.random.rand(1000, 80)
    >>> out = ra.evaluate(main, skip)
    >>> out.shape
    (1000, 100)

    """

    def __init__(self, res_scale=1.0, **kwargs):
        require_shape(self, kwargs, minimum_ndim=2, maximum_ndim=2)
        self.res_scale = res_scale
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Get initial values for `ResAdd.parameters`.

        Layer parameters
        ----------------
        shortcut : ndarray
            Shape (Cin, Cout). Initial value: zero (the skip contributes
            nothing until the projection is fit).
        shortcut_bias : ndarray
            Shape (Cout,). Initial value: zero.
        gamma : ndarray
            Shape (1,). Initial value: one.

        Returns
        -------
        nems.layers.base.Phi

        """
        cout = self.shape[1]
        shortcut = Parameter(
            'shortcut', shape=self.shape,
            prior=Normal(np.zeros(self.shape), np.full(self.shape, 0.1)),
            initial_value=0.0,
            )
        shortcut_bias = Parameter(
            'shortcut_bias', shape=(cout,),
            prior=Normal(np.zeros(cout), np.full(cout, 0.1)),
            initial_value=0.0,
            )
        gamma = Parameter(
            'gamma', shape=(1,), prior=Normal(np.ones(1), np.full(1, 0.1)),
            initial_value=1.0,
            )
        return Phi(shortcut, shortcut_bias, gamma)

    def evaluate(self, main, skip):
        """Combine `main` and a shortcut-projected `skip`.

        Parameters
        ----------
        main : np.ndarray
            Shape (T, Cout) -- the block's post-BatchNorm main path.
        skip : np.ndarray
            Shape (T, Cin) -- the block's original input.

        Returns
        -------
        np.ndarray
            Shape (T, Cout).

        """
        shortcut = self.parameters['shortcut'].values
        shortcut_bias = self.parameters['shortcut_bias'].values
        gamma = self.parameters['gamma'].values[0]
        shortcut_out = np.tensordot(skip, shortcut, axes=(1, 0)) + shortcut_bias
        return gamma * self.res_scale * main + shortcut_out

    @layer('resadd')
    def from_keyword(keyword):
        """Construct ResAdd from keyword.

        Keyword options
        ---------------
        {digit}x{digit} : (Cin, Cout) shape; required.
        {value} : `res_scale`, written with 'd' in place of '.'
            (e.g. '0d1' -> 0.1); required.
        {name} : main-path signal name; required.
        {name} : skip-path (block-input) signal name; required.

        Example: 'resadd.80x100.0d1.b1_main.b1_in'

        See also
        --------
        Layer.from_keyword

        """
        options = keyword.split('.')
        shape = pop_shape(options)
        res_scale = float(options[1].replace('d', '.'))
        main_name, skip_name = options[2], options[3]
        return ResAdd(shape=shape, res_scale=res_scale, input=[main_name, skip_name])

    def as_tensorflow_layer(self, **kwargs):
        """Return a Keras layer applying the same shortcut + scale combination."""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        res_scale = self.res_scale  # Python float captured in closure

        class ResAddTF(NemsKerasLayer):
            def call(self, inputs):
                main, skip = inputs
                shortcut_out = tf.tensordot(
                    skip, self.shortcut, axes=[[2], [0]]
                    ) + self.shortcut_bias
                return self.gamma[0] * res_scale * main + shortcut_out

        return ResAddTF(self, **kwargs)
# [AGENT EDIT END]
