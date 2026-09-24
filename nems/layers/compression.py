import numpy as np

from nems.registry import layer
from .base import Layer, Phi


# [AGENT EDIT START | agent: claude | user: sbp894 | reason: port ACNet's fixed sqrt/log10x gammatone compression as a NEMS Layer, for the ACNet-in-NEMS model port; 2026-09-17 renamed mode='sqrt' (a no-op that was confusing precisely because its name implied an operation it doesn't perform) to mode=None -- compression is now exclusively this layer's concern, gtgram helpers no longer apply any compression themselves | date: 2026-09-16]
class PowerCompress(Layer):
    """Apply a fixed elementwise magnitude compression to gammatone input.

    ACNet trains on gammatone-filterbank magnitude that has already been
    passed through a NEMS-style `gtgram` (see
    `nems.preprocessing.spectrogram.gammatone.gammagram`), whose last step is
    `sqrt(segment_energy.mean())` -- i.e. `gtgram` output is already in the
    sqrt-magnitude domain, not linear magnitude. This is simply what `gtgram`
    produces, not a compression choice -- `PowerCompress` is the only place
    in this port that a compression choice is actually made:

    - `mode=None` (default) : identity. Use the standard `gtgram` output
      as-is. (This used to be spelled `mode='sqrt'`, which was confusing --
      it implied an operation was being applied here, when the sqrt is
      really just an intrinsic property of `gtgram`'s own output, computed
      upstream of this layer, not by it.)
    - `mode='log10x'` : `0.5*log(1 + 10*mag**2)`, recovering linear magnitude
      by squaring the sqrt-domain input first. Matches
      `PT_EncMdl_helpers_v2.MultiTask_BNTDataSet_Site_Nems`'s `log10x` branch
      exactly (`c_gain=0.5`, `c_factor=10`).

    This Layer has no fittable parameters -- the compression mode is a fixed
    choice, not something to optimize (contrast with `nems.layers.LogCompress`,
    which has a learnable `shift`).

    Parameters
    ----------
    mode : str or None; one of {None, 'log10x'}; default=None.

    See also
    --------
    nems.layers.base.Layer

    Examples
    --------
    >>> pc = PowerCompress(mode='log10x')
    >>> gtg = np.random.rand(1000, 32)  # (time, channels), standard nems gtgram
    >>> out = pc.evaluate(gtg)
    >>> out.shape
    (1000, 32)

    """

    def __init__(self, mode=None, **kwargs):
        if mode not in (None, 'log10x'):
            raise ValueError(
                f"PowerCompress mode must be None or 'log10x', got {mode!r}."
                )
        self.mode = mode
        super().__init__(**kwargs)

    def initial_parameters(self):
        """No fittable parameters -- compression mode is fixed at construction.

        Returns
        -------
        nems.layers.base.Phi

        """
        return Phi()

    def evaluate(self, input):
        """Apply the fixed compression to `input`.

        Parameters
        ----------
        input : np.ndarray
            The standard `gtgram` output (sqrt-domain gammatone magnitude).

        Returns
        -------
        np.ndarray

        """
        if self.mode is None:
            return input
        else:
            # log10x: recover linear magnitude (input**2), then compress.
            c_gain, c_factor = 0.5, 10
            return c_gain * np.log(1 + c_factor * input**2)

    @layer('pow')
    def from_keyword(keyword):
        """Construct PowerCompress from keyword.

        Keyword options
        ---------------
        none : mode=None (identity; use the standard gtgram output as-is).
        log10x : mode='log10x'.

        Returns
        -------
        PowerCompress

        See also
        --------
        Layer.from_keyword

        """
        options = keyword.split('.')
        if 'log10x' in options:
            mode = 'log10x'
        elif 'none' in options:
            mode = None
        else:
            raise ValueError(
                f"PowerCompress keyword must specify 'none' or 'log10x', got {keyword!r}."
                )
        return PowerCompress(mode=mode)

    def as_tensorflow_layer(self, **kwargs):
        """Return a Keras layer that applies the same fixed compression."""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        mode = self.mode  # Python str/None captured in closure

        class PowerCompressTF(NemsKerasLayer):
            def call(self, inputs):
                if mode is None:
                    return inputs
                else:
                    c_gain, c_factor = 0.5, 10
                    return c_gain * tf.math.log(
                        1 + c_factor * tf.math.square(inputs)
                        )

        return PowerCompressTF(self, **kwargs)
# [AGENT EDIT END]
