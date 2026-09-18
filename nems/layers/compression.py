import numpy as np

from nems.registry import layer
from .base import Layer, Phi


# [AGENT EDIT START | agent: claude | user: sbp894 | reason: port ACNet's fixed sqrt/log10x gammatone compression as a NEMS Layer, for the ACNet-in-NEMS model port | date: 2026-09-16]
class PowerCompress(Layer):
    """Apply a fixed elementwise magnitude compression to gammatone input.

    ACNet trains on gammatone-filterbank magnitude that has already been
    passed through a NEMS-style `gtgram` (see
    `nems.preprocessing.spectrogram.gammatone.gammagram`), whose last step is
    `sqrt(segment_energy.mean())` -- i.e. `gtgram` output is already in the
    sqrt-magnitude domain, not linear magnitude. `PowerCompress` treats its
    input accordingly:

    - `mode='sqrt'` : identity (input is already `sqrt(mag)`).
    - `mode='log10x'` : `0.5*log(1 + 10*mag**2)`, recovering linear magnitude
      by squaring the sqrt-domain input first. Matches
      `PT_EncMdl_helpers_v2.MultiTask_BNTDataSet_Site_Nems`'s `log10x` branch
      exactly (`c_gain=0.5`, `c_factor=10`).

    This Layer has no fittable parameters -- the compression mode is a fixed
    choice, not something to optimize (contrast with `nems.layers.LogCompress`,
    which has a learnable `shift`).

    Parameters
    ----------
    mode : str; one of {'sqrt', 'log10x'}; default='sqrt'.

    See also
    --------
    nems.layers.base.Layer

    Examples
    --------
    >>> pc = PowerCompress(mode='log10x')
    >>> gtg = np.random.rand(1000, 32)  # (time, channels), sqrt-domain
    >>> out = pc.evaluate(gtg)
    >>> out.shape
    (1000, 32)

    """

    def __init__(self, mode='sqrt', **kwargs):
        if mode not in ('sqrt', 'log10x'):
            raise ValueError(
                f"PowerCompress mode must be 'sqrt' or 'log10x', got {mode!r}."
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
            Gammatone magnitude in the sqrt domain (NEMS `gtgram` convention).

        Returns
        -------
        np.ndarray

        """
        if self.mode == 'sqrt':
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
        sqrt : mode='sqrt' (identity; input already sqrt-domain magnitude).
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
        elif 'sqrt' in options:
            mode = 'sqrt'
        else:
            raise ValueError(
                f"PowerCompress keyword must specify 'sqrt' or 'log10x', got {keyword!r}."
                )
        return PowerCompress(mode=mode)

    def as_tensorflow_layer(self, **kwargs):
        """Return a Keras layer that applies the same fixed compression."""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        mode = self.mode  # Python str captured in closure

        class PowerCompressTF(NemsKerasLayer):
            def call(self, inputs):
                if mode == 'sqrt':
                    return inputs
                else:
                    c_gain, c_factor = 0.5, 10
                    return c_gain * tf.math.log(
                        1 + c_factor * tf.math.square(inputs)
                        )

        return PowerCompressTF(self, **kwargs)
# [AGENT EDIT END]
