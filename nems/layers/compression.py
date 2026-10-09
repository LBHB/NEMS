import numpy as np

from nems.registry import layer
from .base import Layer, Phi


# [AGENT EDIT START | agent: claude | user: sbp894 | reason: port ACNet's fixed sqrt/log10x gammatone compression as a NEMS Layer, for the ACNet-in-NEMS model port; 2026-09-17 renamed mode='sqrt' (a no-op that was confusing precisely because its name implied an operation it doesn't perform) to mode=None -- compression is now exclusively this layer's concern, gtgram helpers no longer apply any compression themselves; 2026-10-09 the None-rename turned out to be wrong for a SECOND, different 'sqrt': MultiTask_BNTDataSet_Site_Nems's compress='sqrt' being a no-op at LOAD time doesn't mean amplitude is untouched -- the archive it loads from was written by export_data.py --compress sqrt, i.e. is ALREADY amplitude**0.5 (see pytorch_models/CLAUDE.md's Data section), so a model trained with compress='sqrt' (v2, v2.2) genuinely consumes sqrt(amplitude), not amplitude. That's a real elementwise transform this layer must apply, distinct from gtgram's own energy->amplitude sqrt (which stays a no-op here, correctly) -- restored as mode='sqrt' | date: 2026-10-09]
class PowerCompress(Layer):
    """Apply a fixed elementwise magnitude compression to gammatone input.

    ACNet trains on gammatone-filterbank magnitude straight from a
    NEMS-style `gtgram` (see
    `nems.preprocessing.spectrogram.gammatone.gammagram`), in AMPLITUDE
    units: `gtgram`'s last step, `sqrt(segment_energy.mean())`, is a
    physical amplitude-from-energy conversion, not a compression choice, so
    its output is plain (uncompressed) amplitude. `PowerCompress` is the
    only place in this port that a compression choice is actually made:

    - `mode=None` (default) : identity. Use the standard `gtgram` output
      (amplitude) as-is -- this is the domain `gtgram`'s own internal
      `sqrt(energy)` step already produces; no further operation happens
      here. Not the training config for any released checkpoint so far
      (v1 uses `'log10x'`, v2/v2.2 use `'sqrt'` below) -- provided for
      completeness and for testing the trunk in isolation.
    - `mode='sqrt'` : `sqrt(|amplitude|)`. Matches `PT_EncMdl_helpers_v2.
      MultiTask_BNTDataSet_Site_Nems`'s `compress='sqrt'` branch -- despite
      being a literal `pass` (no-op) at dataset-*load* time, that's a no-op
      relative to the *archive*, and the archive itself was written via
      `export_data.py --compress sqrt`, i.e. is already `amplitude**0.5`
      (see `pytorch_models/CLAUDE.md`'s Data section: "the archive **is**
      the sqrt-magnitude domain"). So a model trained with `compress='sqrt'`
      (v2, v2.2 -- see `nems.models.ACNet.load_acnet`) genuinely consumes
      `sqrt(amplitude)`, and this mode applies that one real transform on
      top of `gtgram`'s plain-amplitude output. Do not confuse this with
      `gtgram`'s own internal sqrt (energy -> amplitude, step zero, always
      applied, never a choice) -- this is a second, independent sqrt, a
      genuine compression choice, applied on top of amplitude.
    - `mode='log10x'` : `0.5*log(1 + 10*amplitude)`, applied directly to
      `gtgram`'s amplitude output. Matches `MultiTask_BNTDataSet_Site_Nems`'s
      `log10x` branch (`c_gain=0.5`, `c_factor=10`) -- there, the archived
      `amplitude**0.5` is squared back to amplitude first (`self.x_tasks **
      2`) before applying this same formula, undoing the archive's own sqrt
      (unlike the `'sqrt'` branch above, which keeps it). NEMS's
      `acnet_gtgram` never introduces that storage-driven extra sqrt -- it
      returns amplitude directly (see its own docstring) -- so no squaring
      belongs here; doing so would double-compress the input.
      Confirmed 2026-09-28: squaring here silently broke `get_embeddings`
      on a fresh wav-computed `acnet_gtgram` input (r_test ratio ~0.40
      vs. published, across 8 sites/4 animals) despite `acnet_gtgram`'s
      own docstring being correct the whole time -- this layer's `**2`
      was the actual bug, not the front end.

    This Layer has no fittable parameters -- the compression mode is a fixed
    choice, not something to optimize (contrast with `nems.layers.LogCompress`,
    which has a learnable `shift`).

    Parameters
    ----------
    mode : str or None; one of {None, 'sqrt', 'log10x'}; default=None.

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
        if mode not in (None, 'sqrt', 'log10x'):
            raise ValueError(
                f"PowerCompress mode must be None, 'sqrt' or 'log10x', got {mode!r}."
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
            The standard `gtgram` output (amplitude-domain gammatone
            magnitude, uncompressed).

        Returns
        -------
        np.ndarray

        """
        if self.mode is None:
            return input
        elif self.mode == 'sqrt':
            return np.sqrt(np.abs(input))
        else:
            c_gain, c_factor = 0.5, 10
            return c_gain * np.log(1 + c_factor * input)

    @layer('pow')
    def from_keyword(keyword):
        """Construct PowerCompress from keyword.

        Keyword options
        ---------------
        none : mode=None (identity; use the standard gtgram output as-is).
        sqrt : mode='sqrt'.
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
        elif 'none' in options:
            mode = None
        else:
            raise ValueError(
                f"PowerCompress keyword must specify 'none', 'sqrt' or 'log10x', "
                f"got {keyword!r}."
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
                elif mode == 'sqrt':
                    return tf.sqrt(tf.abs(inputs))
                else:
                    c_gain, c_factor = 0.5, 10
                    return c_gain * tf.math.log(1 + c_factor * inputs)

        return PowerCompressTF(self, **kwargs)
# [AGENT EDIT END]
