import numpy as np

from .base import Model
from nems.layers import (
    WeightChannels, LevelShift, DoubleExponential, RectifiedLinear,
    PowerCompress, BatchNorm1d, DepthwiseFIR, ResAdd,
    )


# [AGENT EDIT START | agent: claude | user: sbp894 | reason: assemble ACNet's shared trunk + concatenated readout as a native NEMS Model, built from the new PowerCompress/BatchNorm1d/DepthwiseFIR/ResAdd layers plus reused wc/lvl/relu/dexp -- the ACNet-in-NEMS model port | date: 2026-09-16]
class ACNet(Model):
    """ACNet: a multi-task 1D-ResNet encoding model of ferret auditory cortex.

    A shared stack of causal residual conv blocks (depthwise FIR -> channel
    mix -> BatchNorm -> zero-init-shortcut residual add -> ReLU) followed by
    a single linear + DoubleExponential readout over every recorded neuron.
    Ported from `ACNet_v1.acnet_model.ACNet` (a standalone, torch, single-
    concatenated-readout repackaging of `MT_ResNet_v2`,
    `PT_EncMdl_helpers_v2.py`) -- see that module's `DEFAULT_CONFIG` for the
    released model's exact hyperparameters, which are also this class's
    defaults.

    Expects a single (T, `num_cfs`) gammatone-magnitude spectrogram as input
    (see `nems.preprocessing.spectrogram.acnet_gtgram` for the matching
    front end) and predicts a (T, `n_neurons`) PSTH.

    Block 0 has no residual connection (`ResBlock_CNN1d_W_NL_v1` is
    constructed with `skip=False` there in the original code) -- every other
    block does.

    Parameters
    ----------
    num_cfs : int; default=32.
        Number of gammatone frequency channels (the input's channel count).
    hidden_dim : sequence of int; default=(75, 100, 125, 150, 175, 200).
        Output channel count of each residual block.
    kernel_size : int or sequence of int; default=7.
        Causal FIR kernel size (taps) per block. A scalar is broadcast to
        every block.
    n_neurons : int; default=3124.
        Number of output channels (recorded neurons) in the readout.
    compress : str; one of {'sqrt', 'log10x'}; default='log10x'.
        See `nems.layers.compression.PowerCompress`. `'log10x'` is the
        released checkpoint's actual training config.
    res_scale : float; default=1.0.
        Fixed (non-fittable) residual scale for every block but the first
        (which has no residual). Matches the released checkpoint's config
        (`ACNet_v1.acnet_model.DEFAULT_CONFIG['res_scale']`).
    from_saved : bool; default=False.
        If True, skip layer construction (for loading a saved Model where
        layers will be restored separately).

    See also
    --------
    nems.models.CNN.CNN_pop
    nems.layers.compression.PowerCompress
    nems.layers.batchnorm.BatchNorm1d
    nems.layers.depthwise_fir.DepthwiseFIR
    nems.layers.acnet_block.ResAdd

    Examples
    --------
    >>> model = ACNet(hidden_dim=(8, 10), kernel_size=3, n_neurons=5)
    >>> gtg = np.random.rand(1000, 32)  # (time, num_cfs), gammatone magnitude
    >>> psth = model.evaluate(gtg)
    >>> psth.shape
    (1000, 5)

    """

    def __init__(self, num_cfs=32, hidden_dim=(75, 100, 125, 150, 175, 200),
                 kernel_size=7, n_neurons=3124, compress='log10x', res_scale=1.0,
                 from_saved=False, **model_init_kwargs):
        super().__init__(**model_init_kwargs)
        if from_saved:
            return

        n_blocks = len(hidden_dim)
        if np.isscalar(kernel_size):
            kernel_size = [kernel_size] * n_blocks

        self.add_layers(PowerCompress(mode=compress))

        # Block 0: no residual connection.
        cin, cout = num_cfs, hidden_dim[0]
        self.add_layers(
            DepthwiseFIR(shape=(kernel_size[0], cin)),
            WeightChannels(shape=(cin, cout)),
            LevelShift(shape=(cout,)),
            BatchNorm1d(shape=(cout,)),
            RectifiedLinear(shape=(cout,), output=('b1_in' if n_blocks > 1 else None)),
            )

        # Blocks 1..N-1: depthwise FIR -> channel mix -> BatchNorm -> residual
        # add (zero-init shortcut) -> ReLU. The previous block's ReLU output
        # is named f'b{i}_in' (set above/below) so ResAdd can reference it
        # even though it isn't the layer immediately before ResAdd in the
        # sequence.
        for i in range(1, n_blocks):
            cin, cout = hidden_dim[i - 1], hidden_dim[i]
            main_name = f'b{i}_main'
            next_in_name = f'b{i + 1}_in' if i + 1 < n_blocks else None

            self.add_layers(
                DepthwiseFIR(shape=(kernel_size[i], cin)),
                WeightChannels(shape=(cin, cout)),
                LevelShift(shape=(cout,)),
                BatchNorm1d(shape=(cout,), output=main_name),
                ResAdd(shape=(cin, cout), res_scale=res_scale,
                      input=[main_name, f'b{i}_in']),
                RectifiedLinear(shape=(cout,), output=next_in_name),
                )

        # Single concatenated neural readout (all recorded neurons at once).
        # Named explicitly (and registered as `Model.output_name`) so
        # `predict()`/`evaluate(return_full_data=False)` return this array
        # directly rather than the dict of every named intermediate signal
        # (b*_in/b*_main) the residual wiring above requires.
        self.add_layers(
            WeightChannels(shape=(hidden_dim[-1], n_neurons)),
            LevelShift(shape=(n_neurons,)),
            DoubleExponential(shape=(n_neurons,), output='psth'),
            )
        self.output_name = 'psth'
# [AGENT EDIT END]
