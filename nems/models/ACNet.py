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
    >>> embeddings = model.get_embeddings(gtg)  # trunk output, before the readout
    >>> embeddings.shape
    (1000, 10)

    See also
    --------
    load_acnet_v1_weights
    nems.preprocessing.spectrogram.acnet_gtgram

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
            RectifiedLinear(shape=(cout,), output=('b1_in' if n_blocks > 1 else 'embeddings')),
            )

        # Blocks 1..N-1: depthwise FIR -> channel mix -> BatchNorm -> residual
        # add (zero-init shortcut) -> ReLU. The previous block's ReLU output
        # is named f'b{i}_in' (set above/below) so ResAdd can reference it
        # even though it isn't the layer immediately before ResAdd in the
        # sequence.
        for i in range(1, n_blocks):
            cin, cout = hidden_dim[i - 1], hidden_dim[i]
            main_name = f'b{i}_main'
            next_in_name = f'b{i + 1}_in' if i + 1 < n_blocks else 'embeddings'

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

    def get_embeddings(self, input, **eval_kwargs):
        """Return the shared-trunk ("manifold") embeddings for `input`.

        Equivalent to `ACNet_v1.acnet_model.ACNet.get_mf_embeddings`'s
        `shared_rep` -- the trunk's output just before the readout, i.e. the
        representation shared across every recorded neuron.

        Parameters
        ----------
        input : np.ndarray
            Shape (T, `num_cfs`) -- sqrt-domain gammatone magnitude (see
            `nems.preprocessing.spectrogram.acnet_gtgram`). Compression is
            applied internally by this model's own first layer; do not
            pre-compress `input` yourself.
        eval_kwargs : dict; optional.
            Passed through to `Model.evaluate`.

        Returns
        -------
        np.ndarray
            Shape (T, `hidden_dim[-1]`).

        """
        data = self.evaluate(input, return_full_data=True, **eval_kwargs)
        return data['embeddings']
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: sbp894 | reason: loader for the real released ACNet_v1 checkpoint's weights (exported to a portable npz outside NEMS, since NEMS must stay torch-free) -- needed for the wav/gtg embeddings usage tutorial to demonstrate the actual shipped manifold, not a randomly-initialized one | date: 2026-09-16]
def load_acnet_v1_weights(model, npz_path):
    """Load real ACNet_v1 checkpoint weights (exported to an npz) into `model`.

    The npz must be produced by
    `ACNet_v1/Claude/claude_debug/export_acnet_v1_weights.py` (run separately,
    under a torch env -- this function only ever touches plain numpy arrays).
    `model` must have been constructed with matching `hidden_dim`/`kernel_size`/
    `num_cfs`/`n_neurons` (see the npz's own `hidden_dim`/`kernel_size`/
    `num_cfs`/`n_neurons` entries, which `ACNet`'s defaults already match for
    the released checkpoint).

    Parameters
    ----------
    model : ACNet
    npz_path : str

    Returns
    -------
    ACNet
        `model`, with parameters updated in place.

    See also
    --------
    ACNet
    ACNet.get_embeddings

    """
    fx = np.load(npz_path)
    n_blocks = len(fx['hidden_dim'])

    idx = 1  # index 0 is the PowerCompress layer
    for i in range(n_blocks):
        dfir, wc, lvl, bn = model.layers[idx:idx + 4]
        idx += 4

        dfir.parameters['coefficients'].update(fx[f'block{i}_dfir_coefficients'])
        dfir.parameters['bias'].update(fx[f'block{i}_dfir_bias'])
        wc.parameters['coefficients'].update(fx[f'block{i}_wc_coefficients'])
        lvl.parameters['shift'].update(fx[f'block{i}_lvl_shift'])
        bn.parameters['gamma'].update(fx[f'block{i}_bn_gamma'])
        bn.parameters['beta'].update(fx[f'block{i}_bn_beta'])
        bn.parameters['running_mean'].update(fx[f'block{i}_bn_running_mean'])
        bn.parameters['running_var'].update(fx[f'block{i}_bn_running_var'])

        if i > 0:
            resadd = model.layers[idx]; idx += 1
            resadd.parameters['shortcut'].update(fx[f'block{i}_shortcut_weight'])
            resadd.parameters['shortcut_bias'].update(fx[f'block{i}_shortcut_bias'])
            resadd.parameters['gamma'].update(fx[f'block{i}_gamma'])

        idx += 1  # the block's ReLU has no parameters

    readout_wc, readout_lvl, readout_dexp = model.layers[idx:idx + 3]
    readout_wc.parameters['coefficients'].update(fx['readout_wc'])
    readout_lvl.parameters['shift'].update(fx['readout_lvl'])
    readout_dexp.parameters['base'].update(fx['readout_dexp_base'])
    readout_dexp.parameters['amplitude'].update(fx['readout_dexp_amp'])
    readout_dexp.parameters['kappa'].update(fx['readout_dexp_kappa'])
    # ACNet's DEXP has no shift term (base + amp*exp(-exp(-exp(kappa)*x))) --
    # NEMS's DoubleExponential does (adds shift to x before the innermost
    # exp); zero it out to match exactly.
    readout_dexp.parameters['shift'].update(np.zeros_like(fx['readout_dexp_base']))

    return model
# [AGENT EDIT END]
