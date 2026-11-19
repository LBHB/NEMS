import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter1d

from .base import Model
from nems.registry import layer
from nems.layers.tools import require_shape, pop_shape

from nems.layers import (
    WeightChannels, WeightChannelsGaussian, FiniteImpulseResponse,
    RectifiedLinear, DoubleExponential, LevelShift, Sigmoid
    )

log = logging.getLogger(__name__)

##
## Class defs
##

class LN_reconstruction(Model):

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: svd | reason: add causal option (FIR uses only current and past resp); None defaults so from_saved/json load works (2026-11-19) | date: 2026-11-18]
    def __init__(self, time_bins=None, channels=None, out_channels=None, rank=None,
                 nonlinearity='RectifiedLinear', causal=False,
                 nl_kwargs=None, from_saved=False, **model_init_kwargs):
    # [AGENT EDIT END]
        """Linear-nonlinear Spectro-Temporal Receptive Field model.

        Contains the following layers:
        1. WeightChannels or None (if full rank).
        2. FiniteImpulseResponse
        3. DoubleExponential, RectifiedLinear, or another static nonlinearity.

        Expects a single sound spectrogram as input, with shape (T, N), and a
        single recorded neural response as a target, with shape (T, 1), where
        T and N are the number of time bins and spectral channels, respectively.

        Based on model architectures described in:
        Mesgarani et al 2009

        Parameters
        ----------
        time_bins : int.
            Number of "taps" in FIR filter. We have found that a 150-250ms filter
            typically sufficient for modeling A1 responses, or 15-25 bins for
            a 100 Hz sampling rate.
        channels : int.
            Number of spectral channels in spectrogram.
        rank : int; optional.
            Number of spectral weightings used as input to a reduced-rank filter.
            For example, `rank=1` indicates a frequency-time separable STRF.
            If unspecified, a full-rank filter will be used.
        gaussian : bool; default=True.
            If True, use gaussian functions (1 per `rank`) to parameterize
            spectral weightings. Unused if `rank is None`.
        nonlinearity : str; default='DoubleExponential'.
            Specifies which static nonlinearity to apply after the STRF.
            Default is the double exponential nonlinearity used in the paper
            cited above.
        nl_kwargs : dict; optional.
            Additional keyword arguments for the nonlinearity Layer, like
            `no_shift` or `no_offset` for `RectifiedLinear`.
        causal : bool; default=False.
            If True, the FIR filter is causal: the reconstruction at time t
            uses only neural activity at times <= t (`time_bins` lags, 0 to
            time_bins-1). If False (default), the filter spans roughly
            +-time_bins/2 around t, so it also uses activity after t (the
            response lags the stimulus, so this usually reconstructs better).
        model_init_kwargs : dict; optional.
            Additional keyword arguments for `Model.__init__`, like `dtype`
            or `meta`.

        """

        super().__init__(**model_init_kwargs)
        if from_saved:
            return

        # [AGENT EDIT START | agent: claude-opus-5-5 | user: svd | reason: add causal option (FIR uses only current and past resp) | date: 2026-11-18]
        include_anticausal = not causal
        # Add STRF
        if rank is None:
            # Full-rank finite impulse response
            fir = FiniteImpulseResponse(include_anticausal=include_anticausal,
                                        shape=(time_bins, channels, out_channels))
            self.add_layers(fir)
        else:
            wc = WeightChannels(shape=(channels, rank))
            fir = FiniteImpulseResponse(include_anticausal=include_anticausal,
                                        shape=(time_bins, rank, out_channels))
            self.add_layers(wc, fir)
        # [AGENT EDIT END]

        # Add static nonlinearity
        if nonlinearity in ['DoubleExponential', 'dexp', 'DEXP']:
            nl_class = DoubleExponential
        elif nonlinearity in ['RectifiedLinear', 'relu', 'ReLU']:
            nl_class = RectifiedLinear
        else:
            raise ValueError(
                f'Unrecognized nonlinearity for LN model:  {nonlinearity}.')

        if nl_kwargs is None: nl_kwargs = {}
        nonlinearity = nl_class(shape=(out_channels,), **nl_kwargs)
        self.add_layers(nonlinearity)

    @classmethod
    def from_data(cls, input, filter_duration, sampling_rate=1000, **kwargs):
        channels = input.shape[-1]
        time_bins = int(filter_duration / 1000 * sampling_rate)
        # TODO: modify initial parameters based on stimulus statistics?
        return LN_reconstruction(time_bins, channels, **kwargs)

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: svd | reason: fit_LBHB moved to base Model | date: 2026-11-19]
    # fit_LBHB inherited from nems.models.base.Model, with decoder defaults:
    # stage-2 early-stopping tolerance 5e-4 (stage 1: 5e-3), 3000 stage-1 epochs
    fit_LBHB_defaults = {'early_stopping_tolerance': 5e-4, 'stage1_epochs': 3000}
    # [AGENT EDIT END]

    # TODO
    # @module('CNNrecon')
    def from_keyword(keyword):
        # Return a list of module instances matching this pre-built Model?
        # That way these models can be used with kw system as well, e.g.
        # model = Model.from_keywords('LNSTRF')
        #
        # But would need the .from_keywords method to check for list vs single
        # module returned.
        pass


class CNN_reconstruction(Model):

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: svd | reason: add causal option (FIR uses only current and past resp), as in LN_reconstruction | date: 2026-11-18]
    def __init__(self, time_bins=11, channels=None, out_channels=None, L1=10, L2=0,
                 nonlinearity='RectifiedLinear', causal=False, L1_reps=1,
                 nl_kwargs=None, from_saved=False, **model_init_kwargs):
    # [AGENT EDIT END]
        """CNN-based spectrogram population decoder.

        Contains the following layers:
        1. WeightChannels or None (if full rank).
        2. FiniteImpulseResponse
        3. DoubleExponential, RectifiedLinear, or another static nonlinearity.

        Expects a single sound spectrogram as input, with shape (T, N), and a
        single recorded neural response as a target, with shape (T, 1), where
        T and N are the number of time bins and spectral channels, respectively.

        Based on model architectures described in:
        Mesgarani et al 2009

        Parameters
        ----------
        time_bins : int.
            Number of "taps" in FIR filter. We have found that a 150-250ms filter
            typically sufficient for modeling A1 responses, or 15-25 bins for
            a 100 Hz sampling rate.
        channels : int.
            Number of spectral channels in spectrogram.
        rank : int; optional.
            Number of spectral weightings used as input to a reduced-rank filter.
            For example, `rank=1` indicates a frequency-time separable STRF.
            If unspecified, a full-rank filter will be used.
        gaussian : bool; default=True.
            If True, use gaussian functions (1 per `rank`) to parameterize
            spectral weightings. Unused if `rank is None`.
        nonlinearity : str; default='DoubleExponential'.
            Specifies which static nonlinearity to apply after the STRF.
            Default is the double exponential nonlinearity used in the paper
            cited above.
        nl_kwargs : dict; optional.
            Additional keyword arguments for the nonlinearity Layer, like
            `no_shift` or `no_offset` for `RectifiedLinear`.
        causal : bool; default=False.
            If True, the FIR filter is causal: the reconstruction at time t
            uses only neural activity at times <= t (`time_bins` lags, 0 to
            time_bins-1). If False (default), the filter spans roughly
            +-time_bins/2 around t. Unused if time_bins == 1.
        L1_reps : int; default=1.
            Number of WeightChannels + FIR + ReLU blocks in layer 1 (as in
            CNN_pop). Block 1 maps `channels` -> L1; each later block maps
            L1 -> L1 with its own `time_bins`-tap FIR, so L1_reps=2 gives two
            stacked FIR layers (total temporal span 2*(time_bins-1)+1 bins).
            Requires time_bins > 1 when L1_reps > 1.
        model_init_kwargs : dict; optional.
            Additional keyword arguments for `Model.__init__`, like `dtype`
            or `meta`.

        """

        super().__init__(**model_init_kwargs)
        if from_saved:
            return
        if time_bins>1:
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: svd | reason: causal option; L1_reps stacked wc+fir+relu blocks (copied from CNN_pop) | date: 2026-11-18]
            for rep_idx in range(L1_reps):
                N_in = channels if rep_idx == 0 else L1
                wc1 = WeightChannels(shape=(N_in, 1, L1))
                fir1 = FiniteImpulseResponse(include_anticausal=not causal,
                                             shape=(time_bins, 1, L1))
                relu1 = RectifiedLinear(shape=(L1,), no_offset=False, no_shift=False)
                self.add_layers(wc1, fir1, relu1)
            # [AGENT EDIT END]
        else:
            if L1_reps > 1:
                raise ValueError("L1_reps > 1 requires time_bins > 1")
            relu1 = RectifiedLinear(shape=(L1,), no_offset=False, no_shift=False)
            wc1 = WeightChannels(shape=(channels, L1))
            self.add_layers(wc1, relu1)

        if L2 > 0:

            wc2 = WeightChannels(shape=(L1, L2))
            relu2 = RectifiedLinear(shape=(L2,), no_offset=False, no_shift=False)
            wc3 = WeightChannels(shape=(L2, out_channels))

            self.add_layers(wc2, relu2, wc3)
        else:
            wc2 = WeightChannels(shape=(L1, out_channels))

            self.add_layers(wc2)

        # Add static nonlinearity
        if nonlinearity in ['DoubleExponential', 'dexp', 'DEXP']:
            nl_class = DoubleExponential
        elif nonlinearity in ['RectifiedLinear', 'relu', 'ReLU']:
            nl_class = RectifiedLinear
        else:
            raise ValueError(
                f'Unrecognized nonlinearity for LN model:  {nonlinearity}.')

        if nl_kwargs is None: nl_kwargs = {}
        nonlinearity = nl_class(shape=(out_channels,), **nl_kwargs)
        self.add_layers(nonlinearity)

    @classmethod
    def from_data(cls, input, filter_duration, sampling_rate=1000, **kwargs):
        channels = input.shape[-1]
        time_bins = int(filter_duration / 1000 * sampling_rate)
        # TODO: modify initial parameters based on stimulus statistics?
        return LN_reconstruction(time_bins, channels, **kwargs)

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: svd | reason: fit_LBHB moved to base Model | date: 2026-11-19]
    # fit_LBHB inherited from nems.models.base.Model, with decoder defaults:
    # stage-2 early-stopping tolerance 5e-4 (stage 1: 5e-3), 3000 stage-1 epochs
    fit_LBHB_defaults = {'early_stopping_tolerance': 5e-4, 'stage1_epochs': 3000}
    # [AGENT EDIT END]

    # TODO
    # @module('CNNrecon')
    def from_keyword(keyword):
        # Return a list of module instances matching this pre-built Model?
        # That way these models can be used with kw system as well, e.g.
        # model = Model.from_keywords('LNSTRF')
        #
        # But would need the .from_keywords method to check for list vs single
        # module returned.
        pass

# [AGENT EDIT START | agent: claude-sonnet-4-6 | user: svd | reason: implement binary decoder model | date: 2026-05-14]
class binary(Model):
    """CNN-based binary decoder.

    Architecture mirrors CNN_reconstruction but outputs a sigmoid-activated
    probability, making it suitable for binary classification targets.

    Layers:
    1. WeightChannels  — mix input channels down to L1 features
    2. FiniteImpulseResponse (skipped when time_bins=1) — temporal filter
    3. RectifiedLinear — hidden nonlinearity
    4. WeightChannels  — project to L2 (optional) then to out_channels
    5. [WeightChannels + RectifiedLinear if L2 > 0]
    6. Sigmoid         — squash to (0, 1) for binary probability output

    Parameters
    ----------
    time_bins : int
        FIR filter length in time bins. Set to 1 to skip the FIR layer entirely.
    channels : int
        Number of input channels (e.g. neural population size).
    out_channels : int
        Number of binary outputs.
    L1 : int; default=10
        Number of hidden units in the first layer.
    L2 : int; default=0
        Number of hidden units in a second hidden layer. 0 = no second layer.
    from_saved : bool; default=False
        If True, skip layer construction (used when loading from disk).
    model_init_kwargs : dict
        Passed to `Model.__init__`.
    """

    def __init__(self, time_bins=1, channels=None, out_channels=None,
                 L1=10, L2=0, regularizer=None, from_saved=False,
                 **model_init_kwargs):
        super().__init__(**model_init_kwargs)
        if from_saved:
            return

        relu1 = RectifiedLinear(shape=(L1,), no_offset=False, no_shift=False)

        if time_bins > 1:
            wc1 = WeightChannels(shape=(channels, 1, L1), regularizer=regularizer)
            fir1 = FiniteImpulseResponse(include_anticausal=True, shape=(time_bins, 1, L1))
            core_layers = [wc1, fir1, relu1]
        else:
            wc1 = WeightChannels(shape=(channels, L1), regularizer=regularizer)
            core_layers = [wc1, relu1]

        if L2 > 0:
            wc2 = WeightChannels(shape=(L1, L2), regularizer=regularizer)
            relu2 = RectifiedLinear(shape=(L2,), no_offset=False, no_shift=False)
            wc3 = WeightChannels(shape=(L2, out_channels), regularizer=regularizer)
            output_layers = [wc2, relu2, wc3]
        else:
            wc2 = WeightChannels(shape=(L1, out_channels), regularizer=regularizer)
            output_layers = [wc2]

        sigmoid = Sigmoid(shape=(out_channels,))
        self.add_layers(*core_layers, *output_layers, sigmoid)

    @classmethod
    def from_data(cls, input, filter_duration=0, sampling_rate=1000, **kwargs):
        channels = input.shape[-1]
        time_bins = max(1, int(filter_duration / 1000 * sampling_rate))
        return cls(time_bins=time_bins, channels=channels, **kwargs)

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: svd | reason: fit_LBHB moved to base Model | date: 2026-11-19]
    # single-stage fit (no output-NL skipping: crossentropy needs sigmoid output)
    fit_LBHB_defaults = {'cost_function': 'binary_crossentropy', 'two_stage': False,
                         'early_stopping_tolerance': 5e-4}

    def fit_LBHB(self, X, Y, init_priors=True, **kwargs):
        """Model.fit_LBHB, one stage with binary crossentropy.
        init_priors : old name for init_from_priors (kept for existing callers).
        """
        kwargs.setdefault('init_from_priors', init_priors)
        return super().fit_LBHB(X, Y, **kwargs)
    # [AGENT EDIT END]
# [AGENT EDIT END]

##
## helper functions
##
