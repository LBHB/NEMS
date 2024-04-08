import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter1d

from .base import Model
from nems.registry import layer
from nems.layers.tools import require_shape, pop_shape

from nems.layers import (
    WeightChannels, WeightChannelsGaussian, FiniteImpulseResponse,
    RectifiedLinear, DoubleExponential, LevelShift
    )
from nems.visualization.model import plot_nl

log = logging.getLogger(__name__)

##
## Class defs
##

class LN_reconstruction(Model):

    def __init__(self, time_bins, channels, out_channels, rank=None,
                 nonlinearity='RectifiedLinear',
                 nl_kwargs=None, from_saved=False, **model_init_kwargs):
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
        model_init_kwargs : dict; optional.
            Additional keyword arguments for `Model.__init__`, like `dtype`
            or `meta`.

        """

        super().__init__(**model_init_kwargs)
        if from_saved:
            return

        # Add STRF
        if rank is None:
            # Full-rank finite impulse response
            fir = FiniteImpulseResponse(include_anticausal=True, shape=(time_bins, channels, out_channels))
            self.add_layers(fir)
        else:
            wc = WeightChannels(shape=(channels, rank))
            fir = FiniteImpulseResponse(include_anticausal=True, shape=(time_bins, rank, out_channels))
            self.add_layers(wc, fir)

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

    def fit_LBHB(self, X, Y, cost_function='nmse', fitter='tf'):
        fitter_options = {'cost_function': cost_function,  # 'nmse'
                          'early_stopping_tolerance': 5e-3,
                          'validation_split': 0,
                          'learning_rate': 1e-2, 'epochs': 3000
                          }
        fitter_options2 = {'cost_function': cost_function,
                           'early_stopping_tolerance': 5e-4,
                           'validation_split': 0,
                           'learning_rate': 1e-3, 'epochs': 8000
                           }

        model = self.sample_from_priors()

        model.layers[-1].skip_nonlinearity()
        model = model.fit(input=X, target=Y, backend=fitter,
                          fitter_options=fitter_options, batch_size=None)
        model.layers[-1].unskip_nonlinearity()
        log.info('Fit stage 2: with static output nonlinearity')
        model = model.fit(input=X, target=Y, backend=fitter,
                          verbose=0, fitter_options=fitter_options2, batch_size=None)

        return model

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

    def __init__(self, time_bins=11, channels=None, out_channels=None, L1=10, L2=0,
                 nonlinearity='RectifiedLinear',
                 nl_kwargs=None, from_saved=False, **model_init_kwargs):
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
        model_init_kwargs : dict; optional.
            Additional keyword arguments for `Model.__init__`, like `dtype`
            or `meta`.

        """

        super().__init__(**model_init_kwargs)
        if from_saved:
            return

        wc1 = WeightChannels(shape=(channels, 1, L1))
        fir1 = FiniteImpulseResponse(include_anticausal=True, shape=(time_bins, 1, L1))
        relu1 = RectifiedLinear(shape=(L1,), no_offset=False, no_shift=False)
        if L2 > 0:

            wc2 = WeightChannels(shape=(L1, L2))
            relu2 = RectifiedLinear(shape=(L2,), no_offset=False, no_shift=False)
            wc3 = WeightChannels(shape=(L2, out_channels))

            self.add_layers(wc1, fir1, relu1, wc2, relu2, wc3)
        else:
            wc2 = WeightChannels(shape=(L1, out_channels))

            self.add_layers(wc1, fir1, relu1, wc2)

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

    def fit_LBHB(self, X, Y, cost_function='nmse', fitter='tf'):

        fitter_options = {'cost_function': cost_function,  # 'nmse'
                          'early_stopping_tolerance': 5e-3,
                          'validation_split': 0,
                          'learning_rate': 1e-2, 'epochs': 3000
                          }
        fitter_options2 = {'cost_function': cost_function,
                           'early_stopping_tolerance': 5e-4,
                           'validation_split': 0,
                           'learning_rate': 1e-3, 'epochs': 8000
                           }

        model = self.sample_from_priors()
        #model = model.sample_from_priors()

        model.layers[-1].skip_nonlinearity()
        model = model.fit(input=X, target=Y, backend=fitter,
                          fitter_options=fitter_options, batch_size=None)
        model.layers[-1].unskip_nonlinearity()
        log.info('Fit stage 2: with static output nonlinearity')
        model = model.fit(input=X, target=Y, backend=fitter,
                          verbose=0, fitter_options=fitter_options2, batch_size=None)

        return model

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

##
## helper functions
##
