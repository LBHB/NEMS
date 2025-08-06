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

# Currently a placeholder basd on LN pop

# TODO: CNN forward models -- 1D? single cell? Pop?
class CNN_pop(Model):

    def __init__(self, time_bins=None, channels_in=None, channels_out=None, rank=None, L1=None, L2=None, share_tuning=True,
                 gaussian=False, nonlinearity='DoubleExponential', stride=1, L1_reps=1,
                 nl_kwargs=None, regularizer=None, regularize_fir=True,
                 from_saved=False, **model_init_kwargs):
        """
        1D CNN Spectro-Temporal Receptive Field model.

        Contains the following layers:
        1. WeightChannels, WeightChannelsGaussian, or None (if full rank).
        2. FiniteImpulseResponse
        3. DoubleExponential, RectifiedLinear, or another static nonlinearity.

        Expects a single sound spectrogram as input, with shape (T, N), and a
        single recorded neural response as a target, with shape (T, 1), where
        T and N are the number of time bins and spectral channels, respectively.

        Based on model architectures described in:
        Pennington & David (2023)
        doi: 10.1371/journal.pcbi.1004628

        :param time_bins: int.
            Number of "taps" in L1 FIR filter. We have found that a 150-250ms filter
            typically sufficient for modeling A1 responses, or 15-25 bins for
            a 100 Hz sampling rate.
        :param channels_in: int.
            Number of spectral channels in spectrogram.
        :param channels_out: int.
            Number of output channels (neurons) to predict activity
        :param rank: int; optional.
            Number of spectral weightings used as input to a reduced-rank filter.
            For example, `rank=1` indicates a frequency-time separable STRF.
            If unspecified, a full-rank filter will be used.
        :param L1: 
        :param L2: 
        :param share_tuning: 
        :param gaussian: 
        :param nonlinearity: str; default='RectifiedLinear'.
            Specifies which static nonlinearity to apply after the STRF.
            Default is the double exponential nonlinearity used in the paper
            cited above.
        :param stride: 
        :param L1_reps: 
        :param nl_kwargs: dict; optional.
            Additional keyword arguments for the nonlinearity Layer, like
            `no_shift` or `no_offset` for `RectifiedLinear`.
        :param regularizer: 
        :param from_saved: 
        :param model_init_kwargs: dict; optional.
            Additional keyword arguments for `Model.__init__`, like `dtype`
            or `meta`.
        """

        super().__init__(**model_init_kwargs)
        if from_saved:
            channels_out = self.layers[-1].shape[0]
            self.out_range = [[-1] * channels_out, [3] * channels_out]
            return

        if L1 is None:
            L1 = rank

        # Determine form of initial spectral filters
        if gaussian:
            wc_class1 = WeightChannelsGaussian
            reg1 = None
        else:
            wc_class1 = WeightChannels
            reg1 = regularizer
        if regularize_fir:
            fir_reg1=regularizer
        else:
            fir_reg1=None

        # layer 1
        if stride > 1:
            stride_per_layer = int(stride/L1_reps)
        else:
            stride_per_layer = stride

        for ll1 in range(L1_reps):
            if ll1==0:
                N_in = channels_in
            else:
                N_in = L1
            relu1 = RectifiedLinear(shape=(L1,), no_offset=True, no_shift=False)
            if rank is None:
                # Full-rank finite impulse response, one per output channel
                fir = FiniteImpulseResponse(shape=(time_bins, N_in, L1),
                                            stride=stride, regularizer=regularizer)
                self.add_layers(fir, relu1)
            elif gaussian & (ll1==0):
                wc = wc_class1(shape=(N_in, 1, L1))
                fir = FiniteImpulseResponse(shape=(time_bins, 1, L1),
                                            stride=stride_per_layer, regularizer=fir_reg1)
                self.add_layers(wc, fir, relu1)
            else:
                wc = WeightChannels(shape=(N_in, 1, L1), regularizer=regularizer)
                fir = FiniteImpulseResponse(shape=(time_bins, 1, L1),
                                            stride=stride_per_layer, regularizer=fir_reg1)
                self.add_layers(wc, fir, relu1)

        # layer(s) 2 and/or 3
        if L2 is not None:
            wc2 = WeightChannels(shape=(L1, L2), regularizer=regularizer)
            relu2 = RectifiedLinear(shape=(L2,), no_offset=True, no_shift=False)
            wc3 = WeightChannels(shape=(L2, channels_out), regularizer=regularizer)
            self.add_layers(wc2, relu2, wc3)
        else:
            wc2 = WeightChannels(shape=(L1, channels_out), regularizer=regularizer)
            self.add_layers(wc2)

        # Add static nonlinearity
        if nonlinearity in ['DoubleExponential', 'dexp', 'DEXP']:
            nl_class = DoubleExponential
        elif nonlinearity in ['RectifiedLinear', 'relu', 'ReLU']:
            nl_class = RectifiedLinear
        elif nonlinearity in ['LevelShift', 'lvl']:
            nl_class = LevelShift
        elif nonlinearity in ['', None]:
            nl_class = None
        else:
            raise ValueError(f'Unrecognized nonlinearity for CNN model: {nonlinearity}.')

        if nl_class is not None:
            if nl_kwargs is None: nl_kwargs = {}
            nonlinearity = nl_class(shape=(channels_out,), **nl_kwargs)
            self.add_layers(nonlinearity)

        self.out_range = [[-1] * channels_out, [3] * channels_out]

    @classmethod
    def from_data(cls, input, output, filter_duration, sampling_rate=1000, **kwargs):
        channels_in = input.shape[-1]
        channels_out = output.shape[-1]
        time_bins = int(filter_duration / 1000 * sampling_rate)
        # TODO: modify initial parameters based on stimulus statistics?
        return LN_pop(time_bins, channels_in, channels_out, **kwargs)

    def fit_LBHB(self, X, Y, cost_function='nmse', fitter='tf',
                 learning_rate = 1e-3, epochs=8000, early_stopping_tolerance=1e-4):

        """2-stage fit with freezing/unfreezing NL
        :param Y:
        :param cost_function:
        :param fitter:
        :return:
        """

        fitter_options = {'cost_function': cost_function,  # 'nmse'
                          'early_stopping_tolerance': early_stopping_tolerance*10,
                          'validation_split': 0,
                          'learning_rate': learning_rate*10, 'epochs': int(epochs/2)
                          }
        fitter_options2 = {'cost_function': cost_function,
                           'early_stopping_tolerance': early_stopping_tolerance,
                           'validation_split': 0,
                           'learning_rate': learning_rate, 'epochs': epochs

                           }

        strf = self.sample_from_priors()

        strf.layers[-1].skip_nonlinearity()
        strf = strf.fit(input=X, target=Y, backend=fitter,
                        fitter_options=fitter_options, batch_size=None)
        strf.layers[-1].unskip_nonlinearity()
        log.info('Fit stage 2: with static output nonlinearity')
        strf = strf.fit(input=X, target=Y, backend=fitter,
                        verbose=0, fitter_options=fitter_options2, batch_size=None)

        ymin, ymax = Y.min(axis=tuple(range(Y.ndim - 1))), Y.max(axis=tuple(range(Y.ndim - 1)))
        ydelta = (ymax - ymin) * 0.1
        strf.out_range = [ymin - ydelta, ymax + ydelta]

        return strf

    def get_strf(self, **opts):
        return LNpop_get_strf(self, **opts)

    def plot_strf(self, **opts):
        return LNpop_plot_strf(self, **opts)

    def get_tuning(self, **opts):
        return LNpop_get_tuning(self, **opts)

    @layer('CNNpop')

    def from_keyword(keyword):
        # Return a subclass of Model rather than a layer
        #
        # requires Model.from_keywords method to check for Layer vs. Model
        options = keyword.split('.')
        shape = pop_shape(options)

        d = {}
        if (shape is None):
            raise ValueError("shape must be TxCINxCOUT or TxCINxCOUTxRANK or TxCINxCOUTxRANKxL2SIZE")
        elif len(shape) == 3:
            d['time_bins'] = shape[0]
            d['channels_in'] = shape[1]
            d['channels_out'] = shape[2]
        elif len(shape) == 4:
            d['time_bins'] = shape[0]
            d['channels_in'] = shape[1]
            d['channels_out'] = shape[2]
            d['rank'] = shape[3]
        elif len(shape) == 5:
            d['time_bins'] = shape[0]
            d['channels_in'] = shape[1]
            d['channels_out'] = shape[2]
            d['rank'] = shape[3]
            d['L2'] = shape[4]
        else:
            raise ValueError("shape must be TxCINxCOUT or TxCINxCOUTxRANK")

        for op in options:
            if op == 'g':
                d['gaussian'] = True
            elif op == 'i':
                d['share_tuning'] = False
            elif op in ['lvl', 'dexp', 'relu']:
                d['nonlinearity'] = op
                if op == 'relu':
                    d['nl_kwargs'] = {'no_shift': False, 'no_offset': False}
            elif op.startswith('l2'):
                d['regularizer'] = op
            elif op.startswith('wl2'):
                d['regularizer'] = op[1:]
                d['regularize_fir'] = False
            elif op.startswith('c'):
                d['L1_reps'] = int(op[1:])
        return CNN_pop(**d)

##
## helper functions
##
