import logging
import numpy as np
import matplotlib.pyplot as plt

from .base import Model
from nems.registry import layer
from nems.layers.tools import require_shape, pop_shape

from nems.layers import (
    WeightChannels, WeightChannelsGaussian, FiniteImpulseResponse,
    RectifiedLinear, DoubleExponential, LevelShift
    )
from nems.visualization.model import plot_nl

log = logging.getLogger(__name__)

#
# Class defs
#
class LN_STRF(Model):

    def __init__(self, time_bins, channels, rank=None,
                 gaussian=False, nonlinearity='DoubleExponential',
                 nl_kwargs=None, regularizer=None, **model_init_kwargs):
        """Linear-nonlinear Spectro-Temporal Receptive Field model.

        Contains the following layers:
        1. WeightChannels, WeightChannelsGaussian, or None (if full rank).
        2. FiniteImpulseResponse
        3. DoubleExponential, RectifiedLinear, or another static nonlinearity.

        Expects a single sound spectrogram as input, with shape (T, N), and a
        single recorded neural response as a target, with shape (T, 1), where
        T and N are the number of time bins and spectral channels, respectively.

        Based on model architectures described in:
        Thorson, Lienard and David (2015)
        doi: 10.1371/journal.pcbi.1004628

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
        nonlinearity : str; default='RectifiedLinear'.
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

        # Add STRF
        if rank is None:
            # Full-rank finite impulse response
            fir = FiniteImpulseResponse(shape=(time_bins, channels))
            self.add_layers(fir)
        else:
            wc_class = WeightChannelsGaussian if gaussian else WeightChannels
            wc = wc_class(shape=(channels, rank), regularizer=regularizer)
            fir = FiniteImpulseResponse(shape=(time_bins, rank))
            self.add_layers(wc, fir)

        # Add static nonlinearity
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
            raise ValueError(
                f'Unrecognized nonlinearity for LN model:  {nonlinearity}.')
        if nl_class is not None:
            if nl_kwargs is None: nl_kwargs = {}
            nonlinearity = nl_class(shape=(1,), **nl_kwargs)
            self.add_layers(nonlinearity)
        self.out_range = [[-1], [3]]

    @classmethod
    def from_data(cls, input, filter_duration, sampling_rate=1000, **kwargs):
        channels = input.shape[-1]
        time_bins = int(filter_duration/1000 * sampling_rate)
        # TODO: modify initial parameters based on stimulus statistics?
        return LN_STRF(time_bins, channels, **kwargs)

    def fit_LBHB(self, X,Y, cost_function = 'nmse', fitter='tf'):
        """2-stage fit with freezing/unfreezing NL
        :param Y:
        :param cost_function:
        :param fitter:
        :return:
        """

        fitter_options = {'cost_function': cost_function,  # 'nmse'
                          'early_stopping_tolerance': 1e-3,
                          'validation_split': 0,
                          'learning_rate': 1e-2, 'epochs': 3000
                          }
        fitter_options2 = {'cost_function': cost_function,
                           'early_stopping_tolerance': 1e-4,
                           'validation_split': 0,
                           'learning_rate': 1e-3, 'epochs': 8000
                           }

        strf = self.sample_from_priors()
        log.info('Fit stage 1: w/o static output nonlinearity')
        strf.layers[-1].skip_nonlinearity()
        strf = strf.fit(input=X, target=Y, backend=fitter,
                        fitter_options=fitter_options, batch_size=None)
        strf.layers[-1].unskip_nonlinearity()
        log.info('Fit stage 2: with static output nonlinearity')
        strf = strf.fit(input=X, target=Y, backend=fitter,
                        verbose=0, fitter_options=fitter_options2, batch_size=None)

        ymin, ymax = Y.min(axis=tuple(range(Y.ndim - 1))), Y.max(axis=tuple(range(Y.ndim - 1)))
        ydelta = (ymax-ymin) * 0.1
        strf.out_range = [ymin-ydelta, ymax+ydelta]

        return strf

    def get_strf(self, channels=None):
        wc = self.layers[0].coefficients
        fir = self.layers[1].coefficients
        strf1 = wc @ fir.T

        return strf1

    def plot_strf(self, labels=None):
        strf1 = self.get_strf()

        channels_out = strf1.shape[-1]
        f, ax = plt.subplots(1, 2, figsize=(4, 2))

        mm = np.nanmax(abs(strf1))
        if self.fs is not None:
            extent = [0, strf1.shape[0] / self.fs, 0, strf1.shape[1]]
        else:
            extent = [0, strf1.shape[0], 0, strf1.shape[1]]
        ax[0].imshow(strf1, aspect='auto', interpolation='none', origin='lower',
                     cmap='bwr', vmin=-mm, vmax=mm, extent=extent)

        if self.fs is not None:
            ax[0].set_xlabel('Time lag (s)')
        else:
            ax[0].set_xlabel('Time lag (bins)')
        ax[0].set_ylabel('Input channel')

        ymin, ymax = self.out_range[0][0], self.out_range[1][0]
        plot_nl(self.layers[-1], range=[ymin, ymax], ax=ax[1])
        plt.tight_layout()
        return f

    # TODO
    @layer('LN')
    def from_keyword(keyword):
        # Return a subclass of Model rather than a layer
        #
        # requires Model.from_keywords method to check for Layer vs. Model
        options = keyword.split('.')
        shape = pop_shape(options)

        d={}
        if (shape is None):
            raise ValueError("shape must be TxC or TxCxR")
        elif len(shape)==2:
            d['time_bins']=shape[0]
            d['channels']=shape[1]
        elif len(shape)==3:
            d['time_bins']=shape[0]
            d['channels']=shape[1]
            d['rank']=shape[2]
        else:
            raise ValueError("shape must be TxC or TxCxR")
            
        for op in options:
            if op=='g':
                d['gaussian']=True
            elif op in ['lvl','dexp','relu']:
                d['nonlinearity']=op
                if op=='relu':
                    d['nl_kwargs'] = {'no_shift': False, 'no_offset': False}
            elif op.startswith('l2'):
                d['regularizer']=op
        return LN_STRF(**d)


class LN_pop(Model):

    def __init__(self, time_bins, channels_in, channels_out, rank=None, share_tuning=True,
                 gaussian=False, nonlinearity='DoubleExponential',
                 nl_kwargs=None, regularizer=None, **model_init_kwargs):
        """Linear-nonlinear Spectro-Temporal Receptive Field model.

        Contains the following layers:
        1. WeightChannels, WeightChannelsGaussian, or None (if full rank).
        2. FiniteImpulseResponse
        3. DoubleExponential, RectifiedLinear, or another static nonlinearity.

        Expects a single sound spectrogram as input, with shape (T, N), and a
        single recorded neural response as a target, with shape (T, 1), where
        T and N are the number of time bins and spectral channels, respectively.

        Based on model architectures described in:
        Thorson, Lienard and David (2015)
        doi: 10.1371/journal.pcbi.1004628

        Parameters
        ----------
        time_bins : int.
            Number of "taps" in FIR filter. We have found that a 150-250ms filter
            typically sufficient for modeling A1 responses, or 15-25 bins for
            a 100 Hz sampling rate.
        channels_in : int.
            Number of spectral channels in spectrogram.
        channels_out : int.
            Number of output channels (neurons) to predict activity
        share_tuning : bool; optional.
            If True (default), project channels_in to intermediate space of size rank then FIR, then channels_out
            If False, separate STRFs for each channel (full or partial rank depending on rank parameter)
        rank : int; optional.
            Number of spectral weightings used as input to a reduced-rank filter.
            For example, `rank=1` indicates a frequency-time separable STRF.
            If unspecified, a full-rank filter will be used.
        nonlinearity : str; default='RectifiedLinear'.
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

        # Add STRF
        wc_class = WeightChannelsGaussian if gaussian else WeightChannels

        if rank is None:
            # Full-rank finite impulse response, one per output channel
            fir = FiniteImpulseResponse(shape=(time_bins, channels_in, channels_out))
            self.add_layers(fir)
        elif share_tuning:
            wc = wc_class(shape=(channels_in, 1, rank), regularizer=regularizer)
            fir = FiniteImpulseResponse(shape=(time_bins, 1, rank))
            wc2 = wc_class(shape=(rank, channels_out), regularizer=regularizer)
            self.add_layers(wc, fir, wc2)
        else:
            wc = wc_class(shape=(channels_in, rank, channels_out), regularizer=regularizer)
            fir = FiniteImpulseResponse(shape=(time_bins, rank, channels_out))
            self.add_layers(wc, fir)

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
            raise ValueError(
                f'Unrecognized nonlinearity for LN model:  {nonlinearity}.')
        if nl_class is not None:
            if nl_kwargs is None: nl_kwargs = {}
            nonlinearity = nl_class(shape=(channels_out,), **nl_kwargs)
            self.add_layers(nonlinearity)
        self.out_range = [[-1]*channels_out, [3]*channels_out]

    @classmethod
    def from_data(cls, input, output, filter_duration, sampling_rate=1000, **kwargs):
        channels_in = input.shape[-1]
        channels_out = output.shape[-1]
        time_bins = int(filter_duration / 1000 * sampling_rate)
        # TODO: modify initial parameters based on stimulus statistics?
        return LN_pop(time_bins, channels_in, channels_out, **kwargs)

    def fit_LBHB(self, X,Y, cost_function = 'nmse', fitter='tf'):
        """2-stage fit with freezing/unfreezing NL
        :param Y:
        :param cost_function:
        :param fitter:
        :return:
        """

        fitter_options = {'cost_function': cost_function,  # 'nmse'
                          'early_stopping_tolerance': 1e-3,
                          'validation_split': 0,
                          'learning_rate': 1e-2, 'epochs': 3000
                          }
        fitter_options2 = {'cost_function': cost_function,
                           'early_stopping_tolerance': 1e-4,
                           'validation_split': 0,
                           'learning_rate': 1e-3, 'epochs': 8000
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
        ydelta = (ymax-ymin) * 0.1
        strf.out_range = [ymin-ydelta, ymax+ydelta]

        return strf

    def get_strf(self, **opts):
        return LNpop_get_strf(self, **opts)

    def plot_strf(self, **opts):
        return LNpop_plot_strf(self, **opts)

    @layer('LNpop')
    def from_keyword(keyword):
        # Return a subclass of Model rather than a layer
        #
        # requires Model.from_keywords method to check for Layer vs. Model
        options = keyword.split('.')
        shape = pop_shape(options)

        d={}
        if (shape is None):
            raise ValueError("shape must be TxCINxCOUT or TxCINxCOUTxRANK")
        elif len(shape)==3:
            d['time_bins']=shape[0]
            d['channels_in']=shape[1]
            d['channels_out']=shape[2]
        elif len(shape)==4:
            d['time_bins']=shape[0]
            d['channels_in']=shape[1]
            d['channels_out']=shape[2]
            d['rank']=shape[3]
        else:
            raise ValueError("shape must be TxCINxCOUT or TxCINxCOUTxRANK")
            
        for op in options:
            if op=='g':
                d['gaussian']=True
            elif op=='i':
                d['share_tuning']=False
            elif op in ['lvl','dexp','relu']:
                d['nonlinearity']=op
                if op=='relu':
                    d['nl_kwargs'] = {'no_shift': False, 'no_offset': False}
            elif op.startswith('l2'):
                d['regularizer']=op
        return LN_pop(**d)


class LN_reconstruction(Model):

    def __init__(self, time_bins, channels, out_channels, rank=None,
                 nonlinearity='RectifiedLinear',
                 nl_kwargs=None, **model_init_kwargs):
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
                 nl_kwargs=None, **model_init_kwargs):
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



#
# helper functions
#

def LNpop_get_strf(model, channels=None, layer=2):
    wc = model.layers[0].coefficients
    fir = model.layers[1].coefficients
    wc2 = model.layers[2].coefficients
    filter_count = fir.shape[2]
    strf1 = np.stack([wc[:, :, i] @ fir[:, :, i].T for i in range(filter_count)], axis=2)
    if layer==1:
        return strf1

    strf2 = np.tensordot(strf1, wc2, axes=(2, 0))
    if channels is not None:
        strf2 = strf2[:, :, channels]
    return strf2

def LNpop_plot_strf(model, labels=None, channels=None, layer=2, plot_nl=False, merge=None):
    strf2 = LNpop_get_strf(model, channels=channels, layer=layer)

    hcontra = strf2[:18, :, :]
    hipsi = strf2[18:, :, :]

    norm = strf2.std(axis=(0,1), keepdims=True)
    if merge=='sum':
        strf2=(hcontra+hipsi)
    elif merge=='diff':
        strf2=(hcontra-hipsi)
    elif merge=='both':
        strf2 = np.concatenate(((hcontra+hipsi), (hcontra-hipsi)),axis=0)

    if model.fs is None:
        fs=1
    else:
        fs=model.fs
    wc2 = model.layers[2].coefficients
    wc2std=wc2.std(axis=0,keepdims=True)
    wc2std[wc2std==0]=1
    wc2 /= wc2std
    channels_out = strf2.shape[-1]
    colcount=int(np.ceil(channels_out/10))
    rowcount=np.min([channels_out,10])
    if plot_nl:
        col_mult=2
    else:
        col_mult=1
    f, ax = plt.subplots(rowcount, colcount*col_mult, figsize=(colcount*col_mult,rowcount*0.75),
                         sharex='col', sharey='col')
    if (colcount*col_mult)==1:
        ax=np.array([ax]).T
    for c in range(channels_out):
        rr = c % 10
        cc = int(np.floor(c/10))
        #mm = np.max(np.abs(strf2[:,:,c]))
        mm=norm[0,0,c]*6
        extent = [0, strf2.shape[1]/fs, 0, strf2.shape[0]]
        ax[rr, cc*col_mult].imshow(strf2[:, :, c], aspect='auto', cmap='bwr',
                        origin='lower', interpolation='none', extent=extent,
                        vmin=-mm, vmax=mm)
        if plot_nl:
            if (layer==2):
                xmin, xmax = modelmodel.out_range[0][c], model.out_range[1][c]
                plot_nl(model.layers[-1], range=[xmin, xmax], channel=c, ax=ax[rr, cc*2+1])
            else:
                ax[rr, cc*2+1].plot(wc2[c])
                ax[rr, cc*2+1].axhline(y=0, ls='--', color='black')
            ax[rr, cc * 2+1].set_ylabel('')
            ax[rr, cc * 2+1].set_yticklabels([])
            ax[rr, cc * 2+1].set_xlabel('')

        if labels is not None:
            ax[rr, cc*col_mult].text(extent[0],extent[3],labels[c],va='bottom')
        if cc>0:
            ax[rr,cc*col_mult].set_yticklabels([])

    ax[-1,0].set_xlabel('Time lag')
    plt.suptitle(model.name[:30])
    plt.tight_layout()

    return f
