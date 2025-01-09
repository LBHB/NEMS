import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
import scipy

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

    def __init__(self, time_bins=None, channels=None, rank=None,
                 gaussian=False, nonlinearity='DoubleExponential', stride=1,
                 nl_kwargs=None, regularizer=None, from_saved=False, **model_init_kwargs):
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
        if from_saved:
           self.out_range = [[-1], [3]]
           return
            
        # Add STRF
        if rank is None:
            # Full-rank finite impulse response
            fir = FiniteImpulseResponse(shape=(time_bins, channels), regularizer=regularizer)
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
        self.stride = stride

    @classmethod
    def from_data(cls, input, filter_duration, sampling_rate=1000, **kwargs):
        channels = input.shape[-1]
        time_bins = int(filter_duration/1000 * sampling_rate)
        # TODO: modify initial parameters based on stimulus statistics?
        return LN_STRF(time_bins, channels, **kwargs)

    def fit_LBHB(self, X,Y, cost_function = 'nmse', fitter='tf',
                 learning_rate = 1e-3, epochs=8000, early_stopping_tolerance = 1e-4):
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

    def plot_strf(self, ax=None, labels=None):
        #strf1 = self.get_strf()

        #channels_out = strf1.shape[-1]
        if ax is None:
            f, ax = plt.subplots(1, 2, figsize=(4, 2))
        elif type(ax) is list:
            pass
        else:
            ax=[ax]

        LN_plot_strf(self, ax=ax[0])
        if len(ax)>1:
            ymin, ymax = self.out_range[0][0], self.out_range[1][0]
            plot_nl(self.layers[-1], range=[ymin, ymax], ax=ax[1])
            plt.tight_layout()
        return ax[0].figure


    def get_tuning(self, binaural=None, **tuningargs):
        strf=self.get_strf()
        if binaural is None:
            binaural=is_binaural(self)
        res = {'cellid': self.meta.get('cellid','cell')}
        res.update(get_strf_tuning(strf, binaural=binaural, **tuningargs))
        return res

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

    def __init__(self, time_bins=None, channels_in=None, channels_out=None, rank=None, L1=None, L2=None, share_tuning=True,
                 gaussian=False, nonlinearity='DoubleExponential', stride=1,
                 nl_kwargs=None, regularizer=None, from_saved=False,
                 include_anticausal=False, **model_init_kwargs):
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
        if from_saved:
            channels_out = self.layers[-1].shape[0]
            self.out_range = [[-1]*channels_out, [3]*channels_out]
            return
        if rank is None:
            rank = L1

        # Add STRF
        wc_class = WeightChannelsGaussian if gaussian else WeightChannels
        if gaussian:
            wc_class1 = WeightChannelsGaussian
            reg1 = None
        else:
            wc_class1 = WeightChannels
            reg1 = regularizer
        if rank is None:
            # Full-rank finite impulse response, one per output channel
            fir = FiniteImpulseResponse(shape=(time_bins, channels_in, channels_out),
                                        regularizer=regularizer, include_anticausal=include_anticausal)
            self.add_layers(fir)
        elif share_tuning:
            wc = wc_class1(shape=(channels_in, 1, rank), regularizer=reg1)
            fir = FiniteImpulseResponse(shape=(time_bins, 1, rank),
                                        include_anticausal=include_anticausal, regularizer=regularizer)
            wc2 = WeightChannels(shape=(rank, channels_out), regularizer=regularizer)
            self.add_layers(wc, fir, wc2)
        else:
            wc = wc_class1(shape=(channels_in, rank, channels_out), regularizer=reg1)
            fir = FiniteImpulseResponse(shape=(time_bins, rank, channels_out), include_anticausal=include_anticausal, regularizer=regularizer)
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

        if stride > 1:
            agg = FiniteImpulseResponse(shape=(stride, 1, channels_out), stride=stride)
            agg['coefficients'] = np.ones(agg.shape)/stride
            agg.freeze_parameters('coefficients')
            self.add_layers(agg)

        self.out_range = [[-1]*channels_out, [3]*channels_out]
        self.stride = stride
        self.meta['out_range']=[[-1]*channels_out, [3]*channels_out]
        self.meta['stride']=stride
        self.meta['include_anticausal']=include_anticausal

    @classmethod
    def from_data(cls, input, output, filter_duration, sampling_rate=1000, **kwargs):
        channels_in = input.shape[-1]
        channels_out = output.shape[-1]
        time_bins = int(filter_duration / 1000 * sampling_rate)
        # TODO: modify initial parameters based on stimulus statistics?
        return LN_pop(time_bins, channels_in, channels_out, **kwargs)

    def fit_LBHB(self, X,Y, cost_function = 'nmse', fitter='tf',
                 learning_rate = 1e-3, epochs=8000, early_stopping_tolerance=1e-4):
        """
        2-stage fit with freezing/unfreezing NL
        :param X: (T, channels_in) or (batch, T, channels_in) input
        :param Y: (T, channels_out) or (batch, T, channels_out) ouput
        :param cost_function:
        :param fitter: default is tf
        :param learning_rate:
        :param epochs:
        :param early_stopping_tolerance:
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
        if self.stride > 1:
            nl_layer = -2
        else:
            nl_layer = -1

        strf.layers[nl_layer].skip_nonlinearity()
        strf = strf.fit(input=X, target=Y, backend=fitter,
                        fitter_options=fitter_options, batch_size=None)
        strf.layers[nl_layer].unskip_nonlinearity()
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

    def get_tuning(self, **opts):
        return LNpop_get_tuning(self, **opts)

    def get_gabor_tuning(self, **opts):
        return LNpop_get_gabor_tuning(self, **opts)

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
            elif op.startswith('l1'):
                d['regularizer'] = op
            elif op.startswith('l2'):
                d['regularizer'] = op
            elif op.startswith('ac'):
                d['include_anticausal'] = True
        return LN_pop(**d)


#
# helper functions
#

def LNpop_get_strf(model, channels=None, layer=2):

    if (model.layers[2].name.startswith("WeightChannels")==False) and \
       (model.layers[2].name.startswith("wc")==False):
        layer=1

    wc = model.layers[0].coefficients
    fir = model.layers[1].coefficients
    filter_count = fir.shape[2]
    strf1 = np.stack([wc[:, :, i] @ fir[:, :, i].T for i in range(filter_count)], axis=2)
    if layer == 1:
        if channels is not None:
            return strf1[:, :, channels]
        else:
            return strf1

    wc2 = model.layers[2].coefficients
    if len(wc2.shape) > 2:
        wc2 = np.squeeze(wc2)
    strf2 = np.tensordot(strf1, wc2, axes=(2, 0))
    if channels is not None:
        strf2 = strf2[:, :, channels]
    return strf2

def LN_plot_strf(model=None, channels=None, strf=None,
                 binaural=None, ax=None, ax2=None, fs=100,
                 show_tuning=False, show_gabor=False, label="", x0=0, y0=0, show_label=True,
                 **tuningkwargs):
    if channels is None:
        channels=[0]
    if strf is None:
        strf = model.get_strf(channels=channels)[:,:,0]
    if model is not None:
        rtest = model.meta.get('r_test', np.zeros((np.array(channels).max()+1, 1)))
        rtest = rtest[channels[0], 0]
        if binaural is None:
            binaural = is_binaural(model)
        ac = model.meta.get('include_anticausal', False)
        fmin = model.meta.get('fmin', model.fmin)
        fmax = model.meta.get('fmax', model.fmax)
    else:
        rtest = np.nan
        ac = False
        fmin=200
        fmax=20000
    if binaural is None:
        binaural = False
    if ax is None:
        f, ax = plt.subplots()

    if show_tuning:
        if 'timestep' not in tuningkwargs.keys():
            tuningkwargs['timestep'] = 1/fs
        if binaural:
            res = get_binaural_strf_tuning(strf, **tuningkwargs)
            prelist = ['c', 'i']
        else:
            res = get_strf_tuning(strf, **tuningkwargs)
            res['ipsi_offset']=0
            prelist = ['']
    elif binaural:
        res = get_binaural_strf_tuning(strf, **tuningkwargs)
        
    # mm = np.max(np.abs(strf))
    logf = np.linspace(np.log2(fmin), np.log2(fmax), strf.shape[0] + 1)
    if binaural:
        res['ipsi_offset'] = np.mean(logf)
    tt = np.arange(strf.shape[1])/fs
    if ac:
        tt=tt-tt[int(len(tt)/2)]
    dt=(tt[1]-tt[0])/2
    extent = [tt[0]-dt+x0, tt[-1]+dt+x0, logf[0]+y0, logf[-1]+y0]
    mm = np.max(np.abs(strf))
    if show_gabor:
        mm = mm*1.2

    if binaural:
        m = int(strf.shape[0]/2)
        hcontra = strf[:m, :]
        hipsi = strf[m:, :]
        
    if binaural & (ax2 is not None):
        ax.imshow(hcontra, aspect='auto', cmap='bwr',
                origin='lower', interpolation='none', extent=extent,
                vmin=-mm, vmax=mm)
        ax2.imshow(hipsi, aspect='auto', cmap='bwr',
                origin='lower', interpolation='none', extent=extent,
                vmin=-mm, vmax=mm)
    else:
        ax.imshow(strf, aspect='auto', cmap='bwr',
                origin='lower', interpolation='none', extent=extent,
                vmin=-mm, vmax=mm)
        lf = np.array(ax.get_yticks())[1:-1]
        fr = np.round(2 ** lf / 1000, 1)
        ax.set_yticks(lf, fr)

    if show_tuning:
        f_ = scipy.interpolate.interp1d(np.arange(len(logf)), logf, fill_value="extrapolate")
        for ci, p in enumerate(prelist):
            os = ci * res['ipsi_offset'] + 0.5
            b0 = f_(res[p + 'bfidx'] + os)
            ax.plot(res[p + 'lat'], b0, '.', color='black')
            ax.plot(res[p + 'offlat'], b0, '.', color='black')
            mlat = (res[p + 'lat'] + res[p + 'offlat']) / 2
            ax.plot(mlat, f_(res[p + 'bloidx'] + os), '.', color='black')
            ax.plot(mlat, f_(res[p + 'bhiidx'] + os), '.', color='black')

    if show_gabor:
        if binaural & (ax2 is not None):
            phiopt, E = strf2gabor(hipsi,fs=fs, fmin=model.fmin, fmax=model.fmax)
            label2 = f"{label} I BF={2**phiopt[0][0]/1000:.1f}K"
            plot_gabor(phiopt[0], ax=ax2, logf=logf, t=tt, show_contours=True, x0=x0, y0=y0)

            phiopt, E = strf2gabor(hcontra,fs=fs, fmin=model.fmin, fmax=model.fmax)
            label = f"{label} C BF={2**phiopt[0][0]/1000:.1f}K"
            plot_gabor(phiopt[0], ax=ax, logf=logf, t=tt, show_contours=True, x0=x0, y0=y0)

        else:
            phiopt, E = strf2gabor(strf, binaural=binaural, fs=fs, fmin=model.fmin, fmax=model.fmax)
            #label = f"{label} E={E[0]:.2f}"
            label = f"{label} BF={2**phiopt[0][0]/1000:.1f}K"
            plot_gabor(phiopt[0], ax=ax, logf=logf, t=tt, show_contours=True, x0=x0, y0=y0)
    else:
        label2 = None
    if show_label==False:
        pass
    elif binaural:
        ax.text(extent[0], extent[3],
                f"{label} r={rtest:.2f}\nbdi:{res['bdi']:.2f} mcs:{res['mcs']:.2f}",
                va='bottom', fontsize=6)
        if (ax2 is None):
            ax.axhline(y=res['ipsi_offset'], lw=0.5, ls='--', color='gray')
        elif label2 is not None:
            ax2.text(extent[0], extent[3],
                    f"{label2}",
                    va='bottom', fontsize=6)
            
    elif np.isfinite(rtest):
        ax.text(extent[0], extent[3], f"{label} r={rtest:.2f}")
    else:
        ax.text(extent[0], extent[3], f"{label}")
    ax.set_xlabel('Time lag')


def LNpop_plot_strf(model, labels=None, channels=None, cell_list=None,
                    layer=2, plot_nl=False, merge=None,
                    binaural=None, show_tuning=False,
                    show_gabor=False, x0=0, y0=0, figsize=None):
    if binaural is None:
        binaural = is_binaural(model)

    if cell_list is not None:
        channels = [i for i,c in enumerate(model.meta['cellids']) if c in cell_list]
    if channels is None:
        strf2 = LNpop_get_strf(model, channels=channels, layer=layer)
        channels = np.arange(strf2.shape[-1])
    strf2 = LNpop_get_strf(model, channels=channels, layer=layer)
    channels_out = len(channels)

    m = int(strf2.shape[0]/2)
    hcontra = strf2[:m, :, :]
    hipsi = strf2[m:, :, :]

    norm = strf2.std(axis=(0,1), keepdims=True)
    if merge=='sum':
        strf2=(hcontra+hipsi)
    elif merge=='diff':
        strf2=(hcontra-hipsi)
    elif merge=='both':
        strf2 = np.concatenate(((hcontra+hipsi), (hcontra-hipsi)),axis=0)

    if model.fs is None:
        fs = 100
    else:
        fs = model.fs
    #wc2 = model.layers[2].coefficients
    #wc2std=wc2.std(axis=0,keepdims=True)
    #wc2std[wc2std==0]=1
    #wc2 /= wc2std

    if channels_out > 3:
        rowcount = int(np.ceil(np.sqrt(channels_out)))
        colcount = int(np.ceil(channels_out/rowcount))
    elif channels_out > 16:
        rowcount = np.min([channels_out, 5])
        colcount = int(np.ceil(channels_out / 5))
    else:
        rowcount = np.min([channels_out, 4])
        colcount = int(np.ceil(channels_out / 4))
        
    if plot_nl:
        col_mult = 2
    else:
        col_mult = 1
    if binaural:
        row_mult = 2
        #rowcount = int(np.ceil(rowcount/2))
        #colcount = colcount*2
    else:
        row_mult = 1
        
    if figsize is None:
        if colcount<=6:
            figsize = (colcount*col_mult*2.0, rowcount*1.5)
        else:
            figsize = (colcount*col_mult*1.0, rowcount*0.75)

    f, ax = plt.subplots(rowcount * row_mult, colcount * col_mult, figsize=figsize, sharex=True, sharey=True)
    if row_mult>1:
        ax2=ax[::2]
        ax=ax[1::2]
    else:
        ax2 = ax
    if (colcount*col_mult)==1:
        ax=np.array([ax]).T
    for c, ch in enumerate(channels):
        cc = c % colcount
        rr = int(np.floor(c/colcount))
        # call single-STRF plotter to actually generate the plot
        if labels is not None:
            lbl = labels[c]
        elif 'cellids' in model.meta.keys():
            lbl = model.meta['cellids'][ch]
        else:
            lbl = f"ch{ch}"
        LN_plot_strf(model=model, channels=[ch], strf=strf2[:, :, c],
                     binaural=binaural, ax=ax[rr, cc*col_mult], ax2=ax2[rr, cc*col_mult], fs=fs,
                     show_tuning=show_tuning, show_gabor=show_gabor, label=lbl)

        yl = ax[rr,cc*col_mult].get_ylim()
        if rr<rowcount-1:
            ax[rr,cc*col_mult].set_xlabel('')

        #yl = ax[rr,cc*col_mult].get_ylim()
        if rr<rowcount-1:
            ax[rr, cc*col_mult].set_xlabel('')
        if cc==0:
            lf = ax[rr, cc].get_yticks()
            fr = np.round(2**np.array(lf)/1000, 1)
            ax[rr,cc].set_yticks(lf, fr)
    ax[-1,0].set_xlabel('Time lag')
    plt.suptitle(model.name[:30])
    plt.tight_layout()

    return f

def Gabor1d(x, X0, BW, W, P, g=1):
    g = g * np.exp(-(x-X0)**2 / (2*BW**2))
    s = np.sin(W *2*np.pi * (x-X0) + P)
    return g * s, g, s

def Gabor2D(phi, x=None, t=None):
    logBF, BW, Wf, Pf, t0, BWt, Wt, Pt, g = phi

    if x is None:
        x = np.linspace(np.log2(200), np.log2(20000), 18)
    if t is None:
        t = np.linspace(0, 0.15, 16)

    S, Sg, Ss = Gabor1d(x, logBF, BW, Wf, Pf, g=g)
    T, Tg, Ts = Gabor1d(t, t0, BWt, Wt, Pt, g=1)
    H = S[:, np.newaxis] * T[np.newaxis,:]

    return H

def plot_gabor(phi, ax=None, logf=None, t=None, show_contours=False, frame_on=False, x0=0, y0=0):
    if ax is None:
        f,ax = plt.subplots()

    if logf is None:
        logf = np.linspace(np.log2(200), np.log2(20000), 18)
    if t is None:
        t = np.linspace(0, 0.150, 16)

    H = Gabor2D(phi, x=logf, t=t)
    mm = np.max(np.abs(H))
    if show_contours:
        if mm > 0:
            ax.contour(t+x0, logf+y0, H, levels=[-mm*0.65, mm*0.65], colors=['b','r'])
        if frame_on:
            ax.plot([t[0]+x0, t[-1]+x0, t[-1]+x0, t[0]+x0, t[0]+x0],
                    [logf[0]+y0, logf[0]+y0, logf[-1]+y0, logf[-1]+y0, logf[0]+y0],
                    lw=0.5, color='lightgray')
    else:
        ax.imshow(H, cmap='bwr', vmin=-mm, vmax=mm,
                  origin='lower', aspect='auto',
                  extent=[t[0]+x0, t[-1]+x0, logf[0]+y0, logf[-1]+y0])

    return ax

def fit_gabor_2d(strf, phi0=None, padbins=6, fs=100, fmin=200, fmax=20000, verbose=False):

    logf = np.linspace(np.log2(fmin), np.log2(fmax), strf.shape[0])
    t = np.linspace(0, 1/fs*(strf.shape[1]-1), strf.shape[1])
    tmax = t[-1]
    phlist=['logBF', 'BW', 'Wf', 'Pf', 't0', 'BWt', 'Wt', 'Pt', 'g']
    if phi0 is None:
        c = get_strf_tuning(strf, binaural=False, fmin=fmin, fmax=fmax, timestep=1/fs)
        f_ = scipy.interpolate.interp1d(np.arange(len(logf)), logf, fill_value="extrapolate")

        Wf0 = 0.25 / c['bw']
        Wt0 = 0.5 / (c['offlat'] - c['lat'])
        Pt0 = np.pi * 0.75
        g0 = np.std(strf) * 2 * np.sign(np.mean(strf))
        # phi0 = [logBF, BW, Wf, Pf, maxlat, latwidth, Wt, Pt, g]
        phi0 = [np.log2(c['bf']), c['bw'] / 2, Wf0, np.pi / 2,
                (c['lat'] + c['offlat']) / 2,
                (c['offlat'] - c['lat']) / 2, Wt0, Pt0, g0]
        if verbose:
            log.info("phi0  "+",".join([f"{n}={p:.3f}" for n,p in zip(phlist,phi0)]))

    # padded stuff
    if padbins > 0:
        strfpadded = np.pad(strf, padbins)
        dlogf = logf[1] - logf[0]
        logfpadded = np.concatenate([logf[0] + np.arange(-padbins, 0) * dlogf,
                                     logf,
                                     logf[-1] + np.arange(1, padbins + 1) * dlogf])
        tpadded = np.linspace(0 - 1/fs * padbins, t[-1] + 1/fs * padbins, len(t) + padbins * 2)

        strf, logf, t = strfpadded, logfpadded, tpadded
    # end padded stuff

    #f = 2 ** logf

    # lambda x: np.sum((strf1-Gabor2D(x, x=logf, t=t)**2))
    def Err(x):
        logBF, BW, Wf, Pf, t0, BWt, Wt, Pt, g = x

        # measure big deviations from the bounds, implementing a crude regularizer
        d = 0
        if (logBF > logf[-padbins]):
            d += (logBF - logf[-padbins])
        if (logBF < logf[padbins]):
            d += logf[padbins] - logBF
        if (BW > 5):
            d += BW - 5
        if (BW < 0.1):
            d += -BW+0.1
        if (np.abs(Wt) > 30):
            d += np.abs(Wt) / 30
        if (t0 > t[-padbins]):
            d += t0 - t[-padbins]
        if (t0 < 0):
            d += -t0
        if (BWt > tmax):
            d += (BWt - tmax)
        if (BWt < 0):
            d += -BWt
        if (Wf > 2):
            d += (Wf - 2)

        alpha = 100
        return np.mean((strf - Gabor2D(x, x=logf, t=t)) ** 2) + d / alpha

    cc = 0
    def Callback(xk):
        nonlocal cc
        cc += 1
        if cc % 500 == 1:
            print(cc, Err(xk))

    # minimize(method=’L-BFGS-B’)
    # minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
    # this is the one that works for the most part
    phiopt = scipy.optimize.fmin(func=Err, x0=phi0, ftol=0.0001, maxiter=5000, disp=False) # , callback=Callback)

    # clean up values in phiopt
    logBF, BW, Wf, Pf, t0, BWt, Wt, Pt, g = phiopt

    if Wf < 0:
        Wf = -Wf
        Pf = -Pf
        g = -g
    if Wt < 0:
        Wt = -Wt
        Pt = -Pt
        g = -g
    while Pf > np.pi / 2:
        Pf -= np.pi
        g = -g
    while Pf < -np.pi / 2:
        Pf += np.pi
        g = -g
    while Pt > np.pi / 2:
        Pt -= np.pi
        g = -g
    while Pt < -np.pi / 2:
        Pt += np.pi
        g = -g

    phiopt = np.array([logBF, BW, Wf, Pf, t0, BWt, Wt, Pt, g])
    E = 1 - Err(phiopt) / np.var(strf)

    if verbose:
        plt.figure()
        #plt.imshow(strf, origin='lower', extent=[t[0],t[-1],logf[0],logf[-1]])
        #plot_gabor(phiopt, ax=plt.gca(), t=t, logf=logf, show_contours=True)
        #log.info("phiopt" + ",".join([f"{n}={p:.3f}" for n, p in zip(phlist, phiopt)]))

    return phiopt, E


def strf2gabor(strf, binaural=False, verbose=False, title=None, **fitopts):

    if binaural:
        m=int(strf.shape[0]/2)
        strflist = [strf[:m, :], strf[18:, :]]
        ylabels = ['Contra','Ipsi']
    else:
        strflist = [strf]
        ylabels = ['Freq channel']

    x = [fit_gabor_2d(s, **fitopts) for s in strflist]
    phiopt = [x_[0] for x_ in x]
    E = [x_[1] for x_ in x]
    if verbose:
        f,ax = plt.subplots(len(phiopt), 2, sharex=True, sharey=True)
        if len(phiopt)==1:
            ax=[ax]
        for phi,a,s,yl,e in zip(phiopt,ax,strflist,ylabels,E):
            LN_plot_strf(strf=s, binaural=False, show_tuning=False, ax=a[0], fs=100, label="STRF")
            a[0].set_ylabel(yl)
            H = Gabor2D(phi)
            LN_plot_strf(strf=H, binaural=False, show_tuning=False, ax=a[1], fs=100, label=f"Gabor fit ({e:.3f})")
        if title is not None:
            f.suptitle(title)
    return phiopt, E


def strf_to_components(strf, binaural=False):
    if binaural:
        mm = int(strf.shape[0]/2)
        strfc = strf[:mm,:,i]
        strfi = strf[mm:,:,i] 
        ucf,scf,vcf = np.linalg.svd(strfc)
        uif,sif,vif = np.linalg.svd(strfi)
        return ucf[:,0],vcf[0,:],uif[:,0],vif[0,:]
    else:
        ucf,scf,vcf = np.linalg.svd(strf)
        return ucf[:,0],vcf[0,:]


def get_strf_tuning(strf, binaural=False, fmin=200, fmax=20000, timestep=0.01):
    if binaural:
        return get_binaural_strf_tuning(strf, fmin=fmin, fmax=fmax, timestep=timestep)
    # figure out some tuning properties
    maxoct = int(np.log2(fmax/fmin))

    sf = 4
    strfsmooth = zoom(strf, sf)
    #strfsmooth = gaussian_filter(strfsmooth, sigma=1)
    #strfsmooth[np.abs(strfsmooth) < strfsmooth.std()/2] = 0
    ff = np.exp(np.linspace(np.log(fmin), np.log(fmax), strfsmooth.shape[0]))
    
    onsetbins = int(0.6/timestep*sf)
    mm = np.mean(strfsmooth[:, :onsetbins] * (1*(strfsmooth[:, :onsetbins] > 0)), 1)
    mmneg = -np.mean(strfsmooth[:, :onsetbins] * (1*(strfsmooth[:, :onsetbins] < 0)), 1)

    if (mm.max() < mmneg.max()):
        mm = mmneg
        bfpos = False
    else:
        bfpos = True

    fcurve, tcurve = strf_to_components(strfsmooth[:,:-8])
    
    if 1:
        mm = np.mean(np.abs(strfsmooth[:,:onsetbins]),axis=1)
        msum=np.cumsum(mm)/np.sum(mm)
        bfidx = np.argwhere(msum>=0.5)[0][0]
        blo = np.argwhere(msum>=0.3)[0][0]
        bhi = np.argwhere(msum>=0.7)[0][0]
        bf = np.round(ff[bfidx])

        tt = np.mean(np.abs(strfsmooth[blo:(bhi+1):,:]),axis=0)
        tsum = np.cumsum(tt)/np.sum(tt)
        latbin = np.argwhere(tsum>=0.25)[0][0]/sf
        offlatbin = np.argwhere(tsum>=0.65)[0][0]/sf
        lat=latbin*timestep
        offlat=offlatbin*timestep

    elif sum(np.abs(mm)) > 0:
        bfidx = np.argwhere(mm==mm.max())[0][0]
        bf = np.round(ff[bfidx])
        blo = np.min(np.argwhere(mm>mm.max()/2))
        bhi = np.max(np.argwhere(mm>mm.max()/2))
        #print(bfidx, blo,bhi)
    else:
        bfidx = 0
        bf, blo, bhi = 0, 0, 0

    res={}
    stepsize = maxoct / strf.shape[0]
    res['bf'] = bf # in Hz
    res['lat'] = lat # lat in sec
    res['offlat'] = offlat # offlat in sec
    res['bfidx'] = bfidx/sf  # in STRF bins
    res['latbin'] = latbin # lat in bins
    res['offlatbin'] = offlatbin
    res['bfpos'] = bfpos
    # bw in octaves
    res['bloidx'] = blo/sf
    res['bhiidx'] = bhi/sf
    res['bw'] = (res['bhiidx'] - res['bloidx']) * stepsize
    res['stepsize'] = stepsize
    res['fcurve'] = fcurve
    res['tcurve'] = tcurve

    return res

def get_binaural_strf_tuning(strf, **kwargs):

    # figure out some tuning properties

    if int(strf.shape[0]/2) != strf.shape[0]/2:
        raise ValueError('Binaural strf must have even shape[0]')
    m=int(strf.shape[0]/2)
    contra=strf[:m,:]
    ipsi=strf[m:,:]

    res={}
    ctuning = get_strf_tuning(contra, **kwargs)
    for k in ctuning:
        res['c'+k] = ctuning[k]
    ituning = get_strf_tuning(ipsi, **kwargs)
    for k in ituning:
        res['i'+k] = ituning[k]

    res['bdi'] = np.std(contra-ipsi) / (contra.std()+ipsi.std())
    res['mcs'] = contra.mean() / np.abs(contra).mean()
    res['mis'] = ipsi.mean() / np.abs(ipsi).mean()
    res['ipsi_offset'] = m
    res['fcorr'] = np.abs(np.corrcoef(res['cfcurve'],res['ifcurve'])[0,1])
    res['tcorr'] = np.abs(np.corrcoef(res['ctcurve'],res['itcurve'])[0,1])

    return res

def LNpop_get_tuning(model, channels=None, cell_list=None, layer=2, binaural=None, **tuningargs):
    import pandas as pd
    
    if cell_list is not None:
        channels = [i for i,c in enumerate(model.meta['cellids']) if c in cell_list]
    strf = LNpop_get_strf(model, channels=channels, layer=layer)
    if channels is None:
        channels = np.arange(len(model.meta['cellids']))
    if binaural is None:
        binaural = is_binaural(model)
    dlist=[]
    for i,c in enumerate(channels):
        res = {'cellid': model.meta['cellids'][c],
               'r_test': model.meta['r_test'][c,0], 
               'r_fit': model.meta['r_fit'][c,0]}
        res.update(get_strf_tuning(strf[:,:,i], binaural=binaural, **tuningargs))
        
        dlist.append(res)
    df = pd.DataFrame(dlist)
    df = df.set_index('cellid')
    return df

def LNpop_get_gabor_tuning(model, channels=None, cell_list=None, layer=2, binaural=None, **fitopts):
    import pandas as pd

    if cell_list is not None:
        channels = [i for i,c in enumerate(model.meta['cellids']) if c in cell_list]
    if channels is None:
        channels = np.arange(len(model.meta['cellids']))
    strf = LNpop_get_strf(model, channels=channels, layer=layer)
    if binaural is None:
        binaural = is_binaural(model)
    dlist=[]
    for i, c in enumerate(channels):
        if 'cellids' in model.meta.keys():
            res = {'cellid': model.meta['cellids'][c]}
        else:
            res = {'cellid': 'cell'}
        if 'fmin' not in fitopts.keys():
            fitopts['fmin'] = model.fmin
        if 'fmax' not in fitopts.keys():
            fitopts['fmax'] = model.fmax
        if 'fs' not in fitopts.keys():
            fitopts['fs'] = model.fs
        phiopt, E = strf2gabor(strf[:,:,i], binaural=binaural,
                               title=res['cellid'], **fitopts)
        if binaural:
            logBF, BW, Wf, Pf, t0, BWt, Wt, Pt, g = phiopt[0]
            res.update({'cBF': 2**logBF, 'clogBF': logBF, 'cBW': BW, 'cWf': Wf,
                        'cPf': Pf, 'ct0': t0, 'cBWt': BWt, 'cWt': Wt,
                        'cPt': Pt, 'cg': g, 'cE': E[0]})
            logBF, BW, Wf, Pf, t0, BWt, Wt, Pt, g = phiopt[1]
            res.update({'iBF': 2**logBF, 'ilogBF': logBF, 'iBW': BW, 'iWf': Wf,
                        'iPf': Pf, 'it0': t0, 'iBWt': BWt, 'iWt': Wt,
                        'iPt': Pt, 'ig': g, 'iE': E[0]})
        else:
            logBF, BW, Wf, Pf, t0, BWt, Wt, Pt, g = phiopt[0]
            res.update({'BF': 2**logBF, 'logBF': logBF, 'BW': BW, 'Wf': Wf,
                        'Pf': Pf, 't0': t0, 'BWt': BWt, 'Wt': Wt,
                        'Pt': Pt, 'g': g, 'E': E[0]})
        dlist.append(res)
    df = pd.DataFrame(dlist)
    df = df.set_index('cellid')
    return df



def is_binaural(model):
    loadkey0 = model.meta.get('loader',None)
    if loadkey0 is None:
        loadkey0 = model.meta.get('loadkey',model.name)
    if '.bin' in loadkey0:
        return True
    else:
        return False
