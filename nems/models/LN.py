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

#
# Class defs
#
class LN_STRF(Model):

    def __init__(self, time_bins=None, channels=None, rank=None,
                 gaussian=False, nonlinearity='DoubleExponential', final_stride=1,
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
        self.final_stride = final_stride

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
                 gaussian=False, nonlinearity='DoubleExponential', final_stride=1,
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
            fir = FiniteImpulseResponse(shape=(time_bins, channels_in, channels_out), regularizer=regularizer)
            self.add_layers(fir)
        elif share_tuning:
            wc = wc_class1(shape=(channels_in, 1, rank), regularizer=reg1)
            fir = FiniteImpulseResponse(shape=(time_bins, 1, rank))
            wc2 = WeightChannels(shape=(rank, channels_out), regularizer=regularizer)
            self.add_layers(wc, fir, wc2)
        else:
            wc = wc_class1(shape=(channels_in, rank, channels_out), regularizer=reg1)
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

        if final_stride > 1:
            agg = FiniteImpulseResponse(shape=(final_stride, 1, channels_out), stride=final_stride)
            agg['coefficients'] = np.ones(agg.shape)/final_stride
            agg.freeze_parameters('coefficients')
            self.add_layers(agg)

        self.out_range = [[-1]*channels_out, [3]*channels_out]
        self.final_stride = final_stride

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
        if self.final_stride > 1:
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
                 binaural=None, ax=None, fs=100,
                 show_tuning=True, label="", **tuningkwargs):
    if channels is None:
        channels=[0]
    if strf is None:
        strf = model.get_strf(channels=channels)
    if model is not None:
        rtest = model.meta.get('r_test', np.zeros((np.array(channels).max()+1, 1)))
        rtest = rtest[channels[0], 0]
        if binaural is None:
            binaural = is_binaural(model)
    if binaural is None:
        binaural = False
    if ax is None:
        f, ax = plt.subplots()

    if show_tuning:
        if binaural:
            res = get_binaural_strf_tuning(strf, **tuningkwargs)
            prelist = ['c', 'i']
        else:
            res = get_strf_tuning(strf, **tuningkwargs)
            res['ipsi_offset']=0
            prelist = ['']

    # mm = np.max(np.abs(strf))
    extent = [0, strf.shape[1] / fs, 0, strf.shape[0]]
    mm = np.max(np.abs(strf))
    ax.imshow(strf, aspect='auto', cmap='bwr',
            origin='lower', interpolation='none', extent=extent,
            vmin=-mm, vmax=mm)
    if show_tuning:
        for ci, p in enumerate(prelist):
            os = ci * res['ipsi_offset'] + 0.5
            b0 = res[p + 'bfidx'] + os
            ax.plot(res[p + 'lat'], b0, '.', color='black')
            ax.plot(res[p + 'offlat'], b0, '.', color='black')
            mlat = (res[p + 'lat'] + res[p + 'offlat']) / 2
            ax.plot(mlat, res[p + 'bloidx'] + os, '.', color='black')
            ax.plot(mlat, res[p + 'bhiidx'] + os, '.', color='black')

    if binaural:
        ax.axhline(y=res['ipsi_offset'], lw=0.5, ls='--', color='gray')
        ax.text(extent[0], extent[3],
                f"{label} r={rtest:.2f} bdi:{res['bdi']:.2f} mcs:{res['mcs']:.2f}",
                va='bottom')
    else:
        ax.text(extent[0], extent[3], f"{label} r={rtest:.2f}")
    ax.set_xlabel('Time lag')


def LNpop_plot_strf(model, labels=None, channels=None,
                    layer=2, plot_nl=False, merge=None,
                    binaural=None, show_tuning=True, figsize=None):
    strf2 = LNpop_get_strf(model, channels=channels, layer=layer)
    if binaural is None:
        binaural=is_binaural(model)

    channels_out = strf2.shape[-1]
    if channels is None:
        channels = np.arange(channels_out)

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

    if plot_nl:
        col_mult = 2
    else:
        col_mult = 1

    if channels_out > 3:
        rowcount = int(np.ceil(np.sqrt(channels_out)))
        colcount = int(np.ceil(channels_out/rowcount))
    elif channels_out > 16:
        rowcount = np.min([channels_out, 5])
        colcount = int(np.ceil(channels_out / 5))
    else:
        rowcount = np.min([channels_out, 4])
        colcount = int(np.ceil(channels_out / 4))

    if figsize is None:
        figsize = (colcount*col_mult*1.0, rowcount*0.75)

    f, ax = plt.subplots(rowcount, colcount * col_mult, figsize=figsize, sharex=True, sharey=True)

    if (colcount*col_mult)==1:
        ax=np.array([ax]).T
    for c, ch in enumerate(channels):
        cc = c % colcount
        rr = int(np.floor(c/colcount))
        # call single-STRF plotter to actually generate the plot
        if labels is not None:
            lbl = labels[c]
        else:
            lbl = f"ch{ch}"
        LN_plot_strf(model=model, channels=[ch], strf=strf2[:, :, c],
                     binaural=binaural, ax=ax[rr, cc*col_mult], fs=fs,
                     show_tuning=show_tuning, label=lbl)
        yl = ax[rr,cc*col_mult].get_ylim()
        if rr<rowcount-1:
            ax[rr,cc*col_mult].set_xlabel('')

    ax[-1,0].set_xlabel('Time lag')
    plt.suptitle(model.name[:30])
    plt.tight_layout()

    return f


def get_strf_tuning(strf, binaural=False, fmin=200, fmax=20000, timestep=0.01):
    if binaural:
        return get_binaural_strf_tuning(strf, fmin=fmin, fmax=fmax, timestep=timestep)
    # figure out some tuning properties
    maxoct = int(np.log2(fmax/fmin))

    sf = 4
    strfsmooth = zoom(strf, sf)
    strfsmooth[np.abs(strfsmooth) < strfsmooth.std()/2] = 0
    ff = np.exp(np.linspace(np.log(fmin), np.log(fmax), strfsmooth.shape[0]))

    onsetbins = int(0.6/timestep*sf)
    mm = np.mean(strfsmooth[:, :onsetbins] * (1*(strfsmooth[:, :onsetbins] > 0)), 1)
    mmneg = -np.mean(strfsmooth[:, :onsetbins] * (1*(strfsmooth[:, :onsetbins] < 0)), 1)

    if (mm.max() < mmneg.max()):
        mm = mmneg
        bfpos = False
    else:
        bfpos = True

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
    res['ipsi_offset']=m

    return res

def LNpop_get_tuning(model, channels=None, layer=2, binaural=None, **tuningargs):
    import pandas as pd
    strf = LNpop_get_strf(model, channels=channels, layer=layer)
    if channels is None:
        channels = np.arange(len(model.meta['cellids']))
    if binaural is None:
        binaural = is_binaural(model)
    dlist=[]
    for i,c in enumerate(channels):
        res = {'cellid': model.meta['cellids'][c]}
        res.update(get_strf_tuning(strf[:,:,i], binaural=binaural, **tuningargs))

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
