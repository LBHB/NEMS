import logging
import matplotlib.pyplot as plt
import numpy as np
import nems0.modelspec as ms
from scipy.ndimage import zoom

from nems0.plots.timeseries import plot_timeseries
from nems0.utils import find_module
from nems0.modules.fir import (pz_coefficients, fir_dexp_coefficients,
                              fir_exp_coefficients, _offset_coefficients)
from nems0.plots.utils import ax_remove_box
from nems0 import get_setting
from nems0.gui.decorators import cursor

log = logging.getLogger(__name__)


def plot_heatmap(array, xlabel='Time', ylabel='Channel',
                 ax=None, cmap=None, clim=None, skip=0, title=None, fs=None,
                 interpolation=None, manual_extent=None, show_cbar=True,
                 fontsize=7, **options):
    '''
    A wrapper for matplotlib's plt.imshow() to ensure consistent formatting.
    '''
    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = get_setting('WEIGHTS_CMAP')

    # Make sure array is converted to ndarray if passed as list
    array = np.array(array)

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if manual_extent is not None:
        extent = manual_extent
    elif fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    im=ax.imshow(array, aspect='auto', origin='lower', cmap=cmap,
               clim=clim, interpolation=interpolation, extent=extent)

    # Force integer tick labels, skipping gaps
    #y, x = array.shape

    #plt.xticks(np.arange(skip, x), np.arange(0, x-skip))
    #plt.xticklabels(np.arange(0, x-skip))
    #plt.yticks(np.arange(skip, y), np.arange(0, y-skip))
    #plt.yticklabels(np.arange(0, y-skip))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_cbar:
    # Set the color bar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        cbar.set_label('Gain', fontsize=fontsize)
        cbar.outline.set_edgecolor('white')

    if title is not None:
        ax.set_title(title)

    #ax_remove_box(ax)
    return ax


def _get_wc_coefficients(modelspec, idx=0):
    i = 0
    for m in modelspec:
        if 'weight_channels' in m['fn']:
            if 'fn_coefficients' in m.keys():
                if i == idx:
                    fn = ms._lookup_fn_at(m['fn_coefficients'])
                    kwargs = {**m['fn_kwargs'], **m['phi']}  # Merges dicts
                    return fn(**kwargs)
                else:
                    i += 1
            else:
                if i == idx:
                    return m['phi']['coefficients'].copy()
                else:
                    i += 1
    return None


def _get_fir_coefficients(modelspec, idx=0, fs=None):
    i = 0
    for m in modelspec:
        if 'fir' in m['fn']:
            if 'fn_coefficients' in m.keys():
                fn = ms._lookup_fn_at(m['fn_coefficients'])
                kwargs = {**m['fn_kwargs'], **m['phi']}  # Merges dicts
                return fn(**kwargs)

            #elif 'pole_zero' in m['fn']:
            #    c = pz_coefficients(poles=m['phi']['poles'],
            #                        zeros=m['phi']['zeros'],
            #                        delays=m['phi']['delays'],
            #                        gains=m['phi']['gains'],
            #                        n_coefs=m['fn_kwargs']['n_coefs'], fs=100)
            #    return c
            elif 'dexp' in m['fn']:
                c = fir_dexp_coefficients(phi=m['phi']['phi'],
                                          n_coefs=m['fn_kwargs']['n_coefs'])
                return c
            elif 'exp' in m['fn']:
                tau = m['phi']['tau']

                if 'a' in m['phi']:
                    a = m['phi']['a']
                else:
                    a = m['fn_kwargs']['a']

                if 'b' in m['phi']:
                    b = m['phi']['b']
                else:
                    b = m['fn_kwargs']['b']

                c = fir_exp_coefficients(tau, a=a, b=b,
                                         n_coefs=m['fn_kwargs']['n_coefs'])
                return c
            elif i == idx:
                coefficients = m['phi']['coefficients']
                if 'offsets' in m['phi']:
                    if fs is None:
                        log.warning("couldn't compute offset coefficients for "
                                    "STRF heatmap, no fs provided.")
                    else:
                        coefficients = _offset_coefficients(coefficients,
                                                            m['phi']['offsets'],
                                                            fs=fs)
                return coefficients
            else:
                i += 1
    return None


def weight_channels_heatmap(modelspec, idx=None, ax=None, clim=None, title=None,
                            chan_names=None, wc_idx=0, **options):
    """
    :param modelspec: modelspec object
    :param idx: index into modelspec
    :param ax:
    :param clim:
    :param title:
    :param chan_names: labels for x axis
    :param wc_idx:
    :param options:
    :return:
    """
    if idx is not None:
        # module has been specified
        coefficients = _get_wc_coefficients(modelspec[idx:], idx=0)
    else:
        # weird old way: get the idx-th set of coefficients
        coefficients = _get_wc_coefficients(modelspec, idx=wc_idx)

    # normalize per channel:
    #coefficients /= np.std(coefficients, axis=0, keepdims=True)

    # make bigger dimension horizontal
    if coefficients.shape[0]>coefficients.shape[1]:
        ax = plot_heatmap(coefficients.T, xlabel='Channel Out', ylabel='Channel In',
                     ax=ax, clim=clim, title=title, cmap=get_setting('WEIGHTS_CMAP'))
    else:
        ax = plot_heatmap(coefficients, xlabel='Channel In', ylabel='Channel Out',
                     ax=ax, clim=clim, title=title, cmap=get_setting('WEIGHTS_CMAP'))

    if chan_names is None:
        chan_names = []
    elif type(chan_names) is int:
        chan_names = [chan_names]

    for i, c in enumerate(chan_names):
        plt.text(i, 0, c)

    return ax

def fir_heatmap(modelspec, ax=None, clim=None, title=None, chans=None,
                fir_idx=0, cmap=None, **options):
    coefficients = _get_fir_coefficients(modelspec, idx=fir_idx)
    if cmap is None:
        cmap = get_setting('FILTER_CMAP')
    plot_heatmap(coefficients, xlabel='Time Bin', ylabel='Channel In',
                 ax=ax, clim=clim, cmap=cmap, title=title)
    if chans is not None:
        for i, c in enumerate(chans):
            plt.text(-0.4, i, c, verticalalignment='center')


def nonparametric_strf(modelspec, idx, ax=None, clim=None, title=None, cmap=None, **kwargs):
    coefficients = modelspec[idx]['phi']['coefficients']
    if cmap is None:
        cmap = get_setting('FILTER_CMAP')
    plot_heatmap(coefficients, xlabel='Time Bin', ylabel='Channel In',
                 ax=ax, clim=clim, cmap=cmap, title=title)


def strf_heatmap(modelspec, ax=None, clim=None, show_factorized=True,
                 title='STRF', fs=None, chans=None, wc_idx=0, fir_idx=0,
                 interpolation=None, absolute_value=False, cmap=None,
                 manual_extent=None, show_cbar=True, **options):
    """
    chans: list
       if not None, label each row of the strf with the corresponding
       channel name
    interpolation: string, tuple
       if string, passed on as parameter to imshow
       if tuple, ndimage "zoom" by a factor of (x,y) on each dimension
    """
    if interpolation is None:
        interpolation = get_setting('FILTER_INTERPOLATION')
    if fs is None:
        try:
            fs = modelspec.recording['stim'].fs
        except:
            fs=1
    if cmap is None:
        cmap = get_setting('FILTER_CMAP')
    if cmap is None:
        cmap = 'RdYlBu_r'

    wcc = _get_wc_coefficients(modelspec, idx=wc_idx)
    firc = _get_fir_coefficients(modelspec, idx=fir_idx, fs=fs)
    fir_mod = find_module('fir', modelspec, find_all_matches=True)[fir_idx]

    if wcc is None and firc is None:
        log.warn('Unable to generate STRF.')
        return
    elif wcc is None and firc is not None:
        strf = np.array(firc)
        show_factorized = False
    elif wcc is not None and firc is None:
        strf = np.array(wcc).T
        show_factorized = False
    elif 'filter_bank' in modelspec[fir_mod]['fn']:
        wc_coefs = np.array(wcc).T
        fir_coefs = np.array(firc)

        bank_count = modelspec[fir_mod]['fn_kwargs']['bank_count']
        chan_count = wcc.shape[0]
        bank_chans = int(chan_count / bank_count)
        if wc_coefs.shape[1]==fir_coefs.shape[0]:
            strfs = [wc_coefs[:, (bank_chans*i):(bank_chans*(i+1))] @
                              fir_coefs[(bank_chans*i):(bank_chans*(i+1)), :]
                              for i in range(bank_count)]
            for i in range(bank_count):
                m = np.max(np.abs(strfs[i]))
                if m:
                    strfs[i] = strfs[i] / m
                if i > 0:
                    gap = np.full([strfs[i].shape[0], 1], np.nan)
                    strfs[i] = np.concatenate((gap, strfs[i]/np.max(np.abs(strfs[i]))), axis=1)

            strf = np.concatenate(strfs,axis=1)
        else:
            strf = fir_coefs
        # special case, can't do this stuff for filterbank display
        interpolation=None
        show_factorized = False
    else:
        wc_coefs = np.array(wcc).T
        fir_coefs = np.array(firc)
        if wc_coefs.shape[1] == fir_coefs.shape[0]:
            strf = wc_coefs @ fir_coefs
        else:
            strf = fir_coefs
            show_factorized = False

    if not clim:
        cscale = np.nanmax(np.abs(strf.reshape(-1)))
        clim = [-cscale, cscale]
    else:
        cscale = np.max(np.abs(clim))

    if interpolation is None:
        pass
    elif type(interpolation) is str:
        pass
    else:
        strf = zoom(strf, interpolation)
        fs = fs * float(interpolation[1])
        show_factorized=False
    #import pdb; pdb.set_trace()
    interpolation='none'

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
        skip = nchans + 1
    else:
        everything = strf
        skip = 0

    if absolute_value:
        everything = np.abs(everything)

    plot_heatmap(everything, xlabel='Lag (s)',
                 ylabel='Channel In', ax=ax, skip=skip, title=title, fs=fs,
                 interpolation=interpolation, cmap=cmap,
                 manual_extent=manual_extent, show_cbar=show_cbar)
    #ax_remove_box(left=True, bottom=True, ticks=True)

    if chans is not None:
        for i, c in enumerate(chans):
            plt.text(0, i + nchans + 1, c, verticalalignment='center')


@cursor
def strf_local_lin(rec, modelspec, cursor_time=20, channels=0,
                   **options):
    rec = rec.copy()

    tbin = int(cursor_time * rec['resp'].fs)

    chan_count = rec['stim'].shape[0]
    firmod = find_module('fir', modelspec)
    tbin_count = modelspec.phi[firmod]['coefficients'].shape[1]+2

    use_dstrf = True
    if use_dstrf:
        index = int(cursor_time * rec['resp'].fs)
        strf = modelspec.get_dstrf(rec, index=index, width=20,
                                   out_channel=channels)
    else:
        resp_chan = channels
        d = rec['stim']._data.copy()
        strf = np.zeros((chan_count, tbin_count))
        _p1 = rec['pred']._data[resp_chan, tbin]
        eps = np.nanstd(d) / 100
        eps = 0.01
        #print('eps: {}'.format(eps))
        for c in range(chan_count):
            #eps = np.std(d[c, :])/100
            for t in range(tbin_count):

                _d = d.copy()
                _d[c, tbin - t] *= 1+eps
                rec['stim'] = rec['stim']._modified_copy(data=_d)
                rec = modelspec.evaluate(rec)
                _p2 = rec['pred']._data[resp_chan, tbin]
                strf[c, t] = (_p2 - _p1) / eps
    print('strf min: {} max: {}'.format(np.min(strf), np.max(strf)))
    options['clim'] = np.array([-np.max(np.abs(strf)), np.max(np.abs(strf))])
    plot_heatmap(strf, cmap=get_setting('FILTER_CMAP'), **options)


def strf_timeseries(modelspec, ax=None, show_factorized=True,
                    show_fir_only=True,
                    title=None, fs=1, chans=None, colors=None, **options):
    """
    chans: list
       if not None, label each row of the strf with the corresponding
       channel name
    """
    wcc = _get_wc_coefficients(modelspec)
    firc = _get_fir_coefficients(modelspec)
    if wcc is None and firc is None:
        log.warn('Unable to generate STRF.')
        return
    elif show_fir_only or (wcc is None):
        strf = np.array(firc)
    elif wcc is not None and firc is None:
        strf = np.array(wcc).T
    else:
        wc_coefs = np.array(wcc).T
        fir_coefs = np.array(firc)
        if wc_coefs.shape[1] == fir_coefs.shape[0]:
            strf = wc_coefs @ fir_coefs
        else:
            strf = fir_coefs

    times = [np.arange(strf.shape[1])/fs] * strf.shape[0]
    filters = [strf[i] for i in range(strf.shape[0])]
    if colors is None:
        if strf.shape[0] == 1:
            colors = [[0, 0, 0]]
        elif strf.shape[0] == 2:
            colors = [[254 / 255, 15 / 255, 6 / 255],
                      [129 / 255, 201 / 255, 224 / 255]
                      ]
        elif strf.shape[0] == 3:
            colors = [[254/255, 15/255, 6/255],
                      [217/255, 217/255, 217/255],
                      [129/255, 201/255, 224/255]
                      ]
        elif strf.shape[0] > 3:
            colors = [[254/255, 15/255, 6/255],
                      [217/255, 217/255, 217/255],
                      [129/255, 201/255, 224/255],
                      [128/255, 128/255, 128/255],
                      [32/255, 32/255, 32/255]
                      ]
    #import pdb
    #pdb.set_trace()
    _,strf_h=plot_timeseries(times, filters, xlabel='Time lag', ylabel='Gain',
                    legend=chans, linestyle='-', linewidth=1,
                    ax=ax, title=title, colors=colors)
    plt.plot(times[0][[0, len(times[0])-1]], np.array([0, 0]), linewidth=0.5, color='gray')

    if show_factorized and not show_fir_only:
        wcN=wcc.shape[0]

        ax.set_prop_cycle(None)
        _,fir_h=plot_timeseries([times], [firc.T], xlabel='Time lag', ylabel='Gain',legend=chans, linestyle='--', linewidth=1,ax=ax, title=title)

        ax.set_prop_cycle(None)
        weight_x=np.arange(-1*wcN,0)
        w_h=ax.plot(weight_x, wcc)
        ax.plot(weight_x, np.array([0, 0]), linewidth=0.5, color='gray')
        ax.set_xlim((-1*wcN,len(times)))
        strf_l=['Weighted FIR {}'.format(n+1) for n in range(wcN)]
        fir_l=['Raw FIR {}'.format(n+1) for n in range(wcN)]
        plt.legend(strf_h+fir_h,strf_l+fir_l, loc=1,fontsize='x-small')
        ax.set_xticks(np.hstack((np.arange(-1*wcN,0),np.arange(0,len(times)+1,2))))
        ax.set_xticklabels(np.hstack((np.arange(1,wcN+1),np.arange(0,len(times)+1,2))))
