import numpy as np
import matplotlib.pyplot as plt
import importlib
import nems0.modelspec as ms
import nems0.metrics.api as nm
from nems0.plots.utils import ax_remove_box
from nems0.gui.decorators import cursor

import logging
log = logging.getLogger(__name__)


def plot_nl_io(module=None, xbounds=None, ax=None):

    if module is None:
        return
    if xbounds is None:
        xbounds = np.array([-1, 1])
    if ax:
        #plt.sca(ax)
        pass
    else:
        ax=plt.gca()

    module_name, function_name = module['fn'].rsplit('.', 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, '_' + function_name)
    keys = list(module['phi'].keys())
    chancount = len(module['phi'][keys[0]])
    d_in = np.linspace(xbounds[0], xbounds[1], 100)
    if chancount > 1:
        d_in = np.tile(d_in, (chancount, 1))
    d_out = fn(d_in, **module['phi'])
    ax.plot(d_in.T, d_out.T)

    ax_remove_box(ax)

    return ax


def plot_scatter(sig1, sig2, ax=None, title=None, smoothing_bins=False,
                 channels=0, xlabel=None, ylabel=None, legend=True, text=None,
                 force_square=False, module=None, **options):
    '''
    Uses the channels of sig1 to place points along the x axis, and channels of
    sig2 for distances along the y axis. If sig1 has one channel but sig2 has
    multiple channels, then all of sig2's channels will be plotted against the
    values from sig1. If sig1 has more than 1 channel, then sig2 must have the
    same number of channels, because XY coordinates will be determined from
    the same channel of both sig1 and sig2.

    Optional arguments:
        ax
        smoothing_bins: int
        xlabel
        ylabel
        legend
        module - NEMS module that applies an input-output transformation
          on data plotted from x to y axes. overlay data with curve from the
          module
    '''
    if 'scatter_color' not in options:
        options['scatter_color'] = 'darkgray'

    if (sig1.nchans > 1) or (sig2.nchans > 1):
        log.warning('sig1 or sig2 chancount > 1, using chan 0')
    if ax:
        #plt.sca(ax)
        pass
    ax = plt.gca()

    m1 = sig1.as_continuous()
    m2 = sig2.as_continuous()

    # remove NaNs
    keepidx = np.isfinite(m1[0, :]) * np.isfinite(m2[0, :])
    m1 = m1[channels:(channels+1), keepidx]
    m2 = m2[channels:(channels+1), keepidx]


    for i in range(m2.shape[0]):
        if m1.shape[0] > 1:
            x = m1[[i], :]
        else:
            x = m1[[0], :]
        y = m2[[i], :]

        if smoothing_bins:

            # Concatenate and sort
            s2 = np.append(x, y, 0)
            s2 = s2[:, s2[0, :].argsort()]
            # ????
            bincount = np.min([smoothing_bins, s2.shape[1]])
            T = np.int(np.floor(s2.shape[1] / bincount))
            x0 = np.zeros(bincount)
            y0 = np.zeros(bincount)
            minx=np.min(x)
            stepsize = (np.max(x)-minx)/bincount
            for bb in range(bincount):
                kk = (x>=minx+bb*stepsize) & (x<minx+(bb+1)*stepsize)
                if np.sum(kk):
                    x0[bb]=np.mean(x[kk])
                    y0[bb]=np.mean(y[kk])
            kk = (np.abs(x0)>0) & (np.abs(y0)>0)
            x=x0
            y=y0
#            s2 = s2[:, 0:(T * bincount)]
#            s2 = np.reshape(s2, [2, bincount, T])
#            s2 = np.mean(s2, 2)
#            s2 = np.squeeze(s2)
#            x = s2[0, :]
#            y = s2[1, :]

        chan_name = 'Channel {}'.format(i) if not sig2.chans else sig2.chans[i]
        ax.scatter(x, y, label=chan_name, s=2, color=options['scatter_color'])

    if module is not None:
        xbounds = ax.get_xbound()
        plot_nl_io(module, xbounds, ax)

    xlabel = xlabel if xlabel else sig1.name
    ax.set_xlabel(xlabel)

    ylabel = ylabel if ylabel else sig2.name
    ax.set_ylabel(ylabel)

    if legend and sig2.nchans > 1:
        ax.legend(loc='lower right')

    if title:
        ax.set_title(title)

    if text is not None:
        # Figure out where to align text box
        axes = ax
        ymin, ymax = axes.get_ylim()
        xmin, xmax = axes.get_xlim()
        if ymin == ymax:
            ymax = ymin + 1
        if xmin == xmax:
            xmax = xmin + 1
        x_coord = xmin + (xmax - xmin)/50
        y_coord = ymax - (ymax - ymin)/20
        ax.text(x_coord, y_coord, text, verticalalignment='top')

    if force_square:
        axes = ax
        ymin, ymax = axes.get_ylim()
        xmin, xmax = axes.get_xlim()
        axes.set_aspect(abs(xmax-xmin)/abs(ymax-ymin))

    ax_remove_box(ax)

    return ax

@cursor
def nl_scatter(rec, modelspec, idx, sig_name='pred',
               compare='resp', smoothing_bins=False, cursor_time=None,
               xlabel1=None, ylabel1=None, **options):

    # HACK: shouldn't hardcode 'stim', might be named something else
    #       or not present at all. Need to figure out a better solution
    #       for special case of idx = 0

    if 'mask' in rec.signals.keys():
        before = rec.apply_mask()
    else:
        before = rec.copy()
    if idx == 0:
        # Can't have anything before index 0, so use input stimulus
        sig_name='stim'
        before_sig = before['stim']
        before.name = '**stim'
    else:
        before = ms.evaluate(before, modelspec, start=None, stop=idx)
        before_sig = before[sig_name]

    # compute correlation for pre-module before it's over-written
    if before[sig_name].shape[0] == 1:
        corr1 = nm.corrcoef(before, pred_name=sig_name, resp_name=compare)
    else:
        corr1 = 0
        log.warning('corr coef expects single-dim predictions')

    compare_to = before[compare]

    module = modelspec[idx]
    mod_name = module['fn'].replace('nems0.modules.', '').replace('.', ' ').replace('_', ' ').title()

    title1 = mod_name
    text1 = "r = {0:.5f}".format(np.mean(corr1))

    ax = plot_scatter(before_sig, compare_to, title=title1,
                 smoothing_bins=smoothing_bins, xlabel=xlabel1,
                 ylabel=ylabel1, text=text1, module=module,
                 **options)

    if cursor_time is not None:
        tbin = int(cursor_time * rec[sig_name].fs)
        x = before_sig.as_continuous()[0,tbin]
        ylim=ax.get_ylim()
        ax.plot([x,x],ylim,'r-')
