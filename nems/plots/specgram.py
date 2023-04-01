import logging

import matplotlib.pyplot as plt
import numpy as np

from nems0.gui.decorators import scrollable
from nems0.plots.utils import ax_remove_box

log = logging.getLogger(__name__)


def plot_spectrogram(array, fs=None, ax=None, title=None, time_offset=0,
                     cmap=None, clim=None, extent=True, time_range=None, **options):


    if not ax:
        ax = plt.gca()

    if time_range is not None:
        if fs is not None:
            time_range = np.round(np.array(time_range)*fs).astype(int)
        log.debug('bin range: {}-{}'.format(time_range[0],time_range[1]))
        ax.imshow(array[:, np.arange(time_range[0],time_range[1])],
                  origin='lower', interpolation='none',
                  aspect='auto', cmap=cmap, clim=clim)
    elif extent:
        if fs is None:
            times = np.arange(0, array.shape[1])
        else:
            times = np.arange(0, array.shape[1])/fs-time_offset

        extent = [times[0], times[-1], 1, array.shape[0]]
        if extent[2]==extent[3]:
            extent[3]=2

        ax.imshow(array, origin='lower', interpolation='none',
                  aspect='auto', extent=extent, cmap=cmap, clim=clim)
    else:
        # maybe something had a bug and couldn't plot in seconds?
        ax.imshow(array, origin='lower', interpolation='none',
                  aspect='auto', cmap=cmap, clim=clim)

    ax.margins(x=0)

    # Override x-tic labels to display as real time
    # instead of time bin indices.
    #if fs is not None:
    #    locs = ax.get_xticks()
    #    newlabels = ["{:0.3f}".format(l/fs-time_offset) for l in locs]
    #    ax.set_xticklabels(newlabels)

    # TODO: Is there a way the colorbar can overlap part of the image
    # rather than shift it over?
    # cbar = plt.colorbar(fraction=0.05)
    # cbar.set_label('Amplitude')
    ax.set_xlabel('Time')
    ax.set_ylabel('Channel')
    if title:
        ax.set_title(title)

    ax_remove_box(ax)
    return ax

def spectrogram_from_signal(signal, title=None, ax=None):
    # TODO: How can the colorbar be scaled to match other signals?
    array = signal.as_continuous()
    plot_spectrogram(array, fs=signal.fs, title=title, ax=None)


def spectrogram_from_epoch(signal, epoch, occurrence=0, ax=None, **options):
    if occurrence is None:
        return
    if epoch is None:
        # valid epoch not specified, use whole signal
        array = signal.as_continuous()
    else:
        extracted = signal.extract_epoch(epoch)
        array = extracted[occurrence]
    plot_spectrogram(array, fs=signal.fs, ax=ax, **options)


@scrollable
def spectrogram(rec, sig_name='stim', ax=None, title=None, **options):
    """
    plot a spectrogram of an entire signal (typically stim), **options passed through
    :param rec:
    :param sig_name:
    :param ax:
    :param title:
    :param time_range: if not None, plot time_range[0]:time_range[1] (seconds) of the signal
    :param options: extra dict passed through to plot_spectrogram
    :return:

    TODO: How can the colorbar be scaled to match other signals?
    """
    if 'mask' in rec.signals.keys():
        signal = rec.apply_mask()[sig_name]
    else:
        signal = rec[sig_name]

    array = signal.as_continuous()

    #if time_range is not None:
    #    array = array[:, np.arange(time_range[0],time_range[1])]

    ax = plot_spectrogram(array, fs=signal.fs, title=title, ax=ax, **options)

    return ax


@scrollable
def pred_spectrogram(sig_name='pred', **options):
    """
    wrapper for spectrogram, forces stim_name to be pred. other **options passed through
    :param sig_name:
    :param options: passed through to spectrogram
    :return:
    """
    ax = spectrogram(sig_name=sig_name, **options)

    return ax


@scrollable
def resp_spectrogram(sig_name='resp', **options):
    """
    wrapper for spectrogram, forces stim_name to be resp. other **options passed through
    :param sig_name:
    :param options: passed through to spectrogram
    :return:
    """
    ax = spectrogram(sig_name=sig_name, **options)

    return ax


@scrollable
def spectrogram_output(rec, modelspec, sig_name='pred', idx=0, ax=None, **options):
    '''
    Wrapper for spectrogram, displays signal output by set in modelspec.
    other **options passed through

    Arguments:
    ----------
    rec : recording object
        The dataset to use. See nems/recording.py.

    modelspec : list of dicts
        The transformations to perform. See nems/modelspec.py.

    sig_name : str
        Specifies the signal in 'rec' to be plotted [pred].

    idx : int
        An index into the modelspec. rec[sig_name] will be plotted
        as it exists after step idx-1 and before step idx.

    Returns:
    --------
    ax : axis containing plot
    '''

    trec = modelspec.evaluate(rec, stop=idx+1)

    if 'mask' in rec.signals.keys():
        signal = trec.apply_mask()[sig_name]
    else:
        signal = trec[sig_name]

    array = signal.as_continuous()
    ax = plot_spectrogram(array, ax=ax, fs=signal.fs, **options)

    return ax
