import numpy as np
import matplotlib.pyplot as plt

from nems0.plots.assemble import pad_to_signals
from nems0.plots.timeseries import plot_timeseries

import nems0.modelspec as ms
import nems0.signal as signal
import nems0.recording as recording

def raster(times, values, xlabel='Time', ylabel='Trial', legend=None,
           linestyle='-', linewidth=1,
           ax=None, title=None):
    '''
    Plots a raster with one line for each pair of
    time and value vectors.
    Lines will be auto-colored according to matplotlib defaults.

    times : list of vectors
    values : list of vectors
    xlabel : str
    ylabel : str
    legend : list of strings
    linestyle, linewidth : pass-through options to plt.plot()

    TODO: expand this doc  -jacob 2-17-18
    '''

    if ax is not None:
        plt.sca(ax)

    x = values.copy()
    x = x[np.isfinite(x[:, 0]), :]

    i, j = np.where(x)
    i += 1
    if times is not None:
        t = times[j]
    else:
        t = j

    plt.plot(t, i, 'k.', markersize=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if times is not None:
        plt.xlim(np.min(times), np.max(times))
    plt.ylim(0, x.shape[0]+1)

    if title:
        plt.title(title)


def psth_from_raster(times, values, xlabel='Time', ylabel='Value',
                     legend=None, linestyle='-', linewidth=1,
                     ax=None, title=None, facecolor='lightblue',
                     binsize=1):

    if binsize > 1:
        x = np.reshape(values, [values.shape[0], -1, binsize])
        x = np.nanmean(x, axis=2)
        t = times[int(binsize/2)::binsize]
    else:
        x = values
        t = times

    m = np.nanmean(x, axis=0)
    e = np.nanstd(x, axis=0) / np.sqrt(np.sum(np.isfinite(x[:, 0])))

    if ax is not None:
        plt.sca(ax)

    plt.fill_between(t, m-e, m+e, facecolor=facecolor)

    plot_timeseries([t], [m], xlabel=xlabel, ylabel=ylabel,
                    legend=legend, linestyle=linestyle,
                    linewidth=linewidth, ax=ax, title=title)


def raster_from_vectors(vectors, xlabel='Time', ylabel='Value', fs=None,
                        linestyle='-', linewidth=1, legend=None,
                        ax=None, title=None):
    """TODO: doc"""
    raise NotImplementedError
    times = []
    values = []
    for v in vectors:
        values.append(v)
        if fs is None:
            times.append(np.arange(0, len(v)))
        else:
            times.append(np.arange(0, len(v))/fs)
    plot_timeseries(times, values, xlabel, ylabel, legend=legend,
                    linestyle=linestyle, linewidth=linewidth,
                    ax=ax, title=title)


def raster_from_signals(signals, channels=0, xlabel='Time', ylabel='Value',
                            linestyle='-', linewidth=1,
                            ax=None, title=None):
    """TODO: doc"""
    raise NotImplementedError
    channels = pad_to_signals(signals, channels)

    times = []
    values = []
    legend = []
    for s, c in zip(signals, channels):
        # Get values from specified channel
        value_vector = s.as_continuous()[c]
        # Convert indices to absolute time based on sampling frequency
        time_vector = np.arange(0, len(value_vector)) / s.fs
        times.append(time_vector)
        values.append(value_vector)
        legend.append(s.name+' '+s.chans[c])

    plot_timeseries(times, values, xlabel, ylabel, legend=legend,
                    linestyle=linestyle, linewidth=linewidth,
                    ax=ax, title=title)


def raster_from_epoch(signals, epoch, occurrences=0, channels=0,
                          xlabel='Time', ylabel='Value',
                          linestyle='-', linewidth=1,
                          ax=None, title=None):
    """TODO: doc"""
    raise NotImplementedError
    if occurrences is None:
        return
    occurrences = pad_to_signals(signals, occurrences)
    channels = pad_to_signals(signals, channels)

    legend = [s.name for s in signals]
    times = []
    values = []
    for s, o, c in zip(signals, occurrences, channels):
        # Get occurrences x chans x time
        extracted = s.extract_epoch(epoch)
        # Get values from specified occurrence and channel
        value_vector = extracted[o][c]
        # Convert bins to time (relative to start of epoch)
        # TODO: want this to be absolute time relative to start of data?
        time_vector = np.arange(0, len(value_vector)) / s.fs
        times.append(time_vector)
        values.append(value_vector)
    plot_timeseries(times, values, xlabel, ylabel, legend=legend,
                    linestyle=linestyle, linewidth=linewidth,
                    ax=ax, title=title)


