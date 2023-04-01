import logging
import numpy as np
import matplotlib.pyplot as plt
import copy

from nems0.plots.assemble import pad_to_signals
import nems0.modelspec as ms
import nems0.signal as signal
import nems0.recording as recording
#import nems0.modules.stp as stp
from nems0.metrics.stp import stp_magnitude
from nems0.plots.utils import ax_remove_box
from nems0.gui.decorators import scrollable

log = logging.getLogger(__name__)

def plot_timeseries(times, values, xlabel='Time', ylabel='Value', legend=None,
                    linestyle='-', linewidth=1,
                    ax=None, title=None, colors=None, **options):
    '''
    Plots a simple timeseries with one line for each pair of
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
        pass
        #plt.sca(ax)
    else:
        ax = plt.gca()

    cc = 0
    opt = {}
    h=[]
    mintime = np.inf
    maxtime = 0
    for t, v in zip(times, values):
        if colors is not None:
            opt = {'color': colors[cc % len(colors)]} #Wraparound to avoid crash
        if v.ndim==1:
            v=v[:,np.newaxis]
        for idx in range(v.shape[1]):
            gidx = np.isfinite(v[:,idx])
            h_=ax.plot(t[gidx], v[gidx, idx], linestyle=linestyle,
                        linewidth=linewidth, **opt)
            h = h + h_
        cc += 1
        if gidx.sum() > 0:
            mintime = np.min((mintime, np.min(t[gidx])))
            maxtime = np.max((maxtime, np.max(t[gidx])))
    #ax.set_margins(x=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([mintime, maxtime])
    if legend:
        ax.legend(legend)
    if title:
        ax.set_title(title)

    ax_remove_box(ax)

    return ax, h


def timeseries_from_vectors(vectors, xlabel='Time', ylabel='Value', fs=None,
                            linestyle='-', linewidth=1, legend=None,
                            ax=None, title=None, time_offset=0,
                            colors=None, **options):
    """TODO: doc"""
    times = []
    values = []
    for v in vectors:
        values.append(v)
        if fs is None:
            times.append(np.arange(0, len(v)) - time_offset)
        else:
            times.append(np.arange(0, len(v))/fs - time_offset)
    ax = plot_timeseries(times, values, xlabel, ylabel,
                    legend=legend,
                    linestyle=linestyle, linewidth=linewidth,
                    ax=ax, title=title, colors=colors)
    return ax

@scrollable
def timeseries_from_signals(signals=None, channels=0, no_legend=False,
                            time_range=None, rec=None, sig_name=None, **options):
    """
    Plot one or more timeseries extracted from a list of signals

        :param signals: List of signals to plot
        :param channels: List of channels, one per signal(??)
        :param no_legend: True/False guess what this means?
        :param time_range: if not None, plot time_range[0]:time_range[1] (seconds) of the signal
        :return: Matplotlib axes containing the plot
    """
    if channels is None:
        channels = 0
    if signals is None:
        signals = [rec[sig_name]]
    if type(channels) is int and channels >= signals[0].shape[0]:
        channels=np.arange(signals[0].shape[0])
    channels = pad_to_signals(signals, channels)
    times = []
    values = []
    legend = []
    for i, s in enumerate(signals):
        if len(signals) > 1:
            chanset = [channels[i]]
        else:
            chanset = channels
        for c in chanset:
            if type(c) is str:
                c=0
            # Get values from specified channel
            value_vector = s.as_continuous()[c]
            # Convert indices to absolute time based on sampling frequency
            time_vector = np.arange(0, len(value_vector)) / s.fs
            times.append(time_vector)
            values.append(value_vector)
            if s.chans is not None:
                legend.append(s.name+' '+s.chans[c])

    if no_legend:
        legend = None

    if time_range is not None:
        time_range = np.round(np.array(time_range)*s.fs).astype(int)
        times = [t[np.arange(time_range[0], time_range[1])] for t in times]
        values = [v[np.arange(time_range[0], time_range[1])] for v in values]

    ax = plot_timeseries(times, values, legend=legend, **options)

    return ax


def timeseries_from_epoch(signals, epoch, occurrences=0, channels=0,
                          xlabel='Time', ylabel='Value',
                          linestyle='-', linewidth=1,
                          ax=None, title=None, pre_dur=None, dur=None,
                          PreStimSilence=None, **options):
    """TODO: doc"""
    if occurrences is None:
        return
    occurrences = pad_to_signals(signals, occurrences)
    channels = pad_to_signals(signals, channels)
    if PreStimSilence is None:
        d = signals[0].get_epoch_bounds('PreStimSilence')
        if len(d):
            PreStimSilence = np.mean(np.diff(d))
        else:
            PreStimSilence = 0
    if pre_dur is None:
        pre_dur = PreStimSilence

    legend = [s.name for s in signals]
    times = []
    values = []
    for s, o, c in zip(signals, occurrences, channels):
        if epoch is None:
            # just get the entire signal for this channel
            value_vector = s.as_continuous()[c]
        else:
            # Get occurrences x chans x time
            extracted = s.extract_epoch(epoch)
            # Get values from specified occurrence and channel
            value_vector = extracted[o][c]

        # Convert bins to time (relative to start of epoch)
        # TODO: want this to be absolute time relative to start of data?
        time_vector = np.arange(0, len(value_vector)) / s.fs - PreStimSilence

        # limit time range if specified
        good_bins = (time_vector >= -pre_dur)
        if dur is not None:
            good_bins[time_vector > dur] = False

        times.append(time_vector[good_bins])
        values.append(value_vector[good_bins])

    plot_timeseries(times, values, xlabel, ylabel, legend=legend,
                    linestyle=linestyle, linewidth=linewidth,
                    ax=ax, title=title)


def before_and_after_stp(modelspec=None, mod_index=None, tau=None, u=None, tau2=None, u2=None,
                         ax=None, title=None, colors=None,
                         channels=0, xlabel='Time', ylabel='Value', fs=100, **options):
    '''
    Plots a timeseries of specified signal just before and just after
    the transformation performed at some step in the modelspec.

    Arguments:
    ----------
    u, tau : np arrays of STP parameters. if not specified, use modelspec
    modelspec : modelspec with an STP module
    mod_index : index of STP module to plot (allows models with multiple STPs), d
                default=None, in which case use first STP module
    fs : sampling rate (default 100 Hz)

    Returns:
    --------
    None
    '''

    if (tau is None) or (u is None):
        if mod_index:
            m = modelspec[mod_index]
        else:
            for m in modelspec:
                if 'stp' in m['fn']:
                    break
        tau = m['phi']['tau']
        u = m['phi']['u']
        tau2 = m['phi'].get('tau2', None)
        u2 = m['phi'].get('u2', None)
        urat = m['phi'].get('urat', 0.5)
        quick_eval=m['fn_kwargs'].get('quick_eval',False)

    if type(u) in [float, int]:
        u=np.array([u])
    if type(tau) in [float, int]:
        tau=np.array([tau])
    if type(u2) in [float, int]:
        u2=np.array([u2])
    if type(tau2) in [float, int]:
        tau2=np.array([tau2])
    stp_mag, pred, pred_out = stp_magnitude(tau=tau, u=u, fs=fs, tau2=tau2, u2=u2, urat=urat,
                                            quick_eval=quick_eval)
    c = len(tau)
    pred.name = 'before'
    pred_out.name = 'after'
    signals = []
    channels = []
    for i in range(c):
        signals.append(pred_out)
        channels.append(i)
    signals.append(pred)
    channels.append(0)

    if colors is not None:
        pass
    elif c == 1:
        colors = [[0, 0, 0],
                  [128/255, 128/255, 128/255]]
    elif c == 2:
        colors = [[254 / 255, 15 / 255, 6 / 255],
                  [129 / 255, 201 / 255, 224 / 255],
                  [128/255, 128/255, 128/255]
                  ]
    elif c == 3:
        colors = [[254/255, 15/255, 6/255],
                  [217/255, 217/255, 217/255],
                  [129/255, 201/255, 224/255],
                  [128/255, 128/255, 128/255]
                  ]
    else:
        colors = None

    timeseries_from_signals(signals, channels=channels,
                            xlabel=xlabel, ylabel=ylabel, ax=ax,
                            title=title, colors=colors)


@scrollable
def before_and_after(rec, modelspec, sig_name, ax=None, title=None, idx=0,
                     channels=0, xlabel='Time', ylabel='Value', **options):
    '''
    Plots a timeseries of specified signal just before and just after
    the transformation performed at some step in the modelspec.

    Arguments:
    ----------
    rec : recording object
        The dataset to use. See nems/recording.py.

    modelspec : list of dicts
        The transformations to perform. See nems/modelspec.py.

    sig_name : str
        Specifies the signal in 'rec' to be examined.

    idx : int
        An index into the modelspec. rec[sig_name] will be plotted
        as it exists after step idx-1 and after step idx.

    Returns:
    --------
    None
    '''
    # HACK: shouldn't hardcode 'stim', might be named something else
    #       or not present at all. Need to figure out a better solution
    #       for special case of idx = 0
    if idx == 0:
        input_name = modelspec[0]['fn_kwargs']['i']
        before = rec[input_name].copy()
        before.name += ' before**'
    else:
        before = ms.evaluate(rec.copy(), modelspec, start=None, stop=idx)[sig_name]
        before.name += ' before'

    after = ms.evaluate(rec, modelspec, start=idx, stop=idx+1)[sig_name].copy()
    after.name += ' after'
    timeseries_from_signals([before, after], channels=channels,
                            xlabel=xlabel, ylabel=ylabel, ax=ax,
                            title=title, **options)

@scrollable
def mod_output(rec, modelspec, sig_name='pred', ax=None, title=None, idx=0,
               channels=0, xlabel='Time', ylabel='Value', **options):
    '''
    Plots a time series of specified signal output by step in the modelspec.

    Arguments:
    ----------
    rec : recording object
        The dataset to use. See nems/recording.py.

    modelspec : list of dicts
        The transformations to perform. See nems/modelspec.py.

    sig_name : str or list of strings
        Specifies the signal in 'rec' to be examined.

    idx : int
        An index into the modelspec. rec[sig_name] will be plotted
        as it exists after step idx-1 and after step idx.

    Returns:
    --------
    ax : axis containing plot
    '''
    if type(sig_name) is str:
        sig_name = [sig_name]

    trec = ms.evaluate(rec, modelspec, stop=idx+1)
    if 'mask' in trec.signals.keys():
        trec = trec.apply_mask()

    sigs = [trec[s] for s in sig_name]
    ax = timeseries_from_signals(sigs, channels=channels,
                                 xlabel=xlabel, ylabel=ylabel, ax=ax,
                                 title=title, **options)
    return ax

@scrollable
def mod_output_all(rec, modelspec, sig_name='pred', idx=0, **options):
    '''
    Plots a time series of specified signal output by step in the modelspec.

    Arguments:
    ----------
    rec : recording object
        The dataset to use. See nems/recording.py.

    modelspec : list of dicts
        The transformations to perform. See nems/modelspec.py.

    sig_name : str or list of strings
        Specifies the signal in 'rec' to be examined.

    idx : int
        An index into the modelspec. rec[sig_name] will be plotted
        as it exists after step idx-1 and after step idx.

    Returns:
    --------
    ax : axis containing plot
    '''

    trec = modelspec.evaluate(rec, stop=idx+1)
    if 'mask' in trec.signals.keys():
        trec = trec.apply_mask()

    options['channels']=np.arange(trec[sig_name].shape[0])
    ax = timeseries_from_signals([trec[sig_name]], **options)

    return ax


def before_and_after_signal(rec, modelspec, idx, sig_name='pred'):
    # HACK: shouldn't hardcode 'stim', might be named something else
    #       or not present at all. Need to figure out a better solution
    #       for special case of idx = 0
    if idx == 0:
        # Can't have anything before index 0, so use input stimulus
        before = rec
        before_sig = copy.deepcopy(rec['stim'])
    else:
        before = ms.evaluate(rec, modelspec, start=None, stop=idx)
        before_sig = copy.deepcopy(before[sig_name])

    before_sig.name = 'before'

    after = ms.evaluate(before.copy(), modelspec, start=idx, stop=idx+1)
    after_sig = copy.deepcopy(after[sig_name])
    after_sig.name = 'after'

    return before_sig, after_sig


@scrollable
def fir_output_all(rec, modelspec, sig_name='pred', idx=0, **options):
    """ plot output of fir filter channels separately"""
    # now evaluate next module step

    if 'fir' not in modelspec[idx]['fn']:
        raise ValueError("only works for fir modules")

    ms2 = copy.deepcopy(modelspec)
    ms2[idx]['fn'] = 'nems0.modules.fir.filter_bank'
    chan_count = ms2[idx]['phi']['coefficients'].shape[0]
    ms2[idx]['fn_kwargs']['bank_count'] = chan_count
    before2, after2 = before_and_after_signal(rec, ms2, idx, sig_name)
    value_vector = after2.as_continuous().T
    legend = ['ch'+str(c) for c in range(chan_count)]

    time_vector = np.arange(0, value_vector.shape[0]) / after2.fs

    ax, h = plot_timeseries([time_vector], [value_vector],
                            legend=legend, **options)
    return ax


@scrollable
def pred_resp(rec=None, modelspec=None, ax=None, title=None,
              channels=0, xlabel='Time', ylabel='Value',
              no_legend=False, **options):
    '''
    Plots a time series of prediction overlaid on response.

    Arguments:
    ----------
    rec : recording object
        The dataset to use. See nems/recording.py.

    modelspec : list of dicts
        The transformations to perform. See nems/modelspec.py.

    Returns:
    --------
    ax : axis containing plot
    '''
    sig_list = [modelspec.meta.get('output_name','resp'), 'pred']
    ylabel = f'{"/".join(sig_list)} chan {channels}'
    sigs = [rec[s] for s in sig_list]
    ax = timeseries_from_signals(sigs, channels=channels,
                                 xlabel=xlabel, ylabel=ylabel, ax=ax,
                                 title=title, no_legend=no_legend, **options)
    return ax
