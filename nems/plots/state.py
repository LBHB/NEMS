import matplotlib.pyplot as plt
import numpy as np
import scipy
import logging

from .timeseries import (timeseries_from_signals, timeseries_from_vectors,
                         ax_remove_box)
import nems0.modelspec as ms
from nems0.utils import get_channel_number, find_module
from nems0.metrics.state import state_mod_split
from nems0.plots.utils import ax_remove_box
from nems0.gui.decorators import scrollable 
from .file import save_figure, load_figure_img, load_figure_bytes, fig2BytesIO

log = logging.getLogger(__name__)

line_colors = {'actual_psth': (0,0,0),
               'predicted_psth': 'red',
               #'passive': (255/255, 133/255, 133/255),
               'passive': (216/255, 151/255, 212/255),
               #'active': (196/255, 33/255, 43/255),
               'active': (129/255, 64/255, 138/255),
               'false_alarm': (79/255, 114/255, 184/255),
               'miss': (183/255, 196/255, 229/255),
               'hit': (36/255, 49/255, 103/255),
               'pre': 'green',
               'post': (123/255, 104/255, 238/255),
               'pas1': 'green',
               'pas2': (153/255, 124/255, 248/255),
               'pas3': (173/255, 144/255, 255/255),
               'pas4': (193/255, 164/255, 255/255),
               'pas5': 'green',
               'pas6': (123/255, 104/255, 238/255),
               'hard': (196/255, 149/255, 44/255),
               'easy': (255/255, 206/255, 6/255),
               'puretone': (247/255, 223/255, 164/255),
               'large': (44/255, 125/255, 61/255),
               'small': (181/255, 211/255, 166/255)}
fill_colors = {'actual_psth': (.8,.8,.8),
               'predicted_psth': 'pink',
               #'passive': (226/255, 172/255, 185/255),
               'passive': (234/255, 176/255, 223/255),
               #'active': (244/255, 44/255, 63/255),
               'active': (163/255, 102/255, 163/255),
               'false_alarm': (107/255, 147/255, 204/255),
               'miss': (200/255, 214/255, 237/255),
               'hit': (78/255, 92/255, 135/255),
               'pre': 'green',
               'post': (123/255, 104/255, 238/255),
               'hard':  (229/255, 172/255, 57/255),
               'easy': (255/255, 225/255, 100/255),
               'puretone': (255/255, 231/255, 179/255),
               'large': (69/255, 191/255, 89/255),
               'small': (215/255, 242/255, 199/255)}

@scrollable
def state_vars_timeseries(rec, modelspec, ax=None, state_colors=None,
                          decimate_by=1, channel=None, **options):

    if ax is None:
        ax = plt.gca()

    rec = rec.apply_mask()
    pred = rec['pred']
    resp = rec['resp']
    fs = resp.fs

    chanidx = get_channel_number(resp, channel)

    r1 = resp.as_continuous()[chanidx, :].T * fs
    p1 = pred.as_continuous()[chanidx, :].T * fs
    nnidx = np.isfinite(p1)
    r1 = r1[nnidx]
    p1 = p1[nnidx]

    if decimate_by > 1:
        r1 = scipy.signal.decimate(r1, q=decimate_by, axis=0)
        p1 = scipy.signal.decimate(p1, q=decimate_by, axis=0)
        fs /= decimate_by

    t = np.arange(len(r1)) / fs

    ax.plot(t, r1, linewidth=1, color='gray')
    ax.plot(t, p1, linewidth=1, color='black')
    #print(p1.shape)
    mmax = np.nanmax(p1) * 0.8

    if 'state' in rec.signals.keys():
        s = None
        g = None
        d = None
        sp = None
        for m in modelspec:
            if ('state_dc_gain' in m['fn']) and ('g' in m['phi'].keys()):
                g = np.array(m['phi']['g'])
                d = np.array(m['phi']['d'])
            elif ('state_dc_gain' in m['fn']):
                d = np.array(m['phi']['d'])
            elif ('state_sp_dc_gain' in m['fn']):
                g = np.array(m['phi']['g'])
                d = np.array(m['phi']['d'])
                sp = np.array(m['phi']['sp'])
            elif ('state_gain' in m['fn']):
                g = np.array(m['phi']['g'])

        if g is not None:
            if len(g) < 10:
                s = ",".join(rec["state"].chans)
                g_string = np.array2string(g, precision=3)
                s += " g={} ".format(g_string)
                if d is not None:
                    d_string = np.array2string(d, precision=3)
                    s += "d={} ".format(d_string)
            else:
                s = None

        num_vars = rec['state'].shape[0]
        ts = rec['state'].as_continuous().copy()

        if num_vars > 6:
            ts = scipy.signal.decimate(ts, q=10, axis=1)
            ts = ts / np.nanmax(ts, axis=1, keepdims=True)
            ax.imshow(ts, extent=(0, t[-1], -100, 0))
        else:
            if state_colors is None:
                state_colors = [None] * num_vars
            offset = -1.25 * mmax
            for i in range(1, num_vars):

                st = ts[i, :].T
                if (len(np.unique(st)) == 2) and num_vars > 3:
                    # special, binary variable, keep in one row
                    m = np.array([np.min(st)])
                    st = np.concatenate((m, st, m))
                    dinc = np.argwhere(np.diff(st) > 0)
                    ddec = np.argwhere(np.diff(st) < 0)
                    for x0, x1 in zip(dinc, ddec):
                        ax.plot([x0/fs, x1/fs], [offset, offset],
                                 lw=2, color=state_colors[i-1])
                    tstr = "{}".format(rec['state'].chans[i])
                    ax.text(x0/fs, offset, tstr, fontsize=6)
                    #print("{} {} {}".format(rec['state'].chans[i], x0/fs, offset))
                else:
                    # non-binary variable, plot in own row
                    # figure out gain
                    tstr = "{} (".format(rec['state'].chans[i])
                    if sp is not None:
                        if g.ndim == 1:
                            tstr += "sp={:.3f},d={:.3f},g={:.3f})".format(sp[i], d[i], g[i])
                        else:
                            tstr += "sp={:.3f},d={:.3f},g={:.3f})".format(sp[0, i], d[0, i], g[0, i])
                    elif g is not None:
                        if d is not None:
                            if d.ndim == 1:
                                tstr += "d={:.3f},".format(d[i])
                            else:
                                tstr += "d={:.3f},".format(d[0, i])
                        if g.ndim == 1:
                            tstr += "g={:.3f})".format(g[i])
                        else:
                            tstr += "g={:.3f})".format(g[0, i])

                    if decimate_by > 1:
                        st = scipy.signal.decimate(st[nnidx], q=decimate_by, axis=0)
                    else:
                        st = st[nnidx]

                    st = st / np.nanmax(st) * mmax + offset
                    ax.plot(t, st, linewidth=1, color=state_colors[i-1])
                    ax.text(t[0], offset, tstr, fontsize=6)

                    offset -= 1.25*mmax

        # ax.text(0.5, 0.9, s, transform=ax.transAxes,
        #         horizontalalignment='center')
        # if s:
        #    ax.title(s, fontsize=8)
    ax.set_xlabel('time (s)')
    ax.axis('tight')

    ax_remove_box(ax)


def state_var_psth(rec, psth_name='resp', var_name='pupil', ax=None,
                   channel=None):

    chanidx = get_channel_number(rec[psth_name], channel)

    psth = rec[psth_name]._data[:, chanidx, :]
    fs = rec[psth_name].fs
    var = rec['state'].loc[var_name]._data
    mean = np.nanmean(var)
    low = psth[var < mean]
    high = psth[var >= mean]
    timeseries_from_vectors([low, high], fs=fs, title=var_name, ax=ax)


def state_var_psth_from_epoch(rec, epoch="REFERENCE", psth_name='resp', psth_name2='pred',
                              state_sig='state_raw', state_chan='pupil', ax=None,
                              colors=None, channel=None, decimate_by=1, **options):
    """
    Plot PSTH averaged across all occurences of epoch, grouped by
    above- and below-average values of a state signal (state_sig)
    """

    # TODO: Does using epochs make sense for these?

    if channel is None:
        channel=0
    fs = rec[psth_name].fs

    d = rec[psth_name].get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d)) - 0.5/fs
    d = rec[psth_name].get_epoch_bounds('PostStimSilence')
    if d.size > 0:
        PostStimSilence = np.min(np.diff(d)) - 0.5/fs
        dd = np.diff(d)
        dd = dd[dd > 0]
    else:
        dd = np.array([])
    if dd.size > 0:
        PostStimSilence = np.min(dd) - 0.5/fs
    else:
        PostStimSilence = 0

    low, high = state_mod_split(rec, epoch=epoch, psth_name=psth_name,
                                channel=channel, state_sig=state_sig,
                                state_chan=state_chan)
    if psth_name2 is not None:
        low2, high2 = state_mod_split(rec, epoch=epoch, psth_name=psth_name2,
                                      channel=channel, state_sig=state_sig,
                                      state_chan=state_chan)

    if decimate_by > 1:
        low = scipy.signal.decimate(low, q=decimate_by, axis=1)
        high = scipy.signal.decimate(high, q=decimate_by, axis=1)
        if psth_name2 is not None:
            low2 = scipy.signal.decimate(low2, q=decimate_by, axis=1)
            high2 = scipy.signal.decimate(high2, q=decimate_by, axis=1)
        fs /= decimate_by

    title = state_chan
    if state_chan == 'baseline':
        legend = None
    else:
        legend = ('Lo', 'Hi')

    timeseries_from_vectors([low, high], fs=fs, title=title, ax=ax,
                            legend=legend, time_offset=PreStimSilence,
                            colors=colors, ylabel="sp/sec")

    if psth_name2 is not None:
        timeseries_from_vectors([low2, high2], fs=fs, title=title, ax=ax,
                                linestyle='--', time_offset=PreStimSilence,
                                colors=colors, ylabel="sp/sec")

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot(np.array([0, 0]), ylim, 'k--')
    ax.plot(np.array([xlim[1], xlim[1]])-PostStimSilence, ylim, 'k--')

    if state_chan == 'baseline':
        ax.set_xlabel(epoch)


def state_vars_psth_all(rec, epoch="REFERENCE", psth_name='resp', psth_name2='pred',
                        state_sig='state_raw', ax=None,
                        colors=None, channel=None, decimate_by=1,
                        files_only=False, modelspec=None, max_states=100, **options):

    # TODO: Does using epochs make sense for these?
    if ax is None:
        ax = plt.gca()

    if channel is None:
        channel = 0
    if epoch is None:
        epoch="REFERENCE"

    newrec = rec.copy()
    fn = lambda x: x - newrec['pred']._data
    newrec['error'] = rec['resp'].transform(fn, 'error')

    fs = rec[psth_name].fs

    d = rec[psth_name].get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d)) - 0.5/fs
    d = rec[psth_name].get_epoch_bounds('PostStimSilence')
    if d.size > 0:
        PostStimSilence = np.min(np.diff(d)) - 0.5/fs
        dd = np.diff(d)
        dd = dd[dd > 0]
    else:
        dd = np.array([])
    if dd.size > 0:
        PostStimSilence = np.min(dd) - 0.5/fs
    else:
        PostStimSilence = 0

    state_chan_list = rec[state_sig].chans
    low = np.zeros([0,1])
    high = np.zeros([0,1])
    lowE = np.zeros([0,1])
    highE = np.zeros([0,1])
    low2 = np.zeros([0,1])
    high2 = np.zeros([0,1])
    _high2 = None
    limitset = []
    if files_only: #state_chan_list =['a','p','PASSIVE_1']
        #print(state_chan_list)
        state_chan_list = [s for s in state_chan_list
                           if (s.startswith('FILE') | s.startswith('ACTIVE') |
                               s.startswith('PASSIVE')) ]

    def nan_sem(x, axis=0):
        y = np.nanstd(x, axis=axis) / np.sqrt(np.sum(np.isfinite(x), axis=axis))
        return y

    for state_chan in state_chan_list[:(max_states-1)]:

        _low, _high = state_mod_split(rec, epoch=epoch, psth_name=psth_name,
                                    channel=channel, state_sig=state_sig,
                                    state_chan=state_chan)
        _lowE, _highE = state_mod_split(newrec, epoch=epoch, psth_name='error',
                                    channel=channel, state_sig=state_sig,
                                    state_chan=state_chan, stat=nan_sem)
        if psth_name2 is not None:
            _low2, _high2 = state_mod_split(rec, epoch=epoch, psth_name=psth_name2,
                                            channel=channel, state_sig=state_sig,
                                            state_chan=state_chan)

        gapdur = _low.shape[0]/fs/10
        gap = np.ones([int(np.ceil(fs*gapdur)),1]) * np.nan
        pgap = np.ones(_low.shape) * np.nan
        if files_only:
            if state_chan == state_chan_list[0]:
                low = np.concatenate((high,gap,_low,gap), axis=0)
                lowE = np.concatenate((highE,gap,_lowE,gap), axis=0)
                high = np.concatenate((high,gap,pgap,gap), axis=0)
                highE = np.concatenate((highE,gap,pgap,gap), axis=0)
                if psth_name2 is not None:
                    low2 = np.concatenate((low2,gap,_low2,gap), axis=0)
                    high2 = np.concatenate((high2,gap,pgap,gap), axis=0)
            current_start = high.shape[0]/fs + gapdur
            if state_chan.startswith('ACTIVE'):
                _low = pgap
                _low2 = pgap
            else:
                _low = _high.copy()
                _low2 = _high2.copy()
                _high = pgap
                _high2 = pgap

        else:
            current_start = low.shape[0]/fs + gapdur

        low = np.concatenate((low,gap,_low,gap), axis=0)
        high = np.concatenate((high,gap,_high,gap), axis=0)
        lowE = np.concatenate((lowE,gap,_lowE,gap), axis=0)
        highE = np.concatenate((highE,gap,_highE,gap), axis=0)

        if psth_name2 is not None:
            low2 = np.concatenate((low2,gap,_low2,gap), axis=0)
            high2 = np.concatenate((high2,gap,_high2,gap), axis=0)

        limitset += [[current_start + PreStimSilence,
                     current_start + _low.shape[0]/fs - PostStimSilence]]

    if decimate_by > 1:
        low = scipy.signal.decimate(low, q=decimate_by, axis=1)
        high = scipy.signal.decimate(high, q=decimate_by, axis=1)
        if psth_name2 is not None:
            low2 = scipy.signal.decimate(low2, q=decimate_by, axis=1)
            high2 = scipy.signal.decimate(high2, q=decimate_by, axis=1)
        fs /= decimate_by

    tt = np.arange(high.shape[0])/fs
    ax.fill_between(tt, low[:,0]-lowE[:,0], low[:,0]+lowE[:,0], color=fill_colors['passive'])
    l1, = ax.plot(tt, low, ls='-', lw=1, color=line_colors['passive'])
    ax.fill_between(tt, high[:,0]-highE[:,0], high[:,0]+highE[:,0], color=fill_colors['active'])
    l2, = ax.plot(tt, high, ls='-', lw=1, color=line_colors['active'])
    if psth_name2 is not None:
        ax.plot(tt, low2, ls='--', lw=1, color=line_colors['passive'])
        ax.plot(tt, high2, ls='--', lw=1, color=line_colors['active'])

    if not files_only:
        ax.legend((l1,l2), ('Lo','Hi'))

    ax.set_ylabel('sp/sec')
    ylim = ax.get_ylim()

    for ls, s in zip(limitset, state_chan_list):
        try:
            sc = modelspec[0]['meta']['state_chans']
            mi = modelspec[0]['meta']['state_mod']
            sn = "{} ({:.2f})".format(s,mi[sc.index(s)])
        except:
            sn = s
        ax.plot(ls, [ylim[1], ylim[1]], 'k-', linewidth=2)
        lc = np.mean(ls)
        ax.text(lc, ylim[1], sn, ha='center', va='bottom', fontsize=6)
    ax.set_ylim([ylim[0], ylim[1]*1.1])
    ax_remove_box(ax)


def state_gain_plot(modelspec, rec=None, idx=None, state_sig='state_raw', ax=None, colors=None, clim=None, title=None, **options):

    state_chan_list = rec[state_sig].chans
    state_idx = find_module('state', modelspec)
    g = modelspec.phi_mean[state_idx]['g'].copy()
    ge = modelspec.phi_sem[state_idx]['g']
    if modelspec[state_idx]['fn'] == 'nems0.modules.state.state_gain':
        d = None
        de = None
        gainoffset = modelspec[state_idx]['fn_kwargs']['gainoffset']
        g += modelspec[state_idx]['fn_kwargs']['gainoffset']
    else:
        d = modelspec.phi_mean[state_idx]['d']
        de = modelspec.phi_sem[state_idx]['d']

    MI = modelspec[0]['meta']['state_mod']
    state_chans = modelspec[0]['meta']['state_chans']
    if ax is None:
        ax = plt.gca()
    if g.shape[0] > 1:
        opt={}
        for cc in range(g.shape[1]):
            if colors is not None:
                opt = {'color': colors[cc]}
            if d is not None:
                ax.plot(d[:,cc],'--', **opt)
            ax.plot(g[:,cc], **opt)
    else:
        if d is not None:
            ax.errorbar(np.arange(len(d[0, :])), d[0, :], de[0, :], color='blue')
            dz = np.abs(d[0, :] / de[0, :])
        ax.errorbar(np.arange(len(g[0, :])), g[0, :], ge[0, :], color='red')
        gz = np.abs(g[0, :] / ge[0, :])
        for i in range(len(gz)):
            if gz[i] > 2:
                ax.text(i, g[0, i] + np.sign(g[0, i]) * ge[0, i], state_chans[i],
                        color='red', ha='center', fontsize=6)
            elif d is not None and dz[i] > 2:
                ax.text(i, d[0,i]+np.sign(d[0,i])*de[0,i], state_chans[i],
                        color='blue', ha='center', fontsize=6)

    #ax.plot(MI)
    #ax.xticks(np.arange(len(state_chans)), state_chans, fontsize=6)
    if d is None:
        ax.legend(state_chan_list, frameon=False)
        ax.set_ylabel('Gain')
        ax.plot(np.arange(len(state_chans)), gainoffset*np.ones(len(state_chans)), 'k--',
                linewidth=0.5)
    else:
        ax.legend(('baseline', 'gain'), frameon=False)
        ax.plot(np.arange(len(state_chans)),np.zeros(len(state_chans)),'k--',
                 linewidth=0.5)

    if title is None:
        if 'id' in modelspec[idx].keys():
            title = " " + modelspec[idx]['id']
    if title is not None:
        ax.text(ax.get_xlim()[0],ax.get_ylim()[1],title,
                va='top',ha='left')

    ax_remove_box(ax)


def state_gain_parameters(modelspec=None, idx=None, rec=None,
                          ax=None, title=None, colors=None,
                          **options):
    phi = modelspec.phi[idx]
    if rec is not None:
        if 'state' in rec.signals.keys():
            labels=rec['state'].chans
        else:
            labels=None
    if 'g' in phi.keys():
        x = phi['g']
    elif 'd' in phi.keys():
        x = phi['d']
    else:
        print(f'state_gain_parameters: nothing to plot for mod {idx}')
        return

    if ax is None:
        ax = plt.gca()

    ax.plot(x)
    if labels is not None:
        ax.legend(labels, frameon=False)
    if title is None:
        if 'id' in modelspec[idx].keys():
            title = " " + modelspec[idx]['id']

    if title is not None:
        ax.text(ax.get_xlim()[0],ax.get_ylim()[1],title,
                va='top',ha='left')

    return ax

def model_per_time(ctx, fit_idx=0):
    """
    state_colors : N x 2 list
       color spec for high/low lines in each of the N states
    """
    if type(ctx['val']) is list:
        rec = ctx['val'][0].apply_mask()
    else:
        rec = ctx['val'].apply_mask()

    modelspec = ctx['modelspec']
    modelspec.fit_idx = 0

    epoch="REFERENCE"
    rec = ms.evaluate(rec, modelspec)

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    state_vars_timeseries(rec, modelspec, ax=ax)

    ax = plt.subplot(2, 1, 2)
    state_vars_psth_all(rec, epoch, psth_name='resp',
                        psth_name2='pred', state_sig='state_raw',
                        colors=None, channel=None, decimate_by=1,
                        ax=ax, files_only=True, modelspec=modelspec)


def cc_comp(val, modelspec, ax=None, figures=None, IsReload=False,
            extra_epoch=None, saveto=None, **options):
    if IsReload:
        return {}
    
    ## display noise corr. matrices

    if extra_epoch is not None:
        #rec = val.copy()
        #rec = rec.and_mask(extra_epoch)
        rec = val.apply_mask()
        #large_idx=rec['mask_large'].as_continuous()[0,:].astype(bool)
        #small_idx=rec['mask_small'].as_continuous()[0,:].astype(bool)
        #mask = rec['mask'].as_continuous()[0,:].astype(bool)
        #large_idx *= mask
        #small_idx *= mask
        if type(extra_epoch) is list:
            for i, e in enumerate(extra_epoch):
                if i == 0:
                    large_idx=rec['mask_'+e+'_lg'].as_continuous()[0,:].astype(bool)
                    small_idx=rec['mask_'+e+'_sm'].as_continuous()[0,:].astype(bool)
                else:
                    li = rec['mask_'+e+'_lg'].as_continuous()[0,:].astype(bool)
                    si = rec['mask_'+e+'_sm'].as_continuous()[0,:].astype(bool)
                    large_idx += li
                    small_idx += si
        else:
            large_idx=rec['mask_'+extra_epoch+'_lg'].as_continuous()[0,:].astype(bool)
            small_idx=rec['mask_'+extra_epoch+'_sm'].as_continuous()[0,:].astype(bool)
        print(f"masked {extra_epoch} len from {val['mask'].as_continuous().sum()} to {large_idx.sum()+small_idx.sum()}")
    else:
        rec = val.apply_mask()
        large_idx=rec['mask_large'].as_continuous()[0,:].astype(bool)
        small_idx=rec['mask_small'].as_continuous()[0,:].astype(bool)

    if 'pred0' in rec.signals.keys():
        input_name = 'pred0'
    else:
        input_name = 'psth'

    pred0 = rec[input_name].as_continuous()
    pred = rec['pred'].as_continuous()
    resp = rec['resp'].as_continuous()
    state = rec['state'].as_continuous()

    siteid = modelspec.meta['cellid'].split("-")[0]
    large_cc = np.cov(resp[:, large_idx]-pred0[:, large_idx])
    small_cc = np.cov(resp[:, small_idx]-pred0[:, small_idx])
    mm=np.max(np.abs(small_cc)) * 0.5
    sm_cc = np.cov(pred[:, small_idx]-pred0[:, small_idx])
    lg_cc = np.cov(pred[:, large_idx]-pred0[:, large_idx])

    # large vs. small delta cc
    dact=large_cc-small_cc
    dpred=lg_cc-sm_cc

    # act and pred cc values for all pairs, averaged across states
    triu = np.triu_indices(dpred.shape[0], 1)
    cc_avg = (large_cc[triu] + small_cc[triu])/2
    cc_pred_avg = (lg_cc[triu] + sm_cc[triu])/2

    # distribution of delta cc
    d_each = dact[triu]
    d_each_pred = dpred[triu]

    f, ax = plt.subplots(3, 4, figsize=(10, 7.5))
    imopts = {'aspect': 'equal', 'interpolation': 'none', 'clim': [-mm,mm],
              'cmap': 'bwr', 'origin': 'lower'}

    ax[0, 0].imshow(small_cc, **imopts)
    ax[0, 0].set_title('Actual small pupil')
    ax[0, 0].set_xlabel(f"Unit (sorted by depth)")
    ax[0, 0].set_ylabel(f"Unit (sorted by depth)")
    ax[0, 1].imshow(large_cc, **imopts)
    ax[0, 1].set_title('Actual large pupil')
    ax[0, 1].set_xlabel(f"Unit (sorted by depth)")
    ax[0, 2].imshow(large_cc-small_cc, **imopts)
    ax[0, 2].set_title(f"Large - small")
    #ax[0, 2].set_title(f"lg-sm std={np.mean((large_cc-small_cc)**2):.3f}")
    ax[0, 2].set_xlabel(f"Unit (sorted by depth)")

    # histogram of cc
    h, b = np.histogram(cc_avg, bins=20, range=[-0.12, 0.12])
    hp, bp = np.histogram(cc_pred_avg, bins=20, range=[-0.12, 0.12])
    db = (b[1]-b[0])/2
    ax[0, 3].bar(b[1:]-db, h, width=b[1]-b[0], color='lightgray',
                 label=f'Act ({np.mean(cc_avg):.3f})')
    ax[0, 3].step(bp[1:],hp, label=f'Model ({np.mean(cc_pred_avg):.3f})', color='blue')
    ax[0, 3].axvline(0, color='black', linestyle='--', lw=1)
    ax[0, 3].set_title(f"Overall noise correlation")
    ax[0, 3].legend(frameon=False)
    ax[0, 3].set_ylabel(f"Number of neuron pairs")

    h, b = np.histogram(d_each, bins=20, range=[-0.08, 0.08])
    hp, bp = np.histogram(d_each_pred, bins=20, range=[-0.08, 0.08])
    db = (b[1]-b[0])/2
    ax[1, 3].bar(b[1:]-db, h, width=b[1] - b[0], color='lightgray',
                 label=f'Act ({np.mean(d_each):.3f})')
    ax[1, 3].step(bp[1:], hp, label=f'Model ({np.mean(d_each_pred):.3f})', color='blue')
    ax[1, 3].axvline(0, color='black', linestyle='--', lw=1)
    ax[1, 3].set_title(f"Noise correlation change (lg-sm pupil)")
    ax[1, 3].legend(frameon=False)
    ax[1, 3].set_ylabel(f"Number of neuron pairs")

    ax[1, 0].imshow(sm_cc, aspect='auto', interpolation='none', clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[1, 0].set_title('Model small pupil')
    ax[1, 0].set_xlabel(f"Unit (sorted by depth)")
    ax[1, 0].set_ylabel(f"Unit (sorted by depth)")
    ax[1, 1].imshow(lg_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[1, 1].set_title('Model large pupil')
    ax[1, 1].set_xlabel(f"Unit (sorted by depth)")
    ax[1, 2].imshow((lg_cc-sm_cc),aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[1, 2].set_title("Large - small")
    #ax[1, 2].set_title(f"std={np.mean((lg_cc-sm_cc)**2):.3f}")
    ax[1, 2].set_xlabel(f"Unit (sorted by depth)")

    ax[2, 0].plot(np.diag(dact), label='act')
    ax[2, 0].plot(np.diag(dpred), label='pred')
    ax[2, 0].set_title('mean lg-sm var')
    ax[2, 0].set_xlabel(f"Unit (sorted by depth)")
    ax[2, 0].legend(frameon=False)

    np.fill_diagonal(dact, 0)
    ax[2, 1].plot(dact.mean(axis=0), label='act')
    np.fill_diagonal(dpred, 0)
    ax[2, 1].plot(dpred.mean(axis=0), label='pred')
    ax[2, 1].set_title('mean lg-sm cc')
    ax[2, 1].set_xlabel(f"Unit (sorted by depth)")

    im = ax[2, 2].imshow((large_cc-small_cc) - (lg_cc-sm_cc),aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[2, 2].set_ylabel('d_sim-d_act')
    ax[2, 2].set_title(f"E={np.mean(((large_cc-small_cc) - (lg_cc-sm_cc))**2):.3f}");
    ax[2, 2].set_xlabel(f"Unit (sorted by depth)")
    plt.colorbar(im, ax=ax[2,2])

    lv_weights = modelspec.phi[2]['g']
    smweights = lv_weights[:,0]-lv_weights[:,1]+lv_weights[:,2]
    lgweights = lv_weights[:,0]+lv_weights[:,1]+lv_weights[:,2]
    if smweights.sum()<0:
        smweights *= -1
        lgweights *= -1
    #ax[2, 3].plot(smweights, color='lightgray', label='Small pup.')
    #ax[2, 3].plot(lgweights, color='blue', label='Large pup.')

    baseweights = lv_weights[:,0] * np.sign(np.sum(lv_weights[:,0]))
    pupweights = lv_weights[:,1] * -np.sign(np.sum(lv_weights[:,1]))
    ax[2, 3].plot(baseweights, color='lightgray', label='Baseline')
    ax[2, 3].plot(pupweights, color='blue', label='Pupil mod.')
    ax[2, 3].axhline(0, color='black', linestyle='--', lw=1)
    ax[2, 3].legend(frameon=False)
    ax[2, 3].set_title('LV gain per neuron')
    ax[2, 3].set_xlabel(f"Unit (sorted by depth)")

    f.suptitle(f"{modelspec.meta['cellid']} - {modelspec.meta['modelname']}", fontsize=8)
    plt.tight_layout()

    cc_std_sm = np.std(small_cc[triu])
    cc_std_lg = np.std(large_cc[triu])
    state_std_sm = np.std(state[:,small_idx],axis=1)
    state_std_lg = np.std(state[:,large_idx],axis=1)

    modelspec.meta['cc_std_sm'] = cc_std_sm
    modelspec.meta['cc_std_lg'] = cc_std_lg
    modelspec.meta['state_std_sm'] = state_std_sm
    modelspec.meta['state_std_lg'] = state_std_lg
    modelspec.meta['med_d_each'] = np.median(d_each)
    modelspec.meta['E_dcc'] = np.mean(((large_cc-small_cc) - (lg_cc-sm_cc))**2)
    modelspec.meta['act_dcc_std'] = np.mean((large_cc-small_cc)**2)
    modelspec.meta['pred_dcc_std'] = np.mean((lg_cc-sm_cc)**2)

    if saveto is not None:
        log.info(f'Saving fig to {saveto}')
        f.savefig(saveto)

    # CANNOT initialize figures=[] in optional args our you will create a bug
    if figures is None:
        figures = []
    figures.append(fig2BytesIO(f))

    return {'figures': figures, 'modelspec': modelspec}


def cc_comp_per_mask(val=None, modelspec=None, IsReload=False, figures=None, **options):
    if IsReload:
        return {}

    ## display noise corr. matrices
    f,ax = plt.subplots(4,5, figsize=(12,9))

    rec = val.copy()

    masks = ["_".join(k.split("_")[:-1]) for k in rec.signals.keys()
             if (k.startswith("mask_") and k!="mask_small" and k!="mask_large")]
    masks = list(set(masks))

    if 'pred0' in rec.signals.keys():
        input_name = 'pred0'
    else:
        input_name = 'psth'
    pred0 = rec[input_name].as_continuous()
    pred = rec['pred'].as_continuous()
    resp = rec['resp'].as_continuous()

    for i, m in enumerate(masks[:4]):
        ml = rec[m+"_lg"].as_continuous()[0,:]
        ms = rec[m+"_sm"].as_continuous()[0,:]

        large_cc = np.cov(resp[:,ml]-pred0[:,ml])
        small_cc = np.cov(resp[:,ms]-pred0[:,ms])
        pred_large_cc = np.cov(pred[:,ml]-pred0[:,ml])
        pred_small_cc = np.cov(pred[:,ms]-pred0[:,ms])

        mm=np.max(np.abs(small_cc)) * 1

        ax[i,0].imshow(small_cc,aspect='equal',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
        ax[i,1].imshow(large_cc,aspect='equal',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
        ax[i,2].imshow(pred_small_cc,aspect='equal',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
        ax[i,3].imshow(pred_large_cc,aspect='equal',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')

        dact=large_cc-small_cc
        dpred=pred_large_cc-pred_small_cc

        np.fill_diagonal(dact, 0)
        ax[i,4].plot(dact.mean(axis=0), label='act')
        np.fill_diagonal(dpred, 0)
        ax[i,4].plot(dpred.mean(axis=0), label='pred')
        ax[i,4].set_title('mean lg-sm cc')
        ax[i,4].set_xlabel('unit')
        ax[i,4].legend(frameon=False)

        ax[i,0].set_ylabel(m.replace("mask_STIM",""),fontsize=6)
    ax[0,0].set_title('act sm cc')
    ax[0,1].set_title('act lg cc')
    ax[0,2].set_title('pred sm cc')
    ax[0,3].set_title('pred lg cc')

    f.suptitle(f"{modelspec.meta['cellid']} - {modelspec.meta['modelname']}", fontsize=8)

    # CANNOT initialize figures=[] in optional args our you will create a bug
    if figures is None:
        figures = []
    figures.append(fig2BytesIO(f))

    return {'figures': figures, 'modelspec': modelspec}

