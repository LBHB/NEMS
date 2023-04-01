import warnings
import copy
import logging
import logging

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from scipy.integrate import cumtrapz
import scipy.signal as ss

import nems0.epoch as ep
import nems0.signal
from nems0.recording import Recording
from nems0.utils import smooth

log = logging.getLogger(__name__)


def generate_average_sig(signal_to_average,
                         new_signalname='respavg', epoch_regex='^STIM_', mask=None):
    '''
    Returns a signal with a new signal created by replacing every epoch
    matched in "epoch_regex" with the average of every occurrence in that
    epoch. This is often used to make a response average signal that
    is the same length as the original signal_to_average, usually for plotting.

    Optional arguments:
       signal_to_average   The signal from which you want to create an
                           average signal. It will not be modified.
       new_signalname      The name of the new, average signal.
       epoch_regex         A regex to match which epochs to average across.
    '''

    # 1. Fold matrix over all stimuli, returning a dict where keys are stimuli
    #    and each value in the dictionary is (reps X cell X bins)
    epochs_to_extract = ep.epoch_names_matching(signal_to_average.epochs,
                                                epoch_regex)
    folded_matrices = signal_to_average.extract_epochs(epochs_to_extract, mask=mask)

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    for k in folded_matrices.keys():
        per_stim_psth[k] = np.nanmean(folded_matrices[k], axis=0)

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg = signal_to_average.replace_epochs(per_stim_psth)
    respavg.name = new_signalname

    return respavg


def add_average_sig(rec, signal_to_average='resp',
                    new_signalname='respavg', epoch_regex='^STIM_'):
    '''
    Returns a recording with a new signal created by replacing every epoch
    matched in "epoch_regex" with the average of every occurrence in that
    epoch. This is often used to make a response average signal that
    is the same length as the original signal_to_average, usually for plotting.

    Optional arguments:
       signal_to_average   The signal from which you want to create an
                           average signal. It will not be modified.
       new_signalname      The name of the new, average signal.
       epoch_regex         A regex to match which epochs to average across.
    '''

    # generate the new signal by averaging epochs of the input singals
    respavg = generate_average_sig(rec[signal_to_average],
                                   new_signalname, epoch_regex)

    # Add the signal to the recording
    newrec = rec.copy()
    newrec.add_signal(respavg)

    return newrec


def average_away_epoch_occurrences(recording, epoch_regex='^STIM_', use_mask=True):
    """
    Returns a recording with _all_ signals averaged across epochs that
    match epoch_regex, shortening them so that each epoch occurs only
    once in the new signals. i.e. unlike 'add_average_sig', the new
    recording will have signals 3x shorter if there are 3 occurrences of
    every epoch.

    This has advantages:
    1. Averaging the value of a signal (such as a response) in different
       occurrences will make it behave more like a linear variable with
       gaussian noise, which is advantageous in many circumstances.
    2. There will be less computation needed because the signal is shorter.

    It also has disadvantages:
    1. Stateful filters (FIR, IIR) will be subtly wrong near epoch boundaries
    2. Any ordering of epochs is essentially lost, unless all epochs appear
       in a perfectly repeated order.

    To avoid accidentally averaging away differences in responses to stimuli
    that are based on behavioral state, you may need to create new epochs
    (based on stimulus and behaviorial state, for example) and then match
    the epoch_regex to those.
    """
    if use_mask:
        recording = recording.remove_masked_epochs()

    # need to edit the epochs dataframe, so make a working copy
    temp_epochs = recording['resp'].epochs.copy()

    # only pull out matching epochs
    regex_mask = temp_epochs['name'].str.contains(pat=epoch_regex, na=False, regex=True)
    epoch_stims = temp_epochs[regex_mask]

    # get a list of the unique epoch names
    epoch_names = temp_epochs.loc[regex_mask, 'name'].sort_values().unique()

    #import pdb; pdb.set_trace()

    # what to round to when checking if epoch timings match
    d = int(np.ceil(np.log10(recording[list(recording.signals.keys())[0]].fs))+1)

    # need an end and start to close the bounds for cases where start and end bounds are identical
    s_name_start = pd.Series(epoch_stims['name'].values,
                             pd.IntervalIndex.from_arrays(epoch_stims['start'], epoch_stims['end'], closed='left'))
    s_name_end = pd.Series(epoch_stims['name'].values,
                           pd.IntervalIndex.from_arrays(epoch_stims['start'], epoch_stims['end'], closed='right'))
    s_cat_start = pd.Series(np.arange(len(epoch_stims['start']), dtype='int'),
                            pd.IntervalIndex.from_arrays(epoch_stims['start'], epoch_stims['end'], closed='left'))
    s_cat_end = pd.Series(np.arange(len(epoch_stims['end']), dtype='int'),
                          pd.IntervalIndex.from_arrays(epoch_stims['start'], epoch_stims['end'], closed='right'))

    # add helper columns using the interval index lookups
    temp_epochs['cat'] = temp_epochs['start'].map(s_cat_start)
    temp_epochs['cat_end'] = temp_epochs['end'].map(s_cat_end)
    temp_epochs['stim'] = temp_epochs['start'].map(s_name_start)
    temp_epochs['stim_end'] = temp_epochs['end'].map(s_name_end)

    # only want epochs that fall within a stim epoch, so drop the ones that don't
    drop_mask = temp_epochs['cat'] != temp_epochs['cat_end']
    trial_mask = temp_epochs['name'] == 'TRIAL'  # also dorp this
    temp_epochs = temp_epochs.loc[~drop_mask & ~trial_mask, ['name', 'start', 'end', 'cat', 'stim']]

    temp_epochs['cat'] = temp_epochs['cat'].astype(int)  # cast back to int to make into index

    # build another helper series, to map in times to subtract from start/end
    work_mask = temp_epochs['name'].str.contains(pat=epoch_regex, na=False, regex=True)
    s_starts = pd.Series(temp_epochs.loc[work_mask, 'start'].values, temp_epochs.loc[work_mask, 'cat'].values)
    
    temp_epochs['start'] -= temp_epochs['cat'].map(s_starts)
    temp_epochs['end'] -= temp_epochs['cat'].map(s_starts)
    temp_epochs = temp_epochs.round(d)
    
    expected_max = temp_epochs.loc[temp_epochs['name'].str.contains(pat=epoch_regex, na=False, regex=True),'end'].max()

    concat = []

    offset = 0
    new_epoch_names=[]
    for name, group in temp_epochs.groupby('stim'):
        # build a list of epoch names where all the values are equal
        m_equal =(group.groupby('name').agg({
            'start': lambda x: len(set(x)) == 1,
            'end': lambda x: len(set(x)) == 1,
        }).all(axis=1)
           )
        m_equal = list(m_equal.index[m_equal].values)
        m_equal.extend([name,'REFERENCE','PreStimSilence','PostStimSilence'])
        
        # find the epoch names that are common to every group
        s = set()
        for idx, (cat_name, cat_group) in enumerate(group.groupby('cat')):
            if idx == 0:
                s.update(cat_group['name'])
            else:
                s.intersection_update(cat_group['name'])

        # drop where values across names aren't equal, or where a group is missing an epoch
        keep_mask = (group['name'].isin(m_equal)) & (group['name'].isin(s))

        g = group[keep_mask].drop(['cat', 'stim'], axis=1).drop_duplicates()
        max_end = g['end'].max()
        g[['start', 'end']] += offset
        
        #if max_end>=expected_max:
        concat.append(g)
        offset += max_end
        new_epoch_names.append(name)
        #else:
        #    log.info(f"dropping epoch {name} because it's too short")
            
        if np.isnan(offset):
            log.info('nan offset')

    new_epochs = pd.concat(concat).sort_values(['start', 'end', 'name']).reset_index(drop=True)
    epoch_names=new_epoch_names
    
    # make name the temp_epochs index for quick start/end lookup in loop below
    temp_epochs = (temp_epochs[['name', 'start', 'end']]
                   .drop_duplicates()
                   .set_index('name')
                   .assign(dur=lambda x: (x['end'] - x['start']).astype(float))
                   .drop(['start', 'end'], axis='columns')
                   )

    #averaged_recording = recording.copy()
    averaged_signals = {}
    for signal_name, signal in recording.signals.items():
        # TODO: this may be better done as a method in signal subclasses since
        # some subclasses may have more efficient approaches (e.g.,
        # TiledSignal)

        # Extract all occurences of each epoch, returning a dict where keys are
        # stimuli and each value in the dictionary is (reps X cell X bins)
        #print(signal_name)
        epoch_data = signal.rasterize().extract_epochs(epoch_names)

        fs = signal.fs
        # Average over all occurrences of each epoch
        data = []
        for epoch_name in epoch_names:
            epoch = epoch_data[epoch_name]
            
            # TODO: fix empty matrix error. do epochs align properly?
            if epoch.dtype == bool:
                epoch = epoch[0,...]
            elif np.sum(np.isfinite(epoch)):
                epoch = np.nanmean(epoch, axis=0)
            else:
                epoch = epoch[0,...]

            elen = int(round(np.min(temp_epochs.loc[epoch_name, 'dur'] * fs)))

            if epoch.shape[-1] > elen:
                #log.info('truncating epoch_data for epoch %s', epoch_name)
                #epoch = epoch[..., :elen]
                log.info('NOT truncating epoch_data for epoch %s', epoch_name)
                log.info(f"{epoch}")
            elif epoch.shape[-1]<elen:
                pad = np.zeros((epoch.shape[0], elen-epoch.shape[1])) * np.nan
                epoch = np.concatenate((epoch, pad), axis=1)
                log.info('padding epoch_data for epoch %s with nan', epoch_name)

            data.append(epoch)
            
        data = np.concatenate(data, axis=-1)
        if data.shape[-1] != round(signal.fs * offset):
            raise ValueError('Misalignment issue in averaging signal')

        averaged_signal = signal._modified_copy(data, epochs=new_epochs)
        averaged_signals[signal_name] = averaged_signal

#        # TODO: Eventually need a smarter check for this in case it's named
#        #       something else. Basically just want to preserve spike data.
#        if signal.name == 'resp':
#            spikes = signal.copy()
#            spikes.name = signal.name + ' spikes'
#            averaged_recording.add_signal(spikes)
    averaged_recording = Recording(averaged_signals,
                                   meta=recording.meta,
                                   name=recording.name)
    return averaged_recording


def remove_invalid_segments(rec):
    """
    Currently a specialized function for removing incorrect trials from data
    collected using baphy during behavior.

    TODO: Migrate to nems_lbhb or make a more generic version
    """

    # First, select the appropriate subset of data
    rec['resp'] = rec['resp'].rasterize()
    if 'stim' in rec.signals.keys():
        rec['stim'] = rec['stim'].rasterize()

    sig = rec['resp']

    # get list of start and stop indices (epoch bounds)
    epoch_indices = np.vstack((
            ep.epoch_intersection(sig.get_epoch_indices('REFERENCE'),
                                  sig.get_epoch_indices('HIT_TRIAL')),
            ep.epoch_intersection(sig.get_epoch_indices('REFERENCE'),
                                  sig.get_epoch_indices('PASSIVE_EXPERIMENT'))))

    # Only takes the first of any conflicts (don't think I actually need this)
    epoch_indices = ep.remove_overlap(epoch_indices)

    # merge any epochs that are directly adjacent
    epoch_indices2 = epoch_indices[0:1]
    for i in range(1, epoch_indices.shape[0]):
        if epoch_indices[i, 0] == epoch_indices2[-1, 1]:
            epoch_indices2[-1, 1] = epoch_indices[i, 1]
        else:
            #epoch_indices2 = np.concatenate(
             #       (epoch_indices2, epoch_indices[i:(i + 1), :]), axis=0)
            epoch_indices2=np.append(epoch_indices2,epoch_indices[i:(i+1)], axis=0)

    # convert back to times
    epoch_times = epoch_indices2 / sig.fs

    # add adjusted signals to the recording
    newrec = rec.select_times(epoch_times)

    return newrec


def mask_all_but_correct_references(rec, balance_rep_count=False, include_incorrect=False, 
                                    generate_evoked_mask=False, exclude_partial_ref=True):
    """
    Specialized function for removing incorrect trials from data
    collected using baphy during behavior.
    exclude_nans: remove any REF epoch with nans in the response

    TODO: Migrate to nems_lbhb and/or make a more generic version
    """
    newrec = rec.copy()
    if 'mask' in newrec.signals.keys():
        log.debug('valid bins coming in: %d',np.sum(newrec['mask'].as_continuous()))

    newrec = normalize_epoch_lengths(newrec, resp_sig='resp', epoch_regex='^STIM_|^REF|^TAR',
                                     include_incorrect=include_incorrect)

    newrec['resp'] = newrec['resp'].rasterize()
    if 'stim' in newrec.signals.keys():
        newrec['stim'] = newrec['stim'].rasterize()
    resp = newrec['resp']

    if balance_rep_count:

        epoch_regex = "^STIM_"
        epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
        p=resp.get_epoch_indices("PASSIVE_EXPERIMENT")
        a=np.concatenate((resp.get_epoch_indices("HIT_TRIAL"),
                          resp.get_epoch_indices("CORRECT_REJECT_TRIAL")), axis=0)

        epoch_list=[]
        for s in epochs_to_extract:
            e = resp.get_epoch_indices(s)
            pe = ep.epoch_intersection(e, p)
            ae = ep.epoch_intersection(e, a)
            if len(pe)>len(ae):
                epoch_list.extend(ae)
                subset=np.round(np.linspace(0,len(pe),len(ae)+1)).astype(int)
                for i in subset[:-1]:
                    epoch_list.append(pe[i])
            else:
                subset=np.round(np.linspace(0,len(ae),len(pe)+1)).astype(int)
                for i in subset[:-1]:
                    epoch_list.append(ae[i])
                epoch_list.extend(pe)

        newrec = newrec.create_mask(epoch_list)

    elif include_incorrect:
        log.info('INCLUDING ALL TRIALS (CORRECT AND INCORRECT)')
        newrec = newrec.and_mask(['REFERENCE'])

    else:
        newrec = newrec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL'])
        newrec = newrec.and_mask(['REFERENCE'])

    if exclude_partial_ref:
        mask_data = newrec['mask'].extract_epoch('REFERENCE')
        pp = np.mean(mask_data, axis=2)[:,0]
        # if partial mask, remove completely
        mask_data[(pp>0) & (pp<1),:,:]=0
        tt = (pp>0) & (pp<1) 
        if tt.sum() > 0:
            log.info('removing %d incomplete REFERENCES', tt.sum())
        newrec.signals['mask']=newrec['mask'].replace_epoch('REFERENCE', mask_data)

    # figure out if some actives should be masked out
#    t = ep.epoch_names_matching(resp.epochs, "^TAR_")
#    tm = [tt[:-2] for tt in t]  # trim last digits
#    active_epochs = resp.get_epoch_indices("ACTIVE_EXPERIMENT")
#    if len(set(tm)) > 1 and len(active_epochs) > 1:
#        print('Multiple targets: ', tm)
#        files = ep.epoch_names_matching(resp.epochs, "^FILE_")
#        keep_files = files
#        e = active_epochs[1]
#        for i,f in enumerate(files):
#            fi = resp.get_epoch_indices(f)
#            if any(ep.epoch_contains([e], fi, 'both')):
#                keep_files = files[:i]
#
#        print('Print keeping files: ', keep_files)
#        newrec = newrec.and_mask(keep_files)

    if 'state' in newrec.signals:
        b_states = ['far', 'hit', 'lick',
                    'puretone_trials', 'easy_trials', 'hard_trials']
        trec = newrec.copy()
        trec = trec.and_mask(['ACTIVE_EXPERIMENT'])
        st = trec['state'].as_continuous().copy()
        str = trec['state_raw'].as_continuous().copy()
        mask = trec['mask'].as_continuous()[0, :]
        for s in trec['state'].chans:
            if s in b_states:
                i = trec['state'].chans.index(s)
                m = np.nanmean(st[i, mask])
                sd = np.nanstd(st[i, mask])
                # print("{} {}: m={}, std={}".format(s, i, m, sd))
                # print(np.sum(mask))
                st[i, mask] -= m
                st[i, mask] /= sd
                str[i, mask] -= m
                str[i, mask] /= sd
        newrec['state'] = newrec['state']._modified_copy(st)
        newrec['state_raw'] = newrec['state_raw']._modified_copy(str)

    if generate_evoked_mask:
        mask = newrec['mask'].as_continuous().copy()
        padbins=int(np.round(newrec['resp'].fs * 0.1))

        preidx = resp.get_epoch_indices('PreStimSilence', mask=newrec['mask'])
        posidx = resp.get_epoch_indices('PostStimSilence', mask=newrec['mask'])
        for i,p in enumerate(posidx):
            posidx[i]=(p[0]+padbins, p[1])

        post_mask = newrec['resp'].epoch_to_signal(indices=posidx)
        pre_mask = newrec['resp'].epoch_to_signal(indices=preidx)
        #mask[post_mask.as_continuous()] = False
        ev_mask = mask.copy()
        ev_mask[pre_mask.as_continuous()] = False
        ev_mask[post_mask.as_continuous()] = False
        newrec['sp_mask'] = newrec['mask']._modified_copy(data=mask)
        newrec['ev_mask'] = newrec['mask']._modified_copy(data=ev_mask)

    return newrec


def mask_keep_passive(rec, max_passive_blocks=2):
    """
    Mask out all times that don't fall in PASSIVE_EXPERIMENT epochs.

    TODO: Migrate to nems_lbhb and/or make a more generic version
    """

    newrec = rec.copy()
    newrec['resp'] = newrec['resp'].rasterize()
    if 'stim' in newrec.signals.keys():
        newrec['stim'] = newrec['stim'].rasterize()

    passive_epochs = newrec['resp'].get_epoch_indices("PASSIVE_EXPERIMENT")
    passive_epochs = passive_epochs[:max_passive_blocks]
    newrec = newrec.and_mask(passive_epochs)

    return newrec


def mask_late_passives(rec):
    """
    Mask out all times aren't in active or first passive

    TODO: Migrate to nems_lbhb and/or make a more generic version
    """

    newrec = rec.copy()

    resp = newrec['resp']
    file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")
    passive_indices = resp.get_epoch_indices('PASSIVE_EXPERIMENT')

    if 'mask' in newrec.signals.keys():
        del newrec.signals['mask']

    pcount = 0
    for f in file_epochs:

        # test if passive expt
        f_indices = resp.get_epoch_indices(f)
        epoch_indices = ep.epoch_intersection(f_indices, passive_indices)

        add_file = True
        if epoch_indices.size:
            # this is a passive file
            pcount += 1
            if pcount > 1:
                add_file = False

        if add_file:
            newrec = newrec.or_mask(f_indices)
            log.info("Including %s", f)
        else:
            log.info("Skipping %s", f)

    log.debug('valid bins after removing late passives: %d',np.sum(newrec['mask'].as_continuous()))

    return newrec


def mask_all_but_targets(rec, include_incorrect=True):
    """
    Specialized function for removing incorrect trials from data
    collected using baphy during behavior.

    TODO: Migrate to nems_lbhb and/or make a more generic version
    """

    newrec = rec.copy()
    newrec['resp'] = newrec['resp'].rasterize()
    #newrec = normalize_epoch_lengths(newrec, resp_sig='resp', epoch_regex='TARGET',
    #                                include_incorrect=include_incorrect)
    if 'stim' in newrec.signals.keys():
        newrec['stim'] = newrec['stim'].rasterize()

    #newrec = newrec.or_mask(['TARGET'])
    #newrec = newrec.and_mask(['PASSIVE_EXPERIMENT', 'TARGET'])
    #newrec = newrec.and_mask(['REFERENCE','TARGET'])
    newrec = newrec.and_mask(['TARGET'])

    if not include_incorrect:
        newrec = mask_incorrect(newrec)

    # svd attempt to kludge this masking to work with a lot of code that assumes all relevant epochs are
    # called "REFERENCE"
    #import pdb;pdb.set_trace()
    for k in newrec.signals.keys():
        newrec[k].epochs.name = newrec[k].epochs.name.str.replace("TARGET", "REFERENCE")
    return newrec

def mask_incorrect(rec, include_ITI=True, ITI_sec_to_include=None, **context):
    """
    Specialized function for removing incorrect trials from data
    collected using baphy during behavior.
    """
    newrec = rec.copy()
    
    if include_ITI:
        e=newrec['resp'].epochs
        et=e.loc[e.name=="TRIAL"]
        for i,r in e.loc[e.name.str.endswith("TRIAL")].iterrows():
            next_trial=et.loc[et.start>r.start,'start'].min()
            if ~np.isnan(next_trial):
                # data exists after current trail
                if ITI_sec_to_include is not None:
                    # limit amount of post-target data to include
                    if next_trial > e.at[i,'end']+ITI_sec_to_include:
                        next_trial = e.at[i,'end']+ITI_sec_to_include
                e.at[i,'end']=next_trial
            #print(i, r.start, next_trial)
        for s in list(newrec.signals.keys()):
            newrec[s].epochs = e
    newrec = newrec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL'])

    return newrec

def nan_invalid_segments(rec):
    """
    Currently a specialized signal for removing incorrect trials from data
    collected using baphy during behavior.

    TODO: Complete this function, replicate remove_invalid_segments logic
          Or delete ME
    TODO: Migrate to nems_lbhb or make a more generic version
    """

    # First, select the appropriate subset of data
    rec['resp'] = rec['resp'].rasterize()
    sig = rec['resp']

    # get list of start and stop times (epoch bounds)
    epoch_indices = np.vstack((
            ep.epoch_intersection(sig.get_epoch_bounds('HIT_TRIAL'),
                                  sig.get_epoch_bounds('REFERENCE')),
            ep.epoch_intersection(sig.get_epoch_bounds('REFERENCE'),
                                  sig.get_epoch_bounds('PASSIVE_EXPERIMENT'))))

    # Only takes the first of any conflicts (don't think I actually need this)
    epoch_indices = ep.remove_overlap(epoch_indices)

    epoch_indices2 = epoch_indices[0:1, :]
    for i in range(1, epoch_indices.shape[0]):
        if epoch_indices[i, 0] == epoch_indices2[-1, 1]:
            epoch_indices2[-1, 1] = epoch_indices[i, 0]
        else:
            epoch_indices2 = np.concatenate(
                    (epoch_indices2, epoch_indices[i: (i+1), :]), axis=0
                    )

    # add adjusted signals to the recording
    newrec = rec.nan_times(epoch_indices2)

    return newrec


def generate_stim_from_epochs(rec, new_signal_name='stim',
                              epoch_regex='^STIM_', epoch_shift=0,
                              epoch2_regex=None, epoch2_shift=0,
                              epoch2_shuffle=False,
                              onsets_only=True):

    rec = rec.copy()
    resp = rec['resp'].rasterize()

    epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
    if not epochs_to_extract:
        log.info("generate_stim_from_epochs: no epochs matching regex: %s, skipping ..." % epoch_regex)

    else:
        # sort extracted stim
        try:
            n = {e: float(e.split("_")[1])+100000*(e.split("_")[0]=="TAR") for e in epochs_to_extract}
            def _myfunc(i):
                return(n[i])
            epochs_to_extract.sort(key=_myfunc)
            log.info('sorted epochs')
        except:
            pass

        sigs = []
        for e in epochs_to_extract:
            log.info('Adding to %s: %s with shift = %d',
                     new_signal_name, e, epoch_shift)
            s = resp.epoch_to_signal(e, onsets_only=onsets_only, shift=epoch_shift)
            if epoch_shift:
                s.chans[0] = "{}{:+d}".format(s.chans[0], epoch_shift)
            sigs.append(s)

        if epoch2_regex is not None:
            epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch2_regex)
            for e in epochs_to_extract:
                log.info('Adding to %s: %s with shift = %d',
                         new_signal_name, e, epoch2_shift)
                s = resp.epoch_to_signal(e, onsets_only=onsets_only,
                                         shift=epoch2_shift)
                if epoch2_shuffle:
                    log.info('Shuffling %s', e)
                    s = s.shuffle_time()
                    s.chans[0] = "{}_shf".format(s.chans[0])
                if epoch_shift:
                    s.chans[0] = "{}{:+d}".format(s.chans[0], epoch2_shift)
                sigs.append(s)

        stim = sigs[0].concatenate_channels(sigs)
        stim.name = new_signal_name

        # add_signal operates in place
        rec.add_signal(stim)

    return rec


def integrate_signal_per_epoch(rec, sig='stim', sig_out='stim_int', epoch_regex='^STIM_'):
    '''
    Calculates integral for each epoch of a signal

    if rec['mask'] exists, uses rec['mask'] == True to determine valid epochs
    '''

    newrec = rec.copy()
    sig = newrec[sig].rasterize()

    # compute PSTH response during valid trials
    if type(epoch_regex) == list:
        epochs_to_extract = []
        for rx in epoch_regex:
            eps = ep.epoch_names_matching(sig.epochs, rx)
            epochs_to_extract += eps

    elif type(epoch_regex) == str:
        epochs_to_extract = ep.epoch_names_matching(sig.epochs, epoch_regex)

    folded_matrices = sig.extract_epochs(epochs_to_extract,
                                         mask=newrec['mask'])

    # 2. Average across reps and integrate each stim
    for k, v in folded_matrices.items():
        v = np.nanmean(v, axis=0)
        v = cumtrapz(v, dx=1/sig.fs, initial=0)
        folded_matrices[k] = v

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    new_sig = sig._modified_copy(data=np.zeros_like(sig._data))
    new_sig = new_sig.replace_epochs(folded_matrices)
    new_sig.name = sig_out

    # add the new signals to the recording
    newrec.add_signal(new_sig)

    return newrec


def normalize_epoch_lengths(rec, resp_sig='resp', epoch_regex='^STIM_',
                            include_incorrect=False):
    """
    for each set of epochs matching epoch_regex, figure out minimum length and truncate all
    occurrences to that length
    :param rec:
    :param resp_sig:
    :param epoch_regex:
    :param include_incorrect: (False) not used
    :return:
    """
    newrec = rec.copy()
    resp = newrec[resp_sig].rasterize()
    log.info('normalize_epoch_lengths: %s',epoch_regex)

    if include_incorrect:
        log.info('INCLUDING ALL TRIALS (CORRECT AND INCORRECT)')
        #newrec = newrec.and_mask(['REFERENCE'])
    else:
        newrec = newrec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL'])
        #newrec = newrec.and_mask(['REFERENCE'])

    mask = newrec['mask']
    del newrec.signals['mask']
    mask.epochs = mask.to_epochs()
    mask_bounds = mask.get_epoch_bounds('mask')

    # Figure out list of matching epoch names: epochs_to_extract
    if type(epoch_regex) == list:
        epochs_to_extract = []
        for rx in epoch_regex:
            eps = ep.epoch_names_matching(resp.epochs, rx)
            epochs_to_extract += eps

    elif type(epoch_regex) == str:
        epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
    else:
        raise ValueError('unknown epoch_regex format')

    # find all pre/post-stim silence epochs
    preidx = resp.get_epoch_bounds('PreStimSilence')
    posidx = resp.get_epoch_bounds('PostStimSilence')
    epochs_new = resp.epochs.copy()
    precision = int(np.ceil(np.log(resp.fs)))
    log.debug('decimal precision = %d', precision)
    for ename in epochs_to_extract:
        ematch = resp.get_epoch_bounds(ename)
        # remove events outside of valid trial mask (if excluding incorrect)
        #if not include_incorrect:
        #   ematch = ep.epoch_intersection(ematch, mask_bounds)
        if len(ematch)>0:
           # for CC data, may be "STIM_nnn" tone events that have been excluded in REF analysis
           ematch_new = ematch.copy()
           prematch = np.zeros((ematch.shape[0],1))
           posmatch = np.zeros((ematch.shape[0],1))
           for i,e in enumerate(ematch):
               #x = ep.epoch_intersection(preidx, [e], precision=precision)
               x = preidx[preidx[:,0]==e[0]]

               if len(x):
                   prematch[i] = np.diff(x[0,:])
               else:
                   log.info('pre missing?')
                   prematch[i] = 0
                   #import pdb; pdb.set_trace()
               x = ep.epoch_intersection(posidx, [e], precision=precision)
               if len(x):
                   if x.shape[0]>1:
                      import pdb; pdb.set_trace()
                   posmatch[i] = np.diff(x)
           prematch = np.round(prematch, decimals=precision)
           posmatch = np.round(posmatch, decimals=precision)
           dur = np.round(np.diff(ematch, axis=1)-prematch-posmatch, decimals=precision)

           # weird hack to deal with CC data dropping post-stimsilence
           if sum(posmatch>0)>0:
               udur = np.sort(np.unique(dur[posmatch>0]))
           else:
               udur = np.sort(np.unique(dur))
           mindur = np.min(udur)
           dur[dur < mindur] = mindur

           #log.info('epoch %s: n=%d dur range=%.4f-%.4f', ename,
           #         len(ematch), mindur, np.max(dur))

           #import pdb;pdb.set_trace()
           minpre = np.min(prematch)
           if (posmatch>0).sum():
               minpos = np.min(posmatch[posmatch > 0])
           else:
               minpos = 0
           posmatch[posmatch < minpos] = minpos
           remove_post_stim = False
           if (minpre<np.max(prematch)) & (ematch.shape[0]==prematch.shape[0]):
               log.info('epoch %s pre varies, fixing to %.3f s', ename, minpre)
               ematch_new[:,0] = ematch_new[:,0]-minpre+prematch.T
           if (mindur<np.max(dur)):
               log.info('epoch %s dur varies, fixing to %.3f s', ename, mindur)
               ematch_new[:,1] = ematch_new[:,0]+mindur
               remove_post_stim = True
           elif (minpos<np.max(posmatch)):
               log.info('epoch %s pos varies, fixing to %.3f s', ename, minpos)
               ematch_new[:,1] = ematch_new[:,0]+minpre+dur.T+minpos

           for e_old, e_new in zip(ematch, ematch_new):
               _mask = (np.round(epochs_new['start']-e_old[0], precision)==0) & \
                       (np.round(epochs_new['end']-e_old[1], precision)==0)
               epochs_new.loc[_mask,'start'] = e_new[0]
               epochs_new.loc[_mask,'end'] = e_new[1]
               if remove_post_stim:
                   _mask = epochs_new['name'].str.startswith("PostStimSilence") & \
                           (np.round(epochs_new['end'] - e_old[1], precision) == 0)
                   epochs_new.loc[_mask, 'start'] = e_new[1]
                   epochs_new.loc[_mask, 'end'] = e_new[1]

    if 'mask' in rec.signals.keys():
        newrec.signals['mask'] = rec['mask'].copy()

    # save revised epochs back to all signals in the recording
    for k in newrec.signals.keys():
        newrec[k].epochs = epochs_new.copy()

    return newrec


def generate_psth_from_resp(rec, resp_sig='resp', epoch_regex='^(STIM_|TAR_|CAT_|REF_)',
                            smooth_resp=False, channel_per_stim=False, mean_zero=False):
    '''
    Estimates a PSTH from all responses to each regex match in a recording

    subtract spont rate based on pre-stim silence for ALL estimation data.

    if rec['mask'] exists, uses rec['mask'] == True to determine valid epochs

    Problem: not all the Pre/Dur/Post lengths are the same across reps of a stimulus.
    Shorten everything to minimum of each. If Dur is variable, throw away post-stim silence.

    '''

    newrec = rec.copy()
    newrec[resp_sig] = newrec[resp_sig].rasterize()
    resp = newrec[resp_sig]

    # compute spont rate during valid (non-masked) trials
    if 'mask' in newrec.signals.keys():
        mask = newrec['mask']
    else:
        mask = None

    # Figure out list of matching epoch names: epochs_to_extract
    if type(epoch_regex) == list:
        epochs_to_extract = []
        for rx in epoch_regex:
            eps = ep.epoch_names_matching(resp.epochs, rx)
            epochs_to_extract += eps

    elif type(epoch_regex) == str:
        epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)

    # figure out spont rate for subtraction from PSTH
    if np.sum(resp.epochs.name=='ITI')>0:
        spontname='ITI'
        prestimsilence = resp.extract_epoch(spontname, mask=mask)
    elif np.sum(resp.epochs.name == 'TRIALPreStimSilence') > 0:
        # special case where the epochs included in mask don't have PreStimSilence,
        # so we get it elsewhere. Designed for CPN data...
        spontname = 'TRIALPreStimSilence'
        prestimsilence = resp.extract_epoch(spontname)
    elif np.sum(resp.epochs.name=='PreStimSilence')>0:
        spontname='PreStimSilence'
        prestimsilence = resp.extract_epoch(spontname, mask = mask)
    else:
        raise ValueError("Can't find pre-stim silence to use for PSTH calculation")
    if prestimsilence.shape[-1] > 0:
        if len(prestimsilence.shape) == 3:
            spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
        else:
            spont_rate = np.nanmean(prestimsilence)

    NEW_WAY = True
    if NEW_WAY:
        # already taken care of?
        #newrec = normalize_epoch_lengths(newrec, resp_sig=resp_sig, epoch_regex='^STIM_')
        #resp = newrec[resp_sig].rasterize()
        # find all pre/post-stimsilence epochs
        preidx = resp.get_epoch_indices('PreStimSilence', mask=mask)
        posidx = resp.get_epoch_indices('PostStimSilence', mask=mask)
        prebins = preidx[0][1] - preidx[0][0]
        if posidx.shape[0]>0:
            postbins = posidx[0][1] - posidx[0][0]
        else:
            postbins = 0
    else:
        # find all pre/post-stimsilence epochs
        preidx = resp.get_epoch_indices('PreStimSilence', mask=mask)
        posidx = resp.get_epoch_indices('PostStimSilence', mask=mask)
        dpre=preidx[:,1]-preidx[:,0]
        minpre=np.min(dpre)
        prebins = preidx[0][1] - preidx[0][0]
        dpos=posidx[:,1]-posidx[:,0]
        minpos=np.min(dpre)
        postbins = posidx[0][1] - posidx[0][0]
        #refidx = resp.get_epoch_indices('REFERENCE')

        #import pdb
        #pdb.set_trace()
        for ename in epochs_to_extract:
            ematch = np.argwhere(resp.epochs['name']==ename)
            import pdb; pdb.set_trace()
            ff = resp.get_epoch_indices(ename, mask=mask)
            for i,fe in enumerate(ff):
                re = ((resp.epochs['name']=='REFERENCE') &
                      (resp.epochs['start']==fe[0]/resp.fs))
                pe = ep.epoch_contained(preidx, [fe])
                thispdur = np.diff(preidx[pe])

                if np.sum(pe)==1 and thispdur>minpre:
                    print('adjust {} to {}'.format(thispdur, minpre))
                    print(resp.epochs.loc[ematch[i]])
                    resp.epochs.loc[ematch[i],'start'] += (thispdur[0,0]-minpre)/resp.fs
                    resp.epochs.loc[re,'start'] += (thispdur[0,0]-minpre)/resp.fs
                    print(resp.epochs.loc[ematch[i]])

                pe = ep.epoch_contained(posidx, [fe])
                thispdur = np.diff(posidx[pe])
                if thispdur.shape and thispdur>minpos:
                    print('adjust {} to {}'.format(thispdur, minpos))
                    print(resp.epochs.loc[ematch[i]])
                    resp.epochs.loc[ematch[i],'end'] -= (thispdur[0,0]-minpos)/resp.fs
                    resp.epochs.loc[re,'end'] -= (thispdur[0,0]-minpos)/resp.fs
                    print(resp.epochs.loc[ematch[i]])

        newrec['resp'].epochs = resp.epochs.copy()

    # mask=None means no mask
    folded_matrices = resp.extract_epochs(epochs_to_extract, mask=newrec['mask'])

    log.info('generating PSTHs for %d epochs', len(folded_matrices.keys()))

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    per_stim_psth_spont = dict()
    total = np.zeros((resp.shape[0],1))
    total_n = 0

    for k, v in folded_matrices.items():
        if smooth_resp:
            # replace each epoch (pre, during, post) with average
            v[:, :, :prebins] = np.nanmean(v[:, :, :prebins],
                                           axis=2, keepdims=True)
            v[:, :, prebins:(prebins+2)] = np.nanmean(v[:, :, prebins:(prebins+2)],
                                                      axis=2, keepdims=True)
            v[:, :, (prebins+2):-postbins] = np.nanmean(v[:, :, (prebins+2):-postbins],
                                                        axis=2, keepdims=True)
            v[:, :, -postbins:(-postbins+2)] = np.nanmean(v[:, :, -postbins:(-postbins+2)],
                                                          axis=2, keepdims=True)
            v[:, :, (-postbins+2):] = np.nanmean(v[:, :, (-postbins+2):],
                                                 axis=2, keepdims=True)

        per_stim_psth[k] = np.nanmean(v, axis=0) - spont_rate[:, np.newaxis]
        per_stim_psth_spont[k] = np.nanmean(v, axis=0)
        folded_matrices[k] = v

        total += v.mean(axis=2, keepdims=True).sum(axis=0)
        total_n += v.shape[0]


    #import pdb; pdb.set_trace()
    if mean_zero:
        total=total/total_n
        for k, v in folded_matrices.items():
            per_stim_psth[k] = per_stim_psth_spont[k] - total

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    if channel_per_stim:
        raise ValueError('channel_per_stim not yet supported')

    #respavg = resp.replace_epochs(per_stim_psth, zero_outside=True)
    respavg = resp.replace_epochs(per_stim_psth)
    respavg_with_spont = resp.replace_epochs(per_stim_psth_spont)
    respavg.name = 'psth'
    respavg_with_spont.name = 'psth_sp'
    
    # Fill in a all non-masked periods with 0 (presumably, these are spont
    # periods not contained within stimulus epochs), or spont rate (for the signal
    # containing spont rate)
    respavg_data = respavg.as_continuous().copy()
    respavg_spont_data = respavg_with_spont.as_continuous().copy()

    if 'mask' in newrec.signals.keys():
        mask_data = newrec['mask']._data
    else:
        mask_data = np.ones(respavg_data.shape).astype(np.bool)
    spont_periods = ((np.isnan(respavg_data)) & (mask_data==True))
    respavg_data[:, spont_periods[0,:]] = 0
    # respavg_spont_data[:, spont_periods[0,:]] = spont_rate[:, np.newaxis]

    respavg = respavg._modified_copy(respavg_data)
    respavg_with_spont = respavg_with_spont._modified_copy(respavg_spont_data)

    # add the new signals to the recording
    newrec.add_signal(respavg)
    newrec.add_signal(respavg_with_spont)
    if 'stim' in newrec.signals.keys():
        # add as channel to stim signal if it exists
        newrec['stim'] = newrec['stim'].rasterize()
        newrec = concatenate_state_channel(newrec, respavg, 'stim')
        newrec['stim'].chans[-1] = 'psth'

    if smooth_resp:
        log.info('Replacing resp with smoothed resp')
        resp = resp.replace_epochs(folded_matrices, mask=newrec['mask'])
        newrec.add_signal(resp)

    return newrec


def smooth_signal_epochs(rec, signal='resp', epoch_regex='^STIM_',
                         **context):
    """
    xforms-compatible wrapper for smooth_epoch_segments
    """

    newrec = rec.copy()

    smoothed_sig, respavg, respavg_with_spont = smooth_epoch_segments(
        newrec[signal], epoch_regex=epoch_regex, mask=newrec['mask'])

    newrec.add_signal(smoothed_sig)

    return {'rec': newrec}

def decimate_signal(rec, stride=2, signal='resp', **context):

    new_rec = rec.copy()
    new_rec[signal] = new_rec[signal].decimate(stride=stride)

    return {'rec': new_rec}


def smooth_epoch_segments(sig, epoch_regex='^STIM_', mask=None):
    """
    wonky function that "smooths" signals by computing the mean of the
    pre-stim silence, onset, sustained, and post-stim silence
    Used in PSTH-based models. Duration of onset hard-coded to 2 bins
    :return: (smoothed_sig, respavg, respavg_with_spont)
    smoothed_sig - smoothed signal
    respavg - smoothed signal, averaged across all reps of matching epochs
    """

    # compute spont rate during valid (non-masked) trials
    prestimsilence = sig.extract_epoch('PreStimSilence', mask=mask)

    if len(prestimsilence.shape) == 3:
        spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
    else:
        spont_rate = np.nanmean(prestimsilence)

    preidx = sig.get_epoch_indices('PreStimSilence', mask=mask)
    dpre=preidx[:,1]-preidx[:,0]
    minpre=np.min(dpre)
    prebins = preidx[0][1] - preidx[0][0]
    posidx = sig.get_epoch_indices('PostStimSilence', mask=mask)
    dpos=posidx[:,1]-posidx[:,0]
    minpos=np.min(dpre)
    postbins = posidx[0][1] - posidx[0][0]
    #refidx = sig.get_epoch_indices('REFERENCE')

    # compute PSTH response during valid trials
    if type(epoch_regex) == list:
        epochs_to_extract = []
        for rx in epoch_regex:
            eps = ep.epoch_names_matching(resp.epochs, rx)
            epochs_to_extract += eps

    elif type(epoch_regex) == str:
        epochs_to_extract = ep.epoch_names_matching(sig.epochs, epoch_regex)
    else:
        raise ValueError("invalid epoch_regex")

    #import pdb
    #pdb.set_trace()
    for ename in epochs_to_extract:
        ematch = np.argwhere(sig.epochs['name']==ename)
        ff = sig.get_epoch_indices(ename, mask=mask)
        for i,fe in enumerate(ff):
            re = ((sig.epochs['name'] == 'REFERENCE') &
                  (sig.epochs['start'] == fe[0]/sig.fs))
            pe = ep.epoch_contained(preidx, [fe])
            thispdur = np.diff(preidx[pe])

            if np.sum(pe)==1 and thispdur>minpre:
                print('adjust {} to {}'.format(thispdur, minpre))
                print(sig.epochs.loc[ematch[i]])
                sig.epochs.loc[ematch[i],'start'] += (thispdur[0,0]-minpre)/resp.fs
                sig.epochs.loc[re,'start'] += (thispdur[0,0]-minpre)/resp.fs
                print(sig.epochs.loc[ematch[i]])

            pe = ep.epoch_contained(posidx, [fe])
            thispdur = np.diff(posidx[pe])
            if thispdur.shape and thispdur>minpos:
                print('adjust {} to {}'.format(thispdur, minpos))
                print(sig.epochs.loc[ematch[i]])
                sig.epochs.loc[ematch[i],'end'] -= (thispdur[0,0]-minpos)/resp.fs
                sig.epochs.loc[re,'end'] -= (thispdur[0,0]-minpos)/resp.fs
                print(resp.epochs.loc[ematch[i]])

    smoothed_sig = sig.copy()
    smoothed_sig.epochs = smoothed_sig.epochs.copy()

    folded_matrices = smoothed_sig.extract_epochs(epochs_to_extract, mask=mask)

    # 2. Average over all reps of each epoch and save into dict called psth.
    per_stim_psth = dict()
    per_stim_psth_spont = dict()
    for k, v in folded_matrices.items():
        # replace each epoch (pre, during, post) with average
        v[:, :, :prebins] = np.nanmean(v[:, :, :prebins],
                                       axis=2, keepdims=True)
        v[:, :, prebins:(prebins+2)] = np.nanmean(v[:, :, prebins:(prebins+2)],
                                                  axis=2, keepdims=True)
        v[:, :, (prebins+2):-postbins] = np.nanmean(v[:, :, (prebins+2):-postbins],
                                                    axis=2, keepdims=True)
        v[:, :, -postbins:(-postbins+2)] = np.nanmean(v[:, :, -postbins:(-postbins+2)],
                                                      axis=2, keepdims=True)
        v[:, :, (-postbins+2):] = np.nanmean(v[:, :, (-postbins+2):],
                                             axis=2, keepdims=True)

        per_stim_psth[k] = np.nanmean(v, axis=0) - spont_rate[:, np.newaxis]
        per_stim_psth_spont[k] = np.nanmean(v, axis=0)
        folded_matrices[k] = v

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    log.info('Replacing resp with smoothed resp')
    smoothed_sig = smoothed_sig.replace_epochs(folded_matrices, mask=mask)

    respavg = smoothed_sig.replace_epochs(per_stim_psth)
    respavg.name = 'psth'
    respavg_with_spont = smoothed_sig.replace_epochs(per_stim_psth_spont)
    respavg_with_spont.name = 'psth_sp'

    # Fill in a all non-masked periods with 0 (presumably, these are spont
    # periods not contained within stimulus epochs), or spont rate (for the signal
    # containing spont rate)
    respavg_data = respavg.as_continuous().copy()
    respavg_spont_data = respavg_with_spont.as_continuous().copy()

    if mask is not None:
        mask_data = mask._data
    else:
        mask_data = np.ones(respavg_data.shape).astype(np.bool)

    spont_periods = ((np.isnan(respavg_data)) & (mask_data==True))

    respavg_data[:, spont_periods[0,:]] = 0
    # respavg_spont_data[:, spont_periods[0,:]] = spont_rate[:, np.newaxis]

    respavg = respavg._modified_copy(respavg_data)
    respavg_with_spont = respavg_with_spont._modified_copy(respavg_spont_data)

    return smoothed_sig, respavg, respavg_with_spont


def generate_psth_from_est_for_both_est_and_val(est, val,
                                                epoch_regex='^STIM_', mean_zero=False):
    '''
    Estimates a PSTH from the EST set, and returns two signals based on the
    est and val, in which each repetition of a stim uses the EST PSTH?

    subtract spont rate based on pre-stim silence for ALL estimation data.
    '''

    resp_est = est['resp'].rasterize()
    resp_val = val['resp'].rasterize()

    # compute PSTH response and spont rate during those valid trials
    prestimsilence = resp_est.extract_epoch('PreStimSilence')
    if len(prestimsilence.shape) == 3:
        spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
    else:
        spont_rate = np.nanmean(prestimsilence)

    epochs_to_extract = ep.epoch_names_matching(resp_est.epochs, epoch_regex)
    folded_matrices = resp_est.extract_epochs(epochs_to_extract,
                                              mask=est['mask'])

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    for k in folded_matrices.keys():
        per_stim_psth[k] = np.nanmean(folded_matrices[k], axis=0) - \
            spont_rate[:, np.newaxis]

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg_est = resp_est.replace_epochs(per_stim_psth)
    respavg_est.name = 'psth'

    # add signal to the est recording
    est.add_signal(respavg_est)

    respavg_val = resp_val.replace_epochs(per_stim_psth)
    respavg_val.name = 'psth'

    # add signal to the val recording
    val.add_signal(respavg_val)

    return est, val


def generate_psth_from_est_for_both_est_and_val_nfold(ests, vals,
                                                      epoch_regex='^STIM_', mean_zero=False):
    '''
    call generate_psth_from_est_for_both_est_and_val for each e,v
    pair in ests,vals
    '''
    for e, v in zip(ests, vals):
        e, v = generate_psth_from_est_for_both_est_and_val(e, v, epoch_regex, mean_zero=mean_zero)

    return ests, vals


def resp_to_pc(rec, pc_idx=None, resp_sig='resp', pc_sig='pca',
               pc_count=None, pc_source='all', overwrite_resp=True,
               compute_power='no',
               whiten=True, **context):
    """
    generate PCA transformation of signal, if overwrite_resp==True, replace (multichannel) reference with a single
    pc channel

    :param rec: NEMS recording
    :param pc_idx: subset of pcs to return (default all)
    :param resp_sig: signal to compute PCs
    :param pc_sig: name of signal to save PCs (if not overwrite_resp)
    :param pc_count: number of PCs to save
    :param pc_source: what to compute PCs of (all/psth/noise)
    :param overwrite_resp: (True) if True replace resp_sig with PCs, if False, save in pc_sig
    :param whiten: whiten before PCA
    :param context: NEMS context for xforms compatibility
    :return: copy of rec with PCs
    """
    rec0 = rec.copy()
    resp = rec0[resp_sig]

    # compute duration of spont period
    d = resp.get_epoch_bounds('PreStimSilence')
    if len(d):
        PreStimSilence = np.mean(np.diff(d))
    else:
        PreStimSilence = 0
    prestimbins = int(PreStimSilence * resp.fs)

    # compute PCs only on valid (masked) times
    rec0[resp_sig] = rec0[resp_sig].rasterize()
    rec_masked = generate_psth_from_resp(rec0, resp_sig=resp_sig)

    if 'mask' in rec0.signals:
        rec_masked = rec_masked.apply_mask()

    if pc_source=='all':
        D_ref = rec_masked[resp_sig].as_continuous().T
    elif pc_source=='psth':
        D_ref = rec_masked['psth'].as_continuous().T
    elif pc_source=='noise':
        D_psth = rec_masked['psth'].as_continuous().T
        D_raw = rec_masked[resp_sig].as_continuous().T
        D_ref = D_raw - D_psth
    else:
        raise ValueError('pc_source {} not supported'.format(pc_source))

    # project full response dataset to preserve time
    D = rec0[resp_sig].as_continuous().T

    if pc_count is None:
        pc_count = D_ref.shape[1]
    if pc_idx is None:
        pc_idx=list(np.arange(pc_count))
    elif type(pc_idx) is int:
        pc_idx = [pc_idx]

    log.info(f"PCA: inputs={D_ref.shape[1]}, source={pc_source}, pc_idx(es): {pc_idx}, whiten: {whiten}")

    if False:
        # use sklearn. maybe someday
        pca = PCA(n_components=pc_count)
        pca.fit(D_ref)

        X = pca.transform(D)
    else:
        # each row(??) of v is a PC --weights to project into PC domain
        m = np.nanmean(D_ref, axis=0, keepdims=True)
        if whiten:
            sd = np.nanstd(D_ref, axis=0, keepdims=True)
        else:
            sd = np.ones(m.shape)
        D_ref = D_ref[np.sum(np.isfinite(D_ref),axis=1)>0,:]
        sd[sd==0]=1
        D_ = (D_ref-m)/sd

        #import pdb;
        #pdb.set_trace()

        # A = u @ np.diag(s) @ vh = (u * s) @ vh
        u, s, v = np.linalg.svd(D_.T @ D_, full_matrices=False)
        X = ((D-m) / sd) @ u

        rec0[pc_sig] = rec0[resp_sig]._modified_copy(X.T)

        r = rec0[pc_sig].extract_epoch('REFERENCE', mask=rec0['mask'])
        mr = np.nanmean(r, axis=0)
        if prestimbins>0:
            spont = np.mean(mr[:,:prestimbins],axis=1,keepdims=True)
            mr -= spont
        vs = np.sign(np.sum(mr[:, prestimbins:(prestimbins+10)], axis=1, keepdims=True))
        u *= vs
        X = ((D-m) / sd) @ u

    if compute_power == 'single_trial':
        smwin = 5
        X = smooth(np.abs(X), smwin, axis=0)
        X = X - X.mean(axis=0, keepdims=True)

    pc_chans = [f'PC{i}' for i in range(pc_count)]
    rec0[pc_sig] = rec0[resp_sig]._modified_copy(X.T[np.arange(pc_count),:], chans=pc_chans)

#    r = rec0[pc_sig].extract_epoch('REFERENCE')
#    mr=np.mean(r,axis=0)
#    spont=np.mean(mr[:,:50],axis=1,keepdims=True)
#    mr-=spont
#    plt.figure()
#    plt.plot(mr[:5,:].T)
#    plt.legend(('1','2','3','4','5'))

    rec0.meta['pc_weights'] = u
    rec0.meta['pc_mag'] = s
    if overwrite_resp:
        rec0[resp_sig] = rec0[resp_sig]._modified_copy(X[:, pc_idx].T, 
                                                       chans=[pc_chans[c] for c in pc_idx])
        rec0.meta['pc_idx'] = pc_idx

    return {'rec': rec0}


def make_state_signal(rec, state_signals=['pupil'], permute_signals=[], generate_signals=[],
                      new_signalname='state', sm_win_len=180):
    """
    generate state signal for stategain.S/sdexp.S models

    valid state signals include (incomplete list):
        pupil, pupil_ev, pupil_bs, pupil_psd
        active, each_file, each_passive, each_half
        far, hit, lick, p_x_a

    TODO: Migrate to nems_lbhb or make a more generic version
    """

    newrec = rec.copy()
    resp = newrec['resp'].rasterize()
    state_signals = state_signals.copy()
    permute_signals = permute_signals.copy()
    generate_signals = generate_signals.copy()

    # normalize mean/std of pupil trace if being used
    if any([s.startswith('pupil') for s in state_signals]):
        # save raw pupil trace
        # normalize min-max
        p_raw = newrec["pupil"].as_continuous().copy()
        # p[p < np.nanmax(p)/5] = np.nanmax(p)/5
        # norm to mean 0, variance 1
        p = p_raw - np.nanmean(p_raw)
        p /= np.nanstd(p)
        # hack to make sure state signal matches size of resp
        if 'resp' in newrec.signals.keys():
            #import pdb;pdb.set_trace()
            if p.shape[1] > newrec['resp'].shape[1]:
                p = p[:, :newrec['resp'].shape[1]]
                p_raw = p_raw[:, :newrec['resp'].shape[1]]
        newrec["pupil"] = newrec["pupil"]._modified_copy(p)
        newrec["pupil_raw"] = newrec["pupil"]._modified_copy(p_raw)

        if any([s.startswith('pupil') for s in state_signals]):
            # save raw pupil trace
            # normalize min-max
            p_raw = newrec["pupil"].as_continuous().copy()
            # p[p < np.nanmax(p)/5] = np.nanmax(p)/5
            # norm to mean 0, variance 1
            p = p_raw - np.nanmean(p_raw)
            p /= np.nanstd(p)
            # hack to make sure state signal matches size of resp
            if 'resp' in newrec.signals.keys():
                # import pdb;pdb.set_trace()
                if p.shape[1] > newrec['resp'].shape[1]:
                    p = p[:, :newrec['resp'].shape[1]]
                    p_raw = p_raw[:, :newrec['resp'].shape[1]]
            newrec["pupil"] = newrec["pupil"]._modified_copy(p)
            newrec["pupil_raw"] = newrec["pupil"]._modified_copy(p_raw)

        if 'pupiln' in state_signals:
            log.info('norm pupil min/max = 0/1')
            p_raw = newrec["pupil_raw"].as_continuous().copy()
            p = p_raw - np.nanmin(p_raw)
            p /= np.nanmax(p)
            newrec["pupiln"] = newrec["pupil"]._modified_copy(p)

        for state_signal in [s for s in state_signals if s.startswith('pupil_r')]:
            # copy repetitions of pupil
            newrec[state_signal] = newrec["pupil"]._modified_copy(newrec['pupil']._data)
            newrec[state_signal].chans = [state_signal]
        for state_signal in [s for s in state_signals if s.startswith('pupiln_r')]:
            # copy repetitions of pupil
            newrec[state_signal] = newrec["pupiln"]._modified_copy(newrec['pupiln']._data)
            newrec[state_signal].chans = [state_signal]
        if ('pupil2') in state_signals:
            newrec["pupil2"] = newrec["pupil"]._modified_copy(p ** 2)
            newrec["pupil2"].chans = ['pupil2']
        if ('pupil_dup') in state_signals:
            newrec['pupil_dup']=newrec["pupil"].copy()
            newrec["pupil_dup"].chans = ['pupil_dup']
        if ('pupil_dup2') in state_signals:
            newrec['pupil_dup2']=newrec["pupil"].copy()
            newrec["pupil_dup2"].chans = ['pupil_dup2']

    if ('pupil_psd') in state_signals:
        pup = newrec['pupil'].as_continuous().copy()
        fs = newrec['pupil'].fs
        # get spectrogram of pupil
        nperseg = int(60*fs)
        noverlap = nperseg-1
        f, time, Sxx = ss.spectrogram(pup.squeeze(), fs=fs, nperseg=nperseg,
                                      noverlap=noverlap)
        max_chan = 4 # (np.abs(f - 0.1)).argmin()
        # Keep only first five channels of spectrogram
        #f = interpolate.interp1d(np.arange(0, Sxx.shape[1]), Sxx[:max_chan, :], axis=1)
        #newspec = f(np.linspace(0, Sxx.shape[-1]-1, pup.shape[-1]))
        pad1 = np.ones((max_chan,int(nperseg/2)))*Sxx[:max_chan,[0]]
        pad2 = np.ones((max_chan,int(nperseg/2-1)))*Sxx[:max_chan,[-1]]
        newspec = np.concatenate((pad1,Sxx[:max_chan, :],pad2), axis=1)

        # = np.concatenate((Sxx[:max_chan, :], np.tile(Sxx[:max_chan,-1][:, np.newaxis], [1, noverlap])), axis=1)
        newspec -= np.nanmean(newspec, axis=1, keepdims=True)
        newspec /= np.nanstd(newspec, axis=1, keepdims=True)

        spec_signal = newrec['pupil']._modified_copy(newspec)
        spec_signal.name = 'pupil_psd'
        chan_names = []
        for chan in range(0, newspec.shape[0]):
            chan_names.append('puppsd{0}'.format(chan))
        spec_signal.chans = chan_names

        newrec.add_signal(spec_signal)

    if ('pupil_ev' in state_signals) or ('pupil_bs' in state_signals):
        # generate separate pupil baseline and evoked signals

        prestimsilence = newrec["pupil"].extract_epoch('PreStimSilence')
        spont_bins = prestimsilence.shape[2]
        pupil_trial = newrec["pupil"].extract_epoch('TRIAL')

        pupil_bs = np.zeros(pupil_trial.shape)
        for ii in range(pupil_trial.shape[0]):
            pupil_bs[ii, :, :] = np.mean(
                    pupil_trial[ii, :, :spont_bins])
        pupil_ev = pupil_trial - pupil_bs

        newrec['pupil_ev'] = newrec["pupil"].replace_epoch('TRIAL', pupil_ev)
        newrec['pupil_ev'].chans=['pupil_ev']
        newrec['pupil_bs'] = newrec["pupil"].replace_epoch('TRIAL', pupil_bs)
        newrec['pupil_bs'].chans=['pupil_bs']

    # normalize mean/std of pupil trace if being used
    if any([s.startswith('facepca') for s in state_signals]):
        # save raw facepca trace
        # normalize min-max
        p_raw = newrec["facepca"].as_continuous().copy()
        # p[p < np.nanmax(p)/5] = np.nanmax(p)/5
        # norm to mean 0, variance 1
        p = p_raw - np.nanmean(p_raw, axis=1, keepdims=True)
        p /= np.nanstd(p, axis=1, keepdims=True)
        # hack to make sure state signal matches size of resp
        newrec["facepca"] = newrec["facepca"]._modified_copy(p)
        newrec["facepca_raw"] = newrec["facepca"]._modified_copy(p_raw)

    if ('each_passive' in state_signals):
        file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")
        pset = []
        found_passive1 = False
        for f in file_epochs:
            # test if passive expt
            epoch_indices = ep.epoch_intersection(
                    resp.get_epoch_indices(f),
                    resp.get_epoch_indices('PASSIVE_EXPERIMENT'))
            if epoch_indices.size:
                if not(found_passive1):
                    # skip first passive
                    found_passive1 = True
                else:
                    pset.append(f)
                    newrec[f] = resp.epoch_to_signal(f)
        state_signals.remove('each_passive')
        state_signals.extend(pset)
        if 'each_passive' in permute_signals:
            permute_signals.remove('each_passive')
            permute_signals.extend(pset)

    if ('each_file' in state_signals) or ('each_active' in state_signals):
        file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")
        trial_indices = resp.get_epoch_indices('TRIAL')
        passive_indices = resp.get_epoch_indices('PASSIVE_EXPERIMENT')
        pset = []
        psetx = []
        pcount = 0
        acount = 0
        # pupil interactions

        for f in file_epochs:
            # test if passive expt
            f_indices = resp.get_epoch_indices(f, mask=newrec['mask'])

            epoch_indices = ep.epoch_intersection(f_indices, passive_indices)
            added_signal = False
            if not f_indices.size:
                log.info("Skipping file %s because empty after masking", f)
            elif epoch_indices.size:
                # this is a passive file
                name1 = "PASSIVE_{}".format(pcount)
                pcount += 1
                if pcount == 1:
                    acount = 1  # reset acount for actives after first passive
                else:
                    # use first passive part A as baseline - don't model
                    if ('each_file' in state_signals):
                        pset.append(name1)
                        newrec[name1] = resp.epoch_to_signal(name1, indices=f_indices)
                        added_signal = True
            else:
                name1 = "ACTIVE_{}".format(acount)
                pset.append(name1)
                newrec[name1] = resp.epoch_to_signal(name1, indices=f_indices)
                added_signal = True
                if pcount == 0:
                    acount -= 1
                else:
                    acount += 1
            if ('p_x_f' in state_signals) and added_signal:
                if name1.startswith('ACTIVE') | ('each_file' in state_signals):
                    p = newrec["pupil"].as_continuous()
                    a = newrec[name1].as_continuous()
                    name1x = name1+'Xpup'
                    newrec[name1x] = newrec["pupil"]._modified_copy(p * a)
                    newrec[name1x].chans = [name1x]
                    psetx.append(name1x)

            # test if passive expt
#            epoch_indices = ep.epoch_intersection(
#                    resp.get_epoch_indices(f),
#                    resp.get_epoch_indices('PASSIVE_EXPERIMENT'))
#            if epoch_indices.size and not(found_passive1):
#                # skip first passive
#                found_passive1 = True
#            else:
#                pset.append(f)
#                newrec[f] = resp.epoch_to_signal(f)

        if 'each_file' in state_signals:
            state_signals.remove('each_file')
            state_signals.extend(pset)
        if 'each_active' in state_signals:
            state_signals.remove('each_active')
            state_signals.extend(pset)
        if 'each_file' in permute_signals:
            permute_signals.remove('each_file')
            permute_signals.extend(pset)
        if 'each_active' in permute_signals:
            permute_signals.remove('each_active')
            permute_signals.extend(pset)

        # add interactions to state list if specified
        if ('p_x_f' in state_signals):
            state_signals.remove('p_x_f')
            state_signals.extend(psetx)
            if 'p_x_f' in permute_signals:
                permute_signals.remove('p_x_f')
                permute_signals.extend(psetx)

    if ('each_half' in state_signals):
        file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")
        trial_indices = resp.get_epoch_indices('TRIAL')
        passive_indices = resp.get_epoch_indices('PASSIVE_EXPERIMENT')
        pset = []
        pcount = 0
        acount = 0
        for f in file_epochs:
            # test if passive expt
            f_indices = resp.get_epoch_indices(f)
            epoch_indices = ep.epoch_intersection(f_indices, passive_indices)
            trial_intersect = ep.epoch_intersection(f_indices, trial_indices)
            #trial_count = trial_intersect.shape[0]
            #_split = int(trial_count/2)
            _t1=trial_intersect[0,0]
            _t2=trial_intersect[-1,1]
            _split = int((_t1+_t2)/2)
            epoch1 = np.array([[_t1,_split]])
            epoch2 = np.array([[_split,_t2]])

            if epoch_indices.size:
                # this is a passive file
                name1 = "PASSIVE_{}_{}".format(pcount, 'A')
                name2 = "PASSIVE_{}_{}".format(pcount, 'B')
                pcount += 1
                if pcount == 1:
                    acount = 1 # reset acount for actives after first passive
                else:
                    # don't model PASSIVE_0 A -- baseline
                    pset.append(name1)
                    newrec[name1] = resp.epoch_to_signal(name1, indices=epoch1)

                # do include part B
                pset.append(name2)
                newrec[name2] = resp.epoch_to_signal(name2, indices=epoch2)
            else:
                name1 = "ACTIVE_{}_{}".format(acount, 'A')
                name2 = "ACTIVE_{}_{}".format(acount, 'B')
                pset.append(name1)
                newrec[name1] = resp.epoch_to_signal(name1, indices=epoch1)
                pset.append(name2)
                newrec[name2] = resp.epoch_to_signal(name2, indices=epoch2)

                if pcount == 0:
                    acount -= 1
                else:
                    acount += 1

        state_signals.remove('each_half')
        state_signals.extend(pset)
        if 'each_half' in permute_signals:
            permute_signals.remove('each_half')
            permute_signals.extend(pset)

    # generate task state signals
    if 'pas' in state_signals:
        fpre = (resp.epochs['name'] == "PRE_PASSIVE")
        fpost = (resp.epochs['name'] == "POST_PASSIVE")
        INCLUDE_PRE_POST = (np.sum(fpre) > 0) & (np.sum(fpost) > 0)
        if INCLUDE_PRE_POST:
            # only include pre-passive if post-passive also exists
            # otherwise the regression gets screwed up
            newrec['pre_passive'] = resp.epoch_to_signal('PRE_PASSIVE')
        else:
            # place-holder, all zeros
            newrec['pre_passive'] = resp.epoch_to_signal('XXX')
            newrec['pre_passive'].chans = ['PRE_PASSIVE']
    if 'puretone_trials' in state_signals:
        newrec['puretone_trials'] = resp.epoch_to_signal('PURETONE_BEHAVIOR')
        newrec['puretone_trials'].chans = ['puretone_trials']
    if 'easy_trials' in state_signals:
        newrec['easy_trials'] = resp.epoch_to_signal('EASY_BEHAVIOR')
        newrec['easy_trials'].chans = ['easy_trials']
    if 'hard_trials' in state_signals:
        newrec['hard_trials'] = resp.epoch_to_signal('HARD_BEHAVIOR')
        newrec['hard_trials'].chans = ['hard_trials']
    if ('active' in state_signals) or ('far' in state_signals):
        newrec['active'] = resp.epoch_to_signal('ACTIVE_EXPERIMENT')
        newrec['active'].chans = ['active']
    if (('hit_trials' in state_signals) or ('miss_trials' in state_signals) or
        ('far' in state_signals) or ('hit' in state_signals)):
        newrec['hit_trials'] = resp.epoch_to_signal('HIT_TRIAL')
        newrec['miss_trials'] = resp.epoch_to_signal('MISS_TRIAL')
        newrec['fa_trials'] = resp.epoch_to_signal('FA_TRIAL')

    sm_len = int(sm_win_len * newrec['resp'].fs)
    if 'far' in state_signals:
        log.info('FAR: sm_win_len=%.0f sm_len=%d', sm_win_len, sm_len)
        a = newrec['active'].as_continuous()
        fa = newrec['fa_trials'].as_continuous().astype(float)
        #c = np.concatenate((np.zeros((1,sm_len)), np.ones((1,sm_len+1))),
        #                   axis=1)
        c = np.ones((1,sm_len))/sm_len

        fa = convolve2d(fa, c, mode='same')
        fa[a] -= 0.25 # np.nanmean(fa[a])
        fa[np.logical_not(a)] = 0

        s = newrec['fa_trials']._modified_copy(fa)
        s.chans = ['far']
        s.name='far'
        newrec.add_signal(s)

    if 'hit' in state_signals:
        a = newrec['active'].as_continuous()
        hr = newrec['hit_trials'].as_continuous().astype(float)
        ms = newrec['miss_trials'].as_continuous().astype(float)
        ht = hr-ms

        c = np.ones((1,sm_len))/sm_len

        ht = convolve2d(ht, c, mode='same')
        ht[a] -= 0.1  # np.nanmean(ht[a])
        ht[np.logical_not(a)] = 0

        s = newrec['hit_trials']._modified_copy(ht)
        s.chans = ['hit']
        s.name='hit'
        newrec.add_signal(s)

    if 'lick' in state_signals:
        newrec['lick'] = resp.epoch_to_signal('LICK')

    # pupil interactions
    if ('p_x_a' in state_signals):
        # normalize min-max
        p = newrec["pupil"].as_continuous()
        a = newrec["active"].as_continuous()
        newrec["p_x_a"] = newrec["pupil"]._modified_copy(p * a)
        newrec["p_x_a"].chans = ["p_x_a"]

    if ('pupil_x_population' in state_signals):
        # normalize min-max
        p = newrec["pupil"].as_continuous().copy()
        p -= np.mean(p, axis=1, keepdims=True)
        a = newrec["population"].as_continuous().copy()
        a -= np.mean(a, axis=1, keepdims=True)
        newrec["pupil_x_population"] = newrec["population"]._modified_copy(p * a)
        newrec["pupil_x_population"].chans = ["px"+c for c in newrec["pupil_x_population"].chans]

    if ('active_x_population' in state_signals):
        # normalize min-max
        a = newrec["active"].as_continuous().astype(float)
        a -= np.mean(a, axis=1, keepdims=True)
        p = newrec["population"].as_continuous().copy()
        p -= np.mean(p, axis=1, keepdims=True)
        newrec["active_x_population"] = newrec["population"]._modified_copy(p * a)
        newrec["active_x_population"].chans = ["ax"+c for c in newrec["active_x_population"].chans]

    if ('prw' in state_signals):
        # add channel two of the resp to state and delete it from resp
        if len(rec['resp'].chans) != 2:
            raise ValueError("this is for pairwise fitting")
        else:
            ch2 = rec['resp'].chans[1]
            ch1 = rec['resp'].chans[0]

        newrec['prw'] = newrec['resp'].extract_channels([ch2]).rasterize()
        newrec['resp'] = newrec['resp'].extract_channels([ch1]).rasterize()

    if ('pup_x_prw' in state_signals):
        # interaction term between pupil and the other cell
        if 'prw' not in newrec.signals.keys():
            raise ValueError("Must include prw alone before using interaction")

        else:
            pup = newrec['pupil']._data
            prw = newrec['prw']._data
            sig = newrec['pupil']._modified_copy(pup * prw)
            sig.name = 'pup_x_prw'
            sig.chans = ['pup_x_prw']
            newrec.add_signal(sig)

    if 'drift' in state_signals:
        resp_len = rec['resp'].shape[1]
        drift = np.reshape(np.linspace(0,1,resp_len), (1, -1))
        _s = nems0.signal.RasterizedSignal(fs=rec['resp'].fs, data=drift, name="drift",
                                          recording=rec['resp'].recording, chans=["drift"], epochs=rec['resp'].epochs)
        newrec.add_signal(_s)

    # delete any pre-existing state signal. Is this a good idea??
    if new_signalname in newrec.signals.keys():
        log.info("Deleting existing %s signal before generating new one", new_signalname)
        del newrec.signals[new_signalname]

    for i, x in enumerate(state_signals):
        if x.startswith("dummy"):
            s = rec['resp'].shape[1]
            d_data = np.random.uniform(size=(1, s))
            _s = nems0.signal.RasterizedSignal(fs=rec['resp'].fs, data=d_data, name=x,
                                               recording=rec['resp'].recording, chans=[x],
                                               epochs=rec['resp'].epochs)
            newrec.add_signal(_s)

        if x in permute_signals:
            # kludge: fix random seed to index of state signal in list
            # this avoids using the same seed for each shuffled signal
            # but also makes shuffling reproducible
            newrec = concatenate_state_channel(
                    newrec, newrec[x].shuffle_time(rand_seed=i,
                                  mask=newrec['mask']),
                    state_signal_name=new_signalname)
        elif x in generate_signals:
            # fit a gaussian process to the signal, then generate a new signal using the fit
            newrec = concatenate_state_channel(
                    newrec, _generate_gp(newrec[x], rand_seed=i), 
                    state_signal_name=new_signalname)
        else:
            newrec = concatenate_state_channel(
                    newrec, newrec[x], state_signal_name=new_signalname)

        newrec = concatenate_state_channel(
                newrec, newrec[x], state_signal_name=new_signalname+"_raw")
                
    return newrec


def _generate_gp(signal, rand_seed):
    from sklearn.gaussian_process import GaussianProcessRegressor
    import sklearn.gaussian_process.kernels as k
    cur_state = np.random.get_state()
    np.random.seed(rand_seed)
    data = signal._data
    # fit signal
    log.info(f"Fitting RBF kernel to signal")
    kernel = k.RBF(length_scale=5, length_scale_bounds=(1, 100))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(np.atleast_2d(np.linspace(0, data.shape[1], data.shape[1])).T, data.T)

    log.info(f"RBF kernel length: {gp.kernel_.length_scale}")

    # general new signal
    def rbf_kernel(a, b, length=1):
        square_distance = np.sum((a - b) ** 2)
        return np.exp(-square_distance / (2 * (length**2)))
    log.info("Generating a new signal using GP fit")
    N = data.shape[1]
    cov = np.zeros((N, N))
    for ii in range(N):
        for jj in range(N):
            cov[ii, jj] = rbf_kernel(ii, jj, length=gp.kernel_.length_scale)

    # generate random pupil
    newsig = np.random.multivariate_normal(np.zeros(N), cov, (1,))
    
    # return state
    np.random.set_state(cur_state)

    return signal._modified_copy(newsig)
    


def concatenate_state_channel(rec, sig, state_signal_name='state', generate_baseline=True):

    newrec = rec.copy()

    if (state_signal_name not in rec.signals.keys()):
        if generate_baseline:
            # create an initial state signal of all ones for baseline
            x = np.ones([1, sig.shape[1]])
            ones_sig = sig._modified_copy(x)
            ones_sig.name = "baseline"
            ones_sig.chans = ["baseline"]
    
            state_sig_list = [ones_sig]
        else:
            state_sig_list=[]
    else:
        # start with existing state signal
        state_sig_list = [rec[state_signal_name]]

    state_sig_list.append(sig)

    state = nems0.signal.RasterizedSignal.concatenate_channels(state_sig_list)
    state.name = state_signal_name

    newrec.add_signal(state)

    return newrec


def concatenate_input_channels(rec, input_signals=[], input_name=None, **kwargs):
    newrec = rec.copy()
    input_sig_list = []
    for s in input_signals:
        input_sig_list.append(newrec[s].rasterize())
    input_sig_list.append(newrec[input_name].rasterize())
    log.info(f"Concatenating {input_signals} onto {input_name}")
    input = nems0.signal.RasterizedSignal.concatenate_channels(input_sig_list)
    input.name = input_name

    newrec.add_signal(input)

    return newrec


def add_noise_signal(rec, n_chans=None, T=None, noise_name="indep", ref_signal="resp", chans=None, 
                     rep_count=1, rand_seed=1, distribution="gaussian", sm_win=0, est=None, val=None, **context):
    
    newrec = rec.copy()
    
    if rep_count>1:
        # duplicate signals rep_count times in time to get more variety in the noise signal
    
        for k,s in rec.signals.items():
            newrec.signals[k]=s.concatenate_time([s]*rep_count)
            log.info(f"concat {k}x{rep_count} len: {s.shape[1]} to {newrec.signals[k].shape[1]}")
    if n_chans is None:
        n_chans=newrec[ref_signal].shape[0]
    if chans is None:
        chans = newrec[ref_signal].chans.copy()
    if T is None:
        T = newrec[ref_signal].shape[1]
    
    # set seed to produce frozen noise
    save_state = np.random.get_state()
    rseed=rand_seed+n_chans+T
    np.random.seed(rseed)
    log.info(f"add_noise_signal: name={noise_name} using seed {rseed}")
    if distribution=='gaussian':
        d = np.random.randn(n_chans,T)
    elif distribution=='uniform':
        d = np.random.uniform(size=(n_chans,T))
    else:
        raise ValueError(f"unknown distribution {distribution}")
    
    if sm_win>1:
        box = np.ones((1,sm_win))/sm_win
        d = convolve2d(d, box, mode='same')

    # restore random state
    np.random.set_state(save_state)

    newrec.add_signal(newrec['resp']._modified_copy(data=d, name=noise_name, chans=chans))
    d = {'rec': newrec}
    if est is not None:
        est = est.copy()
        est.add_signal(newrec[noise_name])
        d['est']=est
    if val is not None:
        val = val.copy()
        val.add_signal(newrec[noise_name])
        d['val']=val

    return d


#
# SELECT SUBSETS OF THE DATA
#

def signal_select_channels(rec, sig_name="resp", chans=None):

    newrec = rec.copy()
    if chans is None:
        return newrec

    s = newrec[sig_name].rasterize()
    s = s.extract_channels(chans)
    newrec[sig_name] = s

    return newrec


def split_est_val_for_jackknife(rec, epoch_name='TRIAL', modelspecs=None,
                                njacks=10, IsReload=False, **context):
    """
    take a single recording (est) and define njacks est/val sets using a
    jackknife logic. returns lists est_out and val_out of corresponding
    jackknife subsamples. removed timepoints are replaced with nan
    """
    est = []
    val = []
    # logging.info("Generating {} jackknifes".format(njacks))

    for i in range(njacks):
        # est_out += [est.jackknife_by_time(njacks, i)]
        # val_out += [est.jackknife_by_time(njacks, i, invert=True)]
        est += [rec.jackknife_by_epoch(njacks, i, epoch_name,
                                       tiled=True)]
        val += [rec.jackknife_by_epoch(njacks, i, epoch_name,
                                       tiled=True, invert=True)]

    modelspecs_out = []
    if (not IsReload) and (modelspecs is not None):
        if len(modelspecs) == 1:
            modelspecs_out = [copy.deepcopy(modelspecs[0])
                              for i in range(njacks)]
        elif len(modelspecs) == njacks:
            # assume modelspecs already generated for njacks
            modelspecs_out = modelspecs
        else:
            raise ValueError('modelspecs must be len 1 or njacks')

    return est, val, modelspecs_out


def mask_est_val_for_jackknife(rec, epoch_name='TRIAL', epoch_regex=None,
                               modelspec=None,
                               njacks=10, allow_partial_epochs=False,
                               IsReload=False, **context):
    """
    take a single recording (est) and define njacks est/val sets using a
    jackknife logic. returns lists est_out and val_out of corresponding
    jackknife subsamples. removed timepoints are replaced with nan
    """
    if epoch_regex is None:
        epochs_to_extract = [epoch_name]
    else:
        log.info('jackknife epoch_regex=%s', epoch_regex)
        epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)
    log.info(epochs_to_extract)
    # logging.info("Generating {} jackknifes".format(njacks))
    if rec.get_epoch_indices(epochs_to_extract, allow_partial_epochs=allow_partial_epochs).shape[0]:
        pass
    elif rec.get_epoch_indices('REFERENCE', allow_partial_epochs=allow_partial_epochs).shape[0]:
        log.info('Jackknifing by REFERENCE epochs')
        epoch_name = 'REFERENCE'
    elif rec.get_epoch_indices('TARGET', allow_partial_epochs=allow_partial_epochs).shape[0]:
        log.info('Jackknifing by TARGET epochs')
        epoch_name = 'TARGET'
    elif rec.get_epoch_indices('TRIAL', allow_partial_epochs=allow_partial_epochs).shape[0]:
        log.info('Jackknifing by TRIAL epochs')
        epoch_name = 'TRIAL'
    else:
        raise ValueError('No epochs matching '+epoch_name)

    est = rec.jackknife_masks_by_epoch(njacks, epochs_to_extract, tiled=True,
                                       allow_partial_epochs=allow_partial_epochs)
    val = rec.jackknife_masks_by_epoch(njacks, epochs_to_extract,
                                       tiled=True, invert=True,
                                       allow_partial_epochs=allow_partial_epochs)
    #import pdb;pdb.set_trace()

    if (not IsReload) and (modelspec is not None):
        if modelspec.jack_count == 1:
            modelspec_out = modelspec.tile_jacks(njacks)
        elif modelspec.jack_count == njacks:
            # assume modelspec already generated for njacks
            modelspec_out = modelspec
        else:
            raise ValueError('modelspec.jack_count must be 1 or njacks')
    else:
        modelspec_out = modelspec

    return est, val, modelspec_out


def mask_est_val_for_jackknife_by_time(rec, modelspec=None,
                                       njacks=10, IsReload=False, **context):
    """
    take a single recording (est) and define njacks est/val sets using a
    jackknife logic. returns lists est_out and val_out of corresponding
    jackknife subsamples. removed timepoints are replaced with nan
    """
    #est = []
    #val = []
    #for i in range(njacks):
    #    est += [rec.jackknife_mask_by_time(njacks, i, tiled=True)]
    #    val += [rec.jackknife_mask_by_time(njacks, i, tiled=True, invert=True)]

    est = rec.jackknife_masks_by_time(njacks, tiled=True)
    val = rec.jackknife_masks_by_time(njacks, tiled=True, invert=True)

    modelspec_out = []
    if (not IsReload) and (modelspec is not None):
        if modelspec.jack_count == 1:
            modelspec_out = modelspec.tile_jacks(njacks)
        elif modelspec.jack_count == njacks:
            # assume modelspec already generated for njacks
            modelspec_out = modelspec
        else:
            raise ValueError('modelspec.jack_count must be 1 or njacks')

    return est, val, modelspec_out

def shuffle(sigs, recs=['est','val'], ** context):  
    for r in recs:
        for i,sig in enumerate(sigs):
            context[r][sig]=context[r][sig].shuffle_time(rand_seed=i,mask=context[r]['mask'])
    
    return context
    