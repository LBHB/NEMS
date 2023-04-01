import io
import os
import logging
import json
import re
import math
import copy
import tempfile
import warnings

import pandas as pd
import numpy as np
import h5py

from nems0.epoch import (remove_overlap, merge_epoch, epoch_contained,
                        epoch_intersection, epoch_names_matching)

log = logging.getLogger(__name__)


################################################################################
# Utility methods
################################################################################
def _string_syntax_valid(s):
    '''
    Returns True iff the string is valid for use in signal names,
    recording names, or channel names. Else False.
    '''
    disallowed = re.compile(r'[^a-zA-Z0-9_\-]')
    match = disallowed.match(s)
    if match:
        return False
    else:
        return True


################################################################################
# Indexing support
################################################################################
class BaseSignalIndexer:

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) > 2:
            raise IndexError('Cannot add dimensions')

        # Parse the slice into the channel and time portions
        if isinstance(index, tuple):
            c_slice, t_slice = index
        else:
            c_slice = index
            t_slice = slice(None)

        # Make sure the channel and time slices are slices.
        # If they're integers,
        # that dimension will be dropped. We don't want that as we need to
        # preserve dimensionality to create a new Signal object,
        # so the parsers
        # need to return the appropriate dimensionality-preserving slice.
        c_slice = self._parse_c_slice(c_slice)
        t_slice = self._parse_t_slice(t_slice)

        data = self.obj.as_continuous()[c_slice, t_slice]
        chans = np.array(self.obj.chans)[c_slice].tolist()
        kwattrs = {'chans': chans}
        if t_slice.start is not None:
            kwattrs['t0'] = t_slice.start/self.obj.fs
        return self.obj._modified_copy(data, **kwattrs)

    def _parse_c_slice(self, c_slice):
        raise NotImplementedError

    def _parse_t_slice(self, t_slice):
        raise NotImplementedError


class SimpleSignalIndexer(BaseSignalIndexer):
    '''
    Index-based signal indexer that supports selecting channels and timepoints
    by index
    '''
    def _parse_c_slice(self, c_slice):
        if isinstance(c_slice, int):
            c_slice = slice(c_slice, c_slice+1)
        return c_slice

    def _parse_t_slice(self, t_slice):
        if isinstance(t_slice, int):
            t_slice = slice(t_slice, t_slice+1)
        elif isinstance(t_slice, list):
            m = 'List-based selection on time-based indexers not supported'
            raise IndexError(m)
        return t_slice


class LabelSignalIndexer(BaseSignalIndexer):
    '''
    Label-based signal indexer that supports selecting by named channels and
    time (in seconds).
    '''

    def _parse_t_slice(self, t_slice):
        if isinstance(t_slice, slice):
            if t_slice.step is not None:
                m = 'Strides on time-based indexers not supported'
                raise IndexError(m)
            start = None if t_slice.start is None else \
                int(t_slice.start*self.obj.fs)
            stop = None if t_slice.stop is None else \
                int(t_slice.stop*self.obj.fs)
            return slice(start, stop)
        elif isinstance(t_slice, (float, int)):
            t = int(t_slice*self.obj.fs)
            return slice(t, t+1)
        elif isinstance(t_slice, list):
            m = 'List-based selection on time-based indexers not supported'
            raise IndexError(m)

    def _parse_c_slice(self, c_slice):
        if isinstance(c_slice, slice):
            if c_slice.step is not None:
                m = 'Strides on label-based indexers not supported'
                raise IndexError(m)
            start = None if c_slice.start is None else \
                self.obj.chans.index(c_slice.start)
            # Note that we add 1 to the label-based index becasue we're
            # duplicating Pandas loc behavior (where the end of a label
            # slice is included).
            stop = None if c_slice.stop is None else \
                self.obj.chans.index(c_slice.stop)+1
            return slice(start, stop)
        elif isinstance(c_slice, str):
            return [self.obj.chans.index(c_slice)]
        elif isinstance(c_slice, list):
            return [self.obj.chans.index(c) for c in c_slice]
        else:
            raise IndexError('Slice not understood')


################################################################################
# Signals
################################################################################
class SignalBase:

    def __init__(self, fs, data, name, recording, chans=None, epochs=None,
                 segments=None, meta=None, safety_checks=True,
                 normalization='none', norm_baseline=np.array([[0]]),
                 norm_gain=np.array([[1]]), **other_attributes):
        '''
        Parameters
        ----------
        ... TODO
        chans : list of labels for each channel (string required?)
        epochs : {None, DataFrame}
            Epochs are periods of time that are tagged with a name
            When defined, the DataFrame should have these first three columns:
                 ('start', 'end', 'name')
            denoting the start and end of the time of an epoch (in seconds).
            You may use the same epoch name multiple times; this is common when
            tagging epochs that correspond to occurrences of the same stimulus.
        segments : {None, 2 x M ndarray}
            start and stop time windows to cut out of data matrix
            designed for subsampling existing signals
        meta : {dict, None}
            arbitrary metadata, only JSON-compatible types plus nparrays are
            supported if signal is to be saved
        safety_checks : {bool, True}
            TODO: clarify what this does

        normalization : { string, 'none' }
        norm_baseline : { string, 'none' }
        norm_gain : { string, 'none' }
        signal_type: { string, 'none' }
            passthrough metadata that should be preserved if the signal is copied.

        '''
        self.fs = fs
        self.name = name
        self.recording = recording
        self.chans = chans
        self.n_extradims = []
        self.epochs = epochs
        self.meta = meta
        self.signal_type = str(type(self))
        self._data = data
        self.normalization = normalization
        self.norm_baseline = norm_baseline
        self.norm_gain = norm_gain

        if epochs is not None:
            max_epoch_time = self.epochs["end"].max()
        else:
            max_epoch_time = 0
        if isinstance(data, dict):
            # max_event_times = [max(et) for et in self._data.values()]
            max_event_times = [0]
        else:
            max_event_times = [data.shape[1] / fs]
        max_time = max(max_epoch_time, *max_event_times)

        # svd - kludge warning! rounding to prevent accidental increase in duration
        # due to floating point limits
        self.ntimes = np.int(np.round(fs*max_time))

        if segments is None:
            segments = np.array([[0, self.ntimes]])
        self.segments = segments

        if epochs is None:
            self.add_epoch("SIGNAL",np.array([[0, self.ntimes/self.fs]]))

        if safety_checks:
            self._run_safety_checks()

    def _run_safety_checks(self):
        """
        Additional subclass-specific checks can be added by
        redefiing _run_safety_checks in that class and calling
        super()._run_safety_checks() within that function.
        """
        if not isinstance(self.name, str):
            m = 'Name of signal must be a string: {}'.format(self.name)
            raise ValueError(m)

        if not isinstance(self.recording, str):
            m = 'Name of recording must be a string: {}'.format(self.recording)
            raise ValueError(m)

        if self.chans:
            if type(self.chans) is not list:
                raise ValueError('Chans must be a list.')
            typesok = [(True if type(c) is str else False) for c in self.chans]
            if not all(typesok):
                raise ValueError('Chans must be a list of strings:' +
                                 str(self.chans) + str(typesok))
            # Test that channel names use only
            # lowercase letters and numbers 0-9
            for s in self.chans:
                if s and not _string_syntax_valid(s):
                    raise ValueError("Disallowed characters in: {0}\n"
                                     .format(s))

        # Test that other names use only lowercase letters and numbers 0-9
        for s in [self.name, self.recording]:
            if s and not _string_syntax_valid(s):
                raise ValueError("Disallowed characters in: {0}\n"
                                 .format(s))

        if self.fs < 0:
            m = 'Sampling rate of signal must be a positive number. Got {}.'
            raise ValueError(m.format(self.fs))

        # not implemented yet in epoch.py -- 2/4/2018
        # verify_epoch_integrity(self.epochs)

    ##
    ## I/O method(s)
    ##
    def _save_metadata(self, epoch_fh, md_fh, fmt='%.18e'):
        '''
        Save this signal to a CSV file + JSON sidecar. If desired,
        you may use optional parameter fmt (for example, fmt='%1.3e')
        to alter the precision of the floating point matrices.
        '''

        self.epochs.to_csv(epoch_fh, sep=',', index=False)
        attributes = self._get_attributes()
        del attributes['epochs']
        attributes['segments'] = attributes['segments'].tolist()
        attributes['norm_baseline'] = attributes['norm_baseline'].tolist()
        attributes['norm_gain'] = attributes['norm_gain'].tolist()
        json.dump(attributes, md_fh)

    def _save_metadata_to_dirpath(self, dirpath, fmt='%.18e'):
        # create files
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath, mode=0o0777)

        filebase = self.recording + '.' + self.name
        basepath = os.path.join(dirpath, filebase)
        jsonfilepath = basepath + '.json'
        epochfilepath = basepath + '.epoch.csv'
        with open(jsonfilepath, 'w') as md_fh, open(epochfilepath, 'w') as epoch_fh:
            self._save_metadata(epoch_fh, md_fh, fmt)
        return (jsonfilepath, epochfilepath)

    def _save_data_to_h5(self, dirpath):

        if not os.path.isdir(dirpath):
            os.makedirs(dirpath, mode=0o0777)

        filebase = self.recording + '.' + self.name
        basepath = os.path.join(dirpath, filebase)
        hdf5filepath = basepath + '.h5'

        with h5py.File(hdf5filepath, 'a') as f:
            # TODO: any other attrs we should save?
            for key, array in self._data.items():
                f.create_dataset(key, data=array, compression='gzip')

        return hdf5filepath

    def __getitem__(self, key):
        """Overloaded get.

        :return: _data indexed to key
        """
        return self._data[key]

    def __repr__(self):
        """Overloaded repr.

        :return: Repr of the modelspec `_data` property.
        """
        return repr(self._data)

    def copy(self):
        '''
        Returns a shallow copy of this signal. _data matrix is not copied.
        '''
        return copy.copy(self)

    def get_epoch_bounds(self, epoch, boundary_mode='exclude',
                         fix_overlap=None, overlapping_epoch=None, mask=None):
        '''
        Get boundaries of named epoch.

        Parameters
        ----------
        epoch : {string, list of strings, Nx2 array, pandas series}
            If string, name of epoch (as stored in internal dataframe) to
            extract.
            If list of strings, concatenate all epochs in list
            If Nx2 array, the first column indicates the start time (in
            seconds) and the second column indicates the end time (in seconds)
            to extract.
            If pandas series, pull out bounds of epochs where epoch==True
        boundary_mode : {'exclude', 'trim'}
            If 'exclude', discard all epochs where the boundaries are not fully
            contained within the range of the signal. If 'trim', epochs with
            boundaries falling outside the signal range will be truncated. For
            example, if an epoch runs from -1.5 to 10, it will be truncated to 0
            to 10.
        fix_overlap : {None, 'merge', 'first'}
            Indicates how to handle overlapping epochs. If None, return
            boundaries as-is. If 'merge', merge overlapping epochs into a single
            epoch. If 'first', keep only the first of an overlapping set of
            epochs.
        overlapping_epoch : {None, or string}
            if defined, only return occurences of epoch that are spanned by
            occurences of overlapping_epoch
        complete : boolean
            If True, eliminate any epochs whose boundaries are not fully
            contained within the signal.

        Returns
        -------
        bounds : 2D array (n_occurances x 2)
            Each row in the array corresponds to an occurance of the epoch. The
            first column is the start time and the second column is the end
            time.
        '''
        # If string, pull the epochs out of the internal dataframe.
        if isinstance(epoch, str):
            if self.epochs is None:
                m = "Signal does not have any epochs defined"
                raise ValueError(m)
            _mask = self.epochs['name'] == epoch
            bounds = self.epochs.loc[_mask, ['start', 'end']].values
            bounds = np.round(bounds.astype(float) * self.fs) / self.fs
        elif isinstance(epoch, list):
            # list of strings
            if self.epochs is None:
                m = "Signal does not have any epochs defined"
                raise ValueError(m)
            bounds = np.zeros((0,2))
            for e in epoch:
                _mask = self.epochs['name'] == e
                _bounds = self.epochs.loc[_mask, ['start', 'end']].values
                _bounds = np.round(_bounds.astype(float) * self.fs) / self.fs
                bounds = np.concatenate((bounds, _bounds), axis=0)

        elif isinstance(epoch, pd.core.series.Series):
            bounds = self.epochs.loc[epoch, ['start', 'end']].values
        else:
            bounds = epoch

        if boundary_mode is None:
            raise NotImplementedError
        elif boundary_mode == 'exclude':
            m = epoch_contained(bounds, self.segments)
            bounds = bounds[m]
        elif boundary_mode == 'trim':
            bounds = epoch_intersection(bounds, self.segments)

        if fix_overlap is None:
            pass
        elif fix_overlap == 'merge':
            bounds = merge_epoch(bounds)
        elif fix_overlap == 'first':
            bounds = remove_overlap(bounds)
        else:
            m = 'Unsupported mode, {}, for fix_overlap'.format(fix_overlap)
            raise ValueError(m)

        if overlapping_epoch is None:
            pass
        else:
            # find occurrences of overlapping epoch
            # only keep bounds that fall inside those occurences
            overlap_bounds = self.get_epoch_bounds(overlapping_epoch)
            bounds = epoch_intersection(bounds, overlap_bounds)

        # bounds = np.sort(bounds, axis=0)

        if mask is not None:
            # TODO : allow masking to work here. requires converting mask signal
            # to times (or bound times to samples)
            raise ValueError("mask not supported for get_epoch_bounds yet")

        return bounds

    def get_epoch_indices(self, epoch, boundary_mode='exclude',
                          fix_overlap=None, overlapping_epoch=None,
                          mask=None, allow_incomplete=False):
        '''
        Get boundaries of named epoch as index.

        Parameters
        ----------
        epoch : {string, Nx2 array, pandas series}
            passed through to get_epoch_bounds
            If string, name of epoch (as stored in internal dataframe) to
            extract.
            If list of strings, concatenate all epochs in list
            If Nx2 array, the first column indicates the start time
            (in seconds) and the second column indicates the end time
            (in seconds) to extract.
            If pandas series, pull out bounds of epochs where epoch==True
        boundary_mode : {None, 'exclude', 'trim'}
            If 'exclude', discard all epochs where the boundaries are not fully
            contained within the range of the signal. If 'trim', epochs with
            boundaries falling outside the signal range will be truncated. For
            example, if an epoch runs from -1.5 to 10, it will be truncated to
            0 to 10 (all signals start at time 0). If None, return all epochs.
        fix_overlap : {None, 'merge', 'first'}
            Indicates how to handle overlapping epochs. If None, return
            boundaries as-is. If 'merge', merge overlapping epochs into a
            single epoch. If 'first', keep only the first of an overlapping
            set of epochs.
        overlapping_epoch : {None, or string}
            if defined, only return occurences of epoch that are spanned by
            occurences of overlapping_epoch
        mask : {None or vector or signal}
            only include epochs (fully?) spanned by the mask==True
        allow_incomplete: {True, False}  (added CRH 2/4/2020)
            if True, allow mask to not perfectly match epoch lb and ub. However, epoch ub and lb must
            still span the mask. Right now, the mask MUST be identical across repetions of the epoch though!
            One example use case would be to take out PreStimSilence on each rep of an epoch using a mask.
            If the mask is variable between repetitions, this will throw an error.

            TODO: check how mask interacts with segments. Currently, masking is
            only tested before segmentation (ie, the application of the mask)

        Returns
        -------
        bounds : 2D array (n_occurances x 2)
            Each row in the array corresponds to an occurance of the epoch. The
            first column is the start time and the second column is the end
            time.
        '''
        bounds = self.get_epoch_bounds(epoch, boundary_mode, fix_overlap,
                                       overlapping_epoch)
        # Indices of segments and epochs
        s = 0
        e = 0

        # Running list of segment offsets
        o = 0
        n_segments = len(self.segments)
        n_epochs = len(bounds)
        indices = []

        while True:
            if s >= n_segments:
                break
            if e >= n_epochs:
                break

            s_lb, s_ub = self.segments[s]
            while True:
                # For given segment, loop through epochs that fall within that
                # segment and calculate correct indices.
                # (or span that segment - CRH 12/22/2019)
                e_lb, e_ub = bounds[e]

                if (e_lb <= s_lb) & (e_ub >= s_ub):
                    # epoch spans this segment
                    lb = o
                    ub = round((s_ub-s_lb)*self.fs) + o
                    indices.append((lb, ub))
                    e += 1

                elif s_lb <= e_lb < s_ub:
                    # Be sure to round otherwise an index of 1.999...999 will
                    # get converted to 1 rather than 2.
                    lb = round((e_lb-s_lb)*self.fs) + o
                    if e_ub <= s_ub:
                        ub = round((e_ub-s_lb)*self.fs) + o
                    else:
                        ub = round((s_ub-s_lb)*self.fs) + o
                    indices.append((lb, ub))
                    e += 1

                else:
                    # We are now at an epoch that falls in a different segment.
                    # Update the running offset. Break out of the loop and pull
                    # out the next segment.
                    s += 1
                    o += round((s_ub-s_lb)*self.fs)
                    break
                if e >= n_epochs:
                    break

        indices = np.asarray(indices, dtype='i')

        # exclude segments without data
        if (indices.size != 0) & allow_incomplete:
            zero_data_mask = (indices[:, 0] - indices[:, 1])!=0
            indices = indices[zero_data_mask, :]

        if mask is not None:
            # remove instances of the epoch that do not fall in the mask
            if type(mask) is np.ndarray:
                m_data = mask
            else:
                m_data = mask.as_continuous()

            # get a "reference epoch mask" for safety checking below
            standard_mask = None

            keepidx = []
            for i, (lb, ub) in enumerate(indices):
                #                samples = ub-lb
                if np.all(m_data[0, lb:ub]) & np.all(m_data[0, lb]):
                    keepidx.append(i)

                elif (np.sum(m_data[0, lb:ub]) > 0) & allow_incomplete:
                    # "safety" checks
                    if standard_mask is None:
                        standard_mask = m_data[0, lb:ub]
                    if (m_data[0, lb:ub].sum()!=standard_mask.sum()):
                        raise ValueError("For allow_incomplete=True, masks must all be the same size on each epoch")
                    #if (m_data[0, lb:ub].shape!=standard_mask.shape):
                    #    raise ValueError("For allow_incomplete=True, epochs must all be the same size")
                    #if  ~np.all(m_data[0, lb:ub] == standard_mask):
                    #    raise ValueError("Mask must be identical on each epoch when using allow_incomplete=True")

                    # define new indices
                    idx = np.where(m_data[0, lb:ub])
                    lb_partial = lb + idx[0][0]
                    ub_partial = lb + idx[0][-1] + 1
                    # check to make sure all True in this new range (i.e. can't extract non-continuous chunks of an epoch)
                    if np.all(m_data[0, lb_partial:ub_partial]):
                        keepidx.append(i)
                        indices[i, :] = [lb_partial, ub_partial]

            if len(keepidx) > 0:
                keepidx = np.array(keepidx)
                indices = indices[keepidx]
            else:
                indices = np.zeros((0,2), dtype='i')

        return indices

    def reset_segmented_epochs(self):
        epochs = np.unique(self.epochs.name)
        fs = self.fs
        df = pd.DataFrame([], columns=['start', 'end', 'name'])
        start = []
        end = []
        name = []
        for ep in epochs:
            _ep = self.get_epoch_indices(ep, boundary_mode='trim')
            if _ep.size != 0:
                _times = _ep / fs
                start += _times[:, 0].tolist()
                end += _times[:, 1].tolist()
                name += [ep] * len(_times)

        df['start'] = start
        df['end'] = end
        df['name'] = name
        df = df.sort_values(by='start').reset_index(drop=True)

        sig = self._modified_copy(self._data, epochs=df, segments=None)
        return sig

    def count_epoch(self, epoch, mask=None):
        """Returns the number of occurrences of the given epoch."""
        epoch_indices = self.get_epoch_indices(epoch, mask=mask)
        count = len(epoch_indices)
        return count

    def _get_attributes(self):
        md_attributes = ['name', 'chans', 'fs', 'meta', 'recording', 'epochs',
                         'segments', 'signal_type', 'normalization',
                         'norm_baseline', 'norm_gain']
        return {name: getattr(self, name) for name in md_attributes}

    def add_epoch(self, epoch_name, epoch):
        '''
        Add epoch to the internal epochs dataframe

        Parameters
        ----------
        epoch_name : string
            Name of epoch
        epoch : 2D array of (M x 2)
            The first column is the start time and second column is the end
            time. M is the number of occurrences of the epoch.
        '''
        df = pd.DataFrame(epoch, columns=['start', 'end'])
        df['name'] = epoch_name

        # important to match standard column order in case epochs is empty. Some code requires this order??
        #df = df[['name', 'start', 'end']]
        if self.epochs is not None:
            self.epochs = self.epochs.append(df, ignore_index=True)
        else:
            self.epochs = df

    def _split_epochs(self, split_time):
        if self.epochs is None:
            lepochs = None
            repochs = None
        else:
            mask = self.epochs['start'] < split_time
            lepochs = self.epochs.loc[mask]
            mask = self.epochs['end'] > split_time
            repochs = self.epochs.loc[mask]
            repochs.loc[:, 'start'] -= split_time
            repochs.loc[:, 'end'] -= split_time

            # If epochs were present initially but missing after split,
            # raise a warning.
            portion = None
            if lepochs.size == 0:
                portion = 'first'
            elif repochs.size == 0:
                portion = 'second'
            if portion:
                warnings.warn("Epochs for {0} portion of signal: {1}"
                              "ended up empty after splitting by time."
                              .format(portion, self.name))

        return lepochs, repochs

    @classmethod
    def _merge_epochs(cls, signals):
        # Merge the epoch tables. For all signals after the first signal,
        # we need to offset the start and end indices to ensure that they
        # reflect the correct position of the trial in the merged array.
        offset = 0
        epochs = []
        for signal in signals:
            ti = signal.epochs.copy()
            ti['end'] += offset
            ti['start'] += offset
            #offset += signal.ntimes/signal.fs
            offset += signal.epochs['end'].max()
            epochs.append(ti)
        return pd.concat(epochs, ignore_index=True)

    def average_epoch(self, epoch, mask=None):
        '''
        Returns the average of the epoch.

        Parameters
        ----------
        epoch : {string, Nx2 array}
            If string, name of epoch (as stored in internal dataframe) to
            extract. If Nx2 array, the first column indicates the start time
            (in seconds) and the second column indicates the end time
            (in seconds) to extract.

        mask: {None, signal}
            if provided, onlye extract epochs overlapping periods where
            mask.as_continuous()==True in all time bins

        Returns
        -------
        mean_epoch : 2D array
            Two dimensinonal array of shape C, T where C is the number of
            channels, and T is the maximum length of the epoch in samples.
        '''
        epoch_data = self.extract_epoch(epoch, mask=mask)
        return np.nanmean(epoch_data, axis=0)

    def extract_epochs(self, epoch_names, overlapping_epoch=None, mask=None, allow_incomplete=False):
        '''
        Returns a dictionary of the data matching each element in epoch_names.

        Parameters
        ----------
        epoch_names : list OR string
            if list, list of epoch names to extract. These will be keys in the
            result dictionary.
            if string, will find matches via nems0.epoch.epoch_names_matching

        chans : {None, iterable of strings}
            Names of channels to return. If None, return the full set of
            channels.  If an iterable of strings, return those channels (in the
            order specified by the iterable).

        overlapping_epoch: {None, string}
            if not None, only extracts epochs that overlap with occurrences
            of overlapping epoch

        mask: {None, signal}
            if provided, onlye extract epochs overlapping periods where
            mask.as_continuous()==True in all time bins

        Returns
        -------
        epoch_datasets : dict
            Keys are the names of the epochs, values are 3D arrays created by
            `extract_epoch`.
        '''
        # TODO: Update this to work with a mapping of key -> Nx2 epoch
        # structure as well.

        if type(epoch_names) is str:
            epoch_regex = epoch_names
            epoch_names = epoch_names_matching(self.epochs, epoch_regex)

        data = {}
        for name in epoch_names:
            v = self.extract_epoch(name, allow_empty=True,
                                   overlapping_epoch=overlapping_epoch,
                                   mask=mask, allow_incomplete=allow_incomplete)
            # only return matrices for epochs with non-empty data matrices
            # (deal with possibility that some stimuli are masked out)
            if len(v)>0:
                data[name] = v

        return data
        #
        # return {name: self.extract_epoch(name, allow_empty=True,
        #                                 overlapping_epoch=overlapping_epoch,
        #                                 mask=mask)
        #        for name in epoch_names}

    def generate_epoch_mask(self, epoch=True):
        '''
        inputs:
            epoch: {None, boolean, ndarray, string, list}
             if None, defaults to False
             if False, initialize mask signal to False for all times
             if True, initialize mask signal to False for all times
             if Tx1 ndarray, True where ndarray is true, False elsewhere
             if Nx2 ndarray, True in N epoch times
             if string (eoch name), mask is True for epochs with .name==string
             if list of strings (epoch names), mask is OR combo of all strings
             if list of tuples (epoch times), mask is OR combo of all epoch times
        '''

        mask = np.zeros([1, self.ntimes], dtype=np.bool)

        if (epoch is None) or (epoch is False):
            pass

        elif type(epoch) is str:
            # assuming defaults for boundary_mask and fix_overlap!
            indices = self.get_epoch_indices(epoch)
            for lb, ub in indices:
                mask[:, lb:ub] = True

        elif (type(epoch) is list) and (type(epoch[0]) is tuple):
            # epoch is a list of indices
            for (lb, ub) in epoch:
                mask[:, lb:ub] = True

        elif (type(epoch) is list) and (type(epoch[0]) is str):
            # epoch is a list of epochs
            mask = self.generate_epoch_mask(epoch[0])
            for e in epoch [1:]:
                mask = mask | self.generate_epoch_mask(e)

        elif (type(epoch) is np.ndarray) and (epoch.ndim ==2):
            # epoch is an array of indices
            for (lb, ub) in epoch:
                mask[:, lb:ub] = True

        elif (type(epoch) is np.ndarray) and (epoch.ndim == 1):
            # epoch is an 1d array, set mask to True at all times when array is nonzero
            mask[0, epoch] = True

        elif epoch == True:
            mask[:] = 1

        else:
            raise RuntimeError('Invalid epoch passed to generate_epoch_mask')
        return mask

    def epoch_to_signal(self, epoch='epoch', indices=None, boundary_mode='exclude',
                        fix_overlap=None, onsets_only=False, shift=0,
                        point_process=False):
        '''
        Convert an epoch to a RasterizedSignal using the same sampling rate
        and duration as this signal.

        Parameters
        ----------
        epoch : string
            Name of epoch(s) to convert to a signal
        indices : ndarray or None
            if not None, use this Nx2 array to specify epoch times
            otherwise find indices matching epoch
        onsets_only : boolean
            only return 1 at the start of each epoch
        point_process : boolean
            TODO
            if onsets_only==True, return PointProcess signal
            (need to get masking/splitting worked out for this signal type
            in order for it to be useful)
        Returns
        -------
        signal : instance of Signal
            A signal whose value is 1 for each occurrence of the epoch, 0
            otherwise.
        '''
        if indices is None:
            # find matching epoch periods
            indices = self.get_epoch_indices(epoch, boundary_mode, fix_overlap)

        data = np.zeros([1, self.ntimes], dtype=np.bool)
        for lb, ub in indices:
            if onsets_only:
                data[:, lb] = True
            else:
                data[:, lb:ub] = True
        if shift:
            data = np.roll(data, shift, axis=1)

        epoch_name = epoch if isinstance(epoch, str) else 'epoch'
        attributes = self._get_attributes()
        attributes['chans'] = [epoch_name]
        return RasterizedSignal(data=data, safety_checks=False, **attributes)
        # return self._modified_copy(data, chans=[epoch_name])

    def to_epochs(self):
        raise NotImplementedError

    @property
    def shape(self):
        return tuple([self.nchans]+self.n_extradims+[self.ntimes])

    def select_times(self, times):
        raise NotImplementedError

    def split_at_time(self, fraction):
        raise NotImplementedError

    def extract_channels(self, chans, name=None):
        raise NotImplementedError

    def extract_epoch(self, epoch, allow_empty=True,
                      overlapping_epoch=None, mask=None):
        raise NotImplementedError

    def remove_epochs(self, mask):
        """
        delete epochs falling in False region of mask signal.
        don't do anything to the _data itself
        :param mask:
        :return:
        """
        signal = self.copy()
        fs = mask.fs

        temp_epochs = signal.epochs.copy()
        temp_epochs['start'] = np.round(temp_epochs['start'] * fs).astype(int)
        temp_epochs['end'] = np.round(temp_epochs['end'] * fs).astype(int)

        new_mask = np.full(len(temp_epochs), False)
        mask_data = mask._data[0]

        for idx, (start, end) in enumerate(temp_epochs[['start', 'end']].values):
            if mask_data[start:end].all():
                new_mask[idx] = True

        signal.epochs = self.epochs.loc[new_mask]
        return signal

    @staticmethod
    def load(basepath):
        pass

    @classmethod
    def concatenate_time(cls, signals):
        raise NotImplementedError

    @classmethod
    def concatenate_channels(cls, signals):
        raise NotImplementedError

    def select_times(self, times):
        '''
        Parameters
        ----------
        times : list of tuples
            Times is a list of tuples. Each tuple specifies the lower and upper
            bound (in seconds) of the time subset to extract. This should return
            a new signal containing only the subset of desired data.
        '''
        s = self.rasterize()
        return s.select_times(times)

    def select_channels(self, channels):
        '''
        Parameters
        ----------
        channels : list of channels
            A list of channel names indicating channels to extract. This should
            return a new signal containing only the subset of desired data.
        '''
        raise NotImplementedError

    def as_raster(self):
        '''
        Returns a RasterizedSignal or RasterizedSubsetSignal
        '''
        raise NotImplementedError

    def as_continuous(self):
        '''
        Returns the underlying array -- NOT IMPLEMENTED FOR THIS SIGNAL
        '''
        raise NotImplementedError

    def as_matrix(self, epoch_names, overlapping_epoch=None, mask=None):
        """
        Inputs:
            epoch_names: regex or list of epochs
            overlapping_epoch: require those epochs to overlap with epoch(s)
                matching this name.
                eg, epoch_names="^STIM_", overlapping_epoch="PASSIVE_EXPERIMENT"
                (DEPRECATED?)
            mask: {None, signal}
                if provided, onlye extract epochs overlapping periods where
                mask.as_continuous()==True in all time bins

        Returns:
            d: np.array, stim X reps X channels X time


        TODO: add channel selection option?
        """

        # create dictionary of extracted epochs
        folded_signal = self.extract_epochs(
                epoch_names, overlapping_epoch=overlapping_epoch, mask=mask)
        if type(epoch_names) is list:
            keys = epoch_names
        else:
            keys = list(folded_signal.keys())
            keys.sort()

        reps = []
        chans = []
        lens = []
        for k in keys:
            if k in folded_signal.keys():
                reps.append(folded_signal[k].shape[0])
                chans.append(folded_signal[k].shape[1])
                lens.append(folded_signal[k].shape[2])
            else:
                reps.append(0)
                chans.append(0)
                lens.append(0)

        max_rep = np.max(np.array(reps))
        max_chan = np.max(np.array(chans))
        max_len = np.max(np.array(lens))
        d = np.empty((len(keys),max_rep,max_chan,max_len))
        d[:] = np.nan

        for i, k in enumerate(keys):
            if k in folded_signal.keys():
                d[i,:reps[i],:chans[i],:lens[i]]=folded_signal[k]

        return d

    ## plotting functions
    def plot_mean(self, epoch="TRIAL", channel=None, norm=False, ax=None):
        import matplotlib.pyplot as plt

        e = np.nanmean(self.extract_epoch(epoch), axis=2)
        if channel is not None:
            e=e[:,channel, :]

        #log.info(f"epoch {epoch} eshape={e.shape}")
        if ax is None:
            f, ax = plt.subplots()
        if e.shape[1] == 1:
            ax.plot(e)
            ax.set_xlabel(f'{epoch} occurrence')
            ax.set_ylabel('Mean')
        else:
            if norm:
                im = ax.imshow((e / e.max(axis=0)).T, origin='lower', aspect='auto')
            else:
                im = ax.imshow(e.T, origin='lower', aspect='auto')
            #plt.colorbar(im, ax=ax)
            ax.set_xlabel(f'{epoch} occurrence')
            ax.set_ylabel('Channel')

        ax.set_title(self.name)

    def plot_epoch_avg(self, epoch="TRIAL", channel=None, norm=False, ax=None):
        import matplotlib.pyplot as plt

        e = np.nanmean(self.extract_epoch(epoch), axis=0)

        if channel is not None:
            if type(channel) is int:
                e = e[[channel], :]
            else:
                e = e[channel, :]

        #log.info(f"epoch {epoch} eshape={e.shape}")
        if ax is None:
            f, ax = plt.subplots()
        if e.shape[0] == 1:
            ax.plot(e.T)
            ax.set_xlabel(f'Bins from {epoch} onset')
            ax.set_ylabel('Mean')
        else:
            if norm:
                im = ax.imshow((e / e.max(axis=1)), origin='lower', aspect='auto')
            else:
                im = ax.imshow(e, origin='lower', aspect='auto')
            #plt.colorbar(im, ax=ax)
            ax.set_xlabel(f'Bins from {epoch} onset')
            ax.set_ylabel('Channel')

        ax.set_title(self.name)
        
    def plot_raster(self, epoch="TRIAL", channel=None, ax=None):
        
        from nems0.plots.raster import raster

        if channel is None:
            channel = 0

        if channel in self.chans:
            r_ = self.extract_channels([channel])
            r = r_.extract_epoch(epoch)[:, 0, :]
        else:
            r = self.extract_epoch(epoch)[:, channel, :]

        times = np.arange(r.shape[1]) / self.fs
        title = f"{self.name} chan {channel} epoch {epoch} raster"
        raster(times, r, ax=ax, title=title)


class RasterizedSignal(SignalBase):

    def __init__(self, fs, data, name, recording, chans=None, epochs=None,
                 segments=None, meta=None, safety_checks=True,
                 normalization='none', **other_attributes):
        '''
        Parameters
        ----------
        data : ndarray, 2 dimensional
        epochs : {None, DataFrame}
            Epochs are periods of time that are tagged with a name
            When defined, the DataFrame should have these first three columns:
                 ('start', 'end', 'name')
            denoting the start and end of the time of an epoch (in seconds).
            You may use the same epoch name multiple times; this is common when
            tagging epochs that correspond to occurrences of the same stimulus.
        '''
        if data.ndim==1:
            # assume one dim should be time
            data=np.reshape(data,(1,data.size))
            
        super().__init__(fs, data, name, recording, chans, epochs, segments,
                         meta, safety_checks, normalization)
        self._data.flags.writeable = False

        # Install the indexers
        self.iloc = SimpleSignalIndexer(self)
        self.loc = LabelSignalIndexer(self)
        s = self._data.shape
        self.nchans = s[0]
        self.ntimes = s[-1]
        if len(s) > 2:
            self.n_extradims = list(s[1:-1])
        else:
            self.n_extradims = []
        self.signal_type = str(type(self))

        # Verify that we have a long time series
        if safety_checks and self.ntimes < self.nchans:
            m = 'Incorrect matrix dimensions?: (C, T) is {}. ' \
                'We expect a long time series, but T < C'
            warnings.warn(m.format((self.nchans, self.ntimes)))

        if safety_checks:
            self._run_safety_checks()

    @classmethod
    def from_3darray(cls, fs, array, name, recording, epoch_name='TRIAL',
                     chans=None, meta=None, safety_cheks=True):
        """Initialize RasterizedSignal from 3d array

        Parameters
        ----------
        fs :
        array : ndarray  (n_epochs, n_channels, n_times)
            Data array.
        """
        assert array.ndim == 3
        n_trials, n_channels, n_times = array.shape
        data = np.swapaxes(array, 0, 1)
        data = data.reshape((n_channels, n_trials * n_times))

        out = cls(fs, data, name, recording, chans, meta=meta,
                  safety_checks=safety_cheks)
        times = np.array([[t / fs, (t + n_times) / fs] for t in
                          range(0, n_trials * n_times, n_times)])
        out.add_epoch(epoch_name, times)
        return out

    def _set_cached_props(self):
        """Sets channel_max, channel_min, channel_mean, channel_var,
        and channel_std.

        """
        self.channel_max = np.nanmax(self._data, axis=-1, keepdims=True)
        self.channel_min = np.nanmin(self._data, axis=-1, keepdims=True)
        self.channel_mean = np.nanmean(self._data, axis=-1, keepdims=True)
        self.channel_var = np.nanvar(self._data, axis=-1, keepdims=True)
        self.channel_std = np.nanstd(self._data, axis=-1, keepdims=True)

    def as_file_streams(self, fmt='%.18e'):
        '''
        Returns 3 filestreams for this signal: the csv, json, and epoch.
        TODO: Better docs and a refactoring of this and save()
        '''
        # TODO: actually compute these instead of cheating with a tempfile
        files = {}
        filebase = self.recording + '.' + self.name
        csvfile = filebase + '.csv'
        jsonfile = filebase + '.json'
        epochfile = filebase + '.epoch.csv'
        # Create three streams
        files[csvfile] = io.BytesIO()
        files[jsonfile] = io.StringIO()
        files[epochfile] = io.StringIO()
        # Write to those streams
        # Write the CSV file to a bytesIO buffer
        mat = self.as_continuous()
        mat = np.swapaxes(mat, 0, 1)
        np.savetxt(files[csvfile], mat, delimiter=",", fmt=fmt)
        files[csvfile].seek(0)  # Seek back to start of file

        self._save_metadata(files[epochfile],files[jsonfile], fmt)

        return files

    def save(self, dirpath, fmt='%.18e', prefix=''):
        '''
        Save this signal to a CSV file + JSON sidecar. If desired,
        you may use optional parameter fmt (for example, fmt='%1.3e')
        to alter the precision of the floating point matrices.
        '''

        jsonfilepath,epochfilepath=self._save_metadata_to_dirpath(dirpath)

        filebase = prefix + self.recording + '.' + self.name
        basepath = os.path.join(dirpath, filebase)
        csvfilepath = basepath + '.csv'

        mat = self.as_continuous()
        mat = np.swapaxes(mat, 0, 1)
        # TODO: Why does numpy not support fileobjs like streams?
        np.savetxt(csvfilepath, mat, delimiter=",", fmt=fmt)

        return (csvfilepath, jsonfilepath, epochfilepath)

    @staticmethod
    def load(basepath):
        '''
        Loads the CSV & JSON files at basepath; returns a Signal() object.
        Example: If you want to load
           /tmp/sigs/gus027b13_p_PPS_resp-a1.csv
           /tmp/sigs/gus027b13_p_PPS_resp-a1.json
        then give this function
           /tmp/sigs/gus027b13_p_PPS_resp-a1
        '''
        return load_rasterized_signal(basepath)

    # SVD -- 2018-05-24 -- save to delete?
    """
    @staticmethod
    def load_from_streams(csv_stream, json_stream, epoch_stream=None):
        ''' Loads from BytesIO objects rather than files. '''
        # Read the epochs stream if it exists
        epochs = pd.read_csv(epoch_stream) if epoch_stream else None
        # Read the json metadata
        js = json.load(json_stream)
        # Read the CSV
        mat = pd.read_csv(csv_stream, header=None).values
        mat = mat.astype('float')
        mat = np.swapaxes(mat, 0, 1)
        # mat = np.genfromtxt(csv_stream, delimiter=',')
        # Now build the signal
        s = RasterizedSignal(name=js['name'],
                   chans=js.get('chans', None),
                   epochs=epochs,
                   recording=js['recording'],
                   fs=js['fs'],
                   meta=js['meta'],
                   data=mat)
        s.segments = np.array(js.get('segments', s.segments))
        return s
    """

    @staticmethod
    def list_signals(directory):
        '''
        Returns a list of all CSV/JSON pairs files found in DIRECTORY,
        Paths are relative, not absolute.
        '''
        files = os.listdir(directory)
        return RasterizedSignal._csv_and_json_pairs(files)

    @staticmethod
    def _csv_and_json_pairs(files):
        '''
        Given a list of files, return the file basenames (i.e. no extensions)
        that for which a .CSV and a .JSON file exists.
        '''
        just_fileroot = lambda f: os.path.splitext(os.path.basename(f))[0]
        csvs = [just_fileroot(f) for f in files if f.endswith('.csv')]
        jsons = [just_fileroot(f) for f in files if f.endswith('.json')]
        overlap = set.intersection(set(csvs), set(jsons))
        return list(overlap)

    def _modified_copy(self, data, **kwargs):
        '''
        For internal use when making various immutable copies of this signal.
        '''
        attributes = self._get_attributes()
        attributes.update(kwargs)
        return RasterizedSignal(data=data, safety_checks=False, **attributes)

    def extract_epoch(self, epoch, boundary_mode='exclude',
                      fix_overlap='first', allow_empty=False, trunc_at_min=False,
                      overlapping_epoch=None, mask=None, allow_incomplete=False):
        '''
        Extracts all occurances of epoch from the signal.

        Parameters
        ----------
        epoch : {string, Nx2 array}
            If string, name of epoch (as stored in internal dataframe) to
            extract. If Nx2 array, the first column indicates the start time
            (in seconds) and the second column indicates the end time
            (in seconds) to extract.

            allow_empty: if true, returns empty matrix if no valid epoch
            matches. otherwise, throw error when this happens

        boundary_mode, fix_overlap: parameters passed through to
            get_epoch_indices

        allow_empty: {False, boolean}

        trunc_at_min: {False, boolean} if True, truncate all epochs
             to length of shortest match

        mask: {None, signal}
            if provided, onlye extract epochs overlapping periods where
            mask.as_continuous()==True in all time bins

        Returns
        -------
        epoch_data : 3D array
            Three dimensional array of shape O, C, T where O is the number of
            occurances of the epoch, C is the number of channels, and T is the
            maximum length of the epoch in samples.

        Note
        ----
        Epochs tagged with the same name may have various lengths. Shorter
        epochs will be padded with NaN.
        '''

        if type(epoch) is str:
            epoch_indices = self.get_epoch_indices(epoch,
                                                   boundary_mode=boundary_mode,
                                                   fix_overlap=fix_overlap,
                                                   mask=mask,
                                                   allow_incomplete=allow_incomplete)
        else:
            epoch_indices = epoch

        if epoch_indices.size == 0:
            if allow_empty:
                return np.empty([0, 0, 0])
            else:
                raise IndexError("No matching epochs to extract for: %s\n"
                                 "In signal: %s", epoch, self.name)

        if trunc_at_min:
            n_samples = np.min(epoch_indices[:, 1]-epoch_indices[:, 0])
            epoch_indices[:,1] = epoch_indices[:, 0] + n_samples
        else:
            n_samples = np.max(epoch_indices[:, 1]-epoch_indices[:, 0])
        n_epochs = len(epoch_indices)

        data = self.as_continuous()
        n_chans = data.shape[0]
        epoch_shape = [n_epochs] + list(data.shape[:-1]) + [n_samples]
        if data.dtype == bool:
            epoch_data = np.full(epoch_shape, False, dtype=bool)
        else:
            epoch_data = np.full(epoch_shape, np.nan)
        # print(epoch)

        for i, (lb, ub) in enumerate(epoch_indices):
            if ub > data.shape[-1]:
                ub = data.shape[-1]
            samples = ub-lb
            try:
                epoch_data[i, ..., :samples] = data[..., lb:ub]
            except:
                raise ValueError('Trying to extract invalid range from signal for epoch (out of bounds or negative duration?).')
        return epoch_data

    def normalize(self, normalization='minmax', b=None, g=None, mask=None):
        '''
        Returns a copy of this signal with each channel normalized to have a
        mean of 0 and standard deviation of 1.
        '''
        m = self._data * self.norm_gain + self.norm_baseline
        
        m_normed, b, g = _normalize_data(m, normalization, d=b, g=g, sig=self, mask=mask)
        sig = self._modified_copy(m_normed)
        sig.normalization = normalization
        sig.norm_baseline = b
        sig.norm_gain = g

        return sig

    def normalize_sqrt(self, **opts):
        '''
        Take sqrt() of signal
        '''
        m = self._data / np.sqrt(np.abs(self._data) + (self._data==0))
        sig = self._modified_copy(m)

        return sig

    def normalized_by_mean(self):
        '''
        Returns a copy of this signal with each channel normalized to have a
        mean of 0 and standard deviation of 1.
        '''
        # TODO: this is a workaround/hack until a less expensive
        #       method for calculating these is available.
        if not (hasattr(self, 'channel_mean')
                and hasattr(self, 'channel_std')):
            self._set_cached_props()
        m = self._data
        m_normed = (m - self.channel_mean) / self.channel_std
        return self._modified_copy(m_normed)

    def normalize_spont(self):
        '''
        Returns a copy of this signal with each channel normalized to have a
        mean of 0 during the PreStimSilence period
        '''
        # figure out spont rate for subtraction from PSTH
        try:
            #trec=rec.and_mask('REFERENCE',invert=True)
            #trec=trec.and_mask('TARGET',invert=True)
            #if trec['mask'].as_continuous().sum()>0:
            #    prestimsilence = resp.extract_epoch('PreStimSilence', mask=trec['mask'])
            #else:
            prestimsilence = self.extract_epoch('PreStimSilence', mask=mask)

            if prestimsilence.shape[-1] > 0:
                if len(prestimsilence.shape) == 3:
                    spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
                else:
                    spont_rate = np.nanmean(prestimsilence)
            else:
                try:
                    prestimsilence = resp.extract_epoch('TRIALPreStimSilence')
                    if len(prestimsilence.shape) == 3:
                        spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
                    else:
                        spont_rate = np.nanmean(prestimsilence)
                except:
                    raise ValueError("Can't find prestim silence to use for PSTH calculation")
        except:
            # special case where the epochs included in mask don't have PreStimSilence,
            # so we get it elsewhere. Designed for CPN data...
            try:
                prestimsilence = resp.extract_epoch('TRIALPreStimSilence')
                if len(prestimsilence.shape) == 3:
                    spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
                else:
                    spont_rate = np.nanmean(prestimsilence)
            except:
                raise ValueError("Can't find prestim silence to use for PSTH calculation")

        m_normed = self._data - spont_rate
        return self._modified_copy(m_normed)

    def normalized_by_bounds(self):
        '''
        Returns a copy of this signal with each channel normalized to the range
        [-1, 1]
        '''
        # TODO: this is a workaround/hack until a less expensive
        #       method for calculating these is available.
        if not (hasattr(self, 'channel_max')
                and hasattr(self, 'channel_std')
                and hasattr(self, 'channel_max')):
            self._set_cached_props()
        m = self._data
        ptp = self.channel_max - self.channel_mean
        m_normed = (m - self.channel_min) / ptp - 1
        return self._modified_copy(m_normed)

    def split_at_time(self, fraction):
        '''
        Splits this signal at 'fraction' of the total length of the time series
        to create a tuple of two signals: (before, after).
        Example:
          l, r = mysig.split_at_time(0.8)
          assert(l.ntimes == 0.8 * mysig.ntimes)
          assert(r.ntimes == 0.2 * mysig.ntimes)

        '''
        split_idx = max(1, int(self.ntimes * fraction))
        split_time = split_idx/self.fs

        data = self.as_continuous()
        ldata = data[..., :split_idx]
        rdata = data[..., split_idx:]

        lepochs, repochs = self._split_epochs(split_time)
        lsignal = self._modified_copy(ldata, epochs=lepochs)
        rsignal = self._modified_copy(rdata, epochs=repochs)

        return lsignal, rsignal

    def split_by_epochs(self, epochs_for_est, epochs_for_val):
        '''
        Returns a tuple of estimation and validation data splits: (est, val).
        Arguments should be lists of epochs that define the estimation and
        validation sets. Both est and val will have non-matching data NaN'd
        out.
        '''
        est = self.select_epochs(epochs_for_est)
        val = self.select_epochs(epochs_for_val)
        return (est, val)

    def jackknife_by_epoch(self, njacks, jack_idx, epoch_name,
                           tiled=True,
                           invert=False, excise=False):
        '''
        Returns a new signal, with epochs matching epoch_name NaN'd out.
        Optional argument 'invert' causes everything BUT the matched epochs
        to be NaN'd. njacks determines the number of jackknifes to divide
        the epochs into, and jack_idx determines which one to return.

        'Tiled' makes each jackknife use every njacks'th occurrence, and is
        probably best explained by the following example...

        If there are 18 occurrences of an epoch, njacks=5, invert=False,
        and tiled=True, then the five jackknifes will have these
        epochs NaN'd out:

           jacknife[0]:  0, 5, 10, 15
           jacknife[1]:  1, 6, 11, 16
           jacknife[2]:  2, 7, 12, 17
           jacknife[3]:  3, 8, 13
           jacknife[4]:  4, 9, 14

        Note that the last two jackknifes have one fewer occurrences.

        If tiled=False, then the pattern of NaN'd epochs becomes sequential:

           jacknife[0]:   0,  1,  2,  3
           jacknife[1]:   4,  5,  6,  7,
           jacknife[2]:   8,  9, 10, 11,
           jacknife[3]:  12, 13, 14, 15,
           jacknife[4]:  16, 17

        Here we can see the last jackknife has 2 fewer occurrences.

        In any case, an exception will be thrown if epoch_name is not found,
        or when there are fewer occurrences than njacks.
        '''

        epochs = self.get_epoch_indices(epoch_name)
        occurrences, _ = epochs.shape

        if excise:
            raise ValueError('Excise not supported for jackknife_by_epoch')
        if len(epochs) == 0:
            m = 'No epochs found matching epoch_name. Unable to jackknife.'
            raise ValueError(m)

        if occurrences < njacks:
            raise ValueError("Can't divide {0} occurrences into {1} jackknifes"
                             .format(occurrences, njacks))

        if jack_idx < 0 or njacks < 0:
            raise ValueError("Neither jack_idx nor njacks may be negative")

        nrows = math.ceil(occurrences / njacks)
        idx_data = np.arange(nrows * njacks)

        if tiled:
            idx_data = idx_data.reshape(nrows, njacks)
            idx_data = np.swapaxes(idx_data, 0, 1)
        else:
            idx_data = idx_data.reshape(njacks, nrows)

        data = self.as_continuous().copy()
#        sig_valid_start = np.sum(np.isfinite(data[0,:]))

        mask = np.zeros_like(data, dtype=np.bool)
        mask2 = np.zeros_like(data, dtype=np.bool)

        for ep in epochs:
            lb, ub = ep
            mask2[:,lb:ub]=1

        for idx in idx_data[jack_idx].tolist():
            if idx < occurrences:
                lb, ub = epochs[idx]
                mask[:, lb:ub] = 1

        if invert:
            mask = ~mask

        data[mask] = np.nan
        data[~mask2] = np.nan

        return self._modified_copy(data)

    def jackknifes_by_epoch(self, njacks, epoch_name, tiled=True):
        '''
        Convenience fn. Returns generator that returns njacks tuples of
        (est, val) made using jackknife_by_epoch().
        '''
        jack_idx = 0
        while jack_idx < njacks:
            yield (self.jackknife_by_epoch(njacks, jack_idx, epoch_name,
                                           tiled=tiled, invert=False),
                   self.jackknife_by_epoch(njacks, jack_idx, epoch_name,
                                           tiled=tiled, invert=True))
            jack_idx += 1

    def jackknife_by_time(self, njacks, jack_idx, invert=False, excise=False):
        '''
        Returns a new signal, with some data NaN'd out based on its position
        in the time stream. jack_idx is indexed from 0; if you have 20 splits,
        the first is #0 and the last is #19.
        Optional argument 'invert' causes everything BUT the jackknife to be NaN.
        Optional argument 'excise' removes the elements that would've been NaN
        and thus changes the size of the signal.
        '''
        splitsize = int(self.ntimes / njacks)
        if splitsize < 1:
            m = 'Too many jackknifes? Splitsize was {}'
            raise ValueError(m.format(splitsize))

        split_start = jack_idx * splitsize
        if jack_idx == njacks - 1:
            split_end = self.ntimes
        else:
            split_end = (jack_idx + 1) * splitsize
        m = self.as_continuous().copy()
        if excise:
            if invert:
                o = np.empty((self.nchans, split_end - split_start))
                o = m[:, split_start:split_end]
            else:
                o = np.delete(m, slice(split_start, split_end), axis=-1)
            return self._modified_copy(o.reshape(self.nchans, -1))
        else:
            if not invert:
                m[..., split_start:split_end] = np.nan
            else:
                mask = np.ones_like(m, dtype=np.bool)
                mask[:, split_start:split_end] = 0
                m[mask] = np.nan
            return self._modified_copy(m.reshape(self.nchans, -1))

    def jackknifes_by_time(self, njacks):
        '''
        Convenience fn. Returns generator that returns njacks tuples of
        (est, val) made using jackknife_by_time().
        '''
        jack_idx = 0
        while jack_idx < njacks:
            yield (self.jackknife_by_time(njacks, jack_idx, invert=False),
                   self.jackknife_by_time(njacks, jack_idx, invert=True))
            jack_idx += 1

    @classmethod
    def concatenate_time(cls, signals):
        '''
        Combines the signals along the time axis. All signals must have the
        same number of channels and the same sampling rates.
        '''
        # Make sure all objects passed are instances of the Signal class
        for signal in signals:
            if not isinstance(signal, cls):
                raise ValueError('Cannot merge these signals')

        # Make sure that important attributes match up
        base = signals[0]
        for signal in signals[1:]:
            if not base.fs == signal.fs:
                raise ValueError('Cannot concat signals with unequal fs')
            if not base.chans == signal.chans:
                raise ValueError('Cannot concat signals with unequal # of chans')

        # Now, concatenate data along time axis
        data = np.concatenate([s.as_continuous() for s in signals], axis=-1)
        epochs = cls._merge_epochs(signals)

        return cls(
            name=base.name,
            recording=base.recording,
            chans=base.chans,
            fs=base.fs,
            meta=base.meta,
            data=data,
            epochs=epochs,
            safety_checks=False
        )

    @classmethod
    def concatenate_channels(cls, signals):
        '''
        Given signals=[sig1, sig2, sig3, ..., sigN], concatenate all channels
        of [sig2, ...sigN] as new channels on sig1. All signals must be equal-
        length time series sampled at the same rate (i.e. ntimes and fs are the
        same for all signals).
        '''
        for signal in signals:
            if not isinstance(signal, cls):
                raise ValueError('Cannot merge these signals')

        base = signals[0]
        for signal in signals[1:]:
            if not base.fs == signal.fs:
                raise ValueError('Cannot append signal with different fs')
            if not base.ntimes == signal.ntimes:
                raise ValueError('Cannot concatenate signal with different number of samples')

        data = np.concatenate([s.as_continuous() for s in signals], axis=0)

        chans = []
        for signal in signals:
            if signal.chans:
                chans.extend(signal.chans)

        epochs=signals[0].epochs

        attr=base._get_attributes()
        del attr['epochs']
        del attr['fs']
        del attr['name']
        del attr['recording']
        del attr['chans']
        del attr['signal_type']

        return RasterizedSignal(base.fs, data, base.name, base.recording,
                                epochs=epochs, chans=chans,
                                safety_checks=False, **attr)

    def decimate(self, stride=2, axis=-1):
        l=self.shape[1]
        if l & 0x1:
            new_data = self._data[..., ::stride].copy()
            new_data[:, 0:-1] = (new_data[:, 0:-1] + self._data[..., 1::stride])
        else:
            new_data = (self._data[..., ::stride] + self._data[..., 1::stride])
        new_fs = int(self.fs/stride)
        return self._modified_copy(data=new_data, fs=new_fs)

    def extract_channels(self, chans=None, chan_idx=None, name=None):
        '''
        Returns a new signal object containing only the specified
        channel indices.
        '''
        array = self.as_continuous()
        if chans is not None:
            # s is shorthand for slice. Return a 2D array.
            s = [self.chans.index(c) for c in chans]
        elif chan_idx is not None:
            # numeric indices passed
            s = np.array(chan_idx)
            chans = [self.chans[i] for i in s]

        if name is None:
            name = self.name
        return self._modified_copy(array[s], chans=chans, name=name)


    def replace_epoch(self, epoch, epoch_data, preserve_nan=True, mask=None):
        '''
        Returns a new signal, created by replacing every occurrence of
        epoch with epoch_data, assumed to be a 2D matrix of data
        (chans x time).

        Or if epoch_data is occurence X chans X time, replace each epoch with
        the corresponding occurence in epcoh_data
        '''
        data = self.as_continuous().copy()
        if preserve_nan:
            nan_bins = np.isnan(data[0, :])
        indices = self.get_epoch_indices(epoch, mask=mask)
        if indices.size == 0:
            warnings.warn("No occurrences of epoch were found: \n{}\n"
                          "Nothing to replace.".format(epoch))
        if epoch_data.ndim == 2:
            for lb, ub in indices:
                data[:, lb:ub] = epoch_data
        else:
            ii = 0
            for lb, ub in indices:
                if ub <= data.shape[1]:
                    n = ub-lb
                    data[:, lb:ub] = epoch_data[ii, :, :n]
                else:
                    log.info("Skipping epoch that falls outside of data time window")
                ii += 1

        if preserve_nan:
            data[:, nan_bins] = np.nan

        return self._modified_copy(data)

    def replace_epochs(self, epoch_dict, preserve_nan=True, zero_outside=False, mask=None):
        '''
        Returns a new signal, created by replacing every occurrence of epochs
        in this signal with whatever is found in the replacement_dict under
        the same epoch_name key. Dict values are assumed to be 2D matrices
        (same signal for each occurence) or 3D (different signal for each
        occurence).

        NOTE: segments of the signal outside of any matching epoch are
        set to np.nan

        If the replacement matrix shape is not the same as the original
        epoch being replaced, it will be truncated.

        If overlapping epochs are defined, then they will be replaced in
        the order present in the epochs dataframe (i.e. sorting your
        epochs dataframe may change the results you get!). For this reason,
        we do not recommend replacing overlapping epochs in a single
        operation because there is some ambiguity as to the result.
        '''

        data = self.as_continuous().copy()
        if zero_outside:
            data[:]=0

        if preserve_nan:
            nan_bins = np.isnan(data[0, :])

        # intialize with nans so that any subsequent prediction will be
        # restricted to the specified epochs
        # TODO - remove this - not necessary anymore?
#        if data.dtype == bool:
#            data[:] = False
#        else:
#            data[:] = np.nan

        for epoch, epoch_data in epoch_dict.items():
            indices = self.get_epoch_indices(epoch, mask=mask)
            if epoch_data.ndim == data.ndim:
                # ndim==2: single PSTH to be inserted in every matching epoch
                for lb, ub in indices:
                    if ub-lb > epoch_data.shape[-1]:
                        ub = lb + epoch_data.shape[-1]
                    if ub > data.shape[-1]:
                        ub = data.shape[-1]
                    #print(ub-lb)
                    #print(epoch_data.shape)
                    data[..., lb:ub] = epoch_data[..., :(ub-lb)]

            else:
                # ndim==3: different segment to insert for each epoch
                # (assume epoch_data.shape[1] == len(indices))
                ii = 0
                for lb, ub in indices:
                    if ub > data.shape[-1]:
                        ub = data.shape[-1]
                    n = ub-lb
                    data[..., lb:ub] = epoch_data[ii, ..., :n]
                    ii += 1

        if preserve_nan:
            data[..., nan_bins] = np.nan

        return self._modified_copy(data)

    # def replace_epochs(self, epoch_dict, preserve_nan=True, mask=None):
    #     '''
    #     Returns a new signal, created by replacing every occurrence of epochs
    #     in this signal with whatever is found in the replacement_dict under
    #     the same epoch_name key. Dict values are assumed to be 2D matrices
    #     (same signal for each occurence) or 3D (different signal for each
    #     occurence).
    #
    #     NOTE: segments of the signal outside of any matching epoch are
    #     set to np.nan
    #
    #     If the replacement matrix shape is not the same as the original
    #     epoch being replaced, it will be truncated.
    #
    #     If overlapping epochs are defined, then they will be replaced in
    #     the order present in the epochs dataframe (i.e. sorting your
    #     epochs dataframe may change the results you get!). For this reason,
    #     we do not recommend replacing overlapping epochs in a single
    #     operation because there is some ambiguity as to the result.
    #     '''
    #
    #     data = self.as_continuous().copy()
    #     if preserve_nan:
    #         nan_bins = np.isnan(data[0, :])
    #
    #     epochs = sorted(list(epoch_dict.keys()))
    #     # get the length of the epochs so that we can iterate through the aggregated epoch indices
    #     epoch_data_lens = self.epochs['name'].value_counts()[epochs].values
    #
    #     # need to reorder epochs dataframe to ensure that the indices are returned in the
    #     # same order as the epochs we pulled from the dict; return to original order when done
    #     old_index = self.epochs.index.values
    #     self.epochs = self.epochs.sort_values('name')
    #
    #     epoch_mask = self.epochs['name'].isin(epochs)
    #     indices = self.get_epoch_indices(epoch_mask, mask=mask)
    #
    #     # return epochs df to old order
    #     self.epochs = self.epochs.loc[old_index]
    #
    #     start = 0
    #     for epoch, epoch_data_len in zip(epochs, epoch_data_lens):
    #         epoch_data = epoch_dict[epoch]
    #
    #         if epoch_data.ndim == 2:
    #             for lb, ub in indices[start: start + epoch_data_len]:
    #                 data[:, lb:ub] = epoch_data[:, :(ub-lb)]
    #
    #         elif epoch_data.ndim == 3:
    #             for idx, (lb, ub) in enumerate(indices[start: start + epoch_data_len]):
    #                 data[:, lb:ub] = epoch_data[idx, :, :(ub-lb)]
    #
    #         start += epoch_data_len
    #
    #     if preserve_nan:
    #         data[:, nan_bins] = np.nan
    #
    #     return self._modified_copy(data)

    def select_epoch(self, epoch):
        '''
        Returns a new signal, the same as this, with everything NaN'd
        unless it is tagged with epoch_name.
        '''
        new_data = np.full(self.shape, np.nan)
        for (lb, ub) in self.get_epoch_indices(epoch, trim=True):
            new_data[:, lb:ub] = self._data[:, lb:ub]
        if np.all(np.isnan(new_data)):
            warnings.warn("No matched occurrences for epoch: \n{}\n"
                                 "Returned signal will be only NaN."
                                 .format(epoch))
        return self._modified_copy(new_data)

    def select_epochs(self, list_of_epoch_names):
        '''
        Returns a new signal, the same as this, with everything NaN'd
        unless it is tagged with one of the epoch_names found in
        list_of_epoch_names.
        '''
        # TODO: Update this to work with a mapping of key -> Nx2 epoch
        # structure as well.
        new_data = np.full(self.shape, np.nan, dtype=self._data.dtype)

        mask = self.epochs['name'].isin(list_of_epoch_names)
        for (lb, ub) in self.get_epoch_indices(mask):
            new_data[..., lb:ub] = self._data[..., lb:ub]

        if np.all(np.isnan(new_data)):
            warnings.warn("No matched occurrences for epochs: \n{}\n"
                                 "Returned signal will be only NaN."
                                 .format(list_of_epoch_names))
        return self._modified_copy(new_data)

    def trial_epochs_from_occurrences(self, occurrences=1):
        """
        Creates a generic epochs DataFrame with a number of trials based on
        sample length and number of occurrences specified.

        Example
        -------
        If signal._data has shape 3x100 and the signal is sampled at 100 Hz,
        trial_epochs_from_occurrences(occurrences=5) would generate a DataFrame
        with 5 trials (starting at 0, 0.2, 0.4, 0.6, 0.8 seconds).

        Note
        ----
        * The number of time samples must be evenly divisible by the number of
          occurrences.
        * Epoch indices behave similar to python list indices, so start is
          inclusive while end is exclusive.
        """

        trial_size = self.ntimes/occurrences/self.fs
        if self.ntimes % occurrences:
            m = 'Signal not evenly divisible into fixed-length trials'
            raise ValueError(m)

        starts = np.arange(occurrences)*trial_size
        ends = starts + trial_size
        return pd.DataFrame({
            'start': starts,
            'end': ends,
            'name': 'trial'
        })

    def to_epochs(self):
        """
        create list of epochs for when a signal is non-zero. intended
        for boolean-type signals that are sparsely True, but works
        for any signal type
        :return: epochs list for each segment where self._data > 0
        """
        s = np.array(self._data.shape)
        s[-1] = 1
        nz = np.concatenate((np.zeros(s, dtype=int),
                             (np.abs(self._data) > 0).astype(int),
                             np.zeros(s, dtype=int)),
                            axis=self._data.ndim-1)
        _, starts = np.where(np.diff(nz)>0)
        _, ends = np.where(np.diff(nz)<0)
        return pd.DataFrame({
            'start': starts/self.fs,
            'end': ends/self.fs,
            'name': self.name
        })


    def transform(self, fn, newname=None):
        '''
        Applies this signal's 2d .as_continuous() matrix representation to
        function fn, which must be a pure (curried) function of one argument.

        It then packs the return value of fn into a new signal object,
        identical to this one but with different data.

        Optional argument newname allows a new signal name to be returned.
        '''
        # x = self.as_continuous()   # Always Safe but makes a copy
        x = self._data  # Much faster; TODO: Test if throws warnings
        y = fn(x)
        newsig = self._modified_copy(y)
        if newname:
            newsig.name = newname
        return newsig

    def shuffle_time(self, rand_seed=None, mask=None):
        '''
        Shuffle a signal in time, applying a different shuffle to each channel.
        rand_seed - allows identical/pseudo-random shuffling when called on the same singal twice.
        mask - shuffling will take place only within the segment(s) where mask=True
        '''
        # x = self.as_continuous()   # Always Safe but makes a copy
        x = self._data.copy()  # Much faster; TODO: Test if throws warnings
        arr = np.arange(x.shape[1])
        if mask is None:
            arr0 = arr[np.isfinite(x[0, :])]
        else:
            arr0 = arr[mask.as_continuous()[0,:].astype(bool) & np.isfinite(x[0, :])]

        arr = arr0.copy()

        if rand_seed is not None:
            save_state = np.random.get_state()
            np.random.seed(rand_seed)

        for i in range(x.shape[0]):
            np.random.shuffle(arr)
            #arr=np.roll(arr, int(np.random.rand()*1000))
            x[i, arr0] = x[i, arr]

        if rand_seed is not None:
            # restore random state
            np.random.set_state(save_state)

        newsig = self._modified_copy(x)
        newsig.name = newsig.name + '_shuf'
        return newsig

    def nan_outliers(self, trim_outside_zscore=2.0):
        '''
        Tries to NaN out outliers from the signal. Outliers are defined
        as being values further than trim_outside_zscore stddevs from the mean.

        Arguments:
        ----------
        trim_outside_zscore: float
        Multiple of standard deviation that determines the range of
        'normal' versus 'outlier' values.

        Returns:
        --------
        A new copy of the signal with outliers NaN'd out.
        '''
        m = self.as_continuous()
        std = np.std(m)
        mean = np.mean(m)
        max_val = mean + std * trim_outside_zscore
        min_val = mean - std * trim_outside_zscore
        m[m < max_val] = np.nan
        m[m > min_val] = np.nan
        return self._modified_copy(m)

    def nan_mask(self, mask, remove_epochs=True):
        """
        NaN out all time points where signal mask is False
        :param mask: boolean signal
        :param remove_epochs: (True) if True, remove epochs overlapping the
                              nan-ed periods
        :return: copy of self with nan mask applied
        """
        m = self.as_continuous().copy()
        m[:, mask._data[0, :] == False] = np.nan
        if remove_epochs:
            return self._modified_copy(m).remove_epochs(mask)
        else:
            return self._modified_copy(m)

    def select_times(self, times, padding=0):

        if padding != 0:
            raise NotImplementedError    # TODO

        times = np.asarray(times)
        indices = np.round(times*self.fs).astype('i')
        data = self.as_continuous()

        subsets = [data[..., lb:ub] for lb, ub in indices]
        data = np.concatenate(subsets, axis=-1)
        return self._modified_copy(data, segments=times)

    def nan_times(self, times, padding=0):

        if padding != 0:
            raise NotImplementedError    # TODO

        times = np.asarray(times)
        indices = np.round(times*self.fs).astype('i')
        data = self.as_continuous().copy()
        for lb, ub in indices:
            data[..., lb:ub] = np.nan

        return self._modified_copy(data)

    def rasterize(self, fs=None):
        """
        A pass-through. We don't need to rasterize, since the
        signal is already a raster!
        """
        return self

    def as_continuous(self, mask=None):
        '''
        For SignalBase, return a signal _data variable. -- NOT COPIED!

        but if mask signal provided, do return a copy, with only masked
        portion
        '''
        if mask is None:
            return self._data
        else:
            return self._data[:, mask.as_continuous()[0, :]]


class PointProcess(SignalBase):
    '''
    Expects data to be a dictionary of the form:
        {<string>: <ndarray of spike times, one dimensional>}
    '''
#    @property
#    def _data(self):
#        if self._cached_data is not None:
#            pass
#        else:
#            log.info("matrix doesn't exist yet, "
#                     "generating from time dict")
#            self._generate_data()
#        return self._cached_data

    def __init__(self, fs, data, name, recording, chans=None, epochs=None,
                 segments=None, meta=None, safety_checks=True,
                 normalization='none', **other_attributes):
        '''
        Parameters
        ----------
        data : dictionary of event times in each channel
        epochs : {None, DataFrame}
           same as BaseSignal

        TODO : Safety checks:
            data.keys should match self.chans
            others?
        '''
        super().__init__(fs, data, name, recording, chans, epochs, segments,
                         meta, safety_checks, normalization)

        # number of channels specified by number of entries in data dictionary
        self.nchans = len(list(data.keys()))

        if safety_checks:
            if 'none' != normalization:
                raise ValueError ('normalization not supported for PointProcess signals')


    def _modified_copy(self, data, **kwargs):
        """
        For internal use when making various immutable copies of this signal.
        """
        attributes = self._get_attributes()
        attributes.update(kwargs)
        return PointProcess(data=data, safety_checks=False, **attributes)


    def rasterize(self, fs=None):
        """
        convert list of spike times to a raster of spike rate, with duration
        matching max end time in the event_times list

        by default, fs=self.fs, which can be preset to match other signals in a
        recording
        """
        if not fs:
            fs = self.fs

        if self.epochs is not None:
            max_epoch_time = self.epochs["end"].max()
        else:
            max_epoch_time = 0

        if max_epoch_time==0:
            max_event_times = [max(et) for et in self._data.values()]
            max_time = max(max_epoch_time, *max_event_times)
        else:
            max_time=max_epoch_time

        max_bin = np.int(np.round(fs*max_time))
        unit_count = len(self._data.keys())
        raster = np.zeros([unit_count, max_bin])

        # _data dictionary has one entry per cell.
        # The output raster should be cell X time
        cellids = sorted(self._data)
        for i, key in enumerate(cellids):
            for t in self._data[key]:
                b = int(np.floor(t*fs))
                if b < max_bin:
                    raster[i, b] += 1

        return RasterizedSignal(fs=fs, data=raster, name=self.name,
                                recording=self.recording, chans=cellids,
                                epochs=self.epochs, meta=self.meta)

    def as_continuous(self):
        return self.rasterize()._data

    def transform(self, fn, newname=None):
        '''
        Rasterize this signal then apply fn and return the result as
        a new signal.
        '''
        x = self.rasterize()
        y = fn(x._data)
        newsig = x._modified_copy(y)
        if newname:
            newsig.name = newname
        return newsig

    def save(self, dirpath, fmt='%.18e'):
        '''
        Save this signal to a HDF5 file + JSON sidecar.
        '''

        jsonfilepath,epochfilepath=self._save_metadata_to_dirpath(dirpath)
        hdf5filepath = self._save_data_to_h5(dirpath)

        return (hdf5filepath, jsonfilepath, epochfilepath)

    def as_file_streams(self, fmt='%.18e'):
        '''
        Returns 3 filestreams for this signal: the csv, json, and epoch.
        TODO: Better docs and a refactoring of this and save()
        '''
        # TODO: actually compute these instead of cheating with a tempfile
        files = {}
        filebase = self.recording + '.' + self.name
        h5file = filebase + '.h5'
        jsonfile = filebase + '.json'
        epochfile = filebase + '.epoch.csv'

        tmppath=tempfile.mkdtemp()

        temph5=self._save_data_to_h5(tmppath)
        th=io.open(temph5,'rb')
        files[h5file]=io.BytesIO(th.read())

        # Create textfile streams
        files[jsonfile] = io.StringIO()
        files[epochfile] = io.StringIO()

        self._save_metadata(files[epochfile],files[jsonfile], fmt)

        return files

#    @staticmethod
#    def load(path):
#        with h5py.File(path, 'r') as f:
#            fs = f.attrs['fs']
#            recording = f.attrs['recording']
#            name = f.attrs['name']
#            chans = json.loads(f.attrs['chans'])
#            meta = json.loads(f.attrs['meta'])
#            safety_checks = f.attrs['safety_checks']
#
#            epochs = None
#            data = {}
#            for key, dataset in f.items():
#                if 'epochs' in key:
#                    epochs = pd.read_hdf(path, key=key)
#                else:
#                    data[key] = np.array(dataset[:])
#
#            return PointProcess(fs=fs, data=data, name=name, recording=recording,
#                              chans=chans, epochs=epochs, meta=meta,
#                              safety_checks=safety_checks)

    def split_by_epochs(self, epochs_for_est, epochs_for_val):
        '''
        Returns a tuple of estimation and validation data splits: (est, val).
        Arguments should be lists of epochs that define the estimation and
        validation sets. est and val will have non-matching data NaN'd out.
        '''
        est = self.rasterize().select_epochs(epochs_for_est)
        val = self.rasterize().select_epochs(epochs_for_val)
        return (est, val)

    def jackknife_by_epoch(self, njacks, jack_idx, epoch_name,
                           tiled=True,
                           invert=False, excise=False):
        """
        convert to rasterized signal and create jackknife sets as
        described there.
        """
        sig = self.rasterize()
        return sig.jackknife_by_epoch(njacks, jack_idx, epoch_name,
                                      tiled, invert, excise)

    def decimate(self, stride=2, axis=-1):
        """
        for PointProcess, simply change the sampling rate
        """
        new_fs = int(self.fs/stride)
        return self._modified_copy(data=self._data, fs=new_fs)

    @classmethod
    def concatenate_time(cls, signals):
        '''
        Combines the signals along the time axis. All signals must have the
        same number of channels (and the same sampling rates?).
        '''
        # Make sure all objects passed are instances of the Signal class
        for signal in signals:
            if not isinstance(signal, PointProcess):
                raise ValueError('Cannot merge these signals')

        # Make sure that important attributes match up
        base = signals[0]
        for signal in signals[1:]:
            #if not base.fs == signal.fs:
            #    raise ValueError('Cannot concat signals with unequal fs')
            if not base.chans == signal.chans:
                raise ValueError('Cannot concat signals with unequal # of chans')

        # Now, concatenate data along time axis, adding an offset
        # to each successive signal to account for the duration of
        # the preceeding signals
        offset = 0
        for signal in signals:
            if offset==0:
                data=signal._data
            else:
                cellids = sorted(signal._data)
                for i, key in enumerate(cellids):
                    # append new data to list, after adding offset
                    data[key]=np.concatenate((data[key],(signal._data[key]+offset)))

            # increment offset by duration (sec) of current signal
            #offset += signal.ntimes / signal.fs
            offset += signal.epochs['end'].max()

        # basically do the same thing for epochs, using the Base routine
        epochs = _merge_epochs(signals)

        return PointProcess(
            name=base.name,
            recording=base.recording,
            chans=base.chans,
            fs=base.fs,
            meta=base.meta,
            data=data,
            epochs=epochs,
            safety_checks=False,
        )

    def append_time(self, new_signal):
        '''
        Combines the signals along the time axis. All signals must have the
        same number of channels (and the same sampling rates?).
        '''
        # Make sure all objects passed are instances of the Signal class
        if not isinstance(new_signal, PointProcess):
            raise ValueError('Wrong signal type to append')

        # Make sure that important attributes match up
        #if not self.fs == new_signal.fs:
        #    raise ValueError('Cannot append signal with unequal fs')
        if not self.chans == new_signal.chans:
            raise ValueError('Cannot append signal with unequal # of chans')

        # Now, concatenate data along time axis, adding an offset
        # to each successive signal to account for the duration of
        # the preceeding signals
        new_data=copy.deepcopy(self._data)
        cellids = sorted(self._data)
        #offset = self.ntimes / self.fs
        #offset = np.round(self.epochs['end'].max() * self.fs) / self.fs
        offset = self.epochs['end'].max()
        log.info('concatenate offset: %.15f', offset)

        for key in cellids:
            # append new data to list, after adding offset
            new_data[key]=np.concatenate((new_data[key],(new_signal._data[key]+offset)))

        # basically do the same thing for epochs, using the Base routine
        epochs = _merge_epochs([self,new_signal])

        return PointProcess(
            name=self.name,
            recording=self.recording,
            chans=self.chans,
            fs=self.fs,
            meta=self.meta,
            data=new_data,
            epochs=epochs,
            safety_checks=False
        )


    def extract_channels(self, chans, name=None):
        '''
        Returns a new signal object containing only the specified
        channel indices.
        '''
        # s is shorthand for slice. Return a 2D array.
        s = {c: self._data[c] for c in chans}
        if name is None:
            name = self.name
        return self._modified_copy(s, chans=chans, name=name)

    def extract_epoch(self, epoch, boundary_mode='exclude',
                      fix_overlap='first', allow_empty=False,
                      overlapping_epoch=None, mask=None, allow_incomplete=False):
        '''
        Extracts all occurances of epoch from the signal.

        Parameters
        ----------
        epoch : {string, Nx2 array}
            If string, name of epoch (as stored in internal dataframe) to
            extract. If Nx2 array, the first column indicates the start time
            (in seconds) and the second column indicates the end time
            (in seconds) to extract.

            allow_empty: if true, returns empty matrix if no valid epoch
            matches. otherwise, throw error when this happens

        boundary_mode, fix_overlap: parameters passed through to
            get_epoch_indices

        allow_empty: {False, boolean}

        mask: {None, signal}
            if provided, only extract epochs overlapping periods where
            mask.as_continuous()==True in all time bins

        Returns
        -------
        epoch_data : dictionary of 2D arrays
            epoch_data[chan] = N x 2 array. first col=rep #, second col= time of event within rep

        '''

        if type(epoch) is str:
            epoch_bounds = self.get_epoch_bounds(epoch,
                                                  boundary_mode=boundary_mode,
                                                  fix_overlap=fix_overlap,
                                                  mask=mask,
                                                  allow_incomplete=allow_incomplete)
        else:
            epoch_bounds = epoch

        if epoch_bounds.size == 0:
            if allow_empty:
                return np.empty([0, 0, 0])
            else:
                raise IndexError("No matching epochs to extract for: %s\n"
                                 "In signal: %s", epoch, self.name)

        epoch_data = dict()

        for j, c in enumerate(self._data.keys()):
            epoch_data[c] = np.zeros((0, 2))
            d = np.array([])
            t = np.array([])
            for i, (lb, ub) in enumerate(epoch_bounds):
                _d = self._data[c][(self._data[c] >= lb) & (self._data[c] < ub)] - lb
                d = np.append(d, _d)
                t = np.append(t, np.full_like(_d, i))
            epoch_data[c] = np.concatenate((np.expand_dims(t,1), np.expand_dims(d,1)), axis=1)

        return epoch_data


class TiledSignal(SignalBase):
    '''
    Expects data to be a dictionary of the form:
        {<string>: <ndarray of stim data, two dimensional>}
    '''
    def __init__(self, fs, data, name, recording, chans=None, epochs=None,
                 segments=None, meta=None, safety_checks=True,
                 normalization='none', **other_attributes):
        '''
        Parameters
        ----------
        data : dictionary of event times in each channel
        epochs : {None, DataFrame}
           same as BaseSignal

        TODO : Safety checks:
            data.keys should match self.chans
            others?
        '''
        super().__init__(fs, data, name, recording, chans, epochs, segments,
                         meta, safety_checks, normalization)

        # number of channels dim 0 of each entry in dictionary. need to match!
        chancount = None
        for k, v in data.items():
            this_chancount = v.shape[0]
            if chancount and this_chancount != chancount:
                raise ValueError('channel count does not match across tiled signal dictionary')
            chancount = this_chancount

        self.nchans = chancount

        if safety_checks:
            if 'none' != normalization:
                raise ValueError('normalization not supported for TiledSignal')

    def rasterize(self, fs=None):
        '''
        Create a rasterized version of the signal and return it

        fs is not used but in parameters for compatibility with PointProcess
        '''
        maxtime = np.max(self.epochs["end"])
        maxbin = self.shape[1]
        if self.fs*maxtime > maxbin:
            maxbin = int(self.fs*maxtime)
        tags = list(self._data.keys())
        chancount = list(self._data[tags[0]].shape[:-1])

        z = np.zeros(chancount + [maxbin])
        zsig = RasterizedSignal(fs=self.fs, data=z, name=self.name,
                                recording=self.recording, chans=self.chans,
                                epochs=self.epochs, meta=self.meta)
        signal = zsig.replace_epochs(self._data)

        # replace nans with zeros. Assume that the signal was valid but zero
        s = signal._data.copy()
        s[np.isnan(signal._data)] = 0
        signal = signal._modified_copy(s)

        return signal

    def as_continuous(self):
        return self.rasterize()._data

    def transform(self, fn, newname=None):
        '''
        Rasterize this signal then apply fn and return the result as
        a new signal.
        '''
        x = self.rasterize()
        y = fn(x._data)
        newsig = x._modified_copy(y)
        if newname:
            newsig.name = newname
        return newsig

    def save(self, dirpath, fmt='%.18e'):
        '''
        Save this signal to a HDF5 file + JSON sidecar.
        '''

        jsonfilepath,epochfilepath=self._save_metadata_to_dirpath(dirpath)
        hdf5filepath = self._save_data_to_h5(dirpath)

        return (hdf5filepath, jsonfilepath, epochfilepath)

    def as_file_streams(self, fmt='%.18e'):
        '''
        Returns 3 filestreams for this signal: the csv, json, and epoch.
        TODO: Better docs and a refactoring of this and save()
        '''
        # TODO: actually compute these instead of cheating with a tempfile
        files = {}
        filebase = self.recording + '.' + self.name
        h5file = filebase + '.h5'
        jsonfile = filebase + '.json'
        epochfile = filebase + '.epoch.csv'

        tmppath=tempfile.mkdtemp()

        temph5=self._save_data_to_h5(tmppath)
        th=io.open(temph5,'rb')
        files[h5file]=io.BytesIO(th.read())

        # Create textfile streams
        files[jsonfile] = io.StringIO()
        files[epochfile] = io.StringIO()

        self._save_metadata(files[epochfile],files[jsonfile], fmt)

        return files

#    @staticmethod
#    def load(path):
#        with h5py.File(path, 'r') as f:
#            fs = f.attrs['fs']
#            recording = f.attrs['recording']
#            name = f.attrs['name']
#            chans = json.loads(f.attrs['chans'])
#            meta = json.loads(f.attrs['meta'])
#
#            epochs = None
#            data = {}
#            for key, dataset in f.items():
#                if 'epochs' in key:
#                    epochs = pd.read_hdf(path, key=key)
#                else:
#                    data[key] = np.array(dataset[:])
#
#            return TiledSignal(fs=fs, data=data, name=name,
#                                    recording=recording, chans=chans,
#                                    epochs=epochs, meta=meta)

    def split_by_epochs(self, epochs_for_est, epochs_for_val):
        '''
        Returns a tuple of estimation and validation data splits: (est, val).
        Arguments should be lists of epochs that define the estimation and
        validation sets. Both est and val will have non-matching data NaN'd out.
        '''
        est = self.rasterize().select_epochs(epochs_for_est)
        val = self.rasterize().select_epochs(epochs_for_val)
        return (est, val)

    def jackknife_by_epoch(self, njacks, jack_idx, epoch_name,
                           tiled=True, invert=False, excise=False):
        """
        convert to rasterized signal and create jackknife sets as
        described there.
        """
        sig = self.rasterize()
        return sig.jackknife_by_epoch(njacks, jack_idx, epoch_name,
                                      tiled, invert, excise)

    @classmethod
    def concatenate_time(cls, signals):
        '''
        Combines the signals along the time axis. All signals must have the
        same number of channels (and the same sampling rates?).
        '''
        # Make sure all objects passed are instances of the Signal class
        for signal in signals:
            if not isinstance(signal, TiledSignal):
                raise ValueError('Cannot merge these signals')

        # Make sure that important attributes match up
        base = signals[0]
        for signal in signals[1:]:
            #if not base.fs == signal.fs:
            #    raise ValueError('Cannot concat signals with unequal fs')
            if not base.chans == signal.chans:
                raise ValueError('Cannot concat signals with unequal # of chans')

        # Data is concatenated simply by merging the dictionaries. Assuming
        # no duplicate keys or that if there are duplicates, self supercedes
        # new_signal.
        new_data={}
        for signal in signals:
            new_data={**new_data,**signal._data}

        # append epochs, adding offset to account for length of self
        epochs = _merge_epochs(signals)

        return TiledSignal(
            name=base.name,
            recording=base.recording,
            chans=base.chans,
            fs=base.fs,
            meta=base.meta,
            data=new_data,
            epochs=epochs,
            safety_checks=False
        )

    def append_time(self, new_signal):
        '''
        Combines the signals along the time axis. All signals must have the
        same number of channels (and the same sampling rates?).
        '''
        # Make sure all objects passed are instances of the Signal class
        if not isinstance(new_signal, TiledSignal):
            raise ValueError('Wrong signal type to append')

        # Make sure that important attributes match up
        if not self.fs == new_signal.fs:
            raise ValueError('Cannot append signal with unequal fs')
        if not self.chans == new_signal.chans:
            raise ValueError('Cannot append signal with unequal # of chans')

        # Data is concatenated simply by merging the dictionaries. Assuming
        # no duplicate keys or that if there are duplicates, self supercedes
        # new_signal.
        new_data={**self._data,**new_signal._data}

        # append epochs, adding offset to account for length of self
        epochs = _merge_epochs([self,new_signal])

        return TiledSignal(
            name=self.name,
            recording=self.recording,
            chans=self.chans,
            fs=self.fs,
            meta=self.meta,
            data=new_data,
            epochs=epochs,
            safety_checks=False
        )


class RasterizedSignalSubset(SignalBase):
    '''
    Expects data to be a list of lists.
    '''
    pass

# -----------------------------------------------------------------------------
# Functions that work on multiple signal objects

def list_signals(directory):
    '''
    Returns a list of all CSV/JSON pairs files found in DIRECTORY,
    Paths are relative, not absolute.
    '''
    files = os.listdir(directory)
    return _list_json_files(files)

def _list_json_files(files):
    '''
    Given a list of files, return the file basenames (i.e. no extensions)
    that for which a .CSV and a .JSON file exists.
    '''
    just_fileroot = lambda f: os.path.splitext(os.path.basename(f))[0]
    jsons = {just_fileroot(f) for f in files if f.endswith('.json')}
    csvs = {just_fileroot(f) for f in files if f.endswith('.csv')}
    return list(jsons.intersection(csvs))

def load_signal(basepath):
    '''
    Generic signal loader. Load JSON file, figure out signal type and
    call appropriate loader
    '''
    csvfilepath = basepath + '.csv'
    h5filepath = basepath + '.h5'
    epochfilepath = basepath + '.epoch.csv'
    jsonfilepath = basepath + '.json'
    if os.path.isfile(epochfilepath):
        epochs = pd.read_csv(epochfilepath)
    else:
        epochs = None
    # TODO: reduce code duplication and call load_from_streams
    with open(jsonfilepath, 'r') as f:
        js = json.load(f)

    if 'signal_type' in js.keys():
        signal_type=js['signal_type']
    else:
        signal_type="nems0.signal.RasterizedSignal"

    if 'RasterizedSignal' in signal_type:
        mat = pd.read_csv(csvfilepath, header=None).values
        mat = mat.astype('float')
        mat = np.swapaxes(mat, 0, 1)

        s = RasterizedSignal(name=js['name'],
                    chans=js.get('chans', None),
                    epochs=epochs,
                    recording=js['recording'],
                    fs=js['fs'],
                    meta=js['meta'],
                    data=mat)

    elif 'PointProcess' in signal_type:
        with h5py.File(h5filepath, 'r') as f:
            data = {}
            for key, dataset in f.items():
                data[key] = np.array(dataset[:])

        s = PointProcess(name=js['name'],
                    chans=js.get('chans', None),
                    epochs=epochs,
                    recording=js['recording'],
                    fs=js['fs'],
                    meta=js['meta'],
                    data=data)

    elif 'TiledSignal' in signal_type:
        with h5py.File(h5filepath, 'r') as f:
            data = {}
            for key, dataset in f.items():
                data[key] = np.array(dataset[:])

        s = TiledSignal(name=js['name'],
                    chans=js.get('chans', None),
                    epochs=epochs,
                    recording=js['recording'],
                    fs=js['fs'],
                    meta=js['meta'],
                    data=data)

    else:
        raise ValueError('signal_type unknown')

    # NOTE: Moved this outside of call to initializer because
    #       some saved signals don't have segments in their json sidecar.
    s.segments = np.array(js.get('segments', s.segments))

    return s

def load_signal_from_streams(data_stream, json_stream, epoch_stream=None):
    ''' Loads from BytesIO objects rather than files. epoch stream was formerly
        csv stream, but this could be an hdf5 file (or something else?)
    '''
    # Read the epochs stream if it exists
    epochs = pd.read_csv(epoch_stream) if epoch_stream else None
    # Read the json metadata
    js = json.load(json_stream)

    if 'signal_type' in js.keys():
        signal_type=js['signal_type']
    else:
        signal_type="nems0.signal.RasterizedSignal"

    if 'RasterizedSignal' in signal_type:
        mat = pd.read_csv(data_stream, header=None).values
        mat = mat.astype('float')
        mat = np.swapaxes(mat, 0, 1)

        s = RasterizedSignal(name=js['name'],
                    chans=js.get('chans', None),
                    epochs=epochs,
                    recording=js['recording'],
                    fs=js['fs'],
                    meta=js['meta'],
                    data=mat)

    elif 'PointProcess' in signal_type:
        with h5py.File(data_stream, 'r') as f:
            data = {}
            for key, dataset in f.items():
                data[key] = np.array(dataset[:])

        if not data:
            warnings.warn("Tried to load data stream {0} but data object"
                             "ended up empty. Potential bug upstream?"
                             .format(data_stream))

        s = PointProcess(name=js['name'],
                    chans=js.get('chans', None),
                    epochs=epochs,
                    recording=js['recording'],
                    fs=js['fs'],
                    meta=js['meta'],
                    data=data)
        max_time=epochs['end'].max()

    elif 'TiledSignal' in signal_type:
        with h5py.File(data_stream, 'r') as f:
            data = {}
            for key, dataset in f.items():
                data[key] = np.array(dataset[:])

        s = TiledSignal(name=js['name'],
                    chans=js.get('chans', None),
                    epochs=epochs,
                    recording=js['recording'],
                    fs=js['fs'],
                    meta=js['meta'],
                    data=data)

    else:
        raise ValueError('signal_type unknown')

    s.segments = np.array(js.get('segments', s.segments))

    return s


def load_rasterized_signal(basepath):
    csvfilepath = basepath + '.csv'
    epochfilepath = basepath + '.epoch.csv'
    jsonfilepath = basepath + '.json'
    # TODO: reduce code duplication and call load_from_streams
    mat = pd.read_csv(csvfilepath, header=None).values
    if os.path.isfile(epochfilepath):
        epochs = pd.read_csv(epochfilepath)
    else:
        epochs = None
    mat = mat.astype('float')
    mat = np.swapaxes(mat, 0, 1)
    with open(jsonfilepath, 'r') as f:
        js = json.load(f)
        s = RasterizedSignal(name=js['name'],
                    chans=js.get('chans', None),
                    epochs=epochs,
                    recording=js['recording'],
                    fs=js['fs'],
                    meta=js['meta'],
                    data=mat)
        s.segments = js.get('segments', s.segments)
        return s



################################################################################
# Signals
################################################################################
def merge_selections(signals):
    '''
    Returns a new signal object by combining a list of signals of the same
    shape that are assumed to be non-overlapping selections. The returned
    signal will have identical metadata to the first signal in the list.

    For signals to be non-overlapping, every corresponding element of
    every signal must be NaN unless it is NaN in all other signals, or
    it is identical to all other values that are non-NaN.

    Ex:
    s1: [   1,   2,   3, NaN, NaN, NaN]
    s2: [ NaN, NaN, NaN,   4,   5,   6]

    would merge to:
    s3: [   1,   2,   3,   4,   5,   6]

    But this would result in error:
    s4: [   1,   2,   3,   4, NaN, NaN]
    s5: [ NaN, NaN, NaN,   4,   5,   6]

    Because the index position of  4 was non-NaN in more than one signal.
    '''

    # Check that all signals have the same shape, fs, and chans
    for s in signals:
        if s.shape != signals[0].shape:
            raise ValueError("All signals must have the same shape.")
        if s.fs != signals[0].fs:
            raise ValueError("All signals must have the same fs.")
        if s.chans != signals[0].chans:
            raise ValueError("All signals must have the same chans.")

    # Make a big 3D array from the 2D arrays
    arys = [s.as_continuous() for s in signals]
    bigary = np.stack(arys, axis=2)

    # If there are no overlapping values, then nanmean() will be equal
    # to the value found in each position
    the_mean = np.nanmean(bigary, axis=2)
    if type(signals[0]._data[0][0]) is np.bool_:
        return signals[0]._modified_copy(the_mean)
    else:
        for a in arys:
            if not np.array_equal(a[np.isfinite(a)],
                                  the_mean[np.isfinite(a)]):

                raise ValueError("Overlapping, unequal non-NaN values"
                                 "found in signal {}."
                                 .format(signals[0].name))

        # Use the first signal as a template for setting fs, chans, etc.
        return signals[0]._modified_copy(the_mean)


def jackknife_inverse_merge(sig_list):
    """
    given a list of signals, merge into a single signal.
    superceded by merge_selections?
    """

    m = sig_list[0].as_continuous()
    for s in sig_list[1:]:
        m2 = s.as_continuous()
        gidx = np.isfinite(m2[0, :])
        m[:, gidx] = m2[:, gidx]
    sig_new = sig_list[0]._modified_copy(data=m)
    return sig_new


def join_signal_subsets(subsets):
    # TODO
    raise NotImplementedError
    return Signal(...)


def split_signal_to_subsets(signal):
    # TODO
    # Maybe just existing jackknifing method with minor changes?
    raise NotImplementedError
    return [SignalSubset(...) for data, fakepochs in something]


def concatenate_channels(cls, signals):
    '''
    Given signals=[sig1, sig2, sig3, ..., sigN], concatenate all channels
    of [sig2, ...sigN] as new channels on sig1. All signals must be equal-
    length time series sampled at the same rate (i.e. ntimes and fs are the
    same for all signals).
    '''
    for signal in signals:
        if not isinstance(signal, cls):
            raise ValueError('Cannot merge these signals')

    base = signals[0]
    for signal in signals[1:]:
        if not base.fs == signal.fs:
            raise ValueError('Cannot append signal with different fs')
        if not base.ntimes == signal.ntimes:
            raise ValueError('Cannot append signal with different channels')

    raise NotImplementedError

    # TODO get this working for other subtypes or throw an error.
    # this is called by plotting functions, so I think it only needs to
    # support RasterizedSignal
    data = np.concatenate([s.as_continuous() for s in signals], axis=0)

    chans = []
    for signal in signals:
        if signal.chans:
            chans.extend(signal.chans)

    epochs=signals[0].epochs

    return RasterizedSignal(
        name=base.name,
        recording=base.recording,
        chans=chans,
        fs=base.fs,
        meta=base.meta,
        epochs=epochs,
        data=data,
        safety_checks=False
        )

def _split_epochs(epochs,split_time):
    if epochs is None:
        lepochs = None
        repochs = None
    else:
        mask = epochs['start'] < split_time
        lepochs = epochs.loc[mask]
        mask = epochs['end'] > split_time
        repochs = epochs.loc[mask]
        repochs[['start', 'end']] -= split_time

    # If epochs were present initially but missing after split,
    # raise a warning.
    portion = None
    if lepochs.size == 0:
        portion = 'first'
    elif repochs.size == 0:
        portion = 'second'
    if portion:
        warnings.warn("Epochs for {0} portion of signal"
                             "ended up empty after splitting by time."
                             .format(portion))

    return lepochs, repochs

def _merge_epochs(signals):
    # Merge the epoch tables. For all signals after the first signal,
    # we need to offset the start and end indices to ensure that they
    # reflect the correct position of the trial in the merged array.
    offset = 0
    epochs = []
    for signal in signals:
        ti = signal.epochs.copy()
        ti['end'] += offset
        ti['start'] += offset
        offset += signal.ntimes/signal.fs
        epochs.append(ti)
    return pd.concat(epochs, ignore_index=True)


def _normalize_data(data, normalization='minmax', d=None, g=None, sig=None, mask=None):

    if (d is not None) and (g is not None):
        data_out = (data - d) / g
        
    elif normalization == 'none':
        d = np.zeros([data.shape[0], 1])
        g = np.ones([data.shape[0], 1])
        return data, d, g

    elif normalization == 'minmax':
        d = np.nanmin(data, axis=1, keepdims=True)
        g = np.nanmax(data, axis=1, keepdims=True) - d
        #d = np.nanmin(data)
        #g = np.nanmax(data) - d
        # avoid divide-by-zero
        g[g == 0] = 1
        data_out = (data - d) / g

        # force "quiet" stim to be true zero
        data_out[data_out<1e-6]=0

    elif normalization == 'meanstd':
        d = np.nanmean(data, axis=1, keepdims=True)
        g = np.nanstd(data, axis=1, keepdims=True)
        # avoid divide-by-zero
        g[g == 0] = 1
        data_out = (data - d) / g

    elif normalization == 'spont':
        try:
            prestimsilence = sig.extract_epoch('PreStimSilence', mask=mask)
            if prestimsilence.shape[-1] > 0:
                if len(prestimsilence.shape) == 3:
                    d = np.nanmean(prestimsilence, axis=(0, 2))
                else:
                    d = np.nanmean(prestimsilence)
            else:
                # special case where the epochs included in mask don't have PreStimSilence,
                # so we get it elsewhere. Designed for CPN data...
                prestimsilence = sig.extract_epoch('TRIALPreStimSilence')
                if len(prestimsilence.shape) == 3:
                    d = np.nanmean(prestimsilence, axis=(0, 2))
                else:
                    d = np.nanmean(prestimsilence)
        except:
            raise ValueError("Can't find prestim silence to use for PSTH calculation")

        g = np.array([1.0])
        data_out = data - d

    else:
        raise ValueError(f'Signal normalization format {normalization} unknown')

    return data_out, d, g
