"""Defines modelspec object and helper functions."""

import copy
import importlib
import json
import logging
import os
import re
import typing
from functools import partial
import inspect

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

import nems0
import nems0.uri
import nems0.utils
import nems0.recording
from nems0.fitters.mappers import simple_vector

log = logging.getLogger(__name__)

# Functions for saving, loading, and evaluating modelspecs

# TODO: In retrospect, this should have been a class, just like Recording.
#       Refactoring would not be too hard and would shorten many of these
#       function names. If you do so, see /docs/planning/models.py and
#       bring the ideas into this file, then delete it from docs/planning.


class ModelSpec:
    """Defines a model based on a NEMS modelspec.

    Long goes here. TODO docs
    """

    def __init__(self, raw=None, phis=None, fit_index=0, cell_index=0,
                 jack_index=0, recording=None, cell_count=1, fit_count=1, jack_count=1):
        """Initialize the modelspec.

        TODO more details
        a modelspec can have multiple fits, each of which contains a different set of phi values.

        :param dict raw: Nested list of dictionaries. Equivalent of the old NEMS modelspec. The first level is a
            list of cells, each of which is a list of lists. The second level is a list of fits, each of which
            is a list of dictionaries. The third level is a list of jacknifes, each of which is a list of
            dictionaries. Each dictionary specifies a module, or one step in the model.
        :param list phis: The free parameters.
        :param int fit_index: Index of which fit to reference when multiple are present. Defaults to 0.
        :param int cell_index: Index of which cell to reference when multiple are present. Defaults to 0.
        :param int jack_index: Index of which jacknife to reference when multiple are present. Defaults to 0.
        :param int cell_count: Defaults to 1.
        :param int fit_count: Defaults to 1.
        :param int jack_count: Defaults to 1.
        :param recording: recording for evaluation & plotting. Defaults to None
        """
        if raw is None:
            # initialize with empty lists (no modules)
            raw = np.full((cell_count,fit_count,jack_count), None)
            for i in range(cell_count):
                for j in range(fit_count):
                    for k in range(jack_count):
                        raw[i,j,k] = []
                        
        elif type(raw) is list:
            # compatible with load_modelspec -- read in list of lists
            if type(raw[0]) is not list:
                raw = [raw]
            
            r = np.full((len(raw)), None)
            for i, _r in enumerate(raw):
                r[i] = []
                for __r in _r:
                    # check if this is a new NemsModule object
                    _f = _lookup_fn_at(__r['fn'])
                    if inspect.isclass(_f):
                        r[i].append(_f(**__r))
                    else:
                        # if not, just save dict to module
                        r[i].append(__r)
                        
            # raw array will be reshaped to 1 x 1 x jack_count unless ms_shape is specified in meta dict
            ms_shape = raw[0][0]['meta'].get('shape', [1,1,len(raw)])
            if len(raw) == (ms_shape[0]*ms_shape[1]*ms_shape[2]):
                raw = np.reshape(r, ms_shape)  # cell_count x fit_count x jack_count
            else:
                raw = np.reshape(r,[1,1,len(raw)])
        else:
            cell_count=raw.shape[0]
            fit_count=raw.shape[1]
            jack_count=raw.shape[2]
            
        # otherwise, assume raw is a properly formatted 3D numpy array (cell_count X fit_count X jack_count) of lists of Modules
        
        self.raw = raw
        self.phis = [] if phis is None else phis
        self.rec_list = [None] * cell_count

        # references the default phi
        self.fit_index = fit_index
        self.jack_index = jack_index
        self.cell_index = cell_index
        self.mod_index = 0
        self.plot_epoch = 'REFERENCE'
        self.plot_occurrence = 0
        self.plot_channel = 0   # channel of pred/resp to plot in timeseries (for population models)
        self.recording = recording  # default recording for evaluation & plotting
        self.fast_eval = False
        self.fast_eval_start = 0
        self.freeze_rec = None
        self.shared_count = 0
        
        # cache the tf model if it exists
        self.tf_model = None

    def __getitem__(self, key):
        """Get the given item from the modelspec.

        Overloaded in order to allow accessing of other elements. Key can be either an int,
        a `slice` object, `meta`, or `phi`.

        :param key: Index or object to retrieve from the modelspec.
        :return: Either a module of the modelspec, or the `phi`, the `meta`, or the slice of the data.
        :raises ValueError: Raised if `key` out of bounds or not one of the above.
        """
        try:
            return self.get_module(key)
        except IndexError:
            if type(key) is slice:
                return [self.raw[self.cell_index, self.fit_index, self.jack_index][ii]
                        for ii in range(*key.indices(len(self)))]
            elif key == 'meta':
                return self.meta
            elif key == 'phi':
                return self.phi()
            else:
                raise ValueError('key {} not supported'.format(key))

    def __setitem__(self, key, val):
        """Update the raw dict of the modelspec.

        Updates the current modelspec raw dict at the current `cell_index`, `fit_index`, and `jack_index`.

        :param int key: Which index in the modelspec to update the value of
        :param val: The value to update to.
        :return: Self, updated.
        :raises ValueError: If unable to set
        """
        try:
            # Try converting types like np.int64 instead of just
            # throwing an error.
            self.raw[self.cell_index, self.fit_index, self.jack_index][int(key)] = val
        except ValueError:
            raise ValueError('key {} not supported'.format(key))
        return self

    def __iter__(self):
        """Set the `mod_index` to zero for iterators.

        :return: self, with updated `mod_index`
        """
        self.mod_index = 0
        return self

    def __next__(self):
        """Return the proper index of the modelspec for iterators, and update the `mod_index`.

        :return: The current module `mod_index`.
        """
        try:
            ret = self[self.mod_index]
            self.mod_index += 1
            return ret
        except ValueError:
            raise StopIteration

    def __repr__(self):
        """Overloaded repr.

        :return: Repr of the modelspec `raw` dict.
        """
        return repr(self.raw)

    def __str__(self):
        """Overloaded str.

        :return str: String newline concat of the module functions.
        """
        x = [m['fn'] for m in self.raw[self.cell_index, self.fit_index, self.jack_index]]
        return "\n".join(x)

    def __len__(self):
        """Overloaded len.

        :return int: Length of the raw dict at the current `cell_index`, `fit_index`, and
            `jack_index`..
        """
        return len(self.raw[self.cell_index, self.fit_index, self.jack_index])

    def copy(self, fit_index=None, jack_index=None):
        """Generate a deep copy of the modelspec.

        :param int fit_index:
        :param int jack_index:
        :return: A deep copy of the modelspec (subset of modules if specified).
        """
        raw = copy.deepcopy(self.raw)
        #meta_save = [copy.deepcopy(raw.flatten()[i][0].get('meta',{})) for i in range(len(raw.flatten()))]

        if fit_index is not None:
            raw = raw[:, fit_index:(fit_index+1), :]
            _f = 0
        else:
            _f = self.fit_index
        if jack_index is not None:
            raw = raw[:, :, jack_index:(jack_index + 1)]
            _j = 0
        else:
            _j = self.jack_index

        #for i, r in enumerate(raw.flatten()):
        #    r[0]['meta'] = meta_save[i]

        m = ModelSpec(raw, fit_index=_f, cell_index=self.cell_index,
                      jack_index=_j)
        m.shared_count = self.shared_count
        m.rec_list = self.rec_list.copy()

        return m

    def get_module(self, mod_index=None):
        """Get the requested module.

        Returns the raw dict at the `mod_index`, and current `cell_index`, `fit_index`, and
        `jack_index`.

        :param int mod_index: Index of module to return, defaults to `mod_index` if `None`.
        :return: Single module from current `fit_index`. Does not create a copy.
        """
        if mod_index is None:
            mod_index = self.mod_index
        return self.raw[self.cell_index, self.fit_index, self.jack_index][mod_index]

    def drop_module(self, mod_index=None, in_place=False):
        """Drop a module from the modelspec.

        Return a new modelspec with the module dropped, or optionally drop the module in place.

        :param int mod_index: Index of module ot drop.
        :param bool in_place: Whether or not to drop in place, or return a copy.
        :return: None if in place, otherwise a new modelspec without the dropped module.
        """
        if mod_index is None:
            mod_index = len(self)-1

        if in_place:
            for fit in self.raw.flatten():
                del fit[mod_index]
            return None

        else:
            raw_copy = copy.deepcopy(self.raw)
            for fit in self.raw.flatten():
                del fit[mod_index]
            new_spec = ModelSpec(raw_copy)
            new_spec.cell_count=self.cell_count
            new_spec.recording = self.recording
            return new_spec

    @property
    def modules(self):
        """All of the modules for the current `cell_index`, `fit_index`, and `jack_index`."""
        return self.raw[self.cell_index, self.fit_index, self.jack_index]

    # TODO support for multiple recording views/modelspec jackknifes (jack_count>0)
    #  and multiple fits (fit_count>0)

    def tile_fits(self, fit_count=1):
        """Create `fit_count` sets of fit parameters to allow for multiple fits.

        Useful for n-fold cross validation or starting from multiple initial
        conditions. Values of each phi are copied from the existing first
        value. Applied in-place.

        :param int fit_count: Number of tiles to create.
        :return: Self.
        """
        #meta_save = self.meta

        fits = [copy.deepcopy(self.raw[:, 0:1, :]) for i in range(fit_count)]
        self.raw = np.concatenate(fits, axis=1)

        #for r in self.raw.flatten():
        #    r[0]['meta'] = meta_save

        return self

    def tile_jacks(self, jack_count=0):
        """Create `jack_count` sets of fit parameters to allow for multiple jackknifes.

        Useful for n-fold cross validation or starting from multiple initial
        conditions. Values of each phi are copied from the existing first
        value. Applied in-place.

        :param int jack_count: Number of tiles to create.
        :return: Self.
        """
        #meta_save = self.meta

        jacks = [copy.deepcopy(self.raw[:, :, 0:1]) for i in range(jack_count)]
        self.raw = np.concatenate(jacks, axis=2)

        #for r in self.raw.flatten():
        #    r[0]['meta'] = meta_save

        return self

    @property
    def cell_count(self):
        """Number of cells (sets of phi values) in this modelspec."""
        return self.raw.shape[0]

    @property
    def fit_count(self):
        """Number of fits (sets of phi values) in this modelspec."""
        return self.raw.shape[1]

    @property
    def jack_count(self):
        """Number of jackknifes (sets of phi values) in this modelspec."""
        return self.raw.shape[2]

    def set_cell(self, cell_index=None):
        """Set the `cell_index`. Done in place.

        :param int cell_index: The updated `cell_index`.
        :return: Self.
        """
        if cell_index is not None:
            self.cell_index = cell_index
        return self

    def set_fit(self, fit_index):
        """Set the `fit_index`. Done in place.

        :param int fit_index: The updated `fit_index`.
        :return: Self.
        """
        if fit_index is None:
            pass
        elif fit_index > (self.fit_count - 1):
            raise ValueError('fit_index greater than fit_count-1')
        elif fit_index < self.fit_count*-1:
            raise ValueError('negative fit_index smaller than fit_count')
        else:
            self.fit_index = fit_index

        return self

    def set_jack(self, jack_index=None):
        """Set the `jack_index`. Done in place.

        :param int jack_index: The updated `jack_index`.
        :return: Self.
        """
        if jack_index is not None:
            self.jack_index = jack_index

        return self

    def fits(self):
        """List of modelspecs, one for each fit, for compatibility with some old functions."""
        return [ModelSpec(self.raw, jack_index=f) for f in range(self.jack_count)]

    @property
    def meta(self):
        """Dict of meta information."""
        if self.raw[self.cell_index, 0, 0][0].get('meta') is None:
            self.raw[self.cell_index, 0, 0][0]['meta'] = {}
        return self.raw[self.cell_index, 0, 0][0]['meta']

    @property
    def recording(self):
        """recording for current cell_index."""
        return self.rec_list[self.cell_index]

    @recording.setter
    def recording(self, rec):
        self.rec_list[self.cell_index] = rec
        
    @property
    def modelspecname(self):
        """Name of the modelspec."""
        return '-'.join([m.get('id', 'BLANKID') for m in self.modules])

    def fn(self):
        """List of fn for each module."""
        return [m['fn'] for m in self.raw[self.cell_index, self.fit_index, self.jack_index]]

    @property
    def phi(self, fit_index=None, mod_idx=None):
        """The free parameters for the model.

        :param int fit_index: Which model fit to use (defaults to `fit_index`).
        :return: List of phi dictionaries, or None for modules with no phi.
        """
        if fit_index is None:
            fit_index = self.fit_index
        if mod_idx is None:
            return [m.get('phi') for m in self.raw[self.cell_index, fit_index, self.jack_index]]
        else:
            return self.raw[self.cell_index, fit_index, self.jack_index][mod_idx].get('phi')

    @property
    def phi_mean(self, mod_idx=None):
        """Mean of phi across fit_indexes and/or jack_indexes.

        :param int mod_idx: Which module to use (default all modules).
        :return: List of phi dictionaries, mean of each value.
        """
        if self.jack_count * self.fit_count == 1:
            return self.phi

        if type(mod_idx) is list:
            mod_range = mod_idx
        elif mod_idx is not None:
            mod_range = [mod_idx]
        else:
            mod_range = range(len(self.raw[0]))

        phi = []
        raw = []
        for sublist in self.raw[self.cell_index]:
            for item in sublist:
                raw.append(item)
        for mod_idx in mod_range:
            p = {}
            for k in raw[0][mod_idx]['phi'].keys():
                maxdim = len(raw[0][mod_idx]['phi'][k].shape)
                p[k] = np.mean(np.concatenate([np.expand_dims(f[mod_idx]['phi'][k], axis=maxdim)
                                               for f in raw], axis=maxdim), axis=maxdim, keepdims=False)
            phi.append(p)

        return phi

    @property
    def phi_sem(self, mod_idx=None):
        """SEM of phi across fit_indexes and/or jack_indexes.

        :param int mod_idx: Which module to use (default all modules).
        :return: List of phi dictionaries, jackknife sem of each value.
        """
        modelcount = self.jack_count * self.fit_count
        if modelcount == 1:
            return self.phi

        if type(mod_idx) is list:
            mod_range = mod_idx
        elif mod_idx is not None:
            mod_range = [mod_idx]
        else:
            mod_range = range(len(self.raw[0]))

        phi = []
        raw = []
        for sublist in self.raw[self.cell_index]:
            for item in sublist:
                raw.append(item)
        for mod_idx in mod_range:
            p = {}
            for k in raw[0][mod_idx]['phi'].keys():
                maxdim = len(raw[0][mod_idx]['phi'][k].shape)
                p[k] = np.std(np.concatenate(
                    [np.expand_dims(f[mod_idx]['phi'][k], axis=maxdim) for f in raw],
                    axis=maxdim), axis=maxdim, keepdims=False) / np.sqrt(modelcount-1)
            phi.append(p)

        return phi

    @property
    def phi_vector(self, fit_index=None):
        """Vector of phi across fit_indexes.

        :param int fit_index: Which model fit to use (defaults to `fit_index`).
        :return: Vector of phi values from all modules.
        """
        if fit_index is None:
            fit_index = self.fit_index
        m = self.copy(fit_index)
        packer, unpacker, bounds = simple_vector(m)
        return packer(self)

    #
    # plotting support
    #
    def get_plot_fn(self, mod_index=None, plot_fn_idx=None, fit_index=None):
        """Get the plotting function for the specified module.

        :param int mod_index: Which module in the modelspec to get the plotting function for.
        :param int plot_fn_idx: Which plotting function in the list to get.
        :param int fit_index: Update the fit index if not None.

        :return: A plotting function.
        """
        if mod_index is None:
            mod_index = self.mod_index

        if fit_index is not None:
            self.fit_index = fit_index

        module = self.get_module(mod_index)

        fallback_fn_path = 'nems0.plots.api.mod_output'

        try:
            fn_list = module['plot_fns']
        except KeyError:
            # if no 'plot_fns', then early out
            log.warning(f'No "plot_fns" found for module "{module["fn"]}", defaulting to "{fallback_fn_path}"')
            return _lookup_fn_at(fallback_fn_path)

        if not fn_list:
            # if 'plot_fns' present but empty, then early out
            log.warning(f'Empty "plot_fns" found for module "{module["fn"]}", defaulting to "{fallback_fn_path}"')
            return _lookup_fn_at(fallback_fn_path)

        if plot_fn_idx is None:
            plot_fn_idx = module.get('plot_fn_idx', 0)
        try:
            fn_path = fn_list[plot_fn_idx]
        except IndexError:
            log.warning(f'plot_fn_idx of "{plot_fn_idx}" is out of bounds for module idx {mod_index},'
                        'defaulting to first entry.')
            fn_path = fn_list[0]

        log.debug(f'Found plot fn "{fn_path}" for module "{module["fn"]}"')
        return _lookup_fn_at(fn_path)

    def plot(self, mod_index=0, plot_fn_idx=None, fit_index=None, rec=None,
             sig_name='pred', channels=None, ax=None, **kwargs):
        """Generate the plot for a single module.

        :param mod_index: Which module in the modelspec to generate the plot for.
        :param plot_fn_idx: Which function in the list of plot functions.
        :param fit_index: Update the fit index.
        :param rec: The recording from which to pull the data.
        :param sig_name: Which signal in the recording.
        :param channels: Which channel in the signal.
        :param ax: Axis on which to plot.
        :param kwargs: Optional keyword args.
        """
        if rec is None:
            rec = self.recording

        if channels is None:
            channels = self.plot_channel

        plot_fn = self.get_plot_fn(mod_index, plot_fn_idx, fit_index)
        
        # call the plot func
        return plot_fn(rec=rec, modelspec=self, sig_name=sig_name, idx=mod_index,
                       channels=channels, ax=ax, **kwargs)

    def quickplot(self, rec=None, epoch=None, occurrence=None, fit_index=None,
                  include_input=True, include_output=True, size_mult=(1.0, 2.0),
                  figsize=None, fig=None, time_range=None, sig_names=None,
                  modidx_set=None):

        """Generate a summary plot of a subset of the data.

        :param rec: The recording from which to pull the data.
        :param epoch: Name of epoch from which to extract data.
        :param int occurrence: Which occurrences of the data to plot.
        :param int fit_index: Update the fit index.
        :param bool include_input: Whether to include default plot of the inputs.
        :param bool include_output: Whether to include default plot of the outputs.
        :param tuple size_mult: Scale factors for width and height of figure.
        :param tuple figsize: Size of figure (tuple of inches).
        :param tuple time_range: If not None, plot signals from time_range[0]-time_range[1] sec
        :param sig_names: list of signal name strings (default ['stim'])
        :param modidx_set: list of mod indexes to plot (default all)
        :return: Matplotlib figure.
        """
        if rec is None:
            rec = self.recording

        if fit_index is not None:
            self.fit_index = fit_index

        input_name = self.meta.get('input_name', 'stim')
        output_name = self.meta.get('output_name', 'resp')

        if sig_names is None:
            if include_input:
                sig_names = [input_name]
            else:
                sig_names = []

        # strip out any signals that aren't in the recording
        sig_names = [s for s in sig_names if s in rec.signals.keys()]

        if modidx_set is None:
            modidx_set = range(len(self))

        # if there's no epoch, don't bother
        if rec['resp'].epochs is None:
            pass

        else:
            # list of possible epochs
            available_epochs = rec['resp'].epochs.name.unique()

            # if the epoch is correct, move on
            if (epoch is not None) and (epoch in available_epochs):
                pass

            # otherwise try the default fall backs
            else:
                # order of fallback epochs to search for
                epoch_sequence = [
                    'TRIAL',
                    'REFERENCE',
                    'TARGET',
                    'SIGNAL',
                    'SEQUENCE1',
                    None  # leave None as the last in the sequence, to know when not found
                ]

                for e in epoch_sequence:
                    if e is None:
                        # reached the end of the fallbacks, not found
                        log.warning(f'Quickplot: no epoch specified, and no fallback found. Will not subset data')

                    if e in available_epochs:
                        epoch = e
                        log.info(f'Quickplot: no epoch specified, falling back to "{epoch}"')
                        break

        # data to plot
        if epoch is not None:
            try:
                epoch_bounds = rec['resp'].get_epoch_bounds(epoch, mask=rec['mask'])
            except ValueError:
                # some signal types can't handle masks with epoch bounds
                epoch_bounds = rec['resp'].get_epoch_bounds(epoch)
            except:
                log.warning(f'Quickplot: no valid epochs matching {epoch}. Will not subset data.')
                epoch = None

        if 'mask' in rec.signals.keys():
            rec = rec.apply_mask()

        rec_resp = rec['resp']
        rec_pred = rec['pred']
        rec_stim = rec['stim']

        if epoch is None or len(epoch_bounds) == 0:
            epoch_bounds = np.array([[0, rec['resp'].shape[1] / rec['resp'].fs]])

        # figure out which occurrence
        # not_empty = [np.any(np.isfinite(x)) for x in extracted]  # occurrences containing non inf/nan data
        # possible_occurrences, = np.where(not_empty)
        possible_occurrences = np.arange(epoch_bounds.shape[1])
        # if there's no possible occurrences, then occurrence passed in doesn't matter
        if time_range is not None:
            pass
        elif len(possible_occurrences) == 0:
            # only warn if passed in occurrence
            if occurrence is not None:
                log.warning('Quickplot: no possible occurrences, ignoring passed occurrence')
            occurrence = None
            time_range = epoch_bounds[0]
        else:
            # otherwise, if the passed occurrence is not possible, then default to the first one
            if occurrence not in possible_occurrences:
                # only warn if had passed in occurrence
                if occurrence is not None:
                    log.warning(f'Quickplot: Passed occurrence not possible, defaulting to first possible '
                                f'(idx: {occurrence}).')
                occurrence = possible_occurrences[0]
            time_range = epoch_bounds[occurrence]

        # determine the plot functions
        plot_fn_modules = []
        skip_list = ['nems0.plots.api.null']
        for mod_idx, m in enumerate(self):
            # do some forward checking here for strf: skip gaussian weights if next is strf
            # clunky, better way?
            if mod_idx < len(self) and self[mod_idx]['fn'] == 'nems0.modules.weight_channels.gaussian' and \
                    self[mod_idx + 1]['fn'] == 'nems0.modules.fir.basic':
                continue
            # don't double up on spectrograms
            if m['fn'] == 'nems0.modules.nonlinearity.dlog':
                continue
            if m.get('plot_fns', None) is None:
                continue
            if m['plot_fns'] == []:
                continue
            fn = m['plot_fns'][m['plot_fn_idx']]
            if fn in skip_list:
                continue
            if mod_idx in modidx_set:
                plot_fn_modules.append((mod_idx, self.get_plot_fn(mod_idx)))
            # these plot functions always produce the same thing
            # so should be skipped if they appear more than once
            if fn in ['nems0.plots.api.pred_resp',
                      'nems0.plots.api.state_vars_timeseries',
                      'nems0.plots.api.spectrogram_output',
                      ]:
                skip_list.append(fn)
                
        # use partial so ax can be specified later
        # the format is (fn, col_span), where col_span is 1 for all of these, but will vary for the custom pre-post
        # below fn and col_span should be list, but for simplicity here they are just int and partial and converted
        # in the plotting loop below
        if rec_stim is None:
            opts = {}
        else:
            opts = {'chan_names': rec_stim.chans}
        plot_fns = [
            (partial(plot_fn,
                     rec=rec,
                     modelspec=self,
                     idx=mod_idx,
                     time_range=time_range,
                     **opts), 1)
            for mod_idx, plot_fn in plot_fn_modules
        ]

        for s in sig_names:
            if rec[s].shape[0] > 1:
                plot_fn = _lookup_fn_at('nems0.plots.api.spectrogram')
                title = s + ' spectrogram'
            else:
                plot_fn = _lookup_fn_at('nems0.plots.api.timeseries_from_signals')
                title = s + ' timeseries'

            fn = partial(plot_fn, rec=rec,
                         sig_name=s,
                         epoch=epoch,
                         occurrence=occurrence,
                         time_range=time_range,
                         title=title
                         )
            # add to front
            plot_fns = [(fn, 1)] + plot_fns

        if include_output and (output_name=='stim'):
            if rec[output_name].shape[0] > 1:
                plot_fn = _lookup_fn_at('nems0.plots.api.spectrogram')
                title = output_name + ' spectrogram'
            else:
                plot_fn = _lookup_fn_at('nems0.plots.api.timeseries_from_signals')
                title = output_name + ' timeseries'

            # actual output (stim)
            fn = partial(plot_fn, rec=rec,
                         sig_name=output_name,
                         epoch=epoch,
                         occurrence=occurrence,
                         time_range=time_range,
                         title=title
                         )
            # add to end
            plot_fns.append((fn, 1))

            # recon
            r_test = np.mean(self.meta.get('r_test', [0]))
            title = f'reconstruction r_test: {r_test:.3f}'
            fn = partial(plot_fn, rec=rec,
                         sig_name='pred',
                         epoch=epoch,
                         occurrence=occurrence,
                         time_range=time_range,
                         title=title
                         )
            # add to end
            plot_fns.append((fn, 1))
            
        elif include_output:
            if rec[output_name].shape[0] > 1:
                plot_fn = _lookup_fn_at('nems0.plots.api.spectrogram')
                title = output_name + ' spectrogram'
                fn = partial(plot_fn, rec=rec,
                             sig_name=output_name,
                             epoch=epoch,
                             occurrence=occurrence,
                             time_range=time_range,
                             title=title
                             )
            else:
                if (time_range is not None) or (epoch is None):
                    plot_fn = _lookup_fn_at('nems0.plots.api.timeseries_from_signals')
                else:
                    plot_fn = _lookup_fn_at('nems0.plots.api.timeseries_from_epoch')
                fn = partial(plot_fn,
                             signals=[rec_resp, rec_pred],
                             epoch=epoch,
                             time_range=time_range,
                             occurrences=occurrence,
                             title=f'Prediction vs Response, {epoch} #{occurrence}'
                             )
            # add to end
            plot_fns.append((fn, 1))

            # scatter text
            n_cells = len(self.meta.get('r_test', []))
            r_test = np.mean(self.meta.get('r_test', [0]))
            r_fit = np.mean(self.meta.get('r_fit', [0]))
            scatter_text = f'r_test: {r_test:.3f} (n={n_cells})\nr_fit: {r_fit:.3f}'

            scatter_fn = _lookup_fn_at('nems0.plots.api.plot_scatter')
            n_bins = 100
            fn_smooth = partial(scatter_fn,
                                sig1=rec_pred,
                                sig2=rec_resp,
                                smoothing_bins=n_bins,
                                force_square=False,
                                text=scatter_text,
                                title=f'Smoothed, bins={n_bins}'
                                )

            if rec_resp.shape[0] > 1:
                not_smooth_fn = _lookup_fn_at('nems0.plots.api.perf_per_cell')
                fn_not_smooth = partial(not_smooth_fn, self)
            else:
                fn_not_smooth = partial(scatter_fn,
                                        sig1=rec_pred,
                                        sig2=rec_resp,
                                        smoothing_bins=False,
                                        force_square=False,
                                        text=scatter_text,
                                        title='Unsmoothed'
                                        )

            plot_fns.append(([fn_smooth, fn_not_smooth], [1, 1]))

        # done with plot functions, get figure title
        cellid = self.meta.get('cellid', 'UNKNOWN')
        modelname = self.meta.get('modelname', 'UNKNOWN')
        batch = self.meta.get('batch', 0)

        fig_title = f'Cell: {cellid}, Batch: {batch}, {epoch} #{occurrence}\n{modelname}'

        n_rows = len(plot_fns)

        # make the figure and the grids for the plots
        if figsize is None:
            figsize = (10 * size_mult[0], n_rows * size_mult[1])
        if fig is None:
            fig = plt.figure(figsize=figsize, constrained_layout=True)
        else:
            fig.set_size_inches(figsize[0], figsize[1], forward=True)

        # each module gets a row in the gridspec, giving plots control over subplots, etc.
        gs_rows = gridspec.GridSpec(n_rows, 1, figure=fig)
        max_cols = max([sum(pfn[1]) if type(pfn[1]) is list else 1 for pfn in plot_fns])

        # iterate through the plotting partials and plot them to the gridspec
        for row_idx, (plot_fn, col_spans) in enumerate(plot_fns):
            # plot_fn, col_spans should be list, so convert if necessary
            log.info('plotting row {}/{}'.format(row_idx + 1, len(plot_fns)))
            if isinstance(col_spans, int) and not isinstance(plot_fn, list):
                col_spans = [col_spans]
                plot_fn = [plot_fn]

            n_cols = sum(col_spans)
            gs_cols = gridspec.GridSpecFromSubplotSpec(1, n_cols, gs_rows[row_idx])

            col_idx = 0
            for fn, col_span in zip(plot_fn, col_spans):
                try:
                    # if n_cols == max_cols:
                    #     plot_ind = row_idx * max_cols + col_idx + 1
                    # else:
                    start_ind = row_idx * max_cols + col_idx + 1
                    plot_ind = (start_ind,int(start_ind + max_cols/n_cols + col_span - 2))
                    ax = fig.add_subplot(n_rows, max_cols, plot_ind)

                    #ax = plt.Subplot(fig, gs_cols[0, col_span-1])
                    #ax = fig.add_subplot(gs_cols[0, col_idx:(col_idx+col_span)])
                    #ax = plt.Subplot(fig, gs_cols[0, col_idx:col_idx + col_span])

                    fn(ax=ax)
                    col_idx += col_span
                except:
                    log.warning(f'Quickplot: failed plotting function: for row {row_idx} skipping.')

        # suptitle needs to be after the gridspecs in order to work with constrained_layout
        fig.suptitle(fig_title)
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1, wspace=0)

        log.info('Quickplot: generated fig with title "{}"'.format(fig_title.replace("\n", " ")))
        return fig

    def append(self, module):
        """Append a module to the modelspec.

        :param module: A module dict.
        """
        self.raw[self.cell_index, self.fit_index, self.jack_index].append(module)

    def pop_module(self):
        """Remove the last module from the modelspec."""
        del self.raw[self.cell_index, self.fit_index, self.jack_index][-1]

    def get_priors(self, data):
        """TODO docs.

        :param data: TODO docs
        :return: TODO docs
        """
        # Here, we query each module for it's priors. A prior will be a
        # distribution, so we set phi to the mean (i.e., expected value) for
        # each parameter, then ask the module to evaluate the input data. By
        # doing so, we give each module an opportunity to perform sensible data
        # transforms that allow the following module to initialize its priors as
        # it sees fit.
        result = data.copy()
        priors = []
        for module in self.modules:
            module_priors = module.get_priors(result)
            priors.append(module_priors)

            phi = {k: p.mean() for k, p in module_priors.items()}
            module_output = module.evaluate(result, phi)
            result.update(module_output)

        return priors

    def evaluate(self, rec=None, **kwargs):
        """Evaluate the Model on a recording. essentially a wrapper for `modelspec.evaluate`.

        :param rec: Recording object (default is self.recording preset to val usually)
        :param modelspec: Modelspec object.
        :param start: Start evaluation at module start, assuming `rec['pred']` is in the appropriate
            state to feed into modelspec[start].
        :param stop: Stop at this module.
        :return: `Recording` copy of input with `pred` updated with prediction.
        """
        if rec is None:
            rec = self.recording

        if np.any(['tf_only' in fn for fn in self.fn()]):
            rec = evaluate_tf(rec, self, **kwargs)
        else:
            rec = evaluate(rec, self, **kwargs)

        return rec

    def fast_eval_on(self, rec=None, subset=None):
        """Quickly evaluates a model on a recording.

        Enter fast eval mode, where model is evaluated up through the
        first module that has a fittable phi. Evaluate model on rec up through
        the preceding module and save in `freeze_rec`.

        :param rec: Recording object to evaluate.
        :param subset: Which subset of the data to evaluate.
        """
        if rec is None:
            raise ValueError("Must provide valid rec=<recording> object")
        if subset is not None:
            start_mod = subset[0]
        else:
            start_mod = len(self)-1
            for i in range(len(self)-1, -1, -1):
                if ('phi' in self[i]) and self[i]['phi']:
                    start_mod = i
        # import pdb; pdb.set_trace()
        # eval from 0 to start position and save the result in freeze_rec
        self.fast_eval_start = 0
        self.freeze_rec = evaluate(rec, self, start=0, stop=start_mod)

        # then switch to fast_eval mode
        self.fast_eval = True
        self.fast_eval_start = start_mod
        log.info('Freezing fast rec at start=%d', self.fast_eval_start)

    def fast_eval_off(self):
        """Turn off `fast_eval` and purge `freeze_rec` to free up memory."""
        self.fast_eval = False
        self.freeze_rec = None
        self.fast_eval_start = 0

    def generate_tensor(self, data, phi):
        """Evaluate the module given the input data and phi.

        :param dict data: Dictionary of arrays and/or tensors.
        :param list(dict) phi: list of dictionaries. Each entry in the list maps to the corresponding
            module in the model. If a module does not require any input parameters, use a blank
            dictionary. All elements in phi must be scalars, arrays or tensors.
        :return: dictionary of Signals
        """
        # Loop through each module in the stack and transform the data.
        result = data.copy()
        for module, module_phi in zip(self.modules, phi):
            module_output = module.generate_tensor(result, module_phi)
            result.update(module_output)
        return result

    def get_shortname(self):
        """Get a string that is just the module IDs in this modelspec.

        :return str: Shortname, the module IDs.
        """
        keyword_string = '_'.join([m['id'] for m in self])
        return keyword_string

    def get_longname(self):
        """Return a long name for this modelspec suitable for use in saving to disk without a path.

        :return str: Longname, more details about the modelspec.
        """
        meta = self.meta

        recording_name = meta.get('exptid')
        if recording_name is None:
            recording_name = meta.get('recording', 'unknown_recording')
        if 'modelspecname' in self.meta:
            keyword_string = self.meta['modelspecname']
        else:
            keyword_string = get_modelspec_shortname(self)
        fitter_name = meta.get('fitkey', meta.get('fitter', 'unknown_fitter'))
        date = nems0.utils.iso8601_datestring()
        guess = '.'.join([recording_name, keyword_string, fitter_name, date])

        # remove problematic characters
        guess = re.sub('[:]', '', guess)
        guess = re.sub('[,]', '', guess)

        if len(guess) > 100:
            # If modelname is too long, causes filesystem errors.
            guess = guess[:80] + '...' + str(hash(guess))

        return guess

    def modelspec2tf(self, tps_per_stim=550, feat_dims=1, data_dims=1, state_dims=0,
                     fs=100, net_seed=1, weight_scale=0.1, use_modelspec_init=True,
                     distr='norm',):
        """Converts a modelspec object to Tensorflow layers.

        Maps modelspec modules to Tensorflow layers. Adapted from code by Sam Norman-Haignere.
        https://github.com/snormanhaignere/cnn/blob/master/cnn.py

        :param tps_per_stim:
        :param int feat_dims:
        :param int data_dims:
        :param int state_dims:
        :param fs:
        :param net_seed:
        :param weight_scale:
        :param bool use_modelspec_init:
        """
        import tensorflow as tf
        # placeholders not compatible with eager execution, which is the default in tf 2
        tf.compat.v1.disable_eager_execution()
        # set GPU memory to grow as needed, instead of requesting all available during init
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        # placeholders
        shape = [None, tps_per_stim, feat_dims]
        F = tf.compat.v1.placeholder('float32', shape=shape)
        D = tf.compat.v1.placeholder('float32', shape=shape)
        if state_dims > 0:
            s_shape = [None, tps_per_stim, state_dims]
            S = tf.compat.v1.placeholder('float32', shape=s_shape)

        layers = []
        for idx, m in enumerate(self):
            fn = m['fn']
            log.info(f'Modelspec2tf: {fn}')

            layer = {}
            # input to each layer is output of previous layer
            if idx == 0:
                layer['X'] = F
                # layer['D'] = D
                if state_dims > 0:
                    layer['S'] = S

            else:
                layer['X'] = layers[-1]['Y']
                if 'L' in layers[-1]:
                    layers['L'] = layers[-1]['L']

            n_input_feats = np.int32(layer['X'].shape[2])
            # default integration time is one bin
            layer['time_win_smp'] = 1  # default

            layer = nems0.tf.cnnlink.map_layer(layer=layer, prev_layers=layers, idx=idx, modelspec=m,
                                              n_input_feats=n_input_feats, net_seed=net_seed, weight_scale=weight_scale,
                                              use_modelspec_init=use_modelspec_init, fs=fs, distr=distr)

            # necessary?
            layer['time_win_sec'] = layer['time_win_smp'] / fs

            layers.append(layer)

        return layers

    def modelspec2tf2(self, seed=0, use_modelspec_init=True, fs=100, initializer='random_normal',
                      freeze_layers=None, kernel_regularizer=None):
        """New version

        TODO
        """
        layers = []
        if freeze_layers is None:
            freeze_layers = []

        for i, m in enumerate(self):
            try:
                tf_layer = nems0.utils.lookup_fn_at(m['tf_layer'])
            except KeyError:
                raise NotImplementedError(f'Layer "{m["fn"]}" does not have a tf equivalent.')

            if i in freeze_layers:
                trainable=False
            else:
                trainable=True
            layer = tf_layer.from_ms_layer(m, use_modelspec_init=use_modelspec_init, seed=seed, fs=fs,
                                           initializer=initializer, trainable=trainable,
                                           kernel_regularizer=kernel_regularizer)
            layers.append(layer)

        return layers

    def get_dstrf(self,
                  rec: nems0.recording.Recording,
                  index: int,
                  width: int = 30,
                  rebuild_model: bool = False,
                  out_channel: int = 0,
                  method: str = 'jacobian'
                  ) -> np.array:
        """Creates a tf model from the modelspec and generates the dstrf.

        :param rec: The input recording, of shape [channels, time].
        :param index: The index at which the dstrf is calculated. Must be within the data.
        :param width: The width of the returned dstrf (i.e. time lag from the index). If 0, returns the whole dstrf.
        :rebuild_model: Rebuild the model to avoid using the cached one.
        Zero padded if out of bounds.

        :return: np array of size [channels, width]
        """
        if 'stim' not in rec.signals:
            raise ValueError('No "stim" signal found in recording.')
        # predict response for preceeding D bins, enough time, presumably, for slow nonlinearities to kick in
        D = 50
        data = rec['stim']._data[:,np.max([0,index-D]):(index+1)].T
        chan_count=data.shape[1]
        if 'state' in rec.signals.keys():
            include_state = True
            state_data = rec['state']._data[:, np.max([0, index - D]):(index + 1)].T
        else:
            include_state = False

        if index < D:
            data = np.pad(data, ((D-index, 0), (0, 0)))
            if include_state:
                state_data = np.pad(state_data, ((D - index, 0), (0, 0)))

        # a few safety checks
        if data.ndim != 2:
            raise ValueError('Data must be a recording of shape [channels, time].')
        #if not 0 <= index < width + data.shape[-2]:

        if D > data.shape[-2]:
            raise ValueError(f'Index must be within the bounds of the time channel plus width.')

        need_fourth_dim = np.any(['Conv2D_NEMS' in m['fn'] for m in self])

        #print(f'index: {index} shape: {data.shape}')
        # need to import some tf stuff here so we don't clutter and unnecessarily import tf 
        # (which is slow) when it's not needed
        # TODO: is this best practice? Better way to do this?
        import tensorflow as tf
        from nems0.tf.cnnlink_new import get_jacobian

        if self.tf_model is None or rebuild_model:
            from nems0.tf import modelbuilder
            from nems0.tf.layers import Conv2D_NEMS

            # generate the model
            model_layers = self.modelspec2tf2(use_modelspec_init=True)
            state_shape = None
            if need_fourth_dim:
                # need a "channel" dimension for Conv2D (like rgb channels, not frequency). Only 1 channel for our data.
                data_shape = data[np.newaxis, ..., np.newaxis].shape
                if include_state:
                    state_shape = state_data[np.newaxis, ..., np.newaxis].shape
            else:
                data_shape = data[np.newaxis].shape
                if include_state:
                    state_shape = state_data[np.newaxis].shape
            self.tf_model = modelbuilder.ModelBuilder(
                name='Test-model',
                layers=model_layers,
            ).build_model(input_shape=data_shape, state_shape=state_shape)

        if type(out_channel) is list:
            out_channels = out_channel
        else:
            out_channels = [out_channel]

        if method == 'jacobian':
            # need to convert the data to a tensor
            stensor = None
            if need_fourth_dim:
                tensor = tf.convert_to_tensor(data[np.newaxis, ..., np.newaxis], dtype='float32')
                if include_state:
                    stensor = tf.convert_to_tensor(state_data[np.newaxis, ..., np.newaxis], dtype='float32')
            else:
                tensor = tf.convert_to_tensor(data[np.newaxis], dtype='float32')
                if include_state:
                    stensor = tf.convert_to_tensor(state_data[np.newaxis], dtype='float32')

            if include_state:
                tensor = [tensor, stensor]

            for outidx in out_channels:
                if include_state:
                    w = get_jacobian(self.tf_model, tensor, D, tf.cast(outidx, tf.int32))[0].numpy()[0]
                else:
                    w = get_jacobian(self.tf_model, tensor, D, tf.cast(outidx, tf.int32)).numpy()[0]

                if need_fourth_dim:
                    w = w[:, :, 0]
    
                if width == 0:
                    _w = w.T
                else:
                    # pad only the time axis if necessary
                    padded = np.pad(w, ((width-1, width), (0, 0)))
                    _w = padded[D:D + width, :].T
                if len(out_channels)==1:
                    dstrf = _w
                elif outidx == out_channels[0]:
                    dstrf = _w[..., np.newaxis]
                else:
                    dstrf = np.concatenate((dstrf, _w[..., np.newaxis]), axis=2)
        else:
            dstrf = np.zeros((chan_count, width, len(out_channels)))

            if need_fourth_dim:
                tensor = tf.convert_to_tensor(data[np.newaxis, ..., np.newaxis])
            else:
                tensor = tf.convert_to_tensor(data[np.newaxis])
            p0 = self.tf_model(tensor).numpy()
            eps = 0.0001
            for lag in range(width):
                for c in range(chan_count):
                    d = data.copy()
                    d[-lag, c] += eps
                    if need_fourth_dim:
                        tensor = tf.convert_to_tensor(d[np.newaxis, ..., np.newaxis])
                    else:
                        tensor = tf.convert_to_tensor(d[np.newaxis])
                    p = self.tf_model(tensor).numpy()
                    #print(p.shape)
                    dstrf[c, -lag, :] = p[0, D, out_channels] - p0[0, D, out_channels]
            if len(out_channels) == 1:
                dstrf = dstrf[:, :, 0]

        return dstrf

def get_modelspec_metadata(modelspec):
    """Return a dict of the metadata for this modelspec.

    Purely by convention, metadata info for the entire modelspec is stored in the first module.

    :param modelspec: Modelspec object from which to get metadata.
    :return dict: Modelspec meta dict.
    """
    return modelspec.meta


def set_modelspec_metadata(modelspec, key, value):
    """Set a key/value pair in the modelspec's metadata.

    Purely by convention, metadata info for the entire modelspec is stored in the first module.

    :param modelspec: Modelspec object from which to get metadata.
    :param key: Update key.
    :param value: Update value.
    :param: The modelspec with updated meta.
    """
    if not modelspec.meta:
        modelspec[0]['meta'] = {}
    modelspec[0]['meta'][key] = value
    
    return modelspec


def get_modelspec_shortname(modelspec):
    """Return a string that is just the module ids in this modelspec.

    :param modelspec: Modelspec object from which to get metadata.
    :return str: The modelspec shortname.
    """
    return modelspec.get_shortname()


def get_modelspec_longname(modelspec):
    """Return a LONG name for this modelspec suitable for use in saving to disk without a path.

    :param modelspec: Modelspec object from which to get metadata.
    :return str: The modelspec longname.
    """
    return modelspec.get_longname()


def _modelspec_filename(basepath, number):
    """Append a number to the end of a filepath.

    :param basepath: Path to add number to.
    :param number: Number to add.
    :return: String of basepath with suffix added.
    """
    suffix = '.{:04d}.json'.format(number)
    return basepath + suffix


def save_modelspec(modelspec, filepath):
    """Save a modelspec to filepath. Overwrites any existing file.

    :param modelspec: Modelspec object from which to get metadata.
    :param filepath: Save location.
    """
    if type(modelspec) is list:
        nems0.uri.save_resource(filepath, json=modelspec)
    else:
        nems0.uri.save_resource(filepath, json=modelspec.raw)


def save_modelspecs(directory, modelspecs, basename=None):
    """Save one or more modelspecs to disk with stereotyped filenames.

    Ex:
        directory/basename.0000.json
        directory/basename.0001.json
        directory/basename.0002.json
        ...etc...

    Basename will be automatically generated if not provided.

    :param directory: Save location.
    :param list modelspecs: List of modelspecs to save.
    :param basename: Save name of modelspecs, otherwise will use modelspec long name.
    :return: The filepath of the last saved modelspec.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
        os.chmod(directory, 0o777)

    for idx, modelspec in enumerate(modelspecs):
        if not basename:
            bname = get_modelspec_longname(modelspec)
        else:
            bname = basename
        basepath = os.path.join(directory, bname)
        filepath = _modelspec_filename(basepath, idx)
        if type(modelspec) is list:
            # TODO fix save with array
            save_modelspec(modelspec, filepath)
        else:
            # HACK for backwards compatibility. if saving a modelspecs list
            # then only need a single fit_index from the ModelSpec class
            save_modelspec(modelspec.raw[0, 0, 0], filepath)
    return filepath


def load_modelspec(uri):
    """Return a single modelspecs loaded from uri.

    :param uri: URI of modelspec.
    :return: A new modelspec object loaded form the uri.
    """
    ms = nems0.uri.load_resource(uri)
    return ModelSpec(ms)


def load_modelspecs(directory, basename, regex=None):
    """Return a list of modelspecs loaded from `directory/basename.*.json`.

    :param directory: Directory to search for modelspecs.
    :param basename: Name of modelspecs to match against.
    :param regex: Optional regex matching for modelspec names.
    :return: A new modelspec object.
    """
    # regex = '^' + basename + '\.{\d+}\.json'
    # TODO: fnmatch is not matching pattern correctly, replacing
    #       with basic string matching for now.  -jacob 2/17/2018
    # files = fnmatch.filter(os.listdir(directory), regex)
    #       Also fnmatch was returning list of strings? But
    #       json.load expecting file object
    # modelspecs = [json.load(f) for f in files]
    dir_list = os.listdir(directory)
    if regex:
        # TODO: Not sure why this isn't working? No errors but
        #       still isn't matching the things it should be matching.
        #       ( tested w/ regex='^TAR010c-18-1\.{\d+}\.json')
        #       -jacob 2/25/18
        if isinstance(regex, str):
            regex = re.compile(regex)
        files = [os.path.join(directory, s) for s in dir_list
                 if re.search(regex, s)]
    else:
        files = [os.path.join(directory, s) for s in dir_list
                 if (basename in s and '.json' in s)]
    modelspecs = []
    for file in files:
        with open(file, 'r') as f:
            try:
                m = json.load(f)
                m[0]['meta']['filename'] = file
            except json.JSONDecodeError as e:
                print("Couldn't load modelspec: {0}"
                      "Error: {1}".format(file, e))
            modelspecs.append(m)
    return ModelSpec(modelspecs)
    # return modelspecs


lookup_table = {}  # TODO: Replace with real memoization/joblib later


def _lookup_fn_at(fn_path, ignore_table=False):
    """Private function that returns a function handle found at a given module.

    Basically, a way to import a single function.
    e.g.
        myfn = _lookup_fn_at('nems0.modules.fir.fir_filter')
        myfn(data)
        ...

    :param fn_path: Path to the function.
    :param ignore_table: Whether or not to look up the function in the cache.
    :return: Function handle.
    """
    # default is nems0.xforms.<fn_path>
    if '.' not in fn_path:
        fn_path = 'nems0.xforms.' + fn_path

    if (not ignore_table) and (fn_path in lookup_table):
        fn = lookup_table[fn_path]
    else:
        api, fn_name = nems0.utils.split_to_api_and_fn(fn_path)
        api = api.replace('nems_db.xform', 'nems_lbhb.xform')

        # backwards compatibility for models fit pre-nems lite
        api = api.replace('nems.', 'nems0.')
        try:
            api_obj = importlib.import_module(api)
        
            if ignore_table:
                importlib.reload(api_obj)  # force overwrite old imports
            fn = getattr(api_obj, fn_name)
            if not ignore_table:
                lookup_table[fn_path] = fn
        except:
            log.info(f'failed to import module: {api}')
            fn = None
    return fn


def fit_mode_on(modelspec, rec=None, subset=None):
    """Turn on `norm.recalc` for each module when present.

    TODO docs can this be removed?

    :param modelspec:
    param rec:
    param subset:
    """
    """
    # norm functions deprecated. too messy
    for m in modelspec:
        if 'norm' in m.keys():
            m['norm']['recalc'] = 1
    """
    if rec is None:
        raise ValueError('rec must be specified')
    modelspec.fast_eval_on(rec, subset)


def fit_mode_off(modelspec):
    """Turn off norm.recalc for each module when present.

    TODO docs can this be removed?

    :param modelspec:
    """
    """
    # norm functions deprecated. too messy
    for m in modelspec:
        if 'norm' in m.keys():
            m['norm']['recalc'] = 0
    """
    modelspec.fast_eval_off()


def eval_ms_layer(data: np.ndarray,
                  layer_spec: typing.Union[None, str] = None,
                  state_data: np.ndarray = None,
                  stop: typing.Union[None, int] = None,
                  modelspec: ModelSpec = None
                  ) -> np.ndarray:
    """Takes in a numpy array and applies a single ms layer to it.

    :param data: The input data. Shape of (reps, time, channels).
    :param layer_spec: A layer spec for layers of a modelspec.
    :param state_data: State gain data, optional. Same shape as data.
    :param stop: What layer to eval to. Non inclusive. If not passed, will evaluate the whole layer spec.
    :param modelspec: Optionally use an existing modelspec. Takes precedence over layer_spec.

    :return: The processed data.
    """
    if layer_spec is None and modelspec is None:
        raise ValueError('Either of "layer_spec" or "modelspec" must be specified.')

    if modelspec is not None:
        ms = modelspec
    else:
        ms = nems0.initializers.from_keywords(layer_spec)

    sig = nems0.signal.RasterizedSignal.from_3darray(
        fs=100,
        array=np.swapaxes(data, 1, 2),
        name='temp',
        recording='temp',
        epoch_name='REFERENCE'
    )
    signal_dict = {'stim': sig}

    if state_data is not None:
        state_sig = nems0.signal.RasterizedSignal.from_3darray(
            fs=100,
            array=np.swapaxes(state_data, 1, 2),
            name='state',
            recording='temp',
            epoch_name='REFERENCE'
        )
        signal_dict['state'] = state_sig

    rec = nems0.recording.Recording(signal_dict)
    rec = ms.evaluate(rec=rec, stop=stop)

    pred = np.swapaxes(rec['pred'].extract_epoch('REFERENCE'), 1, 2)
    return pred


def evaluate(rec, modelspec, start=None, stop=None):
    """Given a recording object and a modelspec, return a prediction in a new recording.

    Does not alter modelspec's arguments in any way. Only evaluates modules at indices start through stop-1.
    A value of None for start will include the beginning of the list, and a value of None for stop will include
    the end of the list (whereas a value of -1 for stop will not). Evaluates using cell/fit/jack currently
    selected for modelspec.

    :param rec: Recording object.
    :param modelspec: Modelspec object.
    :param start: Start evaluation at module start, assuming `rec['pred']` is in the appropriate
        state to feed into modelspec[start].
    :param stop: Stop at this module.
    :return: `Recording` copy of input with `pred` updated with prediction.
    """
    if modelspec.fast_eval:
        # still kind of testing this out, though it seems to work
        start = modelspec.fast_eval_start
        d = modelspec.freeze_rec.copy()
        # import pdb
        # pdb.set_trace()
    else:
        # don't need a deep copy, fact that signals are immutable means that there will be an error
        # if evaluation tries to modify a signal in place
        d = rec.copy()

    for m in modelspec[start:stop]:
        if type(m) is dict:
            fn = _lookup_fn_at(m['fn'])
            if fn is None:
                tfn = m['fn'].replace("nems.", "nems0.")
                fn = _lookup_fn_at(tfn)
        else:
            fn = m.eval
        fn_kwargs = m.get('fn_kwargs', {})
        phi = m.get('phi', {})
        kwargs = {**fn_kwargs, **phi}  # Merges both dicts
        new_signals = fn(rec=d, **kwargs)

        # if type(new_signals) is not list:
        #     raise ValueError('Fn did not return list of signals: {}'.format(m))

        """
        # testing normalization
        if 'norm' in m.keys():
            s = new_signals[0]
            k = s.name
            if m['norm']['recalc']:
                if m['norm']['type'] == 'minmax':
                    m['norm']['d'] = np.nanmin(s.as_continuous(), axis=1,
                                               keepdims=True)
                    m['norm']['g'] = np.nanmax(s.as_continuous(), axis=1,
                                               keepdims=True) - \
                        m['norm']['d']
                    m['norm']['g'][m['norm']['g'] <= 0] = 1
                elif m['norm']['type'] == 'none':
                    m['norm']['d'] = np.array([0])
                    m['norm']['g'] = np.array([1])
                else:
                    raise ValueError('norm format not supported')

            fn = lambda x: (x - m['norm']['d']) / m['norm']['g']
            new_signals = [s.transform(fn, k)]
        """

        for s in new_signals:
            d.add_signal(s)
        d.signal_views[d.view_idx] = d.signals

    return d


def evaluate_tf(rec, modelspec, epoch_name='REFERENCE', **kwargs):

    input_name = modelspec[0]['fn_kwargs']['i']
    output_name = modelspec[-1]['fn_kwargs']['o']
    # convert ms to TF model
    if 'mask' in rec.signals:
        mask = rec['mask']
    else:
        mask = nems0.signal.RasterizedSignal(np.ones((1, rec['resp'].shape[-1])))  # select all

    stim_train = np.transpose(rec[input_name].extract_epoch(epoch=epoch_name, mask=mask), [0, 2, 1])
    if np.any(['Conv2D' in m['fn'] for m in modelspec.modules]):
        stim_train = stim_train[..., np.newaxis]
    model_layers = modelspec.modelspec2tf2(fs=rec['resp'].fs)
    pred_stacked = stim_train
    for layer in model_layers:  # looping over layers avoids needing to re-compile entire model
        pred_stacked = layer(pred_stacked)  # stims x time x cells

    output_count = rec['resp'].shape[0]
    pred_size = np.prod(np.array(pred_stacked.shape))
    if rec['resp'].shape[0] * mask.as_continuous().sum() == pred_size:
        output_count = rec['resp'].shape[0]
    else:
        output_count = int(pred_size/mask.as_continuous().sum())
    pred = pred_stacked.numpy().T.swapaxes(1, 2).reshape(output_count, -1)  # output cells x time

    # need to put back into shape of original unmasked signal, otherwise some nems functions break
    padded_pred = np.full((pred.shape[0], mask.shape[-1]), np.nan, dtype=np.float32)
    padded_pred[:, mask.as_continuous().flatten() == 1] = pred
    pred_sig = rec['resp']._modified_copy(padded_pred)
    pred_sig.name = output_name
    rec.add_signal(pred_sig)

    return rec


def summary_stats(modelspecs, mod_key='fn', meta_include=[], stats_keys=[]):
    """Generate summary statistics for a list of modelspecs.

    Each modelspec must be of the same length and contain the same
    modules (though they need not be in the same order).

    For example, ten modelspecs composed of the same modules that
    were fit to ten different datasets can be compared. However, ten
    modelspecs all with different modules fit to the same data cannot
    be compared because there is no guarantee that they contain
    comparable parameter values.

    :param list modelspecs: List of modelspecs
    :param mod_key: TODO docs
    :param stats_keys: TODO docs remove?

    :return: Nested dictionary of stats.
        {'module.function---parameter':
            {'mean':M, 'std':S, 'values':[v1,v2 ...]}}
        Where M, S and v might be scalars or arrays depending on the
        typical type for the parameter.
    """
    # Make sure the modelspecs themselves aren't modified
    # deepcopy for nested container structure
    modelspecs = [copy.deepcopy(m) for m in modelspecs]

    # Modelspecs must have the same length to compare
    length = None
    for m in modelspecs:
        if length:
            if len(m) != length:
                raise ValueError("All modelspecs must have the same length")
        length = len(m)

    # Modelspecs must have the same modules to compare
    fns = [m['fn'] for m in modelspecs[0]]
    for mspec in modelspecs[1:]:
        m_fns = [m['fn'] for m in mspec]
        if not sorted(fns) == sorted(m_fns):
            raise ValueError("All modelspecs must have the same modules")

    # Create a dictionary with a key for each parameter associated with
    # to a list of one value per modelspec
    columns = {}
    for mspec in modelspecs:
        for i, m in enumerate(mspec):
            name = '%d--%s' % (i, m[mod_key]) if mod_key else str(i)
            # Abbreviate by default if using 'fn'
            if name.startswith('nems0.modules.'):
                name = name[13:]

            # Add information from first-module meta
            if i == 0:
                meta = m['meta']
                meta_keys = [k for k in meta.keys() if k in meta_include]
                for k in meta_keys:
                    column_entry = 'meta--%s' % (k)
                    if column_entry in columns.keys():
                        columns[column_entry].append(meta[k])
                    else:
                        columns.update({column_entry: [meta[k]]})

            # Add in phi values
            phi = m['phi']
            params = phi.keys()
            for p in params:
                column_entry = '%s--%s' % (name, p)
                if column_entry in columns.keys():
                    columns[column_entry].append(phi[p])
                else:
                    columns.update({column_entry: [phi[p]]})

    # Convert entries from lists of values to dictionaries
    # containing keys for mean, std and the raw values.
    with_stats = {}

    for col, values in columns.items():
        try:
            mean = try_scalar((np.mean(values, axis=0)))
            std = try_scalar((np.std(values, axis=0)))
            sem = try_scalar((st.sem(values, axis=0)))
            max = try_scalar((np.max(values, axis=0)))
            min = try_scalar((np.min(values, axis=0)))
        except:  # TODO specify error type
            mean = np.nan
            std = np.nan
            sem = np.nan
            max = np.nan
            min = np.nan
        values = try_scalar((np.array(values)))

        with_stats[col] = {
                'mean': mean, 'std': std, 'sem': sem, 'max': max, 'min': min,
                'values': values
                }

    return with_stats


def get_best_modelspec(modelspecs, metakey='r_test', comparison='greatest'):
    """Get the best modelspec ranked by the given metakey.

    Examine the first-module meta information within each modelspec in a list,
    and return a singleton list containing the modelspec with the greatest
    value for the specified metakey by default (or the least value optionally).

    :param list modelspecs: Modelspecs to compare.
    :param metakey: Key to compare across modelspecs.
    :param str comparison: `greatest` or `least`.

    :return list: Modelspec with greatest/least metakey.
    """
    idx = None
    best = None
    for i, m in enumerate(modelspecs):
        if metakey in m[0]['meta']:
            metaval = m[0]['meta'][metakey]
            if comparison == 'greatest':
                if best is None:
                    best = metaval
                    idx = i
                else:
                    if metaval > best:
                        best = metaval
                        idx = i

            elif comparison == 'least':
                if best is None:
                    best = metaval
                    idx = i
                else:
                    if metaval < best:
                        best = metaval
                        idx = i

            else:
                raise NotImplementedError("Only supports 'greatest' or 'least'"
                                          "as arguments for comparison")

    return [modelspecs[idx]]


def sort_modelspecs(modelspecs, metakey='r_test', order='descending'):
    """Sort Modelspecs by given metakey.

    Sorts modelspecs in order of the given metakey, which should be in
    the first-module meta entry of each modelspec.

    :param list modelspecs: List of modelspecs to sort.
    :param metakey: Key to compare across modelspecs.
    :param order: `descending` or `ascending`.

    :return list: Sorted list of modelspecs.
    """
    sort = sorted(modelspecs, key=lambda m: m[0]['meta'][metakey])
    if order.lower() in ['ascending', 'asc', 'a']:
        return sort
    elif order.lower() in ['descending', 'desc', 'd']:
        return list(reversed(sort))
    else:
        raise ValueError("Not a recognized sorting order: %s" % order)


def try_scalar(x):
    """Try to convert x to scalar, in case of ValueError just return x.

    :param x: Value to convert to scalar.
    """
    # TODO: Maybe move this to an appropriate utilities module?
    try:
        x = np.asscalar(x)
    except ValueError:
        pass
    return x

# TODO: Check that the word 'phi' is not used in fn_kwargs
# TODO: Error checking the modelspec before execution;
# TODO: Validation of modules json schema; all require args should be present
