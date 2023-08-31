"""Implements DataSet container for use by Model.evaluate.

TODO: May be a better place to put this?

"""

import numpy as np

from nems.tools.arrays import broadcast_dicts, concatenate_dicts, apply_to_dict


class DataSet:
    # Use these names if `input_name`, `state_name`, etc. are not specified
    # in DataSet.__init__.
    default_input = 'input'
    default_output = 'output'
    default_target = 'target'
    default_state = 'state'

    def __init__(self, input, state=None, target=None, input_name=None,
                 state_name=None, output_name=None, target_name=None,
                 prediction_name=None, dtype=None, has_samples=False,
                 debug_memory=False, **kwargs):
        """Container for tracking dictionaries of data arrays.
        
        See `Model.evaluate` and `Model.fit` for detailed documentation of
        parameters.

        Parameters
        ----------
        input : ndarray or dict.
        state : ndarray; optional.
        target : ndarray or dict; optional.
        input_name : str; optional.
        state_name : str; optional.
        output_name : str; optional.
        target_name : str; optional.
        prediction_name : str; optional.
        dtype : type; optional.
            TODO: WIP. Want to specify a consistent datatype to cast all
                  arrays to.
        has_samples : bool; default=False.
            Indicates if first dimension of inputs (and other optional data)
            represents samples, saved as `DataSet.has_samples`. Methods that
            add or assume a sample dimension will set this attribute to True,
            and methods that remove a sample dimension will set this to False.
            In most cases, this kwarg should not need to be specified at time
            of construction.
        debug_memory : bool; default=False.
            TODO: Not actually implemented yet.
            If True, check `np.shares_memory(array, copied_array)` in several
            places to ensure that memory is not duplicated unintentionally.
        kwargs : dict; optional.
            Extra kwargs are silently ignored for convenience, so that other
            code can use `DataSet(input, **evaluate_kwargs)`.
        
        Attributes
        ----------
        inputs : dict.
            All model inputs, stored as `{key: ndarray}`.
        outputs : dict.
            All saved Layer outputs, stored in the same format. Special key
            '_last_output' will contain the most recent Layer output during
            evaluation, but this key is removed by `DataSet.finalize_data`.
            Will be empty if `save_output` has never been called.
        targets : dict.
            All optimization targets, stored in the same format. Will be empty
            if `target is None`.

        """
        # Set self.<attr>_name to default if <attr>_name is None, otherwise
        # save self.<attr>_name.
        names = zip(['input', 'state', 'output', 'target'],
                    [input_name, state_name, output_name, target_name])
        for attr, name in names:
            if name is None: name = getattr(self, f'default_{attr}')
            setattr(self, f'{attr}_name', name)
        if prediction_name is None: prediction_name = self.output_name
        self.prediction_name = prediction_name

        self.has_samples = has_samples
        self.debug_memory = debug_memory
        self.initialize_data(input, state, target)  # other kwargs too

        if dtype is None:
            # Set to same as first input.
            self.dtype = list(self.inputs.values())[0].dtype
        else:
            self.dtype = dtype

    @property
    def prediction(self):
        """Return only the data that should be compared to a fit target."""
        # TODO: multiple predictions
        return self.outputs[self.prediction_name]

    @property
    def n_samples(self):
        """Get size of first dimension of first input.
        
        If `DataSet.has_samples` is `False`, this method returns `None` to
        indicate that the number of samples is not known. Either there is no
        sample dimension, or there is but `DataSet` doesn't know about it.
        
        """
        if self.has_samples:
            n_samples = len(list(self.inputs.values())[0])
        else:
            n_samples = None
        return n_samples

    def initialize_data(self, input, state=None, target=None):
        """Package data into dictionaries, store in attributes.
        
        Assigns data to `DataSet.inputs, DataSet.outputs, DataSet.targets`.

        Parameters
        ----------
        input : ndarray or dict.
        state : ndarray; optional.
        target : ndarray or dict; optional.
        
        """

        # Initialize inputs
        if isinstance(input, dict):
            # Arrays in shallow copy will share memory, but the new data
            # dictionary will end up with additional keys after evaluation.
            input_dict = input.copy()
        else:
            input_dict = {self.input_name: input}
            if state is not None:
                input_dict[self.state_name] = state
        if isinstance(input, dict) | isinstance(input, np.ndarray):
            self.data_format = 'array'
        else:
            self.data_format = 'fn'

        # Initialize outputs
        output_dict = {}

        # Initialize targets
        if target is None:
            target_dict = {}
        elif isinstance(target, dict):
            target_dict = target.copy()
        else:
            target_dict = {self.target_name: target}

        # Make sure all targets are at least 2D, otherwise undesired broadcasting
        # may occurr in cost functions.
        target_dict = apply_to_dict(
            lambda a: a if a.ndim >= 2 else a[..., np.newaxis],
            target_dict, allow_copies=False
            )

        self.inputs = input_dict
        self.outputs = output_dict
        self.targets = target_dict

    def save_output(self, keys, output):
        """Save `output` in `DataSet.outputs` for keys that are not `None`.
        
        Parameters
        ----------
        keys : list of str.
            Indicates keys in `DataSet.outputs` where array(s) in `output`
            should be stored. If a key is None, the output is not saved
            (except at special key `_last_output`).
        output : ndarray or list of ndarray.
            Output of a call to Layer._evaluate.
        
        """
        # keys is always a list, but output might be a list or one array.
        if isinstance(output, (list, tuple)):
            self.outputs.update({k: v for k, v in zip(keys, output)
                        if k is not None})
        elif keys[0] is not None:
            self.outputs[keys[0]] = output
        # Always save output to _last_output for use by Model.evaluate
        self.outputs['_last_output'] = output

    def finalize_data(self, final_layer):
        """Remove special keys from `DataSet.outputs` and update final_layer.
        
        If `final_layer.output is None`, save the Layer's output to
        `DataSet.output_name` instead and update `Layer.data_map` to reflect
        the new key(s).
        
        """
        # Re-name last output if keys not specified by Layer
        final_output = self.outputs['_last_output']
        if final_layer.output is None:
            # Re-map `data_map.out` so that it reflects `output_name`.
            final_layer.data_map.map_outputs(final_output, self.output_name)
            self.save_output(final_layer.data_map.out, final_output)
        _ = self.outputs.pop('_last_output')

    def as_broadcasted_samples(self):
        """Broadcasts all data arrays against each other for number of samples.
        
        Arrays with shape (1, T, ..., N), where T is the number of time bins
        and N is the number of output channels, will broadcast to shape
        `(S, T, ..., N)`, where S is the number of samples in other data arrays.
        First all inputs are broadcast against other inputs (and outputs against
        other outputs, etc) and then inputs are broadcast against outputs and
        targets, outputs against inputs and targets, etc.

        Examples
        --------
        >>> stimulus = np.random.rand(1, 1000, 18)  # one stimulus sample
        >>> response = np.random.rand(10, 1000, 18) # multiple trials
        >>> data = DataSet(input=stimulus, target=response)
        >>> broadcasted = data.as_broadcasted_samples()
        >>> broadcasted.inputs['stimulus'].shape
        (10, 1000, 18)
        >>> np.shares_memory(broadcasted.inputs['stimulus'], stimulus)
        True
        
        """
        if self.data_format=='fn':
            # don't do anything if data is a generator
            return self

        # In case inputs/outputs and targets have different numbers of samples,
        # broadcast within each category first.
        inputs, outputs, targets = [
            broadcast_dicts(d1, d1, debug_memory=self.debug_memory)
            for d1 in [self.inputs, self.outputs, self.targets]
        ]

        # Then broadcast each category to the others.
        inputs, outputs, targets = [
            broadcast_dicts(d1, d2, debug_memory=self.debug_memory)
            for d1, d2 in [
                (inputs, {**outputs, **targets}),
                (outputs, {**inputs, **targets}),
                (targets, {**inputs, **outputs})
                ]
        ]

        copy = self.modified_copy(inputs, outputs, targets)
        copy.has_samples = True

        return copy

    def as_batches(self, batch_size=None, permute=False):
        """Generate copies of DataSet containing single batches.
        
        Parameters
        ----------
        batch_size : int; optional.
            Number of samples to include in each batch. If the total number of
            samples is not evenly divisble by `batch_size`, then one batch will
            have fewer samples. If `batch_size` is None, `batch_size` will be
            set equal to the total number of samples.
        permute : bool; default=False.
            TODO: Not implemented yet.
            If True, shuffle yield order of batches.

        Yields
        ------
        DataSet
            Dict entries will be lists of samples.

        Notes
        -----
        This implementation results in a list of views into the original data
        (i.e. memory is shared). If changes are made, make sure the new version
        doesn't result in copies (which could increase memory usage
        dramatically).

        Warnings
        --------
        This method assumes the first dimension of all data arrays represents
        samples. Unexpected behavior will result if this assumption is invalid.
        
        """

        # TODO: This handles cases of 1 sample -> many samples or vise-versa,
        #       but do we want to support n samples -> m samples? Would have
        #       to do some kind of tiling that may have side-effects that I'm
        #       not thinking of, and I'm not sure if this would be useful.
        d = self.as_broadcasted_samples()
        d.has_samples = True
        
        # Split data into batches along first axis. Should end up with a list
        # of arrays with shape (B, T, N), where B is `batch_size` (i.e. number
        # of samples per batch), stored at each key.
        batched_inputs, batched_outputs, batched_targets = [
            d._arrays_to_batches(_dict, batch_size)
            for _dict in [d.inputs, d.outputs, d.targets]
        ]

        n_batches = len(list(batched_inputs.values())[0])

        # Index into batched_data instead of collecting a list of batches,
        # to ensure memory is still shared. Also makes permutations easier.
        indices = np.arange(n_batches)
        if permute:
            # Randomly shuffle indices
            np.random.shuffle(indices)
            # TODO: Not quite this simple, have to be able to put the concatenated
            #       outputs back in the right order. So need to store the shuffled
            #       indices somehow.
            # TODO: Should samples be shuffled instead/in addition? Otherwise
            #       the same samples always end up in a batch together.
            raise NotImplementedError("Shuffling batches not implemented yet")

        for i in indices:
            inputs = {k: v[i] for k, v in batched_inputs.items()}
            outputs = {k: v[i] for k, v in batched_outputs.items()}
            targets = {k: v[i] for k, v in batched_targets.items()}
            d.assert_no_copies(inputs, outputs, targets)
            yield d.modified_copy(inputs, outputs, targets)

    def _arrays_to_batches(self, data, batch_size):
        """Internal for `as_batches`.
        
        Parameters
        ----------
        data : dict
        batch_size : int

        Returns
        -------
        dict
        
        """

        if (batch_size is None) and (len(data) > 0):
            # Assume sample dimension exists, set batch_size to force 1 batch
            batch_size = len(list(data.values())[0])
        batched_data = {
            k: np.split(v, np.arange(batch_size, len(v), batch_size))
            for k, v in data.items()
            }

        return batched_data

    def as_samples(self):
        """Generate copies of a batch DataSet, containing single samples.
        
        `DataSet.inputs, .outputs, .targets` should contain a list of arrays at
        each key, as yielded by `DataSet.as_batches`.

        Yields
        ------
        DataSet

        Warnings
        --------
        If this method is used on a DataSet that is *not* in the format yielded
        by `DataSet.as_batches`, unexpected behavior will result.

        """

        n_samples = len(list(self.inputs.values())[0])
        s_inputs, s_outputs, s_targets = [
            {k: np.split(v, n_samples) for k, v in d.items()}
            for d in [self.inputs, self.outputs, self.targets]
            ]

        for i in range(n_samples):
            inputs = {k: v[i].squeeze(axis=0) for k, v in s_inputs.items()}
            outputs = {k: v[i].squeeze(axis=0) for k, v in s_outputs.items()}
            targets = {k: v[i].squeeze(axis=0) for k, v in s_targets.items()}
            if self.debug_memory:
                self.assert_no_copies(inputs, outputs, targets)
            sample = self.modified_copy(inputs, outputs, targets)
            sample.has_samples = False
            yield sample

    def prepend_samples(self):
        """Prepend a singleton sample dimension."""
        data = self.apply(lambda v: v[np.newaxis,...], allow_copies=False)
        data.has_samples = True
        return data

    def squeeze_samples(self):
        """Remove singleton sample dimension from all arrays."""
        data = self.apply(lambda v: np.squeeze(v, axis=0), allow_copies=False)
        data.has_samples = False
        return data

    @staticmethod
    def concatenate_sample_outputs(data_sets):
        """Concatenate key-matched arrays from each `data_set.outputs`.
        
        Parameters
        ----------
        data_sets : list of DataSet

        Returns
        -------
        dict of ndarray, in the format of `DataSet.outputs`.
        
        """
        return concatenate_dicts(*[d.outputs for d in data_sets])

    def modified_copy(self, inputs, outputs, targets):
        """Get a shallow copy of DataSet with new data dictionaries.
        
        Parameters
        ----------
        inputs : dict
        outputs : dict
        targets : dict

        Returns
        -------
        DataSet
        
        """
        data = DataSet(
            inputs, state=None, target=targets, input_name=self.input_name,
            state_name=self.state_name, output_name=self.output_name,
            target_name=self.target_name, dtype=self.dtype,
            has_samples=self.has_samples,
            )
        data.outputs = outputs
        return data

    def copy(self):
        """Get a shallow copy of DataSet."""
        return self.modified_copy(self.inputs, self.outputs, self.targets)

    def apply(self, fn, *args, allow_copies=False, inplace=False, **kwargs):
        """Maps {k: v} -> {k: fn(v, *args, **kwargs)} for all k, v in DataSet.
        
        Parameters
        ----------
        fn : callable
            Must accept a single ndarray as its first positional argument.
        args : N-tuple
            Additional positional arguments for `fn`.
        allow_copies : bool; default=True.
            If False, raise AssertionError if `fn(v, *args, **kwargs)` returns
            an array that does not share memory with `v`.
        kwargs : dict
            Additional keyword arguments for `fn`.

        Returns
        -------
        DataSet
            A modified copy containing the transformed arrays.

        Examples
        --------
        TODO
        
        """
        inputs, outputs, targets = [
            apply_to_dict(fn, d, *args, allow_copies=allow_copies, **kwargs)
            for d in [self.inputs, self.outputs, self.targets]
            ]
        if inplace:
            self.inputs = inputs
            self.outputs = outputs
            self.targets = targets
        else:
            return self.modified_copy(inputs, outputs, targets)

    def assert_no_copies(self, inputs, outputs, targets):
        """Check if arrays in dictionaries share memory with arrays in DataSet.
        
        Useful for debugging memory inflation. Note that keys of arguments are
        iterated over, not keys of DataSet. This means inputs, outputs and
        targets can contain subsets of the keys present in DataSet, in which
        case not all arrays in DataSet will be checked.

        Parameters
        ----------
        inputs : dict
        outputs : dict
        targets : dict

        Raises
        ------
        AssertionError
            If `np.shares_memory(inputs[k], DataSet.inputs[k])` is False for
            any array in inputs. The same comparison is repeated for outputs
            and targets.

        """
        for k in inputs.keys():
            assert np.shares_memory(inputs[k], self.inputs[k])
        for k in outputs.keys():
            assert np.shares_memory(outputs[k], self.outputs[k])
        for k in targets.keys():
            assert np.shares_memory(targets[k], self.targets[k])

    def as_dict(self):
        """Get `DataSet.inputs, .outputs, .targets` as a single dictionary."""
        return {**self.inputs, **self.outputs, **self.targets}

    def as_type(self, dtype, inplace=False):
        """Replace all arrays with a different dtype."""
        # NOTE: Even with `copy=False`, this will result in a copy if dtype
        #       changes precision (e.g. float64 -> float32). 
        return self.apply(lambda a: a.astype(dtype, copy=False), inplace=inplace,
                          allow_copies=True)

    # Pass dict get (but not set) operations to self.inputs, outputs, targets
    def __getitem__(self, key):
        return self.as_dict()[key]
    def get(self, key, default):
        return self.as_dict().get(key, default)
    def items(self):
        return self.as_dict().items()
    def __iter__(self):
        return self.as_dict().__iter__()
    def __len__(self):
        return len(self.as_dict())
