import copy
import textwrap
import itertools

import numpy as np

from nems.registry import keyword_lib
from nems.backends import get_backend
from nems.visualization import plot_model
from .dataset import DataSet
# Temporarily import layers to make sure they're registered in keyword_lib
import nems.layers
del nems.layers


class Model:

    def __init__(self, layers=None, name=None, meta=None):
        """A structured collection of Layers.
        
        This is the primary class for interacting with NEMS. Conceptually, a
        Model encapsulates all computations needed to transform an input into
        a desired output (or prediction).

        TODO: more context here?

        Parameters
        ----------
        layers : list of Layer; optional.
            Layers that will define the Model's data transformations.
        name : str; optional.
            Name for the Model.
        meta : dict; optional.
            A general-purpose dictionary for storing additional information
            about the model.

        Attributes
        ----------
        layers : _LayerDict.
            All model layers in a format that allows both integer and
            string indexing.
        results : FitResults or None.
            A collection of information about the most recent call of
            `Model.fit()`. This will be `None` if `Model.fit()` has never been
            called for this Model.
        backend : Backend or None.
            A container for an equivalent model built using TensorFlow or some
            other supported backend, cached by the most recent call to
            `Model.fit()`. This will be `None` if `Model.fit()` has never been
            called.

        Methods
        -------
        evaluate(input, ...)
            Transform input by applying all Layers in a proscribed order.
        predict(input, ...)
            As `evaluate`, but will only return the output of the final layer
            by default.
            NOTE: Currently this method assumes there is only one model output
                and a single target.
        fit(input, target, ...)
            Produce a copy of this Model with parameters optimized to match
            the output of `Model.evaluate(input)` to `target`.
        plot(input, target=None, ...)
            Visualize the output of each Layer when applied to `input` (or the
            output of a previous Layer). If `target` is also specified, plot
            `target` on the same axes as the output of the final Layer.
            NOTE: Currently this method assumes there is only one model output
                  and a single target.

        See also
        --------
        nems.layers.base.layer.Layer
        nems.backends.base.Backend
        nems.visualization.model.plot_model

        Examples
        --------
        >>> import numpy as np
        >>> from nems import Model
        >>> # Create some fake data
        >>> input = np.random.rand(1000,18)  # 1000 time points, 18 channels
        >>> target = np.random.rand(1000, 1)
        >>> # Compute the model's output given the input data.
        >>> evaluated_data = model.evaluate(input)
        >>> # Get a fitted model
        >>> fit_model = model.fit(input, target)
        >>> # Plot the fitted model
        >>> fig = fit_model.plot(input, target)
        
        """
        self._layers = {}  #  layer.name : layer obj, increment on clashes
        if layers is not None:
            self.add_layers(*layers)
        self.name = name if name is not None else 'UnnamedModel'
        # Store optional metadata. This is a generic dictionary for information
        # about the model. Any type can be stored here as long as it can be
        # encoded by `json.dumps`.
        if meta is None: meta = {}
        self.meta = meta

        self.results = None   # holds FitResults after Model.fit()
        self.backend = None # holds all previous Backends (1 per key)

    @property
    def layers(self):
        """Get all Model Layers. Supports integer or string indexing."""
        return _LayerDict(self._layers)

    @property
    def bounds(self):
        """Get all Model bounds as a dict. See `Layer.bounds`."""
        return {k: v.bounds for k, v in self.layers.items()}

    @property
    def priors(self):
        """Get all Model priors as a dict. See `Layer.priors`."""
        return {k: v.priors for k, v in self.layers.items()}

    @property
    def parameter_count(self):
        """Total size of all Parameters from all Layers.
        
        Note that this is the total number of all parameter values, *not* the
        number of Parameter objects. I.e. a model with a single Parameter of
        shape (2,3) has a parameter_count of 6.
        TODO: rename this to avoid ambiguity? value_count? 

        Returns
        -------
        int

        See also
        --------
        Model.parameter_info
        nems.layers.base.Layer.parameter_count
        
        """
        return sum([layer.parameter_count for layer in self.layers])

    @property
    def parameter_info(self):
        """Sizes of frozen, unfrozen and permanent parameters.
        
        Returns
        -------
        dict
            {'layer_name':  # per layer
                {'total': int, 'unfrozen': int, 'frozen': int, 'permanent': int},
                ...
             'model':       # model totals
                {'total': int, unfrozen': int, ... }  # etc.
                }

        See also
        --------
        Model.parameter_count
        nems.layers.base.Layer.parameter_info
        
        """
        info = {layer.name: layer.parameter_info for layer in self.layers}
        model_info = {k: sum([d[k] for d in info.values()])
                      for k in list(info.values())[0]}
        info['model'] = model_info

        return info

    def add_layers(self, *layers):
        """Add Layers to this Model, stored in `Model._layers`.

        This will also update `Layer.name` for any layers with a name clash,
        so that each Layer in the Model is guaranteed to have a unique name.

        Parameters
        ----------
        layers : N-tuple of Layers

        See also
        --------
        nems.layers.base.Layer
        
        """

        # TODO: need to track name, layer lists instead? Apparently dictionaries
        #       aren't guaranteed to keep the same order. Hasn't caused problems
        #       so far, but...

        for layer in layers:
            layer.model = self  # each layer gets a reference to parent Model
            key = layer.name
            i = 0
            while key in self._layers:
                # Avoid name clashes
                key = f'{layer.name}{i}'
                i += 1
            self._layers[key] = layer
            # Also update `Layer.name` so that there's no mismatch between
            # a Layer's name and its key in the Model.
            layer._name = key

    def get_layer_index(self, name):
        """Get integer index for Layer with `.name == name`."""
        return list(self.layers.keys()).index(name)

    def get_layer(self, key):
        """Get one Layer. Key can be int or string (`Layer.name`)."""
        return self.layers[key]

    # Maybe we don't need to implement this? Would require some refactoring of
    # Model.layers.
    def insert_layer(self, index, name=None):
        """TODO: add layer at specific integer index."""
        raise NotImplementedError

    def evaluate(self, input, state=None, input_name=None, state_name=None,
                 output_name=None, n=None, return_full_data=True,
                 as_dataset=False, use_existing_maps=False, batch_size=0,
                 permute_batches=False):
        """Transform input(s) by invoking `Layer.evaluate` for each Layer.

        Evaluation encapsulates three steps:
            1) Package data and metadata in a single structured container.
            2) Loop over `Model.layers`, invoking `Layer._evaluate` to
               transform the data.
            3) Clean up no-longer-needed data, possibly re-format.
        See `DataSet` and `Model.generate_layer_data` for implementation.

        During the evaluation process, `input` (and `state` if provided) will
        be packaged into dictionaries, with the following structure:
            array_name : ndarray. 
                If `input` is a dict, each array in `input` will be
                shallow-copied with the same key. Otherwise, arrays `input` and
                `state` will be added to with default keys specified by
                the DataSet class.
            _last_output : ndarray, list, or None
                Return value of the most recently evaluated Layer. This key is
                removed after evaluation is complete.
        These dictionaries are tracked separately within the DataSet object
        for inputs, outputs, and targets.

        Parameters
        ----------
        input : ndarray, dict, or DataSet.
            If ndarray, use this as the input to the first Layer.
            Otherwise, use keys specified by `input_name` or `Layer.input` to
            determine the first input.
        state : ndarray; optional.
            Add this to `DataSet.inputs`. This option can only be used in
            conjunction with an array `input`. If other data is needed,
            use a dictionary input containing all data.
        input_name : str; optional.
            Specifies which array should be provided as input to the
            first layer. Note that priority will be given to `Layer.input` in
            case of a clash. I.e. if `input_name is not None`, but also
            `Layer.input is not None` for the first layer, `Layer.input` will
            be used.
            If None : use `Layer.input` of first layer if specified.
            If str  : a single input array at key `input_name`.
        state_name : str; optional.
            If str, and `state is not None`, then `state_name` will be the key
            for `state`.
        output_name : str; optional.
            Specifies name(s) for output(s) of the final layer if
            `Layer.output is None` (incremented if multiple).
        n : int; optional.
            Evaluate the first `n` Layers (all by defualt).
        return_full_data : bool; default=True
            If True, return a dictionary containing all input data and all
            uniquely-keyed Layer outputs.
        as_dataset : bool; default=False.
            Return data in a DataSet instead of a dictionary.
        use_existing_maps : bool; default=False.
            If True, existing DataMaps will be used rather than generating a
            new one for each Layer. This is disabled by default for direct
            evaluation, but enabled by default when called from `Model.fit` to
            eliminate unnecessary overhead for repeated evaluations of the same
            data.
        batch_size : int; default=0.
            Determines batching behavior.
            If 0 (default) : Data will be interpreted as having no sample
                dimension (i.e. first dimension is time). This is equivalent to
                using batch_size=1 with a single sample.
            If None : First dimension is interpreted as samples instead of time.
                Use a single batch with batch_size equal to the size of the
                sample dimension.
            Else : Form batches by select groups of samples such that each batch
                includes `batch_size` samples. If the number of samples is not
                evenly divisible by `batch_size`, then one batch will have
                fewer samples.
            Each sample within each batch will be evaluated separately, then
            their outputs will be concatenated to match the shape of the input.
        permute_batches : bool; default=False.
            TODO, still WIP. Currently doesn't track how indices are shuffled
            so the outputs won't be concatenated in the right order.
            If True, randomly shuffle batches prior to evaluation. Typically
            used during fitting, to shuffle between epochs.

        Returns
        -------
        data : ndarray, list, dict or DataSet
            Type depends on `return_full_data` and `as_dataset` options, and the
            return type of the `evaluate` method of the final layer in the model.

        See also
        --------
        nems.layers.base.Layer._evaluate
        nems.models.dataset.DataSet
        Model.generate_layer_data

        Warnings
        --------
        Since arrays in `data` share memory with `input`, modifying arrays
        arrays in-place is strongly discouraged.
        
        """

        if not isinstance(input, DataSet):
            # Package arrays and/or dicts in a DataSet
            data = DataSet(
                input=input, state=state, input_name=input_name,
                state_name=state_name, output_name=output_name,
            )
        else:
            # Input should already be a properly formatted DataSet
            # (for example, passed by Model.fit)
            data = input

        if batch_size == 0:
            # No reason to do any batching, data should be formatted as
            # (T, ..., N), where T is the number of time bins and N is the
            # number of output channels.
            data_generator = self.generate_layer_data(
                                data, use_existing_maps=use_existing_maps,
                                )
            data = self.get_layer_data(data_generator, n, n)[-1]['data']

        else:
            # Data should be formatted as (S, T, ..., N), where S is the number
            # of samples/trials.

            batch_out = list(self.generate_batch_data(
                input, state=state, input_name=input_name, state_name=state_name,
                output_name=output_name, n=n, batch_size=batch_size,
                permute_batches=permute_batches, use_existing_maps=use_existing_maps
                ))
            all_outputs = DataSet.concatenate_sample_outputs(batch_out)
            # Inputs (and any targets) should not have changed
            data.outputs = all_outputs

        if not return_full_data:
            out = data.outputs
            if len(out) == 1:
                out = list(out.values())[0]
        else:
            if as_dataset:
                out = data
            else:
                # Default option: return all data in a dictionary
                out = data.as_dict()

        return out

    def generate_batch_data(self, input, n=None, batch_size=0,
                            permute_batches=False, use_existing_maps=False,
                            **eval_kwargs):
        """Generate output of final Layer for one batch at a time.
        
        See `Model.evaluate` for detailed documentation of parameters.

        Parameters
        ----------
        input : ndarray, dict, or DataSet
        n : int; optional.
        batch_size : int or None; default=0.
        permute_batches : bool; default=False.
        use_existing_maps : bool; default=False.
        eval_kwargs : dict.
            Additional keyword arguments for `Model.evaluate`.

        Yields
        ------
        batch : DataSet
            Contains all inputs and outputs (and targets if present) for a
            single batch.
        
        """
    
        if not isinstance(input, DataSet):
            # Package arrays and/or dicts in a DataSet
            data = DataSet(input=input, **eval_kwargs)
        else:
            # Input should already be a properly formatted DataSet
            # (for example, passed by Model.fit)
            data = input

        if n is not None: n -= 1
        else: n = len(self.layers)-1

        batches = data.as_batches(batch_size, permute=permute_batches)
        for batch in batches:
            samples = batch.as_samples()
            sample_out = []
            for sample in samples:
                data_generator = self.generate_layer_data(
                    sample, use_existing_maps=use_existing_maps,
                )

                # Get data for the final layer only, to reduce memory use.
                layer_data = self.get_layer_data(data_generator, n, n)[-1]
                sample_out.append(layer_data['data'].prepend_samples())

            batch_out = DataSet.concatenate_sample_outputs(sample_out)
            batch.outputs = batch_out
            yield batch


    # TODO: possibly move this method and any related subroutines to a separate
    #       module (inside a new `base` directory), with simple wrappers in
    #       Model as the public-facing API.
    def generate_layer_data(self, input, copy_data=False,
                            use_existing_maps=False, **eval_kwargs):
        """Generate input and output arrays for each Layer in Model.
        
        This method serves as the core loop for `Model.evaluate`, but is exposed
        separately for use in plotting, debugging, etc. The loop is implemented
        as a generator to reduce memory overhead when only one Layer at a time
        is needed.

        Parameters
        ----------
        input : dict or ndarray
            See `Model.evaluate`.
        copy_data : bool; default=False.
            If True, a deep copy of data will be stored in `layer_data['data']`
            after each `Layer._evaluate`. This will be very memory intensive
            for large data, and is generally not recommended except for
            debugging.
        use_existing_maps : bool; default=False.
            If True, existing DataMaps will be used rather than generating a
            new one for each Layer. This is disabled by default for direct
            evaluation, but enabled by default when called from `Model.fit` to
            eliminate unnecessary overhead for repeated evaluations of the same
            data.
        eval_kwargs : dict
            Additional keyword arguments for `Model._initialize_data` and
            `Model._finalize_data`. See `Model.evaluate` for documentation.

        Yields
        ------
        layer_data : dict
            `layer_data` has the following structure: {
                'index': int, index of Layer within Model.
                'layer': str, Layer.name.
                'args': list of ndarray, positional arguments
                    for `Layer._evaluate`.
                'kwargs': dict of ndarray, keyword arguments
                    for `Layer._evaluate`
                'out': ndarray or list of ndarray, return value of
                    `Layer._evaluate(*args, **kwargs)`.
                'data' : DataSet.
                    See `Model.evaluate` for details.
                }

        Warnings
        --------
        layer_data['data'], is a reference to a data structure that is
        iteratively updated in-place during evaluation. Modifying this
        structure in-place is strongly discouraged, as it can violate the state
        expectations of not-yet-evaluated Layers. To make modifications safely,
        use `copy_data=True`.
        TODO: I think this is partially untrue now since each DataSet should
              be a shallow copy of the previous one. Modifying the underlying
              arrays is still discouraged, however. Need to test the first
              point.

        See also
        --------
        Model.get_layer_data
        Model.print_layer_data

        Examples
        --------
        Get a list of all outputs in memory simultaneously:
        >>> generator = generate_layer_data(input, **eval_kwargs)
        >>> data_list = [d['out'] for d, _ in generator]

        Get positional arguments for the first layer:
        >>> generator = generate_layer_data(input, **eval_kwargs)
        >>> args = next(generator)['args']

        """

        if not isinstance(input, DataSet):
            data = DataSet(input, **eval_kwargs)
        else:
            data = input

        max_n = len(self.layers)
        for n, layer in enumerate(self.layers):
            if not use_existing_maps:
                layer.reset_map()
            a, k, o = self._evaluate_layer(layer, data)
            layer_data = {
                'index': n, 'layer': layer.name,
                'args': a, 'kwargs': k, 'out': o
                }

            if n < (max_n - 1):
                if copy_data:
                    layer_data['data'] = copy.deepcopy(data)
                else:
                    layer_data['data'] = data
                yield layer_data

        # On final layer, only update data after evaluation
        data.finalize_data(final_layer=layer)
        if copy_data:
            layer_data['data'] = copy.deepcopy(data)
        else:
            layer_data['data'] = data

        yield layer_data

    def _evaluate_layer(self, layer, data):
        """Evaluates one Layer. Internal for `Model.generate_layer_data`.
        
        Returns
        -------
        args : list of ndarray
            Positional arguments for `Layer.evaluate`.
        kwargs : dict of ndarray
            Keyword arguments for `Layer.evaluate`.
        output : ndarray or list of ndarray
            Return value of `Layer.evaluate(*args, **kwargs)`.
        
        """
        
        # Get input & output arrays
        args, kwargs, output = layer._evaluate(data)

        # Save output (or don't) based on Layer.DataMap.
        # data_keys is always a list, but output might be a list or one array.
        data_keys = layer.data_map.out
        data.save_output(data_keys, output)

        return args, kwargs, output

    # TODO: maybe remove the data_generator arg and just have this wrap
    #       generate_layer_data? 
    def get_layer_data(self, data_generator, first_index=None, last_index=None):
        """Return data for layers between specified indices (inclusive).
        
        Parameters
        ----------
        data_generator : generator
            Return value of `Model.generate_layer_data`.
        first_index : int; optional.
            Index within Model of the first Layer to get data for.
        last_index : int; optional.
            Index within Model of the last Layer to get data for.

        Returns
        -------
        list of (dict, dict)
            See `Model.generate_layer_data`.

        Examples
        --------
        Get a list of all inputs & outputs in memory simultaneously:
        >>> generator = generate_layer_data(input, **eval_kwargs)
        >>> data_list = get_layer_data(generator)

        Get the keyword arguments for the 3rd Layer:
        >>> generator = generate_layer_data(input, **eval_kwargs)
        >>> kwargs3 = get_layer_data(generator, 3, 3)['kwargs']
        
        """
        if last_index is not None: last_index += 1
        subset = itertools.islice(data_generator, first_index, last_index)
        return [d for d in subset]

    def print_layer_data(self, input, max_char=79, max_array_length=20,
                         show_full_data=False, **eval_kwargs):
        """Pretty-print the return value of `Model.generate_layer_data`.

        Useful for debugging.

        Parameters
        ----------
        input : ndarray, dict, or DataSet.
            See `Model.evaluate`.
        max_char : int; default=79.
            Maximum number of characters to display on each line.
            TODO: separators currently ignore this.
        max_array_length : int; default=20.
            Show truncated arrays if they contain more than this many entries.
            Equivalent to `np.set_printoptions(threshold=max_array_length)`,
            but the previous threshold will be reset after printing.
            TODO: add precision option?
        show_full_data : bool; default=False.
            If True, print the entire DataSet after for each Layer.

        TODO: option to return string instead of printing?
        
        """
        def wrap(k, v):
            return textwrap.fill(f'{k}: {str(v)}', max_char) + '\n' + '-'*16

        current_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=max_array_length)

        for d in self.generate_layer_data(input, **eval_kwargs):
            _data = d.pop('data')

            # Input/output info
            print('_'*36 + f'in/out:' + '_'*36)
            for k, v in d.items():
                if isinstance(v, list):
                    print(f'{k}:')
                    i = 0
                    for val in v:
                        print(wrap(i, val))
                        i += 1
                    if i == 0:
                        print('-'*16)
                elif isinstance(v, dict):
                    print(f'{k}:')
                    j = 0
                    for key, value in v.items():
                        print(wrap(key, value))
                        j += 1
                    if j == 0:
                        print('-'*16)
                else:
                    print(wrap(k, v))
            print('\u203E'*79)

            if show_full_data:
                # Data dictionary
                print('_'*38 + 'data' + '_'*37)
                for k, v in _data.items():
                    print(wrap(k, v))
                print('\u203E'*79 + '\n\n')

        np.set_printoptions(threshold=current_threshold)


    def get_data_maps(self):
        """Get a dictionary of {Layer.name: Layer.DataMap} for all Layers.

        Similar to `Model.layers`, this dictionary is wrapped so that indexing
        with integers, slices, or multiple keys is also possible.
        
        Returns
        -------
        dict

        See also
        --------
        nems.layers.base.map.DataMap
        
        """
        return _LayerDict({layer.name: layer.data_map for layer in self.layers})

    def predict(self, input, return_full_data=False, **eval_kwargs):
        """As `Model.evaluate`, but return only the last output by default.
        
        TODO: This only works for models where the final layer produces the
              only output. Need to think about how to make it work for models
              that produce multiple outputs at different stages.

              Rough idea: return all arrays from data that were not present
              in input.
        
        """
        return self.evaluate(input, return_full_data=return_full_data,
                             **eval_kwargs)

    def fit(self, input, target, target_name=None, backend='scipy',
            fitter_options=None, backend_options=None, **eval_kwargs):
        """Optimize model parameters to match `Model.evaluate(input)` to target.
        
        TODO: where do jackknife indices fit in? possibly use segmentor idea
              from old NEMS that never really got implemented, as an alternative
              to requiring jackknifing to be set up ahead of time.

              simplest solution: don't deal with it here at all. can have a
              separate method/function that just loops through calls to .fit
              and sets indices in between.

        TODO: want to add explicit support for multiple targets,
              E.g. fit to two types of data simultaneously.

        Parameters
        ----------
        input : np.ndarray, dict, or Dataset.
            See `Model.evaluate`.
        target : np.ndarray or dict of np.ndarray.
            TODO: support dict
            If ndarray, this will be the fitter's target data (i.e. try to
            match the model prediction to this). If dict, ... 
            TODO: dict version is more complicated than I was thinking of.
            Would need to also specify a mapping of output -> target.
        target_name : str; optional.
            Key to assign to target, if target is an ndarray.
        backend : str; default='scipy'.
            Determines how the Model will be fit.
            If 'scipy' : Use `scipy.optimize.minimize(method='L-BFGS-B')`.
            If 'tf'    : Use TensorFlow. Also aliased as 'tensorflow'.
            TODO: any other options we want to support?
            See `nems.backends`.
        fitter_options : dict; optional.
            Keyword arguments to pass on to the fitter. For a list of valid
            options, see documentation for `Backend._fit` in the relevant
            Backend subclass.
        backend_options : dict; optional.
            Keyword arguments to pass to the Backend constructor.
        eval_kwargs : dict
            Keyword arguments to supply to `Model.evaluate`.

        Returns
        -------
        new_model : Model
            A copy of this Model with updated parameters. The Backend object
            used during the fit will be saved in `Model.backend`, and a
            FitResults object will be saved in `Model.results`.

        """

        if fitter_options is None: fitter_options = {}
        if backend_options is None: backend_options = {}

        # Initialize DataSet
        data = DataSet(
            input, target=target, target_name=target_name, **eval_kwargs
            )
        if eval_kwargs.get('batch_size', 0) != 0:
            # Broadcast prior to passing to Backend so that those details
            # only have to be tracked once.
            data = data.as_broadcasted_samples()
        # Evaluate once prior to fetching backend, to ensure all DataMaps are
        # up to date and include outputs.
        _ = self.evaluate(
            input, use_existing_maps=False, **eval_kwargs
            )

        # Update parameters of a copy, not the original model.
        new_model = self.copy()
        # Get Backend sublass.
        backend_class = get_backend(name=backend)
        # Build backend model.
        backend_obj = backend_class(new_model, data, eval_kwargs=eval_kwargs,
                                    **backend_options)
        # Fit backend, save results.
        fit_results = backend_obj._fit(
            data, eval_kwargs=eval_kwargs, **fitter_options
            )
        new_model.backend = backend_obj
        new_model.results = fit_results

        return new_model


    def score(self, prediction, target):
        # TODO: this should point to an independent utility function, but
        #       placed here for convenience (and also to provide model defaults).
        raise NotImplementedError

    def get_bounds_vector(self, none_for_inf=True):
        """Get all parameter bounds, formatted as a list of 2-tuples.

        Parameters
        ----------
        none_for_inf : bool
            If True, +/- np.inf is replaced with None
            (for scipy-compatible bounds).

        Returns
        -------
        model_bounds : list of 2-tuple

        See also
        --------
        nems.layers.base.Layer.get_bounds_vector

        """
        # collect all bounds, flatten the intermediate bounds lists
        bounds = [b for layer in self.layers for b in
                  layer.get_bounds_vector(none_for_inf=none_for_inf)]
        return bounds

    def get_parameter_vector(self, as_list=True):
        """Get all parameter values, formatted as a single vector.
        
        Parameters
        ----------
        as_list : bool
            If True, returns a list instead of ndarray
            (for scipy compatibility)

        Returns
        -------
        model_vector : list or ndarray

        See also
        --------
        nems.layers.base.Layer.get_parameter_vector

        """
        # collect all layer vectors
        vectors = []
        for layer in self.layers:
            vector = layer.get_parameter_vector(as_list=as_list)
            vectors.append(vector)
        # flatten list
        if as_list:
            model_vector = [v for vector in vectors for v in vector]
        else:
            model_vector = np.concatenate(vectors)
        
        return model_vector

    def set_parameter_vector(self, vector, ignore_checks=False):
        """Set all parameter values with a single vector.

        Parameters
        ----------
        vector : ndarray or list
            New parameter vector. Size must match the total number of flattened
            parameter values.
        ignore_checks : bool
            If True, set new values without checking size or bounds.
            (intended as a minor optimization for the scipy fitter).

        See also
        --------
        nems.layers.base.Layer.set_parameter_vector
        
        """

        first_index = 0
        for layer in self.layers:
            parameter_size = layer.parameters.unfrozen_size
            last_index = first_index + parameter_size
            layer.set_parameter_vector(vector[first_index:last_index],
                                       ignore_checks=ignore_checks)
            first_index = last_index

    def get_parameter_values(self, *layer_keys):
        """Get all parameter values, formatted as a dict.
        
        Parameters
        ----------
        layer_keys : N-tuple of strings
            Keys indicating which Layers to get parameter values for. If no keys
            are specified, get values for all layers.

        Returns
        -------
        all_values : dict

        See also
        --------
        nems.layers.base.Layer.get_parameter_values
        
        """
        if layer_keys == ():
            layer_keys = self._layers.keys()
        all_values = {}
        for k in layer_keys:
            values = self.layers[k].get_parameter_values(as_dict=True)
            all_values[k] = values
        return all_values

    def set_parameter_values(self, layer_dict):
        """Set new parameter values from key, value pairs.
        
        See also
        --------
        nems.layers.base.Layer.set_parameter_values

        """
        for k, v in layer_dict.items():
            self.layers[k].set_parameter_values(v)

    def sample_from_priors(self, n=1):
        """Get a copy of `Model` with new parameter values sampled from priors.
        
        Parameters
        ----------
        n : int; default=1.
            For `n > 1`, a list of `n` Model copies will be returned. This
            option is ignored if `inplace=True`.

        Returns
        -------
        Model, list of Model.

        See also
        --------
        nems.layers.base.Layer.sample_from_priors

        """

        m = self
        model_copies = []
        for i in range(n):
            # Can't copy all from self, otherwise samples will be the same.
            m = m.copy()
            for layer in m.layers:
                layer.sample_from_priors(inplace=True)
            model_copies.append(m)
        if n == 1:
            # Remove singleton list wrapper
            model_copies = model_copies[0]

        return model_copies

    def mean_of_priors(self, n=1):
        """Get a copy of `Model` with parameters set to mean of priors..
        
        Parameters
        ----------
        n : int; default=1.
            For `n > 1`, a list of `n` Model copies will be returned. This
            option is ignored if `inplace=True`.

        Returns
        -------
        Model, list of Model.

        See also
        --------
        nems.layers.base.Layer.mean_of_priors

        """

        model_copies = [self.copy() for i in range(n)]
        for m in model_copies:
            for layer in m.layers:
                layer.mean_of_priors(inplace=True)
        if n == 1:
            # Remove singleton list wrapper
            model_copies = model_copies[0]

        return model_copies

    def set_index(self, index, new_index='initial'):
        """Change which set of parameter values is referenced.

        Intended for use with jackknifing or other procedures that fit multiple
        iterations of a single model. Rather than using many copies of a full
        Model object, each layer tracks copies of its parameter values.

        Parameters
        ----------
        i : int
            New index for parameter copies. If `i-1` exceeds the number of
            existing copies, then new copies will be added until `i` is a
            valid index.
        new_index : str or None, default='initial'
            Determines how new values are generated if `i` is out of range.
            If `'sample'`   : sample from priors.
            Elif `'mean'`   : mean of priors.
            Elif `'initial'`: initial value of each parameter.
            Elif `'copy'`   : copy of current values.
            Elif `None`     : raise IndexError instead of adding new vectors.

        See also
        --------
        nems.layers.base.Layer.set_index

        """
        for layer in self.layers:
            layer.set_index(index, new_index=new_index)

    def freeze_layers(self, *layer_keys):
        """Invoke `Layer.freeze_parameters()` for each keyed layer.
        
        See also
        --------
        nems.layers.base.Layer.freeze_parameters

        """
        for layer in self.layers.get(*layer_keys):
            layer.freeze_parameters()

    def unfreeze_layers(self, *layer_keys):
        """Invoke `Layer.unfreeze_parameters()` for each keyed layer.
        
        See also
        --------
        nems.layers.base.Layer.unfreeze_parameters

        """
        for layer in self.layers.get(*layer_keys):
            layer.unfreeze_parameters()

    def plot(self, input, **kwargs):
        """Alias for `nems.visualization.model.plot_model`.
        
        By default, the result of each `Layer.evaluate` will be shown.
        
        """
        return plot_model(self, input, **kwargs)

    # added .summary() to mirror tensorflow models, for intuitive comparisons.
    def summary(self):
        """Prints long-form model description (alias for `print(Model)`)."""
        print(self)

    def __str__(self):
        header, tilde_break  = self._repr_helper()
        string = header
        string += tilde_break
        string += ".layers:\n\n"
        # Extra equal_break above first layer, so that its heading looks the
        # same as subsequent layers.
        string += "="*32 + "\n"
        for i, layer in enumerate(self.layers):
            if i != 0:
                # Add blank line between layers if more than one
                string += '\n'
            string += str(layer)
        string += "\n" + tilde_break

        return string

    def __repr__(self):
        header, tilde_break = self._repr_helper()
        string = header
        string += tilde_break
        for i, layer in enumerate(self.layers):
            if i != 0:
                string += "\n\n" # break between layers
            string += layer.__repr__()
        string += "\n" + tilde_break

        return string

    def _repr_helper(self):
        # Get important args/kwargs and string-format as call to constructor.
        # (also attrs, TODO)
        args = []    # TODO  --  what should be here?
        kwargs = {}  # TODO  --  what should be here?
        args_string = ", ".join([f"{a}" for a in args])
        kwargs_string = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        header = f"{type(self).__name__}({args_string}{kwargs_string})\n"
        tilde_break = "~"*64 + "\n"

        return header, tilde_break


    @classmethod
    def from_keywords(cls, *keywords):
        """Construct a Model from a list of keywords or a model string.
        
        Parameters
        ----------
        keywords : N-tuple of strings
            If first string contains hyphens, it will be interpreted as a
            "model string" where each hyphen separates two keywords.

        Returns
        -------
        Model

        See also
        --------
        nems.layers.base.Layer.from_keyword
        nems.scripts.keywords
        
        """
        # Check for kw1-kw2-... (mode; string) format.
        # If so, split into list of keywords
        split = keywords[0].split('-')
        if len(split) > 1:
            keywords = split
        # Get Layer instances by invoking `Layer.from_keyword` through registry.
        layers = [keyword_lib[kw] for kw in keywords]
        return cls(layers=layers)

    # Add compatibility for saving to json
    def to_json(self):
        """Encode a Model as a dictionary.

        TODO: after specifying some built-in Models (e.g. subclasses), determine
              if `Model.<to/from>_json` need to be updated to support those.
              As long as they're just adding specific Layers the base versions
              should work, but not sure exactly how that's going to work yet.

        Returns
        -------
        data : dict

        See also
        --------
        `nems.tools.json`

        """
        # TODO: Should .backend or .results be saved?
        data = {
            'layers': list(self._layers.values()),
            'name': self.name,
            'meta': self.meta
        }

        return data

    @classmethod
    def from_json(cls, json):
        """Decode a Model from a dictionary.

        Returns
        -------
        Model

        See also
        --------
        `nems.tools.json`

        """
        model = cls(layers=json['layers'], name=json['name'], meta=json['meta'])
        return model

    def copy(self):
        """Returns a deep copy of Model without a backend or fit results.
        
        Notes
        -----
        FitResults and Backend objects are removed because they may contain
        state-dependent objects from other packages that cannot copy correctly.
        
        """
        results = self.results
        backend = self.backend
        self.results = None
        self.backend = None
        copied_model = copy.deepcopy(self)
        self.results = results
        self.backend = backend
    
        return copied_model

    def __eq__(self, other):
        if isinstance(other, Model):
            # Layer names don't need to match, but their content does.
            return all([one == two for one, two in
                        zip(self._layers.values(), other._layers.values())])
        else:
            return NotImplemented


class _LayerDict:
    """Wrapper for Layer._layers to enable fancy-indexed gets.

    Supports: integer and string indexing, multi-indexing (one type at a time).
    Note that index assignment is not supported. To change a Model's Layers,
    use `Model.add_layers`, `Model.remove_layers`, etc.

    Examples
    --------
    >>> layers = _LayerDict({'one': 1, 'two': 2, 'three': 3})
    >>> layers
    {'one': 1, 'two': 2, 'three': 3}
    >>> layers[0]
    1
    >>> layers['one']
    1
    >>> layers['one', 'three']
    1, 3
    >>> layers['one', 0]
    KeyError: 0
    >>> layers['two'] = 22
    TypeError: '_LayerDict' object does not support item assignment

    """
    def __init__(self, _dict):
        self._dict = _dict
        self._values = list(_dict.values())

    def __getitem__(self, keys):
        # tuple([]) wrapper to enable multiple keys with Model.layers[] syntax.
        if isinstance(keys, (str, int, slice)):
            keys = tuple([keys])
        value = self.get(*keys, default=None)

        # Raise KeyError if any keys returned None
        if value is None:
            raise KeyError(keys)
        elif isinstance(value, list):
            none_in_list = [x is None for x in value]
            if np.any(none_in_list):
                raise KeyError(keys)

        return value

    def get(self, *keys, default=None):
        # no keys, get all layers
        if keys == ():
            layers = self._values
        elif isinstance(keys[0], slice):
            layers = self._values[keys[0]]
        else:
            # Require all keys str or all int, mixing not allowed.
            # This is meant to discourage hard-to-read code.
            if isinstance(keys[0], int):
                container = self._values
            else:
                container = self._dict
            layers = []
            for key in keys:
                try:
                    layer = container[key]
                except (IndexError, KeyError):
                    layer = default
                layers.append(layer)
        
        # List wrapper (to replace tuple) is just for output consistency should
        # be no practical difference in most cases.
        # Unwrap instead if it's a singleton list, *unless* keys was slice.
        if isinstance(layers, (tuple, list)):
            if (len(layers) == 1) and not isinstance(keys[0], slice):
                layers = layers[0]
            elif isinstance(layers, tuple):
                layers = list(layers)

        return layers

    def __iter__(self):
        """Iterate over Layers (not keys)."""
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._values

    def __repr__(self):
        return self._dict.__repr__()
