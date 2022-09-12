import copy

import numpy as np

from nems.registry import layer
from nems.visualization import plot_layer
from .phi import Phi
from .map import DataMap


# TODO: add more examples, and tests
# TODO: add option to propagate other Parameter options from Layer.__init__,
#       like `default_bounds` and `initial_value`.
class Layer:
    """Encapsulates one data-transformation step in a  Model.

    Attributes
    ----------
    subclasses : dict
        A dictionary of classes that inherit from Layer, used by `from_json`.
    state_arg : str or None; optional.
        If string, `state_arg` will be interpreted as the name of an
        argument for `Layer.evaluate`. During Model evaluation, if
        `Layer.input is None` and a `state` array is provided to
        `Model.evaluate`, then `state` will be added to other inputs as a
        keyword argument, i.e.: `layer.evaluate(*inputs, **state)`.

    """

    # Any subclass of Layer will be registered here, for use by
    # `Layer.from_json`
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    state_arg = None

    def __init__(self, shape=None, input=None, output=None, parameters=None,
                 priors=None, bounds=None, name=None):
        """Encapsulates one data-transformation step of a NEMS ModelSpec.

        Layers are intended to exist as components of a parent Model by
        invoking `Model.add_layer` or `Model.__init__(layers=...)`.

        Parameters
        ----------
        shape : N-tuple of int or None; optional.
            Most Layer subclasses use this keyword argument to specify the
            shape of their Parameters. See individual subclasses for expected
            format. For built-in Layers, an error will typically be raised if
            shape is not specified.
        input : str, list, dict, or None; optional.
            Specifies which data streams should be provided as inputs by
            parent Model during evaluation, where strings refer to keys for a
            dict of arrays provided to `Model.fit`.
            If None : output of previous Layer.
            If str  : a single input array.
            If list : many input arrays.
            If dict : many input arrays, with keys specifying which parameter
                      of `Layer.evaluate` each array is associated with.
            (see examples below)
        output : str, list, or None; optional.
            Specifies name(s) for array output(s) of `Layer.evaluate`.
            If None : use default name specified by parent Model.
            If str  : same name for every output (incremented if multiple).
            If list : one name per output (length must match).
            (see examples below)
        parameters : nems.layers.base.Phi or None; optional.
            Specifies values for fittable parameters used by `Layer.evaluate`.
            If None : Phi returned by `Layer.initial_parameters`.
        priors : dict of Distributions or None; optional.
            Determines prior that each Layer parameter will sample values from.
            Keys must correspond to names of parameters, such that each
            Parameter utilizes `Parameter(name, prior=priors[name])`.
            If `None` : all parameters default to Normal(mean=zero, sd=one),
            where zero and one are appropriately shaped arrays of 0 and 1.
            Individual `None` entries in a `priors` dict result in the same
            behavior for those parameters.
        bounds : dict of 2-tuples or None; optional.
            Determines minimum and maximum values for fittable parameters. Keys
            must correspond to names of parameters, such that each Parameter
            utilizes `Parameter(name, bounds=bounds[name])`.
            If None : use defaults defined in `Layer.initial_parameters`.
        name : str or None; optional.
            A name for the Layer so that it can be referenced through the
            parent Model, in addition to integer indexing.

        Warnings
        --------

        See also
        --------
        nems.models.base.Model
        nems.layers.base.Phi
        nems.layers.base.Parameter

        Examples
        --------
        Subclasses that need to overwrite `__init__()` should specify new
        arguments (if any) in the method definition, followed by **kwargs, and
        invoke super().__init__(**kwargs) to ensure all required attributes are
        set correctly. While not strictly required, this is the easiest way to
        ensure Layers function properly within a Model.

        If `initial_parameters` needs access to new attributes, they should be
        set prior to invoking `super().__init__()`. New options that interact
        with base attributes (like `Layer.parameters` or `Layer.priors`) should
        be coded after invoking `super().__init__()`, to ensure the relevant
        attributes have been set.

        For example:
        >>> def __init__(self, new_arg1, new_kwarg2=None, **kwargs):
        ...     self.something_new = new_arg1
        ...     super().__init__(**kwargs)
        ...     self.do_something_to_priors(new_kwarg2)


        When specifying input for Layers, use:
        `None` to retrieve output of previous Model Layer (default).
        `'data_key'` to retrieve a single specific array.
        `['data_key1', ...]` to retrieve many arrays. This is preferred if
            order of arguments for `Layer.evaluate` is not important
        `{'arg1': 'data_key1', ...}` to map many arrays to specific arguments
            for `Layer.evaluate`.

        >>> data = {'stimulus': stim, 'pupil': pupil, 'state1': state1,
        ...         'state2': state2}
        >>> layers = [
        ...     WeightChannels(shape=(18,4), input='stimulus'),
        ...     FIR(shape=(4, 25)),
        ...     DoubleExponential(output='LN_output'),
        ...     Sum(input=['LN_output', 'state', 'pupil'],
        ...                output='summed_output'),
        ...     LinearWeighting(
        ...         input={
        ...             'pred': 'summed_output',
        ...             'pupil': 'pupil',
        ...             'other_states': ['state1', 'state2']
        ...             }
        ...         )
        ...     ]

        """

        # input/output should be string, list of strings,
        # dict of strings, or None
        self.input = input
        self.output = output

        self.initial_priors = priors
        self.initial_bounds = bounds
        if shape is not None: shape = tuple(shape)
        self.shape = shape
        # In the event of a name clash in a Model, an integer will be appended
        # to `Layer.name` to ensure that all Layer names are unique.
        # TODO: make name property instead and save _name, default_name here.
        #       (so that other code doesn't need to check for None name)
        self._name = name 
        self.default_name = f'{type(self).__name__}'
        self.model = None  # pointer to parent Model

        if parameters is None:
            parameters = self.initial_parameters()
        self.parameters = parameters 

        # Overwrite defaults set by `initial_parameters` if priors, bounds
        # kwargs were specified.
        if priors is not None:
            self.set_priors(**priors)
        if bounds is not None:
            self.set_bounds(**bounds)

        self.data_map = DataMap(self)

    @property
    def name(self):
        """Get Layer.name if specified, or `SubClass(shape=self.shape)."""
        name = self._name
        if name is None:
            name = self.default_name
        return name

    @property
    def parameter_count(self):
        """Total size of all Parameters.
        
        TODO: rename? (see note in Model.parameter_count)
        
        """

        return self.parameters.size

    @property
    def parameter_info(self):
        """Sizes of frozen, unfrozen and permanent parameters.
        
        Returns
        -------
        dict
            {'total': int, 'unfrozen': int, 'frozen': int, 'permanent': int}

        """

        total = self.parameters.size
        unfrozen = self.parameters.unfrozen_size
        frozen = total - unfrozen
        permanent = sum([p.is_permanent*p.size for p in self.parameters])

        # TODO: rename unfrozen -> trainable, frozen -> untrainable?
        return {'total': total, 'unfrozen': unfrozen, 'frozen': frozen,
                'permanent': permanent}

    def set_dtype(self, dtype):
        """Change dtype of all Parameter values in-place."""
        self.parameters.set_dtype(dtype)

    @layer('baseclass')
    def from_keyword(keyword):
        """Construct a Layer corresponding to a registered keyword.

        Each Layer subclass can (optionally) overwrite `from_keyword` to
        enable compatibility with the NEMS keyword system. This is a
        string-based shortcut system for quickly specifying a `Model` without
        needing to import individual Layer subclasses.

        To work correctly within this system, a `from_keyword` method must
        follow these requirements:
        1) The `@layer` decorator, imported from `nems.registry`, must be added
           above the method. The decorator must receive a single string as
           its argument, which serves as the keyword "head": the identifier for
           the relevant Layer subclass. This string must contain only letters
           (i.e. alphabetical characters - no numbers, punctuation, special
           characters, etc). Keyword heads are typically short and all
           lowercase, but this is not enforced.
        2) `from_keyword` must be a static method (i.e. it receives neither
           `cls` nor `self` as an implicit argument).
        3) `from_keyword` must accept a single argument. Within the keyword
           system, this argument will be a string of the form:
           `'{head}.{option1}.{option2}...'`
           where any number of options can be specified, separated by periods.
           Options can contain any* character other than hyphens or underscores,
           which are reserved for composing many keywords at once.
           Options for builtin NEMS layers mostly follow certain formatting
           norms for consistency, but these are not enforced:
           a) Use a lowercase 'x' between dimensions for shape:
                `shape=(3,4)`     -> '3x4'
           # TODO: replace examples for below with examples for layers
           #       (but they illustrate the format in the meantime)
           b) Scientific notation refers to negative exponents:
                `tolerance=0.001` -> 't1e3'
                `max_iter=1000`   -> 'i1000' or 'i1K'
           c) Boolean options use a single lowercase letter if possible:
                `normalize=True`  -> 'n'
           # TODO: any other formatting norms I missed?
           * Users should still be aware that escapes and other characters with
           python- or system-specific meanings should be avoided/used with care.
           Generally, sticking to alpha-numeric characters is safest.
        4) Return an instance of a Layer subclass.
        
        See also
        --------
        nems.registry
        nems.layers.weight_channels.WeightChannels.from_keyword

        Examples
        --------
        A minimal version of `from_keyword` for WeightChannels can be defined
        as follows:
        >>> class WeightChannels(Layer):
        >>>     def __init__(self, shape, **kwargs):
        >>>         self.shape = shape
        >>>         super().__init__(**kwargs)
        
        >>>     @layer('wtchans')
        >>>     @staticmethod
        >>>     def from_keyword(kw):
        ...         options = kw.split('.')
        ...         for op in options:
        ...             if ('x' in op) and (op[0].isdigit()):
        ...                 dims = op.split('x')
        ...                 shape = tuple([int(d) for d in dims])
        ...         # Raises UnboundLocalError if shape is never defined.
        ...         return WeightChannels(shape)

        >>> wtchans = WeightChannels.from_keyword('wtchans.18x2')
        >>> wtchans.shape
        (18, 2)
        
        Note: the actual definition referenced above uses `'wc'` as the keyword
        head, not `'wtchans'`. The key is changed for this example to avoid
        a name clash when testing.

        """
        return Layer()

    def initial_parameters(self):
        """Get initial values for `Layer.parameters`.
        
        Default usage is that `Layer.initial_parameters` will be invoked during
        construction to set `Layer.parameters` if `parameters is None`. Each
        Layer subclass should overwrite this method to initialize appropriate
        values for its parameters, and document the Layer's parameters in the
        overwritten docstring.

        Returns
        -------
        parameters : Phi

        See also
        --------
        nems.layers.weight_channels.WeightChannels.initial_parameters

        Examples
        --------
        A minimal version of `initial_parameters` for WeightChannels can be
        defined as follows:
        >>> class WeightChannels(Layer):
        >>>     def __init__(self, shape, **kwargs):
        >>>         self.shape = shape
        >>>         super().__init__(**kwargs)
        
        >>>     def initial_parameters(self):
        ...         coeffs = Parameter(name='coefficients', shape=self.shape)
        ...         return Phi(coeffs)

        >>> wc = WeightChannels(shape=(2, 1))
        >>> wc.parameters
        Parameter(name=coefficients, shape=(2, 1))
        .values:
        [[0.]
         [0.]]
        
        """
        return Phi()

    def reset_map(self):
        """Overwrite `Layer.data_map` with a fresh copy.
        
        This will remove information about outputs and any data-dependent
        information (like first input).
        
        """
        self.data_map = DataMap(self)

    def _evaluate(self, data):
        """Get inputs from `data`, evaluate them, and update `Layer.data_map`.

        Parameters
        ----------
        data : dict
            See `Model.evaluate` for details on structure.

        Returns
        -------
        args : list of ndarray
            Positional arguments for `Layer.evaluate`.
        kwargs : dict of ndarray
            Keyword arguments for `Layer.evaluate`.
        output : ndarray or list of ndarray
            Return value of `Layer.evaluate(*args, **kwargs)`.
        
        Notes
        -----
        This method is dependent on the state of `data`, so is best thought of
        as internal to `Model.evaluate` even though it technically does not
        rely on the state of a particular Model.

        See also
        --------
        nems.models.base.Model.evaluate
        nems.layers.base.map.DataMap

        """
        args, kwargs = self.data_map.get_inputs(data)
        output = self.evaluate(*args, **kwargs)

        # Add singleton channel axis to each array if missing.
        if isinstance(output, (list, tuple)):
            output = [x[..., np.newaxis] if x.ndim == 1 else x for x in output]
        elif output.ndim == 1:
            output = output[..., np.newaxis]
        
        # Add output information to DataMap
        self.data_map.map_outputs(output)

        # args and kwargs are also returned for easier debugging
        return args, kwargs, output

    def evaluate(self, *args, **kwargs):  
        """Applies some mathematical operation to the argument(s).
        
        Each Layer subclass must overwrite this method. Any number of arguments
        is acceptable, but each should correspond to one name in `self.input`
        at runtime. An arbitrary number of return values is also allowed, and
        each should correspond to one name in `self.output`.
        
        Input and output names will be associated with arguments and return
        values, respectively, in list-order. If `self.input` is a dictionary,
        inputs will instead be mapped to specific arguments (where each
        key is the name of an argument, and each value is the name of an array
        in a `data` dictionary).

        Warnings
        --------
        Evaluate should never modify inputs in-place, as this could change the
        input to other Layers that expect the original data. Intermediate
        operations should always return copies. If a modified copy of the input
        is needed as standalone data, then it should be returned as a separate
        output (either by the same layer or a different one).
        
        Returns
        -------
        N-tuple of numpy.ndarray

        See also
        --------
        Layer.__init__

        Examples
        --------

        >>> class DummyLayer(Layer):
        >>>     def evaluate(self, x, y, z):
        ...         a = x + y
        ...         b = 2*z
        ...         return a, b
        
        >>> x = DummyLayer(input=['one', 'two', 'three'], output=['two', 'new'])
        >>> data = {'one': x, 'two': y, 'three': z}
        
        During fitting, `x.evaluate` would receive `(x, y, z)` as arguments
        (in that order) and return `(x+y, 2*z)`, resulting in:
        >>> data
        {'one': x, 'two': x+y, 'three': z, 'new': 2*z}

        """
        raise NotImplementedError(
            f'{type(self).__name__} has not defined `evaluate`.'
            )

    def description(self):
        """Short description of Layer's function.
        
        Defaults to `Layer.evaluate.__doc__` if not overwritten.

        Example
        -------
        def evaluate(self):
            '''A really long docstring with citations and notes other stuff.'''
            return a + np.exp(b-c)

        def description(self):
            return '''Implements a simple exponential: $a + e^{(b-c)}$'''

        """
        return self.evaluate.__doc__

    def as_tensorflow_layer(self, **kwargs):
        """Build an equivalent TensorFlow layer.

        Builds a subclass of NemsKerasLayer and returns an instance from:
        `SubClass(self, **kwargs)`.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for `NemsKerasLayer` or `tf.keras.layers.Layer`.

        Returns
        -------
        tf.keras.layers.Layer

        See also
        --------
        nems.tf.layer_tools.NemsKerasLayer

        Notes
        -----
        Subclasses should import TensorFlow modules inside this method, *not*
        at the top of the module. This is done to ensure that TensorFlow does
        not need to be installed unless it is being used.

        """
        raise NotImplementedError(
            f'{self.__class__} has not defined a Tensorflow implementation.'
            )

    def as_tf(self):
        """Alias for `as_tensorflow_layer`. This should not be overwritten."""
        return self.as_tensorflow_layer()

    def get_parameter_values(self, *parameter_keys, as_dict=False):
        """Get all parameter values, formatted as a list or dict of arrays.

        Parameters
        ----------
        parameter_keys : N-tuple of strings
            Keys indicating which parameters to get values for. If no keys are
            specified, all parameter values will be fetched.
        as_dict : bool
            If True, return a dict instead of a list with format
            `{parameter_key : Layer.parameters[parameter_key].values}`.

        Returns
        -------
        values : list or dict
        
        See also
        --------
        Phi.get_values
        
        """
        if parameter_keys == ():
            parameter_keys = self.parameters.keys()
        values = self.parameters.get_values(*parameter_keys)
        if as_dict:
            values = {k: v for k, v in zip(parameter_keys, values)}
        return values

    def set_parameter_values(self, *parameter_dict, ignore_bounds=False,
                             **parameter_kwargs):
        """Set new parameter values from key, value pairs.

        Parameters
        ----------
        parameter_dict : dict; optional
            Dictionary containing key-value pairs, where each key is the name
            of a parameter and each value is the new value for that parameter.
        ignore_bounds : bool
            If True, ignore `Parameter.bounds` when setting new values
            (not usually recommended, but useful for testing).
        parameter_kwargs : key-value pairs.
            As `parameter_dict`.

        """
        self.parameters.update(*parameter_dict, ignore_bounds=ignore_bounds,
                               **parameter_kwargs)

    def get_parameter_vector(self, as_list=True):
        """Get all parameter values, formatted as a single vector.
        
        Parameters
        ----------
        as_list : bool
            If True, returns a list instead of ndarray.

        Returns
        -------
        list or ndarray

        See also
        --------
        Phi.get_vector

        """
        return self.parameters.get_vector(as_list=as_list)

    def set_parameter_vector(self, vector, ignore_checks=False):
        """Set parameter values with a single vector.

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
        Phi.set_vector
        
        """
        self.parameters.set_vector(vector, ignore_checks=ignore_checks)

    def get_bounds_vector(self, none_for_inf=True):
        """Get all parameter bounds, formatted as a list of 2-tuples.

        Parameters
        ----------
        none_for_inf : bool
            If True, +/- np.inf is replaced with None
            (for scipy-compatible bounds).

        Returns
        -------
        list of 2-tuple

        See also
        --------
        Phi.get_bounds

        """
        return self.parameters.get_bounds_vector(none_for_inf=none_for_inf)

    @property
    def bounds(self):
        """Get all parameter bounds, as a dict with one key per Parameter."""
        return self.parameters.bounds
    
    def set_bounds(self, *parameter_dict, **parameter_kwargs):
        """Set all parameter bounds fromm key-value pairs.
        
        Warnings
        --------
        This can force the current `Parameter.values` out of bounds. To ensure
        valid values, a subsequent `Parameter.sample` or `Parameter.mean` is
        recommended.

        """
        self.parameters.set_bounds(*parameter_dict, **parameter_kwargs)

    @property
    def priors(self):
        """Get all parameter priors, as a dict with one key per Parameter."""
        return self.parameters.priors

    def set_priors(self, *parameter_dict, **parameter_kwargs):
        """Set all parameter bounds fromm key-value pairs.
        
        Warnings
        --------
        This can force the current `Parameter.values` to be outside the range
        of `Parameter.prior`. To ensure valid values, a subsequent
        `Parameter.sample` or `Parameter.mean` is recommended.

        """
        self.parameters.set_priors(*parameter_dict, **parameter_kwargs)

    def sample_from_priors(self, inplace=True, as_vector=False):
        """Get or set new parameter values by sampling from priors.
        
        Parameters
        ----------
        inplace : bool, default=True
            If True, sampled values will be used to update each Parameter
            (and, in turn, Phi._array) inplace. Otherwise, the
            sampled values will be returned without changing current values.
        as_vector : bool, default=False
            If True, return sampled values as a flattened vector instead of a
            list of arrays.

        Returns
        -------
        samples : ndarray or list of ndarray

        See also
        --------
        nems.layers.base.Phi.sample

        """
        return self.parameters.sample(inplace=inplace, as_vector=as_vector)

    def mean_of_priors(self, inplace=True, as_vector=False):
        """Get, or set parameter values to, mean of priors.
        
        Parameters
        ----------
        inplace : bool, default=True
            If True, mean values will be used to update each Parameter
            (and, in turn, `Phi._array`) inplace. Otherwise, means
            will be returned without changing current values.
        as_vector : bool, default=False
            If True, return means as a flattened vector instead of a
            list of arrays.

        Returns
        -------
        means : ndarray or list of ndarray

        See also
        --------
        nems.layers.base.Phi.mean

        """
        return self.parameters.mean(inplace=inplace, as_vector=as_vector)

    def freeze_parameters(self, *parameter_keys):
        """Use parameter values for evaluation only, do not optimize.

        Parameters
        ----------
        parameter_keys : N-tuple of str
            Each key must match the name of a Parameter in `Layer.parameters`.
            If no keys are specified, all parameters will be frozen.
        
        See also
        --------
        Phi.freeze_parameters
        Parameter.freeze
        
        """
        self.parameters.freeze_parameters(*parameter_keys)

    def unfreeze_parameters(self, *parameter_keys):
        """Make parameter values optimizable.

        Parameters
        ----------
        parameter_keys : N-tuple of str
            Each key must match the name of a Parameter in `Layer.parameters`.
            If no keys are specified, all parameters will be unfrozen.
        
        See also
        --------
        Phi.unfreeze_parameters
        Parameter.unfreeze

        """
        self.parameters.unfreeze_parameters(*parameter_keys)

    def set_permanent_values(self, *parameter_dict, **parameter_kwargs):
        """Set parameters to fixed values. The parameters will not unfreeze."""
        self.parameters.set_permanent_values(*parameter_dict, **parameter_kwargs)

    def set_index(self, i, new_index='initial'):
        """Change which set of parameter values is referenced.

        Intended for use with jackknifing or other procedures that fit multiple
        iterations of a single model. Rather than using many copies of a full
        Model object, each Phi object tracks copies of the underlying vector of
        parameter values.

        Parameters
        ----------
        i : int
            New index for `Phi._array`. If `i >= len(Phi._array)`, then new
            vectors will be appended until `Phi._array` is sufficiently large.
        new_index : str or None, default='initial'
            Determines how new vectors are generated if `i` is out of range.
            If `'sample'`   : invoke `Phi.sample(inplace=False)`.
            Elif `'mean'`   : invoke `Phi.mean(inplace=False)`.
            Elif `'initial'`: set to `[p.initial_value for p in <Parameters>]`.
            Elif `'copy'`   : copy current `Phi.get_vector()`.
            Elif `None`     : raise IndexError instead of adding new vectors.

        See also
        --------
        nems.models.base.Model.set_index

        """
        self.parameters.set_index(i, new_index=new_index)

    def plot(self, output, fig=None, ax=None, **plot_kwargs):
        """Alias for `nems.visualization.model.plot_layer`.

        By default, layer output will be represented by a single 2D line plot
        with time on the x-axis.

        Parameters
        ----------
        output : list of ndarray
            Data returned by `Layer.evaluate`.
        fig : matplotlib.figure.Figure; optional.
            The figure on which the plot will be rendered. If not provided, a
            new figure will be generated.
        plot_kwargs : dict
            Additional keyword arguments to be supplied to
            `matplotlib.axes.Axes.plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure

        See also
        --------
        nems.visualization.model.plot_model
        nems.visualization.model.plot_layer

        Notes
        -----
        Subclasses can overwrite this method to specify a custom plot.
        To be compatible with `nems.tools.plotting.plot_model`, the overwritten
        method must accept `output` as the first argument and `fig` as a keyword
        argument. The plot must be rendered on the provided Figure (unless
        `fig is None`), but `output` does not necessarily need to be used (for
        example, when visualizing model parameters).

        Examples
        --------
        >>> class MyNewLayer(Layer):
        >>>     def evaluate(self, *inputs):
        >>>         return [2*x for x in inputs]
        >>>     def plot(self, output, fig=None):
        >>>         plot_data = some_special_routine(output)
        >>>         if fig is None:
        >>>             fig = plt.figure()
        >>>         ax = fig.subplots(1,1)
        >>>         ax.plot(plot_data)
        >>>         return fig
        >>> # Use evaluate to generate output.
        >>> layer = MyNewLayer()
        >>> output = layer.evaluate(np.random.rand(100, 1))
        >>> layer.plot(output)
        
        """
        return plot_layer(output, fig=fig, ax=ax, **plot_kwargs)

    @property
    def plot_kwargs(self):
        """Get default keyword arguments for `Layer.plot`.
        
        These kwargs will be used by `nems.visualization.model.plot_model`.

        Returns
        -------
        dict
            If `Layer.plot` has *not* been overwritten, each key must correspond
            to one keyword argument for `matplotlib.axes.Axes.plot`, such that
            `ax.plot(..., **Layer.plot_kwargs)` is valid.

            If `Layer.plot` *has* been overwritten, each key must correspond to
            one keyword argument for `Layer.plot`, such that
            `Layer.plot(..., **Layer.plot_kwargs)` is valid.

        See also
        --------
        nems.visualization.model.plot_model
        
        """
        return {}
    
    @property
    def plot_options(self):
        """Get default plot options for `Layer.plot`.

        These options will be used by `nems.visualization.model.plot_model`.

        Returns
        -------
        dict
            Dictionary keys should correspond to a subset of the keys in
            `nems.visualization.model._DEFAULT_PLOT_OPTIONS`. See function
            `set_plot_options` in the same module for behavior.
        
        See also
        --------
        nems.visualization.model.plot_model
        nems.visualization.model.set_plot_options

        Notes
        -----
        To prevent all default plot options from being used (for example, to
        hard-code specific options in an overwritten `Layer.plot` method), set
        `{'skip_plot_options': True}` in the returned dictionary.
        
        """
        return {}

    # Passthrough *some* of the dict-like interface for Layer.parameters
    # NOTE: 'get' operations through this interface return Parameter instances,
    #       not arrays. For array format, reference Parameter.values or use
    #       `Layer.get_parameter_values`.
    def __getitem__(self, key):
        """Get Parameter (not Parameter.values)."""
        return self.parameters[key]
    
    def get(self, key, default=None):
        """Get Parameter (not Parameter.values)."""
        return self.parameters.get(key, default=default)

    def __setitem__(self, key, val):
        """Set Parameter.values (not Parameter itself)."""
        self.parameters[key] = val

    def __iter__(self):
        """Return an iter-wrapped singleton list containing this Layer.

        Supports compatibility for iterating over Model layers when single
        Layers are returned, without the need to repeatedly add `[0]` to the
        end of indexing statements.
        
        NOTE: This does *not* iterate over the Layer's parameters, and is not
              part of the passed-through dict interface for `Layer.Phi`. To
              iterate over parameters, refer iterators directly to
              `Layer.parameters`.
              E.x: `for p in Layer.parameters` instead of `for p in Layer`.

        TODO: This feels kludgy and I don't like it, but it allows for some
              nice syntax for navigating Models.
        
        """
        return iter([self])

    def __str__(self):
        header, equal_break = self._repr_helper()
        string = header + equal_break + "\n"
        if self.state_arg is not None:
            string += f".state_arg:  {self.state_arg}\n"
        string += ".parameters:\n\n"
        p_string = str(self.parameters)
        string += p_string
        if p_string != '':
            # Only add an end-cap equal_break if there was at least one
            # parameter to show
            string += equal_break
        return string

    def __repr__(self):
        header, equal_break = self._repr_helper()
        string = header + equal_break + "\n"
        p_string = self.parameters.__repr__()
        string += p_string
        if p_string != '':
            # Only add an end-cap equal_break if there was at least one
            # parameter to show
            string += equal_break
        return string

    def _repr_helper(self):
        layer_dict = Layer().__dict__
        self_dict = self.__dict__
        # Get attributes that are not part of base class, unless they start
        # with an underscore, then convert to string with "k=v" format
        self_only = ", ".join([f"{k}={v}" for k, v in self_dict.items()
                               if (k not in layer_dict) and not
                               (k.startswith('_'))])
        if self_only != "":
            # Prepend comma to separate from shape
            self_only = ", " + self_only
        header = f"{type(self).__name__}(shape={self.shape}{self_only})\n"
        equal_break = "="*32

        return header, equal_break


    # Add compatibility for saving to .json
    def to_json(self):
        """Encode a Layer as a dictionary.

        This (base class) method encodes all attributes that are common to
        all Layers. Subclasses that need to save additional kwargs or attributes
        should overwrite `to_json`, but invoke `Layer.to_json` within the new
        method as a starting point (see example below).
        
        See also
        --------
        'Layer.from_json`
        `nems.tools.json`
        `nems.layers.weight_channels.WeightChannels.to_json`

        Examples
        --------
        >>> class DummyLayer(Layer):
        >>>     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        ...         important_attr = None
        >>>     def update_important_attr(self, *args):
        ...         important_attr = do_stuff(args)
        ...     # ...
        ...     # Want to preserve the state of `important_attr` in encoding.
        >>>     def to_json(self):
        ...         data = Layer.to_json(self)
        ...         data['attributes'].update(
        ...             important_attr=self.important_attr
        ...             )
        ...         return data

        """
        data = {
            'kwargs': {
                'input': self.input, 'output': self.output,
                'parameters': self.parameters, 'priors': self.priors,
                'bounds': self.bounds, 'name': self.name, 'shape': self.shape,
            },
            'attributes': {},
            'class_name': type(self).__name__
            }

        return data

    @classmethod
    def from_json(cls, json):
        """Decode a Layer from a dictionary.

        Parameters
        ----------
        json : dict
            json data encoded by `Layer.to_json`.

        Returns
        -------
        layer : Layer or subclass
        
        See also
        --------
        `Layer.to_json`
        `nems.tools.json`.

        """
        if json['class_name'] == 'Layer':
            subclass = Layer
        else:
            subclass = cls.subclasses[json['class_name']]
        if subclass.from_json.__qualname__ != Layer.from_json.__qualname__:
            # Subclass has overwritten `from_json`, use that method instead.
            layer = subclass.from_json()
        else:
            layer = subclass(**json['kwargs'])
            for k, v in json['attributes'].items():
                setattr(layer, k, v)

        return layer

    def __eq__(self, other):
        # Must be the same Layer subclass and have the same parameters.
        if type(self) is type(other):
            return self.parameters == other.parameters
        else:
            return NotImplemented
