class DataMap:
    def __init__(self, layer):
        self.layer = layer
        self.name = layer.name
        self.input = layer.input
        if not isinstance(self.input, list):
            self.input = [self.input]
        self.output = layer.output

        # Special check for adding state kwarg when no input is specified
        self.state_arg = layer.state_arg
        self.add_state = False
        if (layer.input is None) and (layer.state_arg is not None):
            self.add_state = True

        self.state = layer.state_arg
        self.out = []
        self._build_input_map()  # assigns self.args, self.kwargs

    def _build_input_map(self):
        """Map `Layer.input` to *args and **kwargs for `Layer.evaluate`.
        
        Adds keys to `self.args` and `self.kwargs`, corresponding to keys in
        `data`.
        
        """
        args = []
        kwargs = {}
        for key in self.input:
            if isinstance(key, dict):
                kwargs.update({k: v for k, v in key.items()})
            else:
                args.append(key)
        if self.add_state:
            # This will be substituted with the actual key when fetching arrays
            kwargs[self.state_arg] = '_state_name'

        self.args = args
        self.kwargs = kwargs

    def map_outputs(self, eval_out, output_name=None):
        """Determine where/whether to save `eval_out` in `data`.

        Adds keys to `self.out`. Entries will be all None (`eval_out` is not
        saved) or all strings (`data[string] = eval_out`). If `self.output` is
        a string and `len(eval_out) > 1`, then that string will be incremented
        in the form of `string.{i}` to get a unique key for each output.
        
        """
        if not isinstance(eval_out, (list, tuple)):
            eval_out = [eval_out]
        output = self.output
        if (output_name is not None) and (output is None):
            # Intended for use by `Model.evaluate` on final layer.
            output = output_name

        if output is not None:
            if isinstance(output, str):
                output_keys = [
                    f"{self.output}.{i}" if i != 0 else f"{output}"
                    for i in range(len(eval_out))
                    ]
            elif isinstance(output, list):
                if len(output) != len(eval_out):
                    raise ValueError(
                        f"Layer {self.layer.name} specified "
                        f"{len(output)} outputs, but .evaluate returned "
                        f"{len(eval_out)} outputs."
                    )
                output_keys = output
            else:
                raise TypeError(
                    f"Unrecognized type for {self.name}.output:"
                    f"{type(output)}"
                    )
        else:
            output_keys = [None]*len(eval_out)
        self.out = output_keys

    def get_inputs(self, data):
        """Get arrays from `data` that correspond to inputs for `Layer.evaluate`.
        
        Parameters
        ----------
        data : dict of ndarray.
            Input data for Model. See `Model.evaluate` for details on structure.
        input_name : str
            Key for input to first layer. Only used if `Layer.input is None`
            and `data['last_output']` is None.

        Returns
        -------
        (args, kwargs, out) : (list, dict, list)
            Entries are ndarrays pulled from `data`.

        Warnings
        --------
        The output of this method is dependent upon the state of `data`, and
        is designed to be used as part of the Model.evaluate` loop. Otherwise,
        care must be taken to ensure that the correct keys are present in `data`
        and are associated with the correct arrays.
        
        """
        last_out = data.get('_last_output', None)
        if (last_out is None) and (self.layer.input is None):
            # First statement should only be True for first Layer
            self.args = [data.input_name]
        missing_input = ("Cannot determine Layer input in Model.evaluate, \n"
                         f"specify `last_out` or `{self.name}.input`.")

        args = []
        for a in self.args:
            if a is None:
                # Supply each of the previous outputs as a separate input,
                # i.e. `layer.evaluate(*last_out)`.
                if last_out is None:
                    raise ValueError(missing_input)
                args.append(last_out)
            elif isinstance(a, list):
                # Supply a list of arrays at all keys as a single argument,
                # i.e. `layer.evaluate([array1, array2, ...])`.
                args.append([data[k] for k in a])
            else: # (a should be a string)
                # Add one array to arguments,
                # i.e. `layer.evaluate(array)`.
                args.append(data[a])

        update_state=None
        kwargs = {}
        for k, v in self.kwargs.items():
            if v is None:
                # Supply keyword `k` with a list of the previous outputs,
                # i.e. `layer.evaluate(k=[out1, out2, ...])`.
                if last_out is None:
                    raise ValueError(missing_input)
                kwargs[k] = last_out
            elif isinstance(v, list):
                # Supply keyword `k` with a list of the indicated arrays,
                # i.e. `layer.evaluate(k=[array1, array2, ...])`.
                kwargs[k] = [data[_k] for _k in v]
            else: # (v should be a string)
                # Supply keyword `k` with a single array,
                # i.e. `layer.evaluate(k=array1)`.
                if v == '_state_name':
                    # Special check for inserting state array
                    v = data.state_name  # replace _state_name with actual key
                    update_state = {k: v}
                kwargs[k] = data[v]
        
        # Replace _state_name
        if update_state is not None:
            self.kwargs.update(update_state)

        return args, kwargs

    @property
    def keys(self):
        return {'args': self.args, 'kwargs': self.kwargs, 'out': self.out}

    def __repr__(self):
        name = self.layer.name
        if name is None:
            name = self.layer.default_name
            
        s = f"DataMap(layer={name})\n"
        s += f".args: {self.args}\n"
        s += f".kwargs: {self.kwargs}\n"
        s += f".out: {self.out}\n"
        s += f"*"*16 + "\n"
        return s

    def __iter__(self):
        """Return an iter-wrapped singleton list containing this DataMap.

        Supports compatibility for iterating over DataMaps returned by
        `Model.get_data_maps` when single Layers are indexed, without the need
        to repeatedly add `[0]` to the end of indexing statements.
        
        NOTE: This does *not* iterate over the DataMap itself.
        TODO: This feels kludgy and I don't like it, but it allows for some
              nice syntax for navigating Models.
        
        """
        return iter([self])
