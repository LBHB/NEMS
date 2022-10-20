import numpy as np

from .parameter import Parameter


# TODO: add examples and tests
#       (for both Phi and Parameter)
# TODO: TF compatibility functions? during fitting, values would have to exist
#       as TF primitives, but need to be able to translate to/from
#       TF and NEMS representations.

class Phi:
    """Stores, and manages updates to, Parameters for one Layer."""

    def __init__(self, *parameters):
        """Stores, and manages updates to, Parameters for one Layer.

        In general, Phi instances should not need to be interacted with directly
        unless implementing a new Layer subclass or a related function. Instead,
        parameters should be accessed through Model- or Layer-level methods.

        Additionally, the set of Parameters assigned to a Phi object is meant
        to be fixed at construction. To get a Phi object with some parameters
        either added or removed, construct a new Phi instance or use
        `Phi.modified_copy`.

        Parameters
        ----------
        parameters : N-tuple of Parameter; optional

        See also
        --------
        nems.models.base.Model
        nems.layers.base.Layer
        nems.layers.base.Parameter
        
        """
        self._array = [[]]
        self._index = 0
        self._dict = {}
        self.size = 0
        # Add parameter values to nested list
        for p in parameters:
            self._add_parameter(p)
        # Convert to ndarray
        self._array = np.array(self._array)
        # Cached indexing into `_array` for `get_vector` and `set_vector`.
        self._vector_mask = None
        self._update_vector_mask()

    @property
    def dtype(self):
        return self._array.dtype

    def set_dtype(self, dtype):
        """Change dtype of all Parameter values in-place."""
        self._array = self._array.astype(dtype)

    # TODO: if it becomes necessary, we can relax this restriction and allow
    #       for adding/removing Parameters. But I can't think of a case where
    #       it would really be needed since Layer parameters aren't meant to
    #       change, and this keeps the implementation of other methods simpler.
    def _add_parameter(self, parameter):
        """Add a new parameter to `Phi._dict` and update `Phi._array`.
        
        Sets `parameter.phi`, `parameter.first_index`, and
        `parameter.last_index` to comply with `Phi.ge_vector()` formatting.
        
        This method should only be invoked during construction, since
        `Phi._array` will be converted to ndarray afterward. This limitation
        is intentional: a Phi object is meant to represent a fixed set of
        model parameters. If different parameters are needed after construction,
        create a new Phi object.
        
        """
        # Start at end of existing vector, track size for next parameter
        parameter.first_index = self.size
        self.size += parameter.size
        parameter.last_index = self.size-1
        # Always only one "row" during construction, so use 0 index.
        self._array[0].extend(parameter.initial_value)
        self._dict[parameter.name] = parameter
        parameter.phi = self

    def _get_mask(self, *index_ranges):
        """Get index mask into current vector within `Phi._array`.

        Parameters
        ----------
        index_ranges : N-tuple of 2-tuples
            First tuple entry = first index, second tuple entry = last index.

        Returns
        -------
        mask : boolean ndarray, shape=Phi._array.shape

        Notes
        -----
        Using `mask` for selection will result in a copy of `Phi._array`.
        Using `mask` for assignment will change values of `Phi._array` itself.
        
        """
        mask = np.full(self._array.shape, False)
        for first, last in index_ranges:
            mask[self._index][first:last+1] = True
        return mask

    def _update_vector_mask(self):
        """Update cached copy of current mask for `Phi.<get/set>_vector`.
        
        This method must be invoked any time there is a change to the indices
        within `Phi._array` to which `Phi.<get/set>_vector` would refer
        (i.e. parameters are frozen/unfrozen, `Phi._index` changes, etc).
        
        """
        parameter_ranges = [
            (p.first_index, p.last_index) for p in self._dict.values()
            if not p.is_frozen
            ]
        self._vector_mask = self._get_mask(*parameter_ranges)

    def get_vector(self, as_list=False):
        """Get a copy of `Phi._array` sliced at `Phi._index`.
        
        Parameters
        ----------
        as_list : bool
            If True, return `vector.tolist()` instead of `vector`.

        Returns
        -------
        vector : ndarray or list
        
        """
        vector = self._array[self._vector_mask]
        if as_list:
            vector = vector.tolist()
        return vector

    def get_parameter_from_index(self, i):
        """Get reference to Parameter corresponding to vector index."""
        j = 0
        for p in self._dict.values():
            j += p.size
            if i >= j:
                # Not part of this Parameter
                continue
            else:
                # Part of this Parameter
                return p

    def get_bounds_vector(self, none_for_inf=True):
        """Return a list of bounds from each parameter in `Phi._dict`.
        
        Will not include bounds for frozen parameters.

        Parameters
        ----------
        none_for_inf : bool, default=True
            If True, replace (+/-)`np.inf` with None for compatibility with
            `scipy.optimize.minimize`.
        
        Returns
        -------
        bounds : list of 2-tuples
        
        """
        # Flatten the list returned by each Parameter
        bounds = [b for p in self._dict.values() if not p.is_frozen
                  for b in p.get_bounds_vector(none_for_inf=none_for_inf)]
        return bounds

    @property
    def bounds(self):
        """Get all parameter bounds, as a dict with one key per Parameter."""
        return {k: v.bounds for k, v in self._dict.items()}

    def set_bounds(self, *dct, **kwargs):
        """Set all parameter bounds fromm key-value pairs.
        
        Warnings
        --------
        This can force the current `Parameter.values` out of bounds. To ensure
        valid values, a subsequent `Parameter.sample` or `Parameter.mean` is
        recommended.

        """
        if dct != ():
            _dct = dct[0]
        else:
            _dct = kwargs
        for k, v in _dct.items():
            if v is not None: v = tuple(v)
            self._dict[k].bounds = v

    @property
    def priors(self):
        """Get all parameter priors, as a dict with one key per Parameter."""
        return {k: v.prior for k, v in self._dict.items()}
    
    def set_priors(self, *dct, **kwargs):
        """Set all parameter bounds fromm key-value pairs.
        
        Warnings
        --------
        This can force the current `Parameter.values` to be outside the range
        of `Parameter.prior`. To ensure valid values, a subsequent
        `Parameter.sample` or `Parameter.mean` is recommended.

        """
        if dct != ():
            _dct = dct[0]
        else:
            _dct = kwargs
        for k, v in _dct.items():
            self._dict[k].priors = v

    def get_indices_outof_range(self, vector, as_bool=True):
        """Get indices where `vector < bounds[0]` or `vector > bounds[1]`."""
        bounds = self.get_bounds_vector(none_for_inf=False)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        if as_bool:
            indices = np.logical_or(vector < lower, vector > upper)
        else:
            check_low = np.argwhere(vector < lower)
            check_high = np.argwhere(vector > upper)
            indices = np.vstack([check_low, check_high]).flatten()
        
        return indices

    def within_bounds(self, vector):
        """False if anywhere `vector < bounds[0]` or `vector > bounds[1]`."""
        return not np.any(self.get_indices_outof_range(vector, as_bool=True))

    def set_vector(self, vector, ignore_checks=False):
        """Set values of `Phi._array` sliced at `Phi._index` to a new vector.

        Parameters
        ----------
        vector : ndarray or list
            New parameter values. Size must match `Phi.get_vector`.
        ignore_checks : bool
            If True, set new values without checking size or bounds.
            (intended as a minor optimization for the scipy fitter).
        
        """
        if not ignore_checks:
            if np.array(vector).size != self.unfrozen_size:
                raise ValueError(f"Size of new vector != Phi.get_vector.")
            if not self.within_bounds(vector):
                bad_indices = self.get_indices_outof_range(vector, as_bool=False)
                raise ValueError("Vector out of bounds at indices:\n"
                                 f"{bad_indices}.")

        self._array[self._vector_mask] = vector

    def _get_parameter_mask(self, p):
        """Get an index mask as in `Phi._get_mask`, but for one Parameter."""
        return self._get_mask((p.first_index, p.last_index))

    def _get_parameter_vector(self, p):
        """Get a sliced copy of `Phi._array` corresponding to one Parameter."""
        mask = self._get_parameter_mask(p)
        return self._array[mask]
        
    def freeze_parameters(self, *parameter_keys):
        """Use parameter values for evaluation only, do not optimize.

        Updates `Phi._vector_mask`.

        Parameters
        ----------
        parameter_keys : N-tuple of str
            Each key must match the name of a Parameter in `Phi._dict`.
            If no keys are specified, all parameters will be frozen.
        
        See also
        --------
        Layer.freeze_parameters
        Parameter.freeze
        
        """
        if parameter_keys == ():
            # no keys given, freeze all parameters
            parameter_keys = list(self._dict.keys())
        for pk in parameter_keys:
            p = self._dict[pk]
            if not p.is_frozen:
                p.freeze()            
        self._update_vector_mask()

    def unfreeze_parameters(self, *parameter_keys):
        """Make parameter values optimizable.

        Updates `Phi._vector_mask`.

        Parameters
        ----------
        parameter_keys : N-tuple of str
            Each key must match the name of a Parameter in `Phi._dict`.
            If no keys are specified, all parameters will be unfrozen.
        
        See also
        --------
        Layer.unfreeze_parameters
        Parameter.unfreeze

        """
        if parameter_keys == ():
            # no keys given, freeze all parameters
            parameter_keys = list(self._dict.keys())
        for pk in parameter_keys:
            p = self._dict[pk]
            if p.is_frozen:
                p.unfreeze()
        self._update_vector_mask()

    @property
    def unfrozen_size(self):
        """Get number of values corresponding to unfrozen parameters."""
        return len(self.get_vector(as_list=True))

    def set_permanent_values(self, *dct, **kwargs):
        """Set parameters to fixed values. The parameters will not unfreeze."""
        if dct != ():
            _dct = dct[0]
        else:
            _dct = kwargs
        for k, v in _dct.items():
            p = self._dict[k]
            p.update(v)
            p.make_permanent()
        
        self._update_vector_mask()

    def sample(self, inplace=False, as_vector=True):
        """Get or set new parameter values by sampling from priors.
        
        Parameters
        ----------
        inplace : bool, default=False
            If True, sampled values will be used to update each Parameter
            (and, in turn, `Phi._array`) inplace. Otherwise, the sampled values
            will be returned without changing current values.
        as_vector : bool, default=True
            If True, return sampled values as a flattened vector instead of a
            list of arrays.

        Return
        ------
        samples : ndarray or list of ndarray

        """
        samples = [p.sample(inplace=inplace) for p in self._dict.values()]
        if not inplace:
            if as_vector:
                unravelled = [np.ravel(s) for s in samples]
                samples = np.concatenate(unravelled)
            return samples

    def mean(self, inplace=False, as_vector=True):
        """Get, or set parameter values to, mean of priors.
        
        Parameters
        ----------
        inplace : bool, default=False
            If True, mean values will be used to update each Parameter
            (and, in turn, `Phi._array`) inplace. Otherwise, means will be
            returned without changing current values.
        as_vector : bool, default=True
            If True, return means as a flattened list instead of a
            list of arrays.

        Return
        ------
        means : list

        """
        means = [p.mean(inplace=inplace) for p in self._dict.values()]
        if not inplace:
            if as_vector:
                unravelled = [np.ravel(m) for m in means]
                means = np.concatenate(unravelled)
            return means


    def set_index(self, i, new_index='initial'):
        """Change which vector to reference within `Phi._array`.

        Updates `Phi._vector_mask`.

        Parameters
        ----------
        i : int
            New index for `Phi._array`. If `i >= len(Phi._array)`, then new
            vectors will be appended until `Phi._array` is sufficiently large.
        new_index : str or None, default='initial'.
            Determines how new vectors are generated if `i` is out of range.
            If `'sample'`   : invoke `Phi.sample()`.
            Elif `'mean'`   : invoke `Phi.mean()`.
            Elif `'initial'`: set to `[p.initial_value for p in <Parameters>]`.
            Elif `'copy'`   : copy current `Phi.get_vector()`.
            Elif `None`     : raise IndexError instead of adding new vectors.

        """
        array_length = len(self._array)
        if i >= array_length:
            # Array isn't that big yet, so add new vector(s).
            new_indices = range(array_length, i+1)
            if new_index == 'sample':
                new_vectors = [self.sample() for j in new_indices]
            elif new_index == 'mean':
                new_vectors = [self.mean() for j in new_indices]
            elif new_index == 'initial':
                new_vectors = [
                    np.concatenate([p.initial_value
                                    for p in self._dict.values()])
                    for j in new_indices
                    ]
            elif new_index == 'copy':
                new_vectors = [self.get_vector() for j in new_indices]
            else:
                # Should be None. Don't add new vectors, raise an error
                # instead. May be useful for testing.
                raise IndexError(f'list index {i} out of range for Phi.')
            # Convert to 2-dim vectors and concatenate after existing vectors
            new_rows = [v[np.newaxis, ...] for v in new_vectors]
            self._array = np.concatenate([self._array] + new_rows)

        self._index = i
        self._update_vector_mask()

    # Provide dict-like interface into Phi._dict
    def __getitem__(self, key):
        return self._dict[key]
    
    def get(self, key, default=None):
        return self._dict.get(key, default)

    def get_values(self, *keys):
        return [self._dict[k].values for k in keys]

    def update(self, *dct, ignore_bounds=False, **kwargs):
        """Update Parameter values (not the Parameters themselves)."""
        if dct != ():
            _dct = dct[0]
        else:
            _dct = kwargs
        for k, v in _dct.items():
            self._dict[k].update(v, ignore_bounds=ignore_bounds)

    def __setitem__(self, key, val):
        """Update Parameter value (not the Parameters itself)."""
        self._dict[key].update(val)

    def __iter__(self):
        return iter(self._dict.values())

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def __str__(self):
        footer = f"Index: {self._index}\n" + "-"*16 + "\n"
        string = ""
        for i, p in enumerate(self._dict.values()):
            if i != 0:
                # Add blank line between parameters if more than one
                string += "\n"
            string += str(p)
            string += footer
        string += "\n"

        return string

    def __repr__(self):
        dash_break = "-"*16 + "\n"
        string = ""
        for i, p in enumerate(self._dict.values()):
            if i != 0:
                string += dash_break
            string += p.__repr__()
        return string


    # Add compatibility for saving to .json
    def to_json(self):
        """Encode Phi object as json. See `nems.tools.json`."""
        p = list(self._dict.values())
        frozen_parameters = [k for k, v in self._dict.items() if v.is_frozen]
        data = {
            'args': p,
            'attributes': {
                '_array': self._array,
                '_index': self._index
            },
            'frozen_parameters': frozen_parameters
        }
        return data

    @classmethod
    def from_json(cls, json):
        """Decode Phi object from json. See `nems.tools.json`."""
        phi = cls(*json['args'])
        for k, v in json['attributes'].items():
            setattr(phi, k, v)
        frozen = json['frozen_parameters']
        if len(frozen) > 0:
            phi.freeze_parameters(*json['frozen_parameters'])
        return phi

    def from_dict(dct, default_bounds='infinite'):
        """Construct Phi from a specially formatted dictionary.
        
        Parameters
        ----------
        dct : dict
            Must contain three nested dictionaries at keys 'values', 'prior',
            and 'bounds'. Each dictionary must have the same keys, which will
            be the parameter names. Values within each dictionary will be used
            as arguments for `initial_value`, `prior`, and `bounds`,
            respectively.
        default_bounds : string, default='infinite'
            Determines behavior when `bounds=None`.
            If `'infinite'`  : set bounds to (-np.inf, np.inf)
            If `'percentile'`: set bounds to tails of `Parameter.prior`.
                (prior.percentile(0.0001), prior.percentile(0.9999))

        See also
        --------
        Parameter.__init__
        
        """
        parameters = []
        for name, value in dct['values'].items():
            value = np.array(value)
            prior = dct['prior'][name]
            bounds = dct['bounds'][name]
            p = Parameter(name, shape=value.shape, prior=prior, bounds=bounds,
                          default_bounds=default_bounds, initial_value=value)
            parameters.append(p)
        phi = Phi(*parameters)
        return phi

    def modified_copy(self, keys_to_keep, parameters_to_add):
        """TODO."""
        #       ref `keys_to_keep` to store Parameter objects,
        #       combine with parameters_to_add,
        #       build new phi,
        #       overwrite part of new array with copy of old array
        #       copy old index
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, Phi):
            try:
                arrays_match = np.allclose(self._array, other._array)
            except:
                arrays_match = False
            try:
                masks_match = np.allclose(self._vector_mask, other._vector_mask)
            except:
                masks_match = False
            conditions = [
                arrays_match, masks_match,
                (self._dict == other._dict),
                (self._index == other._index)
            ]
            return all(conditions)
        else:
            return NotImplemented
