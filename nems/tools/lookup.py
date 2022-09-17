"""Tools for looking up python objects from strings."""
from functools import partial


class FindCallable:

    def __init__(self, _dict, header='Function', ignore_case=True):
        """Matches a name to a callable object within one module or package.
        
        Case-insensitive.

        Parameters
        ----------
        _dict : dict.
            Keys are names of callables, values are the callables.
        header : str; default='Function'.
            Leading string for error messages. By default, messages start with
            "Function name '{name}' ..."
        ignore_case : bool; default=True.
            If False, match case of dictionary keys when looking up names.
        
        """
        self._dict = _dict
        if ignore_case:
            self._lookup = partial(case_insensitive_lookup, return_none=True)
        else:
            self._lookup = lambda d, k: d.get(k, None)

        self.not_found = f"{header} name " + "'{name}' could not be found."
        self.not_callable = f"{header} name " + "'{name}' is not callable." 

    def __call__(self, name):
        """Get a callable corresponding to `name`.
        
        If `name` is a key in `nicknames`, then `nicknames[name]` will be
        returned. Otherwise `globals()[name]` will be returned.

        Parameters
        ----------
        name : str

        Returns
        -------
        callable

        Raises
        ------
        TypeError, if object to be returned is either `None` or not callable.
        
        """
        fn = self._lookup(self._dict, name)
        if fn is None:
            raise TypeError(self.not_found.format(name=name))
        if not callable(fn):
            raise TypeError(self.not_callable.format(name=name))

        return fn

def case_insensitive_lookup(_dict, key, return_none=False):
    """Look for `key` in `_dict` but ignore case.
    
    Parameters
    ----------
    _dict : dict.
        Dictionary with string keys.
    key : str.
    return_none : bool; default=False.
        If True, and `key` is not found in `_dict`, return None instead of
        raising a KeyError.

    Returns
    -------
    _dict[key] or None
        See `return_none` parameter.

    Raises
    ------
    KeyError
        If `key` is not in `_dict` and `return_none is False`.
    
    """
    keys = list(_dict.keys())
    lowercase_keys = [k.lower() for k in keys]
    unique_lowercase = list(set(lowercase_keys))
    if len(unique_lowercase) < len(lowercase_keys):
        # This would return the first matching index, which could cause
        # confusion. Instead, make the user aware of the issue.
        # Example: don't have two functions named `MSE` and `mse`.
        raise ValueError(
            "Dictionary contains keys that only differ by case, cannot "
            "perform a case-insensitive lookup."
            )

    try:
        idx = lowercase_keys.index(key.lower())
        v = _dict[keys[idx]]
    except ValueError:
        # key not in the list
        if return_none:
            v = None
        else:
            raise KeyError('key')

    return v
