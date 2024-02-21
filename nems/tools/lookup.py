"""Tools for looking up python objects from strings."""
import importlib
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

def split_to_api_and_fn(mystring):
    '''
    Returns (api, fn_name) given a string that would be used to import
    a function from a package.
    '''
    matches = mystring.split(sep='.')
    api = '.'.join(matches[:-1])
    fn_name = matches[-1]
    return api, fn_name

lookup_table = {}  # TODO: Replace with real memoization/joblib later

def lookup_fn_at(fn_path, ignore_table=False):
    '''
    Private function that returns a function handle found at a
    given module. Basically, a way to import a single function.
    e.g.
        myfn = _lookup_fn_at('nems0.modules.fir.fir_filter')
        myfn(data)
        ...
    '''

    # default is nems0.xforms.<fn_path>
    if not '.' in fn_path:
        fn_path = 'nems0.xforms.' + fn_path

    if (not ignore_table) and (fn_path in lookup_table):
        fn = lookup_table[fn_path]
    else:
        api, fn_name = split_to_api_and_fn(fn_path)
        api_obj = importlib.import_module(api)
        if ignore_table:
            importlib.reload(api_obj)  # force overwrite old imports
        fn = getattr(api_obj, fn_name)
        if not ignore_table:
            lookup_table[fn_path] = fn
    return fn
