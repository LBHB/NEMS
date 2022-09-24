"""Miscellaneous utilities for manipulating NumPy ndarrays."""

import numpy as np
from numba import njit


# Based on solution by:
# user48956
# https://stackoverflow.com/questions/911871/detect-if-a-numpy-array-contains-at-least-one-non-numeric-value
@njit(nogil=True)
def one_or_more_nan(array):
    """True if an array contains at least one NaN value.
    
    Much faster than native numpy solutions if a NaN occurs 'early', and
    not much slower if no NaNs are present.
    
    """
    for value in array.flat:
        if np.isnan(value): return True
    else:
        return False


@njit(nogil=True)
def one_or_more_negative(array):
    """True if an array contains at least one negative number."""
    for value in array.flat:
        if value < 0: return True
    else:
        return False


def broadcast_axis_shape(array1, array2, axis=0):
    """Get shape where `array1.shape[axis] == array2.shape[axis]`.
    
    Parameters
    ----------
    array1 : np.ndarray
        Array to be broadcast.
    array2 : np.ndarray
        Array to broadcast to.
    axis : int; default=0.
        Axis of array to broadcast.
    
    Returns
    -------
    shape : tuple
        Broadcasted shape for `array1`.

    """
    shape = list(array1.shape)
    shape[axis] = array2.shape[axis]
    return tuple(shape)


def broadcast_axes(array1, array2, axis=0):
    """Broadcast `array1` to `array2.shape`, but only along specified `axis`.
    
    Parameters
    ----------
    array1 : np.ndarray
        Array to be broadcast.
    array2 : np.ndarray
        Array to broadcast to.
    axis : int; default=0.
        Axis of array to broadcast.

    Returns
    -------
    new_array : np.ndarray
        Broadcasted version of `array1`.

    """
    broadcasted_shape = broadcast_axis_shape(array1, array2, axis=axis)
    if broadcasted_shape[axis] == array1.shape[axis]:
        # Don't waste time broadcasting if the shape is the same.
        new_array = array1
    else:
        new_array = np.broadcast_to(array1, broadcasted_shape)

    return new_array


def broadcast_dicts(d1, d2, axis=0, debug_memory=False):
    """Broadcast axis length of all arrays in one dict against another dict.

    Parameters
    ----------
    d1 : dict of np.ndarray.
        Dictionary containing arrays to be broadcast.
    d2 : dict of np.ndarray.
        Dictionary containing arrays to broadcast to.
    axis : int; default=0.
        Axis to broadcast.
    debug_memory : bool; default=False.
        If True, raise AssertionError if broadcasted arrays do not share memory
        with originals.

    Returns
    -------
    dict of np.ndarray.
    
    """

    new_d = {}
    for k, v in d1.items():
        if v.shape[axis] != 1:
            # Can't broadcast
            new_d[k] = v
        else:
            for v2 in d2.values():
                if v2.shape[axis] > 1:
                    # Broadcasting is possible.
                    new_v = broadcast_axes(v, v2, axis=axis)
                    if debug_memory:
                        assert np.shares_memory(new_v, v)
                    new_d[k] = new_v
                    # Shape on axis no longer 1, can only broadcast once.
                    break
            else:
                if k not in new_d:
                    # No compatible arrays for broadcasting.
                    new_d[k] = v
    
    if len(new_d) == 0:
        # There was nothing to broadcast
        new_d = d1.copy()

    return new_d


def concatenate_dicts(*dicts, axis=0, newaxis=False):
    """Concatenate key-matched arrays in `dicts` along `axis`.
    
    Parameters
    ----------
    dicts : N-tuple of dict of np.ndarray.
        Each dict should have the format `{k: arrary}`. All dicts must contain
        the same keys. All arrays at the same key must have the same shape on
        all dimensions except `axis`.
    newaxis : bool; default=False.
        If True, use `np.stack` instead of `np.concatenate`, which will add
        a new axis to concatenate on at the indicated position. All dimensions
        of arrays at the same key must have the same shape, without exception.

    Returns
    -------
    dict of ndarray, in the format of `{k: concatenated_array}`.

    Notes
    -----
    This function will output array copies, even if the arrays in `dicts` are
    views. So far as I can tell, this is an unavoidable limitation from NumPy
    due to the fact that views can be non-continguous in memory, while the
    concatenated arrays must be contiguous.
    
    """

    listed = {k: [d[k] for d in dicts] for k in dicts[0].keys()}
    if newaxis:
        fn = np.stack
    else:
        fn = np.concatenate
    concatenated = {k: fn(v, axis=axis) for k, v in listed.items()}

    return concatenated


def apply_to_dict(fn, d, *args, allow_copies=True, **kwargs):
        """Maps {k: v} -> {k: fn(v, *args, **kwargs)} for all k, v in dict `d`.
        
        Parameters
        ----------
        fn : callable
            Must accept a single ndarray as its first positional argument.
        d : dict of ndarrays.
            Dictionary containing arrays that `fn` will be applied to.
        args : N-tuple
            Additional positional arguments for `fn`.
        allow_copies : bool; default=True.
            If False, raise AssertionError if `fn(v, *args, **kwargs)` returns
            an array that does not share memory with `v`. Useful for debugging
            if you're uncertain whether `fn` will return copies or views.
        kwargs : dict
            Additional keyword arguments for `fn`.

        Returns
        -------
        dict of ndarray
            Same keys as `d`, with values replaced by output of `fn`.

        Examples
        --------
        TODO
        
        """
        new_d = {}
        for k, v in d.items():
            new_v = fn(v, *args, **kwargs)
            if not allow_copies:
                assert np.shares_memory(new_v, v)
            new_d[k] = new_v
        return new_d
