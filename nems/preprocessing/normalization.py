import warnings

import numpy as np


def minmax(x, floor=1e-6, warn_on_nan=True, inplace=False):
    """Normalize array `x` within each output channel to have min 0 and max 1.
    
    For array `x` with `x.ndim == m` and `x.shape == (d_1, d_2, ..., d_m)`, the
    first `m-1` dimensions of `x` will be flattened. Minimum and maximum are
    then determined separately for each of the `d_m` output channels, and `x`
    is normalized according to: `y = (x - min(x)) / max(x - min)`.

    In addition to the normalized array, the per-channel minimum and maximum
    values are returned so that the above normalization can be reversed using
    `x = (y*max) + min`.

    In addition to adjusting the minimum and maximum values, 'small' values are
    set to 0. This is helpful for preventing models from fitting to unimportant
    variations due to recording noise in silent/null padding, for example.
    This portion of `minmax` is not reversible. However, an un-normalized array
    `z` will still satisfy `np.allclose(x, z, atol=e)` for `e >= floor*max(max)`.
    See `floor` parameter for details on usage.

    Parameters
    ----------
    x : np.ndarray.
        Data to be normalized. Shape can be:
        (T,)  or  (T, ..., N)  or  (S, T, ..., N)
        where T is the number of time bins, N is the number of output channels
        or neurons, and S is the number of samples or trials. Shape (T,) is
        interpreted as (T,1).
    floor : float; default=1e-6.
        Values in the normalized array (i.e. already mapped to min 0, max 1)
        will be set to 0 if they are less than this value. If you do not wish
        to zero out any values, use `floor=0`.
        NOTE: This operation is not reversible.
    warn_on_nan : bool; default=True.
        If True, raises RuntimeWarning if `x` contains any NaN values. This is
        done so that unexpected NaNs are not silently ignored, but intentional
        NaNs can be handled (i.e. missing data or NaN-padded ragged trials).
    inplace : bool; default=False.
        If True, normalized values are assigned to `x` in-place instead of
        returning a copy. This option helps reduce memory usage for large
        arrays. This is *not* equivalent to `x = minmax(x, inplace=False)`,
        which still tracks a separate copy of `x` within the scope of the
        `minmax` function.

    Returns
    -------
    normalized : np.ndarray.
        Same shape as `x` with min 0 and max 1.
    min_per_channel : np.ndarray.
        Shape (N,), where N is the number of output channels.
    max_per_channel : np.ndarray.
        Shape (N,).

    Raises
    ------
    AttributeError, if flattening the leading axes of `x` would result in a
    copy. See docs for `numpy.reshape`.

    See also
    --------
    undo_minmax

    """

    # TODO: This is my attempt at a compromise between raising an error
    #       (e.g. using min instead of nanmin) and doing nothing. We want to be
    #       able to handle NaNs for ragged trials, for example, but in general
    #       I'd prefer not to let NaNs silently pass through when they're not
    #       expected, since that makes figuring out where they originated much
    #       more difficult.
    if warn_on_nan and (np.isnan(x).sum() > 0):
        warnings.warn(
            "Input array for `normalization.minmax` contains NaN values.",
            RuntimeWarning, stacklevel=2
            )

    # For large arrays, `inplace=True` should reduce memory usage by half.
    if not inplace:
        x = x.copy()

    original_shape = x.shape
    if x.ndim == 1:
        # Make sure there's an output dimension.
        x = x[..., np.newaxis]
    new_shape = (x[..., 0].size, x.shape[-1])
    # Use this instead of reshape to raise error on copy (only want one copy or
    # zero for the entire scope of the function).
    x.shape = new_shape

    # Shift minimum to 0
    min_per_channel = np.nanmin(x, axis=0, keepdims=True)
    min_zero = np.subtract(x, min_per_channel, out=x)

    # Scale max to 1
    max_per_channel = np.nanmax(min_zero, axis=0, keepdims=True)
    # This can only happen if the whole channel is 0, set to 1 just to avoid
    # 0-division error.
    max_per_channel[max_per_channel == 0] = 1
    try:
        normalized = np.divide(min_zero, max_per_channel, out=x)
    except np.core._exceptions.UFuncTypeError as e:
        # Print info about float dtype
        print('Numpy division resulted in a casting error, make sure the '
              'dtype of the input to `normalization.minmax` is float-like.')
        raise e

    # Force data to be exactly 0 for "small" values.
    normalized[normalized < floor] = 0
    normalized.shape = original_shape

    return normalized, min_per_channel, max_per_channel


def undo_minmax(y, _min, _max):
    """Inverse of minmax normalization, `undo_minmax(minmax(x)) == x`.
    
    Parameters
    ----------
    y, _min, _max : (np.ndarray, np.ndarray, np.ndarray)
        As returned by `minmax`.

    Notes
    -----
    Exact equality is only true for `floor = 0` in `minmax`. Even then, there
    could still be small differences due to floating point rounding errors.

    See also
    --------
    minmax
    
    """
    z = y*_max + _min
    z.shape = y.shape
    return z


def redo_minmax(x, _min, _max, inplace=False):
    """Apply pre-computed normalization to a another array.
    
    TODO: docs.
    
    TODO: N-D support like `minmax`.
    
    """
    if not inplace:
        x = x.copy()
    shifted = np.subtract(x, _min, out=x)
    normalized = np.divide(shifted, _max, out=x)

    return normalized
