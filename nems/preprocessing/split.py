import numpy as np


def indices_by_fraction(data, fraction=0.9, axis=0):
    """Get indices that split `data` into two parts along one axis.

    Parameters
    ----------
    data : numpy.ndarray
        An array to be split.
    fraction : float in range [0,1]
        The proportion of data that will go into the "left" split. The proportion
        in the "right" split will be `1 - fraction`, or as close as possible
        after truncating to the nearest integer bin index.
    axis : int
        The axis on which to perform the split.
    

    Returns
    -------
        2-tuple of numpy.ndarray
            The indices corresponding to before and after the specified
            proportion of the data, suitable as arguments for `split_at_indices`.
    
    """

    split_idx = int(data.shape[axis] * fraction)
    before_fraction = np.arange(0, split_idx)
    after_fraction = np.arange(split_idx, data.shape[axis])

    return before_fraction, after_fraction


def split_at_indices(data, idx1, idx2, axis=0):
    """Split `data` into two parts along one axis by taking slices."""

    first_subset = np.take(data, idx1, axis=axis)
    second_subset = np.take(data, idx2, axis=axis)

    return first_subset, second_subset


# TODO: should this return a generator instead of list?
#       would use less data if the number of indices gets really large
def get_jackknife_indices(data, n, axis=0, full_shuffle=False,
                          shuffle_jacks=True):
    """Generate indices for selecting `n` jackknife replicates from `data`.

    Parameters
    ----------
    data : numpy.ndarray
        Array to make jackknife replicates of.
    n : int
        Number of jackknife replicates to generate indices for.
    axis : int
        Axis on which replicate indices will be generated.
    full_shuffle : bool
        If true, shuffle all array entries along `axis`.
    shuffle_jacks : bool
        If true, shuffle the ordering of the replicate indices. This is useful
        if you want the jackknife replicates to be semi-random but still want
        to (mostly) maintain the original structure of the data.
        Redundant if setting `full_shuffle = True`.

    Returns
    -------
    jack_indices : list
        Each element in the list is a suitable `indices` argument for
        `get_jackknife`.
    
    """
    arrays = []
    indices = np.arange(0, data.shape[axis])
    if full_shuffle:
        indices = np.random.permutation(indices)
    jack_indices = np.array_split(indices, n)
    if shuffle_jacks:
        jack_indices = np.random.permutation(jack_indices)

    return jack_indices

def get_jackknife(data, indices, axis=0):
    """Get a jackknife replicate by deleting slices at `indices` along `axis`."""
    return np.delete(data, obj=indices, axis=axis)

# TODO: what other generic split functions would be useful here?
