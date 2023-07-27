import numpy as np
import types


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

#TODO: Manually split arrays to allow for more optimized generator
def get_jackknife_indices(data, n, batch_size=0, axis=0, full_shuffle=False,
                        full_list=False):
    """Returns a generator that when called will provide 
    a single index set from `n` sets created through `data`..
    

    Parameters
    ----------
    data : numpy.ndarray
        Array to make jackknife replicates of.
    n : int
        Number of jackknife replicates to generate indices for.
    batch_size: int
        Split the data into batches before creating new arrays.
    axis : int
        Axis on which replicate indices will be generated.
    full_shuffle : bool
        If true, shuffle all array entries along `axis`.
    shuffle_jacks : bool
        If true, shuffle the ordering of the replicate indices. This is useful
        if you want the jackknife replicates to be semi-random but still want
        to (mostly) maintain the original structure of the data.
        Redundant if setting `full_shuffle = True`.
    full_list : bool
        If true, changes the output of this generator to be the entire list of
        given indicies. Only to be called once.
        
    Returns
    -------
    jack_indices[x] : list
        Each element in the list is a suitable `indices` argument for
        `get_jackknife`.
    
    """
    def split(arr, n, axis=0):
        '''
        See source numpy.array_split(ary, ...)
        
        A copy of np.array_split as a generator, to return a proportional split at each iteration
        '''
        try:
            arr_len = arr.shape[axis]
        except AttributeError:
            arr_len = len(arr)
        try:
            arr_sections = len(n) + 1
            splits = [0] + list(n) + [arr_len]
        except:
            arr_sections = int(n)
            if arr_sections <= 0:
                raise ValueError('number sections must be larger than 0.') from None
        arr_single, _ = divmod(arr_len, arr_sections)
        arr_section_size = ([0] + _*[arr_single]+(arr_sections-_)*[arr_single])
        splits = np.array(arr_section_size, dtype='int16').cumsum()

        sub_arr = []
        arr_ax = np.swapaxes(arr, axis, 0)
        for i in range(arr_sections):
            start = splits[i]
            end = splits[i+1]
            sub_arr.append(np.swapaxes(arr_ax[start:end], axis, 0))

            yield sub_arr[i]

    data_length = data.shape[axis]
    indices = np.arange(0, data_length)
    if full_shuffle:
        indices = np.random.permutation(indices)

    if full_list:
        full_set = []
        try:
            for index in range(0, n):
                if batch_size > 0:
                    new_data = np.arange(batch_size, data_length, batch_size)
                    jack_set = split(new_data, n, axis=axis)
                else:
                    jack_set = split(indices, n, axis=axis)
                mask = next(jack_set)
                full_set.append(mask)
        except StopIteration:
            pass
        yield full_set
    else:
        for index in range(0, n):
            if batch_size > 0:
                new_data = np.arange(batch_size, data_length, batch_size)
                jack_set = split(new_data, n, axis=axis)
            else:
                jack_set = split(indices, n, axis=axis)
            yield next(jack_set)


def get_jackknife(data, indices, axis=0, pad=False, **pad_kwargs):
    """Get a jackknife replicate by deleting slices at `indices` along `axis`."""
    mask = indices
    if isinstance(mask, types.GeneratorType):
        mask = next(indices)
    # Check if the given indices is a set or generator so we can call next automatically
    if pad:
        jackknife = pad_array(data, indices=mask, **pad_kwargs)
    else:
        jackknife = np.delete(data, obj=mask, axis=axis)

    return jackknife


def get_inverse_jackknife(data, indices, axis=0, pad=False, **pad_kwargs):
    '''Returns the inverse replicate of the given jackknife indices.'''
    inverse_jackknife = None
    inverse_indices = indices

    # Check if the given indices is a set or generator so we can call next automatically
    if isinstance(inverse_indices, types.GeneratorType):
        inverse_indices = next(inverse_indices)

    if pad:
        inverse_jackknife = pad_array(data, indices=inverse_indices, **pad_kwargs)
    else:
        inverse_jackknife = np.take(data, inverse_indices, axis=axis)

    return inverse_jackknife

# TODO: Move this somewhere else? Not a jackknife specific operation
#       Add more pad_type's eg. linear_ramp, reflect, wrap, etc...
def pad_array(array, size=0, axis=0, pad_type='zero', pad_path=None, indices=None):
    '''
    Pads given array using options settings and jackknife indicies

    Parameters
    ----------
    Array: np.array
        A dataset used for model fitting
    Size: int
        The additional amount we wish to pad onto our array
    Axis: int
        Axis used to create list of indices from datasets
    Pad_type: string
        Type of padding method to use
    Pad_path: string
        A string 'start' or 'end' to specifiy where to pad
    Indices: np.array
        An array mask of indices to pad from array

    Returns
    -------
    np.array dataset, padded subset of data

    
    '''
    # Pad based on original array mask given by jackknifes.
    # Otherwise pad to end and start of arrays
    pad_list = {}
    padded_array = array.copy()
    array_length = len(padded_array)
    array_x, array_y = array.shape
    padded_length = array_length + size
    padded_mask = []

    def pad_zero(array, mask, axis=0):
        '''Replaces values at mask location in array with 0'''
        array[mask] = 0
        return array
    

    # TODO: This needs to re-evaluated given multi-dimensional arrays, is it even useful?
    def pad_mean(array, mask, axis=0):
        '''Replaces values at mask location in array with mean array value of given axis'''
        array_mean = np.mean(array, axis=axis)
        array[mask] = array_mean
        return array
    

    def pad_edge(array, mask, **kwargs):
        '''Replaces values at mask location in array with nearby values'''
        for index in mask:
            for_index = array[index+1]
            back_index = array[index-1]
            if for_index is not None:
                array[index] = for_index
            elif back_index is not None:
                array[index] = back_index
            else:
                array[index] = 0
        return array
    

    def pad_random(array, mask, **kwargs):
        '''Replaces values at mask location in array with random array values from min to max'''
        array_max = np.max(array)
        array_min = np.min(array)
        random_array = np.random.uniform(array_min, array_max, len(mask)+1)
        for x, index in enumerate(mask):
            array[index] = random_array[x]
        return array


    if indices is not None:
        padded_mask = indices.copy()

    else:
        # Determines where to pad our data if indicies are not given
        if pad_path is 'start':
            padded_mask = np.arange(0, size)
            padded_array = np.insert(padded_array, obj=padded_mask, values=0, axis=axis)

        elif pad_path is 'end':
            padded_mask = np.arange(array_length, padded_length)
            padded_array = np.append(padded_array, np.zeros((size, array_y)), axis=axis)

        else:
            mask_length = int(size/2)
            start_mask = np.arange(0, mask_length)
            end_mask = np.arange(array_length, padded_length-mask_length)
            padded_mask = np.concatenate((start_mask, end_mask))

            #Expanding our array to allow new padding
            padded_array = np.append(padded_array, np.zeros((mask_length+1, array_y)), axis=axis)
            padded_array = np.insert(padded_array, obj=start_mask, values=0, axis=axis)

    pad_list['zero']   = pad_zero
    pad_list['mean']   = pad_mean
    pad_list['edge']   = pad_edge
    pad_list['random'] = pad_random

    return pad_list[pad_type](padded_array, padded_mask, axis=axis)

# TODO: what other generic split functions would be useful here?

# TODO: Merge functions together ie. lambda, inner def, etc...
def generate_jackknife_data(input, target, samples=5, axis=0, batch_size=0, inverse=False):
    """
    Creates a generator of generators that take a dataset, and # of samples to index to
    return an input or target to be used in fitting. one at a time.

    Parameters
    ----------
    data: np.array
        A dataset used for model fitting
    N: int
        The number of lists we wish to generate from our data
    Axis: int
        Axis used to create list of indices from datasets
    Batch_size:
        Size of batches contained in the dataset, if any
    TODO: Create a list of arguments to adjust data based on things like batches or axis,
            Maybe move this to jackknife 
    Returns
    -------
    np.array dataset, subset of data
    """
    while True:
        if inverse:
            yield (split_jackknife_generator(input, target, samples=samples, axis=axis, batch_size=batch_size, inverse=False), 
                   split_jackknife_generator(input, target, samples=samples, axis=axis, batch_size=batch_size, inverse=True))
        else:
            yield internal_jackknife_generator(input, target, samples=samples, axis=axis, batch_size=batch_size)

# Another internal function for our generator. Used to allow creation of tuple of gens for validation/fit data
def split_jackknife_generator(input, target, samples=5, axis=0, batch_size=0, inverse=False):
    while True:
        if inverse:
            yield internal_jackknife_generator(input, target, samples=samples, axis=axis, batch_size=batch_size, inverse=True)
        yield internal_jackknife_generator(input, target, samples=samples, axis=axis, batch_size=batch_size)


# Testing a generator of generators to allow multiple fits and models to use same base generator
# created by end user. Will be merging with above eventually
def internal_jackknife_generator(input, target, samples=5, axis=0, batch_size=0, inverse=False):
    return_set = None
    dataset = None
    inverse_dataset = None

    mask_gen = get_jackknife_indices(input, samples, axis=axis, batch_size=batch_size)
    for index in range(0, samples):
        mask = next(mask_gen)

        if inverse:
            inverse_input = get_inverse_jackknife(input, mask, axis=axis)
            inverse_dataset = inverse_input
            if target is not None:
                inverse_target = get_inverse_jackknife(target, mask, axis=axis)
                inverse_dataset = (inverse_input, inverse_target)
            return_set = inverse_dataset
        else:
            input_data = get_jackknife(input, mask, axis=axis)
            dataset = input_data
            if target is not None:
                target_data = get_jackknife(target, mask, axis=axis)    
                dataset = (input_data, target_data)
            return_set = dataset

        yield return_set