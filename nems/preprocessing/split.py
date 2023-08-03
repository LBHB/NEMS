import numpy as np
import types
from nems.models.dataset import DataSet


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

class JackknifeIterator:
    """
    Iterator object used to create jackknife subsets of larger datasets

    Attributes
    ----------
    include_inverse : boolean; optional
        A variable used to determine if inverse masks should be provided
    mask_list : ndarray;
        The list of indicies used for modifying our larger dataset into subsets
    index : int; 
        The current index of our iterator
    samples : int; default=5
        The total number of samples we used to create masks
    dataset : DataSet object 
        The dataset object that contains our input/target data

    Methods
    -------
    __next__(self) :
        Returns the next subset of data using our dataset and created masks
    create_jackknife_mask(data, n, axis=0, ...) : 
        Used a given dataset, n samplesize and axis to create a list of masks
    get_jackknife(data, indicies, axis) :
        Uses data and a single mask to return subset of data
    get_jackknife(data, indicies, axis) :
        Uses data and a single mask to return inverse subset of data
    
    """
    def __init__(self, input, samples=5,*, axis=0, inverse=False, state=None, target=None, 
                 input_name=None, state_name=None, output_name=None, target_name=None,
                 prediction_name=None, dtype=None, **kwargs):
        """
        

        """
        self.inverse = inverse
        self.mask_list       = self.create_jackknife_masks(input, samples, axis=axis, **kwargs)
        self.index           = 0
        self.samples         = samples
        self.dataset         = DataSet(input, target=target, state=state, input_name=input_name, state_name=state_name, 
                               output_name=output_name, target_name=target_name, prediction_name=prediction_name, dtype=dtype)

    def __iter__(self):
        return self

    def __next__(self):
        """ Returns jackknifed data with mask at current index, then iterates index  """
        jackknife_data = DataSet(self.get_jackknife(self.dataset['input'], self.mask_list[self.index]), 
                                 target=self.get_jackknife(self.dataset['target'], self.mask_list[self.index]),
                                 state=self.get_jackknife(self.dataset.get('state', None), self.mask_list[self.index]))
        
        if self.inverse is 'both':
            jackknife_data = (jackknife_data, DataSet(self.get_inverse_jackknife(self.dataset['input'], self.mask_list[self.index]), 
                                 target=self.get_inverse_jackknife(self.dataset['target'], self.mask_list[self.index]),
                                 state=self.get_inverse_jackknife(self.dataset.get('state', None), self.mask_list[self.index])))

        self.index = (self.index + 1)%self.samples
        return jackknife_data
        
    def reset_iter(self):
        """ Resets internal index to 0  """
        self.index = 0

    def create_jackknife_masks(self, data, n, axis=0, full_shuffle=False, full_list=False, **kwargs):
        """ returns list of masks for jackknifes """
        masks = np.arange(0, data.shape[axis])
        if full_shuffle:
            masks = np.random.permutation(masks)
        masks = np.array_split(masks, n)
        return masks

    def get_jackknife(self, data, mask, axis=0, pad=False, **pad_kwargs):
        """Get a jackknife replicate by deleting slices at `indices` along `axis`."""
        return np.delete(data, obj=mask, axis=axis) if data is not None else None
    
    def get_inverse_jackknife(self, data, mask, axis=0, pad=False, **pad_kwargs):
        '''Returns the inverse replicate of the given jackknife indices.'''
        return np.take(data, mask, axis=axis) if data is not None else None
    
# TODO: Do we need this for anything?
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