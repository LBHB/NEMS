import numpy as np
import types
from nems.visualization.metrics import jackknife_est_error
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
    def __init__(self, input, samples=15,*, axis=0, inverse=False, shuffle=False, state=None, target=None, 
                 input_name=None, state_name=None, output_name=None, target_name=None,
                 prediction_name=None, dtype=None, **kwargs):
        """
        Initializes our jackknife iterator object. By creating a mask using the first available input,
        converting our data to a DataSet object, and setting initial parameters.

        Parameters
        ----------
        input: np.ndarray or dict
            Given input or dict of inputs to process jackknifes of.
        samples: int
            Given # of jackknife samples to iterator through
        axis: int
            Which axis to process our mask
        inverse: boolean
            If True, include inverse mask and data
        shuffle:
            If True, randomly shuffle which indexes are selected
        state: np.ndarray
            State data to be masked alonside inputs
        target: np.ndarray
            Target data to also be masked with our inputs and state data
        input_name | state_name | output_name: string | prediction_name
            Provided names to use for DataSet object if needed
        dtype: type; optional.
            Type value used within DataSet object
            
        """
        if input is dict:
            self.mask_list       = self.create_jackknife_masks(next(iter(input)), samples, axis=axis, shuffle=shuffle, **kwargs)
        else:
            self.mask_list       = self.create_jackknife_masks(input, samples, axis=axis, shuffle=shuffle, **kwargs)
        self.dataset         = DataSet(input, target=target, state=state, input_name=input_name, state_name=state_name, 
                               output_name=output_name, target_name=target_name, prediction_name=prediction_name, dtype=dtype)
        
        self.inverse         = inverse
        self.fit_list        = None
        self.index           = 0
        self.samples         = samples
        self.max_iter        = samples
        self.axis = axis

    def __iter__(self):
        """Initialization of new iteration state"""
        self.reset_iter()
        return self

    def __next__(self):
        """Returns jackknifed data with mask at current index, then iterates index."""
        if self.index >= self.max_iter:
            raise StopIteration
        jackknifed_data = self.get_indexed_jackknife(index=self.index, inverse=self.inverse)
        self.index += 1
        return jackknifed_data
        
    def reset_iter(self):
        """Resets internal index to 0 """
        self.index = 0

    def set_samples(self, new_sample):
        """Sets new sample amount to iterator"""
        self.samples = new_sample

    def create_jackknife_masks(self, data, n, axis=0, shuffle=False):
        """Returns list of masks for jackknifes"""
        masks = np.arange(0, data.shape[axis])
        if shuffle:
            masks = np.random.permutation(masks)
        masks = np.array_split(masks, n)
        return masks

    def plot_estimate_error(self):
        '''Wrapper for nems.visualization.metrics.jackknife_est_error(model_list, ...)'''
        return jackknife_est_error(self.fit_list, self.samples)

    def get_indexed_jackknife(self, index, inverse=None):
        """Returns the jackknife at a given index, conditionally with inverse."""
        jackknife_data = DataSet(self.get_jackknife(self.dataset['input'], self.mask_list[index]),
                                 target=self.get_jackknife(self.dataset['target'], self.mask_list[index]),
                                 state=self.get_jackknife(self.dataset.get('state', None), self.mask_list[index]))
        
        if inverse == 'both':
            jackknife_data = (jackknife_data, DataSet(self.get_inverse_jackknife(self.dataset['input'], self.mask_list[index]),
                                 target=self.get_inverse_jackknife(self.dataset['target'], self.mask_list[index]),
                                 state=self.get_inverse_jackknife(self.dataset.get('state', None), self.mask_list[index])))
        return jackknife_data

    def get_jackknife(self, data, mask, axis=None):
        """
        Get a jackknife replicate by deleting slices at `indices` along `axis`.
        If multiple inputs are given, jackknife will be applied to each.
        """
        if axis is None:
            axis = self.axis
        jackknifed = None
        if data is not None: 
            if data is dict:
                jackknifed = {np.delete(data[key], obj=mask, axis=axis) for key in data.keys()}
            else:
                jackknifed = np.delete(data, obj=mask, axis=axis)
        return jackknifed
    
    def get_inverse_jackknife(self, data, mask, axis=None):
        """Returns the inverse replicate of the given jackknife indices."""
        if axis is None:
            axis = self.axis
        jackknifed_inverse = None
        if data is not None: 
            if data is dict:
                jackknifed_inverse = {np.take(data[key], mask, axis=axis) for key in data.keys()}
            else:
                jackknifed_inverse = np.take(data, mask, axis=axis)
        return jackknifed_inverse
    
    def get_predicted_jackknifes(self, model_set, **kwargs):
        """
        Returns concatenation of a prediction list created through given model(s) and iterator with the existing target, to compare
        NOTE: Must have or provide a fitted model_list via get_fitted_jackknifes
        """
        if not isinstance(model_set, list):
            model_set = [model_set]

        # Need inverse on iterator to get predictions from validation data
        # Saving state of inverse to reset back after predicting, for jk use consistency
        reset_inverse = False
        if not self.inverse:
            self.inverse = 'both'
            reset_inverse = True

        self.reset_iter()
        predicted = np.concatenate([(model.predict(inverse_set['input'], state=inverse_set.get('state', None), **kwargs)) 
                                    for model, (_, inverse_set) in zip(model_set, self)])
        self.reset_iter()
        target = np.concatenate([(inverse_set['target']) for model, (_, inverse_set) in zip(model_set, self)])
        dataset = {'prediction': predicted, 'target': target}

        if reset_inverse:
            self.inverse = False

        return dataset
    
    def get_fitted_jackknifes(self, model, **kwargs):
        """
        Fits jackknife list to a model and returns a list of fitted models for each jackknife set
        """
        if self.inverse == 'both':
            jackknifed = [model.fit(dataset['input'], dataset['target'], state=dataset.get('state', None), **kwargs) for dataset, _ in self]
        else:
            jackknifed = [model.fit(dataset['input'], dataset['target'], state=dataset.get('state', None), **kwargs) for dataset in self]

        self.fit_list = jackknifed
        return jackknifed