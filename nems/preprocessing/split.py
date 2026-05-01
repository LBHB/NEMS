import types
import logging

import numpy as np

from nems.visualization.metrics import jackknife_est_error
from nems.models.dataset import DataSet

log = logging.getLogger(__name__)


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
    Iterator object used to create jackknife subsets of larger datasets.
    Always slices along axis=0 of the dataset. This can be batches or time, depending
    on how the dateset is configured.

    Attributes
    ----------
    inverse : {boolean, str}; optional
        A variable used to determine if inverse masks should be provided.
        False (default), True, or 'both'
        if inverse=='both' iterator will return a tuple of datasets, with (regular, inverse) slices
    mask_list : ndarray;
        The list of indicies used for modifying our larger dataset into subsets
    index : int; 
        The current index of our iterator
    samples : int; default=5
        The total number of samples we used to create masks
    dataset : DataSet object 
        The dataset object that contains our input/target data
    axis : int
        Default 0

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
    def __init__(self, input, samples=5,*, axis=0, inverse=False, shuffle=False, state=None, target=None,
                 input_name=None, state_name=None, output_name=None, target_name=None,
                 prediction_name=None, dtype=None, stratify=None, mode='auto', **kwargs):
        """
        Parameters
        ----------
        mode : {'auto', 'contiguous', 'interleaved'}
            How each (sub-)group of indices is split into folds.
            'contiguous'  : adjacent indices stay in the same fold (np.array_split).
                Right for continuous time series where neighbouring bins are
                strongly correlated — splitting them apart would leak training
                info into validation.
            'interleaved' : round-robin assignment (idx[i::n]) so each fold is
                spread across the whole range. Right for epoch-batched inputs
                where adjacent epochs are only weakly coupled but session-level
                drift would otherwise load up a single fold.
            'auto' (default): picks 'interleaved' when the input has >=3 dims
                along the splitting axis (i.e. epoch-batched: (n_epochs, T, C)),
                otherwise 'contiguous'. Override explicitly if you know better.
        stratify : array-like, optional
            Per-index label (e.g. 'active'/'passive'). When provided, each
            label group is split into `samples` folds independently using the
            chosen mode and then concatenated so every fold has proportional
            representation from each group. Composes with any `mode`.
        """
        self.inverse         = inverse
        self.axis = axis
        self.fit_list        = None
        self.index           = 0
        self.shuffle = shuffle
        self.samples         = samples
        self.max_iter        = samples
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Stratified jackknife support — when `stratify` labels are provided (one per index along `axis`), each label group is split into `samples` folds independently and concatenated. Guarantees every fold has proportional representation from each group (e.g. active/passive). Added `mode` to pick contiguous vs interleaved splits (auto-detect from input dims) for robustness to session drift in epoch-batched data. | date: 2026-04-19]
        self.stratify = np.asarray(stratify) if stratify is not None else None
        if mode not in ('auto', 'contiguous', 'interleaved'):
            raise ValueError(f"mode must be 'auto', 'contiguous', or 'interleaved'; got {mode!r}")
        self.mode = mode
        # [AGENT EDIT END]
        self.dataset         = DataSet(input, target=target, state=state, input_name=input_name, state_name=state_name,
                               output_name=output_name, target_name=target_name, prediction_name=prediction_name, dtype=dtype)
        k = list(self.dataset.inputs.keys())
        self.create_jackknife_masks(self.dataset.inputs[k[0]])

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        """Returns jackknifed data with mask at current index, then iterates index."""
        if self.index >= self.max_iter:
            raise StopIteration
        jackknifed_data = self.get_indexed_jackknife(index=self.index, inverse=self.inverse)
        #log.info(f"Extracting JK {self.index+1}/{self.max_iter}")
        self.index += 1

        return jackknifed_data
        
    def reset_iter(self):
        """Resets internal index to 0 """
        self.index = 0

    def set_samples(self, new_sample):
        """Sets new sample amount to iterator"""
        self.samples = new_sample

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Resolve 'auto' mode — interleaved for epoch-batched input (>=3D at split axis), else contiguous. Split helper shared by stratified and non-stratified paths. | date: 2026-04-19]
    def _resolve_mode(self):
        if self.mode != 'auto':
            return self.mode
        k = list(self.dataset.inputs.keys())[0]
        return 'interleaved' if self.dataset.inputs[k].ndim >= 3 else 'contiguous'

    def _split_indices(self, idx, n, mode):
        """Split `idx` (1D array of indices) into `n` folds using the given mode."""
        if mode == 'interleaved':
            return [idx[i::n] for i in range(n)]
        return np.array_split(idx, n)
    # [AGENT EDIT END]

    def create_jackknife_masks(self, data):
        """Generates list of masks for jackknifes, saves in self.mask_list

        param: data: np.array
        """
        self.max_index = data.shape[self.axis]
        n = self.samples
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Stratified + mode-aware split. When self.stratify labels are supplied, each label group is split into n folds independently using the resolved mode and concatenated (balanced active/passive). Otherwise splits the full index range with the resolved mode (interleaved for epoch-batched input by default). | date: 2026-04-19]
        mode = self._resolve_mode()
        if self.stratify is not None:
            if len(self.stratify) != self.max_index:
                raise ValueError(
                    f"stratify length {len(self.stratify)} does not match "
                    f"data length {self.max_index} along axis {self.axis}")
            fold_parts = [[] for _ in range(n)]
            for label in np.unique(self.stratify):
                idx = np.where(self.stratify == label)[0]
                if self.shuffle:
                    idx = np.random.permutation(idx)
                for i, chunk in enumerate(self._split_indices(idx, n, mode)):
                    fold_parts[i].append(chunk)
            self.mask_list = [np.concatenate(parts) if len(parts) else np.array([], dtype=int)
                              for parts in fold_parts]
            return self.mask_list
        masks = np.arange(0, self.max_index)
        if self.shuffle:
            masks = np.random.permutation(masks)
        self.mask_list = self._split_indices(masks, n, mode)
        return self.mask_list
        # [AGENT EDIT END]

    def plot_estimate_error(self):
        '''Wrapper for nems.visualization.metrics.jackknife_est_error(model_list, ...)'''
        return jackknife_est_error(self.fit_list, self.samples)

    def get_indexed_jackknife(self, index, inverse=False):
        """Returns the jackknife at a given index, conditionally with inverse."""

        def fn(x):
            return np.delete(x, obj=self.mask_list[index], axis=self.axis)

        def fn2(x):
            return np.take(x, self.mask_list[index], axis=self.axis)

        if inverse==False:
            jackknife_data = self.dataset.apply(fn, allow_copies=True)

        elif inverse==True:
            jackknife_data = self.dataset.apply(fn2, allow_copies=True)

        elif inverse == 'both':
            jackknife_data = (self.dataset.apply(fn, allow_copies=True),
                              self.dataset.apply(fn2, allow_copies=True))
        else:
            raise ValueError(f"Unknown value inverse={inverse}")
        return jackknife_data

    def get_jackknife(self, data, mask, axis=0):
        """Get a jackknife replicate by deleting slices at `indices` along `axis`."""
        return np.delete(data, obj=mask, axis=axis) if data is not None else None
    
    def get_inverse_jackknife(self, data, mask, axis=0):
        """Returns the inverse replicate of the given jackknife indices."""
        return np.take(data, mask, axis=axis) if data is not None else None
    
    def get_predicted_jackknifes(self, model_set, pad_bins=0, **kwargs):
        """Returns concatenation of a prediction list created through given model(s) and iterator with the existing target, to compare.

        pad_bins: number of leading bins per window that are FIR warm-up (input
        was prepadded by lite_input_dict via epoch_prepad). These bins are
        stripped from each per-window prediction and target before stitching,
        so the flat output aligns with the un-padded WINDOW epoch mask.
        """
        if not isinstance(model_set, list):
            model_set = [model_set]
        # Need inverse on iterator to get predictions from validation data
        save_inverse = self.inverse
        self.inverse = True

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: For windowed (3D) data, predict each window independently instead of flattening. Flattening caused FIR to span window boundaries, contaminating the first ~firlen bins of every window with data from an unrelated prior window — especially bad for interleaved/stratified folds where adjacent windows in the flattened stream are temporally disjoint. Per-window prediction mirrors fit-time batching. The pad_bins kwarg lets the caller strip leading FIR warm-up bins from each window's prediction so the stitched flat output matches the un-padded WINDOW epoch mask. | date: 2026-04-27]
        k0 = list(self.dataset.inputs.keys())[0]
        is_windowed = self.dataset.inputs[k0].ndim >= 3
        window_size = self.dataset.inputs[k0].shape[1] if is_windowed else None
        if is_windowed and pad_bins:
            if pad_bins >= window_size:
                raise ValueError(
                    f"pad_bins={pad_bins} >= window_size={window_size}; "
                    f"nothing left after stripping warm-up bins")
            window_size_out = window_size - int(pad_bins)
        else:
            window_size_out = window_size

        if is_windowed:
            log.info(f"get_predicted_jackknifes: windowed mode, "
                     f"n_folds={self.samples}, window_size={window_size}, "
                     f"pad_bins={pad_bins}, window_size_out={window_size_out}")

        # Predict each fold. In 3D mode we keep results as a list of (WS_out, C_out)
        # arrays per fold (one entry per window) — the fold-local order matches
        # mask_list[fold] because inverse_set is built via np.take(..., mask).
        preds = []
        for model, inverse_set in zip(model_set, self):
            if is_windowed:
                n_win = inverse_set.inputs[k0].shape[0]
                win_preds = []
                for w in range(n_win):
                    win_input = {k: v[w] for k, v in inverse_set.inputs.items()}
                    p = model.predict(win_input)
                    if isinstance(p, np.ndarray):
                        p = {'output': p}
                    # Only keep final 'output' — intermediate signals may have
                    # extra dims (e.g. WeightChannels) that don't stitch cleanly.
                    out = p['output']
                    if pad_bins:
                        out = out[pad_bins:]
                    win_preds.append(out)
                pred = {'output': win_preds}  # list of (WS_out, C_out), fold order
            else:
                pred = model.predict(inverse_set)
                if isinstance(pred, np.ndarray):
                    pred = {'output': pred}
            preds.append(pred)

        # Collect targets in the same layout as preds.
        for i, inverse_set in enumerate(self):
            t = inverse_set.targets['target']
            if is_windowed:
                # (n_win, WS, C) → list of (WS_out, C) in fold order, stripped
                preds[i]['target'] = [
                    (t[j][pad_bins:] if pad_bins else t[j])
                    for j in range(t.shape[0])
                ]
            else:
                preds[i]['target'] = t

        # Stitch: in 3D mode, allocate a (n_windows, WS_out, C_out) buffer, place
        # each predicted window at its global window index, then flatten to
        # (n_windows*WS_out, C_out) at the end. This keeps the window→global-index
        # mapping explicit instead of relying on mask-order alignment.
        dataset = {}
        for k in list(preds[0].keys()):
            if is_windowed:
                sample = preds[0][k][0]  # (WS_out, C_out)
                out_ch = sample.shape[-1]
                buf = np.zeros((self.max_index, window_size_out, out_ch),
                               dtype=sample.dtype)
                for p, mask in zip(preds, self.mask_list):
                    for j, w in enumerate(mask):
                        buf[w] = p[k][j]
                dataset[k] = buf.reshape(-1, out_ch)
            else:
                s = list(preds[0][k].shape)
                s[self.axis] = self.max_index
                dataset[k] = np.zeros(s)
                for p, mask in zip(preds, self.mask_list):
                    if self.axis == 0:
                        dataset[k][mask] = p[k]
                    elif self.axis == 1:
                        dataset[k][:, mask] = p[k]
        # [AGENT EDIT END]

        self.inverse = save_inverse

        return dataset
    
    def get_fitted_jackknifes(self, model, **kwargs):
        """Returns a list of fitted models with the given model base."""

        if type(model) is list:
            mlist = model
        else:
            mlist = []
            for i in range(self.samples):
                n = model.name + f"/JK{i+1}"
                mlist.append(model.copy(name=n))
        fit_list = []
        for i, (m, dataset) in enumerate(zip(mlist, self)):
            fit_list.append(m.fit(dataset.inputs, dataset.targets, **kwargs))

        self.fit_list = fit_list

        return fit_list

    
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
        if pad_path == 'start':
            padded_mask = np.arange(0, size)
            padded_array = np.insert(padded_array, obj=padded_mask, values=0, axis=axis)

        elif pad_path == 'end':
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