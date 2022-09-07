import itertools

import numpy as np


# TODO: this doesn't actually support arbitrary arrays, only shape (N,)
def correlation(x, y):
    """Compute Pearson correlation coefficient for arrays x and y.

    Note that this is just a wrapper for `np.corrcoef`.
    TODO: this doesn't actually support arbitrary arrays, only shape (N,)

    Parameters
    ----------
    x, y : np.ndarray
        Arrays must have the same shape. Most commonly, these will be a
        model output (prediction) and a recorded response (target).

    Returns
    -------
    float
    
    """
    return np.corrcoef(x, y)[0,1]


# TODO: fill in examples for single trial split
def noise_corrected_r(prediction, single_trial_responses, n_pairs=None,
                      trial_axis=1, channel_axis=None):
    """Compute noise-corrected correlation coefficient for a model prediction.

    Noise is determined by measuring single-trial correlations in the recorded
    response. Based on method in Hsu and Theusnissen (2004), Network.

    Parameters
    ----------
    prediction : np.ndarray
        Typically the output of `Model.predict`.
    single_trial_responses : list of np.ndarray
        Recorded neural response, split into a list of single trial responses.
        Users are responsible for determining how to perform this split for
        their data, but see TODO for examples.
    n_pairs : int or None; optional.
        Number of random single trial pairs to test. If None, test all pairs.
    trial_axis : int; default=0.
        Axis to concatenate for arrays in `single_trial_responses`.
        For example, if `trial_axis=1` and
        `single_trial_responses = [array(shape=(50, 1)), array(shape=(50, 1))]`
        then `all_trials = array(50, 2)`.
    channel_axis : int or None; optional
        Axis for multiple responses. For example,
        if `single_trial_responses = array(shape=(1000, 10, 50))` representing
        responses of 50 neurons to 10 trials of 1000 time units each, then use
        `channel_axis = 2` to compute this metric for each of the 50 neurons.
        Otherwise if `channel_axis is None`, assume array is 2-dimensional
        (i.e. only trial x time or time x trial)

    Returns
    -------
    corrected_rs : list of float
        One value per channel.
        
    """
    trials = np.concatenate(single_trial_responses, axis=trial_axis)

    if channel_axis is not None:
        n_channels = trials.shape[channel_axis]
    else:
        n_channels = 1
        trials = np.expand_dims(trials, axis=-1)
        channel_axis = -1
    
    corrected_rs = []
    for i in range(n_channels):
        single_channel = np.take(trials, indices=[i], axis=channel_axis)
        trial_pair_r = _paired_single_trial_r(trials, n_pairs=n_pairs,
                                            trial_axis=trial_axis)
        trial_pred_r = _single_trial_r(trials, trial_axis=trial_axis)
        corrected_r= np.mean(trial_pred_r)/np.sqrt(trial_pair_r)
        corrected_rs.append(corrected_r)
    
    return corrected_rs
    
def _paired_single_trial_r(trials, n_pairs=1000, trial_axis=0, limit=0.01):
    """Compute mean correlation of pairs of single trials.

    Internal for `noise_corrected_r`.

    Parameters
    ----------
    trials : ndarray
        Single trial responses concatenated along `trial_axis`.
    n_pairs : int or None; optional.
    trial_axis : int; default=0.
    limit : float
        Minimum value to return, to prevent `noise_corrected_r` from
        blowing up.

    Returns
    -------
    mean_pairwise_correlation : float

    """

    n_repetitions = trials.shape[trial_axis]
    # Get all pairs of indices into trial axis,
    # then randomly shuffle the pairs
    pairs = itertools.permutations(range(n_repetitions), 2)[:n_pairs]
    if pairs is None:
        # Only 1 repetition
        return 0  # TODO: why 0?
    np.random.shuffle(pairs)

    # Compute correlation of each pair of responses,
    pairwise_correlations = []
    for i, j in pairs:
        rep1 = np.take(trials, indices=[i], axis=trial_axis)
        rep2 = np.take(trials, indices=[j], axis=trial_axis)
        pairwise_correlations.append(np.corrcoef(rep1, rep2)[0, 1])
                
    # Hard limit on single-trial correlation to prevent explosion
    # TODO: better logic for this
    mean_pairwise_correlation = np.mean(pairwise_correlations)
    if mean_pairwise_correlation < limit:
        mean_pairwise_correlation = limit

    return mean_pairwise_correlation

def _single_trial_r(prediction, trials, trial_axis=0):
    """Compute mean correlation between prediction and single trial responses.
    
    Internal for `noise_corrected_r`.

    Parameters
    ----------
    prediction : ndarray
    trials : ndarray
        Single trial responses concatenated along `trial_axis`.
    trial_axis : int; default=0.

    Returns
    -------
    mean_trial_correlation : float
    
    """

    n_repetitions = trials.shape[trial_axis]
    trial_correlations = []
    for i in range(n_repetitions):
        single_trial = np.take(trials, indices=[i], axis=trial_axis)
        trial_correlations.append(np.corrcoef(prediction, single_trial)[0, 1])

    return np.mean(trial_correlations)
