"""Cost functions for SciPyBackend."""

import numpy as np


# TODO: Need to have normalization in-place for multi-neuron data
#       (or multi-batch?), otherwise some components will get weighted
#       more heavily. Should be a pre-processing step.

# TODO: I don't love having different argument names for mse vs nmse, but I
#       wanted to make it clear that the ordering doesn't matter for the former
#       (and does for the latter). Similar for other modules in this directory.
def mse(x, y):
    """Compute the mean squared error (MSE) between arrays x and y.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays must have the same shape. Most commonly, these will be a
        model output (prediction) and a recorded response (target).

    Returns
    -------
    mse : float

    Examples
    --------
    >>> prediction = model.predict(data)
    >>> target = data['response']
    >>> error = mse(output, target)

    >>> model.score()

    Notes
    -----
    This implementation is compatible with both numeric (i.e., Numpy)
    and symbolic (i.e., Theano, TensorFlow) computation.

    """
    squared_errors = (x - y)**2
    return np.mean(squared_errors)


def nmse(prediction, target):
    """Compute MSE normalized by the standard deviation of target.

    Because this metric is more computationally expensive than MSE, but is
    otherwise equivalent, we suggest using `mse` for fitting and `nmse` as a
    post-fit performance metric only.

    Parameters
    ----------
    prediction : ndarray
    target : ndarray
        Shape must match prediction. MSE will be divided by the
        standard deviation of this array.

    Returns
    -------
    normalized_error : float

    See also
    --------
    .mse
    
    """
    std_of_target = np.std(target)
    error = np.sqrt(mse(prediction, target))

    if std_of_target == 0:
        # This means the target is a constant. Return raw mse, so that fits will converge
        normalized_error = error
    else:
        normalized_error = error / std_of_target
    return normalized_error


# Add nickname here if desired string name doesn't match
# the name of the function.
cost_nicknames = {}
def get_cost(name):
    if name in cost_nicknames:
        cost = cost_nicknames[name]
    else:
        cost = globals().get(name, None)
    if cost is None:
        raise TypeError(f"Cost function name '{name}' could not be found.")

    return cost
    