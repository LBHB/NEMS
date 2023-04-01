import numpy as np


def phi_to_vector(phi):
    '''
    Convert a list of dictionaries where the values are scalars or array-like
    to a single vector.

    This is a helper function for fitters that use scipy.optimize. The scipy
    optimizers require phi to be a single vector; however, it's more intuitive
    for us to return a list of dictionaries (where each dictionary in the list
    contains the values to be fitted.

    >>> phi = [{'baseline': 0, 'coefs': [32, 41]}, {}, {'a': 1, 'b': 15.1}]
    >>> phi_to_vector(phi)
    [0, 32, 41, 1, 15.1]

    >>> phi = [{'coefs': [[1, 2], [3, 4], [5, 6]]}, {'a': 32}]
    >>> phi_to_vector(phi)
    [1, 2, 3, 4, 5, 6, 32]
    '''
    vector = []
    for p in phi:
        if p is None:
            continue
        for k in sorted(p.keys()):
            value = p[k]
            if np.isscalar(value):
                vector.append(value)
            else:
                flattened_value = np.asanyarray(value).ravel()
                vector.extend(flattened_value)
    return vector


def vector_to_phi(vector, phi_template):
    '''
    Convert vector back to a list of dictionaries given a template for phi

    Example
    -------
    >>> phi_template = [{'baseline': 0, 'coefs': [0, 0]}, {}, {'a': 0, 'b': 0}]
    >>> vector = [0, 32, 41, 1, 15.1]
    >>> vector_to_phi(vector, phi_template)
    [{'baseline': 0, 'coefs': array([32, 41])}, {}, {'a': 1, 'b': 15.1}]
    '''
    # TODO: move this to a unit test instead? Or, find a way to fix the doctest
    # so it passes. Doctest is a bit picky about the formatting, but the correct
    # formatting is difficult to read in a docstring!
    offset = 0
    phi = []
    for p_template in phi_template:
        p = {}
        if p_template is None:
            phi.append(p)
            continue

        for k in sorted(p_template.keys()):
            value_template = p_template[k]
            if np.isscalar(value_template):
                value = vector[offset]
                offset += 1
            else:
                value_template = np.asarray(value_template)
                size = value_template.size
                value = np.asarray(vector[offset:offset+size])
                value.shape = value_template.shape
                offset += size
            p[k] = value
        phi.append(p)
    return phi


def initialize_phi(priors, method='mean'):
    '''
    Create an initial set of values for phi given priors

    Parameters
    ----------
    priors : list of dictionaries
        Priors returned by `model.get_priors`
    method : {'mean', 'sample', float}
        Method for calculating starting value of each coefficient:
        'mean'
            Sets the starting value to the mean (expected) value of the
            distribution.
        'sample'
            Sets the starting value to a random value drawn from the
            distribution.
        float
            Sets the starting value to the given percentile (specified as a
            fraction in the range [0, 1]). Use 0.5 to initialize phi to the
            median.

    Returns
    -------
    phi : list of dictionaries
        Initial values for phi

    Example
    -------
    # First, set up some priors to play with
    >>> from nems0.distributions.api import Normal, Beta
    >>> mu = Normal(0, 0.05)
    >>> scale = Beta([1, 2], [2, 1])
    >>> priors = [{'mu': mu}, {'scale': scale}]

    # Initialize to the mean value
    >>> initialize_phi(priors, 'mean')
    [{'mu': 0.0}, {'scale': array([ 0.33333333,  0.66666667])}]
    '''
    # Generate initial values of phi for fitter
    if method == 'mean':
        return [{n: p.mean() for n, p in mp.items()} for mp in priors]
    elif method == 'sample':
        return [{n: p.sample() for n, p in mp.items()} for mp in priors]
    elif isinstance(method, float):
        return phi_percentile(priors, method)


def phi_percentile(priors, percentile):
    '''
    Create an initial set of values for phi given priors

    Parameters
    ----------
    priors : list of dictionaries
        Priors returned by `model.get_priors`
    percentile : float
        Percentile to calculate for each prior

    Returns
    -------
    percentiles : list of dictionaries
        Percentile for each prior

    Example
    -------
    >>> from nems0.distributions.api import Normal, Beta
    >>> mu = Normal(0, 0.05)
    >>> scale = Beta([1, 2], [2, 1])
    >>> priors = [{'mu': mu}, {'scale': scale}]

    # Get absolute lower bound of distribution
    >>> phi_percentile(priors, 0)
    [{'mu': -inf}, {'scale': array([ 0.,  0.])}]

    # Get absolute upper bound of distribution
    >>> phi_percentile(priors, 1)
    [{'mu': inf}, {'scale': array([ 1.,  1.])}]

    # Get median value
    >>> phi_percentile(priors, 0.5)
    [{'mu': 0.0}, {'scale': array([ 0.29289322,  0.70710678])}]
    '''
    return [{n: p.percentile(percentile) for n, p in m.items()} for m in priors]
