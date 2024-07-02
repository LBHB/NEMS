from functools import wraps
import re
import warnings

import numpy as np
import pandas as pd

import logging
log = logging.getLogger(__name__)


def check_result(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if result.size == 0:
            mesg = 'Result is empty'
            warnings.warn(RuntimeWarning(mesg))
        return result
    return wrapper


def remove_overlap(a):
    '''
    Remove overlapping occurences by taking the first occurence
    '''
    a = a.copy()
    a.sort(axis=0)
    i = 0
    n = len(a)
    trimmed = []
    while i < n:
        lb, ub = a[i]
        i += 1
        trimmed.append((lb, ub))
        while (i < n) and (ub > a[i, 0]):
            i += 1
    return np.array(trimmed)


def merge_epoch(a):
    a = a.copy()
    a.sort(axis=0)
    i = 0
    n = len(a)
    merged = []
    while i < n:
        lb, ub = a[i]
        i += 1
        while (i < n) and (ub >= a[i, 0]):
            ub = a[i, 1]
            i += 1
        merged.append((lb, ub))
    return np.array(merged)


def epoch_union(a, b):
    '''
    Compute the union of the epochs.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.

    Returns
    -------
    union : 2D array of (O x 2)
        The first column is the start time and second column is the end time. O
        is the number of occurances of the union of a and b. Note that O <= M +
        N.

    Example
    -------
    a:       [   ]  [         ]        [ ]
    b:      [   ]       [ ]     []      [    ]
    result: [    ]  [         ] []     [     ]
    '''
    epoch = np.concatenate((a, b), axis=0)
    return merge_epoch(epoch)


## SVD commented out: @check_result
def epoch_difference(a, b):
    '''
    Compute the difference of the epochs. All regions in a which overlap with b
    will be removed.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.

    Returns
    -------
    difference : 2D array of (O x 2)
        The first column is the start time and second column is the end time. O
        is the number of occurances of the difference of a and b.

    Example
    -------
    a:       [   ]  [         ]        [ ]
    b:      [   ]       [ ]     []      [    ]
    result:     []  [  ]  [   ]        []
    '''
    a = a.tolist()
    a.sort(reverse=True)
    b = b.tolist()
    b.sort(reverse=True)

    difference = []
    lb, ub = a.pop()
    lb_b, ub_b = b.pop()

    while True:
        if lb > ub_b:
            #           [ a ]
            #     [ b ]
            # Current epoch in b ends before current epoch in a. Move onto
            # the next epoch in b.
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                difference.append((lb, ub))
                break
        elif ub <= lb_b:
            #   [  a    ]
            #               [ b        ]
            # Current epoch in a ends before current epoch in b. Add bounds
            # and move onto next epoch in a.
            difference.append((lb, ub))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (lb == lb_b) and (ub == ub_b):
            try:
                lb, ub = a.pop()
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub > ub_b):
            #   [  a    ]
            #     [ b ]
            # Current epoch in b is fully contained in the  current epoch
            # from a. Save everything in
            # a up to the beginning of the current epoch of b. However, keep
            # the portion of the current epoch in a
            # that follows the end of the current epoch in b so we can
            # detremine whether there are additional epochs in b that need
            # to be cut out..
            difference.append((lb, lb_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                difference.append((lb, ub))
                break
        elif (lb <= lb_b) and (ub <= ub_b):
            #   [  a    ]
            #     [ b        ]
            # Current epoch in b begins in a, but extends past a.
            difference.append((lb, lb_b))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (ub > lb_b) and (lb <= ub_b):
            #   [  a    ]
            # [       b     ]
            # Current epoch in a is fully contained in b
            lb, ub = a.pop()
        elif (ub > lb_b) and (lb > ub_b):
            #   [  a    ]
            # [ b    ]
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                difference.append((lb, ub))
        else:
            # This should never happen.
            m = 'Unhandled epoch boundary condition. Contact the developers.'
            raise SystemError(m)

    # Add all remaining epochs from a
    difference.extend(a[::-1])
    return np.array(difference)


def epoch_intersection_full(a, b):
    """
    returns all epoch times a that are fully spanned by epoch
    times in b
    """
    a = a.copy().tolist()
    a.sort()
    b = b.copy().tolist()
    b.sort()
    intersection = []
    for lb, ub in a:
        for lb_b, ub_b in b:
            if lb >= lb_b and ub <= ub_b:
                intersection.append([lb, ub])
                break

    result = np.array(intersection)
    return result


## SVD commented out: @check_result
def epoch_intersection(a, b, precision=6):
    '''
    Compute the intersection of the epochs. Only regions in a which overlap with
    b will be kept.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.
    precision : int
        Number of decimal places to use for equality test.

    Returns
    -------
    intersection : 2D array of (O x 2)
        The first column is the start time and second column is the end time. O
        is the number of occurances of the difference of a and b.

    Example
    -------
    a:       [   ]  [         ]        [ ]
    b:      [   ]       [ ]     []      [    ]
    result:  [  ]       [ ]             []
    '''
    # Convert to a list and then sort in reversed order such that pop() walks
    # through the occurences from earliest in time to latest in time.
    a = np.around(a, precision)
    b = np.around(b, precision)
    a = a.tolist()
    a.sort(reverse=True)
    b = b.tolist()
    b.sort(reverse=True)

    intersection = []
    if len(a)==0 or len(b)==0:
        # lists are empty, just exit
        result = np.array([])
        return result

    lb, ub = a.pop()
    lb_b, ub_b = b.pop()
    while True:
        if lb >= ub_b:
            #           [ a ]
            #     [ b ]
            # Current epoch in b ends before current epoch in a. Move onto
            # the next epoch in b.
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif ub <= lb_b:
            #   [  a    ]
            #               [ b        ]
            # Current epoch in a ends before current epoch in b. Add bounds
            # and move onto next epoch in a.
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (lb == lb_b) and (ub == ub_b):
            #   [  a    ]
            #   [  b    ]
            # Current epoch in a matches epoch in b.
            try:
                intersection.append((lb, ub))
                lb, ub = a.pop()
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub >= ub_b):
            #   [  a    ]
            #     [ b ]
            # Current epoch in b is fully contained in the  current epoch
            # from a. Save everything in
            # a up to the beginning of the current epoch of b. However, keep
            # the portion of the current epoch in a
            # that follows the end of the current epoch in b so we can
            # detremine whether there are additional epochs in b that need
            # to be cut out..
            intersection.append((lb_b, ub_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub >= lb_b) and (ub <= ub_b):
            #   [  a    ]
            #     [ b        ]
            # Current epoch in b begins in a, but extends past a.
            intersection.append((lb_b, ub))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (lb > lb_b) and (ub <= ub_b):
            #   [  a    ]
            # [       b     ]
            # Current epoch in a is fully contained in b
            intersection.append((lb, ub))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (lb > lb_b) and (ub > ub_b):
            #   [  a    ]
            # [ b    ]
            intersection.append((lb, ub_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        else:
            # This should never happen.
            m = 'Unhandled epoch boundary condition. Contact the developers.'
            raise SystemError(m)

    result = np.array(intersection)
    return result


def _epoch_contains_mask(a, b):
    '''
    3d array. 1st dimension is index in a. Second dimension is index in b. Third
    dimension is whether start (index 0) or end (index 1) in b falls within the
    corresponding epoch in a.
    '''
    mask = [(b >= lb) & (b <= ub) for lb, ub in a]
    return np.concatenate([m[np.newaxis] for m in mask], axis=0)


def epoch_contains(a, b, mode):
    '''
    Tests whether an occurence of a contains an occurence of b.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.
    mode : {'start', 'end', 'both', 'any'}
        Test to perform.
        - 'start' requires only the start of b to be contained in a
        - 'end' requires only the end of b to be contained in a
        - 'both' requires both start and end in b to be contained in a
        - 'any' is True anywhere b partially or completely overlaps with a

    Returns
    -------
    mask : 1D array of len(a)
        Boolean mask indicating whether the corresponding entry in a meets the
        test criteria.
    '''
    mask = _epoch_contains_mask(a, b)
    if mode == 'start':
        return mask[:, :, 0].any(axis=1)
    elif mode == 'end':
        return mask[:, :, 1].any(axis=1)
    elif mode == 'both':
        return mask.all(axis=2).any(axis=1)
    elif mode == 'any':
        b_in_a = mask.any(axis=2).any(axis=1)
        # This mask will not capture situations where an occurence of a is fully
        # contained in an occurence of b. To test for this, we can flip the
        # epochs and build a new mask to perform this special-case test.
        mask = _epoch_contains_mask(b, a)
        a_in_b = mask.any(axis=2).any(axis=0)
        return b_in_a | a_in_b


def epoch_contained(a, b):
    '''
    Tests whether an occurrence of a is fully contained inside b
    '''
    mask = _epoch_contains_mask(b, a)
    return mask.all(axis=2).any(axis=0)


def adjust_epoch_bounds(a, pre=0, post=0):
    '''

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    pre : scalar
        Value to add to start time of epoch
    post : scalar
        Value to add to end time of epoch

    Example
    -------
    >>> epochs = np.array([[0.5, 1], [2.5, 3]])
    >>> adjust_epoch_bounds(epochs, -0.5)
    [[0, 1],
     [2, 3]]

    >>> adjust_epoch_bounds(epochs, 0.5, 1)
    [[1, 2],
     [3, 4]]
    '''
    return a + np.array([pre, post])


def verify_epoch_integrity(epoch):
    '''
    There are several kinds of pathological epochs:
      1. Epochs with NaN for a start time. (It is OK if end is NaN.)
      2. Epochs where start comes after the end
      3. Epochs which are completely identical to existing triplets
         (i.e. redundant duplicates)
    This function searches for those and throws exceptions about them.
    '''
    # TODO
    raise NotImplementedError


def epoch_names_matching(epochs, regex_str):
    '''
    Returns a list of epoch names that regex match the regex_str.
    '''
    r = re.compile(regex_str)
    names = epochs['name'].tolist()
    matches = filter(r.match, names)

    # convert to list
    matches = [name for name in matches]
    matches = list(set(matches))  # unique values
    matches.sort()

    return matches


def epoch_occurrences(epochs, regex=None):
    '''
    Returns a dataframe of the number of occurrences of each epoch. Optionally,
    provide regex to match only certain epoch names.
    '''
    epoch_counts = epochs.name.value_counts()
    if regex:
        epoch_counts = epoch_counts.filter(regex=regex, axis='rows')
    return epoch_counts


def group_epochs_by_occurrence_counts(epochs, regex=None):
    '''
    Returns a dictionary mapping the number of occurrences to a list of epoch
    names. This is essentially the inverse mapping of epoch_occurrences().
    '''
    d = {}
    # Build a dict of n_occurrences -> [epoch_name1, epoch_name2, etc]
    for row in epoch_occurrences(epochs, regex).items():
        name, count = row
        if count in d:
            d[count].append(name)
        else:
            d[count] = [name]
    return d


def find_common_epochs(epochs, epoch_name, d=12):
    '''
    Finds all epochs contained by `epoch_name` that are common to all
    occurences of `epoch_name`. An epoch is considered "common" to all
    occurences if the name matches and the start and end times, relative to the
    start `epoch_name`, are the same to the number of decimal places indicated.

    Parameters
    ----------
    epochs : dataframe with 'name', 'start', 'end'
        Epochs to filter through
    epoch_name : str
        Name of epoch to use
    d : int
        Number of decimal places to round start and end to. This is important
        when comparing start and end times of different epochs due to
        floating-point errors.

    Result
    ------
    common_epochs : dataframe with 'name', 'start', 'end'
        Epochs common to all occurances of `epoch_name`. The start and end
        times will reflect the time relative to the onset of the epoch.
    '''
    # First, loop through all occurrence of `epoch_name` and find all the
    # epochs contained within that occurrence. Be sure to adjust the start/end
    # time so that they are relative to the beginning of the occurrence of that
    # epoch.
    epoch_subsets = []
    matches = epochs.query('name == "{}"'.format(epoch_name))
    for lb, ub in matches[['start', 'end']].values:
        m = (epochs['start'] >= lb) & (epochs['end'] <= ub)
        epoch_subset = epochs.loc[m].copy()
        epoch_subset['start'] -= lb
        epoch_subset['end'] -= lb
        epoch_subset = set((n, round(s, d), round(e, d)) for (n, s, e) \
                           in epoch_subset[['name', 'start', 'end']].values)

        epoch_subsets.append(epoch_subset)

    # Now, determine which epochs are common to all occurrences.
    common_epochs = epoch_subsets[0].copy()
    for other_epoch in epoch_subsets[1:]:
        common_epochs.intersection_update(other_epoch)

    new_epochs = pd.DataFrame(list(common_epochs),
                              columns=['name', 'start', 'end'])
    new_epochs.sort_values(['start', 'end'], inplace=True)
    return new_epochs


def group_epochs_by_parent(epochs, epoch_name_regex):
    '''
    Iterate through subgroups of the epoches contained by a parent epoch

    Parameters
    ----------
    epochs : dataframe with 'name', 'start', 'end'
        Epochs to filter through
    epoch_name_regex : str
        Regular expression that will be used to identify parent epochs to
        iterate through.

    Returns
    -------
    Iterator yielding a tuple of (parent epoch name, dataframe containing subset
    of epochs contained by parent epoch).

    Example
    '''

    m = epochs.name.str.match(epoch_name_regex)
    for name, start, end in epochs.loc[m, ['name', 'start', 'end']].values:
        m_lb = epochs['start'] >= start
        m_ub = epochs['end'] <= end
        m = m_lb & m_ub
        yield (name, epochs.loc[m])


def add_epoch(df, regex_a, regex_b, new_name=None, operation='intersection'):
    '''
    Add a new epoch based on an operation of two epoch sets, A and B

    Parameters
    ----------
    df : dataframe
        Epoch dataframe with three columns (name, start, end)
    regex_a : string
        Regular expression to match against for A
    regex_b : string
        Regular expression to match against for B
    new_name : {None, string}
        Name to assign to result of operation. If None, name is the
        concatenation of regex_a and regex_b.
    operation : {'intersection', 'difference', 'contained'}
        Operation to perform. See docstring for epoch_{operation} for details.
    '''
    if new_name is None:
        new_name = '{}_{}'.format(regex_a, regex_b)

    mask_a = df['name'].str.contains(regex_a)
    mask_b = df['name'].str.contains(regex_b)
    a = df.loc[mask_a, ['start', 'end']].values
    b = df.loc[mask_b, ['start', 'end']].values

    if operation == 'intersection':
        c = epoch_intersection(a, b)
    elif operation == 'difference':
        c = epoch_difference(a, b)
    elif operation == 'contained':
        c = epoch_contained(a, b)
    else:
        raise ValueError('Unsupported operation {}'.format(operation))

    if len(c) == 0:
        return df.copy()

    new_epochs = pd.DataFrame({
        'name': new_name,
        'start': c[:, 0],
        'end': c[:, 1],
    })
    result = pd.concat((df, new_epochs))
    result.sort_values(['start', 'end', 'name'], inplace=True)
    return result[['name', 'start', 'end']]

def append_epoch(epochs, epoch_name, epoch):
    '''
    Add epoch to the internal epochs dataframe

    Parameters
    ----------
    epochs : DataFrame or None
        existing epochs or None to create new epochs
    epoch_name : string
        Name of epoch
    epoch : 2D array of (M x 2)
        The first column is the start time and second column is the end
        time. M is the number of occurrences of the epoch.
    '''
    # important to match standard column order in case epochs is empty. Some code requires this order??
    #df = df[['name', 'start', 'end']]
    _df = pd.DataFrame({'name': epoch_name, 'start': epoch[0], 'end': epoch[1]}, index=[0])

    if epochs is not None:
        epochs = pd.concat([epochs, _df], ignore_index=True)
    else:
        epochs = _df

    return epochs
