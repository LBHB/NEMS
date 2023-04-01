from functools import partial
import matplotlib.pyplot as plt
import nems0.modelspec as ms
from nems0.signal import concatenate_channels
import numpy as np


def freeze_defaults(plot_fns, recording, modelspec, evaluator):
    return [partial(pf, recording, modelspec, evaluator) for pf in plot_fns]


def simple_grid(partial_plots, nrows=1, ncols=1, figsize=(12, 9)):
    nplots = len(partial_plots)
    fig = plt.figure(figsize=figsize)

    for i in range(nplots):
        plt.subplot(nrows, ncols, i+1)
        partial_plots[i]()

    return fig


def get_predictions(recording, modelspecs, evaluator=ms.evaluate):
    '''
    Given a recording, a list of modelspecs, and optionally an evaluator
    function, returns a list of prediction signals.
    '''
    recs = [evaluator(recording, mspec) for mspec in modelspecs]
    predictions = [rec['pred'] for rec in recs]
    return predictions


def get_modelspec_names(modelspecs):
    """Given a list of modelspecs, returns a list of descriptive names
    for identifying them in plots."""
    names = [ms.get_modelspec_name(m) for m in modelspecs]
    return names


def plot_layout(plot_fn_struct):
    '''
    Accepts a list of lists of functions of 1 argument (ax).
    Basically a fancy subplot that lets you lay out functions without
    worrying about details. See example below: TODO
    '''
    # Count how many plot functions we want
    nrows = len(plot_fn_struct)
    ncols = max([len(row) for row in plot_fn_struct])
    # Set up the subplots
    fig = plt.figure(figsize=(12, 12))
    for r, row in enumerate(plot_fn_struct):
        for c, plotfn in enumerate(row):
            colspan = max(1, int(ncols / len(row)))
            ax = plt.subplot2grid((nrows, ncols), (r, c), colspan=colspan)
            plotfn(ax=ax)
    return fig


def combine_signal_channels(signals, i, j):
    # TODO: Doesn't seem to be working with plots?
    to_concat = signals[i:j]
    del signals[i:j]
    concatenated = concatenate_channels(to_concat)
    signals.append(concatenated)
    return signals


def pad_to_signals(signals, indices):
    if isinstance(indices, int) or isinstance(indices, np.int64):
        indices = [indices]*len(signals)
    elif len(indices) < len(signals):
        diff = len(signals) - len(indices)
        add_on = indices[-1]*diff
        indices.extend(add_on)
    return indices
