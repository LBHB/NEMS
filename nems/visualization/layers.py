"""Tools for visualizing the inputs, outputs, and parameters of Layers."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from nems.layers.stp import ShortTermPlasticity
from nems.visualization.tools import standardize_axes
from nems.preprocessing.normalization import minmax, redo_minmax
from nems.distributions import Normal, HalfNormal


def stp_input_output(n=5, quick_eval=False, figsize=None):
    """Visualize STP input/output relationship for random parameter values.
    
    Creates a 3x3 grid of input/output plots.
    Each column is associated with one random parameter sample.
    Each row is associated with one channel of the synthetic input data:
        Channel 0: Random [0,1)
        Channel 1: Square-wave
        Channel 2: All zero

    Parameters
    ----------
    n : int; default=5.
        Number of random parameter sets to sample. One column is plotted for
        each set, so much larger or smaller values of `n` will likely require
        a change to `figsize`.
    quick_eval : bool; default=False.
        Determines which evaluation algorithm is used (see `nems.layers.stp`).
    figsize : tuple of int; optional.
        (width, height) in inches. Default is ((n*3)+1, 8).
    
    Returns
    -------
    matplotlib.pyplot.Figure
    
    """

    # Generate synthetic input
    input = np.random.rand(100, 3)
    input[:, 1] = 0
    input[10:20, 1] = 1
    input[25:35, 1] = 1
    input[60:90, 1] = 1
    input[:, 2] = 0

    if figsize is None: figsize = ((n*3)+1, 8)
    fig, axes = plt.subplots(3,5, sharey='col', figsize=figsize)
    titles = ['Random [0,1)', 'Square', 'Zeros']

    # Generate output data for 3 random parameter sets and add it to the figure.
    for i in range(n):
        stp = ShortTermPlasticity(shape=(n,), quick_eval=quick_eval)
        # Manually sample parameters.
        u = Normal(mean=0.25, sd=0.25).sample(1).round(3)
        tau = HalfNormal(5).sample(1).round(3)
        if u < 0: u = (0.1*u).round(3)
        if np.abs(u) < 1e-3: u = np.sign(u)*1e-3
        if tau < 0.1: tau = 0.1
        # Make parameter values random, but matched for all channels.
        stp.set_parameter_values(
            u=[u for _ in range(n)],
            tau=[tau for _ in range(n)]
            )

        # Get Layer output
        output = stp.evaluate(input)

        # Normalize between 0 and 1 for easier visualization
        # (already true for depression, but not facilitation)
        this_output, _min, _max = minmax(output)
        this_input = redo_minmax(input, _min, _max)

        # Plot input and output for each channel
        for j, ax in enumerate(axes[:,i]):
            ax.plot( this_input[:, j], color='black', label='Input')
            ax.plot(this_output[:, j], color='slateblue', label='Output')
            ax.xaxis.set_visible(False)
            ax.set_title(titles[j])
            standardize_axes(ax)

        # Label parameter values
        ax.text(95, 0.5, f'u: {u}\ntau: {tau}', ha='right')
        ax.legend(frameon=False)
        ax.xaxis.set_visible(True)
        ax.set_xlabel('Time (bins)')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle('STP input/output')
    fig.tight_layout()

    return fig
