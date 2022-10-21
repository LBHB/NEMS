"""Tools for visualizing the inputs, outputs, and parameters of Layers."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from nems.layers.stp import ShortTermPlasticity
from nems.visualization.tools import standardize_axes
from nems.preprocessing.normalization import minmax, joint_minmax
from nems.distributions import Normal, HalfNormal


def stp_test_input():
    """Generate synthetic data to probe STP's input/output relationship.

    Each channel is structured as:
        Row 0: Random [0,1)
        Row 1: Square-wave
        Row 2: All zero

    Returns
    -------
    np.ndarray
        Shape (100, 3).
    
    """
    input = np.random.rand(100, 3)
    input[:, 1] = 0
    input[10:20, 1] = 1
    input[25:35, 1] = 1
    input[60:90, 1] = 1
    input[:, 2] = 0

    return input


def stp_input_output(n=5, quick_eval=False, figsize=None):
    """Visualize STP input/output relationship for random parameter values.
    
    Creates a 3x3 grid of input/output plots.
    Each column is associated with one parameter sample.
        Column 0: Default parameters.
        Column >= 1: Randomly sampled (*not* `stp.sample_from_priors`.)
                     u:       Normal(mean=0.25, sd=0.25)  s.t.   |u| >= 0.001
                     tau: HalfNormal(           sd=5   )  s.t. |tau| >= 0.100
        For cases where u < 0 (facilitation), input & output will be normalized
        such 
    Each row is associated with one channel of the synthetic input data:
        Row 0: Random [0,1)
        Row 1: Square-wave
        Row 2: All zero.

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
    # 0: random, 1: square, 2: zero
    input = stp_test_input()

    if figsize is None: figsize = ((n*3)+1, 8)
    fig, axes = plt.subplots(3, n, sharey='col', figsize=figsize)
    if n == 1: axes = axes[..., np.newaxis]  # so that subsequent indexing works
    titles = ['Random [0,1)', 'Square', 'Zeros']

    # Generate output data for n parameter sets and add it to the figure.
    for i in range(n):
        stp = ShortTermPlasticity(shape=(3,), quick_eval=quick_eval)

        if i != 0:
            # Make parameter values random, but matched for all channels.
            u, tau = _stp_visualization_parameters(quick_eval=quick_eval)
            stp.set_parameter_values(
                u=[u for _ in range(3)],
                tau=[tau for _ in range(3)]
                )
        else:
            # Use default values, don't resample.
            u = stp.get_parameter_values('u')[0][0]
            tau = stp.get_parameter_values('tau')[0][0]

        # Get Layer output
        output = stp.evaluate(input)

        # Normalize between 0 and 1 for easier visualization
        # (already true for depression, but not facilitation)
        this_input, this_output = joint_minmax(input, output)

        # Plot input and output for each channel
        for j, ax in enumerate(axes[:,i]):
            ax.plot( this_input[:, j], color='black', label='Input')
            ax.plot(this_output[:, j], color='slateblue', label='Output')
            ax.xaxis.set_visible(False)
            ax.set_title(titles[j])
            standardize_axes(ax)

        # Label parameter values
        ax.text(95, 0.5, f'u: {np.round(u,3)}\ntau: {np.round(tau,3)}',
                ha='right')
        ax.legend(frameon=False)
        ax.xaxis.set_visible(True)
        ax.set_xlabel('Time (bins)')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle('STP input/output')
    fig.tight_layout()

    return fig


def _stp_visualization_parameters(quick_eval=False):
    """Sample u and tau. Internal for `stp_input_output`."""
    # Manually sample parameters.
    if not quick_eval:
        u = Normal(mean=0.25, sd=0.25).sample(1)
    else:
        u = HalfNormal(sd=0.1).sample(1)
    tau = HalfNormal(5).sample(1)
    if u < 0: u *= 0.1
    if np.abs(u) < 1e-3: u = np.sign(u)*1e-3
    if tau < 0.1: tau = 0.1

    return u, tau
