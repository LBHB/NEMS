"""Tools for debugging complex routines like model fitting."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np

from nems.visualization import (
    prediction_vs_target, iteration_vs_error, evals_per_iteration,
    parameter_space_pca, input_heatmap, make_axes_plotter
    )
from .io import PrintingBlocked, progress_bar


def debug_fitter(model, input, target, iterations=100, backend='scipy',
                 fitter_options=None, figsize=(14,14), sampling_rate=None,
                 **fit_kwargs):
    """TODO: docs

    Parameters
    ----------
    iterations : int; default=100.
        Number of fit iterations to record. For `backend='scipy'` this is
        interpreted as `{'options': {'maxiter': iterations}}`. For `backend='tf'`
        this is interpreted as `epochs=iterations`.
    sampling_rate : float; optional.
        If given, label time-axis of some plots with real units (seconds
        by default). See `nems.visualization.tools.ax_bins_to_seconds`.
    
    """
    if fitter_options is None: fitter_options = {}

    # Overwrite `iterations` if specified in `fitter_options`,
    # always fit one iteration at a time.
    if backend == 'scipy':
        if 'options' in fitter_options:
            iterations = fitter_options['options'].get('maxiter', iterations)
        else:
            fitter_options['options'] = {}
        fitter_options['options']['maxiter'] = 1
    elif backend == 'tf':
        iterations = fitter_options.get('epochs', iterations)
        fitter_options['epochs'] = 1
    else:
        raise NotImplementedError('backend not implemented for `debug_fitter`.')

    models = [model]
    for _ in progress_bar(range(iterations), prefix="Fitting iteration: "):
        with PrintingBlocked():
            fitted_model = models[-1].fit(
                input, target, fitter_options=fitter_options, **fit_kwargs
                )
        models.append(fitted_model)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = GridSpec(14, 14, figure=fig)
    #                           row,  col
    a1 = fig.add_subplot(grid[  :3,   :8])   # top left timeseries
    a2 = fig.add_subplot(grid[ 3:6,   :8])   # top left time series 2
    a3 = fig.add_subplot(grid[  :6,  8: ])   # top right scatter
    a4 = fig.add_subplot(grid[ 6:9,   : ])   # top bottom timeseries
    a5 = fig.add_subplot(grid[ 9:11,  : ])   # middle bottom timeseries
    a6 = fig.add_subplot(grid[11:14,  : ])   # bottom bottom timeseries

    # Plot iteration vs error, number of fn evaluations.
    # Wrap with `make_axes_plotter` to standardize axes formatting
    # (remove upper and right box border, convert bins to time, etc)
    make_axes_plotter(iteration_vs_error, sampling_rate)(models, ax=a1)
    if backend == 'scipy':
        make_axes_plotter(evals_per_iteration, sampling_rate)(models, ax=a2)

    # Visualize path of optimization through parameter space using PCA
    make_axes_plotter(parameter_space_pca, x_margin=True)(models, ax=a3)

    # Plot predicted vs actual response, input heatmap
    pvt = make_axes_plotter(prediction_vs_target, sampling_rate)
    pvt(input, target, models[0], ax=a4, title='Initial_model')
    pvt(input, target, models[-1], ax=a6, title='Final model')
    make_axes_plotter(input_heatmap, sampling_rate)(input, ax=a5)
    a4.set_xlabel(None)
    a5.set_xlabel(None)


    # TODO: refactor for animation

    return fig, models
