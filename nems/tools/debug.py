"""Tools for debugging complex routines like model fitting."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np

from nems.visualization import (
    prediction_vs_target, iteration_vs_error, evals_per_iteration,
    parameter_space_pca, parameter_pca_table, make_axes_plotter
    )
from .io import PrintingBlocked, progress_bar
from .json import save_model, load_model


def get_model_fits(model, input, target, iterations=None, backend='scipy',
                   fitter_options=None, save_path=None, load_path=None,
                   **fit_kwargs):
    """TODO: docs.

    TODO: save/load won't actually work yet b/c Model.results is not saved since
          it can contain arbitrary objects. need to refactor so that FitResults
          are still saved but w/o any problematic objects, add to/from_json for
          FitResults.
    
    Parameters
    ----------
    save_path : str or Path; optional.
        Must point to a directory.
    load_path : str or Path; optional.
        Must point to a directory.
    
    """

    if load_path is not None:
        directory = Path(load_path).glob('**/*')
        models = [load_model(f) for f in directory if f.is_file()]
        return models

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
    if iterations is None: raise NotImplementedError(
        "Must specify iterations in advance for now"
    )
    for _ in progress_bar(range(iterations), prefix="Fitting iteration: "):
        with PrintingBlocked():
            fitted_model = models[-1].fit(
                input, target, fitter_options=fitter_options, **fit_kwargs
                )
            models.append(fitted_model)

    if save_path is not None:
        directory = Path(save_path)
        for i, m in enumerate(models):
            p = directory / f'{i}.json'
            save_model(m, p)

    return models


def debug_plot(models, input, target, figsize=None, sampling_rate=None,
               xlim=None):
    """TODO: docs
    
    Returns
    -------
    fig : matplotlib.pyplot.figure
    models : list of nems.Model
    xlim : tuple; optional.
        Positional arguments for: `ax.set_xlim(*xlim)` for prediction plot.
        If `sampling_rate` is not None, specified in units of seconds instead
        of bins.
    
    """

    if figsize is None: figsize = (10, 7.5)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = GridSpec(9, 12, figure=fig)

    #                           row,  col
    a1 = fig.add_subplot(grid[  :3,   :7])   # top left timeseries
    a2 = fig.add_subplot(grid[ 3:5,   :7])   # top left time series 2
    a3 = fig.add_subplot(grid[  :5,  7: ])   # top right scatter
    a4 = fig.add_subplot(grid[ 5:9,  : ])   # bottom timeseries

    # Plot iteration vs error, number of fn evaluations.
    # Wrap with `make_axes_plotter` to standardize axes formatting
    # (remove upper and right box border, convert bins to time, etc)
    make_axes_plotter(iteration_vs_error, sampling_rate)(models, ax=a1)
    if models[1].results.backend == 'SciPy':
        make_axes_plotter(evals_per_iteration, sampling_rate)(models, ax=a2)

    # Visualize path of optimization through parameter space using PCA
    make_axes_plotter(parameter_space_pca, x_margin=True)(models, ax=a3)

    if (xlim is not None) and (sampling_rate is not None):
        # Scale xlim from units of seconds to units of bins
        xmin, xmax = xlim
        xlim = (xmin*sampling_rate, xmax*sampling_rate)

    # Plot predicted vs actual response, input heatmap
    make_axes_plotter(prediction_vs_target, sampling_rate)(
        input, target, models[0], ax=a4, title='Model prediction',
        show_input=True, xlim=xlim
        )

    return fig


def debug_fitter(model, input, target, iterations=None, backend='scipy',
                 fitter_options=None, figsize=None, sampling_rate=None,
                 save_path=None, load_path=None, xlim=None, **fit_kwargs):
    """TODO: docs

    Parameters
    ----------
    iterations : int; optional.
        Number of fit iterations to record. For `backend='scipy'` this is
        interpreted as `{'options': {'maxiter': iterations}}`. For `backend='tf'`
        this is interpreted as `epochs=iterations`.
    sampling_rate : float; optional.
        If given, label time-axis of some plots with real units (seconds
        by default). See `nems.visualization.tools.ax_bins_to_seconds`.
    xlim : tuple; optional.
        Positional arguments for: `ax.set_xlim(*xlim)` for prediction plot.
        If `sampling_rate` is not None, specified in units of seconds instead
        of bins.
    
    """

    # Get list of models, starting with initial model and appending one per
    # fit iteration.
    models = get_model_fits(
        model, input, target, iterations=iterations, backend=backend,
        fitter_options=fitter_options, save_path=save_path, load_path=load_path,
        **fit_kwargs
        )

    # Create static plot
    fig = debug_plot(models, input, target, figsize=figsize,
                     sampling_rate=sampling_rate, xlim=xlim)

    return fig, models


def animated_debug(model, input, target, iterations=None, backend='scipy',
                   fitter_options=None, figsize=None, sampling_rate=None,
                   save_path=None, load_path=None, animation_save_path=None,
                   frame_interval=500, xlim=None, **fit_kwargs):
    """TODO: docs

    Parameters
    ----------
    xlim : tuple; optional.
        Positional arguments for: `ax.set_xlim(*xlim)` for prediction plot.
        If `sampling_rate` is not None, specified in units of seconds instead
        of bins.
    
    """
    
    # Get list of models, starting with initial model and appending one per
    # fit iteration.
    models = get_model_fits(
        model, input, target, iterations=iterations, backend=backend,
        fitter_options=fitter_options, save_path=save_path, load_path=load_path,
        **fit_kwargs
        )

    # Start with full figure to set limits and color scales.
    fig = debug_plot(models, input, target, figsize=figsize,
                     sampling_rate=sampling_rate, xlim=xlim)

    # Get references to figure artists and their x/ydata for each iteration.
    artists, xdata, ydata = _get_animation_data(fig, input, models, iterations)

    # On each iteration, update x/ydata but leave all axes, labels, etc in place.
    def animation_step(i):
        for j, a in enumerate(artists):
            if isinstance(a, matplotlib.lines.Line2D):
                a.set_xdata(xdata[i][j])
                a.set_ydata(ydata[i][j])
            else:
                a.set_offsets(xdata[i][j])

        return artists

    anim = animation.FuncAnimation(
        fig, animation_step, frames=iterations, interval=frame_interval,
        blit=True
    )

    if animation_save_path is not None:
        fps = int(np.ceil(1/(frame_interval/1000)))
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist='Me'), bitrate=1800
            )
        anim.save(animation_save_path, writer=writer)

    return anim


def _get_animation_data(fig, input, models, iterations):
    """TODO: docs. Internal for animate_debug."""
    # Extract artists for initial figure, and get list of artist data for each
    # subsequent iteration.
    artists = []
    for i, ax in enumerate(fig.axes):
        artists.extend(ax.lines)
        if i == 2:
            # 3rd subplot is PCA scatter, have to add scatter points separately
            # (index 1). Index 0 is the line, 2 and 3 are initial/final.
            artists.append(ax.get_children()[1])

    updated_xdata = []
    updated_ydata = []
    for i in range(2, iterations+2):
        updated_xdata.append([])
        updated_ydata.append([])
        for j, a in enumerate(artists):
            # TODO: change this if more plots added, and/or refactor to not need
            #       a "magic numbers" here
            # TODO: this will break for TF b/c one line plot will be missing
            if j <= 3:
                if isinstance(a, matplotlib.lines.Line2D):
                    # iterations on x-axis
                    new_x = a.get_xdata()[:i]
                    new_y = a.get_ydata()[:i]
                else:
                    # scatter plot PathCollection, only one set of data.
                    new_x = a.get_offsets().data[:i]
                    new_y = []

            elif j == 4:
                # Actual response doesn't change, input heatmap isn't in artists.
                new_x = a.get_xdata()
                new_y = a.get_ydata()
            else:
                # Get new prediction, same x-values
                new_x = a.get_xdata()
                new_y = models[i-1].predict(input)

            updated_xdata[i-2].append(new_x)
            updated_ydata[i-2].append(new_y)

    return artists, updated_xdata, updated_ydata


# TODO: multiple initial conditions.
# TODO: gradient information. for scipy can use scipy_result.jac
#       (one gradient value per parameter)
# TODO: truncate iterations if stopped b/c under tolerance, otherwise it ends
#       up w/a long tail at all the same error.
# TODO: option to not specify iterations, go until it stops
