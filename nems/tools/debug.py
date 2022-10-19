"""Tools for debugging complex routines like model fitting."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np

from nems.visualization import (
    prediction_vs_target, iteration_vs_error, evals_per_iteration,
    parameter_space_pca, parameter_pca_table, scipy_gradient
    )
from .io import PrintingBlocked, progress_bar
from .json import save_model, load_model


def get_model_fits(model, input, target, iterations=None, save_path=None,
                   load_path=None, backend='scipy', fitter_options=None, 
                   **fit_kwargs):
    """Generate a list of Models by fitting one iteration at a time.
    
    TODO: Truncate iterations if fit is terminating b/c of tolerance criteria.
    TODO: Implement `iterations=None` (i.e. fit until stop critiera are met).

    Parameters
    ----------
    model : nems.Model.
        Initial model to base fits on.
    input : np.ndarray, dict, or DataSet.
        See `nems.models.base.Model.evaluate`.
    target : np.ndarray or dict.
        See `nems.models.base.Model.fit`.
    iterations : int; optional.
        Number of fit iterations to record. For `backend='scipy'` this is
        interpreted as `{'options': {'maxiter': iterations}}`. For `backend='tf'`
        this is interpreted as `epochs=iterations`.
    save_path : str or Path; optional.
        Path where models should be saved after each fit iteration. Must point
        to a directory.
    load_path : str or Path; optional.
        Path to load previously fit models from. Must point to a directory
        containing NEMS models saved to json format.
    backend : str; default='scipy'.
        Name of backend to use for fitting. See `nems.models.base.Model.fit`.
    fitter_options : dict; optional.
        See `nems.models.base.Model.fit` and backend documentation.
    fit_kwargs : dict; optional.
        Additional keyword arguments for `nems.models.base.Model.fit`.

    Returns
    -------
    list of nems.Model.
        Length iterations + 1.

    See also
    --------
    nems.models.base.Model
    nems.tools.json.save_model
    nems.tools.json.load_model
    
    """

    if load_path is not None:
        # TODO: How to ensure these are loaded in the correct order?
        #       Currently assumes they're ordered numerically, increasing.
        directory = Path(load_path).glob('**/*')
        models = [load_model(f) for f in directory if f.is_file()]
        return models[:iterations+1]

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
        raise NotImplementedError(
            f'{backend} backend not implemented for `get_model_fits`.'
            )

    models = [model]
    if iterations is None: raise NotImplementedError(
        "Must specify iterations in advance for now."
    )
    for _ in progress_bar(range(iterations), prefix="Fitting iteration: "):
        with PrintingBlocked():
            fitted_model = models[-1].fit(
                input, target, fitter_options=fitter_options, backend=backend,
                **fit_kwargs
                )
            models.append(fitted_model)

    if save_path is not None:
        directory = Path(save_path)
        for i, m in enumerate(models):
            p = directory / f'{i}.json'
            save_model(m, p)

    return models


def debug_plot(models, input, target, sampling_rate=None, xlim=None,
               figsize=None):
    """Plot error, gradient, parameters, and prediction for each Model.
    
    Parameters
    ----------
    models : list of nems.Model
        Model.results must not be None for all but the first Model, and all
        models must have the same layers and hyperparameters.
        For example, as returned by `get_model_fits`.
    input : np.ndarray, dict, or DataSet.
        See `nems.models.base.Model.evaluate`.
    target : np.ndarray or dict.
        See `nems.models.base.Model.fit`.
    sampling_rate : float; optional.
        If given, label time-axis of some plots with real units (seconds
        by default). See `nems.visualization.tools.ax_bins_to_seconds`.
    xlim : tuple; optional.
        Positional arguments for: `ax.set_xlim(*xlim)` for prediction plot.
        If `sampling_rate` is not None, specified in units of seconds instead
        of bins.
    figsize : 2-tuple of int; optional.
         Figure size in inches, (width, height).

    Returns
    -------
    fig : matplotlib.pyplot.figure
    
    """

    if figsize is None: figsize = (10, 10)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = GridSpec(12, 12, figure=fig)

    #                         row,   col
    a1 = fig.add_subplot(grid[ :3,   :7])   # top left timeseries
    a2 = fig.add_subplot(grid[3:5,   :7])   # top left time series 2
    a3 = fig.add_subplot(grid[ :5,  7: ])   # top right scatter
    a4 = fig.add_subplot(grid[6:8,   :7])   # middle left timeseries
    a5 = fig.add_subplot(grid[6:8,  7: ])   # middle right table
    a6 = fig.add_subplot(grid[8:12,  : ])   # bottom timeseries

    # Plot iteration vs error, number of fn evaluations.
    # Wrap with `make_axes_plotter` to standardize axes formatting
    # (remove upper and right box border, convert bins to time, etc)
    iteration_vs_error(models, ax=a1)
    if models[1].results.backend == 'SciPy':
        # Number of cost function evals vs iterations
        evals_per_iteration(models, ax=a2)
        # Norm of gradient vs iterations
        scipy_gradient(models, ax=a4)

    # Visualize path of optimization through parameter space using PCA
    parameter_space_pca(models, ax=a3)

    # TODO: Can't get this to work right with gridspec
    # Table of first 5 PCs and parameter with largest weight for each PC
    # parameter_pca_table(models, n=5, m=1, fontsize=12, ax=a5)
    a5.axis('off')

    if (xlim is not None) and (sampling_rate is not None):
        # Scale xlim from units of seconds to units of bins
        xmin, xmax = xlim
        xlim = (xmin*sampling_rate, xmax*sampling_rate)

    # Plot predicted vs actual response, input heatmap
    prediction_vs_target(
        input, target, models[0], ax=a6, title='Model prediction',
        show_input=True, sampling_rate=sampling_rate, xlim=xlim
        )

    return fig


def debug_fitter(model, input, target, iterations=None, sampling_rate=None,
                 xlim=None, figsize=None, save_path=None, load_path=None,
                 backend='scipy', fitter_options=None, **fit_kwargs):
    """Plot error, gradient, parameters, and prediction for each fit iteration.

    Parameters
    ----------
    model : nems.Model.
        Initial model to base fits on.
    input : np.ndarray, dict, or DataSet.
        See `nems.models.base.Model.evaluate`.
    target : np.ndarray or dict.
        See `nems.models.base.Model.fit`.
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
    figsize : 2-tuple of int; optional.
         Figure size in inches, (width, height).
    save_path : str or Path; optional.
        Path where models should be saved after each fit iteration. Must point
        to a directory.
    load_path : str or Path; optional.
        Path to load previously fit models from. Must point to a directory
        containing NEMS models saved to json format.
    backend : str; default='scipy'.
        Name of backend to use for fitting. See `nems.models.base.Model.fit`.
    fitter_options : dict; optional.
        See `nems.models.base.Model.fit` and backend documentation.
    fit_kwargs : dict; optional.
        Additional keyword arguments for `nems.models.base.Model.fit`.

    Returns
    -------
    matplotlib.pyplot.figure
    list of nems.Model
        Length iterations + 1.
    
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


def animated_debug(model, input, target, iterations=None, sampling_rate=None,
                   xlim=None, figsize=None, save_path=None, load_path=None,
                   animation_save_path=None, frame_interval=500, backend='scipy',
                   fitter_options=None, **fit_kwargs):
    """Animate error, gradient, parameters, and prediction for each fit iteration.

    Parameters
    ----------
    model : nems.Model.
        Initial model to base fits on.
    input : np.ndarray, dict, or DataSet.
        See `nems.models.base.Model.evaluate`.
    target : np.ndarray or dict.
        See `nems.models.base.Model.fit`.
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
    figsize : 2-tuple of int; optional.
         Figure size in inches, (width, height).
    save_path : str or Path; optional.
        Path where models should be saved after each fit iteration. Must point
        to a directory.
    load_path : str or Path; optional.
        Path to load previously fit models from. Must point to a directory
        containing NEMS models saved to json format.
    animation_save_path : str or Path; optional.
        Path where animation should be saved. Rendering will be handled by
        `matplotlib.animation.FFMpegWriter`.
    frame_interval : int; default=500.
        Delay between frames in milliseconds.
    backend : str; default='scipy'.
        Name of backend to use for fitting. See `nems.models.base.Model.fit`.
    fitter_options : dict; optional.
        See `nems.models.base.Model.fit` and backend documentation.
    fit_kwargs : dict; optional.
        Additional keyword arguments for `nems.models.base.Model.fit`.

    Returns
    -------
    matplotlib.pyplot.figure
    list of nems.Model
        Length iterations + 1.

    See also
    --------
    matplotlib.animation.FuncAnimation
    matplotlib.animation.FFMpegWriter

    Notes
    -----
    Matplotlib animation seems to be very dependent on backend, IDE, OS, etc.
    For example, the animation will likely not play in a jupyter notebook. One
    work-around is to save the animation as a movie to watch separately, which
    is what the `animation_save_path` option is for. However, the FFmpeg encoder
    may or may not already be available on your system. If it is not, try
    `conda install -c conda-forge ffmpeg` (or use your preferred method).

    """
    
    # Get list of models, starting with initial model and appending one per
    # fit iteration.
    models = get_model_fits(
        model, input, target, iterations=iterations, backend=backend,
        fitter_options=fitter_options, save_path=save_path, load_path=load_path,
        **fit_kwargs
        )

    # Start with full figure to set limits and color scales.
    fig = debug_plot(
        models, input, target, figsize=figsize, sampling_rate=sampling_rate,
        xlim=xlim
        )

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
        # NOTE: You may need to install the FFmpeg encoder, for example using
        #       `conda install -c conda-forge ffmpeg`.
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist='Me'), bitrate=1800
            )
        anim.save(animation_save_path, writer=writer)

    return anim, models


def _get_animation_data(fig, input, models, iterations):
    """Get matplotlib artists and data. Internal for `animated_debug`."""
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
            if j <= 4:
                if isinstance(a, matplotlib.lines.Line2D):
                    # iterations on x-axis
                    new_x = a.get_xdata()[:i]
                    new_y = a.get_ydata()[:i]
                else:
                    # scatter plot PathCollection, only one set of data.
                    new_x = a.get_offsets().data[:i]
                    new_y = []

            elif j == 5:
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


# TODO: multiple initial conditions instead of iterations.
# TODO: demo script
