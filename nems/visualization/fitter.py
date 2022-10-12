"""Utilities for visualizing the model fitting process."""

import numpy as np
import matplotlib.pyplot as plt

from .model import input_heatmap


def prediction_vs_target(input, target, model, ax=None, show_input=False,
                         title=None, xlabel='Time (bins)',
                         ylabel='Firing Rate (Hz)'):
    """Plot actual target, overlay with model prediction.

    Parameters
    ----------
    input : np.ndarray, dict, or DataSet.
        Input to `model`. See `nems.Model.evaluate`.
    target : np.ndarray.
        Target of optimization. See `nems.Model.fit`.
    model : nems.Model.
        Model to generate prediction with.
    ax : Matplotlib.axes.Axes; optional.
        Axes on which to plot.
    title : str; optional.
    xlabel : str; default='Time (bins)'.
    ylabel : str; default='Firing Rate (Hz)'.

    Returns
    -------
    None

    Notes
    -----
    Designed for targets with shape (T, 1). Other shapes may work but the plot
    may be difficult to see.
    
    """

    if ax is None: ax = plt.gca()
    ax.plot(target, c='black', alpha=0.3, label='actual')
    ax.plot(model.predict(input), c='black', label='predicted')

    if show_input:
        # Add input heatmap above prediction
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        extent = [xmin, xmax, ymax*(21/20), ymax*(5/4)]
        input_heatmap(input, ax=ax, extent=extent, add_colorbar=False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax*(5/4))

    ax.legend(frameon=False, bbox_to_anchor=(1.0, 0.65), loc='lower right')
    # Check not None so that existing title is not erased.
    if title is not None:  ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def get_model_errors(model_list):
    """Extract final fit error from FitResults of each Model in list."""
    errors = [m.results.final_error for m in model_list[1:]]
    errors.insert(0, model_list[1].results.initial_error)
    return errors


def iteration_vs_error(model_list, ax=None):
    """Plot error (cost) after each fit iteration.
    
    Parameters
    ----------
    model_list : list of nems.Model.
        Must contain at least 2 Models. The Model at index 0 is interpreted as
        the "original" (un-fit) Model. Other models are interpreted as the
        results of a successive fit iterations. In particular, `model.results`
        must not be None for Models at indices 1 or greater.
    ax : Matplotlib.axes.Axes.
        Axes on which to plot.

    Notes
    -----
    While the Model at index 0 is not actually used by this function, it is
    included because it is used by related plotting functions like
    `parameter_space_pca`.
    
    """

    if ax is None: ax = plt.gca()
    errors = get_model_errors(model_list)
    ax.plot(errors, c='black')
    ax.set_ylabel('Error')


def evals_per_iteration(model_list, ax=None):
    """Plot number of cost function evaluations in each fit iteration.
    
    Only works for models fit using `backend='scipy'`.

    Parameters
    ----------
    model_list : list of nems.Model.
        Must contain at least 2 Models. The Model at index 0 is interpreted as
        the "original" (un-fit) Model. Other models are interpreted as the
        results of a successive fit iterations. In particular, `model.results`
        must not be None for Models at indices 1 or greater.
    ax : Matplotlib.axes.Axes.
        Axes on which to plot.

    Notes
    -----
    While the Model at index 0 is not actually used by this function, it is
    included because it is used by related plotting functions like
    `parameter_space_pca`.
    
    """

    try:
        evals_per_iter = [m.results.misc['scipy_fit_result'].nfev
                          for m in model_list[1:]]
        evals_per_iter.insert(0, np.nan)
    except KeyError:
        raise TypeError(
            '`evals_per_iteration` models must be fit with `backend="scipy"`'
            )

    if ax is None: ax = plt.gca()
    ax.plot(evals_per_iter, c='red')
    ax.set_ylabel('Evaluations per iteration')
    ax.set_xlabel('Iteration')


def get_parameter_pcs(model_list):
    """Treat models as observations, project parameters onto PC space.
    
    For a list of M models with N parameters each (i.e. length of
    `Model.get_parameter_vector`), projection matrix will have shape (M, N).
    If each Model (in order) represents the result of a successive fit iteration,
    then each column represents the path of optimization projected onto that
    PC.

    Parameters
    ----------
    model_list : list of nems.Model.
        Must contain at least 2 Models. The Model at index 0 is interpreted as
        the "original" (un-fit) Model. Other models are interpreted as the
        results of a successive fit iterations. In particular, `model.results`
        must not be None for Models at indices 1 or greater.

    Returns
    -------
    projections : np.ndarray.
    percent_variance : np.ndarray.

    """

    # Get all parameter vectors as an array with models as observations (rows)
    # and individual parameters as features/variables (columns).
    parameters = np.array([m.results.final_parameters if m.results is not None
                           else m.get_parameter_vector() for m in model_list])

    # Normalize within parameter, compute covariance matrix.
    parameters -= parameters.mean(axis=0)  # mean 0
    parameters /= parameters.std(axis=0)   #  std 1
    cov = np.cov(parameters.T)
    
    # Get eigenvalues and eigenvectors, project parameters and compute 
    # percent variance explained.
    evals, evecs = np.linalg.eigh(cov)  # cov always real, symmetric, P.S.D.
    # Switch from increasing to decreasing order.
    evals = np.flip(evals, axis=0)
    evecs = np.flip(evecs, axis=1)
    # Project parameters on to principal components, get pct variance explained
    projections = parameters @ evecs
    percent_variance = [e/np.sum(evals)*100 for e in evals]

    return projections, percent_variance


def parameter_space_pca(model_list, ax=None):
    """TODO: docs
    
    Parameters
    ----------
    model_list : list of nems.Model.
        Must contain at least 2 Models. The Model at index 0 is interpreted as
        the "original" (un-fit) Model. Other models are interpreted as the
        results of a successive fit iterations. In particular, `model.results`
        must not be None for Models at indices 1 or greater.
    ax : Matplotlib.axes.Axes.
        Axes on which to plot.

    """

    pcs, percent_variance = get_parameter_pcs(model_list)
    errors = get_model_errors(model_list)

    if ax is None: ax = plt.gca()
    ax.plot(pcs[:,0], pcs[:,1], c='black', zorder=-1)
    ax.scatter(pcs[:,0], pcs[:,1], c=errors, s=100)
    ax.scatter(pcs[0,0], pcs[0,1], marker='o', edgecolors='red',
               facecolors='none', s=200, label='Initial')
    ax.scatter(pcs[-1,0], pcs[-1,1], marker='D', edgecolors='red',
               facecolors='none', s=200, label='Final')

    colors = ax.get_children()[2]
    plt.colorbar(colors, label='Error', ax=ax)
    ax.legend(frameon=False)
    ax.set_xlabel(f'Parameter PC1 ({percent_variance[0]:.1f}% var)')
    ax.set_ylabel(f'Parameter PC2 ({percent_variance[1]:.1f}% var)')
    ax.set_xticks([])
    ax.set_yticks([])
