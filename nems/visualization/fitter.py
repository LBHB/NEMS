"""Utilities for visualizing the model fitting process."""

import numpy as np
import matplotlib.pyplot as plt

from .model import input_heatmap
from .tools import standardize_axes


def prediction_vs_target(input, target, model, ax=None, sampling_rate=None,
                         show_input=False, xlim=None, title=None,
                         xlabel='Time (bins)', ylabel='Firing Rate (Hz)'):
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
    show_input : bool; default=False.
        If True, add a heatmap of input above response and prediction.
    xlim : tuple; optional.
        Positional arguments for: `ax.set_xlim(*xlim)`.
    title : str; optional.
    xlabel : str; default='Time (bins)'.
    ylabel : str; default='Firing Rate (Hz)'.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    Designed for targets with shape (T, 1). Other shapes may work but the plot
    may be difficult to see.
    
    """

    if ax is None: ax = plt.gca()
    if xlim is None: xlim = (None, None)
    ax.plot(target, c='black', alpha=0.3, label='actual')
    ax.plot(model.predict(input), c='black', label='predicted')
    # Have to do this after each plot to get margins to adjust correctly.
    standardize_axes(ax, sampling_rate=None)

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
    ax.set_xlim(*xlim)
    standardize_axes(ax, sampling_rate=sampling_rate)

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

    Returns
    -------
    matplotlib.axes.Axes

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
    standardize_axes(ax)

    return ax


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
    ax.set_xlim(0, len(model_list)-1)
    standardize_axes(ax)

    return ax


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
    evecs : np.ndarray

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

    return projections, percent_variance, evecs


def parameter_space_pca(model_list, ax=None):
    """Plot parameters of each model projected onto first two PCs.

    Principal components are determined by treating each entry in a length-N
    vector returned by `Model.get_parameter_vector()` as a feature, and the
    vector associated with each of the M models in `model_list` as one
    observation, then computing principal components for the resulting
    M x N matrix.
    
    Parameters
    ----------
    model_list : list of nems.Model.
        Must contain at least 2 Models. The Model at index 0 is interpreted as
        the "original" (un-fit) Model. Other models are interpreted as the
        results of a successive fit iterations. In particular, `model.results`
        must not be None for Models at indices 1 or greater.
    ax : Matplotlib.axes.Axes.
        Axes on which to plot.

    Returns
    -------
    matplotlib.axes.Axes

    """

    pcs, percent_variance, _ = get_parameter_pcs(model_list)
    errors = get_model_errors(model_list)

    if ax is None: ax = plt.gca()
    ax.plot(pcs[:,0], pcs[:,1], c='black', zorder=-1)
    ax.scatter(pcs[:,0], pcs[:,1], c=errors, s=100)
    ax.scatter(pcs[0,0], pcs[0,1], marker='o', edgecolors='red',
               facecolors='none', s=200, label='Initial')
    ax.scatter(pcs[-1,0], pcs[-1,1], marker='D', edgecolors='red',
               facecolors='none', s=200, label='Final')

    colors = ax.get_children()[1]
    plt.colorbar(colors, label='Error', ax=ax)
    ax.legend(frameon=False)
    ax.set_xlabel(f'Parameter PC1 ({percent_variance[0]:.1f}% var)')
    ax.set_ylabel(f'Parameter PC2 ({percent_variance[1]:.1f}% var)')
    ax.set_xticks([])
    ax.set_yticks([])
    standardize_axes(ax, x_margin=True)

    return ax


def parameter_pca_table(model_list, n=None, m=3, ax=None, fontsize=16):
    """Generate a table showing Parameters with top 5 weights from first n PCs.
    
    Parameters
    ----------
    model_list : list of nems.Model.
        Must contain at least 2 Models. The Model at index 0 is interpreted as
        the "original" (un-fit) Model. Other models are interpreted as the
        results of a successive fit iterations. In particular, `model.results`
        must not be None for Models at indices 1 or greater.
    n : int; optional.
        Number of PCs to show. All shown by default.
    m : int; default=3.
        List Parameters with the top `m` weights.
    ax : Matplotlib.axes.Axes.
        Axes on which to plot.

    Returns
    -------
    matplotlib.axes.Axes

    """

    _, percent_variance, evecs = get_parameter_pcs(model_list)
    first_n_pcs = evecs[:,:n].T
    top_m_idx = [np.flip(np.argsort(np.abs(e)))[:m] for e in first_n_pcs]
    row_labels = [f'PC{i} ({percent_variance[i-1]:.1f}%)' for i in range(1,n+1)]
    table_text = []

    for i, (ev, idx) in enumerate(zip(first_n_pcs, top_m_idx)):
        table_text.append(['']*m)
        parameters = [
            f'{k}: {p.name}' for k, p in
            model_list[0].get_parameter_from_index(*idx).items()
            ]

        for j, p in enumerate(parameters):
            table_text[i][j] = (p + f';   {ev[j]:.3f}')

    if ax is None: ax = plt.gca()
    ax.imshow([[0]], cmap='Greys')
    table = ax.table(table_text, rowLabels=row_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(fontsize/10, fontsize/10)
    ax.axis('off')
    ax.axis('tight')

    return ax


def scipy_gradient(model_list, ax=None):
    """TODO: docs"""

    grads = np.array([m.results.misc['scipy_fit_result'].jac
                      for m in model_list[1:]])
    # Use all nan gradient for first iteration (not fit).
    all_nan = np.full(shape=(1, grads.shape[1]), fill_value=np.nan)
    grads = np.concatenate([all_nan, grads])
    norm = np.sqrt(np.sum(grads**2, axis=1))

    if ax is None: ax = plt.gca()
    ax.plot(norm, c='black')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('L2 Norm of Gradient')
    ax.set_xlim(0, len(model_list)-1)
    standardize_axes(ax)

    return ax
