"""Tools for debugging complex routines like model fitting."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from .io import PrintingBlocked, progress_bar


def debug_fitter(model, input, target, iterations=100, backend='scipy',
                 fitter_options=None, **fit_kwargs):
    """TODO: docs

    Parameters
    ----------
    iterations : int; default=100.
        Number of fit iterations to record. For `backend='scipy'` this is
        interpreted as `{'options': {'maxiter': iterations}}`. For `backend='tf'`
        this is interpreted as `epochs=iterations`.
    
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

    fig = plt.figure(figsize=(12,5))
    fig1, fig2 = fig.subfigures(1,2)
    a1, a2 = fig1.subplots(2,1)

    # Plot iterations vs error
    errors = [m.results.final_error for m in models[1:]]
    errors.insert(0, models[1].results.initial_error)
    a1.plot(errors, c='black')
    a1.set_ylabel('error')

    if backend == 'scipy':
        # Plot number of function evaluations used in each iteration
        evals_per_iter = [m.results.misc['scipy_fit_result'].nfev
                          for m in models[1:]]
        evals_per_iter.insert(0, np.nan)
        a2.plot(evals_per_iter, c='red')
        a2.set_ylabel('evaluations per iteration')
        a2.set_xlabel('iteration')
    else:
        a1.set_xlabel('iteration')

    # Project parameters at each iteration onto first and second PC,
    # treating individual parameters as "variables" and iterations as
    # "observations". Color by error.
    parameters = np.array([m.results.final_parameters if m.results is not None
                        else m.get_parameter_vector() for m in models])
    # TODO: currently treating every coef of FIR, for example, as an individual parameter.
    #       maybe better to combine each parameter array into one number and then use
    #       those to form the vector?
    parameters -= parameters.mean(axis=0)  # mean 0 for each parameter
    parameters /= parameters.std(axis=0)   # std 1
    cov = np.cov(parameters.T)
    evals, evecs = np.linalg.eig(cov)
    pcs = parameters @ evecs  # principal component projections
    percent_variance = [e/np.sum(evals)*100 for e in evals]

    a3 = fig2.subplots(1,1)
    a3.plot(pcs[:,0], pcs[:,1], c='black', zorder=-1)
    a3.scatter(pcs[:,0], pcs[:,1], c=errors, s=50)
    a3.scatter(pcs[0,0], pcs[0,1], marker='o', edgecolors='red',
               facecolors='none', s=200, label='initial')
    a3.scatter(pcs[-1,0], pcs[-1,1], marker='D', edgecolors='red',
               facecolors='none', s=200, label='final')

    colors = a3.get_children()[2]
    plt.colorbar(colors, label='error', ax=a3)
    a3.legend()
    a3.set_xlabel(f'Parameter PC1 ({percent_variance[0]:.0f}% var)')
    a3.set_ylabel(f'Parameter PC2 ({percent_variance[1]:.0f}% var)')
    a3.set_xticks([])
    a3.set_yticks([])

    # TODO: add initial and final pred vs actual
    # TODO: refactor for animation
    # TODO: split pieces off into subroutines

    return fig, models
