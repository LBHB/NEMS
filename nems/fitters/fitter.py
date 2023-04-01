import logging
import numpy as np
import scipy as scp

import nems0.fitters.termination_conditions as tc

log = logging.getLogger(__name__)


def dummy_fitter(sigma, cost_fn, bounds=None, fixed=None):
    '''
    This fitter does not actually take meaningful steps; it merely
    varies the first element of the sigma vector to be equal to the step
    number. It is intended purely for testing and example purposes so
    that you can see how to re-use termination conditions to write
    your own fitter.
    '''
    # Define a stepinfo and termination condition function 'stop_fit'
    stepinfo, update_stepinfo = tc.create_stepinfo()
    stop_fit = lambda : (tc.error_non_decreasing(stepinfo, 1e-5) or
                         tc.max_iterations_reached(stepinfo, 1000))

    while not stop_fit():
        sigma[0] = stepinfo['stepnum']  # Take a fake step
        err = cost_fn(sigma)
        update_stepinfo(err=err, sigma=sigma)

    return sigma


def coordinate_descent(sigma, cost_fn, step_size=0.1, step_change=0.5,
                       step_min=1e-5, tolerance=1e-5, max_iter=100,
                       bounds=None, **kwargs):

    if bounds is not None:
        bounds = list(zip(*bounds))
    delta=0
    stepinfo, update_stepinfo = tc.create_stepinfo()
    stop_fit = lambda : (tc.error_non_decreasing(stepinfo, tolerance)
                         or tc.max_iterations_reached(stepinfo, max_iter)
                         or tc.less_than_equal(step_size, step_min))
    #stop_fit = lambda : (tc.max_iterations_reached(stepinfo, max_iter)
    #                     or tc.less_than_equal(step_size, step_min))

    this_sigma = sigma.copy()
    n_parameters = len(sigma)
    step_errors = np.empty([n_parameters, 2])
    log.info("CD intializing: step_size=%.2f, tolerance=%e, max_iter=%d",
             step_size, tolerance, max_iter)
    this_steps = 0
    while not stop_fit():
        for i in range(0, n_parameters):
            if bounds is None:
                lower = np.NINF
                upper = np.inf
            else:
                lower = bounds[i][0] if bounds[i][0] is not None else np.NINF
                upper = bounds[i][1] if bounds[i][1] is not None else np.inf
            # Try shifting each parameter both negatively and positively
            # proportional to step_size, and save both the new
            # sigma vectors and resulting cost_fn outputs
            this_sigma[i] = sigma[i] + step_size
            if this_sigma[i] > upper:
                this_sigma[i] = upper
            step_errors[i, 0] = cost_fn(this_sigma)
            this_sigma[i] = sigma[i] - step_size
            if this_sigma[i] < lower:
                this_sigma[i] = lower
            step_errors[i, 1] = cost_fn(this_sigma)
            this_sigma[i] = sigma[i]
        # Get index tuple for the lowest error that resulted,
        # and keep the corresponding sigma vector for the next iteration
        i_param, j_sign = np.unravel_index(
                                step_errors.argmin(), step_errors.shape
                                )
        err = step_errors[i_param, j_sign]
        delta = stepinfo['err'] - err

        # If change was negative, try reducing step size.
        if delta < 0:
            log.info("Error worse, reducing step size from %.06f to %.06f",
                     step_size, step_size * step_change)
            step_size *= step_change
            this_steps = 0
            continue
        else:
            this_steps += 1
            if this_steps > 20:
                log.info("Increasing step size from %.06f to %.6f",
                         step_size, step_size / np.sqrt(step_change))
                this_steps = 0
                step_size /= np.sqrt(step_change)

            # If j is 1, shift was negative, otherwise it was 0 for positive.
            if j_sign == 1:
                sigma[i_param] = this_sigma[i_param] = sigma[i_param] - step_size
            else:
                sigma[i_param] = this_sigma[i_param] = sigma[i_param] + step_size

            update_stepinfo(err=err)
            log.debug("step=%d", stepinfo["stepnum"])
            if stepinfo['stepnum'] % 20 == 0:
                log.debug("sigma is now: %s", sigma)

    log.info("Final error: %.06f (step size %.06f)\n",
             stepinfo['err'], step_size)

    return sigma


def scipy_minimize(sigma, cost_fn, tolerance=None, max_iter=None,
                   bounds=None, method='L-BFGS-B', options={}):
    """
    Wrapper for scipy.optimize.minimize to normalize format with
    NEMS fitters.

    TODO: finish this doc

    Does not currently use the stepinfo/termination_conditions
    paradigm.

    TODO: Pull in code from scipy.py in docs/planning to
          expose more output during iteration.
    """
    # Convert NEMS' tolerance and max_iter kwargs to scipy options,
    # but also allow passing options directly
    options = options.copy()
    if tolerance is not None:
        if 'ftol' in options:
            log.warn("Both <tolerance> and <options: ftol> provided for\n"
                     "scipy_minimize, using <tolerance> by default: %.2E",
                     tolerance)
        options['ftol'] = tolerance
        options['gtol'] = tolerance/10
    elif tolerance is None and 'ftol' not in options:
        options['ftol'] = 1e-7

    if max_iter is not None:
        if 'maxiter' in options:
            log.warn("Both <max_iter> and <options: maxiter> provided for\n"
                     "scipy_minimize, using <max_iter> by default: %d",
                     max_iter)
        options['maxiter'] = max_iter
    elif max_iter is None and 'maxiter' not in options:
        options['maxiter'] = 1000

    options['maxfun'] = options['maxiter']*10
    log.info('options %s', options)
    start_err = cost_fn(sigma)

    if np.isnan(np.array(sigma)).any():
        raise ValueError('Sigma contains NaN!')

    # convert to format required by scipy
    bounds = list(zip(*bounds))
    log.info("Start sigma: %s", np.round(sigma, 4))
    result = scp.optimize.minimize(cost_fn, sigma, method=method,
                                   bounds=bounds, options=options)
    sigma = result.x
    final_err = cost_fn(sigma)
    log.info("Stopped due to: %s", result.message)
    log.info("Starting error: %.06f -- Final error: %.06f", start_err, final_err)
    log.info("Final sigma: %s", np.round(sigma, 4))

    return sigma
