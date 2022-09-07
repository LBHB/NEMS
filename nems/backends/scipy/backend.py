import scipy.optimize

from .cost import get_cost
from ..base import Backend, FitResults


class SciPyBackend(Backend):
    """Default backend: just passthrough NEMS model methods except fit."""
    def _build(self, data=None, eval_kwargs=None, **backend_options):
        """Duplicate reference to `SciPyBackend.nems_model`."""
        # This is the default backend, so there's nothing to build.
        # SciPyBackend.model and .nems_model are the same.
        return self.nems_model

    def _fit(self, data, eval_kwargs=None, cost_function='nmse',
             epochs=1, log_spacing=5, **fitter_options):
        """Fit a Model using `scipy.optimize.minimize`.

        Parameters
        ----------
        data : DataSet
            Model inputs, outputs and targets.
        eval_kwargs : dict
            Keyword arguments for `nems_model.evaluate`.
        cost_function : str or function; default='nmse'
            Specifies which metric to use for computing error while fitting.
            Default is mean squared error normalized by the standard deviation
            of the target.
            If str      : Replace with `get_cost(str)`.
            If function : Use this function to compute errors. Should accept
                          two array arguments and return float.
        epochs : int; default=1.
            Number of "outer loops" to repeat optimization over. Note that this
            is redundant with `fitter_options={'options': {'maxiter': N}}` if
            data contains a single batch (or no batches).
        log_spacing : int; default=5.
            Log progress of fitter every `log_spacing` iterations.
            NOTE: this is the number of iterations, not the number of cost
            function evaluations (of which there may be many per iteration).

        Returns
        -------
        FitResults

        Notes
        -----
        For this Backend, additional **fitter_options are captured because they
        need to be passed directly to `scipy.optimize.minimize`.

        Currently this fitter is designed to work with `method='L-BFGS-B'`.
        Other methods can be used by specifying a 'method' key in
        `fitter_options`, but syntax compatibility with other minimize methods
        has not been tested.

        See also
        --------
        nems.models.base.Model.fit

        """

        if eval_kwargs is None: eval_kwargs = {}
        eval_kwargs['use_existing_maps'] = True
        eval_kwargs['as_dataset'] = True
        batch_size = eval_kwargs.get('batch_size', 0)

        if isinstance(cost_function, str):
            # Convert string reference to loss function
            cost_function = get_cost(cost_function)
        wrapper = _FitWrapper(
            cost_function, self.nems_model, eval_kwargs, log_spacing
            )
        
        # TODO: check tolerance change between epochs.
        for ep in range(epochs):
            print(f"Epoch {ep}")
            print("="*30)
            if batch_size == 0:
                # Don't need to mess with batching.
                _data = data.copy()
                fit_result = wrapper.get_fit_result(_data, **fitter_options)
            else:
                # TODO: maybe better to only generate batches once? Would have
                #       to load them all in a list, but they should be views
                #       anyway.
                batch_generator = self.nems_model.generate_batch_data(
                    data, **eval_kwargs
                    )
                for i, _data in enumerate(batch_generator):
                    print(" "*4 + f"Batch {i}")
                    print(" "*4 + "-"*26)
                    fit_result = wrapper.get_fit_result(_data, **fitter_options)

        print(
            f"Fit successful: {fit_result.success}\n"
            f"Status: {fit_result.status}\n"
            f"Message: {fit_result.message}"
        )
        self.model.set_parameter_vector(fit_result.x)
        
        initial_parameters = wrapper.initial_parameters
        final_parameters = wrapper.model.get_parameter_vector()
        initial_error = wrapper(initial_parameters)
        final_error = wrapper(final_parameters)
        nems_fit_results = FitResults(
            initial_parameters, final_parameters, initial_error, final_error,
            backend_name='SciPy', scipy_fit_result=fit_result
        )

        return nems_fit_results

    def predict(self, input, **eval_kwargs):
        """Alias for `SciPyBackend.nems_model.predict`."""
        self.nems_model.predict(input, **eval_kwargs)


class _FitWrapper:

    def __init__(self, fn, model, eval_kwargs, log_spacing):
        """Internal for `SciPyBackend._fit`."""
        self.fn = fn
        try:
            self.name = fn.__name__
        except AttributeError:
            self.name = 'none'
        self.model = model
        self.initial_parameters = model.get_parameter_vector(as_list=True)
        self.bounds = model.get_bounds_vector(none_for_inf=True)
        self.eval_kwargs = eval_kwargs
        self.log_spacing = log_spacing
        self.data = None
        self.iteration = 0

    def __call__(self, vector):
        """Update model parameters with new vector, compute cost."""
        self.model.set_parameter_vector(vector, ignore_checks=True)
        evaluated_data = self.model.evaluate(self.data, **self.eval_kwargs)
        self.data = evaluated_data
        cost = self.compute_cost()
        return cost

    # TODO: better callback scheme, use logging
    def callback(self, vector):
        """Print error after most recent iteration based on `log_spacing`."""
        if self.iteration % self.log_spacing == 0:
            # Shouldn't need to set parameter vector, that should have
            # been done by the optimization iteration.
            cost = self.__call__(vector)
            print(" "*8 + f"Iteration {self.iteration},"
                    f" error is: {cost:.8f}...")
        self.iteration += 1

    def compute_cost(self):
        """Compute cost given current Model parameters."""
        prediction_list, target_list = self._get_arrays()
        if (len(prediction_list) == 1) and (len(target_list) == 1):
            cost = self.fn(prediction_list[0], target_list[0])
        else:
            # Dict keys are not stored in a guaranteed order, so can't expect
            # .values() to match up even if the lengths are the same. Need to
            # provide a separate mapping of {'pred_key' -> 'target_key'}
            # (and do something different instead of just getting lists).
            raise NotImplementedError(
                "TODO: SciPy cost function for multiple predictions/targets."
                )

        return cost

    def _get_arrays(self):
        """Retrieve outputs and targets from `data` as lists of arrays."""
        prediction = self.data.outputs
        target = self.data.targets

        if len(prediction) == 0:
            # No predictions, error
            raise ValueError(f"{self.name}: No predictions found in data.")
        else:
            predictions = list(prediction.values())
        
        if len(target) == 0:
            # No target, error
            raise ValueError(f"{self.name}: No targets found in data.")
        else:
            targets = list(target.values())

        return predictions, targets

    def get_fit_result(self, data, **fitter_options):
        """Process one optimization epoch."""
        self.data = data
        self.iteration = 0

        if 'method' not in fitter_options:
            fitter_options['method'] = 'L-BFGS-B'
        fit_result = scipy.optimize.minimize(
            self, self.initial_parameters, bounds=self.bounds,
            callback=self.callback, **fitter_options
            )

        return fit_result
