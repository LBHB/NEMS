import textwrap

import numpy as np


class Backend:

    def __init__(self, nems_model, data, verbose=1, eval_kwargs=None, **backend_options):
        """Interface between NEMS Models and optimization libraries.

        Each Backend must implement `_build`, `_fit`, and `predict` methods.

        Parameters
        ----------
        nems_model : nems.models.base.Model
            `Backend._build` should return an "equivalent" representation of
            this Model.
        data : DataSet
            Model inputs, outputs and targets for use by `Backend._build`.
        eval_kwargs : dict
            Keyword arguments for `nems_model.evaluate`.
        backend_options : dict
            Additional options for Backend subclasses.
        
        """
        self.nems_model = nems_model
        self.verbose = verbose
        if eval_kwargs is None: eval_kwargs = {}
        self.model = self._build(
            data, eval_kwargs=eval_kwargs, **backend_options
            )

    def _build(self, data, eval_kwargs=None, **backend_options):
        """Construct whatever object this Backend uses for fitting/predicting.
        
        Parameters
        ----------
        data : DataSet
            Model inputs, outputs and targets.
        eval_kwargs : dict
            Keyword arguments for `Backend.nems_model.evaluate`.
        backend_options : dict
            Additional options for Backend subclasses.

        Returns
        -------
        object
            Some object defined by the Backend that handles evaluating data
            and optimizing parameters (comparable to a NEMS Model).

        """
        raise NotImplementedError
        
    def _fit(self, data, eval_kwargs=None, **kwargs):
        """Call Backend.model.fit() or equivalent.
        
        This method should update parameters in-place for `Backend.model` and
        `Backend.nems_model`, and store information about optimization results
        in the returned FitResults object.

        Parameters
        ----------
        data : DataSet
            Model inputs, outputs and targets.
        eval_kwargs : dict
            Keyword arguments for `Backend.nems_model.evaluate`.
        kwargs : dict
            Subclasses can specify additional keyword arguments that will be
            supplied by `Model.fit` as `Backend._fit(..., **fitter_options)`.

        Returns
        -------
        FitResults
        
        """
        raise NotImplementedError

    def predict(self, input, **eval_kwargs):
        """Call _model.predict() or equivalent.
        
        Parameters
        ----------
        input : ndarray, dict, or DataSet
            See `nems.models.base.Model.evaluate` for details.
        eval_kwargs : dict
            Additional keyword arguments for `Backend.nems_model.evaluate`.

        Returns
        -------
        prediction : (varies)
            Some form of data representing the model's output given `input`.
            No specific type is enforced at this time, but it should be readily
            convertible to `np.ndarray` (or a list or dict of ndarrays).
        
        """
        raise NotImplementedError


class FitResults:

    def __init__(self, initial_parameters, final_parameters, initial_error,
                 final_error, backend_name, **misc):
        """Collection of information about one fit of a NEMS Model.

        Parameters
        ----------
        initial_parameters : list or np.ndarray
            Parameters of `Backend.nems_model` prior to fitting, formatted as
            a flattened list or 1D ndarray.
        final_parameters : list of np.ndarray
            Parameters of `Backend.nems_model` after fitting.
        initial_error : float
            `cost_function(input, target)` prior to fitting.
        final_error : float
            `cost_function(input, target)` after fitting.
        backend_name : str
            Name indicating which Backend was used to fit the model.
        misc : dict
            Additional fit-related information to store.

        Attributes
        ----------
        initial_parameters : np.ndarray
        final_parameters : np.ndarray
        initial_error : float
        final_error : float
        backend : str
        n_parameters : int
            Number of parameters in the model.
        n_parameters_changed : int
            Number of indices `i` for which `initial_parameters[i]` does not
            match `final_parameters[i]`.
        
        """
        self.initial_parameters = np.array(initial_parameters)
        self.final_parameters = np.array(final_parameters)
        self.initial_error = initial_error
        self.final_error = final_error
        self.n_parameters = self.initial_parameters.size
        self.n_parameters_changed = np.sum(
            (self.initial_parameters != self.final_parameters)
            )
        self.backend = backend_name
        self.misc = misc

    def __repr__(self):
        attrs = self.__dict__.copy()
        x0 = attrs.pop('initial_parameters')  # TODO: maybe not worth printing?
        x1 = attrs.pop('final_parameters')
        misc = attrs.pop('misc')

        string = "Fit Results:\n"
        string += "="*11 + "\n"
        for k, v in attrs.items():
            string += f"{k}: {v}\n"
        string += f"Misc:\n"
        for k, v in misc.items():
            string += '-'*11 + '\n'
            string += f'{k}:\n{v}\n'
        string += "="*11

        return string

    def to_json(self):
        args = [
            self.initial_parameters,
            self.final_parameters,
            self.initial_error,
            self.final_error,
            self.backend,
        ]

        kwargs = self.misc.copy()
        if self.backend == 'SciPy':
            h = kwargs['scipy_fit_result'].hess_inv
            kwargs['scipy_fit_result'].hess_inv = h.todense()  # np array

        return {'args': args, 'kwargs': kwargs}

    @classmethod
    def from_json(cls, json):
        if json['args'][-1] == 'SciPy':  # backend
            # Convert dict back to object
            class ReloadedOptimizeResult:
                pass
            o = ReloadedOptimizeResult()

            for k, v in json['kwargs']['scipy_fit_result'].items():
                setattr(o, k, v)
            json['kwargs']['scipy_fit_result'] = o

        return FitResults(*json['args'], **json['kwargs'])
