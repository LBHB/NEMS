import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input

from ..base import Backend, FitResults
from .cost import get_cost


class TensorFlowBackend(Backend):

    def __init__(self, nems_model, data, eval_kwargs=None):
        # TODO: Remove this after issues with using float32 in scipy have
        #       been fixed.
        nems_model.set_dtype(np.float32)
        super().__init__(nems_model, data, eval_kwargs=eval_kwargs)

    def _build(self, data, eval_kwargs=None):
        """Build a TensorFlow Keras model that corresponds to a NEMS Model. 

        Parameters
        ----------
        data : DataSet
            Model inputs, outputs and targets.
        eval_kwargs : dict
            Keyword arguments for `nems_model.evaluate`.

        Returns
        -------
        tensorflow.keras.Model

        Notes
        -----
        This method will only work if all Layers in the NEMS Model have
        implemented `as_tensorflow_layer`.
        
        """
        # TODO: what backend options to accept?

        batch_size = eval_kwargs.get('batch_size', 0)
        if batch_size == 0:
            data = data.prepend_samples()
            batch_size = None
        elif batch_size is not None:
            raise NotImplementedError(
                "tf.tensordot is failing for multiple batches b/c the axis "
                "numbers shift. Need to fix that before this will work."
            )

        inputs = data.inputs

        # Convert inputs to TensorFlow format
        tf_input_dict = {}
        for k, v in inputs.items():
            # Skip trial/sample dimension when determining shape.
            tf_in = Input(shape=v.shape[1:], name=k, batch_size=batch_size,
                          dtype=self.nems_model.dtype)
            tf_input_dict[k] = tf_in
        unused_inputs = list(tf_input_dict.keys())

        # Pass through Keras functional API to map inputs & outputs.
        last_output = None
        tf_data = tf_input_dict.copy()  # Need to keep actual Inputs separate
        tf_kwargs = {}  # TODO, regularizer etc.
        for layer in self.nems_model.layers:
            # Get all `data` keys associated with Layer args and kwargs
            # TODO: how are Layers supposed to know which one is which?
            #       have to check the name?
            layer_map = layer.data_map
            all_data_keys = layer_map.args + list(layer_map.kwargs.values())
            all_data_keys = np.array(all_data_keys).flatten().tolist()

            layer_inputs = []
            for k in all_data_keys:
                if k is None:
                    # Add last output
                    layer_inputs.append(last_output)
                else:
                    # Add Input with matching key
                    layer_inputs.append(tf_data[k])
                if k in unused_inputs:
                    unused_inputs.pop(unused_inputs.index(k))

            # Construct TF layer, provide input_shape for Layers that need that
            # extra information. If inputs is a singleton, unwrap it. Otherwise,
            # layers that expect a single input can break.
            input_shape = [keras.backend.int_shape(x) for x in layer_inputs]
            if len(layer_inputs) == 1:
                layer_inputs = layer_inputs[0]
                input_shape = input_shape[0]

            tf_layer = layer.as_tensorflow_layer(
                input_shape=input_shape, **tf_kwargs
                )
            last_output = tf_layer(layer_inputs)
            
            if isinstance(last_output, (list, tuple)):
                tf_data.update(
                    {k: v for k, v in zip(layer_map.out, last_output)
                    if k is not None}  # indicates unsaved intermediate output
                    )
            elif layer_map.out[0] is not None:
                tf_data[layer_map.out[0]] = last_output

        # Don't include inputs that were never actually passed to any Layers.
        tf_inputs = [v for k, v in tf_input_dict.items() if k not in unused_inputs]
        # For outputs, get all data entries that aren't inputs
        tf_outputs = [v for k, v in tf_data.items() if k not in tf_input_dict]

        model = tf.keras.Model(inputs=tf_inputs, outputs=tf_outputs,
                               name=self.nems_model.name)

        print('TF model built...')
        print(model.summary())

        return model

    def _fit(self, data, eval_kwargs=None, cost_function='squared_error',
             epochs=1, learning_rate=0.001, early_stopping_delay=100,
             early_stopping_patience=150, early_stopping_tolerance=5e-4,
             validation_split=0.0, validation_data=None):
        """Optimize `TensorFlowBackend.nems_model` using Adam SGD.
        
        Currently the use of other TensorFlow optimizers is not exposed as an
        option, but that may be added at a later time.

        TODO: allow loss functions other than mean squared error.

        Parameters
        ----------
        data : DataSet
            Model inputs, outputs and targets. Data must have shape (S, T, ...)
            where S is the number of samples/trials/etc, even if `S=1`.
        eval_kwargs : dict
            Keyword arguments for `nems_model.evaluate`.
        cost_function : str or function; default='squared_error'
            Specifies which metric to use for computing error while fitting.
            Uses mean squared error by default.
            If str      : Replace with `get_loss(str)`.
            If function : Use this function to compute errors. Should accept
                          two array arguments and return float.
        epochs : int
            Number of optimization iterations to perform.
        learning_rate : float; default=0.001.
            See docs for `tensorflow.keras.optimizers.Adam`.
        early_stopping_delay : int
            Minimum epoch before early stopping criteria are used. Set
            delay to 0 to use early stopping immediately.
        early_stopping_tolerance : float
            Minimum change in error between epochs considered to be an
            improvement (`min_delta` for `tf.keras.callbacks.EarlyStopping`).
            To disable early stopping, set tolerance to 0.
        early_stopping_patience : int
            Number of epochs to continue optimization for without improvement.
            (`patience` for `tf.keras.callbacks.EarlyStopping`.)
        validation_split : float
            Proportion of data to treat as temporary validation data
            (`validation_split` for `tf.keras.Model.fit`).
        validation_data : tuple of ndarray or tensors; optional.
            Specify validation data manually (overrides `validation_split`).
            See `tf.keras.Model.fit` for details.

        Returns
        -------
        FitResults

        """

        batch_size = eval_kwargs.get('batch_size', 0)
        if batch_size == 0:
            data = data.prepend_samples()
            batch_size = None

        # Replace cost_function name with function object.
        if isinstance(cost_function, str):
            print(f"cost_function: {cost_function}")
            cost_function = get_cost(cost_function)

        # TODO: support more keys in `fitter_options`.
        # Store initial parameters, compile optimizer and loss for model.
        initial_parameters = self.nems_model.get_parameter_vector()
        final_layer = self.nems_model.layers[-1].name
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={final_layer: cost_function}
        )

        # Build callbacks for early stopping, ... (what else?)
        callbacks = []
        loss_name = 'loss'
        if (validation_split > 0) or (validation_data is not None):
            loss_name = 'val_loss'
        if early_stopping_tolerance != 0:
            early_stopping = DelayedStopper(
                monitor=loss_name, patience=early_stopping_patience,
                min_delta=early_stopping_tolerance, verbose=1,
                restore_best_weights=True, start_epoch=early_stopping_delay,
                )
            callbacks.append(early_stopping)
        # Change back to default None if no callbacks were specified
        if len(callbacks) == 0: callbacks = None

        # TODO: This assumes a single output (like our usual models).
        #       Need to tweak this to be able to fit outputs from multiple
        #       layers. _build would need to establish a mapping I guess, since
        #       it has the information about which layer generates which output.
        inputs = data.inputs
        if len(data.targets) > 1:
            raise NotImplementedError("Only one target supported currently.")
        target = list(data.targets.values())[0]

        loss_fn = self.model.loss[final_layer]
        initial_error = loss_fn(
            tf.constant(target, dtype=tf.float32),
            tf.constant(self.model.predict(inputs), dtype=tf.float32)
            ).numpy()
        print(f"Initialial loss: {initial_error:.5f}")
        
        history = self.model.fit(
            inputs, {final_layer: target}, epochs=epochs,
            validation_split=validation_split, callbacks=callbacks,
            validation_data=validation_data
        )

        # Save weights back to NEMS model
        # Skip input layers.
        # TODO: This assumes there aren't any other extra layers added
        #       by TF. That might not be the case for some model types. Better
        #       approach would be to track keys for the specific layers that
        #       have the parameters we need.
        tf_model_layers = [
            layer for layer in self.model.layers
            if not layer.name in [x.name for x in self.model.inputs]
            ]
        layer_iter = zip(self.nems_model.layers, tf_model_layers)
        for nems_layer, tf_layer in layer_iter:
            nems_layer.set_parameter_values(tf_layer.weights_to_values(), ignore_bounds=True)

        final_parameters = self.nems_model.get_parameter_vector()
        final_error = history.history[loss_name][-1]
        nems_fit_results = FitResults(
            initial_parameters, final_parameters, initial_error, final_error,
            backend_name='TensorFlow',
            misc={'TensorFlow History': history.history}
        )

        return nems_fit_results

    def predict(self, input, batch_size=0, **eval_kwargs):
        """Get output of `TensorFlowBackend.model` given `input`.
        
        Parameters
        ----------
        input : ndarray or list of ndarray, tensor or list of tensor.
        batch_size : int or None; default=0.
        eval_kwargs : dict
            Additional keyword arguments for `Model.evaluate`. Silently ignored.

        Returns
        -------
        np.ndarray
            Outpt of the associated Keras model.

        """
        if batch_size == 0:
            # Prepend samples
            if isinstance(input, (np.ndarray)):
                input = input[np.newaxis, ...]
                batch_size = None
            elif isinstance(input, list) and isinstance(input[0], np.ndarray):
                input = [x[np.newaxis, ...] for x in input]
                batch_size = None

        # TODO: Any kwargs needed here?
        return self.model.predict(input, batch_size=batch_size)


class DelayedStopper(tf.keras.callbacks.EarlyStopping):
    """Early stopper that waits before kicking in."""
    def __init__(self, start_epoch=100, **kwargs):
        super(DelayedStopper, self).__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)
