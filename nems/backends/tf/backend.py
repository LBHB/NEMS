import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.python.keras import regularizers
import logging

from ..base import Backend, FitResults
from .cost import get_cost
from .cost import pearson as pearsonR

log = logging.getLogger(__name__)

class TensorFlowBackend(Backend):

    def __init__(self, nems_model, data, verbose=1, eval_kwargs=None):
        # TODO: Remove this after issues with using float32 in scipy have
        #       been fixed.
        nems_model.set_dtype(np.float32)
        super().__init__(nems_model, data, verbose=verbose, eval_kwargs=eval_kwargs)

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
        #elif batch_size is not None:
        #    raise NotImplementedError(
        #        "tf.tensordot is failing for multiple batches b/c the axis "
        #        "numbers shift. Need to fix that before this will work."
        #    )
        inputs = data.inputs
        if data.data_format=='array':

            # Convert inputs to TensorFlow format
            tf_input_dict = {}
            for k, v in inputs.items():
                # Skip trial/sample dimension when determining shape.
                tf_in = Input(shape=v[0].shape, name=k, batch_size=batch_size,
                              dtype=self.nems_model.dtype)
                tf_input_dict[k] = tf_in
            unused_inputs = list(tf_input_dict.keys())
        else:
            k = list(inputs.keys())[0]
            tf_input_dict = {k: Input(shape=inputs[k][0][0].shape[1:], name=k, batch_size=batch_size,
                                            dtype=self.nems_model.dtype)}
            unused_inputs = []

        # Pass through Keras functional API to map inputs & outputs.
        last_output = None
        tf_data = tf_input_dict.copy()  # Need to keep actual Inputs separate
        for layer in self.nems_model.layers:
            if layer.regularizer is not None:
                reg = layer.regularizer
                reg_ops = reg.split(':')[1:]
                p = 0.001
                p2 = 0.001
                if len(reg_ops)>=2:
                    p=10 ** (-float(reg_ops[0]))
                    p2=10 ** (-float(reg_ops[1]))
                elif len(reg_ops)>=1:
                    p=10 ** (-float(reg_ops[0]))
                elif len(reg)>2:
                    p = 10 ** (-float(reg[2:]))
                if reg.startswith('l1l2'):
                    tf_kwargs = {'regularizer': regularizers.l1_l2(l1=p,l2=p2)}
                elif reg.startswith('l2'):
                    tf_kwargs = {'regularizer': regularizers.l2(l2=p)}
                elif reg.startswith('l1'):
                    tf_kwargs = {'regularizer': regularizers.l1(l1=p)}
                else:
                    raise ValueError(f"Unknown regularizer {reg}")
                log.info(f"Applying regularizer {reg} (p={p}) to {layer.name}")

            else:
                tf_kwargs = {}  # TODO, regularizer etc.

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
            #log.info(f"INPUT SHAPE: {input_shape}")
            #log.info(f"batch size: {batch_size}")

            tf_layer = layer.as_tensorflow_layer(
                input_shape=input_shape, **tf_kwargs
                )
            #log.info(tf_layer.name)
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

        log.info(f'TF model built. (verbose={self.verbose})')
        if self.verbose:
            stringlist=[]
            model.summary(print_fn=lambda x: stringlist.append(x))
            for s in stringlist:
                if len(s.strip(" "))>0:
                    log.info(s)

        return model

    def _fit(self, data, eval_kwargs=None, cost_function='squared_error',
             epochs=1000, learning_rate=0.001, early_stopping_delay=100,
             early_stopping_patience=150, early_stopping_tolerance=5e-4,
             validation_split=0.0, validation_data=None, shuffle=False, verbose=1):
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
        log.info("Starting tf.backend.fit...")
        batch_size = eval_kwargs.get('batch_size', 0)
        if batch_size == 0:
            data = data.prepend_samples()
            batch_size = None

        # Replace cost_function name with function object.
        if isinstance(cost_function, str):
            log.info(f"Cost function: {cost_function}")
            cost_function = get_cost(cost_function)

        # TODO: support more keys in `fitter_options`.
        # Store initial parameters, compile optimizer and loss for model.
        initial_parameters = self.nems_model.get_parameter_vector()
        final_layer = self.nems_model.layers[-1].name
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss={final_layer: cost_function}
        )
        #, metrics = [pearsonR]
        log.info(f"TF model compiled")

        # Build callbacks for early stopping, ... (what else?)
        loss_name = 'loss'
        if (validation_split > 0) or (validation_data is not None):
            loss_name = 'val_loss'
        callbacks = [ProgressCallback(monitor=loss_name, report_frequency=50, epochs=epochs),
                     TerminateOnNaNWeights(),
        ]
        #log.info(f"{callbacks}")
        if early_stopping_tolerance != 0:
            early_stopping = DelayedStopper(
                monitor=loss_name, patience=early_stopping_patience,
                min_delta=early_stopping_tolerance, verbose=1,
                mode='min',
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

        if data.data_format == 'array':
            if len(data.targets) > 1:
                raise NotImplementedError("Only one target supported currently.")
            elif len(list(data.targets.values())) == 0:
                target = None
            else:
                target = list(data.targets.values())[0]

            initial_error = self.model.evaluate(inputs, target, batch_size=batch_size, return_dict=False, verbose=False)
            if type(initial_error) is float:
                initial_error = np.array([initial_error])
            log.info(f"Init loss: {initial_error[0]:.3f}, tol: {early_stopping_tolerance}, batch_size: {batch_size}, shuffle: {shuffle}")

            history = self.model.fit(
                inputs, {final_layer: target}, epochs=epochs, verbose=0,
                validation_split=validation_split, callbacks=callbacks,
                validation_data=validation_data, batch_size=batch_size, shuffle=shuffle)
        else:
            X_ = inputs['input'][0][0]
            Y_ = inputs['input'][0][1]
            initial_error = self.model.evaluate({'input': X_}, Y_, return_dict=False,
                                                verbose=self.verbose)
            if type(initial_error) is float:
                initial_error=np.array([initial_error])
            log.info(f"Initial loss: {initial_error[0]} batch_size: {batch_size} shuffle: {shuffle}")
            if validation_split==0:
                history = self.model.fit(
                    inputs['input'], None, epochs=epochs, verbose=0,
                    validation_split=validation_split, callbacks=callbacks,
                    validation_data=validation_data, batch_size=batch_size, shuffle=shuffle
                )
            else:
                g_est = inputs['input'].copy()
                g_est.frac=1-validation_split
                g_val = inputs['input'].copy()
                g_val.frac=-validation_split
                history = self.model.fit(
                    g_est,
                    validation_data=g_val, batch_size=batch_size, 
                    epochs=epochs, verbose=0,
                    callbacks=callbacks,
                    shuffle=shuffle
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
        
        # SVD fix to allow tf layer order to change randomly (can't force it to match?)
        for nems_layer in self.nems_model.layers:
            for tf_layer in tf_model_layers:
                if nems_layer.name==tf_layer.name:
                    #log.info(f"fixed order: {nems_layer.name}, {tf_layer.name}")
                    nems_layer.set_parameter_values(tf_layer.weights_to_values(), ignore_bounds=True)

        # OLD
        #layer_iter = zip(self.nems_model.layers, tf_model_layers)
        #for nems_layer, tf_layer in layer_iter:
        #    if nems_layer.name==tf_layer.name:
        #        log.info(f"worked in order: {nems_layer.name}, {tf_layer.name}")
        #        nems_layer.set_parameter_values(tf_layer.weights_to_values(), ignore_bounds=True)
        #    else:
        #        for tfl in tf_model_layers:
        #            if nems_layer.name==tfl.name:
        #                log.info(f"fixed order: {nems_layer.name}, {tfl.name}")
        #                nems_layer.set_parameter_values(tfl.weights_to_values(), ignore_bounds=True)

        final_parameters = self.nems_model.get_parameter_vector()
        final_error = np.nanmin(history.history[loss_name])
        log.info(f'Final loss: {final_error:.4f}')
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
            Output of the associated Keras model.

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

    def initialize_model(self, cost_function='squared_error', eval_kwargs=None):
        log.info("Starting tf.backend.fit...")

        self._build(data, eval_kwargs=eval_kwargs)

        r = self.predict(data, **eval_kwargs)

    def dstrf(self, input, t=0, e=0, D=10,
              out_channel=0, method='jacobian', batch_size=0, **eval_kwargs):
        """Creates a tf model from the modelspec and generates the dstrf.

        :param input: The input stimulus [trial X space/freq/etc ... X time
        :param t: The index at which the dstrf is calculated. Must be within the data.
        :param e: trial/epoch
        :param D: The duration of the returned dstrf (i.e. time lag from the index).  If 0, returns the whole dstrf.
        :rebuild_model: Rebuild the model to avoid using the cached one.
        Zero padded if out of bounds.

        :return: np array of size [channels, width]
        """
        if 'stim' not in rec.signals:
            raise ValueError('No "stim" signal found in recording.')
        # predict response for preceeding D bins, enough time, presumably, for slow nonlinearities to kick in
        D = 50
        data = rec['stim']._data[:, np.max([0, index - D]):(index + 1)].T
        chan_count = data.shape[1]
        if 'state' in rec.signals.keys():
            include_state = True
            state_data = rec['state']._data[:, np.max([0, index - D]):(index + 1)].T
        else:
            include_state = False

        if index < D:
            data = np.pad(data, ((D - index, 0), (0, 0)))
            if include_state:
                state_data = np.pad(state_data, ((D - index, 0), (0, 0)))

        # a few safety checks
        if data.ndim != 2:
            raise ValueError('Data must be a recording of shape [channels, time].')
        # if not 0 <= index < width + data.shape[-2]:

        if D > data.shape[-2]:
            raise ValueError(f'Index must be within the bounds of the time channel plus width.')

        need_fourth_dim = np.any(['Conv2D_NEMS' in m['fn'] for m in self])

        # print(f'index: {index} shape: {data.shape}')
        # need to import some tf stuff here so we don't clutter and unnecessarily import tf
        # (which is slow) when it's not needed
        # TODO: is this best practice? Better way to do this?
        import tensorflow as tf
        from nems0.tf.cnnlink_new import get_jacobian

        if self.tf_model is None or rebuild_model:
            from nems0.tf import modelbuilder
            from nems0.tf.layers import Conv2D_NEMS

            # generate the model
            model_layers = self.modelspec2tf2(use_modelspec_init=True)
            state_shape = None
            if need_fourth_dim:
                # need a "channel" dimension for Conv2D (like rgb channels, not frequency). Only 1 channel for our data.
                data_shape = data[np.newaxis, ..., np.newaxis].shape
                if include_state:
                    state_shape = state_data[np.newaxis, ..., np.newaxis].shape
            else:
                data_shape = data[np.newaxis].shape
                if include_state:
                    state_shape = state_data[np.newaxis].shape
            self.tf_model = modelbuilder.ModelBuilder(
                name='Test-model',
                layers=model_layers,
            ).build_model(input_shape=data_shape, state_shape=state_shape)

        if type(out_channel) is list:
            out_channels = out_channel
        else:
            out_channels = [out_channel]

        if method == 'jacobian':
            # need to convert the data to a tensor
            stensor = None
            if need_fourth_dim:
                tensor = tf.convert_to_tensor(data[np.newaxis, ..., np.newaxis], dtype='float32')
                if include_state:
                    stensor = tf.convert_to_tensor(state_data[np.newaxis, ..., np.newaxis], dtype='float32')
            else:
                tensor = tf.convert_to_tensor(data[np.newaxis], dtype='float32')
                if include_state:
                    stensor = tf.convert_to_tensor(state_data[np.newaxis], dtype='float32')

            if include_state:
                tensor = [tensor, stensor]

            for outidx in out_channels:
                if include_state:
                    w = get_jacobian(self.tf_model, tensor, D, tf.cast(outidx, tf.int32))[0].numpy()[0]
                else:
                    w = get_jacobian(self.tf_model, tensor, D, tf.cast(outidx, tf.int32)).numpy()[0]

                if need_fourth_dim:
                    w = w[:, :, 0]

                if width == 0:
                    _w = w.T
                else:
                    # pad only the time axis if necessary
                    padded = np.pad(w, ((width - 1, width), (0, 0)))
                    _w = padded[D:D + width, :].T
                if len(out_channels) == 1:
                    dstrf = _w
                elif outidx == out_channels[0]:
                    dstrf = _w[..., np.newaxis]
                else:
                    dstrf = np.concatenate((dstrf, _w[..., np.newaxis]), axis=2)
        else:
            dstrf = np.zeros((chan_count, width, len(out_channels)))

            if need_fourth_dim:
                tensor = tf.convert_to_tensor(data[np.newaxis, ..., np.newaxis])
            else:
                tensor = tf.convert_to_tensor(data[np.newaxis])
            p0 = self.tf_model(tensor).numpy()
            eps = 0.0001
            for lag in range(width):
                for c in range(chan_count):
                    d = data.copy()
                    d[-lag, c] += eps
                    if need_fourth_dim:
                        tensor = tf.convert_to_tensor(d[np.newaxis, ..., np.newaxis])
                    else:
                        tensor = tf.convert_to_tensor(d[np.newaxis])
                    p = self.tf_model(tensor).numpy()
                    # print(p.shape)
                    dstrf[c, -lag, :] = p[0, D, out_channels] - p0[0, D, out_channels]
            if len(out_channels) == 1:
                dstrf = dstrf[:, :, 0]

        return dstrf

    @tf.function
    def get_jacobian(self, input, out_channel=0):
        """
        Gets the jacobian at the given index.

        This needs to be a tf.function for a huge speed increase.
        """
        #print(out_channel)
        #if type(out_channel) is int:
        #    oc = tf.constant([out_channel])
        #else:
        #    oc = tf.constant(out_channel)
            
        # support for multiple inputs
        if type(input) is list:
            tensor = [tf.cast(i, tf.float32) for i in input]
        else:
            tensor = tf.cast(input, tf.float32)
            
        with tf.GradientTape(persistent=True) as g:
            g.watch(tensor)
            z = self.model(tensor)

            # assume we only care about first output (think this is NEMS standard)
            if type(z) is list:
                z = tf.gather(z[-1][0, -1, :], indices=out_channel, axis=0)
            else:
                z = tf.gather(z[0, -1, :], indices=out_channel, axis=0)
            res = g.jacobian(z, tensor)

        return res


class DelayedStopper(tf.keras.callbacks.EarlyStopping):
    """Early stopper that waits before kicking in."""
    def __init__(self, start_epoch=100, **kwargs):
        super(DelayedStopper, self).__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor='loss', report_frequency=50, epochs=0):
        self.monitor = monitor
        self.report_frequency = report_frequency
        self.epochs = epochs
        self.leading_zeros = int(np.log10(self.epochs)) + 1

    def on_epoch_end(self, epoch, logs=None):
        if (epoch) % self.report_frequency == 0:

            info = 'Epoch {epoch:0>{zeros}}/{total}'.format(epoch=epoch, zeros=self.leading_zeros, total=self.epochs)
            for k, v in logs.items():
                info += ' - %s:' % k
                if v > 1e-3:
                    info += ' %.4f' % v
                else:
                    info += ' %.4e' % v

            log.info(info)

class TerminateOnNaNWeights(tf.keras.callbacks.Callback):
    """Termiantes on NaN weights, or inf. Modeled on tf.keras.callbacks.TerminateOnNan."""
    def __init__(self, **kwargs):
        super(TerminateOnNaNWeights, self).__init__(**kwargs)
        self.safe_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        """Goes through weights looking for any NaNs."""
        found_nan = None
        for weight in self.model.weights:
            if tf.math.reduce_any(tf.math.is_nan(weight)) or tf.math.reduce_any(tf.math.is_inf(weight)):
                log.info(f'Epoch {epoch}: Invalid weights in "{weight.name}", terminating training')
                log.info(f'Weights {weight}')
                self.model.early_terminated = True
                self.model.stop_training = True
                found_nan = weight.name
                break
        if found_nan is not None:
            if self.safe_weights is not None:
                log.info(f"RESTORING SAFE WEIGHTS??")
                self.model.set_weights(self.safe_weights)
                for weight in self.model.weights:
                    if weight.name==found_nan:
                        log.info(f'Weights {weight}')
        else:
            self.safe_weights = self.model.get_weights()
