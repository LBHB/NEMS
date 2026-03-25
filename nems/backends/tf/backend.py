import logging
from types import SimpleNamespace
import time
import re
import logging

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input
from keras import regularizers

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

        #elif batch_size is not None:
        #    raise NotImplementedError(
        #        "tf.tensordot is failing for multiple batches b/c the axis "
        #        "numbers shift. Need to fix that before this will work."
        #    )

        data_format = data.data_format
        batch_size = eval_kwargs.get('batch_size', 0)
        if (batch_size == 0) and (data_format not in ('tf.data.Dataset', 'tf.keras.utils.Sequence')):
            data = data.prepend_samples()
            batch_size = None

        if data_format=='array':
            inputs = data.inputs
            # Convert inputs to TensorFlow format
            tf_input_dict = {}
            for k, v in inputs.items():
                # Skip trial/sample dimension when determining shape.
                tf_in = Input(shape=v[0].shape, name=k, batch_size=batch_size,
                              dtype=self.nems_model.dtype)
                tf_input_dict[k] = tf_in
            unused_inputs = list(tf_input_dict.keys())

        elif (data_format =='tf.data.Dataset') or (data_format=='tf.keras.utils.Sequence'):
            itdata = iter(data)
            slice_data = next(itdata)
            if isinstance(slice_data, tuple):
                slice_data = slice_data[0]
            not_inputs = ['task_id', 'output']
            if (sum([ni in slice_data.keys() for ni in not_inputs])>0):
                inputs = {k: v for (k, v) in slice_data.items() if k not in not_inputs}
            else:
                inputs = slice_data
            tf_input_dict = {}
            for k, v in inputs.items():
                tf_in = Input(shape=v[0].shape, name=k, batch_size=batch_size,
                              dtype=self.nems_model.dtype)
                tf_input_dict[k] = tf_in
            unused_inputs = list(tf_input_dict.keys())

        else:
            inputs = data.inputs
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
                    p=10 ** (-float(reg_ops[0].replace("d", ".")))
                    p2=10 ** (-float(reg_ops[1].replace("d", ".")))
                elif len(reg_ops)>=1:
                    p=10 ** (-float(reg_ops[0].replace("d",".")))
                elif len(reg)>2:
                    p = 10 ** (-float(reg[2:].replace("d",".")))
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
            input_shape = [tuple(x.shape) for x in layer_inputs]
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
        # Only expose the final output. Keras 3 applies the compiled loss to
        # every model output, so including intermediate outputs causes evaluate()
        # and fit() to expect matching targets for each one.
        all_outputs = [v for k, v in tf_data.items() if k not in tf_input_dict]
        if len(all_outputs) > 1:
            log.info(
                f"TF model has {len(all_outputs)} intermediate+final outputs but only the last "
                f"will be used as the model output for fitting (see tf.backend._build)."
            )
        tf_outputs = [last_output]
        safe_name = re.sub(r'[^A-Za-z0-9_.>-]', '_', self.nems_model.name or 'nems_model')
        model = tf.keras.Model(inputs=tf_inputs, outputs=tf_outputs, name=safe_name)

        log.info(f'TF model built. (verbose={self.verbose})')
        if self.verbose:
            stringlist=[]
            model.summary(print_fn=lambda x: stringlist.append(x), show_trainable=True)
            for s in stringlist:
                if len(s.strip(" "))>0:
                    log.info(s)

            log.info('')
            log.info('Per-layer trainability:')
            log.info(f'  {"Layer":<25s} {"Total":>8s} {"Trainable":>10s} {"Frozen":>8s}')
            log.info(f'  {"=" * 60}')
            for layer in model.layers:
                trainable_count = int(sum(np.prod(w.shape) for w in layer.trainable_weights))
                frozen_count = int(sum(np.prod(w.shape) for w in layer.non_trainable_weights))
                total_count = trainable_count + frozen_count
                log.info(f'  {layer.name:<25s} {total_count:>8d} {trainable_count:>10d} {frozen_count:>8d}')

        return model

    def _fit(self, data, eval_kwargs=None, cost_function='squared_error',
             epochs=1000, learning_rate=0.001, early_stopping_delay=100,
             early_stopping_patience=150, early_stopping_tolerance=5e-4,
             validation_split=0.0, validation_data=None, shuffle=False, verbose=1, grad_clipnorm=1.0,
             fit_algorithm=None,
             **kwargs):
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
        fit_algorithm : if None, default
            'can': use keras.Model.fit()

        Returns
        -------
        FitResults

        """
        log.info("Starting tf.backend.fit...")
        if kwargs:
            log.warning(f"TF backend ignoring unrecognized fitter_options: {list(kwargs.keys())}")
        if (validation_split>0) & (validation_data is not None):
            log.warning("Both validation_split and validation_data are specified. validation_data will be used for early stopping")

        batch_size = eval_kwargs.get('batch_size', 0)
        if (batch_size == 0) and (data.data_format not in ('tf.data.Dataset', 'tf.keras.utils.Sequence')):
            data = data.prepend_samples()
            batch_size = None

        # Replace cost_function name with function object.
        if isinstance(cost_function, str):
            log.info(f"Cost function: {cost_function}")
            cost_function = get_cost(cost_function)

        # TODO: support more keys in `fitter_options`.
        # Store initial parameters.
        initial_parameters = self.nems_model.get_parameter_vector()
        loss_name = 'loss'  # overridden to 'val_loss' in the array+validation branch

        def _make_callbacks(loss_name):
            cbs = [ProgressCallback(monitor=loss_name, report_frequency=50, epochs=epochs),
                   TerminateOnNaNWeights()]
            if early_stopping_tolerance != 0:
                cbs.append(DelayedStopper(
                    monitor=loss_name, patience=early_stopping_patience,
                    min_delta=early_stopping_tolerance, verbose=1, mode='min',
                    restore_best_weights=True, start_epoch=early_stopping_delay))
            return cbs

        def _to_array(err):
            return np.array([err]) if isinstance(err, float) else err

        # ----------------------------------------------------------------
        # Unified training loop (2026-03-17 JCW/claude).
        # Array data uses plain numpy batching instead of tf.data.Dataset
        # to avoid a known TF 2.10+ memory leak where Dataset graph
        # objects accumulate across repeated fits and are not freed by
        # clear_session() (see keras-team/tf-keras#286).
        # Dataset/Sequence objects passed in externally are used as-is.
        # ----------------------------------------------------------------

        # Convert data to numpy arrays for batching.
        use_numpy_batching = False
        val_np_inputs = None
        val_np_target = None
        if data.data_format == 'array':
            if len(data.targets) > 1:
                raise NotImplementedError("Only one target supported currently.")
            inputs = data.inputs
            target = list(data.targets.values())[0] if data.targets else None
            n_samples = target.shape[0]
            effective_bs = min(batch_size, n_samples) if (batch_size is not None and batch_size > 0) else min(32, n_samples)

            # Pre-convert to float32 numpy arrays (no tf.data.Dataset).
            # np.asarray with dtype returns a view when already float32 — no copy.
            np_inputs = {k: np.asarray(v, dtype=np.float32) for k, v in inputs.items()}
            np_target = np.asarray(target, dtype=np.float32)

            # Apply validation_split by carving off the last fraction of samples.
            # Explicit validation_data (handled below) takes precedence.
            if validation_split > 0 and validation_data is None:
                split_idx = int(n_samples * (1 - validation_split))
                val_np_inputs = {k: v[split_idx:] for k, v in np_inputs.items()}
                val_np_target = np_target[split_idx:]
                np_inputs = {k: v[:split_idx] for k, v in np_inputs.items()}
                np_target = np_target[:split_idx]
                loss_name = 'val_loss'
                log.info(f"Training: {split_idx} samples, val: {n_samples - split_idx}, effective batch_size={effective_bs}")
            else:
                log.info(f"Training: {n_samples} samples, effective batch_size={effective_bs}")

            use_numpy_batching = True

        elif data.data_format in ('tf.data.Dataset', 'tf.keras.utils.Sequence'):
            train_ds = data
            use_numpy_batching = False
            log.info(f"Training: data_format: {data.data_format}, batch_count: {len(data)}")

        else:
            # 'fn' format — extract the generator/Sequence stored in the DataSet wrapper.
            inputs = data.inputs
            train_ds = inputs['input'][0]
            use_numpy_batching = False
            #log.info(f"Training: {n_samples} samples, effective batch_size={effective_bs}")

        # Build optional validation data (also avoid tf.data.Dataset).
        # val_np_inputs/val_np_target may already be set by the array+validation_split
        # branch above — only initialize them here for non-array paths.
        val_ds = None
        if validation_data is not None:
            if isinstance(validation_data, tf.data.Dataset):
                # Already batched externally — use as-is, no re-batching.
                val_ds = validation_data
            else:
                val_x, val_y = validation_data
                if isinstance(val_x, dict):
                    val_np_inputs = {k: np.asarray(v, dtype=np.float32) for k, v in val_x.items()}
                    val_np_target = np.asarray(val_y, dtype=np.float32)
                elif isinstance(val_x, np.ndarray):
                    val_np_inputs = np.asarray(val_x, dtype=np.float32)
                    val_np_target = np.asarray(val_y, dtype=np.float32)
                else:
                    raise TypeError(
                        f"validation_data must be a (x, y) tuple or tf.data.Dataset, got {type(val_x)}"
                    )
            loss_name = 'val_loss'

        def _numpy_batches(np_in, np_tgt, bs, do_shuffle=False):
            """Yield (dict-of-tensors, tensor) batches from numpy arrays."""
            n = np_tgt.shape[0]
            idx = np.random.permutation(n) if do_shuffle else np.arange(n)
            for start in range(0, n, bs):
                sl = idx[start:start + bs]
                bx = {k: tf.constant(v[sl]) for k, v in np_in.items()} if isinstance(np_in, dict) else tf.constant(np_in[sl])
                by = tf.constant(np_tgt[sl])
                yield bx, by

        # Compute initial error — same accounting as _train_step (task loss + regularization).
        init_loss = 0.0
        init_n = 0
        for bx, by in (_numpy_batches(np_inputs, np_target, effective_bs) if use_numpy_batching else train_ds):
            n = by.shape[0]
            batch_loss = float(cost_function(by, self.model(bx, training=False)))
            if self.model.losses:
                batch_loss += float(tf.add_n(self.model.losses))
            init_loss += batch_loss * n
            init_n += n
        initial_error = np.array([init_loss / max(init_n, 1)])
        log.info(f"Init loss: {initial_error[0]:.3f}, tol: {early_stopping_tolerance}, learning_rate: {learning_rate}")
        log.info(f"  batch_size: {batch_size}, shuffle: {shuffle}, grad_clipnorm: {grad_clipnorm}")

        # Custom training loop with GradientTape.
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=grad_clipnorm)
        model_ref = self.model

        if (fit_algorithm is None):  # None, default fit is custom loop
            
            log.info("Using custom loop fitter")

            # Training step on a single batch — @tf.function with fixed
            # input signature so TF traces once and reuses across epochs/fits.
            @tf.function
            def _train_step(batch_x, batch_y):
                with tf.GradientTape() as tape:
                    preds = model_ref(batch_x, training=True)
                    if isinstance(preds, (list, tuple)):
                        preds = preds[-1]
                    loss = cost_function(batch_y, preds)
                    if model_ref.losses:
                        loss += tf.add_n(model_ref.losses)
                grads = tape.gradient(loss, model_ref.trainable_variables)
                # Filter None gradients (variables not in the loss graph) so the
                # Adam step counter is not wasted on no-op updates, matching the
                # behaviour of Keras compiled train_step. 2026-03-18.
                grads_and_vars = [(g, v) for g, v in zip(grads, model_ref.trainable_variables)
                                  if g is not None]
                optimizer.apply_gradients(grads_and_vars)
                return loss

            @tf.function
            def _val_step(batch_x, batch_y):
                preds = model_ref(batch_x, training=False)
                if isinstance(preds, (list, tuple)):
                    preds = preds[-1]
                loss = cost_function(batch_y, preds)
                if model_ref.losses:
                    loss += tf.add_n(model_ref.losses)
                return loss

            def _run_train_epoch():
                total_loss = 0.0
                total_samples = 0
                if use_numpy_batching:
                    for bx, by in _numpy_batches(np_inputs, np_target, effective_bs, do_shuffle=shuffle):
                        #log.info(f"bx shape: {bx['input'].shape} by shape: {by.shape}")
                        n = by.shape[0]
                        total_loss += float(_train_step(bx, by)) * n
                        total_samples += n
                else:
                    for bx, by in train_ds:
                        n = by.shape[0]
                        total_loss += float(_train_step(bx, by)) * n
                        total_samples += n
                return total_loss / max(total_samples, 1)

            def _run_val_epoch():
                total_loss = 0.0
                total_samples = 0
                if val_np_inputs is not None:
                    val_bs = effective_bs if use_numpy_batching else 32
                    for bx, by in _numpy_batches(val_np_inputs, val_np_target, val_bs):
                        n = by.shape[0]
                        total_loss += float(_val_step(bx, by)) * n
                        total_samples += n
                elif val_ds is not None:
                    for bx, by in val_ds:
                        n = by.shape[0]
                        total_loss += float(_val_step(bx, by)) * n
                        total_samples += n
                return total_loss / max(total_samples, 1)

            fit_start_time = time.time()

            use_validation = (val_ds is not None) or (val_np_inputs is not None)
            loss_history = []
            val_loss_history = [] if use_validation else None
            best_loss = np.inf
            best_weights = self.model.get_weights()
            patience_count = 0
            leading_zeros = int(np.log10(max(epochs, 1))) + 1

            for epoch in range(epochs):
                loss_val = _run_train_epoch()
                loss_history.append(loss_val)

                # Reshuffle indices for Sequence objects between epochs.
                if not use_numpy_batching and hasattr(train_ds, 'on_epoch_end'):
                    train_ds.on_epoch_end()

                if use_validation:
                    val_loss_val = _run_val_epoch()
                    val_loss_history.append(val_loss_val)
                    monitor_val = val_loss_val
                    log_suffix = f' - val_loss: {val_loss_val:.4e}'
                else:
                    monitor_val = loss_val
                    log_suffix = ''

                if epoch % 50 == 0:
                    last_time = time.time()
                    dsec = last_time - fit_start_time
                    info = f"Epoch {epoch:0>{leading_zeros}}/{epochs} T: {dsec:.2f} s"
                    log.info(f'{info} - loss: {loss_val:.4e}{log_suffix}')
                    # Log individual layer losses (e.g. HRTF regularization terms).
                    if model_ref.losses:
                        for metric in model_ref.metrics:
                            if hasattr(metric, 'result'):
                                log.info(f'  {metric.name}: {float(metric.result()):.4e}')

                if np.isnan(monitor_val) or np.isinf(monitor_val):
                    log.info(f'Epoch {epoch}: NaN/Inf {loss_name}, restoring best weights')
                    self.model.set_weights(best_weights)
                    break

                # Do not track best_loss/best_weights during the delay period.
                # This replicates DelayedStopper (used in fit_algorithm='can'):
                # best_loss stays at inf until the delay expires, so the first
                # post-delay epoch always resets the baseline and patience counter.
                # Previously the elif branch updated best_loss/best_weights during
                # the delay, causing early stopping to compare against a pre-delay
                # minimum — a harder threshold that could trigger premature stopping.
                # Removed 2026-03-18.
                if epoch >= early_stopping_delay and early_stopping_tolerance != 0:
                    if monitor_val < best_loss - early_stopping_tolerance:
                        best_loss = monitor_val
                        best_weights = self.model.get_weights()
                        patience_count = 0
                    else:
                        patience_count += 1
                        if patience_count >= early_stopping_patience:
                            log.info(f'Early stopping at epoch {epoch}')
                            self.model.set_weights(best_weights)
                            break
                # elif monitor_val < best_loss:
                #     best_loss = monitor_val
                #     best_weights = self.model.get_weights()


            hist_dict = {'loss': loss_history}
            if use_validation:
                hist_dict['val_loss'] = val_loss_history
            history = SimpleNamespace(history=hist_dict)

        elif fit_algorithm=='can':
            # Canned keras.Model.fit()
            log.info("Using keras built-in fitter")
            final_layer = model_ref.layers[-1].name
            model_ref.compile(optimizer=optimizer, loss={final_layer: cost_function})
            fit_start_time = time.time()
            if data.data_format == 'array':
                # pass inputs, target -- needed so that keras can perform the validation_split
                # validation_data forced to be None
                history = model_ref.fit(
                    inputs, {final_layer: target}, epochs=epochs, verbose=0,
                    validation_split=validation_split, callbacks=_make_callbacks(loss_name),
                    validation_data=None, batch_size=effective_bs, shuffle=shuffle)
            else:
                # pass data sequence (batch_size and shuffle ignored?)
                history = self.model.fit(
                    data, epochs=epochs, verbose=0,
                    validation_split=validation_split, callbacks=_make_callbacks(loss_name),
                    validation_data=val_ds, batch_size=batch_size, shuffle=shuffle)
            epoch = len(history.history['loss'])
            
        else:
            raise ValueError(f"Unkown tf fit_algorithm {fit_algorithm}")
            
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

        final_parameters = self.nems_model.get_parameter_vector()
        final_error = np.nanmin(history.history[loss_name])
        elapsed = time.time() - fit_start_time
        log.info(f'Final loss: {final_error:.4f} ({int(elapsed//60)}m {int(elapsed%60)}s, {epoch+1} epochs)')
        nems_fit_results = FitResults(
            initial_parameters, final_parameters, initial_error, final_error,
            backend_name='TensorFlow',
            misc={'TensorFlow History': history.history}
        )

        # Break closure references held by @tf.function traces so that
        # TF resources can be freed by clear_session + gc.collect in
        # Model.fit().
        if fit_algorithm is None:
            del _train_step, _val_step, _run_train_epoch, _run_val_epoch
            del model_ref, optimizer
            if not use_numpy_batching:
                del train_ds
            if val_ds is not None:
                del val_ds

        return nems_fit_results

    def evaluate_all_layers(self, input):
        """Run a forward pass and return a dict of outputs for every NEMS layer.

        Builds a temporary multi-output Keras model that exposes each layer's
        output tensor, runs one forward pass, and returns results keyed by the
        NEMS data_map output name — matching the structure of DataSet returned
        by Model.evaluate().

        Parameters
        ----------
        input : dict of ndarray
            Model inputs without a leading sample dimension (same format as
            passed to Model.evaluate).

        Returns
        -------
        dict
            Keys are NEMS data_map output names; values are squeezed ndarrays.
        """
        tf_layer_map = {l.name: l for l in self.model.layers}
        debug_outputs = []
        debug_keys = []
        for nems_layer in self.nems_model.layers:
            tf_l = tf_layer_map.get(nems_layer.name)
            if tf_l is None:
                continue
            out_key = nems_layer.data_map.out[0] if nems_layer.data_map.out else None
            if out_key is None:
                continue
            try:
                debug_outputs.append(tf_l.output)
                debug_keys.append(out_key)
            except AttributeError:
                pass
        if not debug_outputs:
            return {}
        debug_model = tf.keras.Model(inputs=self.model.inputs, outputs=debug_outputs)
        sample_val = next(iter(input.values())) if isinstance(input, dict) else input
        model_input_ndim = len(self.model.inputs[0].shape)
        if sample_val.ndim < model_input_ndim:
            tf_input = {k: v[np.newaxis] for k, v in input.items()} \
                if isinstance(input, dict) else input[np.newaxis]
        else:
            tf_input = input
        preds = debug_model.predict(tf_input)
        if not isinstance(preds, list):
            preds = [preds]
        return {key: np.squeeze(pred) for key, pred in zip(debug_keys, preds)}

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

    # def dstrf(self, input, t=0, e=0, D=10,
    #           out_channel=0, method='jacobian', batch_size=0, **eval_kwargs):
    #     """Creates a tf model from the modelspec and generates the dstrf.
    #
    #     :param input: The input stimulus [trial X space/freq/etc ... X time
    #     :param t: The index at which the dstrf is calculated. Must be within the data.
    #     :param e: trial/epoch
    #     :param D: The duration of the returned dstrf (i.e. time lag from the index).  If 0, returns the whole dstrf.
    #     :rebuild_model: Rebuild the model to avoid using the cached one.
    #     Zero padded if out of bounds.
    #
    #     :return: np array of size [channels, width]
    #     """
    #     if 'stim' not in rec.signals:
    #         raise ValueError('No "stim" signal found in recording.')
    #     # predict response for preceeding D bins, enough time, presumably, for slow nonlinearities to kick in
    #     D = 50
    #     data = rec['stim']._data[:, np.max([0, index - D]):(index + 1)].T
    #     chan_count = data.shape[1]
    #     if 'state' in rec.signals.keys():
    #         include_state = True
    #         state_data = rec['state']._data[:, np.max([0, index - D]):(index + 1)].T
    #     else:
    #         include_state = False
    #
    #     if index < D:
    #         data = np.pad(data, ((D - index, 0), (0, 0)))
    #         if include_state:
    #             state_data = np.pad(state_data, ((D - index, 0), (0, 0)))
    #
    #     # a few safety checks
    #     if data.ndim != 2:
    #         raise ValueError('Data must be a recording of shape [channels, time].')
    #     # if not 0 <= index < width + data.shape[-2]:
    #
    #     if D > data.shape[-2]:
    #         raise ValueError(f'Index must be within the bounds of the time channel plus width.')
    #
    #     need_fourth_dim = np.any(['Conv2D_NEMS' in m['fn'] for m in self])
    #
    #     # print(f'index: {index} shape: {data.shape}')
    #     # need to import some tf stuff here so we don't clutter and unnecessarily import tf
    #     # (which is slow) when it's not needed
    #     # TODO: is this best practice? Better way to do this?
    #     import tensorflow as tf
    #     from nems0.tf.cnnlink_new import get_jacobian
    #
    #     if self.tf_model is None or rebuild_model:
    #         from nems0.tf import modelbuilder
    #         from nems0.tf.layers import Conv2D_NEMS
    #
    #         # generate the model
    #         model_layers = self.modelspec2tf2(use_modelspec_init=True)
    #         state_shape = None
    #         if need_fourth_dim:
    #             # need a "channel" dimension for Conv2D (like rgb channels, not frequency). Only 1 channel for our data.
    #             data_shape = data[np.newaxis, ..., np.newaxis].shape
    #             if include_state:
    #                 state_shape = state_data[np.newaxis, ..., np.newaxis].shape
    #         else:
    #             data_shape = data[np.newaxis].shape
    #             if include_state:
    #                 state_shape = state_data[np.newaxis].shape
    #         self.tf_model = modelbuilder.ModelBuilder(
    #             name='Test-model',
    #             layers=model_layers,
    #         ).build_model(input_shape=data_shape, state_shape=state_shape)
    #
    #     if type(out_channel) is list:
    #         out_channels = out_channel
    #     else:
    #         out_channels = [out_channel]
    #
    #     if method == 'jacobian':
    #         # need to convert the data to a tensor
    #         stensor = None
    #         if need_fourth_dim:
    #             tensor = tf.convert_to_tensor(data[np.newaxis, ..., np.newaxis], dtype='float32')
    #             if include_state:
    #                 stensor = tf.convert_to_tensor(state_data[np.newaxis, ..., np.newaxis], dtype='float32')
    #         else:
    #             tensor = tf.convert_to_tensor(data[np.newaxis], dtype='float32')
    #             if include_state:
    #                 stensor = tf.convert_to_tensor(state_data[np.newaxis], dtype='float32')
    #
    #         if include_state:
    #             tensor = [tensor, stensor]
    #
    #         for outidx in out_channels:
    #             if include_state:
    #                 w = get_jacobian(self.tf_model, tensor, D, tf.cast(outidx, tf.int32))[0].numpy()[0]
    #             else:
    #                 w = get_jacobian(self.tf_model, tensor, D, tf.cast(outidx, tf.int32)).numpy()[0]
    #
    #             if need_fourth_dim:
    #                 w = w[:, :, 0]
    #
    #             if width == 0:
    #                 _w = w.T
    #             else:
    #                 # pad only the time axis if necessary
    #                 padded = np.pad(w, ((width - 1, width), (0, 0)))
    #                 _w = padded[D:D + width, :].T
    #             if len(out_channels) == 1:
    #                 dstrf = _w
    #             elif outidx == out_channels[0]:
    #                 dstrf = _w[..., np.newaxis]
    #             else:
    #                 dstrf = np.concatenate((dstrf, _w[..., np.newaxis]), axis=2)
    #     else:
    #         dstrf = np.zeros((chan_count, width, len(out_channels)))
    #
    #         if need_fourth_dim:
    #             tensor = tf.convert_to_tensor(data[np.newaxis, ..., np.newaxis])
    #         else:
    #             tensor = tf.convert_to_tensor(data[np.newaxis])
    #         p0 = self.tf_model(tensor).numpy()
    #         eps = 0.0001
    #         for lag in range(width):
    #             for c in range(chan_count):
    #                 d = data.copy()
    #                 d[-lag, c] += eps
    #                 if need_fourth_dim:
    #                     tensor = tf.convert_to_tensor(d[np.newaxis, ..., np.newaxis])
    #                 else:
    #                     tensor = tf.convert_to_tensor(d[np.newaxis])
    #                 p = self.tf_model(tensor).numpy()
    #                 # print(p.shape)
    #                 dstrf[c, -lag, :] = p[0, D, out_channels] - p0[0, D, out_channels]
    #         if len(out_channels) == 1:
    #             dstrf = dstrf[:, :, 0]
    #
    #     return dstrf

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

    def get_jacobian_multi(self, input, out_channel=0, layer_name=None, time_index=-1):
        """
        Gets the jacobian at the given index, with support for multi-input models
        and intermediate layer access.

        Parameters:
        -----------
        input : tensor or list of tensors
            Input data for the model
        out_channel : int
            Output channel to compute jacobian for
        layer_name : str, optional
            Name of intermediate layer to compute jacobian to. If None, uses final output.
        time_index : int
            Time index to use for jacobian computation (default: -1 for last time step)

        Returns:
        --------
        jacobian : tensor or list of tensors
            Jacobian with respect to input(s)
        """

        # Support for multiple inputs - convert to list format for consistency
        if not isinstance(input, list):
            input = [input]

        # Ensure all inputs are float32 tensors
        tensor_list = [tf.cast(i, tf.float32) for i in input]

        # Create intermediate model if layer_name is specified
        if layer_name is not None:
            # Find the target layer
            target_layer = None
            for layer in self.model.layers:
                if layer.name == layer_name:
                    target_layer = layer
                    break

            if target_layer is None:
                raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: "
                               f"{[layer.name for layer in self.model.layers]}")

            # Create intermediate model up to target layer
            if len(tensor_list) == 1:
                intermediate_model = tf.keras.Model(inputs=self.model.input,
                                                  outputs=target_layer.output)
            else:
                intermediate_model = tf.keras.Model(inputs=self.model.inputs,
                                                  outputs=target_layer.output)
        else:
            intermediate_model = self.model

        # Use tf.function decorator for performance
        @tf.function
        def _compute_jacobian(tensor_list, out_channel, time_index):
            with tf.GradientTape(persistent=True) as g:
                # Watch all input tensors
                for tensor in tensor_list:
                    g.watch(tensor)

                # Forward pass through model (or intermediate model)
                if len(tensor_list) == 1:
                    z = intermediate_model(tensor_list[0])
                else:
                    z = intermediate_model(tensor_list)

                # Handle different output formats
                if isinstance(z, list):
                    # Multiple outputs - use the last one
                    output = z[-1]
                else:
                    output = z

                # Handle time indexing
                if len(output.shape) >= 3:  # (batch, time, channels)
                    if time_index == -1:
                        target_output = output[0, -1, out_channel]
                    else:
                        target_output = output[0, time_index, out_channel]
                elif len(output.shape) == 2:  # (batch, channels)
                    target_output = output[0, out_channel]
                else:
                    raise ValueError(f"Unexpected output shape: {output.shape}")

                # Compute jacobian with respect to each input
                jacobians = []
                for tensor in tensor_list:
                    jac = g.jacobian(target_output, tensor)
                    jacobians.append(jac)

            return jacobians if len(jacobians) > 1 else jacobians[0]

        return _compute_jacobian(tensor_list, out_channel, time_index)

    def get_intermediate_jacobian(self, input, out_channel=0, layer_name=None, time_index=-1):
        """
        Computes jacobian of final output with respect to intermediate layer representation.

        This computes ∂(final_output)/∂(intermediate_layer_output), giving the sensitivity
        of the final neural response to each element of the intermediate representation.

        Parameters:
        -----------
        input : tensor or list of tensors
            Input data for the model
        out_channel : int
            Output channel to compute jacobian for
        layer_name : str
            Name of intermediate layer to use as the jacobian "input"
        time_index : int
            Time index to use for jacobian computation (default: -1 for last time step)

        Returns:
        --------
        jacobian : tensor
            Jacobian ∂(final_output)/∂(intermediate_representation)
        """
        # Support for multiple inputs
        if not isinstance(input, list):
            input = [input]

        # Ensure all inputs are float32 tensors
        tensor_list = [tf.cast(i, tf.float32) for i in input]

        if layer_name is None:
            raise ValueError("layer_name must be specified for intermediate jacobian computation")

        # Find the target layer
        target_layer = None
        target_layer_index = None
        for i, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                target_layer = layer
                target_layer_index = i
                break

        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: "
                           f"{[layer.name for layer in self.model.layers]}")

        # Create a model up to the target layer to get intermediate representation
        if len(tensor_list) == 1:
            input_layer = self.model.input
            intermediate_model = tf.keras.Model(inputs=input_layer, outputs=target_layer.output)
        else:
            input_layers = self.model.inputs
            intermediate_model = tf.keras.Model(inputs=input_layers, outputs=target_layer.output)

        # Create a model from the target layer to the final output
        # We need to find all layers after the target layer
        layers_after_target = []
        target_found = False
        for layer in self.model.layers:
            if layer == target_layer:
                target_found = True
                continue
            if target_found:
                layers_after_target.append(layer)

        @tf.function
        def _compute_intermediate_jacobian(tensor_list, out_channel, time_index):
            with tf.GradientTape(persistent=True) as g:
                # Get intermediate representation using the intermediate model
                if len(tensor_list) == 1:
                    intermediate_representation = intermediate_model(tensor_list[0])
                else:
                    intermediate_representation = intermediate_model(tensor_list)

                # Watch the intermediate representation
                g.watch(intermediate_representation)

                # Continue forward pass through remaining layers
                current_output = intermediate_representation
                for layer in layers_after_target:
                    current_output = layer(current_output)

                final_output = current_output

                # Handle different output formats
                if isinstance(final_output, list):
                    output = final_output[-1]
                else:
                    output = final_output

                # Handle time indexing
                if len(output.shape) >= 3:  # (batch, time, channels)
                    if time_index == -1:
                        target_output = output[0, -1, out_channel]
                    else:
                        target_output = output[0, time_index, out_channel]
                elif len(output.shape) == 2:  # (batch, channels)
                    target_output = output[0, out_channel]
                else:
                    raise ValueError(f"Unexpected output shape: {output.shape}")

                # Compute jacobian with respect to intermediate representation
                jacobian = g.jacobian(target_output, intermediate_representation)

            return jacobian

        return _compute_intermediate_jacobian(tensor_list, out_channel, time_index)

    def _create_intermediate_models(self, layer_name):
        """
        Create and cache models for efficient intermediate jacobian computation.

        Parameters:
        -----------
        layer_name : str
            Name of the target intermediate layer

        Returns:
        --------
        tuple : (intermediate_model, final_model, target_layer_index)
            intermediate_model: inputs -> intermediate_layer_output
            final_model: intermediate_layer_output -> final_output
            target_layer_index: index of target layer in original model
        """
        import tensorflow as tf

        # Find the target layer
        target_layer = None
        target_layer_index = None
        for i, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                target_layer = layer
                target_layer_index = i
                break

        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: "
                           f"{[layer.name for layer in self.model.layers]}")

        # Create intermediate model (inputs -> target layer)
        if len(self.model.inputs) == 1:
            input_layer = self.model.input
            intermediate_model = tf.keras.Model(inputs=input_layer, outputs=target_layer.output)
        else:
            input_layers = self.model.inputs
            intermediate_model = tf.keras.Model(inputs=input_layers, outputs=target_layer.output)

        # Create final model (target layer output -> final output)
        # We need to build a model that takes intermediate representation as input
        intermediate_shape = target_layer.output_shape[1:]  # Remove batch dimension
        intermediate_input = tf.keras.Input(shape=intermediate_shape, name='intermediate_input')

        # Find layers after target layer
        layers_after_target = []
        target_found = False
        for layer in self.model.layers:
            if layer == target_layer:
                target_found = True
                continue
            if target_found:
                layers_after_target.append(layer)

        # Build the final model by chaining layers
        current_output = intermediate_input
        for layer in layers_after_target:
            current_output = layer(current_output)

        final_model = tf.keras.Model(inputs=intermediate_input, outputs=current_output)

        return intermediate_model, final_model, target_layer_index

    def get_batch_intermediate_representations(self, input_batch, layer_name):
        """
        Efficiently compute intermediate representations for a batch of inputs.

        Parameters:
        -----------
        input_batch : list of tensors
            List of input tensors for different time points
        layer_name : str
            Name of target intermediate layer

        Returns:
        --------
        intermediate_representations : tensor
            Batch of intermediate representations [batch, time, channels]
        """
        import tensorflow as tf

        # Create or get cached models
        cache_key = f"intermediate_models_{layer_name}"
        if not hasattr(self, '_model_cache'):
            self._model_cache = {}

        if cache_key not in self._model_cache:
            intermediate_model, final_model, target_idx = self._create_intermediate_models(layer_name)
            self._model_cache[cache_key] = (intermediate_model, final_model, target_idx)
        else:
            intermediate_model, final_model, target_idx = self._model_cache[cache_key]

        # Concatenate all time samples into a batch (remove individual batch dims)
        if isinstance(input_batch[0], list):
            # Multi-input case: concatenate each input type separately
            batched_inputs = []
            for input_idx in range(len(input_batch[0])):
                # Each sample[input_idx] has shape [1, time, features], squeeze batch dim and stack
                stacked_input = tf.concat([sample[input_idx] for sample in input_batch], axis=0)
                batched_inputs.append(stacked_input)
        else:
            # Single input case: each sample has shape [1, time, features]
            batched_inputs = tf.concat(input_batch, axis=0)

        # Get all intermediate representations in one forward pass
        intermediate_representations = intermediate_model(batched_inputs)

        return intermediate_representations, final_model

    def get_batch_intermediate_jacobians(self, input_batch, out_channels, layer_name, time_index=-1):
        """
        Efficiently compute intermediate jacobians for multiple time points and output channels.

        Parameters:
        -----------
        input_batch : list of tensors
            List of input tensors for different time points
        out_channels : list of int
            Output channels to compute jacobians for
        layer_name : str
            Name of target intermediate layer
        time_index : int
            Time index for jacobian computation

        Returns:
        --------
        jacobians : tensor
            Jacobians [time_points, out_channels, intermediate_dims...]
        """
        import tensorflow as tf

        # Get intermediate representations and final model
        intermediate_representations, final_model = self.get_batch_intermediate_representations(input_batch, layer_name)

        # Prepare for jacobian computation
        batch_size = len(input_batch)
        if len(intermediate_representations.shape) == 4:  # [batch, time, height, width]
            # Flatten spatial dimensions for jacobian computation
            intermediate_flat = tf.reshape(intermediate_representations,
                                         [batch_size, -1])
            original_shape = intermediate_representations.shape[1:]  # Include time dimension
        elif len(intermediate_representations.shape) == 3:  # [batch, time, features]
            # Flatten time and feature dimensions
            intermediate_flat = tf.reshape(intermediate_representations,
                                         [batch_size, -1])
            original_shape = intermediate_representations.shape[1:]  # Include time dimension
        else:  # [batch, features]
            intermediate_flat = intermediate_representations
            original_shape = intermediate_representations.shape[1:]

        jacobians = []

        @tf.function
        def _compute_jacobian_batch(intermediate_batch, out_channel):
            with tf.GradientTape() as g:
                # Reshape back to original intermediate shape
                intermediate_reshaped = tf.reshape(intermediate_batch,
                                                 [batch_size] + list(original_shape))

                g.watch(intermediate_reshaped)

                # Forward pass through final model
                final_output = final_model(intermediate_reshaped)

                # Handle different output formats
                if isinstance(final_output, list):
                    output = final_output[-1]
                else:
                    output = final_output

                # Handle time indexing and output channel selection
                if len(output.shape) >= 3:  # [batch, time, channels]
                    if time_index == -1:
                        target_outputs = output[:, -1, out_channel]
                    else:
                        target_outputs = output[:, time_index, out_channel]
                elif len(output.shape) == 2:  # [batch, channels]
                    target_outputs = output[:, out_channel]
                else:
                    raise ValueError(f"Unexpected output shape: {output.shape}")

                # Sum over batch to get single scalar for jacobian computation
                target_scalar = tf.reduce_sum(target_outputs)

            jacobian = g.gradient(target_scalar, intermediate_reshaped)
            return jacobian

        # Compute jacobians for all output channels
        for out_channel in out_channels:
            jacobian = _compute_jacobian_batch(intermediate_flat, out_channel)
            jacobians.append(jacobian)

        # Stack jacobians: [out_channels, batch, intermediate_features...]
        jacobians = tf.stack(jacobians, axis=0)

        # Reshape to [batch, out_channels, intermediate_features...]
        jacobians = tf.transpose(jacobians, [1, 0] + list(range(2, len(jacobians.shape))))

        return jacobians

    def dstrf_multi(self, input_data, index, D=10, width=0,
                   out_channel=0, method='jacobian', target_layer=None,
                   rebuild_model=False, **eval_kwargs):
        """
        Multi-input compatible dSTRF computation using get_jacobian_multi.

        Creates a tf model from the modelspec and generates the dstrf with support
        for multi-input models (like HRTF models) and intermediate layer analysis.

        Parameters:
        -----------
        input_data : dict or array
            Dictionary of input signals (e.g., {'stim': array, 'disttheta': array})
            or single array for backward compatibility
        index : int
            The time index at which the dstrf is calculated
        D : int
            The duration/memory of the returned dstrf (time lags from index)
        width : int
            Width of the output dstrf window (default: same as D)
        out_channel : int or list
            Output channel(s) to analyze
        method : str
            'jacobian' or 'perturbation'
        target_layer : str, optional
            Name of intermediate layer for analysis (e.g., 'concatenate' for HRTF+DLC)
        rebuild_model : bool
            Rebuild the model to avoid using cached version

        Returns:
        --------
        dstrf : np.array
            Array of size [channels, width] or [channels, width, out_channels]
        """
        import tensorflow as tf

        # Handle backward compatibility - convert single array to dict format
        if not isinstance(input_data, dict):
            input_data = {'stim': input_data}

        # Determine the primary input signal
        primary_key = 'stim' if 'stim' in input_data else list(input_data.keys())[0]
        primary_data = input_data[primary_key]

        # Extract data around the time point of interest for all inputs
        data_dict = {}
        for key, signal_data in input_data.items():
            data_slice = signal_data[:, np.max([0, index - D]):(index + 1)].T
            if index < D:
                # Pad if we don't have enough history
                pad_width = ((D - index, 0), (0, 0))
                data_slice = np.pad(data_slice, pad_width)
            data_dict[key] = data_slice

        # Get channel count from primary signal
        chan_count = data_dict[primary_key].shape[1]

        if width == 0:
            width = D

        # Safety checks
        for key, data in data_dict.items():
            if data.ndim != 2:
                raise ValueError(f'Data for {key} must be of shape [time, channels].')
            if D > data.shape[0]:
                raise ValueError(f'D must be within bounds of time dimension for {key}.')

        # Check if we need 4D tensors for Conv2D layers
        need_fourth_dim = np.any(['Conv2D_NEMS' in m['fn'] for m in self])

        # Build or use cached model
        if self.model is None or rebuild_model:
            # Note: This uses the NEMS 2.x backend's model, not the legacy tf_model
            # The model should already be built during backend initialization
            if self.model is None:
                raise ValueError("Model not found. Ensure the backend was properly initialized.")

        # Handle output channels
        if isinstance(out_channel, list):
            out_channels = out_channel
        else:
            out_channels = [out_channel]

        if method == 'jacobian':
            # Convert data to tensors with proper formatting
            tensor_dict = {}
            for key, data in data_dict.items():
                if need_fourth_dim:
                    tensor = tf.convert_to_tensor(data[np.newaxis, ..., np.newaxis], dtype='float32')
                else:
                    tensor = tf.convert_to_tensor(data[np.newaxis], dtype='float32')
                tensor_dict[key] = tensor

            # Convert to list format for get_jacobian_multi
            tensor_list = [tensor_dict[key] for key in sorted(tensor_dict.keys())]

            # Compute dSTRF for each output channel
            dstrf_results = []
            for outidx in out_channels:
                # Choose jacobian method based on whether we want intermediate analysis
                if target_layer is not None:
                    # Use intermediate jacobian: ∂(final_output)/∂(intermediate_representation)
                    jacobians = self.get_intermediate_jacobian(
                        tensor_list,
                        out_channel=outidx,
                        layer_name=target_layer,
                        time_index=D-1
                    )
                else:
                    # Use standard jacobian: ∂(final_output)/∂(original_inputs)
                    jacobians = self.get_jacobian_multi(
                        tensor_list,
                        out_channel=outidx,
                        layer_name=None,
                        time_index=D-1
                    )

                # Handle multiple jacobians (one per input)
                if isinstance(jacobians, list):
                    # For multi-input models, we typically want the jacobian w.r.t. the primary input
                    # but we could concatenate or handle differently based on needs
                    primary_idx = 0 if primary_key == sorted(tensor_dict.keys())[0] else \
                                 sorted(tensor_dict.keys()).index(primary_key)
                    w = jacobians[primary_idx].numpy()[0]
                else:
                    w = jacobians.numpy()[0]

                # Remove 4th dimension if it was added for Conv2D
                if need_fourth_dim and len(w.shape) > 2:
                    w = w[:, :, 0]

                # Format output based on width parameter
                if width == 0:
                    _w = w.T
                else:
                    # Pad only the time axis if necessary
                    padded = np.pad(w, ((width - 1, width), (0, 0)))
                    _w = padded[D:D + width, :].T

                dstrf_results.append(_w)

            # Combine results for multiple output channels
            if len(out_channels) == 1:
                dstrf = dstrf_results[0]
            else:
                # Stack along a new axis for multiple output channels
                dstrf = np.stack(dstrf_results, axis=-1)

        else:  # perturbation method
            dstrf = np.zeros((chan_count, width, len(out_channels)))

            # Convert primary data to tensor for baseline prediction
            primary_tensor_data = data_dict[primary_key]
            if need_fourth_dim:
                tensor = tf.convert_to_tensor(primary_tensor_data[np.newaxis, ..., np.newaxis])
            else:
                tensor = tf.convert_to_tensor(primary_tensor_data[np.newaxis])

            # Get baseline prediction
            p0 = self.model(tensor).numpy()

            # Perturbation analysis
            eps = 0.0001
            for lag in range(width):
                for c in range(chan_count):
                    d = primary_tensor_data.copy()
                    d[-lag, c] += eps

                    if need_fourth_dim:
                        tensor = tf.convert_to_tensor(d[np.newaxis, ..., np.newaxis])
                    else:
                        tensor = tf.convert_to_tensor(d[np.newaxis])

                    p = self.model(tensor).numpy()
                    dstrf[c, -lag, :] = p[0, D, out_channels] - p0[0, D, out_channels]

            if len(out_channels) == 1:
                dstrf = dstrf[:, :, 0]

        return dstrf

#
# callbacks
#
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
        self.start_time = time.time()
        self.last_time = self.start_time

    def on_epoch_end(self, epoch, logs=None):
        if (epoch) % self.report_frequency == 0:
            self.last_time = time.time()
            dsec = self.last_time - self.start_time
            info = f"Epoch {epoch:0>{self.leading_zeros}}/{self.epochs} T: {dsec:.2f} s"
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
