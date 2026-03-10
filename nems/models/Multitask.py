from nems.models import Model
from nems.layers.base import Layer
import numpy as np


class MultiTaskModel(Model):
    """Multi-task variant of NEMS Model supporting multiple output heads"""

    def __init__(self, shared_layers=None, tasks=None, task_layers=None, **kwargs):
        super().__init__(**kwargs)

        # Shared layers
        if shared_layers:
            self.add_layers(*shared_layers)
            self.final_shared = self.layers[-1]
        else:
            self.final_shared = None

        # add task heads
        if tasks is not None:
            self.add_heads(tasks, task_layers=None)
        else:
            self.task_heads = []

    @classmethod
    def from_parent_model(cls, parent_model, tasks, task_layers=None):
        """
        Convert a base Model to MultiTaskModel.

        Args:
            parent_model: Instance of nems.Model
            task_sizes: List[int] - Output sizes for each task
            shared_layer_names: List[str] - Which layers to treat as shared
        """
        from copy import deepcopy
        # Copy critical attributes
        new_model = cls(
            name=parent_model.name + "_multitask",
            dtype=parent_model.dtype,
            meta=parent_model.meta.copy(),
            fs=parent_model.fs
        )

        # Deep copy layers
        for layer in parent_model.layers:
            new_layer = deepcopy(layer)
            new_model.add_layers(new_layer)
            new_model.print_layer_data

        # rename output layer output?
        new_model.layers[-1].output = 'shared_output'
        new_model.layers[-1].reset_map()
        new_model.final_shared = new_model.layers[-1]
        print("adding task heads...")
        # Add task heads
        new_model.add_heads(tasks, task_layers=task_layers)

        # Preserve backend if exists
        if parent_model.backend is not None:
            new_model.backend = parent_model.backend.__class__(new_model)

        return new_model

    def add_heads(self, tasks, task_layers=None):
        # Task-specific output heads
        import nems_lbhb.initializers

        self.task_heads = []
        for i, (siteid, site_data) in enumerate(tasks.items()):
            size = site_data['target'].shape[1]
            if task_layers:
                lkw = task_layers
            else:
                finalnl = f"relu.{size}.o.s"
                reg = ".l2:4"
                if self.final_shared is not None:
                    print(self.layers[-1])
                    lkw = f'wc.{self.final_shared.shape[0]}x{size}.{reg}-{finalnl}'
                else:
                    lkw = f'wc.1x{size}.{reg}-{finalnl}'
            lkw = nems_lbhb.initializers.fill_keyword_string_values(lkw)
            lkw = lkw.split('-')
            head = [keyword_lib[kw] for kw in lkw][0]

            print('adding head')
            print(head)
            head._name = f'{siteid}'
            head.input = self.final_shared.output
            head.output = f'output_{siteid}'
            head.reset_map()
            self.add_layers(head)
            self.layers[-1].sample_from_priors(inplace=True)
            self.task_heads.append(head.name)

    def evaluate(self, input, task_index=None, **kwargs):
        """Override evaluate to handle multiple outputs"""
        # Get shared features
        shared_output = super().evaluate(input, **kwargs)

        # Compute task-specific outputs
        outputs = []
        for i, (siteid, site_data) in enumerate(tasks.items()):
            if task_index is None or i == task_index:
                hn = [hn for hn in self.task_heads if siteid in hn][0]
                outputs.append(shared_output[f'output_{hn}'])

        return outputs if task_index is None else outputs[0]

    def fit(self, tasks, cost_function='squared_error', grad_method='loss_sum', prepend_samples=True, slice_size=5000,
            buffer_size=1000, batch_size=10,
            epochs=1000, learning_rate=0.001, early_stopping_delay=100,
            early_stopping_patience=150, early_stopping_tolerance=5e-4,
            validation_split=0.0, validation_data=None, shuffle=False, verbose=1):

        import tensorflow as tf
        import tensorflow.keras as keras
        from tensorflow.keras import Input
        from keras import regularizers
        import logging
        from nems.backends.base import Backend, FitResults
        from nems.backends.tf.cost import get_cost
        from nems.backends.tf.cost import pearson as pearsonR
        import nems_lbhb.preprocessing

        log = logging.getLogger(__name__)
        from nems.backends import get_backend
        from nems.backends.tf.backend import ProgressCallback
        from nems.backends.tf.PCGrad_tf import PCGrad
        """Override fit for multi-task training"""

        # Handle multiple targets
        log.info("Starting mutlitask fit...")

        # process task data
        mt_dataset, slice_number = create_global_dataset(tasks, slice_size=slice_size, pad_size=firlen,
                                                         drop_remainder=True, prepend_samples=True)

        # set ragged output
        mt_dataset = mt_dataset.map(convert_ragged_output)

        # shuffle
        shuffled_dataset = nems_lbhb.preprocessing.shuffle(buffer_size=buffer_size)

        # batch
        batched_data = shuffled_dataset.batch(batch_size, drop_remainder=True)

        # take slice to test
        data_iter = iter(batched_data)
        slice_data = next(data_iter)

        # Replace cost_function name with function object.
        if isinstance(cost_function, str):
            log.info(f"Cost function: {cost_function}")
            cost_function = get_cost(cost_function)

        initial_parameters = self.get_parameter_vector()

        optimizer = PCGrad(tf.keras.optimizers.Adam(learning_rate=learning_rate))

        eval_kwargs = {'batch_size': batch_size}

        input_keys = [list(v['input'].keys()) for (k, v) in tasks.items()][0]

        data_iter = iter(mt_dataset)
        slice_data = next(data_iter)

        tinput = {k: slice_data[k] for k in input_keys}

        # gen batch shaped input for initial pass
        if (batch_size > 0) & prepend_samples:
            tinput = {k: np.concatenate(batch_size * [v], axis=0) for (k, v) in tinput.items()}

        # perform one forward pass which also fills in layer input/outputs that aren't named?
        _ = self.evaluate(tinput, use_existing_maps=False, **eval_kwargs)

        # build keras model from nems layers
        new_model = self.copy()
        backend_class = get_backend(name='tf')
        backend_obj = backend_class(
            new_model, mt_dataset, verbose=verbose, eval_kwargs=eval_kwargs)
        new_model.backend = backend_obj
        model = new_model.backend._build(mt_dataset, eval_kwargs=eval_kwargs)
        model_output_order = {out._name.split('/')[0]: out_ind for out_ind, out in enumerate(model.output)}
        # grab random batches (i,e. accumulate gradients over random slices regardless of task)
        # Training parameters
        # accum_grads = [tf.zeros_like(var) for var in model.trainable_variables]

        steps_per_epoch = -(slice_number // -batch_size)

        # set tf native enumeration in dataset
        mt_dataset = mt_dataset.enumerate()
        # Custom training loop
        for epoch in range(epochs):
            if grad_method == 'loss_sum':
                chunks = []
                for step_counter, chunk in mt_dataset:
                    chunks.append(chunk)
                    if len(chunks) == batch_size:
                        batched_stim = tf.concat([c['stim'] for c in chunks], axis=0)
                        batched_dlc = tf.concat([c['dlc'] for c in chunks], axis=0)
                        batched_outputs = tf.ragged.stack([tf.squeeze(c['output'], axis=0) for c in chunks], axis=0)
                        batched_tasks = [model_output_order[c['task_id'].numpy().decode('utf-8')] for c in chunks]
                        with tf.GradientTape() as tape:
                            # Forward pass
                            preds = model({'stim': batched_stim,
                                           'dlc': batched_dlc}, training=True)

                            # pull predictions for valid output head given chunk/task_id
                            task_preds = tf.ragged.stack([preds[i][j, :, :] for j, i in enumerate(batched_tasks)],
                                                         axis=0)

                            # compute task specific loss - mean loss across task output
                            total_loss = tf.reduce_mean(cost_function(batched_outputs, task_preds))

                        # calculate gradients
                        acc_grad = tape.gradient(total_loss, model.trainable_variables)
                        # apply updates
                        optimizer.apply_gradients(zip(acc_grad, model.trainable_variables))
                        # reset chunks
                        chunks = []
            if epoch % 50 == 0:
                print(f'Epoch {epoch}: train loss: {np.round(total_loss.numpy()[0], decimals=2)}')

