"""Helpers to build a Tensorflow Keras model from a modelspec."""

import copy
import logging
import os
import shutil
import glob
import typing
from pathlib import Path
from packaging import version

import numpy as np
import tensorflow as tf

import nems0.utils
from nems0 import initializers, recording, get_setting
from nems0 import modelspec as mslib
from nems0.tf import callbacks, loss_functions, modelbuilder
from nems0.tf.layers import Conv2D_NEMS, WeightChannelsNew
from nems0.initializers import init_static_nl

log = logging.getLogger(__name__)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def tf2modelspec(model, modelspec):
    """Populates the modelspec phis with the model layers.

    Does some checking on dims and names as well.
    """
    log.info('Populating modelspec with model weights.')

    # need to keep track of difference between tf layer indices and ms indices because of additional input layers
    idx_offset = 0

    # first layer is the input shape, so skip it
    for idx, layer in enumerate(model.layers):
        # skip input layers (input layers can be midway if using state inputs
        if isinstance(layer, tf.keras.layers.InputLayer):
            idx_offset += 1
            continue

        ms = modelspec[idx - idx_offset]

        if layer.ms_name != ms['fn']:
            raise AssertionError('Model layers and modelspec layers do not match up!')

        phis = layer.weights_to_phi()
        if np.any(['tf_only' in fn for fn in modelspec.fn()]):
            # These TF layers don't have corresponding phi pre-set, so there's nothing to check against.
            # TODO: format phi properly for this model version
            ms['phi'] = phis

        elif 'phi' in ms.keys():
            # check that phis/weights match up in name
            if phis.keys() != ms['phi'].keys():
                raise AssertionError(f'Model layer "{layer.ms_name}" weights and modelspec phis do not have matching names!')

            # check that phis/weights match up in shape
            for model_weights, ms_phis in zip(phis.values(), ms['phi'].values()):
                if model_weights.shape != ms_phis.shape:
                    if layer.ms_name == 'nems0.modules.nonlinearity.double_exponential':
                        continue  # dexp has weird weight shapes due to basic init
                    raise AssertionError(f'Model layer "{layer.ms_name}" weights and modelspec phis do not have matching '
                                         f'shapes!')

            # for non trainable layers, check that values didn't change (within tolerance of gpu computation)
            # TODO: for non trainable layers, should we just not update the modelspec weights?
            if not layer.trainable:
                for model_weights, ms_phis in zip(phis.values(), ms['phi'].values()):
                    if not np.allclose(model_weights, ms_phis, rtol=5e-2, atol=5e-2):
                        #log.warning(f'Frozen layer weights changed:\n{ms_phis}\n{model_weights}')
                        log.warning(f'Model layer "{layer.ms_name}" weights changed significantly despite being frozen!')

            ms['phi'] = phis

    return modelspec


def eval_tf(model: tf.keras.Model, stim: np.ndarray) -> np.ndarray:
    """Evaluate the mode on the stim."""
    return model.predict(stim)


def compare_ms_tf(ms, model, rec_ms, train_data):
    """Compares the evaluation between a modelspec and a keras model.

    For the modelspec, uses a recording object. For the tf model, uses the formatted data from fit_tf,
    which may include state data."""

    pred_tf = model.predict(train_data)
    #import pdb; pdb.set_trace()
    pred_ms = np.swapaxes(ms.evaluate(rec_ms.apply_mask())['pred']._data, 0, 1).reshape(pred_tf.shape)

    # for idx, layer in enumerate(model.layers[1:]):
    #     for weight in layer.weights:
    #         if np.any(np.isnan(weight.numpy())):
    #             print(idx, layer.name, weight.name)

    allclose = np.allclose(pred_ms, pred_tf, rtol=1e-5, atol=1e-5)
    # if not allclose and not np.isnan(allclose):
    #     import matplotlib.pyplot as plt
    #     d = np.abs(pred_ms - pred_tf)
    #     ind = np.unravel_index(np.argmax(d, axis=None), d.shape)
    #     ind = np.s_[ind[0], :, ind[2]]
    #
    #     fig, axes = plt.subplots(figsize=(20, 8), nrows=2)
    #
    #     upto = 200
    #
    #     disp_ms = pred_ms[ind]
    #     disp_tf = pred_tf[ind]
    #
    #     axes[0].plot(disp_ms[:upto], alpha=0.5, label='ms')
    #     axes[0].plot(disp_tf[:upto], alpha=0.5, label='tf')
    #     _ = axes[0].legend()
    #
    #     diff = (disp_ms - disp_tf)[:upto]
    #     axes[1].plot(np.abs(diff), label='diff')
    #     _ = axes[1].legend()
    #     plt.show()

    return np.nanstd(pred_ms.flatten() - pred_tf.flatten())


def _get_tf_data_matrix(rec, signal, epoch_name=None):
    """
    extract signal data and reshape to batch X time X channel matrix to work with TF specs
    """
    if (epoch_name is not None) and (epoch_name != ""):
        # extract out the raw data, and reshape to (batch, time, channel)
        # one batch per occurrence of epoch
        tf_data = np.transpose(rec[signal].extract_epoch(epoch=epoch_name, mask=rec['mask']), [0, 2, 1])
    else:
        # cotinuous, single batch
        tf_data = np.transpose(rec.apply_mask()[signal].as_continuous()[np.newaxis, ...], [0, 2, 1])

    #check for nans



def fit_tf(
        modelspec,
        est: recording.Recording,
        use_modelspec_init: bool = True,
        optimizer: str = 'adam',
        max_iter: int = 10000,
        cost_function: str = 'squared_error',
        early_stopping_steps: int = 5,
        early_stopping_tolerance: float = 5e-4,
        early_stopping_val_split: float = 0,
        learning_rate: float = 1e-4,
        variable_learning_rate: bool = False,
        batch_size: typing.Union[None, int] = None,
        seed: int = 0,
        initializer: str = 'random_normal',
        filepath: typing.Union[str, Path] = None,
        freeze_layers: typing.Union[None, list] = None,
        IsReload: bool = False,
        epoch_name: str = "REFERENCE",
        use_tensorboard: bool = False,
        kernel_regularizer: str = None,
        **context
        ) -> dict:
    """TODO

    :param est:
    :param modelspec:
    :param use_modelspec_init:
    :param optimizer:
    :param max_iter:
    :param cost_function:
    :param early_stopping_steps:
    :param early_stopping_tolerance:
    :param learning_rate:
    :param batch_size:
    :param seed:
    :param filepath:
    :param freeze_layers: Indexes of layers to freeze prior to training. Indexes are modelspec indexes, so are offset
      from model layer indexes.
    :param IsReload:
    :param epoch_name
    :param context:

    :return: dict {'modelspec': modelspec}
    """

    if IsReload:
        return {}

    tf.random.set_seed(seed)
    np.random.seed(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'   # makes output deterministic, but reduces prediction accuracy

    log.info('Building tensorflow keras model from modelspec.')
    nems0.utils.progress_fun()

    # figure out where to save model checkpoints
    job_id = os.environ.get('SLURM_JOBID', None)
    if job_id is not None:
        # if job is running on slurm, need to change model checkpoint dir
        # keep a record of the job id
        modelspec.meta['slurm_jobid'] = job_id

        log_dir_root = Path('/mnt/scratch')
        assert log_dir_root.exists()
        log_dir_base = log_dir_root / Path('SLURM_JOBID' + job_id)
        log_dir_sub = Path(str(modelspec.meta['batch'])) \
                / modelspec.meta.get('cellid', "NOCELL") \
                / modelspec.get_longname()
        filepath = log_dir_base / log_dir_sub
        tbroot = filepath / 'logs'
    elif filepath is None:
        filepath = modelspec.meta['modelpath']
        tbroot = Path(f'/auto/data/tmp/tensorboard/')
    else:
        tbroot = Path(f'/auto/data/tmp/tensorboard/')
    
    filepath = Path(filepath)
    if not filepath.exists():
        filepath.mkdir(exist_ok=True, parents=True)
    cellid = modelspec.meta.get('cellid', 'CELL')
    tbpath = tbroot / (str(modelspec.meta['batch']) + '_' + cellid + '_' + modelspec.meta['modelname'])
    # TODO: should this code just be deleted then?
    if 0 & use_tensorboard:
        # disabled, this is dumb. it deletes the previous round of fitting (eg, tfinit)
        fileList = glob.glob(str(tbpath / '*' / '*'))
        for filePath in fileList:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)

    checkpoint_filepath = filepath / 'weights.hdf5'
    tensorboard_filepath = tbpath
    gradient_filepath = filepath / 'gradients'

    # update seed based on fit index
    seed += modelspec.fit_index

    if False and (freeze_layers is not None) and len(freeze_layers) and (len(freeze_layers)==freeze_layers[-1]+1):
        truncate_model=True
        modelspec_trunc, est_trunc = \
            initializers.modelspec_remove_input_layers(modelspec, est, remove_count=len(freeze_layers))
        modelspec_original = modelspec
        est_original = est
        modelspec = modelspec_trunc
        est = est_trunc
        freeze_layers = None
        log.info(f"Special case of freezing: truncating model. fit_index={modelspec.fit_index} cell_index={modelspec.cell_index}")
    else:
        truncate_model = False
    
    input_name = modelspec.meta.get('input_name', 'stim')
    output_name = modelspec.meta.get('output_name', 'resp')

    # also grab the fs
    fs = est[input_name].fs

    if (epoch_name is not None) and (epoch_name != ""):
        # extract out the raw data, and reshape to (batch, time, channel)
        stim_train = np.transpose(est[input_name].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
        resp_train = np.transpose(est[output_name].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
    else:
        # extract data as a single batch size (1, time, channel)
        stim_train = np.transpose(est.apply_mask()[input_name].as_continuous()[np.newaxis, ...], [0, 2, 1])
        resp_train = np.transpose(est.apply_mask()[output_name].as_continuous()[np.newaxis, ...], [0, 2, 1])

    log.info(f'Feature dimensions: {stim_train.shape}; Data dimensions: {resp_train.shape}.')

    if True:
        log.info("adding a tiny bit of noise to resp_train")
        resp_train = resp_train + np.random.randn(*resp_train.shape)/10000
    # get state if present, and setup training data
    if 'state' in est.signals:
        if (epoch_name is not None) and (epoch_name != ""):
           state_train = np.transpose(est['state'].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
        else:
           state_train = np.transpose(est.apply_mask()['state'].as_continuous()[np.newaxis, ...], [0, 2, 1])
        state_shape = state_train.shape
        log.info(f'State dimensions: {state_shape}')
        train_data = [stim_train, state_train]
    else:
        state_train, state_shape = None, None
        train_data = stim_train
        

    # get the layers and build the model
    log.info(f"Cost function: {cost_function}")
    cost_fn = loss_functions.get_loss_fn(cost_function)
    #model_layers = modelspec.modelspec2tf2(
    #    use_modelspec_init=use_modelspec_init, seed=seed, fs=fs,
    #    initializer=initializer, freeze_layers=freeze_layers,
    #    kernel_regularizer=kernel_regularizer)
    model_layers = modelbuilder.modelspec2tf(modelspec,
        use_modelspec_init=use_modelspec_init, seed=seed, fs=fs,
        initializer=initializer, freeze_layers=freeze_layers,
        kernel_regularizer=kernel_regularizer)

    if np.any([isinstance(layer, Conv2D_NEMS) or isinstance(layer, WeightChannelsNew) for layer in model_layers]):
        # need a "channel" dimension for Conv2D (like rgb channels, not frequency). Only 1 channel for our data.
        stim_train = stim_train[..., np.newaxis]
        train_data = train_data[..., np.newaxis]

    # do some batch sizing logic
    batch_size = stim_train.shape[0] if batch_size == 0 else batch_size

    if variable_learning_rate:
        # TODO: allow other schedule options instead of hard-coding exp decay?
        # TODO: expose exp decay kwargs as kw options? not clear how to choose these parameters
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.9
        )

    from nems0.tf.loss_functions import pearson
    model = modelbuilder.ModelBuilder(
        name='Test-model',
        layers=model_layers,
        learning_rate=learning_rate,
        loss_fn=cost_fn,
        optimizer=optimizer,
        metrics=[pearson],
    ).build_model(input_shape=stim_train.shape, state_shape=state_shape, batch_size=batch_size)

    if freeze_layers is not None:
        for freeze_index in freeze_layers:
            log.info(f'TF layer #{freeze_index}: "{model.layers[freeze_index + 1].name}" is not trainable.')

    # tracking early termination
    model.early_terminated = False

    # create the callbacks
    early_stopping = callbacks.DelayedStopper(monitor='val_loss',
                                              patience=30 * early_stopping_steps,
                                              min_delta=early_stopping_tolerance,
                                              verbose=1,
                                              restore_best_weights=True)
    regular_stopping = callbacks.DelayedStopper(monitor='loss',
                                              patience=30 * early_stopping_steps,
                                              min_delta=early_stopping_tolerance,
                                              verbose=1,
                                              restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_filepath),
                                                    save_best_only=False,
                                                    save_weights_only=True,
                                                    save_freq=100 * stim_train.shape[0],
                                                    monitor='loss',
                                                    verbose=0)
    sparse_logger = callbacks.SparseProgbarLogger(n_iters=50)
    nan_terminate = tf.keras.callbacks.TerminateOnNaN()
    nan_weight_terminate = callbacks.TerminateOnNaNWeights()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_filepath),  # TODO: generic tensorboard dir?
                                                histogram_freq=0,  # record the distribution of the weights
                                                write_graph=False,
                                                update_freq='epoch',
                                                profile_batch=0)
    # gradient_logger = callbacks.GradientLogger(filepath=str(gradient_filepath),
    #                                            train_input=stim_train,
    #                                            model=model)

    # save an initial set of weights before freezing, in case of termination before any checkpoints
    #log.info('saving weights to : %s', str(checkpoint_filepath) )
    model.save_weights(str(checkpoint_filepath), overwrite=True)

    if version.parse(tf.__version__)>=version.parse("2.2.0"):
        callback0 = [sparse_logger]
        verbose=0
    else:
        callback0 = []
        verbose = 2
    # enable the below to log tracked parameters to tensorboard
    if use_tensorboard:
        callback0.append(tensorboard)
        log.info(f'Enabling tensorboard, log: {str(tensorboard_filepath)}')
        # enable the below to record gradients to visualize in tensorboard; this is very slow,
        # and loading all this into tensorboard can use A LOT of memory
        # callback0.append(gradient_logger)

    if early_stopping_val_split > 0:
        callback0.append(early_stopping)
        log.info(f'Enabling early stopping, val split: {str(early_stopping_val_split)}')
    else:
        callback0.append(regular_stopping)
        log.info(f'Stop tolerance: min_delta={early_stopping_tolerance}')
                 
                 
    log.info(f'Fitting model (batch_size={batch_size})...')
    history = model.fit(
        train_data,
        resp_train,
        validation_split=early_stopping_val_split,
        verbose=verbose,
        epochs=max_iter,
        callbacks=callback0 + [
            nan_terminate,
            nan_weight_terminate,
            checkpoint,
        ],
        batch_size=batch_size
    )

    # did we terminate on a nan loss or weights? Load checkpoint if so
    if np.all(np.isnan(model.predict(train_data))) or model.early_terminated:  # TODO: should this be np.any()?
        log.warning('Model terminated on nan loss or weights, restoring saved weights.')
        try:
            # this can fail if it nans out before a single checkpoint gets saved, either because no saved weights
            # exist, or it tries to load a in different model from the init
            model.load_weights(str(checkpoint_filepath))
            log.warning('Reloaded previous saved weights after nan loss.')
        except (tf.errors.NotFoundError, ValueError):
            pass

    modelspec = tf2modelspec(model, modelspec)
    
    if truncate_model:
        log.info("Special case of freezing: restoring truncated model!!!")
        #modelspec_restored, rec_restored = modelspec_restore_input_layers(modelspec_trunc, rec_trunc, modelspec_original)
        modelspec_restored, est_restored = initializers.modelspec_restore_input_layers(modelspec, est, modelspec_original)
        est = est_original
        modelspec = modelspec_restored

    # debug: dump modelspec parameters
    #for i in range(len(modelspec)):
    #    log.info(modelspec.phi[i])
        
    contains_tf_only_layers = np.any(['tf_only' in m['fn'] for m in modelspec.modules])
    if not contains_tf_only_layers:
        # compare the predictions from the model and modelspec
        error = compare_ms_tf(modelspec, model, est, train_data)
        if error > 1e-5:
            log.warning(f'Mean difference between NEMS and TF model prediction: {error}')
        else:
            log.info(f'Mean difference between NEMS and TF model prediction: {error}')
    else:
        # nothing to compare, ms evaluation is not implemented for this type of model
        pass

    # add in some relevant meta information
    modelspec.meta['n_parms'] = len(modelspec.phi_vector)
    try:
        n_epochs = len(history.history['loss'])
        if 'val_loss' in history.history.keys():
            #val_stop = np.argmin(history.history['val_loss'])
            #loss = history.history['loss'][val_stop]
            loss = np.nanmin(history.history['val_loss'])
        else:
            loss = np.nanmin(history.history['loss'])
        
    except KeyError:
        n_epochs = 0
        loss = 0
    if modelspec.fit_count == 1:
        modelspec.meta['n_epochs'] = n_epochs
        modelspec.meta['loss'] = loss
    else:
        if modelspec.fit_index == 0:
            modelspec.meta['n_epochs'] = np.zeros(modelspec.fit_count)
            modelspec.meta['loss'] = np.zeros(modelspec.fit_count)
        modelspec.meta['n_epochs'][modelspec.fit_index] = n_epochs
        modelspec.meta['loss'][modelspec.fit_index] = loss
        
    try:
        max_iter = modelspec.meta['extra_results']
        modelspec.meta['extra_results'] = max(max_iter, n_epochs)
    except KeyError:
        modelspec.meta['extra_results'] = n_epochs

    nems0.utils.progress_fun()

    # clean up temp files
    if job_id is not None:
        log.info('removing temporary weights file(s)')
        shutil.rmtree(log_dir_base)

    return {'modelspec': modelspec}

def fit_tf_iterate(modelspec,
                   est: recording.Recording,
                   est_list: list,
                   early_stopping_tolerance: float = 5e-4,
                   max_iter: int = 10000,
                   extra_tail_fit: str = 'none',
                   proportional_iter: bool = True,
                   iters_per_loop: typing.Union[None, int] = None,
                   freeze_layers: typing.Union[None, list] = None,
                   IsReload: bool = False,
                   **context
                ) -> dict:
    """
    wrapper for fit_tf that can handle est_list and modelspec.cell_count>1

    :return:
    """
    if IsReload:
        return {}
    if iters_per_loop is None:
        iters_per_loop = int(max_iter/100)

    elif iters_per_loop == 0:
        iters_per_loop = max_iter
        freeze_layers = list(range(modelspec.shared_count))

    n_outer_loops = int(max_iter/iters_per_loop)
    freeze_layers2 = list(range(modelspec.shared_count))

    log.info(f"**** fit_tf_iterate, STARTING STAGE 1: iterate fits")
    outer_n = 0
    done = False
    previous_losses = np.ones(modelspec.cell_count)
    
    est_sizes = np.zeros(modelspec.cell_count)
    output_name=modelspec.meta['output_name']
    for i, e in enumerate(est_list):
        est_sizes[i] = np.prod(e[output_name].shape)
    #est_sizes += est_sizes.max()/20  # tweak ratios a bit to not completely skip rare sites
    est_sizes = np.cumsum(est_sizes/np.sum(est_sizes))
    
    while (outer_n < n_outer_loops) and (not done):
        epoch_counts = []
        losses = previous_losses.copy()
        if proportional_iter:
            for i in range(outer_n*modelspec.cell_count):
                _i = np.random.rand()
            cell_range = np.array([np.argmax(est_sizes>np.random.rand()) for i in range(modelspec.cell_count)])
        else:
            cell_range = np.arange(modelspec.cell_count)
            np.random.shuffle(cell_range)
        for i,cell_idx in enumerate(cell_range):
            
            log.info(f"**** fit_tf_iterate outer/inner {outer_n}/{i}: cell {cell_idx}      ****")
            log.info(f"***********************************************************************************")
            est = est_list[cell_idx]
            modelspec.cell_index = cell_idx

            #if (extra_tail_fit == 'pre') and ((i>0) or (outer_n>0)) and \
            if (extra_tail_fit == 'pre') and (outer_n > 1) and \
                ((freeze_layers is None) or (len(freeze_layers2) > len(freeze_layers))):
                log.info(f"**** pre-fitting non-shared layers {modelspec.shared_count}-{len(modelspec)}")
                modelspec = fit_tf(modelspec, est, max_iter=iters_per_loop*20,
                                   early_stopping_tolerance=early_stopping_tolerance*20,
                                   freeze_layers=freeze_layers2,
                                   **context)['modelspec']
                log.info(f"***********************************************************************************")
            elif extra_tail_fit == 'pre_reset' and ((i>0) or (outer_n>0)) and \
              ((freeze_layers is None) or (len(freeze_layers2)>len(freeze_layers))):
                log.info(f"**** resetting non-shared layers {modelspec.shared_count}-{len(modelspec)} to prior")
                for _modidx in range(modelspec.shared_count, len(modelspec)):
                    if 'phi' in modelspec[_modidx].keys():
                        for param_name,dist in modelspec[_modidx]['prior'].items():
                            modelspec[_modidx]['phi'][param_name] = dist[1]['mean']
                
                modelspec = fit_tf(modelspec, est, max_iter=max_iter,
                                   early_stopping_tolerance=early_stopping_tolerance*10,
                                   freeze_layers=freeze_layers2,
                                   **context)['modelspec']
                log.info(f"***********************************************************************************")

            #previous_losses[i] = modelspec.meta.get('loss',1)
            modelspec = fit_tf(modelspec, est, max_iter=iters_per_loop,
                               early_stopping_tolerance=early_stopping_tolerance,
                               freeze_layers=freeze_layers,
                               **context)['modelspec']

            epoch_counts.append(modelspec.meta['n_epochs'])
            losses[cell_idx] = modelspec.meta['loss']

            if (modelspec.shared_count > 0):
                log.info(f'copying first {modelspec.shared_count} mods from modelspec cell_index {modelspec.cell_index}')
                for cc in range(modelspec.cell_count):
                    if cc != cell_idx:
                        for _modidx in range(modelspec.shared_count):
                            if 'phi' in modelspec[_modidx].keys():
                                modelspec.raw[cc, modelspec.fit_index, modelspec.jack_index][_modidx]['phi'] = \
                                    copy.deepcopy(modelspec[_modidx]['phi'])
                        if extra_tail_fit == 'post':
                            modelspec.cell_index = cc
                            est = est_list[cc]
                            log.info(f'*** cc={cc}')
                            modelspec = fit_tf(modelspec, est, max_iter=iters_per_loop*4,
                                               early_stopping_tolerance=early_stopping_tolerance,
                                               freeze_layers=freeze_layers2,
                                               **context)['modelspec']
            log.info(f"**** outer/inner {outer_n}/{i}: cell {cell_idx} prev loss={previous_losses[cell_idx]:4f} new loss={losses[cell_idx]:4f}     ****")
            log.info(f"***********************************************************************************")

        
        mean_delta_loss = np.mean(previous_losses[cell_range]-losses[cell_range]) / iters_per_loop
        prev_loop_sum_str = ", ".join([f'{i}: {l:6.4f}' for i,l in enumerate(previous_losses)])
        loop_sum_str = ", ".join([f'{i}: {l:6.4f}' for i,l in enumerate(losses)])
        previous_losses = losses.copy()

        log.info(f"**** prev_loss={prev_loop_sum_str}")
        log.info(f"**** this_loss={loop_sum_str}")
        log.info(f"**** fit_tf_iterate, outer n={outer_n}, inner={np.max(epoch_counts)} complete, mean_delta_loss={mean_delta_loss:.6f}")
        log.info(f"**** ")

        outer_n += 1
        if mean_delta_loss < early_stopping_tolerance:
            done = True
        if np.max(epoch_counts) < 6:
            done = True

    log.info(f"***********************************************************************************")
    log.info(f"**** fit_tf_iterate, STARTING STAGE 2: fit branches")
    iters_per_loop2 = max_iter
    freeze_layers2 = list(range(modelspec.shared_count))
    for cell_idx in range(modelspec.cell_count):
        log.info(f"**** fit_tf_iterate, STAGE 2 fit_index={modelspec.fit_index} cell_index={cell_idx}")
        est = est_list[cell_idx]
        modelspec.cell_index = cell_idx

        modelspec = fit_tf(modelspec, est, max_iter=iters_per_loop2,
                           early_stopping_tolerance=early_stopping_tolerance,
                           freeze_layers=freeze_layers2,
                           **context)['modelspec']

    modelspec.cell_index = 0

    return {'modelspec': modelspec}


def fit_tf_init(
        modelspec,
        est: recording.Recording,
        est_list: typing.Union[None, list] = None,
        nl_init: str = 'tf',
        IsReload: bool = False,
        isolate_NL: bool = False,
        skip_init: bool = False,
        up_to_idx=None,
        **kwargs
        ) -> dict:
    """Inits a model using tf.

    Makes a new model up to the last relu or first levelshift, in the process setting levelshift to the mean of the
    resp. Excludes in the new model stp, rdt_gain, state_dc_gain, state_gain. Fits this. Then runs init_static_nl ,
    which looks at the last 2 layers of the original model, and if any of dexp, relu, log_sig, sat_rect are in those
    last two, only fits the first it encounters (freezes all other layers).
    """

    if IsReload or skip_init:
        return {}

    def first_substring_index(strings, substring):
        try:
            return next(i for i, string in enumerate(strings) if substring in string)
        except StopIteration:
            return None

    modelspec.cell_index = 0
    if est_list is None:
        est_list=[est]

    # find the first 'lvl' or last 'relu'
    ms_modules = [ms['fn'] for ms in modelspec]
    if up_to_idx is None:
        #up_to_idx = first_substring_index(ms_modules, 'levelshift')
        relu_idx = first_substring_index(reversed(ms_modules), 'relu')  #fit to last relu
        lvl_idx = first_substring_index(reversed(ms_modules), 'levelshift') #fit to last levelshift
        if lvl_idx is not None:
            relu_idx=None
        _idxs = [i for i in [relu_idx, lvl_idx, len(modelspec)-1] if i is not None]
        up_to_idx = len(modelspec) - 1 - np.min(_idxs)
        #last_idx = np.min([relu_idx, lvl_idx])
        #up_to_idx = len(modelspec) - 1 - up_to_idx

    log.info('up_to_idx=%d (%s)', up_to_idx, modelspec[up_to_idx]['fn'])

    # do the +1 here to avoid adding to None
    up_to_idx += 1
    # exclude the following from the init
    exclude = ['rdt_gain']  # , 'state_dc_gain', 'state_gain', 'sdexp']
    freeze = ['stp']

    # more complex version of first_substring_index: checks for not membership in init_static_nl_layers
    init_idxes = [idx for idx, ms in enumerate(ms_modules[:up_to_idx]) if not any(sub in ms for sub in exclude)]
    freeze_idxes = []

    # make a temp modelspec
    log.info('Creating temporary model for init')
    temp_ms = mslib.ModelSpec(cell_count=modelspec.cell_count)
    temp_ms.shared_count = modelspec.shared_count
    for cell_idx in range(modelspec.cell_count):
        modelspec.cell_index = cell_idx
        temp_ms.cell_index = cell_idx
        est = est_list[cell_idx]

        output_name = modelspec.meta.get('output_name', 'resp')
        try:
            mean_resp = np.nanmean(est[output_name].as_continuous(), axis=1, keepdims=True)
        except NotImplementedError:
            # as_continuous only available for RasterizedSignal
            mean_resp = np.nanmean(est[output_name].rasterize().as_continuous(), axis=1, keepdims=True)
        mean_added = False

        for idx in init_idxes:
            # TODO: handle 'merge_channels'
            ms = copy.deepcopy(modelspec[idx])
            log.info(f'{ms["fn"]}')

            # fix dc to mean_resp if present
            if (not mean_added) and (idx == init_idxes[-1]) and \
                    ('levelshift' in ms['fn']) and (len(ms['phi']['level']) == len(mean_resp)):
                log.info(f'Fixing "{ms["fn"]}" to: {mean_resp.flatten()[0]:.3f}')
                ms['phi']['level'][:] = mean_resp
                mean_added = True
            elif (not mean_added) and ('state_dc' in ms['fn']) and \
                    (ms['phi']['d'].shape[0] == len(mean_resp)):
                log.info(f'Fixing "{ms["fn"]}[d][:,0]" to mean_resp')
                ms['phi']['d'][:, [0]] = mean_resp
                mean_added = True

            temp_ms.append(ms)
            if any(fr in ms['fn'] for fr in freeze):
                freeze_idxes.append(len(temp_ms)-1)

    # important to reset cell_index to zero???
    est=est_list[0]
    temp_ms.cell_index = 0

    log.info('Running first init fit: model up to last lvl/relu without stp/gain.')
    log.debug('freeze_idxes: %s', freeze_idxes)
    filepath = Path(modelspec.meta['modelpath']) / 'init_part1'
    
    if 'freeze_layers' in kwargs.keys():
        force_freeze = kwargs.pop('freeze_layers')              # can't pass freeze_layers twice,
    else:
        force_freeze = None
    if force_freeze is not None:
        freeze_idxes = list(set(force_freeze + freeze_idxes))  # but also need to take union with freeze_idxes
    if modelspec.cell_count>1:
        temp_ms = fit_tf_iterate(temp_ms, est, est_list=est_list, freeze_layers=freeze_idxes, filepath=filepath, **kwargs)['modelspec']
    else:
        temp_ms = fit_tf(temp_ms, est, est_list=est_list, freeze_layers=freeze_idxes, filepath=filepath, **kwargs)['modelspec']

    for cell_idx in range(temp_ms.cell_count):
        log.info(f"***********************************************************************************")
        log.info(f"****   fit_tf_init, fit_index={modelspec.fit_index} cell_index={cell_idx} fitting output NL    ****")
        modelspec.cell_index = cell_idx
        temp_ms.cell_index = cell_idx
        est = est_list[cell_idx]

        # put back into original modelspec
        meta_save = modelspec.meta.copy()
        for ms_idx, temp_ms_module in zip(init_idxes, temp_ms):
            modelspec[ms_idx] = temp_ms_module

        if modelspec.fit_count > 1:
            if modelspec.fit_index == 0:
                meta_save['n_epochs'] = np.zeros(modelspec.fit_count)
                meta_save['loss'] = np.zeros(modelspec.fit_count)
            meta_save['n_epochs'][modelspec.fit_index] = temp_ms.meta['n_epochs']
            meta_save['loss'][modelspec.fit_index] = temp_ms.meta['loss']

        modelspec.meta.update(meta_save)

        if nl_init == 'skip':
            pass
            # to support multiple cell_indexes, don't return here,
            # wait til done looping instead
            #return {'modelspec': modelspec}

        elif nl_init == 'scipy':
            # pre-fit static NL if it exists
            _d = init_static_nl(est=est, modelspec=modelspec)
            modelspec = _d['modelspec']
            # TODO : Initialize relu in some intelligent way?

            log.info('finished fit_tf_init, fit_idx=%d/%d', modelspec.fit_index + 1, modelspec.fit_count)
            #return {'modelspec': modelspec}

        else:
            # init the static nl
            init_static_nl_mapping = {
                'double_exponential': initializers.init_dexp,
                'relu': None,
                'logistic_sigmoid': initializers.init_logsig,
                'saturated_rectifier': initializers.init_relsat,
            }
            # first find the first occurrence of a static nl in last two layers
            # if present, remove it from the idxes of the modules to freeze, init the nl and fit, and return the modelspec
            stage2_pending = True
            for idx, ms in enumerate(modelspec[-2:], len(modelspec)-2):
                for init_static_layer, init_fn in init_static_nl_mapping.items():
                    if stage2_pending and (init_static_layer in ms['fn']):
                        log.info(f'Initializing static nl "{ms["fn"]}" at layer #{idx}')
                        # relu has a custom init
                        if init_static_layer == 'relu':
                            ms['phi']['offset'][:] = -0.1
                        else:
                            modelspec = init_fn(est, modelspec, nl_mode=4)

                        if force_freeze is not None:
                            log.info(f'Running second init fit: force_freeze: {force_freeze}.')
                            static_nl_idx_not = force_freeze
                        elif isolate_NL:
                            log.info('Running second init fit: all frozen but static nl.')
                            static_nl_idx_not = list(set(range(len(modelspec))) - set([idx]))
                        elif modelspec.cell_count>1:
                            log.info(f'Titan model: freezing first {modelspec.shared_count} layers.')
                            static_nl_idx_not = list(range(modelspec.shared_count))
                        else:
                            log.info('Running second init fit: not frozen but coarser tolerance.')
                            static_nl_idx_not = []

                        # don't overwrite the phis in the modelspec
                        kwargs['use_modelspec_init'] = True
                        filepath = Path(modelspec.meta['modelpath']) / 'init_part2'
                        modelspec = fit_tf(modelspec, est, freeze_layers=static_nl_idx_not,
                                           filepath=filepath, **kwargs)['modelspec']
                        stage2_pending = False

    modelspec.cell_index = 0
    return {'modelspec': modelspec}


def eval_tf_layer(data: np.ndarray,
                  layer_spec: typing.Union[None, str] = None,
                  state_data: np.ndarray = None,
                  stop: typing.Union[None, int] = None,
                  modelspec: mslib.ModelSpec = None,
                  **kwargs,  # temporary until state data is implemented
                  ) -> np.ndarray:
    """Takes in a numpy array and applies a single tf layer to it.

    :param data: The input data. Shape of (reps, time, channels).
    :param layer_spec: A layer spec for layers of a modelspec.
    :param state_data: The input state data. Shape of (reps, time, channels).
    :param stop: What layer to eval to. Non inclusive. If not passed, will evaluate the whole layer spec.
    :param modelspec: Optionally use an existing modelspec. Takes precedence over layer_spec.

    :return: The processed data.
    """
    if layer_spec is None and modelspec is None:
        raise ValueError('Either of "layer_spec" or "modelspec" must be specified.')

    if modelspec is not None:
        ms = modelspec
    else:
        ms = initializers.from_keywords(layer_spec)

    layers = modelbuilder.modelspec2tf(ms, use_modelspec_init=True)[:stop]

    data_shape = data.shape
    state_shape = None
    if state_data is not None:
        state_shape = state_data.shape
        data = [data, state_data]

    model = modelbuilder.ModelBuilder(layers=layers).build_model(input_shape=data_shape, state_shape=state_shape)

    pred = model.predict(data)
    return pred


# TODO: move this into tf.utils and whatever else needs to go there too
@tf.function
def get_jacobian(model: tf.keras.Model,
                 tensor: tf.Tensor,
                 index: int,
                 out_channel: int = 0
                 ) -> np.array:
    """
    Gets the jacobian at the given index.
    
    This needs to be a tf.function for a huge speed increase.
    """
    
    out_channel=tf.cast(out_channel,tf.int32)
    with tf.GradientTape(persistent=True) as g:
        g.watch(tensor)
        z = model(tensor)[0, index, out_channel]

    return g.jacobian(z, tensor)
