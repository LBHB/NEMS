import tensorflow as tf
import logging

from nems0.tf import loss_functions
from nems0.utils import lookup_fn_at

log = logging.getLogger(__name__)


class ModelBuilder:
    """Basic wrapper to build tensorflow keras models.

    Not an actual subclass of tf.keras.Model, but instead a set of helper
    methods to prepare layers and then return a model when ready.
    """
    def __init__(self,
                 name='model',
                 layers=None,
                 learning_rate=1e-3,
                 optimizer='adam',
                 loss_fn=loss_functions.loss_se,
                 metrics=None,
                 *args,
                 **kwargs
                 ):
        """Layers don't need to be added at init, only prior to fitting.

        :param name:
        :param args:
        :param kwargs:
        """
        self.optimizer_dict = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
        }

        if layers is not None:
            self.model_layers = layers
        else:
            self.model_layers = []

        self.name = name
        self.loss_fn = loss_fn
        self.metrics = metrics

        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def add_layer(self, layer, idx=None):
        """Adds a layer to the list of layers that will be part of the model.

        :param layer: Layer to add.
        :param idx: Insertion index.
        """
        if idx is None:
            self.model_layers.append(layer)
            log.debug(f'Added "{layer.ms_name}" to end of model.')
        else:
            self.model_layers.insert(idx, layer)
            log.debug(f'Inserted "{layer.ms_name}" to idx "{idx}".')

    def build_model(self, input_shape, state_shape=None, batch_size=None):
        """Creates a tf.keras.Model instance. Simple single headed model.

        :param input_shape: Allows summary creation. Shape should be a tuple of (batch, time, channel).
        :param state_shape: Same as input_shape, but for state data. Shape should be a tuple of (batch, time, channel).
        :param batch_size: Size of batch.

        :return: tf.keras.Model instance.
        """
        stim_input = tf.keras.Input(shape=input_shape[1:], name='stim', batch_size=batch_size, dtype='float32')
        if state_shape is not None:
            state_input = tf.keras.Input(shape=state_shape[1:], name='state', batch_size=batch_size,
                                         dtype='float32')
            inputs = [stim_input, state_input]
        else:
            inputs = stim_input

        # TODO: make this work with state as the first layer
        outputs = stim_input
        for layer in self.model_layers:
            if layer._STATE_LAYER:
                outputs = layer([outputs, state_input])
            else:
                outputs = layer(outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        log.info('Built model, printing summary.')
        model.summary(print_fn=log.info)
        # for help understanding "Connected to" column with multi-input models: https://stackoverflow.com/a/53944525

        # build the optimizer
        try:
            optimizer = self.optimizer_dict[self.optimizer](learning_rate=self.learning_rate, clipnorm=1.)
            log.debug(f"optimizer {self.optimizer}: learning_rate={self.learning_rate}, clipnorm={1.}")
        except KeyError:
            optimizer = self.optimizer

        model.compile(optimizer, loss=self.loss_fn, metrics=self.metrics)

        return model

    
def modelspec2tf(modelspec, seed=0, use_modelspec_init=True, fs=100, 
                 initializer='random_normal',
                 freeze_layers=None, kernel_regularizer=None):
    """
    Was in nems0.modelspec, but this is so tightly coupled to tf libraries that
    it probably belongs here instead.

    """
    if kernel_regularizer is not None:
        regstr = kernel_regularizer.split(":")
        kernel_regularizer_ops = {}
        if len(regstr) > 2:
            if regstr[2] == 'firwc':
                kernel_regularizer_ops['modulenames'] = ['weight_channels.basic', 'filter_bank', 'fir.basic']
        else:
            kernel_regularizer_ops['modulenames'] = [
                'weight_channels.basic', 'Conv2D', 'WeightChannelsNew', 'state_dc_gain']


    layers = []
    if freeze_layers is None:
        freeze_layers = []

    for i, m in enumerate(modelspec):
        try:
            tf_layer = lookup_fn_at(m['tf_layer'])
        except KeyError:
            raise NotImplementedError(f'Layer "{m["fn"]}" does not have a tf equivalent.')

        if i in freeze_layers:
            trainable = False
        else:
            trainable = True

        if (kernel_regularizer is not None) and \
            any(fn in m['fn']  for fn in kernel_regularizer_ops['modulenames']):
                log.info(f"Including {kernel_regularizer} regularizer for {m['fn']}")
                layer = tf_layer.from_ms_layer(m, use_modelspec_init=use_modelspec_init, seed=seed, fs=fs,
                                               initializer=initializer, trainable=trainable,
                                               kernel_regularizer=kernel_regularizer)
        else:
            # don't pass kernel_regularizer (set to None) if not fir or weight chans
            layer = tf_layer.from_ms_layer(m, use_modelspec_init=use_modelspec_init, seed=seed, fs=fs,
                                           initializer=initializer, trainable=trainable)
        if ('Conv2D' in m['fn']):
            if 'offset' in m['phi'].keys():
                log.debug(f"Conv2D initializing offset: {m['phi']['offset']}")
            else:
                log.debug("Conv2D initializing offset pseudo-randomly")
        layers.append(layer)

    return layers
