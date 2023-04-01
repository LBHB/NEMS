import logging

import numpy as np
from functools import partial

import tensorflow as tf
from tensorflow import config
from tensorflow.keras.layers import Conv2D, Dense, Dropout
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.keras.constraints import Constraint
from tensorflow.python.ops import array_ops, nn_ops, nn
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


log = logging.getLogger(__name__)


class BaseLayer(tf.keras.layers.Layer):
    """Base layer with parent methods for converting from modelspec layer to tf layers and back."""

    _TF_ONLY = False  # set true in subclasses for which there is no mapping to NEMS modules
    _STATE_LAYER = False  # set true in subclasses that need to also accept state layer

    def __init__(self, *args, **kwargs):
        """Catch args/kwargs that aren't allowed kwargs of keras.layers.Layers"""
        self.fs = kwargs.pop('fs', None)
        self.ms_name = kwargs.pop('ms_name', None)
        kr = kwargs.pop('kernel_regularizer', None)
        if kr is not None:
            log.info(f"kernel regularizer: {kr}")
            #import pdb; pdb.set_trace()

        self.kernel_regularizer = kr

        # Overwrite in other layers' __init__ to change which fn_kwargs get copied
        # (see self.copy_valid_fn_kwargs)

        super(BaseLayer, self).__init__(*args, **kwargs)

    @classmethod
    def from_ms_layer(cls,
                      ms_layer,
                      use_modelspec_init: bool = True,
                      seed: int = 0,
                      fs: int = 100,
                      initializer: str = 'random_normal',
                      kernel_regularizer: str = None,
                      trainable: bool = True,
                      ):
        """Parses modelspec layer to generate layer class.

        :param ms_layer: A layer of a modelspec.
        :param use_modelspec_init: Whether to use the modelspec's initialization or use a tf builtin.
        :param fs: The seed for the inits if not using modelspec init.
        :param fs: The sampling rate of the data.
        :param initializer: What initializer to use. Only used if use_modelspec_init is False.
        """
        log.debug(f'Building tf layer for "{ms_layer["fn"]}".')

        kwargs = {
            'ms_name': ms_layer['fn'],
            'fs': fs,
            'seed': seed,
            'trainable': trainable,
        }

        if kernel_regularizer is not None:
            regstr = kernel_regularizer.split(":")
            kernel_regularizer = regstr[0]
            if len(regstr)>1:
                parm = 10**(-int(regstr[1]))
            else:
                parm = 0.001
            if kernel_regularizer.lower() == 'l2':
                log.info(f'Setting kernel_regularizer to {kernel_regularizer} (l={parm})')
                kwargs['kernel_regularizer'] = regularizers.l2(l=parm)
            else:
                raise ValueError(f"Need to add support for regularizer {kernel_regularizer}")
                #kwargs['kernel_regularizer'] = regularizers.get(kernel_regularizer)(l2=0.001)

        # TODO: clean this up, maybe separate kwargs/fn_kwargs, or method to split out valid tf kwargs from rest
        if 'bounds' in ms_layer:
            kwargs['bounds'] = ms_layer['bounds']
        if 'chans' in ms_layer['fn_kwargs']:
            kwargs['units'] = ms_layer['fn_kwargs']['chans']
        if 'bank_count' in ms_layer['fn_kwargs']:
            kwargs['banks'] = ms_layer['fn_kwargs']['bank_count']
        if 'reset_signal' in ms_layer['fn_kwargs']:
            # kwargs['reset_signal'] = ms_layer['fn_kwargs']['reset_signal']
            kwargs['reset_signal'] = None
        if 'non_causal' in ms_layer['fn_kwargs']:
            kwargs['non_causal'] = ms_layer['fn_kwargs']['non_causal']
        if 'var_offset' in ms_layer['fn_kwargs']:
            kwargs['var_offset'] = ms_layer['fn_kwargs']['var_offset']
        if 'state_type' in ms_layer['fn_kwargs']:
            kwargs['state_type'] = ms_layer['fn_kwargs']['state_type']
        if 'exclude_chans' in ms_layer['fn_kwargs']:
            kwargs['exclude_chans'] = ms_layer['fn_kwargs']['exclude_chans']
        if 'per_channel' in ms_layer['fn_kwargs']:
            kwargs['per_channel'] = ms_layer['fn_kwargs']['per_channel']

        # TODO: this approach could cause issues if there are name clashes with NEMS kwargs
        pass_through_keys = ['n_inputs', 'crosstalk', 'filters', 'kernel_size',
                             'activation', 'units', 'padding', 'rate']
        pass_throughs = {k: v for k, v in ms_layer['fn_kwargs'].items() if k in pass_through_keys}
        kwargs.update(pass_throughs)

        if not cls._TF_ONLY:  # TODO: implement proper phi formatting for TF_ONLY layers so that this isn't necessary
            #import pdb; pdb.set_trace()
            if use_modelspec_init:
                # convert the phis to tf constants
                if ms_layer.get('phi',None):
                    kwargs['initializer'] = {k: tf.constant_initializer(v)
                                             for k, v in ms_layer['phi'].items()}
                else:
                    kwargs['initializer'] = {}
                if 'WeightChannelsGaussian' in ms_layer['tf_layer']:
                    # Per SVD: kludge to get TF optimizer to play nice with sd parameter,
                    kwargs['initializer']['sd'] = tf.constant_initializer(ms_layer['phi']['sd'] * 10)
            else:
                # if want custom inits for each layer, remove this and change the inits in each layer
                # kwargs['initializer'] = {k: 'truncated_normal' for k in ms_layer['phi'].keys()}
                kwargs['initializer'] = {k: initializer for k in ms_layer['phi'].keys()}

            instance = cls(**kwargs)
        else:
            # skip init step, not currently implemented.
            # Instead, directly copy
            if len(ms_layer['phi']) > 0:
                # Phi empty until model has been fit
                weights = [v for k, v in ms_layer['phi'].items()]
                instance = cls(weights=weights, **kwargs)
            else:
                instance = cls(**kwargs)

        return instance

    @property
    def layer_values(self):
        """Returns key value pairs of the weight names and their values."""
        weight_names = [
            # extract the actual layer name from the tf layer naming format
            # ex: "nems0.modules.nonlinearity.double_exponential/base:0"
            weight.name.split('/')[1].split(':')[0]
            for weight in self.weights
        ]

        layer_values = {layer_name: weight.numpy() for layer_name, weight in zip(weight_names, self.weights)}
        return layer_values

    def weights_to_phi(self):
        """In subclass, use self.weight_dict to get a dict of weight_name: weights."""
        raise NotImplementedError


class Dlog(BaseLayer):
    """Simple dlog nonlinearity."""
    def __init__(self,
                 units=None,
                 initializer=None,
                 seed=0,
                 var_offset=True,
                 *args,
                 **kwargs,
                 ):
        super(Dlog, self).__init__(*args, **kwargs)

        # try to infer the number of units if not specified
        if units is None and initializer is None:
            self.units = 1
        elif units is None:
            self.units = initializer['offset'].value.shape[1]
        else:
            self.units = units

        self.var_offset = var_offset

        if self.var_offset:
            self.initializer = {'offset': tf.random_normal_initializer(seed=seed)}
            if initializer is not None:
                self.initializer.update(initializer)
        else:
            self.initializer = None
            self.fixed_offset = -1.0

    def build(self, input_shape):
        if self.var_offset:
            self.offset = self.add_weight(name='offset',
                                          shape=(self.units,),
                                          dtype='float32',
                                          initializer=self.initializer['offset'],
                                          trainable=True,
                                          )

    def call(self, inputs, training=True):
        if self.var_offset:
            return tf.nn.relu(inputs - self.offset)

    def call(self, inputs, training=True):
        # clip bounds at ±2 to avoid huge compression/expansion
        if self.var_offset:
            eb = tf.math.pow(tf.constant(10, dtype='float32'), tf.clip_by_value(self.offset, -2, 2))
        else:
            eb = tf.constant(np.power(10, self.fixed_offset), dtype='float32')

        return tf.math.log((inputs + eb) / eb)

    def weights_to_phi(self):
        layer_values = self.layer_values
        if self.var_offset:
            layer_values['offset'] = layer_values['offset'].reshape((-1, 1))
            log.debug(f'Converted {self.name} to modelspec phis.')
        return layer_values


class Levelshift(BaseLayer):
    """Simple levelshift nonlinearity."""
    def __init__(self,
                 units=None,
                 initializer=None,
                 seed=0,
                 *args,
                 **kwargs,
                 ):
        super(Levelshift, self).__init__(*args, **kwargs)

        # try to infer the number of units if not specified
        if units is None and initializer is None:
            self.units = 1
        elif units is None:
            self.units = initializer['level'].value.shape[1]
        else:
            self.units = units

        self.initializer = {'level': tf.random_normal_initializer(seed=seed)}
        if initializer is not None:
            self.initializer.update(initializer)

    def build(self, input_shape):
        self.level = self.add_weight(name='level',
                                     shape=(self.units,),
                                     dtype='float32',
                                     initializer=self.initializer['level'],
                                     trainable=True,
                                     )

    def call(self, inputs, training=True):
        return tf.identity(inputs + self.level)

    def weights_to_phi(self):
        layer_values = self.layer_values
        layer_values['level'] = layer_values['level'].reshape((-1, 1))
        log.debug(f'Converted {self.name} to modelspec phis.')
        return layer_values


class Relu(BaseLayer):
    """Simple relu nonlinearity."""
    def __init__(self,
                 initializer=None,
                 seed=0,
                 var_offset=True,
                 *args,
                 **kwargs,
                 ):
        super(Relu, self).__init__(*args, **kwargs)

        self.var_offset = var_offset
        if self.var_offset:
            self.initializer = {'offset': tf.random_normal_initializer(seed=None)}
            if initializer is not None:
                self.initializer.update(initializer)
        else:
            self.initializer = None

    def build(self, input_shape):
        if self.var_offset:
            self.offset = self.add_weight(name='offset',
                                          shape=(input_shape[-1],),
                                          dtype='float32',
                                          initializer=self.initializer['offset'],
                                          trainable=True,
                                          )

    def call(self, inputs, training=True):
        if self.var_offset:
            return tf.nn.relu(inputs - self.offset)
        else:
            return tf.nn.relu(inputs)

    def weights_to_phi(self):
        layer_values = self.layer_values
        if self.var_offset:
            layer_values['offset'] = layer_values['offset'].reshape((-1, 1))
            log.debug(f'Converted {self.name} to modelspec phis.')
        return layer_values


class DoubleExponential(BaseLayer):
    """Basic double exponential nonlinearity."""
    def __init__(self,
                 units=None,
                 initializer=None,
                 seed=0,
                 *args,
                 **kwargs,
                 ):
        super(DoubleExponential, self).__init__(*args, **kwargs)

        # try to infer the number of units if not specified
        if units is None and initializer is None:
            self.units = 1
        elif units is None:
            self.units = initializer['base'].value.shape[1]
        else:
            self.units = units

        self.initializer = {
                'base': tf.random_normal_initializer(seed=seed),
                'amplitude': tf.random_normal_initializer(seed=seed + 1),
                'shift': tf.random_normal_initializer(seed=seed + 2),
                'kappa': tf.random_normal_initializer(seed=seed + 3),
            }

        if initializer is not None:
            self.initializer.update(initializer)

    def build(self, input_shape):
        self.base = self.add_weight(name='base',
                                    shape=(self.units,),
                                    dtype='float32',
                                    initializer=self.initializer['base'],
                                    trainable=True,
                                    )
        self.amplitude = self.add_weight(name='amplitude',
                                         shape=(self.units,),
                                         dtype='float32',
                                         initializer=self.initializer['amplitude'],
                                         trainable=True,
                                         )
        self.shift = self.add_weight(name='shift',
                                     shape=(self.units,),
                                     dtype='float32',
                                     initializer=self.initializer['shift'],
                                     trainable=True,
                                     )
        self.kappa = self.add_weight(name='kappa',
                                     shape=(self.units,),
                                     dtype='float32',
                                     initializer=self.initializer['kappa'],
                                     trainable=True,
                                     )

    def call(self, inputs, training=True):
        # formula: base + amp * e^(-e^(-e^kappa * (inputs - shift)))
        return self.base + self.amplitude * tf.math.exp(-tf.math.exp(-tf.math.exp(self.kappa) * (inputs - self.shift)))

    def weights_to_phi(self):
        layer_values = self.layer_values

        # if self.units == 1:
        #     shape = (1,)
        # else:
        #     shape = (-1, 1)
        shape = (-1, 1)

        layer_values['amplitude'] = layer_values['amplitude'].reshape(shape)
        layer_values['base'] = layer_values['base'].reshape(shape)
        layer_values['kappa'] = layer_values['kappa'].reshape(shape)
        layer_values['shift'] = layer_values['shift'].reshape(shape)

        log.debug(f'Converted {self.name} to modelspec phis.')
        return layer_values


class WeightChannelsBasic(BaseLayer):
    """Basic weight channels."""
    def __init__(self,
                 # kind='basic',
                 units=None,
                 initializer=None,
                 seed=0,
                 *args,
                 **kwargs,
                 ):
        super(WeightChannelsBasic, self).__init__(*args, **kwargs)

        # try to infer the number of units if not specified
        if units is None and initializer is None:
            self.units = 1
        elif units is None:
            self.units = initializer['coefficients'].value.shape[1]
        else:
            self.units = units

        self.initializer = {'coefficients': tf.random_normal_initializer(seed=seed)}
        if initializer is not None:
            self.initializer.update(initializer)

    def build(self, input_shape):
        self.coefficients = self.add_weight(name='coefficients',
                                            shape=(self.units, input_shape[-1]),
                                            dtype='float32',
                                            initializer=self.initializer['coefficients'],
                                            trainable=True,
                                            regularizer=self.kernel_regularizer,
                                            )

    def call(self, inputs, training=True):
        transposed = tf.transpose(self.coefficients)
        return tf.nn.conv1d(inputs, tf.expand_dims(transposed, 0), stride=1, padding='SAME')

    def weights_to_phi(self):
        layer_values = self.layer_values
        log.debug(f'Converted {self.name} to modelspec phis.')
        return layer_values


class WeightChannelsGaussian(BaseLayer):
    """Basic weight channels."""
    # TODO: convert per https://stackoverflow.com/a/52012658/1510542 in order to handle banks
    def __init__(self,
                 # kind='gaussian',
                 bounds,
                 units=None,
                 initializer=None,
                 seed=0,
                 *args,
                 **kwargs,
                 ):
        super(WeightChannelsGaussian, self).__init__(*args, **kwargs)

        # try to infer the number of units if not specified
        if units is None and initializer is None:
            self.units = 1
        elif units is None:
            self.units = initializer['mean'].value.shape[0]
        else:
            self.units = units

        self.initializer = {
                'mean': tf.random_normal_initializer(seed=seed),
                'sd': tf.random_normal_initializer(seed=seed + 1),  # this is halfnorm in NEMS
            }

        if initializer is not None:
            self.initializer.update(initializer)

        # constraints assumes bounds built with np.full
        self.mean_constraint = lambda t: tf.clip_by_value(t, bounds['mean'][0][0], bounds['mean'][1][0])
        self.sd_constraint = lambda t: tf.clip_by_value(t, bounds['sd'][0][0], bounds['sd'][1][0] * 10)

    def build(self, input_shape):
        self.mean = self.add_weight(name='mean',
                                    shape=(self.units,),
                                    dtype='float32',
                                    initializer=self.initializer['mean'],
                                    constraint=self.mean_constraint,
                                    trainable=True,
                                    regularizer=self.kernel_regularizer,
                                    )
        self.sd = self.add_weight(name='sd',
                                  shape=(self.units,),
                                  dtype='float32',
                                  initializer=self.initializer['sd'],
                                  constraint=self.sd_constraint,
                                  trainable=True,
                                  regularizer=self.kernel_regularizer,
                                  )

    def call(self, inputs, training=True):
        input_features = tf.cast(tf.shape(inputs)[-1], dtype='float32')
        temp = tf.range(input_features) / input_features
        temp = (tf.reshape(temp, [1, input_features, 1]) - self.mean) / (self.sd/10)
        temp = tf.math.exp(-0.5 * tf.math.square(temp))
        kernel = temp / tf.math.reduce_sum(temp, axis=1)

        return tf.nn.conv1d(inputs, kernel, stride=1, padding='SAME')

    def weights_to_phi(self):
        layer_values = self.layer_values
        layer_values['sd'] = layer_values['sd'] / 10  # reverses *10 kludge in initialization
        # don't need to do any reshaping
        log.debug(f'Converted {self.name} to modelspec phis.')
        return layer_values


class FIR(BaseLayer):
    """Basic FIR filter."""
    # TODO: organize params
    def __init__(self,
                 units=None,
                 banks=1,
                 n_inputs=1,
                 initializer=None,
                 seed=0,
                 non_causal=0,
                 *args,
                 **kwargs,
                 ):
        super(FIR, self).__init__(*args, **kwargs)

        # try to infer the number of units if not specified
        if units is None and initializer is None:
            self.units = 1
        elif units is None:
            self.units = initializer['coefficients'].value.shape[1]
        else:
            self.units = units

        self.banks = banks
        self.n_inputs = n_inputs
        self.non_causal = non_causal
        if non_causal >= units:
            raise ValueError("FIR: non_causal bin count must be < filter length (units)")

        self.initializer = {'coefficients': tf.random_normal_initializer(seed=seed)}
        if initializer is not None:
            self.initializer.update(initializer)

    def build(self, input_shape):
        """Adds some logic to handle depthwise convolution shapes."""
        if self.banks == 1 or input_shape[-1] != self.banks * self.n_inputs:
            shape = (self.banks, input_shape[-1], self.units)

        else:
            shape = (self.banks, self.n_inputs, self.units)

        self.coefficients = self.add_weight(name='coefficients',
                                            shape=shape,
                                            dtype='float32',
                                            initializer=self.initializer['coefficients'],
                                            trainable=True,
                                            regularizer=self.kernel_regularizer,
                                            )

    def _call(self, inputs, training=True):
        """Normal call."""
        pad_size0 = self.units - 1 - self.non_causal
        pad_size1 = self.non_causal
        padded_input = tf.pad(inputs, [[0, 0], [pad_size0, pad_size1], [0, 0]])
        transposed = tf.transpose(tf.reverse(self.coefficients, axis=[-1]))
        Y = tf.nn.conv1d(padded_input, transposed, stride=1, padding='VALID')
        return Y

    def call(self, inputs, training=True):
        """Normal call."""
        pad_size0 = self.units - 1 - self.non_causal
        pad_size1 = self.non_causal
        padded_input = tf.pad(inputs, [[0, 0], [pad_size0, pad_size1], [0, 0]])

        # SVD inserting
        DEBUG = False
        if DEBUG:
            print("Banks:", self.banks)
            print("Units:", self.units)
            print("N_inputs: ", self.n_inputs)
        if config.list_physical_devices('GPU') or \
                (self.n_inputs == padded_input.shape[-1]):
            # this implementation does not evaluate on a CPU if mapping subsets of
            # the input into the different FIR filters.
            transposed = tf.transpose(tf.reverse(self.coefficients, axis=[-1]))
            Y = tf.nn.conv1d(padded_input, transposed, stride=1, padding='VALID')
            if DEBUG:
                print("padded input: ", padded_input.shape)
                print("transposed kernel: ", transposed.shape)
                print("Y: ", Y.shape)
            return Y

        # alternative, kludgy implementation, evaluate each filter separately on the
        # corresponding subset of inputs, concatenate outputs when done
        transposed = tf.transpose(tf.reverse(self.coefficients, axis=[-1]))
        if DEBUG:
            print("padded input: ", padded_input.shape)
            print("transposed kernel: ", transposed.shape)

        L = []
        for i in range(transposed.shape[2]):
            W = transposed.shape[1]
            A = padded_input[:, :, (i*W):((i+1)*W)]
            B = transposed[:, :, i:(i+1)]
            L.append(tf.nn.conv1d(A, B, stride=1, padding='VALID'))
        Y = tf.concat(L, axis=2)
        if DEBUG:
            print("L[0]: ", L[0].shape)
            print("Y: ", Y.shape)

        return Y
        # SVD stopped inserting
        #return tf.nn.conv1d(padded_input, transposed, stride=1, padding='VALID')

    def weights_to_phi(self):
        layer_values = self.layer_values
        layer_values['coefficients'] = layer_values['coefficients'].reshape((-1, self.units))
        # don't need to do any reshaping
        log.debug(f'Converted {self.name} to modelspec phis.')
        return layer_values


# class DampedOscillator(BaseLayer):
#     """Basic weight channels."""
#     # TODO: organize params
#     def __init__(self,
#                  bounds,
#                  units=None,
#                  banks=1,
#                  n_inputs=1,
#                  initializer=None,
#                  seed=0,
#                  *args,
#                  **kwargs,
#                  ):
#         super(FIR, self).__init__(*args, **kwargs)
#
#         # try to infer the number of units if not specified
#         if units is None and initializer is None:
#             self.units = 1
#         elif units is None:
#             self.units = initializer['f1'].value.shape[1]
#         else:
#             self.units = units
#
#         self.banks = banks
#         self.n_inputs = n_inputs
#
#         self.initializer = {
#                 'f1': tf.random_normal_initializer(seed=seed),
#                 'tau': tf.random_normal_initializer(seed=seed + 1),
#                 'delay': tf.random_normal_initializer(seed=seed + 2),
#                 'gain': tf.random_normal_initializer(seed=seed + 3),
#             }
#
#         if initializer is not None:
#             self.initializer.update(initializer)
#
#         # constraints assumes bounds build with np.full
#         self.f1_constraint = tf.keras.constraints.MinMaxNorm(
#             min_value=bounds['f1s'][0][0],
#             max_value=bounds['f1s'][1][0])
#         self.tau_constraint = tf.keras.constraints.MinMaxNorm(
#             min_value=bounds['taus'][0][0],
#             max_value=bounds['taus'][1][0])
#         self.delay_constraint = tf.keras.constraints.MinMaxNorm(
#             min_value=bounds['delays'][0][0],
#             max_value=bounds['delays'][1][0])
#         self.gain_constraint = tf.keras.constraints.MinMaxNorm(
#             min_value=bounds['gains'][0][0],
#             max_value=bounds['gains'][1][0])
#
#     def build(self, input_shape):
#         """Adds some logic to handle depthwise convolution shapes."""
#         if self.banks == 1 or input_shape[-1] != self.banks * self.n_inputs:
#             shape = (self.banks, input_shape[-1], self.units)
#
#         else:
#             shape = (self.banks, self.n_inputs, self.units)
#
#         self.f1 = self.add_weight(name='f1',
#                                   shape=shape,
#                                   dtype='float32',
#                                   initializer=self.initializer['f1'],
#                                   trainable=True,
#                                   )
#         self.tau = self.add_weight(name='tau',
#                                    shape=shape,
#                                    dtype='float32',
#                                    initializer=self.initializer['tau'],
#                                    trainable=True,
#                                    )
#         self.delay = self.add_weight(name='delay',
#                                      shape=shape,
#                                      dtype='float32',
#                                      initializer=self.initializer['delay'],
#                                      trainable=True,
#                                      )
#         self.gain = self.add_weight(name='gain',
#                                     shape=shape,
#                                     dtype='float32',
#                                     initializer=self.initializer['gain'],
#                                     trainable=True,
#                                     )
#
#     def call(self, inputs, training=True):
#         """Normal call."""
#         pad_size = self.units - 1
#         padded_input = tf.pad(inputs, [[0, 0], [pad_size, 0], [0, 0]])
#         transposed = tf.transpose(tf.reverse(self.coefficients, axis=[-1]))
#         return tf.nn.conv1d(padded_input, transposed, stride=1, padding='VALID')
#
#     def weights_to_phi(self):
#         layer_values = self.layer_values
#         layer_values['coefficients'] = layer_values['coefficients'].reshape((-1, self.units))
#         # don't need to do any reshaping
#         log.info(f'Converted {self.name} to modelspec phis.')
#         return layer_values


class STPQuick(BaseLayer):
    """Quick version of STP."""
    def __init__(self,
                 bounds,
                 crosstalk=False,
                 reset_signal=None,
                 units=None,
                 initializer=None,
                 seed=0,
                 *args,
                 **kwargs,
                 ):
        super(STPQuick, self).__init__(*args, **kwargs)

        self.crosstalk = crosstalk
        self.reset_signal = reset_signal

        # try to infer the number of units if not specified
        if units is None and initializer is None:
            self.units = 1
        elif units is None:
            self.units = initializer['u'].value.shape[1]
        else:
            self.units = units

        self.initializer = {
                'u': tf.random_normal_initializer(seed=seed),
                'tau': tf.random_normal_initializer(seed=seed + 1),
                'x0': None
            }

        if initializer is not None:
            self.initializer.update(initializer)

        self.u_constraint = lambda u: tf.abs(u)
        # max_tau = tf.maximum(tf.abs(self.initializer['tau'](self.units)), 2.001 / self.fs)
        # self.tau_constraint = tf.keras.constraints.MaxNorm(max_value=2.001 / self.fs)
        # self.tau_constraint = tf.keras.constraints.NonNeg()
        self.tau_constraint = lambda tau: tf.maximum(tf.abs(tau), 2.001 / self.fs)

    def build(self, input_shape):
        self.u = self.add_weight(name='u',
                                 shape=(self.units,),
                                 dtype='float32',
                                 initializer=self.initializer['u'],
                                 constraint=self.u_constraint,
                                 trainable=True,
                                 )
        self.tau = self.add_weight(name='tau',
                                   shape=(self.units,),
                                   dtype='float32',
                                   initializer=self.initializer['tau'],
                                   constraint=self.tau_constraint,
                                   trainable=True,
                                   )
        if self.initializer['x0'] is None:
            self.x0 = None
        else:
            self.x0 = self.add_weight(name='x0',
                                      shape=(self.units,),
                                      dtype='float32',
                                      initializer=self.initializer['x0'],
                                      trainable=True,
                                      )

    def call(self, inputs, training=True):
        _zero = tf.constant(0.0, dtype='float32')
        _nan = tf.constant(0.0, dtype='float32')

        s = inputs.shape
        tstim = tf.where(tf.math.is_nan(inputs), _zero, inputs)

        if self.x0 is not None:  # x0 should be tf variable to avoid retraces
            # TODO: is this expanding along the right dim? tstim dims: (None, time, chans)
            tstim = tstim - tf.expand_dims(self.x0, axis=1)

        # convert a & tau units from sec to bins
        ui = tf.math.abs(tf.reshape(self.u, (1, -1))) / self.fs * 100
        taui = tf.math.abs(tf.reshape(self.tau, (1, -1))) * self.fs

        # convert chunksize from sec to bins
        chunksize = 5
        chunksize = int(chunksize * self.fs)

        if self.crosstalk:
            # assumes dim of u is 1 !
            tstim = tf.math.reduce_mean(tstim, axis=0, keepdims=True)

        ui = tf.expand_dims(ui, axis=0)
        taui = tf.expand_dims(taui, axis=0)

        @tf.function
        def _cumtrapz(x, dx=1., initial=0.):
            x = (x[:, :-1] + x[:, 1:]) / 2.0
            x = tf.pad(x, ((0, 0), (1, 0), (0, 0)), constant_values=initial)
            return tf.cumsum(x, axis=1) * dx

        a = tf.cast(1.0 / taui, 'float64')
        x = ui * tstim

        if self.reset_signal is None:
            reset_times = tf.range(0, s[1] + chunksize - 1, chunksize)
        else:
            reset_times = tf.where(self.reset_signal[0, :])[:, 0]
            reset_times = tf.pad(reset_times, ((0, 1),), constant_values=s[1])

        td = []
        x0, imu0 = 0.0, 0.0
        for j in range(reset_times.shape[0] - 1):
            xi = tf.cast(x[:, reset_times[j]:reset_times[j + 1], :], 'float64')
            ix = _cumtrapz(a + xi, dx=1, initial=0) + a + (x0 + xi[:, :1]) / 2.0

            mu = tf.exp(ix)
            imu = _cumtrapz(mu * xi, dx=1, initial=0) + (x0 + mu[:, :1] * xi[:, :1]) / 2.0 + imu0

            valid = tf.logical_and(mu > 0.0, imu > 0.0)
            mu = tf.where(valid, mu, 1.0)
            imu = tf.where(valid, imu, 1.0)
            _td = 1 - tf.exp(tf.math.log(imu) - tf.math.log(mu))
            _td = tf.where(valid, _td, 1.0)

            x0 = xi[:, -1:]
            imu0 = imu[:, -1:] / mu[:, -1:]
            td.append(tf.cast(_td, 'float32'))
        td = tf.concat(td, axis=1)

        #ret = tstim * td
        # offset depression by one to allow transients
        ret = tstim * tf.pad(td[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=1.0)
        ret = tf.where(tf.math.is_nan(inputs), _nan, ret)

        return ret

    def weights_to_phi(self):
        layer_values = self.layer_values
        # don't need to do any reshaping
        log.debug(f'Converted {self.name} to modelspec phis.')
        return layer_values


class StateDCGain(BaseLayer):
    """Simple dc stategain."""

    _STATE_LAYER = True

    def __init__(self,
                 units=None,
                 n_inputs=1,
                 initializer=None,
                 seed=0,
                 state_type='both',
                 bounds=None,
                 exclude_chans=None,
                 per_channel=False,
                 *args,
                 **kwargs,
                 ):
        super(StateDCGain, self).__init__(*args, **kwargs)

        self.state_type = state_type
        self.exclude_chans = exclude_chans
        
        # try to infer the number of units if not specified
        if units is None and initializer is None:
            self.units = 1
        elif units is None:
            self.units = initializer['g'].value.shape[1]
        else:
            self.units = units

        self.n_inputs = n_inputs
        self.per_channel = per_channel
        self.initializer = {
                'g': tf.random_normal_initializer(seed=seed),
                'd': tf.random_normal_initializer(seed=seed + 1),  # this is halfnorm in NEMS
            }
        if initializer is not None:
            self.initializer.update(initializer)
        self.d_constraint = None
        self.g_constraint = None
        if bounds is not None:
            # constraints assumes bounds built with np.full
            if 'd' in bounds.keys():
                self.d_constraint = lambda t: tf.clip_by_value(t, bounds['d'][0], bounds['d'][1])
            if 'g' in bounds.keys():
                self.g_constraint = lambda t: tf.clip_by_value(t, bounds['g'][0], bounds['g'][1])

    def build(self, input_shape):
        input_shape, state_shape = input_shape

        if self.state_type != 'dc_only':
            self.g = self.add_weight(name='g',
                                     shape=(self.n_inputs, self.units),
                                     # shape=(self.units, input_shape[-1]),
                                     dtype='float32',
                                     initializer=self.initializer['g'],
                                     constraint=self.g_constraint,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True,
                                     )
        else:
            self.g = np.zeros((self.n_inputs, self.units))
            self.g[:, 0] = 1
            self.g = tf.constant(self.g, dtype='float32')

        if self.state_type != 'gain_only':
            # don't need a d param if we only want gain
            self.d = self.add_weight(name='d',
                                     shape=(self.n_inputs, self.units),
                                     # shape=(self.units, input_shape[-1]),
                                     dtype='float32',
                                     initializer=self.initializer['d'],
                                     constraint=self.d_constraint,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True,
                                     )

    def call(self, inputs, training=True):
        inputs, state_inputs = inputs

        if self.exclude_chans is not None:
            keep_chans = list(np.setdiff1d(np.arange(state_inputs.shape[2]), self.exclude_chans))
            
            print(f'StateDCGain keep_chans {keep_chans}')
            print(f'state_inputs shape {state_inputs.shape}')

            s_ = tf.gather(state_inputs, indices=keep_chans, axis=2)
            print(f's_ shape {s_.shape}')
        else:
            s_ = tf.identity(state_inputs)
        #import pdb;pdb.set_trace()
        if self.per_channel:
            print(f'state gain per channel')
            g_conv = s_ * tf.expand_dims(self.g, 0)            
        else:
            g_transposed = tf.transpose(self.g)
            g_conv = tf.nn.conv1d(s_, tf.expand_dims(g_transposed, 0), stride=1, padding='SAME')
        if self.state_type == 'gain_only':
            return inputs * g_conv

        if self.per_channel:
            print(f'state dc per channel')
            d_conv = s_ * tf.expand_dims(self.d, 0)            
        else:
            d_transposed = tf.transpose(self.d)
            d_conv = tf.nn.conv1d(s_, tf.expand_dims(d_transposed, 0), stride=1, padding='SAME')

        return inputs * g_conv + d_conv

    def weights_to_phi(self):
        layer_values = self.layer_values
        log.debug(f'Converted {self.name} to modelspec phis.')
        return layer_values


class Sum(BaseLayer):
    """Subclass of tf.keras Add layer"""
    def __init__(self, initializer=None, seed=0, *args, **kwargs):
        super(Sum, self).__init__(*args, **kwargs)

    def call(self, inputs, training=True):
        if type(inputs) is list:
            inputs, state_inputs = inputs
        return tf.math.reduce_sum(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        input_shape[-1] = 1
        return input_shape

    def weights_to_phi(self):
        log.info(f'No phis to convert for {self.name}.')
        return {}


class Conv2D_NEMS_new(BaseLayer):
    _TF_ONLY = True
    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               initializer=None, seed=0,
               *args,
               **kwargs):
        #initializer=None, seed=0, *args, **kwargs):
        '''Identical to the stock keras Conv2D but with NEMS compatibility added on, but without a matching module.'''
        super(Conv2D_NEMS, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # force symmetrical padding in frequency, causal in time
        pad_top = kernel_size[0]-1
        pad_bottom = 0
        pad_left = int((kernel_size[1]-1)/2)
        pad_right = kernel_size[1]-pad_left-1
        self.padding = [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]

        #self.rank = rank
        #self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        #self.padding = conv_utils.normalize_padding(padding)
        #if (self.padding == 'causal' and not isinstance(self,
        #                                                (Conv1D, SeparableConv1D))):
        #  raise ValueError('Causal padding is only supported for `Conv1D`'
        #                   'and ``SeparableConv1D`.')
        #self.data_format = conv_utils.normalize_data_format(data_format)
        #self.dilation_rate = conv_utils.normalize_tuple(
        #    dilation_rate, rank, 'dilation_rate')
        #self.activation = activations.get(activation)
        #self.input_spec = InputSpec(ndim=self.rank + 2)


    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = input_shape[-1]
        kernel_shape = self.kernel_size + [input_channel, self.filters]

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

    # Given an input tensor of shape [batch, in_height, in_width, in_channels]
    # and a filter / kernel tensor of shape
    # [filter_height, filter_width, in_channels, out_channels],
    # this op performs the following:
    def call(self, inputs, training=True):
        # inputs should be [batch, in_height, in_width, in_channels]
        # self.weights[0] should be [filter_height, filter_width, in_channels, out_channels]
        log.info(inputs.shape)
        x = tf.nn.conv2d(inputs, self.kernel, [1, 1, 1, 1], padding=self.padding)
        log.info(x.shape)
        # x should be [batch, in_height, in_width, out_channels]

        # self.weights[1] should be [1, 1, 1, out_channels]
        if self.bias is not None:
            return tf.nn.relu(x - tf.reshape(self.bias, [1, 1, 1, -1]))
        else:
            return tf.nn.relu(x)

    def weights_to_phi(self):
        phi = {'kernel': self.kernel.numpy(),
               'bias': self.bias.numpy()}
        log.debug(f'Converted {self.name} to modelspec phis.')

        return phi


class Conv2D_NEMS(BaseLayer, Conv2D):
    _TF_ONLY = False
    def __init__(self, initializer=None, seed=0, *args, **kwargs):
        '''
        Fork of Keras Conv2D module. Currently only works with specialized conditions (relu may be required) but
        implements a causal filter on the time axis. (SVD updated from JP original code 2020-07-27.)
        '''
        super(Conv2D_NEMS, self).__init__(*args, **kwargs)
        # force symmetrical padding in frequency, causal in time
        pad_top = self.kernel_size[0]-1
        pad_bottom = 0
        pad_left = int((self.kernel_size[1]-1)/2)
        pad_right = self.kernel_size[1]-pad_left-1
        self.padding = [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]

        self.initializer = {'coefficients': tf.random_normal_initializer(seed=seed),
                            'offset': tf.random_normal_initializer(seed=seed)}
        if initializer is not None:
            log.debug(f'Conv2D initializing to previous values: {list(initializer.keys())}')
            self.initializer.update(initializer)
        if kwargs.get('kernel_regularizer',None) is not None:
            log.info('kernel regularizer is not None')
            self.kernel_regularizer = kwargs.get('kernel_regularizer',None)
            #import pdb; pdb.set_trace()
        
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = input_shape[-1]
        kernel_shape = list(self.kernel_size) + [input_channel, self.filters]
        log.info(f'Conv2D build: kernel reg: {self.kernel_regularizer}')
        #import pdb; pdb.set_trace()
        self.coefficients = self.add_weight(
            name='coefficients',
            shape=kernel_shape,
            initializer=self.initializer['coefficients'],
            regularizer=self.kernel_regularizer,
            #constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.offset = self.add_weight(
                name='offset',
                shape=(1,1,1,self.filters),
                initializer=self.initializer['offset'],
                #constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.offset = None

    def call(self, inputs, training=True):
        # inputs should be [batch, in_height, in_width, in_channels]
        # self.weights[0] should be [filter_height, filter_width, in_channels, out_channels]
        x = tf.nn.conv2d(inputs, self.coefficients, [1, 1, 1, 1], padding=self.padding)
        # x should be [batch, in_height, in_width, out_channels]

        # self.weights[1] should be [1, 1, 1, out_channels]
        if self.offset is not None:
            return tf.nn.relu(x - tf.reshape(self.offset, [1, 1, 1, -1]))
        else:
            return tf.nn.relu(x)

    def weights_to_phi(self):
        phi = {'coefficients': self.coefficients.numpy(),
               'offset': self.offset.numpy()}
        log.debug(f'Converted {self.name} to modelspec phis.')
        log.debug(f"Offset : {phi['offset']}")
        return phi

class Conv2D_NEMS_old(BaseLayer, Conv2D):
    _TF_ONLY = True
    def __init__(self, initializer=None, seed=0, *args, **kwargs):
        '''
        Fork of Keras Conv2D module. Currently only works with specialized conditions (relu may be required) but
        implements a causal filter on the time axis. (SVD updated from JP original code 2020-07-27.)
        '''
        super(Conv2D_NEMS_old, self).__init__(*args, **kwargs)
        # force symmetrical padding in frequency, causal in time
        pad_top = self.kernel_size[0]-1
        pad_bottom = 0
        pad_left = int((self.kernel_size[1]-1)/2)
        pad_right = self.kernel_size[1]-pad_left-1
        self.padding = [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]

    def call(self, inputs, training=True):
        # inputs should be [batch, in_height, in_width, in_channels]
        # self.weights[0] should be [filter_height, filter_width, in_channels, out_channels]
        x = tf.nn.conv2d(inputs, self.weights[0], [1, 1, 1, 1], padding=self.padding)
        # x should be [batch, in_height, in_width, out_channels]

        # self.weights[1] should be [1, 1, 1, out_channels]
        if self.bias is not None:
            return tf.nn.relu(x - tf.reshape(self.weights[1], [1, 1, 1, -1]))
        else:
            return tf.nn.relu(x)

    def weights_to_phi(self):
        phi = {'kernels': self.weights[0].numpy(),
               'activations': self.weights[1].numpy()}
        log.debug(f'Converted {self.name} to modelspec phis.')
        return phi


class Dense_NEMS(BaseLayer, Dense):
    _TF_ONLY = True
    def __init__(self, initializer=None, seed=0, *args, **kwargs):
        '''Identical to the stock keras Dense but with NEMS compatibility added on, but without a matching module.'''
        super(Dense_NEMS, self).__init__(*args, **kwargs)

    def weights_to_phi(self):
        phi = {'kernels': self.weights[0].numpy(), 'activations': self.weights[1].numpy()}
        log.debug(f'Converted {self.name} to modelspec phis.')
        return phi


class WeightChannelsPerBank(BaseLayer):
    _TF_ONLY = True
    def __init__(self, units=None, initializer=None, seed=0, *args, **kwargs):
        '''Similar to WeightChannelsBasic but implemented as a weighted sum, and does not map to a NEMS module.'''
        super(WeightChannelsNew, self).__init__(*args, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.coefficients = self.add_weight(name='coefficients',
                                            shape=(1, input_shape[-2], self.units),
                                            dtype='float32',
                                            initializer=tf.random_normal_initializer,
                                            constraint=tf.keras.constraints.NonNeg(),
                                            trainable=True,
                                            )

    def call(self, inputs):
        # Weighted sum along frequency dimension
        # Assumes input formated as time x frequency x num_inputs
        return tf.reduce_sum(inputs * self.coefficients, axis=-2)

    def weights_to_phi(self):
        phi = {'coefficients': self.coefficients.numpy()}
        log.debug(f'Converted {self.name} to modelspec phis.')
        return phi


class WeightChannelsNew(BaseLayer):
    _TF_ONLY = True
    def __init__(self, units=None, initializer=None, seed=0, *args, **kwargs):
        '''Similar to WeightChannelsBasic but implemented as a weighted sum, and does not map to a NEMS module.'''
        super(WeightChannelsNew, self).__init__(*args, **kwargs)
        self.units = units

    def build(self, input_shape):
        ##constraint=tf.keras.constraints.NonNeg(),
        log.info(f'WeightChannelsNew build: kernel reg: {self.kernel_regularizer}')
        self.coefficients = self.add_weight(name='coefficients',
                                            shape=(input_shape[-2]*input_shape[-1], self.units),
                                            dtype='float32',
                                            initializer=tf.random_normal_initializer,
                                            regularizer=self.kernel_regularizer,
                                            trainable=True,
                                            )

    def call(self, inputs):
        # Weighted sum along frequency dimension
        # Assumes input formated as time x frequency x num_inputs
        batch, time, channels, units = tuple(inputs.shape)
        flat_inputs = array_ops.reshape(inputs, (array_ops.shape(inputs)[0],) + (time, channels*units))
        return tf.linalg.matmul(flat_inputs, self.coefficients)

    def weights_to_phi(self):
        phi = {'coefficients': self.coefficients.numpy()}
        log.debug(f'Converted {self.name} to modelspec phis.')
        return phi





class FlattenChannels(BaseLayer):
    _TF_ONLY = True
    def __init__(self, initializer=None, seed=0, *args, **kwargs):
        # no weights or initializer to deal with
        super(FlattenChannels, self).__init__(*args, **kwargs)

    def call(self, inputs):
        batch, time, channels, units = tuple(inputs.shape)
        return array_ops.reshape(inputs, (array_ops.shape(inputs)[0],) + (time, channels*units))

    def weights_to_phi(self):
        return {}  # no trainable weights


class Dropout_NEMS(BaseLayer, Dropout):
    '''A NEMS wrapper-layer for tensorflow.keras.layers.Dropout.'''
    _TF_ONLY = True
    def __init__(self, initializer=None, seed=0, *args, **kwargs):
        # no weights or initializer to deal with
        super(Dropout_NEMS, self).__init__(*args, **kwargs)

    def weights_to_phi(self):
        return {}  # no trainable weights
