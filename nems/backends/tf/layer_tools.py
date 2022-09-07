import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint


# TODO: handle frozen parameters
class NemsKerasLayer(Layer):

    def __init__(self, nems_layer, new_values=None, new_bounds=None,
                 regularizer=None, *args, **kwargs):
        """TODO: docs."""
        # Don't pass 'input_shape' to Keras Layer constructor. Will still be in here
        # if `nems_layer.as_tensorflow_layer` didn't use it.
        _ = kwargs.pop('input_shape', None)
        super().__init__(name=nems_layer.name, *args, **kwargs)
        if new_values is None: new_values = {}
        if new_bounds is None: new_bounds = {}
        for p in nems_layer.parameters:
            self._weight_from_nems(
                p, new_values.get(p.name, None), new_bounds.get(p.name, None),
                regularizer
                )

    def _weight_from_nems(self, p, value=None, bounds=None, regularizer=None):
        """Convert Parameter of NEMS Layer to a Keras Layer weight."""
        if value is None: value = p.values
        if bounds is None: bounds = p.bounds
        init = tf.constant_initializer(value)
        constraint = Bounds(bounds[0], bounds[1])
        trainable = not p.is_frozen
        setattr(
            self, p.name, self.add_weight(
                shape=value.shape, initializer=init, trainable=trainable,
                regularizer=regularizer, name=p.name, constraint=constraint
                )
            )

    @property
    def parameter_values(self):
        """Returns key value pairs of the weight names and their values."""
        # TF appendss :<int> to the weight names.
        values = {weight.name.split(':')[0]: weight.numpy()
                    for weight in self.weights} 
        return values

    def weights_to_values(self):
        return self.parameter_values


class Bounds(Constraint):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    def __call__(self, w):
        # TODO: This is not a great way to do constraints, but I'm not sure
        #       where alternatives would be implemented (like modifying the
        #       cost function to grow very large when nearing the bound).
        # TODO: Maybe try this version:
        # https://www.tensorflow.org/probability/api_docs/python/tfp/math/clip_by_value_preserve_gradient
        return tf.clip_by_value(w, self.lower, self.upper)
