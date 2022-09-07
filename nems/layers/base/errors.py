

class ShapeError(Exception):
    def __init__(self, layer, **shapes):
        """Raise in Layer.evaluate if data or parameter shapes are misaligned.
        
        Parameters
        ----------
        layer : .layer.Layer
            Layer that the exception was raised from.
        shapes : dict
            Keys indicate names of shapes in the error message, values indicate
            the associated shape.

        See also
        --------
        nems.layers.WeightChannels.evaluate  (example usage)
        
        """
        self.layer = layer
        self.shapes = shapes
    def __str__(self):
        string = (
            f"Shape mismatch in {type(self.layer).__name__}.evaluate.\n")
        for k, v in self.shapes.items():
            string += f"{k} has shape: {v.__repr__()}\n"
        return string


def require_shape(layer, kwargs, minimum_ndim=0, maximum_ndim=None):
    """Raise `MissingShapeError` if `shape` is not specified."""
    shape = kwargs.get('shape', None)
    layer_name = type(layer).__name__
    message = None

    if not isinstance(shape, (tuple, list)):
        message = f'Layer {layer_name} requires a `shape` kwarg.'
    elif len(shape) < minimum_ndim:
        message = (f'Layer {layer_name}.shape must have at least '
                   f'{minimum_ndim} dimensions.')
    elif (maximum_ndim is not None) and (len(shape) > maximum_ndim):
        message = (f'Layer {layer_name}.shape must have no more than '
                   f'{maximum_ndim} dimensions.')

    if message is not None:
        raise TypeError(message)
