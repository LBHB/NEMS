

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
