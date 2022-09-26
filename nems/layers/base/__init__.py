"""Base structures for representing data transformation steps of Models.

Layer     : Top-level component. Implements an `evaluate` method, which carries
            out the transformation.
Phi       : Stores and manages fittable parameters for the transformation.
Parameter : Low-level representation of individual parameters.

"""

from .layer import Layer
from .phi import Phi
from .parameter import Parameter
from .map import DataMap
from .errors import ShapeError
