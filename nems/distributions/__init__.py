"""Collection of sampling Distributions.

Intended to serve as priors for  `nems.layers.base.Parameter`.
Modified from code by Dr. Brad Buran (github: bburan).

"""

from .normal import Normal
from .half_normal import HalfNormal
from .uniform import Uniform
from .beta import Beta
from .gamma import Gamma
from .exponential import Exponential
