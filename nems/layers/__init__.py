'''
Collects commonly-used Modules for easier import.

Ex: `from nems.modules import WeightChannels, FIR, DoubleExponential`

'''

from .nonlinearity import LevelShift, DoubleExponential, RectifiedLinear, ReLU
from .filter import FiniteImpulseResponse, FIR, STRF
from .weight_channels import WeightChannels, WeightChannelsMulti, WeightChannelsGaussian, WeightChannelsMultiGaussian
from .state import StateGain
from .numpy import NumPy
from .stp import ShortTermPlasticity, STP
from .algebra import SwapDims

from .base import Layer, Phi, Parameter
