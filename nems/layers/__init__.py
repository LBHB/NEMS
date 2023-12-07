'''
Collects commonly-used Modules for easier import.

Ex: `from nems.modules import WeightChannels, FIR, DoubleExponential`

'''

from .nonlinearity import LevelShift, DoubleExponential, RectifiedLinear, ReLU, Sigmoid
from .filter import FiniteImpulseResponse, FIR, STRF
from .weight_channels import WeightChannels, WeightChannelsMulti, WeightChannelsGaussian, \
    WeightChannelsMultiGaussian, WeightGaussianExpand
from .state import StateGain, StateDexp, StateHinge
from .numpy import NumPy
from .stp import ShortTermPlasticity, STP
from .algebra import SwapDims, ConcatSignals, MultiplySignals, MultiplyByExp
from .conv2d import Conv2d

from .base import Layer, Phi, Parameter
