"""Demonstrates how to build models using keywords."""

from nems import Model
from nems.layers import STRF, WeightChannels, FIR, DoubleExponential
from nems.registry import keyword_lib

# View all defined keywords
keyword_lib.list


# Build a model from keywords. The following are equivalent.
model = Model.from_keywords('wc.18x1', 'fir.15x1', 'dexp.1')
model = Model.from_keywords('wc.18x1-fir.15x1-dexp.1')

keywords = ['wc.18x1', 'fir.15x1', 'dexp.1']
layers = [keyword_lib[k] for k in keywords]
model = Model(layers=layers)

layers = [WeightChannels(shape=(18,1)), FIR(shape=(15,1)), 
          DoubleExponential(shape=(1,))]
model = Model(layers=layers)


# Specify layer options using keywords. The following are equivalent.
# TODO: add a str vs Module check in add_layer instead of using
#       a separate .from_keywords method?
model.add_layer('wc.4x18.g')
model.add_layer(WeightChannels(shape=(4,18), parameterization='gaussian'))
