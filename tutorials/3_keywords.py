"""Demonstrates how to build models using keywords."""

from nems import Model
from nems.layers import STRF, WeightChannels, FIR, DoubleExponential, WeightChannelsGaussian
from nems.registry import keyword_lib

# View all defined keywords
keyword_lib.list


###########################
# Models can be built in the same ways previously show, but instead 
# of calling the class of layers, we can provide a keyword pattern
# Ex:
# '{layerKW}.{Shape}.{PositionalParam1}.{PositionalParam2}... etc....'
#
# These keywords can be passed as layers to add_layers or model directly as well
#
# See more at: https://temp.website.net/nems_scripts_keywords
###########################
model = Model.from_keywords('wc.18x1', 'fir.15x1', 'dexp.1')
model = Model.from_keywords('wc.18x1-fir.15x1-dexp.1')

# Keyword and Traditional models below would be equivalent
keyword_model = Model(layers=['wc.18x1', 'fir.15x1', 'dexp.1'])

layers = [WeightChannels(shape=(18,1)), FIR(shape=(15,1)), 
          DoubleExponential(shape=(1,))]
traditional_model = Model(layers=layers)


# Specify layer options using keywords. The following are equivalent.
# TODO: add a str vs Module check in add_layer instead of using
#       a separate .from_keywords method?  ####Should be done -Isaac
keyword_model.add_layers('wc.4x18.g')
traditional_model.add_layers(WeightChannelsGaussian(shape=(4,18)))
