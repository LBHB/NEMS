"""Demonstrates how to build models using keywords."""

from nems import Model
from nems.layers import STRF, WeightChannels, FIR, DoubleExponential, WeightChannelsGaussian
from nems.registry import keyword_lib

# View all defined keywords
keyword_lib.list

########################################################
# Using Keywords
# NEMS allows a set of keywords to be used for building models
# quickly and with highly reduced bloat. This system can also
# be expanded to include your own set of layers, and provide
# specific layer parameters or instances 
#
# @layer():
# Attached to your own layers or modules, this decorator will
# register a given keyword to a specific layer instance defined
# by the function below it.
#
# Model().from_keywords():
# Will allow you to call layers and build models via keywords
#
# Model(**Keywords):
# Lets you initialize your model with layers from the start
########################################################



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
single_keyword_model = Model(layers='wc.18x1-fir.15x1-dexp.1')

layers = [WeightChannels(shape=(18,1)), FIR(shape=(15,1)), 
          DoubleExponential(shape=(1,))]
traditional_model = Model(layers=layers)

###########################
# Building your own keywords
# We will be covering the specifics of making own layers
# in another tutorial, but it's important to know that you
# can build your own keywords in this system, or specify
# special/specific keywords for personal use
#
# The core 2 rules that need to be followed are 
# 1. Maintain the current keyword structure "layer.param.param.etc ..."
# 2. The function defined below the keyword MUST return an instanced layer
###########################

# This module is required to register your own keywords, it will need
# to be included at the top of your layer modules
from nems.registry import layer

# The pattern here can be directly followed for now, for more specifics 
# you will want to look at our new_layers tutorial.

# This will create a special keyword to apply some specific logic to our layer
# 'wcd' is the keyword you will need to provide in the future

# You would need to import this file into your main script to access this specific keyword
# since you are modifying a predefined layer
# !! This only applies to layers already defined inside nems, you can add these directly
# into your own layers normally which would register when your layer is imported
@layer('wcd')
def from_keyword(keyword):
    options = keyword.split('.')
    # We are directly copying the already existing keyword logic from WeightChannels with
    # normal calls
    print(f"The list we've recieved from splitting our keywords: {options}")
    print(f"The keyword we just sent: wc.{options[1]}")
    layer_instance = WeightChannels.from_keyword(f"wc.{options[1]}")

    # Now that we have a create instance of a layer, we can apply some new conditions or changes.
    # Here we are deciding that our new conditions are wcd.18x2.a.b would become a layer with inputs
    # labeled a and outputs labeled b
    kwargs = {}
    if not options[2] and not options[3]:
        raise AttributeError('WDC requires 2 additional variables to name inputs/outputs')
    if options[2]:
        kwargs['input_name'] = options[2]
    if options[3]:
        kwargs['output_name'] = options[3]
    
    layer_instance.input = kwargs['input_name']
    layer_instance.output = kwargs['output_name']

    return layer_instance
WeightChannels.from_keyword2 = from_keyword
    
test = Model.from_keywords('wcd.18x2.a.b')
print(f"Our input name is {test.layers[0].input}, and output is {test.layers[0].output}")

# and now using our keyword we can set our input/output names to anything we want for this layer
test = Model.from_keywords('wcd.18x2.IHateInputs.ILoveOutputs')
print(f"Our input name is {test.layers[0].input}, and output is {test.layers[0].output}")

# Specify layer options using keywords. The following are equivalent.
# TODO: add a str vs Module check in add_layer instead of using
#       a separate .from_keywords method?  ####Should be done -Isaac
keyword_model.add_layers('wc.4x18.g')
traditional_model.add_layers(WeightChannelsGaussian(shape=(4,18)))
