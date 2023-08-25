"""Demonstrates fitting Layers with multiple inputs, and Layer subclassing."""

import numpy as np
import matplotlib.pyplot as plt

from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential
from nems.layers.base import Layer, Phi, Parameter
from nems import visualization

# Fast-running toy fit options for demonstrations.
options = {'options': {'maxiter': 2, 'ftol': 1e-2}}

# This indicates that our code is interactive, allowing a
# matplotlib backend to show graphs
#plt.ion()

# Dummy data
def my_complicated_data_loader(file_path):
    TIME = 100
    CHANNELS = 18

    spectrogram = np.random.rand(TIME, CHANNELS)
    response = np.stack(spectrogram[:, :5])
    pupil_size = np.random.rand(TIME, 1)
    other_state = np.random.rand(TIME, 1)
    return spectrogram, response, pupil_size, other_state, TIME, CHANNELS
stim, resp, pupil, state, TIME, CHANNELS = my_complicated_data_loader('/path/data.csv')


########################################################
# Creating custom layers
# Sometimes you may need to deal with more data than our Input, Output, and State.
# We can create our own layers built from our base layer, to include that new data
# NOTE: This will explored more in Tutorial 10
########################################################

###########################
# Sum Layer, dummy layer that allows arbitrary number of inputs
#   evaluate: Allows you to specify wanted inputs, and define potential operations
#       *inputs: Any number of data, gets pushed into a flat evaluation with no operation
###########################
class Sum(Layer):
    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.
        return np.sum(inputs)
    
###########################
# Linear Weighting Layer, dummy layer that allows some manipulation of provided data
#   initial_parameters: Allows you to evoke a set of parameters that are passed to layers.parameter
#       a: a parameter value of shape(1,)
#       b: a second parameter value of shape(1,)
#       output: Phi(a, b)
#   evaluate: Same as before, In this case returns a multiplication of some set of inputs to parameters
#       a,b: provided parameters initialized in our first function
#       prediction, pupil, other_state: 3 various inputs
#       output: parameter1*input1 + parameter2*input2 + input3
###########################
class LinearWeighting(Layer):

    def initial_parameters(self):
        a = Parameter('a', shape=(1,))
        b = Parameter('b', shape=(1,))
        return Phi(a, b)

    def evaluate(self, prediction, pupil, other_state):
        # `prediction` and `pupil` are treated one way, whereas `other_state`
        # is treated another way.
        a, b = self.get_parameter_values('a', 'b')
        return a*prediction + b*pupil + other_state
    
###########################
# A model sometimes requires multiple inputs which requires a Dict of 
# keywords/values ex,
#   {
#       keyword1: input1,
#       keyword2: input2,
#       keyword3: input3,
#       etc....
#   }
###########################
input = {'stimulus': stim, 'pupil': pupil, 'state': state}

###########################
# Layers for our model, This time in the form of a list which
# can then be passed to our model instead of via .add_layers()
#
# Setting up layers in this way allows you to specify inputs for specific layers
#   WeightedChannels: Takes 'stimulus' from dict as an input
#   FIR: Takes the preceding layer as it's input
#   Double Exp: Also takes preceding layer as input, but outputs LN for future inputs
#   Sum: Adds up a bunch of inputs including the LN output
#   LinearWeighting: Sets evaluation parameters from dict keys
#       Also keeps parameters empty, so they are infered from other layers(DOUBLE CHECK THIS)
###########################
layers = [
    WeightChannels(shape=(18,4), input='stimulus'),
    FIR(shape=(15, 4)),
    DoubleExponential(shape=(1,), output='LN_output'),
    Sum(input=['LN_output', 'state', 'pupil'],
                    output='summed_output'),
    LinearWeighting(input={'prediction': 'summed_output', 'pupil': 'pupil',
                           'other_state': 'state'})
]
model = Model(layers=layers)
# Note that we also passed a list of `Layer` instances to `Model.__init__`
# instead of using the `add_layers()` method. These approaches are
# interchangeable.

# TODO: this won't work yet since there are multiple outputs.
# We fit as before, but provide the `input` dictionary in place of individual
# variables. The necessary inputs are already specified in the layers, so we
# only need to tell the model what data to match its output to (`resp`).
fitted_model = model.fit(input=input, target=resp, fitter_options=options)
prediction = model.predict(input)

# plots our fitted model, and graphs of raw data before model fitting
fitted_model.plot(input=input, target=resp, figure_kwargs={'figsize': (12,8)})

## Uncomment if you don't have an interactive backend installed
#plt.show()

