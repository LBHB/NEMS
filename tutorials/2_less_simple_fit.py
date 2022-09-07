"""Demonstrates fitting Layers with multiple inputs, and Layer subclassing."""

import numpy as np

from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential
from nems.layers.base import Layer, Phi, Parameter

# Fast-running toy fit options for demonstrations.
options = {'options': {'maxiter': 2, 'ftol': 1e-2}}


# Some models will need more data than input, output, and state. In that case,
# the fit process requires a couple of extra steps. First, we'll define some
# simple example Layers that require additional inputs and then load data again.

# Dummy layer that can make use of an arbitrary number of inputs.
class Sum(Layer):
    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.
        return np.sum(inputs)

# Dummy layer that makes use of its inputs in different ways.
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


def my_complicated_data_loader(file_path):
    # Dummy function to demonstrate the data format.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.rand(100, 18)
    response = np.random.rand(100, 5)
    pupil = np.random.rand(100, 1)
    other_state = np.random.rand(100, 1)

    return spectrogram, response, pupil, other_state

stim, resp, pupil, state = my_complicated_data_loader('/path/data.csv')


# For a model that uses multiple inputs, we package the input data into
# a dictionary. This way, layers can specify which inputs they need using the
# assigned keys.
input = {'stimulus': stim, 'pupil': pupil, 'state': state}
target = resp


# Now we build the Model as before, but we can specify which Layer receives
# which input(s) during fitting. We'll also use a factorized, parameterized STRF
# in place of the full-rank version to demonstrate usage. For layers that only
# need a single input, we provide the approprtiate dictionary key. In this case,
# we named the spectrogram data 'stimulus' so that will be the input to
# `WeightChannels`. The inputs for `FIR` and `DoubleExponential` should be the
# outputs of their preceding layers, so we don't need to specify an input.
#
# We do want to track the output of the LN portion of the model
# (`WeightChannels`, `FIR`, and `DoubleExponential`) separately from the rest,
# so we specify `output='LN_output'` for `DoubleExponential`.
#
# We want to apply the `Sum` layer to the output of the LN portion of the
# model and to both of our state variables. The order of the data variables
# doesn't matter for this layer, so we provide a list of the dictionary keys.
#
# Order does matter for the `LinearWeighting` layer, so we provide a dictionary
# instead to ensure the correct data is mapped to each parameter
# for `LinearWeighting.evaluate`.
# NOTE: We could also have used a list as long as the keys were specified in
#       the same order as in the method definition. However, using a dictionary
#       clarifies what the data is mapped to without needing to refer back to
#       the Layer implementation.
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
model.fit(input=input, target=resp, fitter_options=options)
prediction = model.predict(input)
