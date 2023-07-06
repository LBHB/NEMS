"""Demonstrates the key steps needed to create new Layer subclasses."""

import numpy as np
import matplotlib.pyplot as plt

from nems import Model
from nems.distributions import Normal, HalfNormal
from nems.registry import layer
from nems.layers.base import Layer, Phi, Parameter

########################################################
# Creating new layers
#
# Using Layer as a parent, we can create new layer subclasses for your own usecases.
# like any subclass, a newly created layer will inherit many features, so make sure
# to look at the base layer function too.
# As an example, we'll define applies exponential decay to its input.
#   - initial_parameters: Lets you specify trainable parameters, bounds, or priors
#       - s: Parameter(), provide name, shape, prior, bounds etc... 
#       - t: another Parameter()
#       - Phi: Returned 'List' of parameters and info
#   - evaluate: Defines the tranformation applied to inputs
#       - evaluate(self, input): need at least one argument, but can specify more
#       - s,t: recieve paramter values from initial_parameters
#       - Returns data after operations have been performed
#   - from_keyword(keyword): 
#       - @layer('name'): Decorator for setting up keyword layers
#       - keyword: Refered by the decorator, could be used for a parser or other modifications
#       - Returns an instance of our new layer
# See more at: https://temp.website.net/nems_base_Layer
########################################################
class ExponentialDecay(Layer):

    def initial_parameters(self):
        # Create two Parameters: one for the initial value, and one for
        # the time constant of the decay. Each Parameter requires a name as
        # its first positional argument. If no `shape` is specified, it defaults
        # to `shape=()` (a scalar). Priors must be specified using Distributions,
        # which are wrappers for `scipy.stats`.
        s = Parameter('scale', shape=(), prior=Normal(mean=1, sd=1))
        t = Parameter('tau', shape=(), prior=HalfNormal(sd=10))
        # Package both Parameters into a Phi object. In essence, this is a dict
        # that also tracks Parameter values and other information.
        return Phi(s, t)

    def evaluate(self, input):
        # Evaluate must accept at least one argument, but otherwise the method
        # signature is flexible. However, only one argument will be supplied
        # during Model evaluation unless the `Layer.input` attribute is set.
        s, t = self.get_parameter_values()
        return s * np.exp(-1/t * input)

    # Optionally, we can also define a keyword for our Layer using the `layer`
    # decorator. Technically this method can be called anything (and doesn't 
    # even need to be a method of the class), but by convention we define a
    # method named `from_keyword`.
    @layer('decay')
    def from_keyword(keyword):
        # This must return an instance of the desired Layer class.
        # `Layer.name` will be overwritten to match the keyword ('decay').
        return ExponentialDecay()
    
# By default, `scale = 1` and `tau = 10*sqrt(2/pi)` (means of priors)
decay = ExponentialDecay()

# Now we can use our new Layer to:

###########################
# Set up our data
#   input: Linear set of data with length 1000
#   target: random non-linear decay dataset to represent decay
#   y: Output of our input being thrown directly into our new layer
###########################
input = np.arange(1000)[..., np.newaxis]
target = -6*np.exp(-1/100*input) + np.random.rand(1000, 1)
y = decay.evaluate(input)

# Here is a comparison of our values before we run any models
print(f"""
Our first inital input is: {input[0]}\n
Our first target is: {target[0]}\n
The first output of our decay layer with no adjustments or fitting is: {y[0]}""")

###########################
# Sample new parameter values
# sample_from_priors: creates layer with new parameter values sampled from priors
#   inplace: When true, update and adjust parameters inplace
###########################
decay.sample_from_priors(inplace=True)

###########################
# Building models
#   via list of layers
#   direct model initialization
#   from_keywords() list
###########################
model = Model()
model.add_layers(decay)
model = Model('decay')
# NOTE: `Model.from_keywords` will use a *new* instance of ExponentialDecay.
model = Model().from_keywords('decay')

###########################
# Fit our model to optimize parameters
#   fitter_options: Same as other tutorials, used for demonstration
###########################
fitted_model = model.fit(
    input=input, target=target,
    # Set low iteration count to speed up fitting demonstration.
    fitter_options={'options': {'maxiter': 50, 'ftol': 1e-10}}
    )

###########################
# Predict our model
###########################
predicted_model = model.predict(input=target)

###########################
# Plots results
# This will show 2 simple graphs
#   Our linear input graph
#   Our randomized adjustment graph
#   A second line on our second graph for the output of our decay model
###########################
fitted_model.plot(input=input, target=target)

plt.show()
