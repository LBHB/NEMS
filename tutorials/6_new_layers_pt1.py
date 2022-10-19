"""Demonstrates the key steps needed to create new Layer subclasses."""

import numpy as np

from nems import Model
from nems.distributions import Normal, HalfNormal
from nems.registry import layer
from nems.layers.base import Layer, Phi, Parameter


# Creating a new Layer subclass is the first step in expanding NEMS to include
# new types of models. Most new subclasses will need to define two methods:
# 1) `initial_parameters`
#    Specifies the Layer's trainable Parameters and default bounds & priors.
# 2) `evaluate`
#    Defines the transformation that the Layer applies to inputs.

# As an example, we'll define applies exponential decay to its input.
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

# 1) Transform data:
x = np.arange(1000)[..., np.newaxis]
y = decay.evaluate(x)

# 2) Sample new parameter values:
decay.sample_from_priors(inplace=True)

# 3) Build models:
model1 = Model()
model1.add_layers(decay)
# NOTE: `Model.from_keywords` will use a *new* instance of ExponentialDecay.
model2 = Model.from_keywords('decay')

# 4) Optimize parameters:
z = -6*np.exp(-1/100*x) + np.random.rand(1000, 1) # Target
fitted = model2.fit(
    x, target=z,
    # Set low iteration count to speed up fitting demonstration.
    fitter_options={'options': {'maxiter': 50, 'ftol': 1e-10}}
    )

# Plot results:
fitted.plot(input=x, target=z)
