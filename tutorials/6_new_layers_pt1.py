"""Demonstrates the key steps needed to create new Layer subclasses."""

import numpy as np

from nems import Model
from nems.distributions import Normal
from nems.registry import layer
from nems.layers.base import Layer, Phi, Parameter


# Creating a new Layer subclass is the first step in expanding NEMS to include
# new types of models. Most new subclasses will need to define two methods:
# 1) `initial_parameters`
#    Specifies the Layer's trainable Parameters and default bounds & priors.
# 2) `evaluate`
#    Defines the transformation that the Layer applies to inputs.

# As an example, we'll define a new Layer that adds a sinusoid to its input.
class AddSine(Layer):

    def initial_parameters(self):
        # Create two Parameters: one to control the amplitude of the sine wave,
        # another to control its frequency. Each Parameter requires a name as
        # its first positional argument. If no `shape` is specified, it defaults
        # to `shape=()` (a scalar). Priors must be specified using Distributions,
        # which are wrappers for `scipy.stats`.
        a = Parameter('amplitude', shape=(), prior=Normal(mean=1, sd=1))
        f = Parameter('frequency', shape=(), prior=Normal(mean=0.1, sd=1))
        # Package both Parameters into a Phi object. In essence, this is a dict
        # that also tracks Parameter values and other information.
        return Phi(a, f)

    def evaluate(self, input):
        # Evaluate must accept at least one argument, but otherwise the method
        # signature is flexible. However, only one argument will be supplied
        # during Model evaluation unless the `Layer.input` attribute is set.
        a, f = self.get_parameter_values()
        n_times = input.shape[0]
        sine = a * np.sin(f * np.arange(n_times))[..., np.newaxis]
        # Since `sine` has shape (T, 1), it will broadcast across all N output
        # channels for an input with shape (T, N).
        return input + sine

    # Optionally, we can also define a keyword for our Layer using the `layer`
    # decorator. Technically this method can be called anything (and doesn't 
    # even need to be a method of the class), but by convention we define a
    # method named `from_keyword`.
    @layer('sine')
    def from_keyword(keyword):
        # This must return an instance of the desired Layer class.
        # `Layer.name` will be overwritten to match the keyword ('sine').
        return AddSine()
    
# By default, `amplitude = 1` and `frequency = 0.1` (means of priors)
sine = AddSine()

# Now we can use our new Layer to:

# 1) Transform data:
x = np.zeros(shape=(1000, 1))
y = sine.evaluate(x)

# 2) Sample new parameter values:
sine.sample_from_priors(inplace=True)

# 3) Build models:
model1 = Model()
model1.add_layers(sine)
# NOTE: `Model.from_keywords` will use a *new* instance of AddSine.
model2 = Model.from_keywords('sine')

# 4) Optimize parameters:
z = 4 * np.sin(0.5 * np.arange(1000))[..., np.newaxis]  # Target
fitted = model2.fit(
    x, target=z,
    # Set low iteration count to speed up fitting demonstration.
    fitter_options={'options': {'maxiter': 50, 'ftol': 1e-10}}
    )

# TODO: Not clear why this is failing so miserably for such a simple fit,
#       definitely needs to be looked into.
# Plot results:
fitted.plot(input=x, target=z)
