"""Shows how to freeze Layer parameters in a multi-step fitting process."""

import numpy as np

from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential


# In the past, we've found that our models predict our data best if we split
# the fitting process into linear and nonlinear steps. We can do this with
# NEMS with `StaticNonlinearity.skip_nonlinearity()` and repeated `model.fit()`.
# NOTE: We will use the `maxiter` option for `scipy.optimize.minimize` to run
#       the script in a reasonable amount of time. For a real fit, a much larger
#       number should be used (see `scipy` documentation for defaults`).

# "Load" the data
input = np.random.rand(1000, 18)  # 18-channel spectrogram stimulus
target = np.random.rand(1000, 1)   # PSTH response of a single neuron

# Build a simple linear model.
model = Model()
model.add_layers(
    WeightChannels(shape=(18,4), name='wc'),
    FIR(shape=(25, 4), name='fir'),
    DoubleExponential(shape=(1,), name='dexp'),
)

# Fit the linear model using a coarse tolerance.
# NOTE: The nested dictionary structure is due to the syntax for
#       `scipy.optimize.minimize` kwargs, see their documentation.
initialization = {'options': {'ftol': 1e3, 'maxiter': 10}}
# This will freeze (make untrainable) all nonlinear parameters of
# DoubleExponential, but still fit the `shift` prameter (which effectively
# results in a scalar offset: `output = input + shift`).
model.layers['dexp'].skip_nonlinearity()

# Show the number of frozen (untrainable), unfrozen (trainable),
# permanent (always untrainable), and total parameter values for each layer.
print(f'Parameters for first fit:\n{model.parameter_info}\n')

# Note that fit returns a copy of the model. In this case we choose to re-use
# the same model name (i.e. the unfit model will be overwritten).
model = model.fit(input=input, target=target, fitter_options=initialization)

# Add the nonlinear component to the end of the model.
model.layers['dexp'].unskip_nonlinearity()

# Freeze the parameters of the linear portion of the model.
# The values of these parameters will not be changed during fitting.
model.freeze_layers('wc', 'fir')
# model.freeze_layers(0, 1)  # integer indexing would also work

print(f'\nParameters for second fit:\n{model.parameter_info}\n')

# Fit the model again, using the previous fit as a starting point.
model = model.fit(input=input, target=target, fitter_options=initialization)

# Unfreeze the linear portion.
model.unfreeze_layers('wc', 'fir')
# In this case, this is equivalent to unfreezing all layers, which we can
# do by *not* specifying any names or indices:
# model.unfreeze_layers()

print(f'\nParameters for final fit:\n{model.parameter_info}\n')

# Now perform a final fit of all parameters simultaneously, using a finer
# optimization tolerance.
final_fit = {'options': {'ftol': 1e4, 'maxiter': 10}}
model = model.fit(input=input, target=target, fitter_options=final_fit)

# NOTE: In this example, we always froze all parameters of a given module using
#       the model-level method. If we had wanted to freeze some of a module's
#       parameters but not others, we could use the module-level method.
#       For example:
# Just freeze DoubleExponential's kappa parameter:
model.layers['dexp'].freeze_parameters('kappa')
