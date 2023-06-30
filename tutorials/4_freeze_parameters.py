"""Shows how to freeze Layer parameters in a multi-step fitting process."""

import numpy as np

from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential

########################################################
# Freezing Parameters
# When fitting models we've found that seperating linear and non-linear steps
# will improve the fit
#
# We have a variety of ways to freeze certain parts of our model
#   - Freezing Linear/Non-linear parameters in all layers
#   - Freezing specific layers or a list of layers
#   - Freezing specific parameters within any specific layer
#
# Note: It will be helpful to keep track of your layers index and/or provide names
########################################################

# Creating fake data, see "1_simple_fit" tutorial for more information
input = np.random.rand(1000, 18)  # 18-channel spectrogram stimulus
target = np.stack(input[:, :5])   # PSTH response of a single neuron

###########################
# Building a standard linear model
#
# - Layers have parameter name="some_name"
#   - This will be used for freezing specific layers
# - Double Exponential shape=(x,) creates a non-linear layer
###########################
model = Model()
model.add_layers(
    WeightChannels(shape=(18,4), name='wc'),
    FIR(shape=(25, 4), name='fir'),
    DoubleExponential(shape=(1,), name='dexp'),
)

# Options parameters for demonstration, see more at: scipy.optimize.minimize
initialization = {'options': {'ftol': 1e3, 'maxiter': 10}}

# Layer function will skip non-linear parameters, while still modifying linear ones
# for example: this results in a scalar offset: `output = input + shift`).
model.layers['dexp'].skip_nonlinearity()

# Show the number of frozen (untrainable), unfrozen (trainable),
# permanent (always untrainable), and total parameter values for each layer.
print(f'Parameters for first fit:\n{model.parameter_info}\n')

# Note that fit returns a copy of the model. In this case we choose to re-use
# the same model name (i.e. the unfit model will be overwritten).
model = model.fit(input=input, target=target, fitter_options=initialization)

# Opens our non-linear paramenters back up to modification again
model.layers['dexp'].unskip_nonlinearity()

###########################
# Freezing groups of layers
# 
# Often we will need to activate large groups of layers or parameters
#   - Integer: We can do this my providing indexs of layers
#   - Keyword: We can use our defined names for layers instead
###########################
model.freeze_layers('wc', 'fir')
print(f'\nParameters for second fit:\n{model.parameter_info}\n')

# Fit the model again, using the previous fit as a starting point.
model = model.fit(input=input, target=target, fitter_options=initialization)

###########################
# Unfreezing Layers
#
# Follows the same pattern as freezing in addtion;
#   - unfreeze_layers() will unfreeze every layer
###########################
model.unfreeze_layers('wc', 'fir')
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
print(f'\nParameters for final fit:\n{model.parameter_info}\n')
