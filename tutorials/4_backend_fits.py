# TODO: This entire document
import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential
from nems import visualization

## This indicates that our code is interactive, allowing a matplotlib
## backend to show graphs. Uncomment if you don't see any graphs
#plt.ion()

########################################################
# Setting Backend Parameters
#
# A significant part of creating accurate and optimized models
# is picking parameters to be run through your model. There are
# a lot of different paramenters, here we provide some examples
# and recommendations
#
# Finding the right parameter is both time consuming and often
# inconsistent, we have tools in place to help pick out good
# starts for parameters
########################################################

###########################
# Setting up Demo Data, see auditory data tutorial
###########################
nems.download_demo()
training_dict, test_dict = nems.load_demo()

spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response']
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response']

# Creating a basic Rank2LNSTRF model to test parameters on
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(15, 1)),  # 15 taps, 1 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)

###########################
# Picking your parameters
#
# Currently we have 2 core backends available, with
# their own parameters and options. 
#
#   tf_options:
#       - cost_function:
#       - early_stopping:
#           - delay:
#           - patience:
#           - tolerance:
#       - validation_split:
#       - learning_rate:
#       - epochs: 
#   scipy:
#       - maxiter:
#       - ftol: 
#   
###########################
backend = 'scipy'
## Uncomment if you have TensorFlow installed
#backend='tf'
if backend=='tf':
    options = {'cost_function': 'squared_error', 'early_stopping_delay': 50, 'early_stopping_patience': 10,
                  'early_stopping_tolerance': 1e-3, 'validation_split': 0,
                  'learning_rate': 5e-3, 'epochs': 2000}
    tf_model = model.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')

else:
    options = {'options': {'maxiter': 100, 'ftol': 1e-4}}
    scipy_model = model.fit(spectrogram_fit, response_fit, fitter_options=options)

###########################
# Model.sample_from_priors()
#
# Returns a single, or list of models with randomized
# parameters
#   - N: The number of models you want to create
###########################
N = 5
sample_array = model.sample_from_priors(N)
fitted_array = []
best_fit = None

# We also want to print out each of our models predictions on the same data, to compare
f, ax = plt.subplots(N+2, 1, sharex='col')
ax[0].imshow(spectrogram_test.T,aspect='auto', interpolation='none',origin='lower')
ax[0].set_ylabel('Test stimulus')
ax[1].plot(response_test, label='actual response')
ax[1].set_ylabel('Test response')

# This loops fits each model, compares them to find best fit, and shows how they compare on a graph
iter = 0
for x in sample_array:
    fitted_array.append(x.fit(spectrogram_fit, response_fit))
    pred_model = x.predict(spectrogram_test)
    if best_fit is None or best_fit.results.final_error > fitted_array[iter].results.final_error:
        best_fit = fitted_array[iter]

    ax[iter+2].plot(pred_model, label='predicted')
    ax[iter+2].set_ylabel(f'fit {iter}')
    iter += 1

# Plotting the model of our best fit
best_fit.name = "Best Fit"
best_fit.plot(spectrogram_test, target=response_test)

## Uncomment if you don't have an interactive backend installed
#plt.show()
