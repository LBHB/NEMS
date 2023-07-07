import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential
from nems import visualization

# This indicates that our code is interactive, allowing a
# matplotlib backend to show graphs
plt.ion()

###########################
# Setting up our data
# Setting up demo data, see last tutorial
# 
# load_demo(): Provides our tuple of training/testing dictionaries
#   Each dictionary contains a 100 hz natural sound spectrogram and
#   the PSTH / firing rate of the recorded spiking response.
###########################
nems.download_demo()
training_dict, test_dict = nems.load_demo()

spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response']
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response']

# Creating our model to test with our demo data
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 2)),  # 18 spectral channels->2 composite channels
    FiniteImpulseResponse(shape=(15, 2)),  # 15 taps, 2 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)
model.name = "Rank2LNSTRF"

# random initial conditions
model = model.sample_from_priors()

###########################
# Setting up backend parameters
# Since we can have multiple backends for our model,
# we've create 2 sets of optimized model paramenters
#
# TF: Tensorflow backend, as opposed to scipy
#   When using tensorflow we will want different paramters than Scipy
###########################

backend = 'scipy'
## Uncomment if you have TensorFlow installed
#backend='tf'
if backend=='tf':
    options = {'cost_function': 'squared_error', 'early_stopping_delay': 50, 'early_stopping_patience': 10,
                  'early_stopping_tolerance': 1e-3, 'validation_split': 0,
                  'learning_rate': 5e-3, 'epochs': 2000}
    fitted_model = model.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')

else:
    options = {'options': {'maxiter': 100, 'ftol': 1e-4}}
    fitted_model = model.fit(spectrogram_fit, response_fit, fitter_options=options)

# evaluate model on test stimulus
pred_test = fitted_model.predict(spectrogram_test)

###########################
# Ways to view result
# Using our completed models, testing data, and matplotlib we can
# view our data and results of our model in a few different ways
#
# imshow: Creating an image using spectrogram_test.T data
# plot1: plotting response_test to a graph on ax[1]
# plot2: plotting pred_test to graph our prediction on ax[2]
###########################
f, ax = plt.subplots(3, 1, sharex='col')
ax[0].imshow(spectrogram_test.T,aspect='auto', interpolation='none',origin='lower')
ax[0].set_ylabel('Test stimulus')
ax[1].plot(response_test, label='actual response')
ax[1].set_ylabel('Test response')
ax[2].plot(pred_test, label='predicted')
ax[2].set_ylabel('Prediction')

###########################
# Built- in visualization
# We have created several utilities for plotting and
# visualization of NEMS objects
#
#   plot_model: Creates a variety of graphs and plots for a given model and layers
#   simple_strf: Gets FIR and WeightChannels from Model
###########################
visualization.plot_model(fitted_model, spectrogram_test, response_test)
visualization.simple_strf(fitted_model)

## Uncomment if you don't have an interactive backend installed
#plt.show()
