import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential
from nems import visualization

## This indicates that our code is interactive, allowing a matplotlib
## backend to show graphs. Uncomment if you don't see any graphs
#plt.ion()

# Basic options to quickly fit our models
options = {'options': {'maxiter': 100, 'ftol': 1e-4}}

###########################
# Setting up Demo Data instead of dummy data
# 
# load_demo(): Provides our tuple of training/testing dictionaries
#   Each dictionary contains a 100 hz natural sound spectrogram and
#   the PSTH / firing rate of the recorded spiking response.
#   - Several datasets exist for load_demo, these can be specified
#   as a parameter: load_demo(test_dataset)
# See more at: nems.download_demo
###########################
nems.download_demo()
training_dict, test_dict = nems.load_demo()

spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response']
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response']

############GETTING STARTED###############
###########################
# Complex Models and Data
# Our initial tutorial used a very simple set of layers and
# data to show how a model and fits might work. Here you can see
# a quick example of a more complex model that can fit real data
#
# Using a set of FIR and DEXP layers is how we've set up this complex model,
# see advanced section and nems.layers for more information on these layers
#
# NOTE: I've commented out the fits/predicts here to save you some time
# see advanced section for actual fits
###########################
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(15, 1)),  # 15 taps, 1 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)
# fitted_model = model.fit(spectrogram_fit, response_fit, fitter_options=options, backend='scipy')
# pred_test = fitted_model.predict(spectrogram_test)




############ADVANCED###############
###########################
# Creating a Higher Ranked Non-linear STRF Model
# 
# Instead of creating a linear model using Weighted Channels, we are
# created a more complex non-linear model and using real data to measure
# it's effectiveness
#
#   FiniteImpulseResponse: Convolve linear filter(s) with inputs
#       - Can specifiy "Time-Bins", "Rank/Input-Channels", "Filters", etc...
#   DoubleExponential: Sets inputs as a exponential function to the power of some constant
# See more at: nems.layers
###########################
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(15, 1)),  # 15 taps, 1 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)
model.name = "Rank2LNSTRF-PreFit"

# Fitting our Spectrogram data to our target response data
fitted_model = model.fit(spectrogram_fit, response_fit, fitter_options=options, backend='scipy')
fitted_model.name = "Rank2LNSTRF-PostFit"

###########################
# Built- in visualization
# We have created several utilities for plotting and
# visualization of NEMS objects
#
#   plot_model: Creates a variety of graphs and plots for a given model and layers
#   simple_strf: Gets FIR and WeightChannels from Model
###########################

# A plot of our model before anything has been fit
# NOTE: It may be very hard to see the blue line, try zooming in very close
model.plot(spectrogram_fit, target = response_fit)

visualization.plot_model(fitted_model, spectrogram_test, response_test)
visualization.simple_strf(fitted_model)

# Some other important information before fitting our model.
# This time looking at how our FIR layer interacts before/after fits
print(f"""
    Rank2LNSTRF Layers:\n {model.layers}
    FIR shape:\n {model.layers[1].shape}
    FIR priors:\n {model.layers[1].priors}
    FIR bounds:\n {model.layers[1].bounds}
    FIR coefficients:\n {model.layers[1].coefficients}
""")

# Seeing our FIR inputs after our fit has been applied
print(f"""
    Fitted Rank2LNSTRF Layers:\n {fitted_model.layers}
    Fitted FIR shape:\n {fitted_model.layers[1].shape}
    Fitted FIR priors:\n {fitted_model.layers[1].priors}
    Fitted FIR bounds:\n {fitted_model.layers[1].bounds}
    Fitted FIR coefficients:\n {fitted_model.layers[1].coefficients}
""")

# evaluate model on test stimulus
pred_fitted_model = fitted_model.predict(spectrogram_test)
pred_model = model.predict(spectrogram_test)


###########################
# Visualizing multiple models
# Using plot_model_list we can provide a list of models
# and are returned a set of plotted models with given 
# inputs/targets to compare between.
###########################

# Here we just compare our unfitted model next to that same model after being fitted to data
visualization.plot_predictions([pred_model, pred_fitted_model], input=spectrogram_test, target=response_test)

## Uncomment if you don't have an interactive backend installed
#plt.show()
