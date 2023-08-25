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
# load_demo(): Provides a tuple of training/testing dictionaries, each
#   containing:
#   spectrogram: cochleagram of a natural sound stimulus, sampled at
#      18 log-spaced spectral channels and 100 s^-1 in time
#   response: the PSTH / time-varying spike rate of the recorded neuron,
#      also sampled at 100 s^-1 and aligned with the spectrogram in time.
#   cellid: string identifier or recorded neuron (useful for subsequent
#      large datasets.
#  Other datasets are accessible using load_demo, these can be specified
#   as a parameter: load_demo(test_dataset)
###########################
nems.download_demo()
training_dict, test_dict = nems.load_demo()

spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response']
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response']

############GETTING STARTED###############
###########################
# Fit a Rank-1 linear/nonlinear model to the actual neural data.
#
# We first use a very simple model to demonstrate fitting.
#
###########################
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # 18 spectral channels->1 composite channels
)
model.name = "Rank1LNSTRF"

# fit the model. Warning: This can take time!
fitted_model = model.fit(spectrogram_fit, response_fit, fitter_options=options, backend='scipy')

# display model fit parameters and prediction of test data
visualization.plot_model(fitted_model, spectrogram_test, response_test)


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
###########################
model2 = Model()
model2.add_layers(
    WeightChannels(shape=(18, 2)),  # 18 spectral channels->2 composite channels
    FiniteImpulseResponse(shape=(15, 2)),  # 15 taps, 2 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)
model2.name = "Rank2LNSTRF-PreFit"

# Optimize model parameters with fit data
fitted_model2 = model2.fit(spectrogram_fit, response_fit, fitter_options=options, backend='scipy')
fitted_model2.name = "Rank2LNSTRF"

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
model2.plot(spectrogram_test, target = response_test)

fitted_model2.plot(spectrogram_test, target = response_test)

# Some other important information before fitting our model.
# This time looking at how our FIR layer interacts before/after fits
print(f"""
    Rank2LNSTRF Layers:\n {model2.layers}
    FIR shape:\n {model2.layers[1].shape}
    FIR priors:\n {model2.layers[1].priors}
    FIR bounds:\n {model2.layers[1].bounds}
    FIR coefficients:\n {model2.layers[1].coefficients}
""")

# Seeing our FIR inputs after our fit has been applied
print(f"""
    Fitted Rank2LNSTRF Layers:\n {fitted_model2.layers}
    Fitted FIR shape:\n {fitted_model2.layers[1].shape}
    Fitted FIR priors:\n {fitted_model2.layers[1].priors}
    Fitted FIR bounds:\n {fitted_model2.layers[1].bounds}
    Fitted FIR coefficients:\n {fitted_model2.layers[1].coefficients}
""")

# evaluate model on test stimulus
pred_model = fitted_model.predict(spectrogram_test)


###########################
# Visualizing groups of data
# plot_predictions
# Allows us to plot groups of predictions together and compare them
#   predictions: A single, or list of, predictions to plot
#   input: The given input to our predictions, if we wish to show it
#   target: Our target value, if we want to separately graph this alone
#   correlation: If true, adds correlation coefficients to our model titles
#   show_titles: If true, will add titles to the top of each graph
#
# NOTE: If predictions are a dictionary, the keys will be titles for each prediction
###########################

# For example, we can compare our original unfitted model next to our fitted models results
pred_model2 = model2.predict(spectrogram_test)
pred_fitted_model2  = fitted_model2.predict(spectrogram_test)
visualization.plot_predictions({model2.name: pred_model2,
                                fitted_model.name: pred_model,
                                fitted_model2.name: pred_fitted_model2},
                               input=spectrogram_test, target=response_test, correlation=True)

## Uncomment if you don't have an interactive backend installed
#plt.show()
