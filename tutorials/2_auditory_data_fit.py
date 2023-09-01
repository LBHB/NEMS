import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential
from nems import visualization

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
###########################
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
model = Model(name="Rank1LNSTRF")
model.add_layers(
    WeightChannels(shape=(18, 1)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(15, 1)),  # 15 taps, 1 spectral channel
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)

# fit the model. Warning: This can take time!
fitted_model = model.fit(spectrogram_fit, response_fit, fitter_options=options, backend='scipy')

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
model2 = Model(name="Rank2LNSTRF")
model2.add_layers(
    WeightChannels(shape=(18, 2)),  # 18 spectral channels->2 composite channels
    FiniteImpulseResponse(shape=(15, 2)),  # 15 taps, 2 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)

# Optimize model parameters with fit data
fitted_model2 = model2.fit(spectrogram_fit, response_fit, fitter_options=options, backend='scipy')

fitted_model.plot(spectrogram_test, target=response_test)
fitted_model2.plot(spectrogram_test, target=response_test)

# Small comparison of model2's first layer coefficients pre vs. post fit.
# model.layers and model.layers[x].(coefficients/prior/bounds) will provide
# useful information regarding the given model.
print(f"""
    FIR coefficients:\n {model2.layers[0].coefficients}
    Fitted FIR coefficients:\n {fitted_model2.layers[0].coefficients}
""")

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
pred_model = fitted_model.predict(spectrogram_test)
pred_model2  = fitted_model2.predict(spectrogram_test)
visualization.plot_predictions({fitted_model.name: pred_model,
                                fitted_model2.name: pred_model2},
                               input=spectrogram_test, target=response_test, correlation=True)
