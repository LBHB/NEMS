"""Demonstrates how to create and fit a basic model using NEMS."""

import numpy as np
import matplotlib.pyplot as plt

from nems import Model
from nems.layers import LevelShift, WeightChannels
from nems import visualization

# NOTE For all tutorials:
# If you are having trouble viewing graphs, add plt.ion() or plt.show() to script e.g.
plt.ion()

# Fast-running toy fit options for demonstrations.
# See tutorial 4. for more
options = {'options': {'maxiter': 2, 'ftol': 1e-2}}

###########################
# A dummy representation of potential LBHB data
#   Spectrogram: A representation of sound stimulus and neuron response
#   Response: Fake target values that our model will try to fit. These
#             are values relative to our spectrogram, so you can see how
#             our model changes the input.
###########################
def my_data_loader(file_path):
    print(f'Loading data from {file_path}, but not really...')

    # TIME = Representation of some x time to use in our layers
    # CHANNELS = Representation of some y channels for # of inputs in our layers
    TIME = 1000
    CHANNELS = 18

    # Creation of random 2D numpy array with X time representation and Y channel representations
    spectrogram = np.random.randn(TIME, CHANNELS)

    # Creating our target data to fit the model to. The first neuron channel - the
    # values of the 5th neuron channel + some random values, all shifted by 0.5
    response = spectrogram[:,[1]] - spectrogram[:,[5]]*0.1 + np.random.randn(1000, 1)*0.1 + 0.5
    
    return spectrogram, response, TIME, CHANNELS
# Create variables from our data import function
spectrogram, response, TIME, CHANNELS = my_data_loader('/path/to/my/data.csv')

############GETTING STARTED###############
###########################
# Models
# Models are at their core an object that can be built with a few simple steps
# to start:
#   1. Create a Model()
#   2. Add layers to our model based on your needs
#
# Using models is also fairly simple at its core:
#   1. Fit models using Model.fit()
#   2. Predict models using Model.predict()
# There is much more going on with models, but this should let you get started
###########################
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    LevelShift(shape=(1,))  # WeightChannels will provide 1 input to shift
)

# fit model parameters using the input/target data
fitted_model = model.fit(input=spectrogram, target=response,
                         fitter_options=options)

# generate a prediction from the fitted model
prediction = fitted_model.predict(spectrogram)




############ADVANCED###############
########################################################
# Typical nems.Layers data structure:
# layer_data_example(TIME, CHANNEL)
#
# TIME can be any representation relevant to your data
# CHANNEL is some space representing one or more dimensions of inputs,
#   e.g., neuron, spectral channel, etc...
#
# Examples: 
#   1. Tutorial #2: Spiking responses of a single neuron is set up as shape(TIME, 1)
#   2. Tutorial #5: Spiking responses of multiple neuron is set up as shape(TIME, NEURON)
#   3. Tutorial #13: An additional state input is defined as (TIME, 1) and integrated with
#      a sensory response
########################################################


###########################
# Model can take in a (Usually) sequential set of layers
#   WeightChannels: Computes linear weights of input channels
#                   comparable to a Dense layer.
#   LevelShift: Applies a scalar shift to all inputs
###########################
model = Model(name='SimpleLinModel')
model.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    LevelShift(shape=(1,)) # WeightChannels will provide 1 input to shift
)

###########################
# Fitting our data to the model and layers
#   Input: Our 2D numpy array representing some form of TIME x CHANNEL structure
#   Target: The model will try to fit Input to Target using layers and fitter options
#   Fitter Options: Provide options for modifying how our model might fit the data
#
# NOTE: When fitting(& Predicting) our model, a copy of our original model is created
# with the fit or prediction. This copy is returned and needs to be saved to be used.
# For example: If we called our model again below, nothing will have changed. We need
# to call fitted_model to see any changes. 
###########################
fitted_model = model.fit(input=spectrogram, target=response,
                      fitter_options=options)

###########################
# Viewing the model and data
#   Model.plot(): Takes input and many possible KWarg's to plot layer data and evaluation outputs
#
# Here we will plot our current data, and it's target before we actually fit the model.
# You can see our model has done nothing to the blue line whose output is clearly false.
# Our blue and orange lines should be similar or the same if our model is working.
#
# We can also view a lot of data directly from the model and it's layers, these can be
# seen inside something like IPython, or printed out directly as well
###########################

# plot_data from the vizualization library lets us quickly look at time
# series like the response and prediction.
visualization.plot_data(prediction, label='Prediction', title='Model Prediction', target=response)

# We are now viewing our data after it has fit the input to our target
fitted_model.plot(spectrogram, target=response)
