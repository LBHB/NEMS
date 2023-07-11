"""Demonstrates how to create and fit a basic model using NEMS."""

import numpy as np
import matplotlib.pyplot as plt

from nems import Model
from nems.layers import LevelShift, WeightChannels

## This indicates that our code is interactive, allowing a matplotlib
## backend to show graphs. Uncomment if you don't see any graphs
#plt.ion()

# Fast-running toy fit options for demonstrations.
options = {'options': {'maxiter': 2, 'ftol': 1e-2}}

########################################################
# Typical nems.Layers data structure:
# layer_data_example(TIME, CHANNEL)
#
# (X Axis, 2D Numpy Array): TIME can be any representation relevent to your data
# (Y Axis, 2D Numpy Array): CHANNEL is some seperation of data inputs ie... Neuron, Sepctral Channel, etc...
#
# Examples: 
#   1. Spiking responses of neurons is set up as shape(TIME, NEURONS)
#   2. Pupil Size is represented as shape(TIME, PUPIL_STATES)
# See more at: https://temp.website.net/nems.Layers
########################################################

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

###########################
# Model can take in a (Usually) sequential set of layers
#   WeightChannels: Computes linear weights of input channels
#                   comparable to a Dense layer.
#   LevelShift: Applies a scalar shift to all inputs
# See more at: https://temp.website.net/nems_model
###########################
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    LevelShift(shape=(1,)) # WeightChannels will provide 1 input to shift
)

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

# Plotting our model via .plot
model.plot(spectrogram, target=response)

# Viewing various data from our model and it's layers
print(f"""
    Model Layers:\n {model.layers}
    layer shapes:\n {model.layers[0].shape}
    layer priors:\n {model.layers[0].priors}
    layer bounds:\n {model.layers[0].bounds}
""")

###########################
# Fitting our data to the model and layers
#   Input: Our 2D numpy array representing some form of TIME x CHANNEL structure
#   Target: The model will try to fit Input to Target using layers and fitter options
#   Fitter Options: Provide options for modifying how our model might fit the data
# See more at: https://temp.website.net/nems_model_fit
###########################
fit_model = model.fit(input=spectrogram, target=response,
                      fitter_options=options)

# We are now viewing our data after it has fit the input to our target
fit_model.plot(spectrogram, target=response)

print(f"""
    layer shapes:\n {fit_model.layers[0].shape}
    layer priors:\n {fit_model.layers[0].priors}
    layer bounds:\n {fit_model.layers[0].bounds}
    pre-fit weighted layer values:\n {model.layers[0].coefficients}
    post-fit weighted layer values:\n {fit_model.layers[0].coefficients}

""")
 
###########################
# Sets the model to predict data provided after fitting the model
#   Backend(Not seen here): Can be specified to decide an Optimizer, default = scipy.optimize.minimize()
#       - Other options exist such as tf for tensorflow
#   Spectrogram: The data we will performing a prediction with
# See more at: https://temp.website.net/nems_model_predict and
# See more at: https://temp.website.net/nems_backends
###########################
state_fit = model.predict(input=spectrogram, backend='scipy',
                      fitter_options=options)

# Finally viewing our last fit and relevant data
fig = state_fit.plot(spectrogram)

## Uncomment if you don't have an interactive backend installed
#plt.show()