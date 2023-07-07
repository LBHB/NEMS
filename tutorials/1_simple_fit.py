"""Demonstrates how to fit a basic LN-STRF model using NEMS."""

import numpy as np
import matplotlib.pyplot as plt

from nems import Model
from nems.layers import STRF, DoubleExponential, StateGain
from nems.models import LN_STRF

# This indicates that our code is interactive, allowing a
# matplotlib backend to show graphs
plt.ion()

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
#   Pupil_size: A measurement of arousal
###########################
def my_data_loader(file_path):
    print(f'Loading data from {file_path}, but not really...')

    # TIME = Representation of some x time to use in our layers
    # CHANNELS = Representation of some y channels for # of inputs in our layers
    TIME = 1000
    CHANNELS = 18

    # Creation of random 2D numpy array with X time representation and Y channel representations
    spectrogram = np.random.rand(TIME, CHANNELS)
    # Using our Spectrogram to create a target set of data that our model will attempt to fit
    response = np.stack(spectrogram[:, :5])
    # An example of using states to fit our model together
    pupil_size = np.random.rand(TIME, 1)
    
    return spectrogram, response, pupil_size, TIME, CHANNELS

# Create variables from our data import function
spectrogram, response, pupil_size, TIME, CHANNELS = my_data_loader('/path/to/my/data.csv')

###########################
# Model can take in a (Usually) sequential set of layers
#   STRF: Our inital layer, with provided channels
#   DoubleExponential: The second layer taking the previous layer as input
# See more at: https://temp.website.net/nems_model
###########################
model = Model()
model.add_layers(
    STRF(shape=(25,18)),    # Full-rank STRF, 25 temporal x 18 spectral channels
    DoubleExponential(shape=(5,)) # Double-exponential nonlinearity, 100 outputs
)

###########################
# Fitting our data to the model and layers
#   Input: Our 2D numpy array representing some form of TIME x CHANNEL structure
#   Target: The model will try to fit Input to Target using layers and fitter options
#   Fitter Options: Provide options for modifying how our model might fit the data
# See more at: https://temp.website.net/nems_model_fit
###########################
fit_model = model.fit(input=spectrogram, target=response,
                      fitter_options=options)

###########################
# Certain layers may request or need more information ex:
#   StateGain: Requires some form of state data, for example Pupil Size
# Only specific models using these layers will require state
# See more at: https://temp.website.net/nems_layers_stategain
###########################
model.add_layers(StateGain(shape=(1,1)))
state_fit = model.fit(input=spectrogram, target=response, state=pupil_size, backend='scipy',
                      fitter_options=options)
 
###########################
# Sets the model to predict data provided after fitting the model
#   Backend(Not seen here): Can be specified to decide an Optimizer, default = scipy.optimize.minimize()
#       - Other options exist such as tf for tensorflow
#   Spectrogram: The data we will performing a prediction with
#   State: Provided state for StateGain layer
# See more at: https://temp.website.net/nems_model_predict and
# See more at: https://temp.website.net/nems_backends
###########################
prediction = state_fit.predict(spectrogram, state=pupil_size)

###########################
# Pre-built models can be used as well
#   LN_STRF: An already built model for use with time_bins x channels data/axes
# Model can then be fit and predicted in the same way as your own model
# See more at: https://temp.website.net/nems_model_lnstrf
###########################
prefit_model = LN_STRF(time_bins=TIME, channels=CHANNELS)
fitted_LN = prefit_model.fit(input=spectrogram, target=response, output_name='pred')
prefit_prediction = prefit_model.predict(spectrogram)

# TODO: Set this up for a pre-fit LN model so that the plots actually look nice
#       without needing to run a long fit in the tutorial.
# Plot the output of each Layer in order, and compare the final output to
# the neural response. ion will use whatever backend is currently available

fig = state_fit.plot(spectrogram, state=pupil_size, target=response, figure_kwargs={'figsize': (12,8)})
fig = fitted_LN.plot(spectrogram, target=response, figure_kwargs={'figsize': (12,8)})
fig = fit_model.plot(spectrogram,target=response, figure_kwargs={'figsize': (12,8)})

# Plots out 9 channels from our spectrogram dataset to graphs before any models have used it
raw_plot, ax = plt.subplots(3, 3, figsize=(12,8))
for i in range(0, 9):
    ax[int(np.ceil(i/3)-1)][i%3].plot(range(0,TIME), (spectrogram[:, i]*10).astype(int)) 
## Uncomment if you don't have an interactive backend installed
#plt.show()