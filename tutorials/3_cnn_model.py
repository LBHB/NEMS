import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, RectifiedLinear
from nems import visualization

# Basic options to quickly fit our models for tf backend
# NOTE: This will be explored in the next tutorial
options = {'cost_function': 'squared_error', 'early_stopping_delay': 50, 'early_stopping_patience': 10,
                  'early_stopping_tolerance': 1e-3, 'validation_split': 0,
                  'learning_rate': 5e-3, 'epochs': 2000}

###########################
# Setting up Demo Data, see last tutorial
###########################
nems.download_demo()
training_dict, test_dict = nems.load_demo()

spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response']
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response']

###########################
# Creating CNN models in NEMS
# 
# One of the core use cases of NEMS is creating CNN's that
# fit onto Spectrograms. Here is a small example:
#
#   Model(): Like any other model used so far
#       - WeightChannels: A linear set of weights applied to our channels
#       - FiniteImpulseResponse: Convolve our linear filters onto inputs with 3 filters
#       - RectifiedLinear: Apply ReLU activation to inputs, along 3 channels
#
###########################
cnn = Model()
cnn.add_layers(
    WeightChannels(shape=(18, 1, 3)),  # 18 spectral channels->2 composite channels->3rd dimension channel
    FiniteImpulseResponse(shape=(15, 1, 3)),  # 15 taps, 1 spectral channels, 3 filters
    RectifiedLinear(shape=(3,)), # Takes FIR 3 output filter and applies ReLU function
    WeightChannels(shape=(3, 1)), # Another set of weights to apply
    RectifiedLinear(shape=(1,)) # A final ReLU applied to our last input
)
cnn.name = "MiniCNN"

# Plotting our CNN before anything any fitting
cnn.plot(spectrogram_fit, target=response_fit)

# We can also see the additional dimension added to our FIR layer
print(f'FIR coefficients: {cnn.layers[1].coefficients}')

# Fit our model to some real data provided by Demo
# We use 'tf' backend to improve training speed.
# See the next tutorial for more info
fitted_cnn = cnn.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')

# Plotting our CNN again after fits
fitted_cnn.plot(spectrogram_fit, target=response_fit)
visualization.simple_strf(fitted_cnn)

# Our FIR Coefficients after we've fit the model
print(f'FIR coefficients: {fitted_cnn.layers[1].coefficients}')

# Now we can predict some new data
pred_cnn = cnn.predict(spectrogram_test)

# A quick print out of our prediction, stimulus, and response data
f, ax = plt.subplots(3, 1, sharex='col')
ax[0].imshow(spectrogram_test.T,aspect='auto', interpolation='none',origin='lower')
ax[0].set_ylabel('Test stimulus')
ax[1].plot(response_test, label='actual response')
ax[1].set_ylabel('Test response')
ax[2].plot(pred_cnn, label='predicted')
ax[2].set_ylabel('Prediction')
