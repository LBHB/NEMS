import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, RectifiedLinear
from nems import visualization
from nems.metrics import correlation

## This indicates that our code is interactive, allowing a matplotlib
## backend to show graphs. Uncomment if you don't see any graphs
#plt.ion()

# Basic options to quickly fit our models for tf backend
# NOTE: This will be explored in the next tutorial
options = {'options': {'maxiter': 100, 'ftol': 1e-4}}
options = {'cost_function': 'squared_error', 'early_stopping_delay': 50, 'early_stopping_patience': 100,
                  'early_stopping_tolerance': 1e-3, 'validation_split': 0,
                  'learning_rate': 5e-3, 'epochs': 2000}


###########################
# Setting up Demo Data instead of dummy data
# 
# load_demo(): Provides our tuple of training/testing dictionaries
#   This time we will be loading a different data set, and finding a 
#   specific cell that contains data worth testing
#
#   "TAR010c_data.npz": The data parameter to load our different dataset
#   [:, [cid]]: Splitting the dataset to get a specific cells data
#   ['cellid'][cid]: Get cellid at cid for labeling and formatting
###########################
nems.download_demo()
training_dict, test_dict = nems.load_demo("TAR010c_data.npz")

cid=29 # Picking our cell to pull data from
cellid = training_dict['cellid'][cid]
spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response'][:,[cid]]
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response'][:,[cid]]

###########################
# Creating CNN models in NEMS
# 
# One of the core use cases of NEMS is creating CNN's we can use
# to model and predict spectrograms given sets of data
#
#   Model(): Like any other model used so far
#       - WeightChannels: A linear set of weights applied to our channels
#       - FiniteImpulseResponse: Convolve our linear filters onto inputs with 3 filters
#       - RectifiedLinear: Apply ReLU activation to inputs, along 3 channels
###########################
# TODO: replace "cnn_simple" with "ln" or "ln_model"
cnn_simple = Model()
cnn_simple.add_layers(
    WeightChannels(shape=(18, 3)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(10, 3)),  # 15 taps, 1 spectral channels
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False)           # static nonlinearity, 1 output
)
cnn_simple.name = f"{cellid}-Rank3LNSTRF-3Layer"

# A similar model, with a few more layers this time.
cnn = Model()
cnn.add_layers(
    WeightChannels(shape=(18, 1, 3)),  # 18 spectral channels->2 composite channels->3rd dimension channel
    FiniteImpulseResponse(shape=(15, 1, 3)),  # 15 taps, 1 spectral channels, 3 filters
    RectifiedLinear(shape=(3,)), # Takes FIR 3 output filter and applies ReLU function
    WeightChannels(shape=(3, 1)), # Another set of weights to apply
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False) # A final ReLU applied to our last input
)
cnn.name = f"{cellid}-Rank3LNSTRF-5Layer"


###########################
# Initializing our model parameters
#
# We've yet to cover how to set up model parameters and 
# fitter options, so we're using a tool to randomize these.
#
#   sample_from_priors(): Randomizes model parameters & fitter options
###########################
cnn_simple = cnn_simple.sample_from_priors()
cnn = cnn.sample_from_priors()

# Plotting and comparing our 2 CNN's before anything any fitting
cnn.plot(spectrogram_test, target=response_test)

# We can specifiy some plotting parameters by providing figure_kwargs to our plot
cnn_simple.plot(spectrogram_test, target=response_test, figure_kwargs={'facecolor': 'papayawhip'})


# We can also see the additional dimension added to our FIR layer,
# compared to how our simpler model is set up
print(f'FIR coefficient shape: {cnn_simple.layers[1].coefficients.shape}')
print(f'FIR coefficient shape: {cnn.layers[1].coefficients.shape}')

# Fit our model to some real data provided by Demo
# We use 'tf' backend to improve training speed.
# See the next tutorial for more info
fitted_cnn_simple = cnn_simple.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')
fitted_cnn_simple.name += "-Fitted"

# We can also compare how each model predicts the same information, given more or less layers
fitted_cnn = cnn.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')
fitted_cnn.name += "-Fitted"

# Plotting our CNN's again after fits
fitted_cnn_simple.plot(spectrogram_test, target=response_test, figure_kwargs={'facecolor': 'papayawhip'})
fitted_cnn.plot(spectrogram_test, target=response_test)

# Our FIR Coefficients after we've fit the model
print(f'FIR coefficients: {fitted_cnn.layers[1].coefficients}')

# Now we can predict some new data
pred_ln = fitted_cnn_simple.predict(spectrogram_test)
pred_cnn = fitted_cnn.predict(spectrogram_test)
pred_cc_ln = correlation(pred_ln, response_test)
pred_cc_cnn = correlation(pred_cnn, response_test)

results_cnn = np.corrcoef(pred_cnn[:, 0], response_test[:, 0])[0, 1]


# A quick print out of our prediction, stimulus, and response data for our 5-layer model
f, ax = plt.subplots(3, 1, sharex='col')
ax[0].imshow(spectrogram_test.T,aspect='auto', interpolation='none',origin='lower')
ax[0].set_ylabel('Test stimulus')
ax[1].plot(response_test, label='actual response')
ax[1].plot(pred_cnn_simple, label='predicted')
ax[1].set_ylabel('Test response')
ax[1].set_title(f"LN correlation={pred_cc_ln:.3f}")
ax[1].legend()
ax[2].plot(response_test, label='actual response')
ax[2].plot(pred_cnn, label='predicted')
ax[2].set_ylabel('Prediction')
ax[2].set_title(f"CNN correlation={pred_cc_cnn:.3f}")
ax[2].legend()
plt.tight_layout()

## Uncomment if you don't have an interactive backend installed
#plt.show()
