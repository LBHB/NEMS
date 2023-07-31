import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential, RectifiedLinear
from nems import visualization

## This indicates that our code is interactive, allowing a matplotlib
## backend to show graphs. Uncomment if you don't see any graphs
#plt.ion()

# More specific options for our model here. You will 
# need TensorFlow installed, see install instructions. 
# We also recommend having a GPU set up for these fits
options = {'cost_function': 'squared_error', 'early_stopping_delay': 50, 'early_stopping_patience': 100,
                  'early_stopping_tolerance': 1e-3, 'validation_split': 0,
                  'learning_rate': 5e-3, 'epochs': 2000}

########################################################
# Auditory Population Fitting
# A more recent focus has been the fitting of larger populations of inputs.
# Functionally, we are fitting and plotting our data as normal, but the 
# plotted data is impossible to really understand
#
# This type of data will take much longer to process and the resulting data
# will be much more dense than our previous tutorials.
#
# In this example, the given download data 'spectrogram' 
# provides 55 neurons to fit.
########################################################

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
# Download TAR010c_data.npz, which provides 55 Neurons of input data to fit
training_dict, test_dict = nems.load_demo("TAR010c_data.npz")

spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response']
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response']

# Here we can see the dimensions of our input
print(f'The shape of our input data is: {spectrogram_fit.shape}')
print(f' and the shape of our target data is: {response_fit.shape}')

# Creating a typical CNN model. See more at: cnn_model tutorial
cnn = Model()
cnn.add_layers(
    WeightChannels(shape=(18, 1, 30)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(15, 1, 30)),  # 15 taps, 1 spectral channels
    RectifiedLinear(shape=(30,)),
    WeightChannels(shape=(30, 55)),  # 18 spectral channels->1 composite channels
    RectifiedLinear(shape=(55,), no_shift=False, no_offset=False),
)
cnn.name = "Population_CNN"
cnn = cnn.sample_from_priors()

fitted_cnn = cnn.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')

# As you can see here, we have many overlapping inputs attempting to fit to our target data
visualization.plot_model(fitted_cnn, spectrogram_test, response_test)

pred_cnn = fitted_cnn.predict(spectrogram_test)

# A quick print out of our prediction, stimulus, and response data
# Again all the information here is basically unreadable at the moment
f, ax = plt.subplots(3, 1, sharex='col')
ax[0].imshow(spectrogram_test.T,aspect='auto', interpolation='none',origin='lower')
ax[0].set_ylabel('Test stimulus')

# When we have a large population of inputs/targets, it's a good idea to represent these as something other then graphs
ax[1].imshow(response_test.T, label='actual response', aspect='auto', interpolation='none', origin='lower')
ax[1].set_ylabel('Test response')
ax[2].imshow(pred_cnn.T, label='predicted', aspect='auto', interpolation='none', origin='lower')

## Uncomment if you don't have an interactive backend installed
#plt.show()
