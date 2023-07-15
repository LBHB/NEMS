import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential, RectifiedLinear
from nems import visualization

## This indicates that our code is interactive, allowing a matplotlib
## backend to show graphs. Uncomment if you don't see any graphs
#plt.ion()

# Basic options to quickly fit our models
options = {'cost_function': 'squared_error', 'early_stopping_delay': 50, 'early_stopping_patience': 100,
                  'early_stopping_tolerance': 1e-3, 'validation_split': 0,
                  'learning_rate': 5e-3, 'epochs': 2000}

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
training_dict, test_dict = nems.load_demo("TAR010c_data.npz")

spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response']
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response']


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
visualization.plot_model(fitted_cnn, spectrogram_test, response_test)
