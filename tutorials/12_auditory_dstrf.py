# NOTE: Unfinished, provided comments are at best only partially correct and missing context  
# TODO: Finish filling out data, confirm its correct, and make sure imports are working properly
import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, RectifiedLinear
from nems import visualization

# nems_db import module
from nems.backends import get_backend
from nems.models.dataset import DataSet

# This indicates that our code is interactive, allowing a
# matplotlib backend to show graphs
#plt.ion()

# Importing Demo Data
nems.download_demo()
training_dict, test_dict = nems.load_demo("TAR010c_data.npz")

cid=29 # Picking our cell to pull data from
cellid = training_dict['cellid'][cid]
spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response'][:,[cid]]
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response'][:,[cid]]

########################################################
# Auditory DSTRF
# Creating locally linear data to evaluate an strf model
# 
#   - Create a model and fit it for strf data
#   - Save and provide a dataset containing layer data
#   - Get the jacobian output of our datasets
#   - Visualize our individual layers and data
########################################################

# TODO : use model from CNN fit tutorial, focus on comparison of
#  dstrfs from LN and CNN

# Basic Linear Model
ln = Model()
ln.add_layers(
    WeightChannels(shape=(18, 3)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(10, 3)),  # 15 taps, 1 spectral channels
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False)           # static nonlinearity, 1 output
)
ln.name = f"LN_Model"

# A None linear CNN model
cnn = Model()
cnn.add_layers(
    WeightChannels(shape=(18, 1, 3)),  # 18 spectral channels->2 composite channels->3rd dimension channel
    FiniteImpulseResponse(shape=(15, 1, 3)),  # 15 taps, 1 spectral channels, 3 filters
    RectifiedLinear(shape=(3,)), # Takes FIR 3 output filter and applies ReLU function
    WeightChannels(shape=(3, 1)), # Another set of weights to apply
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False) # A final ReLU applied to our last input
)
cnn.name = f"CNN_Model"

###########################
# DSTRF
# 
#
#
###########################

# Fitting both our models to given fit data from our demo data import
fitted_ln = ln.fit(spectrogram_fit, response_fit, backend='tf')
fitted_cnn = cnn.fit(spectrogram_fit, response_fit, backend='tf')

ln_dstrf = fitted_ln.dstrf(spectrogram_test, D=5, reset_backend=True)
visualization.plot_dstrf(ln_dstrf)

cnn_dstrf = fitted_cnn.dstrf(spectrogram_test, D=15, reset_backend=True)
visualization.plot_dstrf(cnn_dstrf)


## Uncomment if you don't have an interactive backend installed
#plt.show()
