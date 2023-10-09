"""Demonstrates several preprocessing utilities.
NOTE: This script may not actually work yet!
"""
import numpy as np
import matplotlib.pyplot as plt

from nems.preprocessing import (
    indices_by_fraction, split_at_indices, JackknifeIterator)
from nems import Model
from nems.layers import LevelShift, WeightChannels, StateGain
from nems.metrics import correlation
from nems import visualization

# Basic fitter options for testing
fitter_options = {'options': {'maxiter': 100, 'ftol': 1e-5}}

###########################
# States
# Using state data we can specifiy sections of our data
# where one state or another is impacting the provided inputs
#
# In this example we've provided a set of state data that 
# as an example that contains 0's and 1's depending on the current
# "states" our inputs would be in. 
###########################
def my_data_loader2(file_path=None):
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(1000, 18))
    response = spectrogram[:,[1]] - spectrogram[:,[7]]*0.5 + np.random.randn(1000, 1)*0.1 + 0.5

    # Creating a state array that contains 1000 tuples, the first 500 of 1, 0 and rest as 1, 1
    # This allows our model adjust based on state data at a specific index
    state = np.ones((len(response), 2))
    state[:500,1] = 0

    # Shifts our response using our state values
    response = response + state[:,[1]]

    return spectrogram, response, state
spectrogram, response, state = my_data_loader2('path/to_data.csv')

###########################
# State Models
# Utilising state within our models requires us to use specific layers created
# to adjust our model fits based on the use of state data.
#   
# StateGain()
#   shape: Tuple of (size of last dimension of state, size of last dimension of input)
# Applies a gain and offset to shift our model fit to the format of a state dependent input/target
# See more at nems.layers.state
###########################
model_no_state = Model()
model_no_state.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    LevelShift(shape=(1,))
)
model_no_state.name = "Standard Model"
# StateGain is the basis of a our model that needs to account for some state data
# nems.layers.state contains several layer types that can account for state in different ways
model_state = Model()
model_state.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    StateGain(shape=(2,1))
)
model_state.name = "State Model"

fit_model_no_state = model_no_state.fit(spectrogram, target=response,
                                    fitter_options=fitter_options)
fit_model_state = model_state.fit(spectrogram, target=response, state=state,
                                    fitter_options=fitter_options)

pred_no_state = fit_model_no_state.predict(spectrogram)
pred_state = fit_model_state.predict(spectrogram, state=state)

# Here we can see the difference in our models performance between 
# 1. A model with no state information attempting to fit to our data clearly shifted by some state
# 2. A model that contains StateGain and State data can actually fit around data that has states
visualization.plot_predictions([pred_no_state, pred_state], spectrogram, response, correlation=True, display_ratio=1)

###########################
# Jackknife with state
# Using state we can jackknife across our inputs, targets, and state itself.
# Below we cover several ways, and an example of why, to implement state 
# with our jackknife iterators
#
###########################

# Standard State vs No-State jackknifing, fiting, predicting
jackknife_iter = JackknifeIterator(spectrogram, state=state, target=response, samples=5, axis=0, inverse='both')
model_fit_list = jackknife_iter.get_fitted_jackknifes(model_state)
prediction_dataset = jackknife_iter.get_predicted_jackknifes(model_fit_list)

jk_no_state = JackknifeIterator(spectrogram, target=response, samples=5, axis=0, inverse='both')
model_fit_list = jk_no_state.get_fitted_jackknifes(model_no_state)
prediction_dataset = jk_no_state.get_predicted_jackknifes(model_fit_list)

jk_no_state.plot_estimate_error()
jackknife_iter.plot_estimate_error()

# Below is a more manual use of our iterator to create specific fit and prediction to see state in action on a per-iter case
jackknife_multi = JackknifeIterator(spectrogram, state=state, target=response, samples=5, axis=0, inverse='both')
multi_est, multi_val = next(jackknife_multi)

model_fit = model_no_state.fit(multi_est['input'], target=multi_est['target'])
state_model_fit = model_state.fit(multi_est['input'], target=multi_est['target'], state=multi_est['state'])

pred_no_state = model_fit.predict(multi_est['input'])
pred_state = state_model_fit.predict(multi_est['input'], state=multi_est['state'])

visualization.plot_predictions({'jk_no_state':pred_no_state, 'jk_state':pred_state}, multi_est['input'], multi_est['target'], correlation=True, display_ratio=1)