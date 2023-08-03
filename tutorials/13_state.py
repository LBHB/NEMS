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

## This indicates that our code is interactive, allowing a matplotlib
## backend to show graphs. Uncomment if you don't see any graphs
#plt.ion()

# Basic fitter options for testing
options = {'options': {'maxiter': 50, 'ftol': 1e-4}}

# Dummy data loader. Refer to tutorial 1 for more info
def my_data_loader(file_path=None):
    # Dummy function to demonstrate the data format.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(1000, 18))
    response = spectrogram[:,[1]] - spectrogram[:,[7]]*0.5 + np.random.randn(1000, 1)*1.0 + 0.5

    return spectrogram, response



def my_data_loader2(file_path=None):
    # Dummy function to demonstrate the data format.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(1000, 18))
    response = spectrogram[:,[1]] - spectrogram[:,[7]]*0.5 + np.random.randn(1000, 1)*0.1 + 0.5

    state = np.ones((len(response), 2))
    state[:500,1] = 0
    response = response + state[:,[1]]

    return spectrogram, response, state


fitter_options = {'options': {'maxiter': 100, 'ftol': 1e-5}}

spectrogram, response, state = my_data_loader2('path/to_data.csv')
print(f'Our original dataset size is {spectrogram.shape}')

model_no_state = Model()
model_no_state.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    LevelShift(shape=(1,))
)

model_state = Model()
model_state.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    StateGain(shape=(2,1))
)


model_no_state = model_no_state.fit(spectrogram, target=response,
                                    fitter_options=fitter_options)
model_state = model_state.fit(spectrogram, target=response, state=state,
                                    fitter_options=fitter_options)

pred_no_state = model_no_state.predict(input=spectrogram)
pred_state = model_state.predict(input=spectrogram, state=state)


f,ax=plt.subplots(2,1)
ax[0].plot(response)
ax[0].plot(pred_no_state)
ax[0].set_title(f"Model without state r={correlation(response, pred_no_state):.3f}")

ax[1].plot(response)
ax[1].plot(pred_state)
ax[1].set_title(f"Model with state r={correlation(response, pred_state):.3f}")
