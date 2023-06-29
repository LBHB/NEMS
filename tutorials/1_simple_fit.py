"""Demonstrates how to fit a basic LN-STRF model using NEMS."""

import numpy as np
import matplotlib.pyplot as plt

from nems import Model
from nems.layers import STRF, DoubleExponential, StateGain
from nems.models import LN_STRF

# Fast-running toy fit options for demonstrations.
options = {'options': {'maxiter': 2, 'ftol': 1e-2}}

# All data should be loaded as 2D numpy arrays. We leave this step up to
# individual users, since data formats are so variable. Built-in model layers
# (see `nems.layers`) expect time to be represented on the first axis, and
# other data dimensions (neurons, spectral channels, etc.) to be represented on
# subsequent axes. For example, spiking responses from a population of neurons
# should have shape (T, N) where T is the number of time bins and N is the
# number of neurons. State data (like pupil size) should have shape (T, M) where
# M is the number of state variables.
# NOTE: shape (S, T, N), where S is the number of samples/trials, is also
#       supported through optional arguments, but that is not covered here.
# TODO: add a separate tutorial covering batched optimization, point to it here.
def my_data_loader(file_path):
    # Dummy function to demonstrate the data format. This resembles LBHB data,
    # which includes a sound stimulus (assumed here to be pre-converted to a
    # spectrogram), spiking responses recorded with implanted electrodes
    # (assumed here to be pre-converted to firing rate / PSTH), and pupil size
    # as a measure of arousal.
    print(f'Loading data from {file_path}, but not really...')

    # Representation of Time & Channels for fake dataset and prefitted model.
    # Actual values may vary
    TIME = 1000
    CHANNELS = 18

    spectrogram = np.random.rand(TIME, CHANNELS)
    response = np.stack(spectrogram[:, :5])
    pupil_size = np.random.rand(TIME, 1)
    
    return spectrogram, response, pupil_size, TIME, CHANNELS

spectrogram, response, pupil_size, TIME, CHANNELS = my_data_loader('/path/to/my/data.csv')


# Build a Model instance, which composes the (typically sequential)
# operations of several Layer instances.
model = Model()
model.add_layers(
    STRF(shape=(25,18)),    # Full-rank STRF, 25 temporal x 18 spectral channels
    DoubleExponential(shape=(5,)) # Double-exponential nonlinearity, 100 outputs
)


# Fit the model to the data. Any data preprocessing should happen separately,
# before this step. No inputs or outputs were specified for the Layers,
# so `input` will be the input to the first layer, the output of the first layer
# will be the input to the second layer and so on. The fitter will try to match
# the final output (of the DoubleExponential Layer in this case) to `target`.
fit_model = model.fit(input=spectrogram, target=response,
                      fitter_options=options)

# Some layers, like StateGain, expect state data to be present in some form
# as an additional model input, so we can specify `pupil_size` as state data.
# (this is not necessary for models with no Layers that require state).
model.add_layers(StateGain(shape=(1,1)))
state_fit = model.fit(input=spectrogram, target=response, state=pupil_size, backend='scipy',
                      fitter_options=options)
 
# By default, `scipy.optimize.minimize` will be used for optimization
# (which can also be specified using the `backend` parameter). This also tells
# the model to use each Layer's standard `evaluate` method for transforming
# inputs (whereas `backend='tf'`, for example, would use
# `Layer.as_tensorflow_layer()`). See `nems.models.base.Model.fit` for
# additional fitting options.

# Predict the response to the inputs using the fitted model.
prediction = state_fit.predict(spectrogram, state=pupil_size)


# Instead of specifying a custom model, we can also use a pre-built model.
# In this case we've also specified `output_name`. Now the output of any
# layer that doesn't specify an output name will be called 'pred' instead
# of the default, 'output'. Our standard LN_STRF model only needs the stimulus
# spectrogram as input and neural response (as firing rates / PSTH) as a target.
prefit_model = LN_STRF(time_bins=TIME, channels=CHANNELS)
#fitted_LN = prefit_model.fit(input=spectrogram, target=response, output_name='pred')
#prediction = prefit_model.predict(spectrogram)

# TODO: Set this up for a pre-fit LN model so that the plots actually look nice
#       without needing to run a long fit in the tutorial.
# Plot the output of each Layer in order, and compare the final output to
# the neural response.
fig = state_fit.plot(spectrogram, state=pupil_size, target=response, figure_kwargs={'figsize': (12,8)})
fig = fitted_LN.plot(spectrogram, target=response, figure_kwargs={'figsize': (12,8)})
fig = fit_model.plot(spectrogram,target=response, figure_kwargs={'figsize': (12,8)})

# Plots out 9 channels from our dataset to a graph before any models have used it
raw_plot, ax = plt.subplots(3, 3, figsize=(12,8))
for i in range(0, 9):
    ax[int(np.ceil(i/3)-1)][i%3].plot(range(0,1000), (spectrogram[:, i]*10).astype(int)) 

plt.show()
