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
options = {'options': {'maxiter': 100, 'ftol': 1e-4}}
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

cid=29
cellid = training_dict['cellid'][cid]
spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response'][:,[cid]]
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response'][:,[cid]]

###########################
# Creating a Higher Ranked Non-linear STRF Model
# 
# Instead of creating a linear model using Weighted Channels, we are
# created a more complex non-linear model and using real data to measure
# it's effectiveness
#
#   FiniteImpulseResponse: Convolve linear filter(s) with inputs
#       - Can specifiy "Time-Bins", "Rank/Input-Channels", "Filters", etc...
#   DoubleExponential: Sets inputs as a exponential function to the power of some constant
# See more at: nems.layers
###########################
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 3)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(10, 3)),  # 15 taps, 1 spectral channels
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False)           # static nonlinearity, 1 output
)
model.name = f"{cellid}-Rank3LNSTRF"
model = model.sample_from_priors()

# A plot of our model before anything has been fit
# NOTE: It will be very hard to see the blue line, try zooming in very close
#model.plot(spectrogram_fit, target = response_fit)

# Some other important information before fitting our model.
# This time looking at how our FIR layer interacts before/after fits
print(f"""
    Rank2LNSTRF Layers:\n {model.layers}
    FIR shape:\n {model.layers[1].shape}
    FIR priors:\n {model.layers[1].priors}
    FIR bounds:\n {model.layers[1].bounds}
    FIR coefficients:\n {model.layers[1].coefficients}
""")

# Fitting our Spectrogram data to our target response data
fitted_model = model.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')
pred_ln = fitted_model.predict(spectrogram_test)
r_test_ln = np.corrcoef(pred_ln[:, 0], response_test[:, 0])[0, 1]

###########################
# Built- in visualization
# We have created several utilities for plotting and
# visualization of NEMS objects
#
#   plot_model: Creates a variety of graphs and plots for a given model and layers
#   simple_strf: Gets FIR and WeightChannels from Model
###########################
visualization.plot_model(fitted_model, spectrogram_test, response_test)
#visualization.simple_strf(fitted_model)

cnn = Model()
cnn.add_layers(
    WeightChannels(shape=(18, 1, 3)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(10, 1, 3)),  # 15 taps, 1 spectral channels
    RectifiedLinear(shape=(3,)),
    WeightChannels(shape=(3, 1)),  # 18 spectral channels->1 composite channels
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False),
)

# TODO: package the procedure for fits from multiple ICs into a tidy function

fitcount=5
cnn_list = cnn.sample_from_priors(fitcount)
fitted_cnn_list = []
r_test = []
for fitidx, cnn in enumerate(cnn_list):
    cnn.name = f"{cellid}_CNN_fit-{fitidx}"
    fitted_cnn = cnn.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')
    visualization.plot_model(fitted_cnn, spectrogram_test, response_test)
    fitted_cnn_list.append(fitted_cnn)

    # TODO: function for computing r_test
    pred_cnn = fitted_cnn.predict(spectrogram_test)
    r_test.append(np.corrcoef(pred_cnn[:, 0], response_test[:, 0])[0, 1])

print(f"LN final E={fitted_model.results.final_error:.3f} r test={r_test_ln:.3f}")

for fitidx, cnn in enumerate(fitted_cnn_list):
    print(f"CNN fit {fitidx} final E={cnn.results.final_error:.3f} r test={r_test[fitidx]:.3f}")
