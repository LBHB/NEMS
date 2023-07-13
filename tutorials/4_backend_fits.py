# TODO: This entire document
import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential
from nems import visualization

## This indicates that our code is interactive, allowing a matplotlib
## backend to show graphs. Uncomment if you don't see any graphs
#plt.ion()

########################################################
# Setting Backend Parameters
#
# A significant part of creating accurate and optimized models
# is picking parameters to be run through your model. There are
# a lot of different paramenters, here we provide some examples
# and recommendations
#
# Finding the right parameter is both time consuming and often
# inconsistent, we have tools in place to help pick out good
# starts for parameters
########################################################

###########################
# Setting up Demo Data instead of dummy data
# 
# load_demo(): Provides our tuple of training/testing dictionaries
#   This time we will be loading a different data set, and finding a 
#   specific cell that contains data worth testing
#
#   "TAR010c_data.npz": The data parameter to load our different dataset
#   [:, [cid]]: Splitting the dataset to get a specific cells data
#   ['cellid'][cid]: Get cellid at cid for labeling and f+ormatting
###########################
nems.download_demo()
training_dict, test_dict = nems.load_demo("TAR010c_data.npz")

cid=29 # Picking our cell to pull data from
cellid = training_dict['cellid'][cid]
spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response'][:,[cid]]
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response'][:,[cid]]


# Creating a basic Rank2LNSTRF model to test parameters on
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(15, 1)),  # 15 taps, 1 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)

###########################
# Picking your parameters
#
# Currently we have 2 core backends available, with
# their own parameters and options. There are a lot
# of available parameters,
# See more info at: nems.backends.{backend}._fit
#
#   model.fit()
#       - fitter_options: Lets you provide a dictionary of options for the fitter
#       - backend: Specify which backend you will be using
#
#   tf_options:
#       - cost_function: Function used for determining the error rate of our model
#       - early_stopping: Determining when and how to stop your model early to avoid overfitting
#           - delay: Min epochs run before early-stopping is possible
#           - patience: How many epochs to iterate without improvement before we end early
#           - tolerance: Min change in error to be considered improvement
#       - validation_split: Ratio to split our data into testing and validation sets 
#       - learning_rate: The scale your weights will adjust each iteration
#       - epochs: Number of time we want to train our model on a given dataset
#   scipy:
#       - maxiter: Maximum iterations to be performed
#       - ftol: Float-Point precision to stop our iterations at
#   
###########################
backend = 'scipy'
## Uncomment if you have TensorFlow installed
#backend='tf'
if backend=='tf':
    options = {'cost_function': 'squared_error', 'early_stopping_delay': 50, 'early_stopping_patience': 10,
                  'early_stopping_tolerance': 1e-3, 'validation_split': 0,
                  'learning_rate': 5e-3, 'epochs': 2000}
    tf_model = model.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')

else:
    options = {'options': {'maxiter': 100, 'ftol': 1e-4}}
    scipy_model = model.fit(spectrogram_fit, response_fit, fitter_options=options)

###########################
# Model.sample_from_priors()
#
# Returns a single, or list of models with randomized
# parameters
#   - N: The number of models you want to create
###########################
N = 5
sample_list = model.sample_from_priors(N)
fitted_list = []
best_fit = None
r_test_list = []

# We also want to print out each of our models predictions on the same data, to compare
f, ax = plt.subplots(N+2, 1, sharex='col')
ax[0].imshow(spectrogram_test.T,aspect='auto', interpolation='none',origin='lower')
ax[0].set_ylabel('Test stimulus')
ax[1].plot(response_test, label='actual response')
ax[1].set_ylabel('Test response')

# This loops fits each model, compares them to find best fit, and shows how they compare on a graph
# In practice you will want to print entire models, not just predictions, to find your best model
for fitidx, cnn_model in enumerate(sample_list):
    cnn_model.name = f"{cellid}_CNN_fit-{fitidx}"
    fitted_list.append(cnn_model.fit(spectrogram_fit, response_fit, backend=backend, fitter_options=options))
    pred_cnn = cnn_model.predict(spectrogram_test)
    r_test_list.append(np.corrcoef(pred_cnn[:, 0], response_test[:, 0])[0, 1])
    if best_fit is None or best_fit.results.final_error > fitted_list[fitidx].results.final_error:
        best_fit = fitted_list[fitidx]

    ax[fitidx+2].plot(pred_cnn, label='predicted')
    ax[fitidx+2].set_ylabel(f'fit {fitidx}')

# Plotting the model of our best fit
best_fit.name = f"Best Fit, #{fitidx}"
best_fit.plot(spectrogram_test, target=response_test)

for fitidx, cnn in enumerate(fitted_list):
    print(f"CNN fit {fitidx} final E={cnn.results.final_error:.3f} r test={r_test_list[fitidx]:.3f}")

## Uncomment if you don't have an interactive backend installed
#plt.show()
