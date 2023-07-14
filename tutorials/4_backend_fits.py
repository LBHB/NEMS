# TODO: This entire document
import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model, Model_List
from nems.layers import WeightChannels, FiniteImpulseResponse, RectifiedLinear
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
    WeightChannels(shape=(18, 1, 3)),  # 18 spectral channels->2 composite channels->3rd dimension channel
    FiniteImpulseResponse(shape=(15, 1, 3)),  # 15 taps, 1 spectral channels, 3 filters
    RectifiedLinear(shape=(3,)), # Takes FIR 3 output filter and applies ReLU function
    WeightChannels(shape=(3, 1)), # Another set of weights to apply
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False) # A final ReLU applied to our last input
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
backend='tf'
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

###########################
# Model_List
# 
# A way of creating and modifying groups of models, current
# use is limited.
#   __init__(): Can create a group via existing list or providing sample amount
#   fit_models(): Fits group of models, using same inputs as normal fit
#   plot_models(): Plots all models in group
###########################
list_of_models = Model_List(model, samples=N)

existing_list_of_models = Model_List(sample_list)

list_of_models.fit_models(spectrogram_fit, response_fit, backend=backend, fitter_options=options)

###########################
# Visualizing all our models
#
# With so many models being created and compared, it's worth taking
# a look at how they all compare and get an idea of how parameters change
# your models outcome.
#
# Here we're looping through our model list, comparing values to find the
# best fitted model and showing a few graphs to see the differences.
# !!! In practice, you will want to print the full model plots to compare models
###########################
list_of_models.plot_models(input=spectrogram_test, target=response_test)

## Something like this will be more useful in practice
# list_of_models.plot_models(input=spectrogram_test, target=response_test, plot_comparitive=False, plot_full=True)

# Plotting the model of our best fit
best_fit = list_of_models.get_best_fit
best_fit.name = f"Best Fit"
best_fit.plot(spectrogram_test, target=response_test)

# Printing out error rates and correlation coeffecients of our models to see how they compare
# Using .get_list we can retrieve the raw list of models
for fitidx, cnn in enumerate(list_of_models.get_list):
    pred_cnn = cnn.predict(spectrogram_test)
    print(f"CNN fit {fitidx} final E={cnn.results.final_error:.3f} r test={np.corrcoef(pred_cnn[:, 0], response_test[:, 0])[0, 1]:.3f}")

## Uncomment if you don't have an interactive backend installed
#plt.show()
