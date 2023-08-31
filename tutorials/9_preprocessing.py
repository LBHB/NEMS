"""Demonstrates several preprocessing utilities.
NOTE: This script may not actually work yet!

"""
import numpy as np
import matplotlib.pyplot as plt

from nems.preprocessing import (
    indices_by_fraction, split_at_indices, JackknifeIterator)
from nems import Model
from nems.layers import LevelShift, WeightChannels
from nems.metrics import correlation

# Basic fitter options for testing
options = {'options': {'maxiter': 50, 'ftol': 1e-4}}

# Dummy data loader. Refer to tutorial 1 for more info
def my_data_loader(file_path=None):
    # Dummy function to demonstrate the data format.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(1000, 18))
    response = spectrogram[:,[1]] - spectrogram[:,[7]]*0.5 + np.random.randn(1000, 1)*1.0 + 0.5

    return spectrogram, response



########################################################
# Creating validation/estimation data
#
# We've created tools that allow you to split your formatted data into
# "Estimation" and "Validation" sets, used to fit and validate models
#
# In this example, we are creating 90% Estimation data, 10% Validation data
# using different tools to find out where to split data, and give you options
# for how to do so.
########################################################
spectrogram, response = my_data_loader('path/to_data.csv')

###########################
# indices_by_fraction
# Allows you to find the point to split your data
#   - response: The dataset we are finding indices for
#   - fraction: The ratio you want to split the data by
#   - axis: Which axis to split the data by
###########################
idx_before, idx_after = indices_by_fraction(response, fraction=0.9, axis=0)

###########################
# split_at_indices
# Actually splits data based on indices, creating copies of the arrays
#   - spectrogram: The dataset we are splitting
#   - idx1, idx2: The points to split our data at
#   - axis: Which axis to split the data by
###########################
est_spectrogram, val_spectrogram = split_at_indices(
    spectrogram, idx1=idx_before, idx2=idx_after, axis=0
    )
est_response, val_response = split_at_indices(
    response, idx1=idx_before, idx2=idx_after, axis=0
)

# Alternatively, we could use 100% of the duration of the data, but assign
# 90% of neurons to the estimation set and 10% of neurons to validation
# (by changing `axis=0` to `axis=1`).
est_neurons, val_neurons = indices_by_fraction(response, fraction=0.9, axis=1)
est_response, val_response = split_at_indices(
    response, idx1=est_neurons, idx2=val_neurons, axis=1
    )

########################################################
# Jackknifing
# The process of creating many sub-arrays from large sets of data. 
# We do this by creating a mask of indicies(0's and 1's) and deleting, 
# or taking, from an dataset using that mask.
#
# Jackknifing is typically used as a way to measure and compare
# a models stability over the course of many smaller fits when
# compared to it's target data.
#
########################################################

###########################
# JackknifeIterator
# The core of jackknifing is done through our JackknifeIterator.
# This class allows you to create indicies for a given dataset
# and return smaller subsets that are iterated over the main data
# and given indicies.
#
#   input: Input data we wish to modify
#   target: Target data we wish to modify
#   samples: The number of samples to create masks of
#   axis: The axis on which we wish to create our mask and modify data from
#   Inverse: Indicate if you would like the inverse masks of given data
###########################

############GETTING STARTED###############
spectrogram, response = my_data_loader('path/to_data.csv')
print(f'Our original dataset size is {spectrogram.shape}')
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    LevelShift(shape=(1,)) # WeightChannels will provide 1 input to shift
)

# This creates an iterator that we can use to modify our data with 5 samples at axis 0
jack_samples = 5
jackknife_iterator = JackknifeIterator(spectrogram, target=response, samples=jack_samples, axis=0)

# We can then fit this iterator directly and return a list of fitted model
# This will fit range(0, samples) models with given masks before returning a list of fitted models
model_fit_list = jackknife_iterator.get_fitted_jackknifes(model)
print(len(model_fit_list))

# Predictions can be done across the entire list of fitted models, using our sets of masks as well
prediction_set = jackknife_iterator.get_predicted_jackknifes(model_fit_list)
predictions = prediction_set['prediction']
targets = prediction_set['target']
print(len(predictions))




############ADVANCED###############
# Calling next will return a single sub-array of our data from our jackknife masks
jackknife_iterator.reset_iter()
jackknife_single = next(jackknife_iterator)
model.fit(jackknife_single['input'], jackknife_single['target'])

# The object returned on each iteration is actually a Dataset object which can used like below:
print(type(jackknife_single))
input_data = jackknife_single['input']
target_data = jackknife_single['target']

# Each time we call next, we iterate over our object and can see it's current index via:
print(f'Current index: {jackknife_iterator.index} \n')

# We can reset our index to start back at 0, normally this index will loop back to 0 after 
# 'Samples' # of iterations but it may be good to keep it at 0 when starting a new process
jackknife_iterator.reset_iter()

# We can also find and use our sample amounts to iterate over the intended amount of masks
jackknife_dataset = [next(jackknife_iterator) for x in range(jackknife_iterator.samples)]

print(f'Our index: {jackknife_iterator.index} \n Dataset size: {len(jackknife_dataset)} \n Data shape: {jackknife_dataset[0]["input"].shape}\n')

# Once we have a fitted list, we can predict on one of our models and compare it to another, or to
# inverse data.

# Getting the inverse masks requires you to pass 'both' to our inverse attribute. Then when you iterate,
# you will recieve a tuple that gives you base and inverse for each mask
jackknife_iterator_both = JackknifeIterator(spectrogram, samples=jack_samples, axis=0, target=response, inverse='both')
est, val = next(jackknife_iterator_both)

# Now we can compare two different models predictions
print(f" Model 1 vs Model 2: {correlation(model_fit_list[0].predict(val['input']), model_fit_list[1].predict(val['input']))}")

# We can also compare our models prediction to the target data
print(f" Model 1 vs Validation 1: {correlation(model_fit_list[0].predict(val['input']), val['target'])}")

# We can also choose to shuffle the groups of indicies so the removed data is less sequential
shuffle_set = JackknifeIterator(spectrogram, samples=jack_samples, axis=0, target=response, inverse='both', shuffle=True)
print(shuffle_set.mask_list[0])

# Here we can plot some data from a fitted model set and predictions to see how our model is performing
prediction_set = jackknife_iterator.get_predicted_jackknifes(model_fit_list)

# The plot itself will use it's internal set of fitted models and samples amount to plot the data
jackknife_iterator.plot_estimate_error()
