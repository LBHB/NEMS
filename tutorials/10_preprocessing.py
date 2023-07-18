"""Demonstrates several preprocessing utilities.
NOTE: This script may not actually work yet!

"""
import numpy as np
import matplotlib.pyplot as plt

from nems.preprocessing import (
    indices_by_fraction, split_at_indices, get_jackknife_indices, get_jackknife, generate_jackknife_data
)
from nems import Model, Model_List
from nems.layers import LevelShift, WeightChannels

## This indicates that our code is interactive, allowing a matplotlib
## backend to show graphs. Uncomment if you don't see any graphs
#plt.ion()

# Basic fitter options for testing
options = {'options': {'maxiter': 50, 'ftol': 1e-4}}

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

# Assume data has been loaded. Refer to tutorial 1 for more info
def my_data_loader(file_path):
    # Dummy function to demonstrate the data format.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(1000, 18))
    response = spectrogram[:,[1]] - spectrogram[:,[5]]*0.1 + np.random.randn(1000, 1)*0.1 + 0.5

    return spectrogram, response
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


# TODO: jackknifing, other stuff

###########################
# get_jackknife_indices
# Using this tool we get a list of indices for creating n datasets
#   - data: The data we want to perform operations on
#   - n: The number of samples we wish to create from our data
#   - full_shuffle: Allows us to shuffle all entries in array
#   - shuffle_jacks: Shuffles the indices created
# See more at: https://temp.website.net/split_jackknifing
###########################
jack_list = get_jackknife_indices(spectrogram, 12, axis=0,
                          shuffle_jacks=True)
# Note: Jackknifing can be used for creating N-1 datasets from a dataset of size N

###########################
# get_jackknife
# This uses given index and creates a jackknife dataset from it
#   - data: The data we will be seperating
#   - x: The index to seperate
#   - axis: The axis we wish to seperate
###########################
jack_dataset = [get_jackknife(spectrogram, x, axis=0) for x in jack_list]
# Since get_jackknife_indices returns a list of indexs, we must iterate on that list
# to get all of our data sets

# This just prints some info on the new data we made
print(f"Number of new datasets: {len(jack_dataset)}, and shape of datasets: {jack_dataset[0].shape}")
[print(f'set {x+1}: {jack_dataset[x].shape}') for x in range(len(jack_dataset))]

###########################
# generate_jackknife_data
# One way to utilize jackknifing is by taking advantage of
# our generators to create more sets of data from our inputs 
# and targets
#   - data: Our input data to be split
#   - samples: The number of sets we wish to generate
#   - axis: What axis to split the data on
#   - batch_size: The size of individual batches to take into account
# NOTE: This function actually returns a second generator that performs the above work.
#       To return relevant data, call next(next(test_gen_var))
###########################
input_gen = generate_jackknife_data(spectrogram, 5)
target_gen = generate_jackknife_data(response, 5)

###########################
# fit_from_generator
# Utilizing our new generators, we will fit our model
# through a series of datasets
#   # If you do not pass your own generated data
#   - input: The input data our default generator will use
#   - target: The target data for our default generator
#   # Else, you can specify your own generators
#   - input_gen: The generator called for inputs
#   - target_gen: The generator called for target data
#   - **fit_options: Our usual fit_options we will pass to Model().fit
###########################

# Basic model for testing
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    LevelShift(shape=(1,)) # WeightChannels will provide 1 input to shift
)


# There are 2 ways to call a fit using generators

#1. Using our prebuilt generators, we can call it directly with fit
gen_model = model.fit(input_gen, target_gen, fitter_options=options, backend='scipy')

#2. Using fit_from_generator we can provide simply an input, output and a default generator will be used
gen_model = model.fit_from_generator(spectrogram, response, 5, fitter_options=options, backend='scipy')

# Keep in mind, you cannot reuse a declared generator, so you must create a new set for each model
# Visualizing our model after 5 fits using our data generator
gen_model.plot(spectrogram, target=response)

# We can also apply all of this to our gen_model_list
# NOTE: You must provide a list of generators, or set samples 
# as (models*samples) so 5 samples in this list is 5*2 = 10 samples
input_gen = generate_jackknife_data(spectrogram, 10)
target_gen = generate_jackknife_data(response, 10)

gen_model_list = Model_List(model)
gen_model_list.fit(input_gen, target_gen, fitter_options=options, backend='scipy')

# Comparitive plot of our 5 graphs, with 5 fits each, process through generated data
gen_model_list.plot(spectrogram, response)

## Uncomment if you don't have an interactive backend installed
#plt.show()