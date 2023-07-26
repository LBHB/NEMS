"""Demonstrates several preprocessing utilities.
NOTE: This script may not actually work yet!

"""
import numpy as np
import matplotlib.pyplot as plt

from nems.preprocessing import (
    indices_by_fraction, split_at_indices, get_jackknife_indices, get_jackknife, generate_jackknife_data, get_inverse_jackknife,
    pad_array
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

########################################################
# Jackknifing
# Using jackknifes we can create subsets of our data to
# more accurately fit models and provide predication data
#
# The core of our jackknife functions are based around generators. It is
# important to keep in mind a few things about generators:
#   1. Using next(generator_function) will provide you with the next iteration of a generator
#   2. Generators may have limited iterations, which require you to create new ones
#       See, nems.preprocessing.split.internal_jackknife_generator(data, ...)
#   3. Generators only provide a single iteration of data, this saves on memory but also
#      means you need call the next iteration for more data.
########################################################

###########################
# get_jackknife_indices
# Using this tool we can create a generator that provides sets of jackknifed
# indicies. These indices can then be used to create datasets using get_jackknife
#   - data: The data we want to perform operations on
#   - n: The number of samples we wish to create from our data
#   - batch_size: The size of batches to organize indicies by
#   - full_shuffle: Allows us to shuffle all entries in array
#   - shuffle_jacks: Shuffles the indices created
#   - full_list: Provides entire lists of jackknife sets, instead of generator
# See more at: https://temp.website.net/split_jackknifing
###########################
jack_generator = get_jackknife_indices(spectrogram, 12, axis=0, shuffle_jacks=True)

# Here we're printing a single set of jackknife indices
print(next(jack_generator).shape)

# Creating a list of all our generated lists
jack_list = []
for indices_set in jack_generator:
    jack_list.append(indices_set)

# Or pull a whole list at once
jack_list = get_jackknife_indices(spectrogram, 12, axis=0, shuffle_jacks=True, full_list=True)
jack_list = next(jack_list)

###########################
# get_jackknife
# This uses given index or generator and creates a jackknife dataset from it
#   - data: The data we will be seperating
#   - x: The index to seperate
#   - axis: The axis we wish to seperate
###########################
jack_single = get_jackknife(spectrogram, jack_list[0], axis=0)
jack_dataset = [get_jackknife(spectrogram, x, axis=0) for x in jack_list]
# Since get_jackknife_indices returns a list of indexs, we must iterate on that list
# to get all of our data sets

# This just prints some info on the new data we made
print(f"Number of new datasets: {len(jack_dataset)}, and shape of datasets: {jack_dataset[0].shape}")
[print(f'set {x+1}: {jack_dataset[x].shape}') for x in range(len(jack_dataset))]

###########################
# get_inverse_jackknife
# Uses the given jackknife indicies or generator and provides the inverse dataset
#   - data: The data we will be seperating
#   - x: The original index to inverse
#   - axis: The axis we wish to seperate
###########################
inverse_single = get_inverse_jackknife(spectrogram, jack_list[0], axis=0)
inverse_jack_dataset = [get_inverse_jackknife(spectrogram, index_set, axis=0) for index_set in jack_list]

###########################
# generate_jackknife_data
# One way to utilize jackknifing is by taking advantage of
# our generators to create available sets of inputs and targets
#   - input: Data we wish to fit in a model
#   - target: Our target data to fit our input onto
#   - samples: The number of sets we wish to generate
#   - axis: What axis to split the data on
#   - batch_size: The size of individual batches to take into account
#   - inverse: Allows you to access a tuple that provides the inverse data 
# NOTE: This function actually returns a second generator that performs the above work.
#       To return relevant data, call next(next(test_gen_var))
###########################
jackknife_dataset_generator = generate_jackknife_data(spectrogram, response, 5)

# Genereate_jackknife_data creates a generator of generators so we can make new ones with the same parameters.
# Calling next again will provide us with an actual generator that returns values
jackknife_dataset_generator = next(jackknife_dataset_generator)

# An example of how this generator will provide data,
# typically this is for internal use by our fits:
input_gen, target_gen = next(jackknife_dataset_generator)

# We can also specify and have our jackknifes return inverse masks
jackknife_full_generator = generate_jackknife_data(spectrogram, response, 5, inverse=True)

# Our inverse sets will be used for validation
est, val = next(jackknife_full_generator)

# A generator we can pass to our model fitters
est_gen = next(est)
est_input, est_target = next(est_gen)

# Our validation set
val_input, val_target = next(val)

###########################
# fit_from_generator
# Utilizing our new generators, we will fit our model
# through a series of datasets
#   # If you do not pass your own generated data
#   - input: The input data our default generator will use
#   - target: The target data for our default generatorjackknife_dataset_generator = jackknife_dataset_generator

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

#1. We can fit our individual data sets as normal
gen_model = model.fit(est_input, est_target, fitter_options=options, backend='scipy')

#2. We can also fit using their generators directly, looping through all the given samples at once
gen_model = gen_model.fit(est, fitter_options=options, backend='scipy')

# Visualizing our model after 5 fits using our data generator
gen_model.plot(spectrogram, target=response)

# We can also apply all of this to our gen_model_list

gen_model_list = Model_List(model)
gen_model_list.fit(est, fitter_options=options, backend='scipy')

# Comparitive plot of our 5 graphs, with 5 fits each, process through generated data
gen_model_list.plot(spectrogram, response)





# TODO: Rewrite this

###########################
# pad_array
# Often we may need to pad both our original jackknifes, and our inverse
# sets, so we can properly fit models together. Using pad_array you can
# fill our new data with empty data so you can start fitting models.
#   - array: The original array
#   - size: Lets you specify how much bigger you want the end array to be
#   - axis: Lets you choose which axis to use for certain pad_types ie. Mean
#   - pad_type: Pick which way you wish to pad your data
#   - pad_path: Choose to pad only the end or start of the array if you wish
#   - indicies: Provide the mask used to replace indicies of the array with fake data
###########################
test_array = spectrogram
test_array = pad_array(test_array, size=25, pad_type='zero')
print(f'Our original array size: {spectrogram.shape}\n Our padded array size: {test_array.shape}')

# This is all important because if you want to split your data into groups and also fit your model to 
# the inverse of that data, you need to make sure there is an equal sized axis to measure for each

# For example:
print(f'Base data: {jack_single.shape}\n Inverse data: {inverse_single.shape}')

# Now if we try to fit these it won't work, but we can pad them both.

# Here we are replacing our data with padding, instead of deleting it
jack_single = get_jackknife(spectrogram, jack_list[0], axis=0, pad=True)

# This will create our inverse as normal, but instead of deleting the other data it will replace
# it with our padding
inverse_single = get_inverse_jackknife(spectrogram, jack_list[0], axis=0, pad=True)

# Now they should both have the same size
print(f'Base data: {jack_single.shape}\n Inverse data: {inverse_single.shape}')


# We have a few other ways of padding our data too:
# Mean pads with the mean value along a given axis
test_array = pad_array(test_array, size=25, pad_type='mean')

# Edge will use the the immediate values from the left or right of each added index
test_array = pad_array(test_array, size=25, pad_type='edge')

# Random will select some random value between the min and max of the array at a given axis
test_array = pad_array(test_array, size=25, pad_type='random')










## Uncomment if you don't have an interactive backend installed
#plt.show()