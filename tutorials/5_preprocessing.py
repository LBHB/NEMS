"""Demonstrates several preprocessing utilities.
NOTE: This script may not actually work yet!

"""
import numpy as np

from nems.preprocessing import (
    indices_by_fraction, split_at_indices, get_jackknife_indices, get_jackknife
)

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
    spectrogram = np.random.random(size=(10000, 18))
    response = np.stack(spectrogram[:, :5])

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
print(f"Number of new datasets: {len(jack_dataset)}, and shape of datasets: ")
[print(f'set {x+1}: {jack_dataset[x].shape}') for x in range(len(jack_dataset))]
