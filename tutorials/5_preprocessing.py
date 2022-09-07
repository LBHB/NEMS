"""Demonstrates several preprocessing utilities.
NOTE: This script may not actually work yet!

"""

import numpy as np

from nems.preprocessing import (
    indices_by_fraction, split_at_indices, get_jackknife_indices, get_jackknife
)


# Assume data has been loaded.
def my_data_loader(file_path):
    # Dummy function to demonstrate the data format.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(10000, 18))
    response = np.random.random(size=(10000, 100))

    return spectrogram, response

spectrogram, response = my_data_loader('path/to_data.csv')


# Split the data into estimation and validation sets, using tools found in
# `nems.preprocessing.split`. In this case, we'll use 90% of the data (in time)
# for estimation and the remaining 10% for validation.
# 
# First, we get the indices that can be used to perform the split. This step is
# separate from actually generating the split data so that memory usage can be
# reduced when needed.
idx_before, idx_after = indices_by_fraction(response, fraction=0.9, axis=0)

# Next, generate the split for each data array. This will produce separate
# copies of the data.
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
