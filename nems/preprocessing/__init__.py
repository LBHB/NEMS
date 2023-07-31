"""A collection of preprocessing utilities.

# TODO: These contents don't all exist yet, sketching this out based on
#       the strfy proposal document.
Contents:
    `spectrogram`: Convert sound waveforms to spectrograms.
    `filters.py`: Transform data using smoothing, high-pass filter, etc.
    `normalization.py`: Scale data to a standard dynamic range.
    `raster.py`: Convert data to continuous time series (ex: spikes -> PSTH).
    `split.py`: Separate data into estimation & validation sets.
    `mask.py`: Exclude subsets of data based on various criteria.
    `merge.py`: Align and combine datasets for fitting a single large model.

"""

from .spectrogram import spectrogram, fft_gammagram, gammagram
from .normalization import minmax, undo_minmax, joint_minmax
from .raster import raster_to_spike_times, spike_times_to_raster
from .split import (indices_by_fraction, split_at_indices,
                    get_jackknife_indices, get_jackknife, 
                    generate_jackknife_data, get_inverse_jackknife, pad_array)
