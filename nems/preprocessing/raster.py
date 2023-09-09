import math

import numpy as np


def raster_to_spike_times(raster, fs):
    """Convert spike raster to a list of spike time arrays.
    
    Parameters
    ----------
    raster : np.ndarray.
        Binary spike raster of shape (T, N), where T is the number of time bins
        and N is the number of neurons.
    fs : int.
        Sampling rate of `raster`. Determines precision of spike times.

    Returns
    -------
    spike_times_in_seconds : list of ndarray.
    
    """
    spike_times_in_seconds = [
        raster[..., i].nonzero()[0] / fs
        for i in range(raster.shape[1])
        ]

    return spike_times_in_seconds


def spike_times_to_raster(spike_times, fs, duration=None):
    """Convert nested list of spike times to a spike raster.

    TODO: Add option to convert times from other units (i.e conversion factor
          similar to plotting utility).

    TODO: This is my first-pass naive implementation. There are likely some
          details I'm not considering, ask Stephen to revise.

    Parameters
    ----------
    spike_times : list of list, or list of np.ndarray.
        Each list element should represent all spike times in seconds for one
        neuron, in order of increasing time.
    fs : int.
        Sampling rate for spike raster.
    duration : float.
        Amount of time in seconds that the raster should span. If not specified,
        duration will be determined by the largest spike time.

    Returns
    -------
    raster : np.ndarray, dtype=bool.
    
    """
    if duration is not None:
        n_time_bins = math.ceil(duration*fs)
    else:
        last_spike_time = max([times[-1] for times in spike_times])
        n_time_bins = math.ceil(last_spike_time*fs)

    raster_columns = []
    duplicate_spikes = 0
    truncated_spikes = 0
    for i, times in enumerate(spike_times):
        column = np.zeros(shape=(n_time_bins, 1), dtype=bool)

        for j, t in enumerate(times):
            spike_bin = round(t*fs)
            if spike_bin == n_time_bins:
                # Exactly at end of duration, round down instead of up.
                spike_bin -= 1
            elif spike_bin > n_time_bins:
                # Spike times past duration, truncate.
                truncated_spikes += len(times[j:])
                break

            # NOTE: Multiple spikes in the same bin will be overwritten.
            if column[spike_bin, 0] == 1:
                # Already a spike here, log the count for debugging.
                duplicate_spikes += 1
            else:
                column[spike_bin, 0] = 1

        raster_columns.append(column)

    print(f'{duplicate_spikes} spikes dropped due to bin number rounding.')
    print(f'{truncated_spikes} spikes dropped due to truncated duration.')
    raster = np.concatenate(raster_columns, axis=1)

    return raster
