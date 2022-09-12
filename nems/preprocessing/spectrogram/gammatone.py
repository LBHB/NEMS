# This code is derived from the gammatone toolkit, licensed under the 3-clause
# BSD license: https://github.com/detly/gammatone/blob/master/COPYING

"""Renders spectrograms which use gammatone filterbanks."""

from __future__ import division
import numpy as np

from .filters import make_erb_filters, centre_freqs, erb_filterbank


def gammagram(wave, fs=44000, window_time=0.01, hop_time=0.01, channels=18,
              f_min=200.0, f_max=None):
    """Calculate a spectrogram-like array based on gammatone subband filters.
    
    The waveform `wave` (at sample rate `fs`) is passed through a multi-channel
    gammatone auditory model filterbank, with lowest frequency `f_min` and
    highest frequency `f_max`. The outputs of each band then have their energy
    integrated over windows of `window_time` seconds, advancing by `hop_time`
    seconds for successive columns. These magnitudes are returned as a
    nonnegative real matrix with `channels` rows.

    Parameters
    ----------
    wave : np.ndarray.
        Sound waveform with shape (T,) or (T,1).
    fs : int; default=44000.
        Sampling frequency. `gtgram_strides` uses this to scale `window_time`
        and `hop_time` to convert them to units of bins.
    window_time : float; default=0.01.
        Length of integration window. Default of 0.01 corresponods to 1/100hz,
        where 100hz is the most common spike raster sampling rate used by LBHB.
    hop_time : float; default=0.01.
        Stepsize of window advancement for successive columns. Default of 0.01
        corresponods to 1/100hz, where 100hz is the most common spike raster
        sampling rate used by LBHB.
    channels : int.
        Number of frequency channels in the spectrogram-like output.
    f_min : float; default=200.0.
        Lower frequency cutoff.
    f_max : float; optional.
        Upper frequency cutoff.

    Returns
    -------
    np.ndarray
        With shape (T, `channels`)
    
    Copyright
    ---------
    2009-02-23 Dan Ellis dpwe@ee.columbia.edu
    (c) 2013 Jason Heeris (Python implementation)

    """

    xe = _gtgram_xe(wave, fs, channels, f_min, f_max)
    
    nwin, hop_samples, ncols = gtgram_strides(
        fs,
        window_time,
        hop_time,
        xe.shape[1]
    )
    
    y = np.zeros((channels, ncols))
    
    for cnum in range(ncols):
        segment = xe[:, cnum * hop_samples + np.arange(nwin)]
        y[:, cnum] = np.sqrt(segment.mean(1))
    
    return y


def _gtgram_xe(wave, fs, channels, f_min, f_max=None, verbose=False):
    """Calculate the intermediate ERB filterbank processed matrix.
    
    Internal for `gammagram`.
    
    """

    cfs = centre_freqs(fs, channels, f_min, f_max)
    if verbose:
        print('cfs: ', cfs)
    fcoefs = np.flipud(make_erb_filters(fs, cfs))
    xf = erb_filterbank(wave, fcoefs)
    xe = np.power(xf, 2)

    return xe


def gtgram_strides(fs, window_time, hop_time, filterbank_cols):
    """Calculate the window size for a gammatone filter spectrogram.
    
    Parameters
    ----------
    fs : int.
        Sampling frequency.
    window_time : float; default=0.01.
        Length of integration window. Default of 0.01 corresponods to 1/100hz,
        where 100hz is the most common spike raster sampling rate used by LBHB.
    hop_time : float; default=0.01.
        Stepsize of window advancement for successive columns. Default of 0.01
        corresponods to 1/100hz, where 100hz is the most common spike raster
        sampling rate used by LBHB.
    filterbank_cols : int.

    Returns
    -------
    (window_size, hop_samples, output_columns)

    """

    nwin        = int(round_half_away_from_zero(window_time * fs))
    hop_samples = int(round_half_away_from_zero(hop_time * fs))
    columns     = int(np.floor((filterbank_cols - nwin)/hop_samples)) + 1

    return (nwin, hop_samples, columns)


def round_half_away_from_zero(num):
    """Implements the "round-half-away-from-zero" rule.
    
    Fractional parts of 0.5 result in rounding up to the nearest positive
    integer for positive numbers, and down to the nearest negative number for
    negative integers.

    Parameters
    ----------
    num : float

    Returns
    -------
    int

    """
    return np.sign(num) * np.floor(np.abs(num) + 0.5)
