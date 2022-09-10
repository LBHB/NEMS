# This code is derived from the gammatone toolkit, licensed under the 3-clause
# BSD license: https://github.com/detly/gammatone/blob/master/COPYING

"""Renders spectrograms which use gammatone filterbanks or an FFT approximation.

SVD 2020-04-09 : Modified for NEMS to allow for explicit `f_max` specification.

"""

from __future__ import division
import numpy as np

from .spectrogram import spectrogram
from .filters import make_erb_filters, centre_freqs, erb_filterbank, erb_space


def gammagram(wave, fs, window_time, hop_time, channels, f_min, f_max=None):
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
    fs : int.
        Sampling frequency.
    window_time : float.
    hop_time : float.
    channels : int.
        Number of frequency channels in the spectrogram-like output.
    f_min : float.
    f_max : float; optional.

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
    TODO

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


def fft_gammagram(wave, fs, window_time, hop_time, channels, f_min):
    """Approximate a gammatone filter spectrogram using FFT.

    A matrix of weightings is calculated using `fft_weights`, and applied to
    the FFT of the input signal, `wave`, using sample rate `fs`. The result is
    an approximation of full filtering using an ERB gammatone filterbank
    (as per `gammagram`).

    `f_min` determines the frequency cutoff for the corresponding gammatone
    filterbank. `window_time` and `hop_time` (both in seconds) are the size
    and overlap of the spectrogram columns.

    Parameters
    ----------
    wave : np.ndarray.
        Sound waveform with shape (T,) or (T,1).
    fs : int.
        Sampling frequency.
    window_time : float.
    hop_time : float.
    channels : int.
        Number of frequency channels in the spectrogram-like output.
    f_min : float.

    Returns
    -------
    np.ndarray
        With shape (T, `channels`)

    Copyright
    ---------
    2009-02-23 Dan Ellis dpwe@ee.columbia.edu
    (c) 2013 Jason Heeris (Python implementation)

    """

    nfft = int(2 ** (np.ceil(np.log2(2 * window_time * fs))))
    nwin, nhop, _ = gtgram_strides(fs, window_time, hop_time, 0)

    gt_weights, _ = fft_weights(
        nfft, fs, channels, f_min, f_max=fs/2, maxlen=(nfft/2 + 1)
        )

    sgram = spectrogram(wave, nfft, fs, nwin, nhop)
    result = gt_weights.dot(np.abs(sgram)) / nfft

    return result


def fft_weights(nfft, fs, channels, f_min, f_max, maxlen, width=1):
    """Generate a matrix of weights to combine FFT bins into Gammatone bins.

    Parameters
    ----------
    nfft : int.
        Size of the source FFT.
    fs : int.
        Sampling frequency.
    channels : int.
        Number of output bands required.
    f_min : lower limit of frequencies (Hz)
    f_max : upper limit of frequencies (Hz)
    maxlen : int.
        Number of bins to truncate the columns to.
    width : int; default=1.
        Constant width of each band in Bark.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Weight matrices and gain vectors.
    
    Copyright
    ---------
    (c) 2004-2009 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    (c) 2012 Jason Heeris (Python implementation)

    """
    ucirc = np.exp(1j*2*np.pi*np.arange(0, nfft/2 + 1)/nfft)[np.newaxis, ...]
    
    # Common ERB filter code factored out
    cf_array = erb_space(f_min, f_max, channels)[::-1]

    _, A11, A12, A13, A14, _, _, _, B2, gain = (
        make_erb_filters(fs, cf_array, width).T
    )
    
    A11, A12, A13, A14 = (A11[..., np.newaxis], A12[..., np.newaxis],
                          A13[..., np.newaxis], A14[..., np.newaxis])

    r = np.sqrt(B2)
    theta = 2 * np.pi * cf_array / fs    
    pole = (r * np.exp(1j * theta))[..., None]
    
    GTord = 4
    
    weights = np.zeros((channels, nfft))

    weights[:, 0:ucirc.shape[1]] = (
          np.abs(ucirc + A11 * fs) * np.abs(ucirc + A12 * fs)
        * np.abs(ucirc + A13 * fs) * np.abs(ucirc + A14 * fs)
        * np.abs(fs * (pole - ucirc) * (pole.conj() - ucirc)) ** (-GTord)
        / gain[..., None]
    )

    weights = weights[:, 0:int(maxlen)]

    return weights, gain
