# This code is derived from the gammatone toolkit, licensed under the 3-clause
# BSD license: https://github.com/detly/gammatone/blob/master/COPYING

from __future__ import division
import numpy as np

from .spectrogram import specgram
from .filters import make_erb_filters, centre_freqs, erb_filterbank, erb_space

"""
This module contains functions for rendering "spectrograms" which use gammatone
filterbanks instead of Fourier transforms.

SVD 2020-04-09 : Modified for NEMS to allow for explicit f_max specification
"""

def round_half_away_from_zero(num):
    """ Implement the round-half-away-from-zero rule, where fractional parts of
    0.5 result in rounding up to the nearest positive integer for positive
    numbers, and down to the nearest negative number for negative integers.
    """
    return np.sign(num) * np.floor(np.abs(num) + 0.5)


def gtgram_strides(fs, window_time, hop_time, filterbank_cols):
    """
    Calculates the window size for a gammatonegram.
    
    @return a tuple of (window_size, hop_samples, output_columns)
    """
    nwin        = int(round_half_away_from_zero(window_time * fs))
    hop_samples = int(round_half_away_from_zero(hop_time * fs))
    columns     = (1
                    + int(
                        np.floor(
                            (filterbank_cols - nwin)
                            / hop_samples
                        )
                    )
                  )
        
    return (nwin, hop_samples, columns)


def gtgram_xe(wave, fs, channels, f_min, f_max=None, verbose=False):
    """ Calculate the intermediate ERB filterbank processed matrix """
    cfs = centre_freqs(fs, channels, f_min, f_max)
    if verbose:
        print('cfs: ', cfs)
    fcoefs = np.flipud(make_erb_filters(fs, cfs))
    xf = erb_filterbank(wave, fcoefs)
    xe = np.power(xf, 2)
    return xe


def gtgram(
    wave,
    fs,
    window_time, hop_time,
    channels,
    f_min, f_max=None):
    """
    Calculate a spectrogram-like time frequency magnitude array based on
    gammatone subband filters. The waveform ``wave`` (at sample rate ``fs``) is
    passed through an multi-channel gammatone auditory model filterbank, with
    lowest frequency ``f_min`` and highest frequency ``f_max``. The outputs of
    each band then have their energy integrated over windows of ``window_time``
    seconds, advancing by ``hop_time`` secs for successive columns. These
    magnitudes are returned as a nonnegative real matrix with ``channels`` rows.
    
    | 2009-02-23 Dan Ellis dpwe@ee.columbia.edu
    |
    | (c) 2013 Jason Heeris (Python implementation)
    """
    xe = gtgram_xe(wave, fs, channels, f_min, f_max)
    
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


def fft_weights(
    nfft,
    fs,
    nfilts,
    width,
    fmin,
    fmax,
    maxlen):
    """
    :param nfft: the source FFT size
    :param sr: sampling rate (Hz)
    :param nfilts: the number of output bands required (default 64)
    :param width: the constant width of each band in Bark (default 1)
    :param fmin: lower limit of frequencies (Hz)
    :param fmax: upper limit of frequencies (Hz)
    :param maxlen: number of bins to truncate the rows to
    
    :return: a tuple `weights`, `gain` with the calculated weight matrices and
             gain vectors
    
    Generate a matrix of weights to combine FFT bins into Gammatone bins.
    
    Note about `maxlen` parameter: While wts has nfft columns, the second half
    are all zero. Hence, aud spectrum is::
    
        fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft))
    
    `maxlen` truncates the rows to this many bins.
    
    | (c) 2004-2009 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    | (c) 2012 Jason Heeris (Python implementation)
    """
    ucirc = np.exp(1j * 2 * np.pi * np.arange(0, nfft / 2 + 1) / nfft)[None, ...]
    
    # Common ERB filter code factored out
    cf_array = erb_space(fmin, fmax, nfilts)[::-1]

    _, A11, A12, A13, A14, _, _, _, B2, gain = (
        make_erb_filters(fs, cf_array, width).T
    )
    
    A11, A12, A13, A14 = A11[..., None], A12[..., None], A13[..., None], A14[..., None]

    r = np.sqrt(B2)
    theta = 2 * np.pi * cf_array / fs    
    pole = (r * np.exp(1j * theta))[..., None]
    
    GTord = 4
    
    weights = np.zeros((nfilts, nfft))

    weights[:, 0:ucirc.shape[1]] = (
          np.abs(ucirc + A11 * fs) * np.abs(ucirc + A12 * fs)
        * np.abs(ucirc + A13 * fs) * np.abs(ucirc + A14 * fs)
        * np.abs(fs * (pole - ucirc) * (pole.conj() - ucirc)) ** (-GTord)
        / gain[..., None]
    )

    weights = weights[:, 0:int(maxlen)]

    return weights, gain


def fft_gtgram(
    wave,
    fs,
    window_time, hop_time,
    channels,
    f_min):
    """
    Calculate a spectrogram-like time frequency magnitude array based on
    an FFT-based approximation to gammatone subband filters.

    A matrix of weightings is calculated (using :func:`gtgram.fft_weights`), and
    applied to the FFT of the input signal (``wave``, using sample rate ``fs``).
    The result is an approximation of full filtering using an ERB gammatone
    filterbank (as per :func:`gtgram.gtgram`).

    ``f_min`` determines the frequency cutoff for the corresponding gammatone
    filterbank. ``window_time`` and ``hop_time`` (both in seconds) are the size
    and overlap of the spectrogram columns.

    | 2009-02-23 Dan Ellis dpwe@ee.columbia.edu
    |
    | (c) 2013 Jason Heeris (Python implementation)
    """
    width = 1 # Was a parameter in the MATLAB code

    nfft = int(2 ** (np.ceil(np.log2(2 * window_time * fs))))
    nwin, nhop, _ = gtgram.gtgram_strides(fs, window_time, hop_time, 0);

    gt_weights, _ = fft_weights(
            nfft,
            fs,
            channels,
            width,
            f_min,
            fs / 2,
            nfft / 2 + 1
        )

    sgram = specgram(wave, nfft, fs, nwin, nhop)

    result = gt_weights.dot(np.abs(sgram)) / nfft

    return result
