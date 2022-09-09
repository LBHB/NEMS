# This code is derived from the gammatone toolkit, licensed under the 3-clause
# BSD license: https://github.com/detly/gammatone/blob/master/COPYING

"""
This module contains functions for calculating weights to approximate a
gammatone filterbank-like "spectrogram" from a Fourier transform.
"""
from __future__ import division
import numpy as np


def specgram_window(
        nfft,
        nwin,
    ):
    """
    Window calculation used in specgram replacement function. Hann window of
    width `nwin` centred in an array of width `nfft`.
    """
    halflen = nwin // 2
    halff = nfft // 2 # midpoint of win
    acthalflen = int(np.floor(min(halff, halflen)))
    halfwin = 0.5 * ( 1 + np.cos(np.pi * np.arange(0, halflen+1)/halflen))
    win = np.zeros((nfft,))
    win[halff:halff+acthalflen] = halfwin[0:acthalflen];
    win[halff:halff-acthalflen:-1] = halfwin[0:acthalflen];
    return win


def specgram(x, n, sr, w, h):
    """ Substitute for Matlab's specgram, calculates a simple spectrogram.

    :param x: The signal to analyse
    :param n: The FFT length
    :param sr: The sampling rate
    :param w: The window length (see :func:`specgram_window`)
    :param h: The hop size (must be greater than zero)
    """
    # Based on Dan Ellis' myspecgram.m,v 1.1 2002/08/04
    assert h > 0, "Must have a hop size greater than 0"

    s = x.shape[0]
    win = specgram_window(n, w)

    c = 0

    # pre-allocate output array
    ncols = 1 + int(np.floor((s - n)/h))
    d = np.zeros(((1 + n // 2), ncols), np.dtype(complex))

    for b in range(0, s - n, h):
      u = win * x[b : b + n]
      t = np.fft.fft(u)
      d[:, c] = t[0 : (1 + n // 2)].T
      c = c + 1

    return d
