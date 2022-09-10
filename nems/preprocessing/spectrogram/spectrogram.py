# This code is derived from the gammatone toolkit, licensed under the 3-clause
# BSD license: https://github.com/detly/gammatone/blob/master/COPYING

"""Represent waveforms with spectrograms using Fast Fourier Transform."""

from __future__ import division
import numpy as np


def spectrogram(wave, fft_length, fs, window_time, hop_time):
    """Compute a simple spectrogram for sound `wave`.

    Parameters
    ----------
    wave : np.ndarray.
        Sound waveform with shape (T,) or (T,1).
    fft_length : int.
        Size of the source FFT.
    fs : int.
        Sampling frequency.
    window_time : float.
    hop_time : float > 0.

    Returns
    -------
    np.ndarray
        Shape (T, )

    Copyright
    ---------
    Based on Dan Ellis' myspecgram.m,v 1.1 2002/08/04.

    """

    if hop_time <= 0: raise ValueError("Must have a hop size greater than 0")

    sound_length = wave.shape[0]
    win = _spectrogram_window(fft_length, window_time)

    c = 0

    # pre-allocate output array
    ncols = 1 + int(np.floor((sound_length - fft_length)/hop_time))
    d = np.zeros(((1 + fft_length // 2), ncols), np.dtype(complex))

    for b in range(0, sound_length - fft_length, hop_time):
      u = win * wave[b : b + fft_length]
      t = np.fft.fft(u)
      d[:, c] = t[0 : (1 + fft_length // 2)].T
      c = c + 1

    return d


def _spectrogram_window(nfft, nwin):
    """Window calculation used in specgram replacement function.
    
    Hann window of width `nwin` centred in an array of width `nfft`.
    Internal for `spectrogram`.

    """
    halflen = nwin // 2
    halff = nfft // 2 # midpoint of win
    acthalflen = int(np.floor(min(halff, halflen)))
    halfwin = 0.5 * ( 1 + np.cos(np.pi * np.arange(0, halflen+1)/halflen))
    win = np.zeros((nfft,))
    win[halff : halff + acthalflen] = halfwin[0 : acthalflen]
    win[halff : halff - acthalflen : -1] = halfwin[0 : acthalflen]

    return win
