# This code is derived from the gammatone toolkit, licensed under the 3-clause
# BSD license: https://github.com/detly/gammatone/blob/master/COPYING

"""Represent waveforms with spectrograms using the Fast Fourier Transform."""

from __future__ import division
import numpy as np

from .filters import erb_space, make_erb_filters
from .gammatone import gtgram_strides


def spectrogram(wave, fft_size=None, fs=44000, window_time=0.01,
                hop_time=0.01):
    """Compute a simple spectrogram for sound `wave`.

    Parameters
    ----------
    wave : np.ndarray.
        Sound waveform with shape (T,) or (T,1).
    fft_size : int, divisible by 2; optional.
        Number of points for the Discrete Fourier Transform.
        Default is `2^ceil(log_2(2*window_time*fs))`.
        (`n` in Dan Ellis `specgram`).
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

    Returns
    -------
    np.ndarray, dtype=complex.
        Shape (Tau, fft_size/2 + 1)
        Where `Tau = 1 + floor((sound_length - fft_size)/hop_bins))`.
        If the default value is used for `fft_size`, this is equivalent to
        downsampling the spectrogram to a sampling rate of `1/hop_time`, with
        some rounding error to get integer bins.

    Copyright
    ---------
    Based on Dan Ellis' myspecgram.m,v 1.1 2002/08/04.

    """

    if hop_time <= 0: raise ValueError("Must have a hop size greater than 0")
    if fft_size is None:
        fft_size = _default_fft_size(window_time, fs)

    sound_length = wave.shape[0]
    window_bins, hop_bins, _ = gtgram_strides(fs, window_time, hop_time, 0)
    win = _spectrogram_window(fft_size, window_bins)

    t = 0

    # pre-allocate output array
    n_time_bins = 1 + int(np.floor((sound_length - fft_size)/hop_bins))
    spectrogram = np.zeros((n_time_bins, (1 + fft_size//2)), np.dtype(complex))

    for b in range(0, sound_length - fft_size, hop_bins):
      u = win * wave[b : b + fft_size]
      fft = np.fft.fft(u)
      spectrogram[t, :] = fft[0 : (1 + fft_size//2)]
      t = t + 1

    return spectrogram


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

def _default_fft_size(window_time, fs):
    """Get default number of points for DFT, internal for `spectrogram`."""
    return int(2 ** (np.ceil(np.log2(2 * window_time * fs))))


def fft_gammagram(wave, fs=44000, window_time=0.01, hop_time=0.01,
                  channels=18, f_min=200.0, f_max=None):
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
        Upper frequency cutoff. If not specified, will be set to `fs/2`.

    Returns
    -------
    np.ndarray
        With shape (Tau, `channels`)
        Where `Tau = 1 + floor((sound_length - fft_size)/hop_bins))`.
        This is equivalent to downsampling the spectrogram to a sampling rate
        of `1/hop_time`, with some rounding error to get integer bins.

    Copyright
    ---------
    2009-02-23 Dan Ellis dpwe@ee.columbia.edu
    (c) 2013 Jason Heeris (Python implementation)

    """

    fft_size = _default_fft_size(window_time, fs)
    if f_max is None: f_max = fs/2
    gt_weights, _ = fft_weights(
        fft_size, fs, channels, f_min, f_max=fs/2, maxlen=(fft_size/2 + 1)
        )

    sgram = spectrogram(wave, fft_size, fs, window_time, hop_time)
    result = np.abs(sgram).dot(gt_weights) / fft_size

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

    return weights.T, gain.T  # Transposed to conform with NEMS shapes.
