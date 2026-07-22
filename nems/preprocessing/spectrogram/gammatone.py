# This code is derived from the gammatone toolkit, licensed under the 3-clause
# BSD license: https://github.com/detly/gammatone/blob/master/COPYING

"""Renders spectrograms which use gammatone filterbanks."""

from __future__ import division
import numpy as np

from .filters import (
    make_erb_filters, centre_freqs, erb_filterbank, stateful_erb_filterbank,
)
from nems.tools.utils import joblib_memory


# [AGENT EDIT START | agent: claude | user: svd | reason: add time_axis param to gammagram and gtgram alias, to unify with nems0's (channels, T) convention on request | date: 2026-07-16]
def gammagram(wave, fs=44000, window_time=0.01, hop_time=0.01, channels=18,
              f_min=200.0, f_max=None, time_axis=0):
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
    time_axis : int; default=0.
        Axis of the time dimension in the returned array. `0` returns shape
        (T, `channels`) (NEMS default convention). `1` returns shape
        (`channels`, T) (legacy gammatone-toolkit convention).

    Returns
    -------
    np.ndarray
        With shape (T, `channels`) if `time_axis=0`, or (`channels`, T) if
        `time_axis=1`.

    Copyright
    ---------
    2009-02-23 Dan Ellis dpwe@ee.columbia.edu
    (c) 2013 Jason Heeris (Python implementation)

    """
    if time_axis not in (0, 1):
        raise ValueError("time_axis must be 0 or 1")

    xe = gtgram_xe(wave, fs, channels, f_min, f_max)

    nwin, hop_samples, ncols = gtgram_strides(
        fs,
        window_time,
        hop_time,
        xe.shape[0]
    )

    y = np.zeros((ncols, channels))

    for cnum in range(ncols):
        segment = xe[cnum * hop_samples + np.arange(nwin), :]
        y[cnum, :] = np.sqrt(segment.mean(axis=0))

    if time_axis == 1:
        y = y.T

    return y


# Alias matching the legacy nems0/gammatone-toolkit name.
gtgram = gammagram
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: svd | reason: add joblib-cached gtgram/chunked_gtgram wrappers, migrated from nems_lbhb.runclass so caching lives in nems rather than nems_db; use the @joblib_memory() decorator so the cache location resolves lazily on each call (rather than once at import time), letting a cache_path set at runtime (e.g. by nems_db) take effect immediately, regardless of import order | date: 2026-07-16]
@joblib_memory()
def c_gtgram(*args, **kwargs):
    """Joblib-cached `gtgram`/`gammagram`. See `gammagram` for parameters.

    Caches to disk (see `nems.tools.utils.joblib_memory`) so repeated calls
    with identical arguments skip re-computing the gammatone filterbank.

    Defaults `time_axis` to 1 (legacy `(channels, T)` convention) since that
    is what existing callers (mostly in nems_lbhb) expect; pass `time_axis=0`
    explicitly for the NEMS default `(T, channels)` convention.

    """
    # joblib's argument hashing can't handle a keyword-only param declared
    # after *args, so default time_axis via kwargs instead of the signature.
    kwargs.setdefault('time_axis', 1)
    return gammagram(*args, **kwargs)


@joblib_memory()
def c_chunked_gtgram(*args, **kwargs):
    """Joblib-cached `chunked_gtgram`. See `chunked_gtgram` for parameters.

    Defaults `time_axis` to 1, as in `c_gtgram`.

    """
    kwargs.setdefault('time_axis', 1)
    return chunked_gtgram(*args, **kwargs)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: svd | reason: make gtgram_xe public (was _gtgram_xe, internal-only) to match nems0.analysis.gammatone.gtgram.gtgram_xe's public API, needed by nems_lbhb/projects/spatial/elizabeth/hrtf_test.py | date: 2026-07-16]
def gtgram_xe(wave, fs, channels, f_min, f_max=None, verbose=False, time_axis=0):
    """Calculate the intermediate ERB filterbank processed matrix (energy per
    channel, before windowing/downsampling to a gammagram). Used internally by
    `gammagram`, but also useful standalone (e.g. for per-sample analysis).

    Parameters
    ----------
    wave, fs, channels, f_min, f_max, verbose : see `gammagram`.
    time_axis : int; default=0.
        Axis of the time dimension in the returned array. `0` returns shape
        (T, `channels`) (NEMS default convention). `1` returns shape
        (`channels`, T) (legacy gammatone-toolkit convention).

    Returns
    -------
    np.ndarray

    """
    if time_axis not in (0, 1):
        raise ValueError("time_axis must be 0 or 1")

    cfs = centre_freqs(fs, channels, f_min, f_max)
    if verbose:
        print('cfs: ', cfs)
    fcoefs = np.flipud(make_erb_filters(fs, cfs))
    xf = erb_filterbank(wave, fcoefs)
    xe = np.power(xf, 2)

    if time_axis == 1:
        xe = xe.T

    return xe
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: svd | reason: port chunked_gtgram from nems0.analysis.gammatone.gtgram, adapted to nems's (T, N) internal convention with time_axis support | date: 2026-07-16]
def chunked_gtgram(wave, fs, window_time, hop_time, channels, f_min, f_max=None,
                    chunk_size=44100, force_overlap=None, time_axis=0):
    """Calculate a `gammagram`/`gtgram` in chunks, carrying gammatone filter
    state across chunk boundaries.

    Equivalent to `gammagram`, but processes `wave` in overlapping chunks of
    `chunk_size` samples instead of filtering the whole waveform at once. This
    is useful for long waveforms where running the full ERB filterbank in one
    pass would use too much memory. Each chunk overlaps the previous one by
    enough samples for the lowest-frequency filter to settle (based on its
    ERB bandwidth), and the overlap is discarded after filtering so chunk
    boundaries don't introduce artifacts.

    Parameters
    ----------
    wave : np.ndarray.
        Sound waveform with shape (T,) or (T,1).
    fs : int.
        Sampling frequency.
    window_time : float.
        Length of integration window, in seconds.
    hop_time : float.
        Stepsize of window advancement for successive columns, in seconds.
    channels : int.
        Number of frequency channels in the spectrogram-like output.
    f_min : float.
        Lower frequency cutoff.
    f_max : float; optional.
        Upper frequency cutoff.
    chunk_size : int; default=44100.
        Number of waveform samples to process per chunk.
    force_overlap : float; optional.
        Minimum chunk overlap, in seconds, in case the automatically
        estimated filter settling time is not long enough.
    time_axis : int; default=0.
        Axis of the time dimension in the returned array. `0` returns shape
        (T, `channels`) (NEMS default convention). `1` returns shape
        (`channels`, T) (legacy gammatone-toolkit convention).

    Returns
    -------
    np.ndarray
        With shape (T, `channels`) if `time_axis=0`, or (`channels`, T) if
        `time_axis=1`.

    """
    if time_axis not in (0, 1):
        raise ValueError("time_axis must be 0 or 1")

    # Calculate base parameters
    nwin, hop_samples, _ = gtgram_strides(fs, window_time, hop_time, chunk_size)

    # Estimate safe settling time for the lowest frequency filter
    cfs = centre_freqs(fs, channels, f_min, f_max)
    fcoefs = np.flipud(make_erb_filters(fs, cfs))
    # Bandwidth estimation based on the Glasberg & Moore (1990) formula for
    # the Equivalent Rectangular Bandwidth (ERB) of auditory filters.
    erb_bandwidths = [1.019 * 2 * np.pi * ((cfs[i] / 9.26449) + 24.7) for i in range(channels)]
    min_bandwidth = erb_bandwidths[-1]
    settling_time_s = 5 / (2 * np.pi * min_bandwidth)
    settling_samples = int(np.ceil(settling_time_s * fs))

    if force_overlap is not None:
        # Use the larger of the estimated settling time and the forced
        # minimum overlap (nems0's version discarded force_overlap here).
        forced_samples = int(np.ceil((force_overlap * fs) / hop_samples) * hop_samples)
        settling_samples = max(settling_samples, forced_samples)
    # Ensure overlap is at least settling time and a multiple of hop_samples
    overlap_samples = int(np.ceil(settling_samples / hop_samples) * hop_samples)

    # Initialize filter states
    zi = None

    y_list = []
    first_chunk = True

    for start in range(0, len(wave), chunk_size - overlap_samples):
        end = min(start + chunk_size, len(wave))
        chunk = wave[start:end]

        # Stateful processing; xf_chunk has shape (T_chunk, channels)
        xf_chunk, zi = stateful_erb_filterbank(chunk, fcoefs, zi)
        xe_chunk = np.power(xf_chunk, 2)

        # Calculate columns
        chunk_cols = (xe_chunk.shape[0] - nwin) // hop_samples + 1
        y_part = np.zeros((chunk_cols, channels))

        for c in range(chunk_cols):
            segment = xe_chunk[c * hop_samples : c * hop_samples + nwin, :]
            y_part[c, :] = np.sqrt(segment.mean(axis=0))

        # Discard overlap columns (except for the first chunk)
        overlap_cols = overlap_samples // hop_samples
        if not first_chunk:
            y_part = y_part[overlap_cols:, :]
        else:
            first_chunk = False

        y_list.append(y_part)

    # Concatenate and trim to match full-length gammagram
    expected_cols = gtgram_strides(fs, window_time, hop_time, len(wave))[2]
    y_concat = np.vstack(y_list)[:expected_cols, :]

    if time_axis == 1:
        y_concat = y_concat.T

    return y_concat
# [AGENT EDIT END]


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
