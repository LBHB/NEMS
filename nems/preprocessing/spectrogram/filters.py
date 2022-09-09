# This code is derived from the gammatone toolkit, licensed under the 3-clause
# BSD license: https://github.com/detly/gammatone/blob/master/COPYING

"""Constructs sets of equivalent rectangular bandwidth gammatone filters."""

from __future__ import division
from collections import namedtuple

import numpy as np
import scipy as sp
from scipy import signal as sgn


def erb_space(low_freq=100.0, high_freq=44100/4, n=100):
    """Compute `n` uniformly spaced frequencies on an ERB scale.
    
    For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
    "Suggested formulae for calculating auditory-filter bandwidths and
    excitation patterns," J. Acoust. Soc. Am. 74, 750-753.

    Parameters
    ----------
    low_freq : float; default=100.0.
    high_freq : float; deafult=44100/4.
    n : int; default=100.

    Returns
    -------
    np.ndarray, dtype=float.

    """
    return erb_point(low_freq, high_freq, np.arange(1, n+1)/n)


def erb_point(low_freq, high_freq, fraction, glasberg_moore=None):
    """Calculates point(s) on an ERB scale.
    
    If `fraction` is a scalar a single point is returned, but if `fraction` is
    an array then an array of points is returned. Each point will be between
    `low_freq` and `high_freq`, with the exact position determined by `fraction`.
    If `fraction == 1`, `low_freq` is used. If `fraction == 0`, `high_freq` is
    used.
    
    `fraction` can actually be outside the range `[0, 1]`, which in general
    isn't very meaningful, but might be useful when `fraction` is rounded a
    little above or below `[0, 1]` (eg. for plot axis labels).

    Parameters
    ----------
    low_freq : float.
    high_freq : float.
    fraction : float or np.ndarray.
    glasberg_moore : dict; optional.
        A dictionary with keys 'ear_q', 'min_bw', and 'order'. Change these
        values if you wish to use a different ERB scale.

        Defaults are Glasberg and Moore parameters:
        'ear_q': 9.26449
        'min_bw': 24.7
        'order': 1

    Returns
    -------
    float or np.ndarray.

    """

    if glasberg_moore is None: glasberg_moore = {}
    # Default to Glasberg and Moore parameters.
    ear_q = glasberg_moore.get('ear_q', 9.26449)
    min_bw = glasberg_moore.get('min_bw', 24.7)
    order = glasberg_moore.get('order', 1)

    # All of the following expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear Filter
    # Bank." See pages 33-34.
    erb_point = (
        -ear_q * min_bw
        + np.exp(
            fraction * (
                -np.log(high_freq + ear_q * min_bw)
                + np.log(low_freq + ear_q * min_bw)
                )
        ) *
        (high_freq + ear_q * min_bw)
    )
    
    return erb_point


def centre_freqs(fs, num_freqs, cutoff, f_max=None):
    """Calculate centre frequencies for use by `make_erb_filters`.

    Parameters
    ----------
    fs : int.
        Sampling frequency.
    num_freqs : int.
        Number of centre frequencies to calculate.
    cutoff : float.
        Lower cutoff frequency.
    f_max : float; optional.
        Upper cutoff frequency. Added by SVD, 2020-04-09.

    Returns
    -------
    np.ndarray, dtype=float.

    """
    if f_max is None: f_max = fs/2
    return erb_space(cutoff, f_max, num_freqs)


def make_erb_filters(fs, centre_freqs, width=1.0, glasberg_moore=None):
    """Compute filter coefficients for a bank of Gammatone filters.
    
    These filters were defined by Patterson and Holdworth for simulating the
    cochlea. Each row of the filter arrays contains the coefficients for four
    second order filters. The transfer function for these four filters share the
    same denominator (poles) but have different numerators (zeros). All of these
    coefficients are assembled into one vector that `erb_filterbank` can use
    to implement the filter.

    Parameters
    ----------
    fs : int.
        Sampling frequency.
    centre_freqs : np.ndarray, dtype=float.
        Center frequencies returned by `centre_freqs(fs, ...)`.
    width : float; default=1.0.
        TODO: what does width affect?
    glasberg_moore : dict; optional.
        A dictionary with keys 'ear_q', 'min_bw', and 'order'. Change these
        values if you wish to use a different ERB scale.

        Defaults are Glasberg and Moore parameters:
        'ear_q': 9.26449
        'min_bw': 24.7
        'order': 1

    Returns
    -------
    np.ndarray, dtype=float.
    
    Notes
    -----
    This implementation fixes a problem in the original code by computing four
    separate second order filters. This avoids a big problem with round off
    errors in cases of very small cfs (100Hz) and large sample rates (44kHz).
    The problem is caused by roundoff error when a number of poles are combined,
    all very close to the unit circle. Small errors in the eigth order
    coefficient, are multiplied when the eigth root is taken to give the pole
    location. These small errors lead to poles outside the unit circle and
    instability. Thanks to Julius Smith for leading me to the proper explanation.
    
    Examples
    --------
    Evaluate the response of a 10-channel filterbank:
    >>> cf = centre_freqs(fs=16000, num_freqs=10, cutoff=100)
    >>> filter_coefficients = make_erb_filters(fs=16000, cf, width=1)
    >>> erb_filterbank(np.sin(np.arange(500)), filter_coefficients).shape
    (500, 10)
    
    Copyright
    ---------
    Rewritten by Malcolm Slaney@Interval.  June 11, 1998.
    (c) 1998 Interval Research Corporation
    (c) 2012 Jason Heeris (Python implementation)

    """
    T = 1 / fs

    if glasberg_moore is None: glasberg_moore = {}
    # Default to Glasberg and Moore parameters.
    ear_q = glasberg_moore.get('ear_q', 9.26449)
    min_bw = glasberg_moore.get('min_bw', 24.7)
    order = glasberg_moore.get('order', 1)

    erb = width*((centre_freqs/ear_q)**order + min_bw**order)**(1/order)
    B = 1.019 * 2 * np.pi * erb

    arg = 2 * centre_freqs * np.pi * T
    vec = np.exp(2j * arg)

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * np.cos(arg) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)
    
    rt_pos = np.sqrt(3 + 2 ** 1.5)
    rt_neg = np.sqrt(3 - 2 ** 1.5)
    
    common = -T * np.exp(-(B * T))
    
    # TODO: This could be simplified to a matrix calculation involving the
    # constant first term and the alternating rt_pos/rt_neg and +/-1 second
    # terms
    k11 = np.cos(arg) + rt_pos * np.sin(arg)
    k12 = np.cos(arg) - rt_pos * np.sin(arg)
    k13 = np.cos(arg) + rt_neg * np.sin(arg)
    k14 = np.cos(arg) - rt_neg * np.sin(arg)

    A11 = common * k11
    A12 = common * k12
    A13 = common * k13
    A14 = common * k14

    gain_arg = np.exp(1j * arg - B * T)

    gain = np.abs(
            (vec - gain_arg * k11)
          * (vec - gain_arg * k12)
          * (vec - gain_arg * k13)
          * (vec - gain_arg * k14)
          * (  T * np.exp(B * T)
             / (-1 / np.exp(B * T) + 1 + vec * (1 - np.exp(B * T)))
            )**4
        )

    allfilts = np.ones_like(centre_freqs)
    
    fcoefs = np.column_stack([
        A0 * allfilts, A11, A12, A13, A14, A2*allfilts,
        B0 * allfilts, B1, B2,
        gain
    ])
    
    return fcoefs


def erb_filterbank(wave, coefs):
    """Process an input waveform with a gammatone filter bank.

    Parameters
    ----------
    wave : np.ndarray.
        Sound waveform with shape (T,) or (T,1).  TODO: confirm
    coefs : np.ndarray.
        Gammatone filter coefficients returned by `make_erb_filters`,
        with shape (`num_freqs`, 10)
    
    Returns
    -------
    np.ndarray
        Filter outputs, with shape (T, `num_freqs`).

    Copyright
    ---------
    Malcolm Slaney @ Interval, June 11, 1998.
    (c) 1998 Interval Research Corporation
    Thanks to Alain de Cheveigne' for his suggestions and improvements.
    (c) 2013 Jason Heeris (Python implementation)

    """

    # TODO: why the hard-coded 9?
    output = np.zeros((coefs[:,9].shape[0], wave.shape[0]))
    
    gain = coefs[:, 9]
    # A0, A11, A2
    As1 = coefs[:, (0, 1, 5)]
    # A0, A12, A2
    As2 = coefs[:, (0, 2, 5)]
    # A0, A13, A2
    As3 = coefs[:, (0, 3, 5)]
    # A0, A14, A2
    As4 = coefs[:, (0, 4, 5)]
    # B0, B1, B2
    Bs = coefs[:, 6:9]
    
    # Loop over channels
    for idx in range(coefs.shape[0]):
        # These seem to be reversed (in the sense of A/B order), but that's what
        # the original code did...
        # Replacing these with polynomial multiplications reduces both accuracy
        # and speed.
        y1 = sgn.lfilter(As1[idx], Bs[idx], wave)
        y2 = sgn.lfilter(As2[idx], Bs[idx], y1)
        y3 = sgn.lfilter(As3[idx], Bs[idx], y2)
        y4 = sgn.lfilter(As4[idx], Bs[idx], y3)
        output[idx, :] = y4 / gain[idx]
        
    return output.T  # Transpose to conform w/ NEMS data format
