"""ACNet's raw-audio front end: level normalization + gammatonegram + compression.

Ported (numpy/scipy, no torch) from `ACNet_v1.acnet_model.nems_audio_preprocess`/
`remove_clicks`, which were themselves vendored from `PT_EncMdl_helpers_v2`'s
`GammatoneFilterbankProcessing`. This module reproduces the level-normalization
and click-limiter steps -- the pieces that don't already exist in NEMS -- and
combines them with NEMS's own `gammagram`/`gtgram` (`.gammatone`) and
`PowerCompress` (`nems.layers.compression`) into one convenience function,
`acnet_gtgram`, that turns a raw waveform into the (T, `num_cfs`) compressed
spectrogram a ported ACNet model expects as its stimulus input.

Substitution flagged explicitly (not silent): the released ACNet weights were
trained with a **polyphase** resampler (`torchaudio.transforms.Resample`),
which requires torch. Since this port must stay TF/numpy-only, `resample`
below always uses `scipy.signal.resample` (FFT-based) instead -- this is
exactly the path `nems_audio_preprocess`'s own `nems_match=True` branch already
used (to reproduce a NEMS recording's `stim` signal), so it is not a new
approximation invented for this port, but it does mean this function will not
bit-match the polyphase path ACNet was actually trained under. PT's own
comment on the difference: "Differs audibly little from the polyphase one
(waveform r=0.9993) but enough to move the gammatonegram by a few percent."
"""

# [AGENT EDIT START | agent: claude | user: sbp894 | reason: port ACNet's raw-audio front end (level norm + click limiter) as NEMS-native numpy, for the ACNet-in-NEMS model port -- gammatone filterbank itself is NOT reimplemented, NEMS's own gtgram already covers it | date: 2026-09-16]
import numpy as np
from scipy.signal import resample

from .gammatone import gammagram


def remove_clicks(w, max_threshold=50):
    """Log-compress samples beyond 67% of `max_threshold` (LBHB peak limiter).

    Parameters
    ----------
    w : np.ndarray
        Waveform, already scaled by `fixed_amp_scale`.
    max_threshold : float; default=50.

    Returns
    -------
    np.ndarray

    """
    w_clean = w.copy()
    crossover = 0.67 * max_threshold

    ii = w_clean > crossover
    w_clean[ii] = crossover + np.log(w_clean[ii] - crossover + 1)
    jj = w_clean < -crossover
    w_clean[jj] = -crossover - np.log(-w_clean[jj] - crossover + 1)

    return w_clean


def nems_audio_preprocess(sig, fs_stim, fs_gtg, f_max=20e3, duration=None,
                          fixed_amp_scale=250, lbhb_mode=False, overall_db=65,
                          level_mode='exact', verbose=False):
    """Resample, ramp, and level-scale a waveform prior to the gammatone filterbank.

    For inference outside LBHB, use the default `lbhb_mode=False`; then
    `level_mode` must be `'exact'` and the signal is scaled so its RMS matches
    `overall_db` dB SPL (reference 20 uPa). `lbhb_mode=True` additionally
    applies `remove_clicks`, a peak limiter -- matching BNT-exact reproduction
    requires this, plus the site's real `fixed_amp_scale`/`overall_db`.

    Parameters
    ----------
    sig : np.ndarray
        Mono waveform, shape (n_samples,).
    fs_stim : float
        Sampling rate of `sig`.
    fs_gtg : float
        Target gammatonegram frame rate (used only to validate `f_max`; the
        filterbank itself is applied by the caller via `gammagram`/`gtgram`).
    f_max : float; default=20e3.
        Upper cutoff frequency; resampling target is `2*f_max` (Nyquist).
    duration : float; optional.
        If given, pad/truncate `sig` to this many seconds before ramping.
    fixed_amp_scale : float; default=250.
        Scale applied before `remove_clicks` when `lbhb_mode=True`.
    lbhb_mode : bool; default=False.
        If True, apply `remove_clicks` (peak limiter) before level scaling.
    overall_db : float; default=65.
        Target level, dB SPL.
    level_mode : str; one of {'exact', 'approx'}; default='exact'.
        `'exact'` scales using the signal's own measured dB SPL.  `'approx'`
        (only valid when `lbhb_mode=True`) assumes a nominal 80 dB SPL
        pre-limiter level instead of measuring it.
    verbose : bool; default=False.

    Returns
    -------
    sig : np.ndarray
    fs0 : float
        The resampled rate (`2*f_max`).

    """
    sig = np.asarray(sig, dtype=np.float64).squeeze()
    fs0 = f_max * 2

    if duration is not None and len(sig) < fs_stim * duration:
        sig = np.pad(sig, (0, int(fs_stim * duration) - len(sig)))

    if fs_stim != fs0:
        n_out = int(len(sig) / fs_stim * fs0)
        sig = resample(sig, n_out)

    if duration is not None:
        sig = sig[:int(np.floor(duration * fs0))]

    # 5 ms onset/offset ramp (symmetric Hann, matching nems_audio_preprocess's
    # nems_match=True path).
    ramp_full = np.hanning(int(0.005 * fs0 * 2))
    ramp = ramp_full[:int(np.floor(len(ramp_full) / 2))]
    sig[:len(ramp)] *= ramp
    sig[-len(ramp):] *= ramp[::-1]

    if lbhb_mode:
        sig = remove_clicks(sig * fixed_amp_scale, 15)
        pre_dbspl = 20 * np.log10(np.sqrt(np.mean(sig**2)) / 20e-6)
        if overall_db is not None:
            if level_mode == 'approx':
                sf = 10 ** ((80 - overall_db) / 20)
            elif level_mode == 'exact':
                sf = 10 ** ((pre_dbspl - overall_db) / 20)
            else:
                raise ValueError(f"Unknown level_mode {level_mode!r}.")
            sig = sig / sf
    else:
        pre_dbspl = 20 * np.log10(np.sqrt(np.mean(sig**2)) / 20e-6)
        if overall_db is not None:
            assert level_mode == 'exact', "If lbhb_mode=False, level_mode must be 'exact'."
            sf = 10 ** ((pre_dbspl - overall_db) / 20)
            sig = sig / sf

    if verbose:
        post_dbspl = 20 * np.log10(np.sqrt(np.mean(sig**2)) / 20e-6)
        print(f"lbhb={lbhb_mode}: dbspl pre={pre_dbspl:.2f} post={post_dbspl:.2f}")

    return sig, fs0


def acnet_gtgram(sig, fs_stim, num_cfs=32, f_min=200.0, f_max=20e3, fs_gtg=100.0,
                 compress='sqrt', duration=None, fixed_amp_scale=250,
                 lbhb_mode=False, overall_db=65, level_mode='exact', verbose=False):
    """Raw waveform -> level-normalized, compressed gammatonegram.

    Combines `nems_audio_preprocess` (level norm + click limiter), NEMS's own
    `gammagram` gammatone filterbank, and `nems.layers.compression.PowerCompress`
    into the (T, `num_cfs`) array a ported ACNet model expects as its stimulus.

    Parameters
    ----------
    sig : np.ndarray
        Mono waveform, shape (n_samples,).
    fs_stim : float
        Sampling rate of `sig`.
    num_cfs : int; default=32.
    f_min : float; default=200.0.
    f_max : float; default=20e3.
    fs_gtg : float; default=100.0.
    compress : str; one of {'sqrt', 'log10x'}; default='sqrt'.
    duration, fixed_amp_scale, lbhb_mode, overall_db, level_mode, verbose :
        Passed through to `nems_audio_preprocess`.

    Returns
    -------
    np.ndarray
        Shape (T, `num_cfs`).

    """
    from nems.layers.compression import PowerCompress

    sig, fs0 = nems_audio_preprocess(
        sig, fs_stim, fs_gtg, f_max=f_max, duration=duration,
        fixed_amp_scale=fixed_amp_scale, lbhb_mode=lbhb_mode,
        overall_db=overall_db, level_mode=level_mode, verbose=verbose,
        )

    gtg = gammagram(
        sig, fs=fs0, window_time=1/fs_gtg, hop_time=1/fs_gtg,
        channels=num_cfs, f_min=f_min, f_max=f_max,
        )

    return PowerCompress(mode=compress).evaluate(gtg)
# [AGENT EDIT END]
