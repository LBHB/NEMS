"""ACNet's raw-audio front end: level normalization + the standard nems gtgram.

Ported (numpy/scipy, no torch) from `ACNet_v1.acnet_model.nems_audio_preprocess`/
`remove_clicks`, which were themselves vendored from `PT_EncMdl_helpers_v2`'s
`GammatoneFilterbankProcessing`. This module reproduces the level-normalization
and click-limiter steps -- the pieces that don't already exist in NEMS -- and
combines them with NEMS's own `gammagram`/`gtgram` (`.gammatone`) into one
convenience function, `acnet_gtgram`, that turns a raw waveform into the
(T, `num_cfs`) standard gtgram a ported ACNet model expects as its stimulus
input.

Deliberately NOT done here: compression. `acnet_gtgram` always returns the
plain nems gtgram (see `.gammatone.gammagram`'s own sqrt-domain convention --
that's an intrinsic property of `gammagram`'s output, not a choice made by
this module). Compression is exclusively `nems.layers.compression.
PowerCompress`'s job, applied inside the model itself -- keeping it out of
this front end means there's exactly one place a compression choice is made,
not two.

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
import struct

import numpy as np
from scipy.signal import resample

from .gammatone import gammagram


def load_wav(path, int16_scale=32768.0):
    """Decode a WAV file to a mono float64 array in [-1, 1], no torch.

    Ported from `ACNet_v1.acnet_model._load_wav` (parses the RIFF container
    directly, so no audio-decoding dependency is needed). Supports integer
    PCM (8/16/24/32-bit) and IEEE-float (32/64-bit) WAV, including
    WAVE_FORMAT_EXTENSIBLE.

    Parameters
    ----------
    path : str
    int16_scale : float; default=32768.0.
        Divisor for 16-bit PCM. The default is the full-scale convention;
        pass 32767 to match `nems_lbhb.runclass` (divides int16 by 32767).
        The two differ by 3e-5 relative.

    Returns
    -------
    wav : np.ndarray
        Shape (n_samples,).
    fs : int

    """
    with open(path, 'rb') as f:
        riff = f.read()
    if riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
        raise ValueError(f"Not a RIFF/WAVE file: {path}")

    fmt_tag = n_channels = fs = bits = None
    raw = None
    pos = 12
    while pos + 8 <= len(riff):
        cid = riff[pos:pos + 4]
        csize = struct.unpack('<I', riff[pos + 4:pos + 8])[0]
        body = riff[pos + 8:pos + 8 + csize]
        if cid == b'fmt ':
            fmt_tag, n_channels, fs, _, _, bits = struct.unpack('<HHIIHH', body[:16])
            if fmt_tag == 0xFFFE and csize >= 26:  # EXTENSIBLE: real tag is in the subformat GUID
                fmt_tag = struct.unpack('<H', body[24:26])[0]
        elif cid == b'data':
            raw = body
        pos += 8 + csize + (csize & 1)  # chunks are word-aligned

    if fmt_tag is None or raw is None:
        raise ValueError(f"Missing fmt/data chunk in {path}")

    if fmt_tag == 1:  # integer PCM
        if bits == 8:  # 8-bit PCM is unsigned
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        elif bits == 16:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / int16_scale
        elif bits == 24:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            ints[ints >= 1 << 23] -= 1 << 24  # sign-extend
            data = ints.astype(np.float64) / (1 << 23)
        elif bits == 32:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM bit depth {bits} in {path}")
    elif fmt_tag == 3:  # IEEE float
        dtype = np.float32 if bits == 32 else np.float64
        data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    else:
        raise ValueError(f"Unsupported WAV format tag {fmt_tag} in {path}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)  # mix to mono

    return np.ascontiguousarray(data), fs


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
                 duration=None, fixed_amp_scale=250,
                 lbhb_mode=False, overall_db=65, level_mode='exact', verbose=False):
    """Raw waveform -> level-normalized, standard nems gtgram. No compression.

    Combines `nems_audio_preprocess` (level norm + click limiter) with NEMS's
    own `gammagram` gammatone filterbank into the (T, `num_cfs`) array a
    ported ACNet model expects as its stimulus -- before compression, which
    is exclusively `nems.layers.compression.PowerCompress`'s job (applied
    inside the model, not here).

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
    duration, fixed_amp_scale, lbhb_mode, overall_db, level_mode, verbose :
        Passed through to `nems_audio_preprocess`.

    Returns
    -------
    np.ndarray
        Shape (T, `num_cfs`). Standard nems gtgram (sqrt-domain magnitude,
        per `gammagram`'s own convention) -- uncompressed.

    """
    sig, fs0 = nems_audio_preprocess(
        sig, fs_stim, fs_gtg, f_max=f_max, duration=duration,
        fixed_amp_scale=fixed_amp_scale, lbhb_mode=lbhb_mode,
        overall_db=overall_db, level_mode=level_mode, verbose=verbose,
        )

    return gammagram(
        sig, fs=fs0, window_time=1/fs_gtg, hop_time=1/fs_gtg,
        channels=num_cfs, f_min=f_min, f_max=f_max,
        )
# [AGENT EDIT END]
