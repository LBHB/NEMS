import wave

import pytest
import numpy as np

from nems.layers.compression import PowerCompress
from nems.preprocessing.spectrogram import (
    load_wav, remove_clicks, nems_audio_preprocess, acnet_gtgram,
    resolve_site_calibration,
)
from nems.preprocessing.spectrogram.gammatone import gammagram


class TestLoadWav:

    def _write_wav(self, path, fs=40000, duration=0.1, n_channels=1, bits=16):
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        sig = 0.5 * np.sin(2 * np.pi * 1000 * t)
        with wave.open(str(path), 'wb') as w:
            w.setnchannels(n_channels)
            w.setsampwidth(bits // 8)
            w.setframerate(fs)
            if n_channels > 1:
                sig = np.tile(sig[:, None], (1, n_channels)).flatten()
            ints = (sig * 32767).astype(np.int16)
            w.writeframes(ints.tobytes())
        return sig, fs

    def test_mono_16bit_roundtrip(self, tmp_path):
        path = tmp_path / 'test.wav'
        expected, fs = self._write_wav(path)
        wav, fs_out = load_wav(str(path))
        assert fs_out == fs
        assert wav.shape == expected.shape
        assert np.allclose(wav, expected, atol=1e-4)

    def test_stereo_mixed_to_mono(self, tmp_path):
        path = tmp_path / 'stereo.wav'
        self._write_wav(path, n_channels=2)
        wav, fs = load_wav(str(path))
        assert wav.ndim == 1

    def test_not_a_wav_raises(self, tmp_path):
        path = tmp_path / 'not_a_wav.txt'
        path.write_bytes(b'not a riff file')
        with pytest.raises(ValueError):
            load_wav(str(path))


class TestRemoveClicks:

    def test_below_threshold_unchanged(self):
        w = np.array([-5.0, 0.0, 5.0, 9.0])
        out = remove_clicks(w, max_threshold=15)
        # crossover = 0.67*15 = 10.05, nothing here exceeds it
        assert np.allclose(out, w)

    def test_above_threshold_log_compressed(self):
        w = np.array([20.0, -20.0])
        out = remove_clicks(w, max_threshold=15)
        crossover = 0.67 * 15
        expected_pos = crossover + np.log(20.0 - crossover + 1)
        expected_neg = -crossover - np.log(20.0 - crossover + 1)
        assert np.allclose(out, [expected_pos, expected_neg])

    def test_does_not_modify_input(self):
        w = np.array([20.0, -20.0])
        w_orig = w.copy()
        remove_clicks(w, max_threshold=15)
        assert np.array_equal(w, w_orig)


class TestNemsAudioPreprocess:

    def test_lbhb_false_requires_exact(self):
        sig = np.random.randn(4000)
        with pytest.raises(AssertionError):
            nems_audio_preprocess(sig, fs_stim=40000, fs_gtg=100, level_mode='approx',
                                  lbhb_mode=False)

    def test_output_level_matches_target(self):
        # A pure sine at an arbitrary level should end up at the requested
        # overall_db (RMS-based, reference 20 uPa), regardless of lbhb_mode.
        fs_stim = 40000
        t = np.arange(fs_stim) / fs_stim
        sig = 0.01 * np.sin(2 * np.pi * 1000 * t)
        target_db = 65
        out, fs0 = nems_audio_preprocess(sig, fs_stim, fs_gtg=100, f_max=fs_stim/2,
                                         overall_db=target_db, level_mode='exact',
                                         lbhb_mode=False)
        out_db = 20 * np.log10(np.sqrt(np.mean(out**2)) / 20e-6)
        assert np.isclose(out_db, target_db, atol=0.5)

    def test_resamples_to_2x_fmax(self):
        sig = np.random.randn(4000)
        _, fs0 = nems_audio_preprocess(sig, fs_stim=40000, fs_gtg=100, f_max=10e3,
                                       overall_db=None, lbhb_mode=False,
                                       level_mode='exact')
        assert fs0 == 20e3

    # [AGENT EDIT START | agent: claude | user: svd | reason: cover new level_mode 'max'/'rms' (baphy OLP NormalizeRMS No/Yes) | date: 2026-10-02]
    def _noise(self, fs_stim=40000, scale=0.01):
        rng = np.random.default_rng(0)
        return scale * rng.standard_normal(fs_stim)

    def test_max_mode_peak_5_at_80db(self):
        out, _ = nems_audio_preprocess(self._noise(), 40000, fs_gtg=100, f_max=20e3,
                                       lbhb_mode=True, level_mode='max', overall_db=80)
        assert np.isclose(np.max(np.abs(out)), 5.0)

    def test_rms_mode_rms_3p5349_at_80db(self):
        out, _ = nems_audio_preprocess(self._noise(), 40000, fs_gtg=100, f_max=20e3,
                                       lbhb_mode=True, level_mode='rms', overall_db=80)
        # remove_clicks only touches samples > 10.05 SD, so RMS is ~exact
        assert np.isclose(np.sqrt(np.mean(out**2)), 3.5349, rtol=1e-3)

    @pytest.mark.parametrize('level_mode', ['max', 'rms'])
    def test_input_scale_invariant_and_overall_db_attenuates(self, level_mode):
        kw = dict(fs_stim=40000, fs_gtg=100, f_max=20e3, lbhb_mode=True, level_mode=level_mode)
        a, _ = nems_audio_preprocess(self._noise(scale=0.01), overall_db=65, **kw)
        b, _ = nems_audio_preprocess(self._noise(scale=0.3), overall_db=65, **kw)
        c, _ = nems_audio_preprocess(self._noise(scale=0.01), overall_db=80, **kw)
        assert np.allclose(a, b)
        assert np.allclose(c, a * 10 ** (15 / 20))

    @pytest.mark.parametrize('level_mode', ['max', 'rms'])
    def test_new_modes_require_lbhb_mode(self, level_mode):
        with pytest.raises(AssertionError):
            nems_audio_preprocess(self._noise(), 40000, fs_gtg=100, level_mode=level_mode,
                                  lbhb_mode=False)
    # [AGENT EDIT END]


class TestAcnetGtgram:

    def test_output_shape(self):
        fs_stim = 40000
        duration = 0.5
        sig = np.random.randn(int(fs_stim * duration))
        out = acnet_gtgram(sig, fs_stim, num_cfs=32, f_min=200.0, f_max=10e3,
                           fs_gtg=100.0, duration=duration,
                           overall_db=65, level_mode='exact', lbhb_mode=False)
        assert out.shape[1] == 32
        assert out.shape[0] > 0

    def test_no_compress_kwarg(self):
        # acnet_gtgram never applies compression -- that's PowerCompress's
        # job, inside the model. Passing compress here should be a hard
        # error (no such parameter), not silently ignored.
        fs_stim = 40000
        sig = np.random.randn(fs_stim)
        with pytest.raises(TypeError):
            acnet_gtgram(sig, fs_stim, compress='log10x')

    def test_output_matches_gammagram_exactly(self):
        # acnet_gtgram must be a pure pass-through of gammagram's amplitude
        # output -- no extra transform of its own. This is the contract
        # PowerCompress relies on (see its own docstring); a regression
        # here would silently double- or under-compress every caller.
        fs_stim = 40000
        sig = 0.05 * np.random.randn(int(fs_stim * 0.5))
        gtg = acnet_gtgram(sig, fs_stim, num_cfs=32, f_min=200.0, f_max=10e3,
                           fs_gtg=100.0, overall_db=65, level_mode='exact',
                           lbhb_mode=False)
        preprocessed, fs0 = nems_audio_preprocess(
            sig, fs_stim, fs_gtg=100.0, f_max=10e3, overall_db=65,
            level_mode='exact', lbhb_mode=False,
            )
        expected = gammagram(preprocessed, fs=fs0, window_time=1 / 100.0,
                             hop_time=1 / 100.0, channels=32, f_min=200.0,
                             f_max=10e3)
        assert np.array_equal(gtg, expected)

    def test_output_feeds_powercompress_log10x_without_conversion(self):
        # Regression test for the 2026-09-28 bug: PowerCompress(mode=
        # 'log10x') used to square its input, assuming acnet_gtgram's
        # amplitude output needed recovering from a deeper "sqrt" domain
        # that acnet_gtgram never actually produced. That mismatch was
        # confirmed on real BigNat site data (get_embeddings/predict r_test
        # ratio ~0.40 vs. published, across 8 sites spanning 4 animals) --
        # this test pins the contract with no external data required:
        # acnet_gtgram's raw output must go straight into PowerCompress,
        # matching PT_EncMdl_helpers_v2's real training formula exactly.
        fs_stim = 40000
        sig = 0.05 * np.random.randn(fs_stim)
        gtg = acnet_gtgram(sig, fs_stim, num_cfs=32, f_min=200.0, f_max=10e3,
                           fs_gtg=100.0, overall_db=65, level_mode='exact',
                           lbhb_mode=False)
        out = PowerCompress(mode='log10x').evaluate(gtg)
        expected = 0.5 * np.log(1 + 10 * gtg)
        assert np.allclose(out, expected)


class TestResolveSiteCalibration:

    def test_unlisted_site_passes_through(self):
        assert resolve_site_calibration('PRN007a', 65, 250) == (65, 250)

    def test_listed_site_overridden(self):
        # REI058a: confirmed 2026-09-28 via direct celldb query -- reports
        # OveralldB=50 (Reishi rig hardware bug), actually recorded at 65.
        assert resolve_site_calibration('REI058a', 50, 250) == (65, 250)

    def test_listed_site_query_mismatch_still_trusts_table(self):
        # Even if the live query no longer matches what the table recorded
        # as the query value (celldb changed, or a typo), the table's
        # true_* values are still what's returned -- just with a warning.
        with pytest.warns(UserWarning):
            out = resolve_site_calibration('REI058a', 999, 250)
        assert out == (65, 250)

    def test_overrides_csv_none_disables_lookup(self):
        assert resolve_site_calibration('REI058a', 50, 250, overrides_csv=None) == (50, 250)

    def test_missing_csv_path_passes_through(self):
        assert resolve_site_calibration('REI058a', 50, 250,
                                        overrides_csv='/nonexistent/path.csv') == (50, 250)
