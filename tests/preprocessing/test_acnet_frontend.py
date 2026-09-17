import pytest
import numpy as np

from nems.preprocessing.spectrogram import (
    remove_clicks, nems_audio_preprocess, acnet_gtgram,
)


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


class TestAcnetGtgram:

    def test_output_shape(self):
        fs_stim = 40000
        duration = 0.5
        sig = np.random.randn(int(fs_stim * duration))
        out = acnet_gtgram(sig, fs_stim, num_cfs=32, f_min=200.0, f_max=10e3,
                           fs_gtg=100.0, compress='sqrt', duration=duration,
                           overall_db=65, level_mode='exact', lbhb_mode=False)
        assert out.shape[1] == 32
        assert out.shape[0] > 0

    def test_compress_modes_differ(self):
        fs_stim = 40000
        duration = 0.5
        np.random.seed(0)
        sig = np.random.randn(int(fs_stim * duration))
        sqrt_out = acnet_gtgram(sig, fs_stim, num_cfs=8, f_max=10e3, duration=duration,
                                overall_db=65, level_mode='exact', lbhb_mode=False,
                                compress='sqrt')
        log_out = acnet_gtgram(sig, fs_stim, num_cfs=8, f_max=10e3, duration=duration,
                               overall_db=65, level_mode='exact', lbhb_mode=False,
                               compress='log10x')
        assert sqrt_out.shape == log_out.shape
        assert not np.allclose(sqrt_out, log_out)
        # Exact relationship: log10x = 0.5*log(1 + 10*sqrt_out**2)
        assert np.allclose(log_out, 0.5 * np.log(1 + 10 * sqrt_out**2))
