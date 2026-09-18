import pytest
import numpy as np

from nems.layers.batchnorm import BatchNorm1d


def test_constructors():
    with pytest.raises(TypeError):
        BatchNorm1d()  # no shape specified
    bn = BatchNorm1d(shape=(8,))
    assert bn.momentum == 0.1
    assert bn.eps == 1e-5


def test_from_keyword():
    bn = BatchNorm1d.from_keyword('bn.16')
    assert bn.shape == (16,)


class TestEvaluate:

    def test_shape_preserved(self, spectrogram):
        n = spectrogram.shape[-1]
        bn = BatchNorm1d(shape=(n,))
        out = bn.evaluate(spectrogram)
        assert out.shape == spectrogram.shape

    def test_eval_mode_is_default_and_uses_running_stats(self, spectrogram):
        n = spectrogram.shape[-1]
        bn = BatchNorm1d(shape=(n,))
        # Running stats start at mean=0, var=1 (the priors' means), gamma=1, beta=0
        # -- eval-mode output should just be `input` unchanged at these defaults.
        out = bn.evaluate(spectrogram)
        assert np.allclose(out, spectrogram, atol=1e-4)

    def test_train_mode_matches_hand_computed_formula(self, spectrogram):
        n = spectrogram.shape[-1]
        bn = BatchNorm1d(shape=(n,), momentum=0.1)
        bn.train_mode()
        gamma, beta, running_mean0, running_var0 = bn.get_parameter_values()

        out = bn.evaluate(spectrogram)

        batch_mean = spectrogram.mean(axis=0)
        batch_var = spectrogram.var(axis=0)  # biased
        expected = gamma * (spectrogram - batch_mean) / np.sqrt(batch_var + bn.eps) + beta
        assert np.allclose(out, expected)

        # Running stats should have moved from their initial values toward the
        # batch statistics (PyTorch BatchNorm1d convention: unbiased variance
        # for the running update).
        _, _, running_mean1, running_var1 = bn.get_parameter_values()
        n_elem = spectrogram.shape[0]
        unbiased_var = batch_var * n_elem / (n_elem - 1)
        expected_running_mean = 0.9 * running_mean0 + 0.1 * batch_mean
        expected_running_var = 0.9 * running_var0 + 0.1 * unbiased_var
        assert np.allclose(running_mean1, expected_running_mean)
        assert np.allclose(running_var1, expected_running_var)

    def test_eval_mode_uses_updated_running_stats(self, spectrogram):
        n = spectrogram.shape[-1]
        bn = BatchNorm1d(shape=(n,))
        bn.train_mode()
        bn.evaluate(spectrogram)  # updates running stats in place
        bn.eval_mode()

        _, _, running_mean, running_var = bn.get_parameter_values()
        gamma, beta, _, _ = bn.get_parameter_values()
        out = bn.evaluate(spectrogram)
        expected = gamma * (spectrogram - running_mean) / np.sqrt(running_var + bn.eps) + beta
        assert np.allclose(out, expected)

    def test_running_stats_are_permanent(self, spectrogram):
        n = spectrogram.shape[-1]
        bn = BatchNorm1d(shape=(n,))
        assert bn.parameters['running_mean'].is_frozen
        assert bn.parameters['running_var'].is_frozen
        assert not bn.parameters['gamma'].is_frozen
        assert not bn.parameters['beta'].is_frozen
