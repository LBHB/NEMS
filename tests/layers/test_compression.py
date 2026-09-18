import pytest
import numpy as np

from nems.layers.compression import PowerCompress


def test_constructors():
    # No shape needed -- compression is a fixed elementwise transform.
    pc = PowerCompress()
    assert pc.mode == 'sqrt'
    pc = PowerCompress(mode='log10x')
    assert pc.mode == 'log10x'

    with pytest.raises(ValueError):
        PowerCompress(mode='not_a_mode')


def test_from_keyword():
    pc = PowerCompress.from_keyword('pow.sqrt')
    assert pc.mode == 'sqrt'
    pc = PowerCompress.from_keyword('pow.log10x')
    assert pc.mode == 'log10x'

    with pytest.raises(ValueError):
        PowerCompress.from_keyword('pow')


class TestEvaluate:

    def test_sqrt_is_identity(self, spectrogram):
        pc = PowerCompress(mode='sqrt')
        out = pc.evaluate(spectrogram)
        assert out.shape == spectrogram.shape
        assert np.array_equal(out, spectrogram)

    def test_log10x_formula(self, spectrogram):
        # Matches PT_EncMdl_helpers_v2.MultiTask_BNTDataSet_Site_Nems's
        # log10x branch: c_gain=0.5, c_factor=10, applied to linear magnitude
        # recovered by squaring the sqrt-domain input.
        pc = PowerCompress(mode='log10x')
        out = pc.evaluate(spectrogram)
        expected = 0.5 * np.log(1 + 10 * spectrogram**2)
        assert out.shape == spectrogram.shape
        assert np.allclose(out, expected)

    def test_no_parameters(self, spectrogram):
        pc = PowerCompress(mode='log10x')
        assert list(pc.parameters.keys()) == []
