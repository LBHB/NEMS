import pytest
import numpy as np

from nems.layers.compression import PowerCompress


def test_constructors():
    # No shape needed -- compression is a fixed elementwise transform.
    pc = PowerCompress()
    assert pc.mode is None
    pc = PowerCompress(mode='sqrt')
    assert pc.mode == 'sqrt'
    pc = PowerCompress(mode='log10x')
    assert pc.mode == 'log10x'

    with pytest.raises(ValueError):
        PowerCompress(mode='not_a_mode')


def test_from_keyword():
    pc = PowerCompress.from_keyword('pow.none')
    assert pc.mode is None
    pc = PowerCompress.from_keyword('pow.sqrt')
    assert pc.mode == 'sqrt'
    pc = PowerCompress.from_keyword('pow.log10x')
    assert pc.mode == 'log10x'

    with pytest.raises(ValueError):
        PowerCompress.from_keyword('pow')


class TestEvaluate:

    def test_none_is_identity(self, spectrogram):
        pc = PowerCompress(mode=None)
        out = pc.evaluate(spectrogram)
        assert out.shape == spectrogram.shape
        assert np.array_equal(out, spectrogram)

    def test_sqrt_formula(self, spectrogram):
        # Matches MultiTask_BNTDataSet_Site_Nems's compress='sqrt' branch --
        # a literal no-op at dataset *load* time, but the archive it loads
        # from was itself written via export_data.py --compress sqrt (i.e.
        # is already amplitude**0.5), so a model trained this way (v2, v2.2
        # -- see nems.models.ACNet.load_acnet) genuinely consumes
        # sqrt(amplitude). This is a second, independent sqrt from
        # gtgram's own energy->amplitude step (always applied upstream,
        # never a choice) -- see this layer's own docstring for why the
        # 2026-09-17 None-rename didn't account for this case.
        pc = PowerCompress(mode='sqrt')
        out = pc.evaluate(spectrogram)
        expected = np.sqrt(np.abs(spectrogram))
        assert out.shape == spectrogram.shape
        assert np.allclose(out, expected)

    def test_log10x_formula(self, spectrogram):
        # Matches PT_EncMdl_helpers_v2.MultiTask_BNTDataSet_Site_Nems's
        # log10x branch: c_gain=0.5, c_factor=10, applied directly to the
        # standard gtgram's amplitude-domain input. PT's own archived
        # training data happened to be cached as amplitude**0.5 for storage
        # efficiency (squared back before this formula there) -- that is
        # not part of the compression itself, and `acnet_gtgram` never
        # introduces it, so no squaring belongs here (see PowerCompress's
        # own docstring for the 2026-09-28 bug this used to have).
        pc = PowerCompress(mode='log10x')
        out = pc.evaluate(spectrogram)
        expected = 0.5 * np.log(1 + 10 * spectrogram)
        assert out.shape == spectrogram.shape
        assert np.allclose(out, expected)

    def test_no_parameters(self, spectrogram):
        pc = PowerCompress(mode='log10x')
        assert list(pc.parameters.keys()) == []
