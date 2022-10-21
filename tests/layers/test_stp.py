import pytest
import numpy as np

from nems import Model
from nems.layers.stp import ShortTermPlasticity
from nems.visualization.layers import stp_test_input


def test_constructor():
    with pytest.raises(TypeError):
        # No shape
        stp = ShortTermPlasticity()
    # This should raise no errors.
    stp = ShortTermPlasticity(shape=(2,1))
    

class TestEvaluate:

    def test_one(self, spectrogram):
        time, spectral = spectrogram.shape
        spectrogram[spectrogram < 0] = 0  # STP only supports >= 0 for now.
        model = Model.from_keywords(f'wc.{spectral}x1-stp.1')
        out = model.predict(spectrogram)
        assert out.shape == (time, 1)

    def test_full(self, spectrogram):
        time, spectral = spectrogram.shape
        stp = ShortTermPlasticity(shape=(18,))
        out = stp.evaluate(spectrogram)
        assert out.shape == spectrogram.shape

    def test_quick(self, spectrogram):
        time, spectral = spectrogram.shape
        stp = ShortTermPlasticity(shape=(spectral,), quick_eval=True)
        out = stp.evaluate(spectrogram)

    def test_dep_only(self, spectrogram):
        time, spectral = spectrogram.shape
        stp = ShortTermPlasticity(shape=(spectral,), dep_only=True)
        out = stp.evaluate(spectrogram)
        assert out.max() <= spectrogram.max()

    def test_input(self):
        # 0: randomm, 1: square, 2: all zero
        input = stp_test_input()
        stp = ShortTermPlasticity(shape=(3,))
        u, t = stp.get_parameter_values()
        # default should be depression
        assert np.all(u > 0)

        output = stp.evaluate(input)
        # Depression should never increase 
        assert output.max() <= input.max()
        assert output.min() <= input.min()
        # All-zero input should be unchanged
        assert np.allclose(input[:, 2], output[:, 2])
