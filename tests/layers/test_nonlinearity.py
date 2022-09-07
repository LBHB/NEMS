import pytest
import numpy as np

from nems.layers.nonlinearity import DoubleExponential, RectifiedLinear


def test_constructors():
    # dexp
    with pytest.raises(TypeError):
        # No shape specified.
        dexp = DoubleExponential()
    dexp = DoubleExponential(shape=(5,))
    dexp = DoubleExponential(shape=(2,1))

    # ReLU
    with pytest.raises(TypeError):
        # Shape has too few dimensions.
        relu = RectifiedLinear(shape=())
    relu = RectifiedLinear(shape=(1,))
    relu = RectifiedLinear(shape=(1,2,3))

class TestEvaluate:

    def test_dexp1(self, spectrogram):
        dexp = DoubleExponential(shape=(1,))
        out = dexp.evaluate(spectrogram)
        # This should work b/c dexp should broadcast to however many channels
        # spectrogram has. Nonlinearities shouldn't change shape of inputs.
        assert out.shape == spectrogram.shape

    def test_dexp3(self, spectrogram):
        dexp = DoubleExponential(shape=(3,))
        if (spectrogram.shape[-1] == 3) or (spectrogram.shape[-1] == 1):
            out = dexp.evaluate(spectrogram)
            assert out.shape == spectrogram.shape
        else:
            with pytest.raises(ValueError):
                # Can't broadcast unless spectrogram has 1 output channel.
                dexp.evaluate(spectrogram)

    def test_dexp_to_1(self, spectrogram):
        dexp = DoubleExponential(shape=(2,))
        newax = spectrogram[..., np.newaxis]
        out = dexp.evaluate(newax)  # spectrogram should broadcast
        assert out.shape == spectrogram.shape + (2,)

    def test_relu1(self, spectrogram):
        relu = RectifiedLinear(shape=(1,))
        out = relu.evaluate(spectrogram)
        assert out.shape == spectrogram.shape

    def test_relu_multi(self, spectrogram):
        newax = spectrogram[..., np.newaxis]
        relu = RectifiedLinear(shape=(1,3))
        out = relu.evaluate(newax)
        # Multi-dim nonlinearities should also work. ReLU should broadcast to
        # input for 1->spectral, input should broadcast to relu for 1->3.
        assert out.shape == spectrogram.shape + (3,)
