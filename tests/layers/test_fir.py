import pytest
import numpy as np

from nems.layers.filter import FiniteImpulseResponse, PoleZeroFIR


def test_constructor():
    with pytest.raises(TypeError):
        # No shape
        fir = FiniteImpulseResponse()
    with pytest.raises(TypeError):
        # Shape has too few dimensions
        fir = FiniteImpulseResponse(shape=(2,))
    # But this should raise no errors.
    fir = FiniteImpulseResponse(shape=(1,2,3))
    

class TestEvaluate:

    def test_full_rank(self, spectrogram):
        time, spectral = spectrogram.shape
        # 50 ms filter, 18 spectral channels (full-rank STRF)
        fir = FiniteImpulseResponse(shape=(5, spectral))
        out = fir.evaluate(spectrogram)
        assert out.shape == (time, 1)

    def test_full_with_outputs(self, spectrogram):
        # Same but with 3 outputs (i.e. filter-bank)
        time, spectral = spectrogram.shape
        fir_bank = FiniteImpulseResponse(shape=(5, spectral, 3))
        bank_out = fir_bank.evaluate(spectrogram)
        # Rank dimension should be squeezed out.
        assert bank_out.shape == (time, 3)

    def test_rank_with_outputs(self, spectrogram):
        # Pretend spectrogram has output dimension, rank 1
        time, spectral = spectrogram.shape
        spectrogram_with_outputs = spectrogram.reshape(time, 1, spectral)
        fir_with_outputs = FiniteImpulseResponse(shape=(5, 1, spectral))
        outputs_out = fir_with_outputs.evaluate(spectrogram_with_outputs)
        # Rank dimension should be squeezed out.
        assert outputs_out.shape == (time, spectral)

    def test_reapply_same_filter(self, spectrogram):
        # Pretend spectrogram has output dimension, rank 1
        time, spectral = spectrogram.shape
        spectrogram_with_outputs = spectrogram.reshape(time, 1, spectral)
        fir_no_outputs = FiniteImpulseResponse(shape=(5, 1))  # 1 implied output
        no_outputs_out = fir_no_outputs.evaluate(spectrogram_with_outputs)
        # Rank dimension should be squeezed out, but outputs of spectrogram
        # should be preserved
        assert no_outputs_out.shape == (time, spectral)

    def test_pole_zero(self, spectrogram):
        time, spectral = spectrogram.shape
        fir_no_outputs = PoleZeroFIR(
            shape=(15, spectral), n_poles=3, n_zeros=1, fs=100
            )
        out = fir_no_outputs.evaluate(spectrogram)
        assert out.shape == (time, 1)

        # pretend spectrogram has an output dimension, rank 1
        spectrogram_with_outputs = spectrogram.reshape(time, 1, spectral)
        fir_with_outputs = PoleZeroFIR(
            shape=(15, 1, spectral), n_poles=2, n_zeros=2, fs=10
        )
        out2 = fir_with_outputs.evaluate(spectrogram_with_outputs)
        assert out2.shape == (time, spectral)
