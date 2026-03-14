import pytest
import numpy as np

from nems.layers.weight_channels import WeightChannels
from nems.layers.filter import FiniteImpulseResponse, STRF, PoleZeroFIR
from nems.tools.debug import generate_random_input, compare_layer_eval

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

    def test_full_rank(self):
        spectrogram = generate_random_input((100,4))
        time, spectral = spectrogram.shape
        # 50 ms filter, 18 spectral channels (full-rank STRF)
        fir = FiniteImpulseResponse(shape=(5, spectral))
        out = fir.evaluate(spectrogram)
        assert out.shape == (time, 1)

    def test_full_with_outputs(self):
        # Same but with 3 outputs (i.e. filter-bank)
        spectrogram = generate_random_input((100,4))
        time, spectral = spectrogram.shape
        fir_bank = FiniteImpulseResponse(shape=(5, spectral, 3))
        bank_out = fir_bank.evaluate(spectrogram)
        # Rank dimension should be squeezed out.
        assert bank_out.shape == (time, 3)

    def test_rank_with_outputs(self):
        # Pretend spectrogram has output dimension, rank 1
        spectrogram = generate_random_input((100,4))
        time, spectral = spectrogram.shape
        spectrogram_with_outputs = spectrogram.reshape(time, 1, spectral)
        fir_with_outputs = FiniteImpulseResponse(shape=(5, 1, spectral))
        outputs_out = fir_with_outputs.evaluate(spectrogram_with_outputs)
        # Rank dimension should be squeezed out.
        assert outputs_out.shape == (time, spectral)

    def test_reapply_same_filter(self):
        # Pretend spectrogram has output dimension, rank 1
        spectrogram = generate_random_input((100,4))
        time, spectral = spectrogram.shape
        spectrogram_with_outputs = spectrogram.reshape(time, 1, spectral)
        fir_no_outputs = FiniteImpulseResponse(shape=(5, 1))  # 1 implied output
        no_outputs_out = fir_no_outputs.evaluate(spectrogram_with_outputs)
        # Rank dimension should be squeezed out, but outputs of spectrogram
        # should be preserved
        assert no_outputs_out.shape == (time, spectral)

    def test_pole_zero(self):
        spectrogram = generate_random_input((100,4))
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

    def test_strf(self):
        spectrogram = generate_random_input((100,6))
        time, spectral = spectrogram.shape

        shape = (spectral, 2, 10, 3) # (C, R, T, N)
        wshape = (shape[0], shape[1], shape[3])
        fshape = (shape[2], shape[1], shape[3])

        strf = STRF(shape=shape)
        out1 = strf.evaluate(spectrogram)

        wc = WeightChannels(shape=wshape)
        fir = FiniteImpulseResponse(shape=fshape)
        out2 = fir.evaluate(wc.evaluate(spectrogram))

        assert out1.shape==out2.shape
        assert np.all(out1==out2)

    def test_strf_kw(self):
        spectrogram = generate_random_input((100,6))
        time, spectral = spectrogram.shape

        shape = (spectral, 2, 10, 3) # (C, R, T, N)
        shape = [str(s) for s in shape]
        wshape = "x".join((shape[0], shape[1], shape[3]))
        fshape = "x".join((shape[2], shape[1], shape[3]))
        shape = "x".join(shape)

        strf = STRF.from_keyword(f"strf.{shape}")
        out1 = strf.evaluate(spectrogram)

        wc = WeightChannels.from_keyword(f"wc.{wshape}")
        fir = FiniteImpulseResponse.from_keyword(f"fir.{fshape}")
        out2 = fir.evaluate(wc.evaluate(spectrogram))

        assert out1.shape==out2.shape
        assert np.all(out1==out2)


    def test_backends(self):
        spectrogram = generate_random_input((100,6))
        time, spectral = spectrogram.shape

        shape = (spectral, 2, 10, 3) # (C, R, T, N)
        strf = STRF(shape=shape)
        strf.set_dtype('float32')

        numpy_out, tf_out = compare_layer_eval(strf, spectrogram)

        assert np.mean((numpy_out.flatten() - tf_out.flatten()) ** 2) < 1e-4
