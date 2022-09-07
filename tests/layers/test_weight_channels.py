import pytest
import numpy as np

from nems.layers.weight_channels import (
    WeightChannels, WeightChannelsGaussian, WeightChannelsMulti
    )


def test_constructor():
    with pytest.raises(TypeError):
        # No shape
        wc = WeightChannels()
    with pytest.raises(TypeError):
        # No shape
        wc = WeightChannels(None)
    with pytest.raises(TypeError):
        # Too few dimensions
        # TODO: may decide to allow as few as 1 dims in the future
        wc = WeightChannels(shape=(2,))

    # These should raise no errors
    wc = WeightChannels(shape=(18, 2))
    wc = WeightChannels(shape=(10, 3, 5))
    wcg = WeightChannelsGaussian(shape=(18, 2))


class TestEvaluate:

    def test_rank1(self, spectrogram):
        time, spectral = spectrogram.shape
        wc = WeightChannels(shape=(spectral, 1))
        out = wc.evaluate(spectrogram)
        assert out.shape == (time, 1)

    def test_rank3(self, spectrogram):
        time, spectral = spectrogram.shape
        wc = WeightChannels(shape=(spectral, 3))
        out = wc.evaluate(spectrogram)
        assert out.shape == (time, 3)

    def test_multi_rank1(self, spectrogram_with_samples):
        samples, time, spectral = spectrogram_with_samples.shape
        # pretend samples are output channels
        spectrogram = np.moveaxis(spectrogram_with_samples, 0,-1)
        wcm = WeightChannelsMulti(shape=(spectral, 1, samples))
        out = wcm.evaluate(spectrogram)
        assert out.shape == (time, 1, samples)

    def test_multi_rank4(self, spectrogram_with_samples):
        samples, time, spectral = spectrogram_with_samples.shape
        # pretend samples are output channels
        spectrogram = np.moveaxis(spectrogram_with_samples, 0, -1)
        wcm = WeightChannelsMulti(shape=(spectral, 4, samples))
        out = wcm.evaluate(spectrogram)
        assert out.shape == (time, 4, samples)

    def test_gaussian_rank1(self, spectrogram):
        time, spectral = spectrogram.shape
        wcg = WeightChannelsGaussian(shape=(spectral, 1))
        out = wcg.evaluate(spectrogram)
        assert out.shape == (time, 1)

    def test_gaussian_rank5(self, spectrogram):
        time, spectral = spectrogram.shape
        wcg = WeightChannelsGaussian(shape=(spectral, 5))
        out = wcg.evaluate(spectrogram)
        assert out.shape == (time, 5)
