import pytest
import numpy as np

from nems.layers.weight_channels import (
    WeightChannels, WeightChannelsGaussian, WeightChannelsMulti,
    WeightGaussianExpand
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

"""
import matplotlib.pyplot as plt
import importlib
from nems.layers import weight_channels
importlib.reload(weight_channels)

spectrogram = np.random.randn(100,2)
time, spectral = spectrogram.shape
wcg = weight_channels.WeightGaussianExpand(shape=(spectral, 18, 2))
wcg['mean']=np.array([[0.1, 0.2],[0.5, 0.6]]).T
wcg['sd']=np.array([[0.1, 0.1],[0.2, 0.2]]).T
out = wcg.evaluate(spectrogram)
print(out.shape)

f,ax=plt.subplots(1,2)
ax[0].imshow(out[:,0,:].T,origin='lower')
ax[1].imshow(out[:,1,:].T,origin='lower')

wcg = weight_channels.WeightGaussianExpand(shape=(spectral, 18))
wcg['mean']=np.array([0.1, 0.7])
wcg['sd']=np.array([0.1, 0.2])
wcg['amp']=np.array([1, 2])
out = wcg.evaluate(spectrogram)
print(out.shape)

f,ax=plt.subplots()
ax.imshow(out.T,origin='lower')
"""
