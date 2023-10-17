import pytest
import numpy as np
from nems.preprocessing import (
    indices_by_fraction, split_at_indices, JackknifeIterator
)
"""
Pytests for preprocessing split functions & classes including;
JackknifeIterator, indicies_by_fraction, split_at_indices
"""

class TestJackknifeIterator:
    """
    Testing several input types, parameter values, and 
    core functionality of JackknifeIterator Object
    """
    def test_single_input(self, spectrogram, response):
        # Test that the creation of a single input jk is successful with no error
        jackknife_iterator = JackknifeIterator(spectrogram, target=response, samples=5, axis=0)

    def test_single_iter(self, spectrogram, response):
        # A single iteration for a standard jk should return a sliced ndarray
        jackknife_iterator = JackknifeIterator(spectrogram, target=response, samples=5, axis=0)
        iter = next(jackknife_iterator)
        assert isinstance(iter['input'], np.ndarray)

    def test_samples(self, spectrogram, response):
        # Similar to above, but we are using a higher sample count and asserting that it
        # produces smaller subsets of arrays
        samples = 25
        jackknife_iterator = JackknifeIterator(spectrogram, target=response, samples=samples, axis=0)
        lower_jackknife_iterator = JackknifeIterator(spectrogram, target=response, samples=5, axis=0)

        iter = next(jackknife_iterator)
        lower_iter = next(lower_jackknife_iterator)

        assert iter['input'].shape[0] == (spectrogram.shape[0] - spectrogram.shape[0]/samples) and iter['input'].shape[0] > lower_iter['input'].shape[0]
    
    def test_axis(self):
        # Confirm that jk actually cuts along the axis given
        samples = 2
        spectrogram = np.random.rand(100, 10, 18)
        response = np.random.rand(100, 10, 1)

        jackknife_iterator = JackknifeIterator(spectrogram, target=response, samples=samples, axis=1)
        iter = next(jackknife_iterator)

        assert iter['input'].shape[1] == (spectrogram.shape[1] - spectrogram.shape[1]/samples)

    def test_rand_idx(self, spectrogram, response):
        shuffle_jackknife_iterator = JackknifeIterator(spectrogram, target=response, samples=5, axis=0, shuffle=True)
        jackknife_iterator = JackknifeIterator(spectrogram, target=response, samples=5, axis=0, shuffle=False)
        shuffle_idx = shuffle_jackknife_iterator.mask_list[0]
        idx = jackknife_iterator.mask_list[0]

        assert np.sum(idx) != np.sum(shuffle_idx)

    def test_multi_input(self, spectrogram_with_samples, response):
        # Test that the creation of a multi input jk is successful with no error
        jackknife_iterator = JackknifeIterator(spectrogram_with_samples, target=response, samples=5, axis=0)

    def test_inverse(self, spectrogram, response):
        jackknife_iterator = JackknifeIterator(spectrogram, target=response, samples=5, axis=0, inverse=True)
        est, val = next(jackknife_iterator)

        assert est is not None and val is not None


