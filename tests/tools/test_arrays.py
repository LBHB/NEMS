import pytest
import numpy as np

from nems.tools.arrays import (
    broadcast_axis_shape, broadcast_axes, broadcast_dicts, concatenate_dicts,
    apply_to_dict
    )


class TestBroadcast:

    x = np.random.rand(100, 1, 2, 3)
    y = np.random.rand(1, 100, 3, 2)

    def test_axes_shape(self):
        shape = broadcast_axis_shape(self.x, self.y, axis=0)
        # Even though this isn't a valid broadcast, this should just replace
        # x.shape[0] with y.shape[0], so there should be no error.
        assert shape == (1, 1, 2, 3)

        shape = broadcast_axis_shape(self.y, self.x, axis=0)
        # This is the goal for broadcasting: on ax 0, broadcast 1->100, other
        # axes stay the same.
        assert shape == (100, 100, 3, 2)

        shape = broadcast_axis_shape(self.x, self.y, axis=-1)
        # Also not a valid broadcast, just a different axis, but should still
        # not complain.
        assert shape == (100, 1, 2, 2)

    def test_broadcast_axes(self):
        with pytest.raises(ValueError):
            # Can't broadcast 100 -> 1
            broadcast_axes(self.x, self.y, axis=0)

        # But 1 -> 100 should be fine
        array = broadcast_axes(self.y, self.x, axis=0)
        assert array.shape == (100, 100, 3, 2)

        # 2 -> 3 also should not work
        with pytest.raises(ValueError):
            broadcast_axes(self.x, self.y, axis=2)

    def test_broadcast_dicts(self):
        d1 = {'array': self.y}
        d2 = {'array': self.x}
        # Should broadcast 1 -> 100
        d3 = broadcast_dicts(d1, d2, axis=0, debug_memory=True)
        assert d3['array'].shape == (100, 100, 3, 2)

        d1['array2'] = self.x
        d2['array2'] = self.y
        # x -> y on axis should fail a direct broadcast, but shouldn't raise
        # an error here because broadcast_dicts only broadcasts for arrays where
        # the broadcasting is supposed to succeed.
        d4 = broadcast_dicts(d1, d2, axis=0, debug_memory=True)
        assert d4['array'].shape == (100, 100, 3, 2)
        assert d4['array2'].shape == d4['array2'].shape


def test_concatenate_dicts(spectrogram_with_samples, response_with_samples):
    n_samples, n_times, n_out = response_with_samples.shape
    n_spectral = spectrogram_with_samples.shape[-1]
    d1 = {'x': response_with_samples, 'y': spectrogram_with_samples}
    d2 = d1.copy()

    # Only specified axis should change size
    d3 = concatenate_dicts(d1, d2, axis=0)
    assert d3['x'].shape == (n_samples*2, n_times, n_out)
    assert d3['y'].shape == (n_samples*2, n_times, n_spectral)
    d4 = concatenate_dicts(d1, d2, axis=2)
    assert d4['x'].shape == (n_samples, n_times, 2*n_out)
    assert d4['y'].shape == (n_samples, n_times, 2*n_spectral)

    # Here a new axis should be added, other axes should stay the same.
    d5 = concatenate_dicts(d1, d2, axis=-1, newaxis=True)
    assert d5['x'].ndim == response_with_samples.ndim + 1
    assert d5['x'].shape[:-1] == response_with_samples.shape
    assert d5['y'].ndim == spectrogram_with_samples.ndim + 1
    assert d5['y'].shape[:-1] == spectrogram_with_samples.shape


def test_apply_dict(spectrogram, response):
    d1 = {'input': spectrogram, 'target': response}
    # Add 1 to all arrays
    f1 = lambda a: a + 1
    d2 = apply_to_dict(f1, d1)
    assert np.allclose(d2['input'], spectrogram + 1)
    assert np.allclose(d2['target'], response + 1)

    # Should pass shape (-1,) as arg, flatten all arrays
    d3 = apply_to_dict(np.reshape, d1, (-1,))
    assert d3['input'].shape[0] == spectrogram.size
    assert d3['target'].shape[0] == response.size

    # Pass fill_value as kwarg, replace with all 5's
    d4 = apply_to_dict(np.full_like, d1, fill_value=5)
    assert np.allclose(d4['input'], np.full_like(spectrogram, fill_value=5))
    assert np.allclose(d4['target'], np.full_like(response, fill_value=5))
