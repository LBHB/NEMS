import pytest
import numpy as np

from nems.preprocessing.normalization import minmax, undo_minmax


class TestMinmax:

    def test_min_equal_max(self):
        # pathological case, just don't want this to throw an error
        x = np.zeros(shape=(10,))
        y, _, _ = minmax(x)

    def test_already_normalized(self):
        x = np.zeros(shape=(10,))
        x[-1] = 1
        y, _, _ = minmax(x)
        assert np.all(x == y)  # no change, already had min 0 max 1

    def test_warn_nan(self):
        x = np.array([1, 2, 3, np.nan, 5])
        with pytest.warns(RuntimeWarning):
            y, _min, _max = minmax(x, warn_on_nan=True)
        # All should match except nan
        match = (y == np.array([0, 1/4, 2/4, np.nan, 1]))
        assert np.sum(match) == 4
        assert match[3] == False
        # Min, max should not be changed by presence of nan
        assert _min[0] == 1
        assert _max[0] == 4

    def test_dtype_error(self):
        x = np.arange(10)
        with pytest.raises(np.core._exceptions.UFuncTypeError):
            # Error due to integer dtype
            y, _, _ = minmax(x)

        x = x.astype(np.float16)
        # But any float dtype should be fine
        y, _, _ = minmax(x)

    def test_data(self, spectrogram, response):
        s, s_min, s_max = minmax(spectrogram)
        assert s.shape == spectrogram.shape
        assert s_min.shape == (1, spectrogram.shape[-1])
        assert not np.shares_memory(s, spectrogram)
        
        r, r_min, r_max = minmax(response)
        assert r.shape == response.shape
        assert r_min.shape == (1, response.shape[-1])
        assert not np.shares_memory(r, response)

    def test_inplace(self):
        x = np.random.rand(10, 2)
        y, _, _ = minmax(x, inplace=True)
        assert y.shape == x.shape
        assert np.shares_memory(x,y)

    def test_undo(self):
        x = np.random.rand(10,2)
        y, _min, _max = minmax(x, floor=0)
        z = undo_minmax(y, _min, _max)  # undo normalization
        assert np.allclose(z, x)

        # Should also work for other shapes
        x2 = np.random.rand(10,)
        y2, _min2, _max2 = minmax(x2, floor=0)
        z2 = undo_minmax(y2, _min2, _max2)
        assert np.allclose(z2, x2)

        x3 = np.random.rand(5, 10, 3, 8)
        y3, _min3, _max3 = minmax(x3, floor=0)
        z3 = undo_minmax(y3, _min3, _max3)
        assert np.allclose(z3, x3)

    def test_floor(self):
        x = np.array([0, 0.99, 100])               # ->  [0, 0.009, 1]
        floor = 0.01                               # ->  [0, 0,     1]
        y, _min, _max = minmax(x, floor=floor)     # 1 value is clipped
        z = undo_minmax(y, _min, _max)             # ->  [0, 0,   100]
        assert np.allclose(x, z, atol=floor*_max)  # 0.99 <= floor*_max=1
