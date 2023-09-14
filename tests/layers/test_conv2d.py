import pytest
import numpy as np
import tensorflow as tf

from nems.layers.conv2d import Conv2d


def test_constructor():
    with pytest.raises(TypeError):
        # No shape
        conv2d = Conv2d()
    with pytest.raises(TypeError):
        # Shape has too few dimensions
        conv2d = Conv2d(shape=(2,))
    # But this should raise no errors.
    conv2d = Conv2d(shape=(1,2,3))


class TestEvaluate:

    def test_full_rank(self):
        input_shape = (4, 28, 28, 3)
        x = tf.random.normal(input_shape)

        # 50 ms filter, 18 spectral channels (full-rank STRF)
        conv = Conv2d(shape=(4, 4, 1))
        out = conv.evaluate(x)
        assert out.shape == (4, 24, 24, 3)
