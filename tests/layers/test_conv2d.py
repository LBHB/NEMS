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

    def test_2d(self):
        input_shape = (1000, 28)
        x = tf.random.normal(input_shape)

        conv = Conv2d(shape=(4, 4, 4))
        out = conv.evaluate(x)
        assert out.shape == (1000, 28)

    def test_3d(self):
        input_shape = (2, 1000, 28)
        x = tf.random.normal(input_shape)

        conv = Conv2d(shape=(4, 4, 4))
        out = conv.evaluate(x)
        assert out.shape == (1000, 28)

    def test_4d(self):
        input_shape = (2, 1000, 28, 4)
        x = tf.random.normal(input_shape)

        conv = Conv2d(shape=(4, 4, 4))
        out = conv.evaluate(x)
        assert out.shape == (1000, 28)