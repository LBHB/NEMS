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

    def test_filter_single(self):
        input_shape = (3, 5)
        input = tf.random.normal(input_shape)

        conv = Conv2d(shape=(3, 5, 1))
        output = conv.evaluate(input)
        assert output == input

    def test_filter_multi(self):
        input_shape = (5, 3)
        input = np.ones(input_shape)

        conv = Conv2d(shape=(5, 3, 1), pool_type='SUM')
        conv2 = Conv2d(shape=(5, 3, 1), pool_type='SUM')
        conv.coefficients = np.ones((5,3,1))
        conv2.coefficients = np.ones((5,3,1))
        conv_output = conv.evaluate(input)
        conv2_output = conv_output + conv2.evaluate(conv_output)
        assert conv2_output == input*2
        pass

    def test_tf_vs_scipy(self):
        pass

    """WIP
    
    
    import numpy as np
#import tensorflow as tf

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



def test_2d():
    input_shape = (1000, 28)
    x = np.random.normal(input_shape)

    conv = Conv2d(shape=(4, 4, 4))
    out = conv.evaluate(x)
    assert out.shape == (1000, 28)

def test_3d():
    input_shape = (2, 1000, 28)
    x = tf.random.normal(input_shape)

    conv = Conv2d(shape=(4, 4, 4))
    out = conv.evaluate(x)
    assert out.shape == (1000, 28)

def test_4d():
    input_shape = (2, 1000, 28, 4)
    x = tf.random.normal(input_shape)

    conv = Conv2d(shape=(4, 4, 4))
    out = conv.evaluate(x)
    assert out.shape == (1000, 28)

def test_filter_single():
    input_shape = (3, 5)
    input = tf.random.normal(input_shape)

    conv = Conv2d(shape=(3, 5, 1))
    output = conv.evaluate(input)
    assert output == input

def test_filter_multi():
    input_shape = (5, 3)
    input = np.ones(input_shape)

    conv = Conv2d(shape=(5, 3, 1), pool_type='SUM')
    conv2 = Conv2d(shape=(5, 3, 1), pool_type='SUM')
    conv.coefficients = np.ones((5,3,1))
    conv2.coefficients = np.ones((5,3,1))
    conv_output = conv.evaluate(input)
    conv2_output = conv_output + conv2.evaluate(conv_output)
    assert conv2_output == input*2
    pass

def test_tf_vs_scipy():
    pass



x1 = np.random.randn(1, 1000, 28)
input_shape = x1.shape

conv1 = Conv2d(shape=(4, 4, 4))
out1 = conv1.evaluate(x1)
print(out1.shape)


import matplotlib.pyplot as plt

x = np.zeros((1,40,18))
x[0,20,8]=1
conv = Conv2d(shape=(5, 3, 2), pool_type='STACK')
c = conv.get_parameter_values()[0]
c[:,:,0]=0
c[2,0,0]=-0.5
conv['coefficients']=c

y=conv.evaluate(x)
e=conv.as_tensorflow_layer(x.shape)
yt=e(x).numpy()

imargs = {'interpolation': 'none', 'aspect': 'auto', 'origin': 'lower', 'cmap': 'gray'}
f,ax = plt.subplots(1,3)
ax[0].imshow(x[0], **imargs)
ax[1].imshow(y[0], **imargs)
ax[2].imshow(yt[0], **imargs)


"""