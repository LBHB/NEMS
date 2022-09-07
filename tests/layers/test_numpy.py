import numpy as np

from nems import Model
from nems.layers import NumPy


# Picked a few numpy functions to test the general behavior, but this is not
# an exhaustive test of all possible numpy operations.

def test_numpy_ops():
    x = np.arange(12).reshape(3, 4)
    y = np.split(x, 3, axis=0)
    split = NumPy('split', 3, axis=0)
    z = split.evaluate(x)
    assert np.allclose(y, z)

    # This should be the same as the first split
    split2 = NumPy.from_keyword('np.split.3.axis0')
    z2 = split2.evaluate(x)
    assert np.allclose(y, z2)

    y3 = np.vstack(y)
    vstack = NumPy('vstack')
    z3 = vstack.evaluate(z)
    assert np.allclose(y3, z3)

    a = np.arange(10)
    b = np.sum(a)
    model = Model.from_keywords('np.sum')
    c = model.layers['np.sum'].evaluate(a)
    assert np.allclose(b, c)

    # Should be no bounds
    assert model.get_bounds_vector() == []
    # No parameters
    assert model.get_parameter_values()['np.sum'] == {}

    d = np.ones(shape=(4,3))
    e = np.ones(shape=(3,5))*2
    f = np.matmul(d,e)
    model2 = Model.from_keywords('np.matmul')
    g = model2.layers['np.matmul'].evaluate(d, e)
    assert np.allclose(f, g)

    h = np.transpose(g, axes=[1,0])
    model3 = Model.from_keywords('np.transpose.axes[1,0]')
    i = model3.layers['np.transpose'].evaluate(g)
    assert np.allclose(h, i)
