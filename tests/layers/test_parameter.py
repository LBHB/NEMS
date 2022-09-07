import pytest
import numpy as np

from nems.layers.base import Parameter, Phi


def test_constructor():
    # Name should be only required argument, shape defaults to ()
    p = Parameter('test')
    assert p.shape == ()

    p = Parameter('test', shape=(2,3))
    assert p.shape == (2,3)
    assert p.prior.shape == (2,3)
    

def test_epsilon_and_bounds():
    test1 = Parameter('test1', bounds=(0, 1))
    assert test1.bounds == (0, 1)
    # 0 should be replaced by machine epsilon for float32,
    # `np.finfo(np.float32).eps`
    test2 = Parameter('test2', bounds=(0, 1), zero_to_epsilon=True)
    assert test2.bounds[0] > 0

    with pytest.raises(ValueError):
        # Value error b/c 0 should be below minimum bound.
        test2.update(0)

    with pytest.raises(ValueError):
        # Value error b/c 2 should be above maximum bound.
        test1.update(2)

    with pytest.raises(AttributeError):
        # Attribute error b/c no Phi is assigned, so Parameter
        # doesn't actually have any values
        test2.update(0.5)


def test_priors():
    test = Parameter('test', shape=(2,3))
    s = test.sample()
    assert s.shape == test.shape

    m = test.mean()
    assert m.shape == test.shape


def test_bounds_vector():
    p = Parameter('test', shape=(2,3,4))
    b = p.get_bounds_vector()
    assert len(b) == 2*3*4
    assert all([isinstance(t, tuple) for t in b])


# TODO: maybe not worth maintaining the array passthroughs?
#       they don't cover everything, so might just be confusing when
#       .values is required.
def test_parameter_math():
    p = Parameter('mathtest', shape=(3,2), initial_value=1)
    p2 = Parameter('mathtest2', shape=(3,2))
    p3 = Parameter('mathtest3', shape=(2,3))
    phi = Phi(p, p2, p3)  # so that Parameter.values works

    assert np.allclose(p + p2, p.values)
    assert np.allclose(p + 1, p.values + 1)
    assert np.allclose(p*2, np.full(shape=(3,2), fill_value=2))
    assert np.allclose(p/3, np.full(shape=(3,2), fill_value=1/3))
    assert np.allclose(p @ p3, np.zeros(shape=(3,3)))
    with pytest.raises(ValueError):
        # shape mismatch
        p @ p2
    assert np.allclose(p % 5, np.ones(shape=p.shape))
    assert np.allclose((p*2)**3, np.full(shape=p.shape, fill_value=8))
