import pytest
import numpy as np

from nems import Model
from nems.layers.base import Parameter, Phi, Layer


@pytest.fixture
def p1():
    return Parameter('one', shape=(2,3), initial_value=5)

@pytest.fixture
def p2():
    return Parameter('two', shape=(2,3), initial_value=5)

@pytest.fixture
def p3():
    return Parameter('one', shape=(2,3), initial_value=7)

@pytest.fixture
def p4():
    return Parameter('two', shape=(4,))

@pytest.fixture
def p5():
    return Parameter('one', shape=(2,3), initial_value=5)

@pytest.fixture
def phi1(p1, p2):
    return Phi(p1, p2)

@pytest.fixture
def phi2(p3, p2):
    return Phi(p3, p2)

@pytest.fixture
def phi3(p1, p2, p3):
    return Phi(p1, p2, p3)

@pytest.fixture
def phi4(p5, p2):
    return Phi(p5, p2)

@pytest.fixture
def layer1(phi1):
    return Layer(name='one', parameters=phi1)

@pytest.fixture
def layer2(phi4):
    return Layer(name='two', parameters=phi4)

@pytest.fixture
def layer3(phi3):
    return Layer(name='three', parameters=phi3)

@pytest.fixture
def layer4(phi1):
    class Tester(Layer):
        pass
    return Tester(parameters=phi1)

@pytest.fixture
def model1(layer1, layer3):
    return Model(layers=[layer1, layer3])

@pytest.fixture
def model2(layer2, layer3):
    return Model(layers=[layer2, layer3])

@pytest.fixture
def model3(layer1, layer2):
    return Model(layers=[layer1, layer2])


def test_parameter(p1, p2, p3, p4):
    assert p1 != p2  # different names
    assert p1 == p3  # different values shouldn't matter
    assert p2 != p4  # different shapes

    p1.freeze()
    assert p1 != p3  # frozen != unfrozen
    p1.unfreeze()
    assert p1 == p3  # both unfrozen again
    p1.make_permanent()
    assert p1 != p3  # permanent != non-permanent


def test_phi(phi1, phi2, phi3, phi4):
    assert phi1 != phi2  # parameters all equal, but not values
    assert phi1 == phi4  # everything equal
    assert phi1 != phi3  # different parameters

    phi1.freeze_parameters('one')
    assert phi1 != phi4  # parameters now unequal
    phi1.unfreeze_parameters()
    assert phi1 == phi4  # should be equal again
    phi1._vector_mask = np.logical_not(phi2._vector_mask)
    assert phi1 != phi4  # masks don't match


def test_layer(layer1, layer2, layer3, layer4):
    assert layer1 == layer2  # same class, same parameters
    assert layer1 != layer3  # different parameters
    assert layer1 != layer4  # different class


def test_model(model1, model2, model3):
    assert model1 == model2  # layers are all equal
    assert model1 != model3  # layers not equal
    model4 = model1.copy()
    assert model1 == model4  # copies should be equal to each other
    model5 = model4.sample_from_priors()
    assert model1 != model5  # if parameters change, layers shouldn't be equal

def test_keywords():
    keyword_string = 'wc.18x1-fir.15x1-dexp.1'
    first_model = Model.from_keywords(keyword_string)
    second_model = Model.from_keywords(keyword_string)
    assert first_model == second_model  # same keywords, same layers
