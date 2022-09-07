import numpy as np

from nems import Model
from nems.layers.base import Parameter, Phi, Layer
from nems.distributions import Normal
from nems.tools.json import nems_to_json, nems_from_json


def test_save_load():
    shape1 = (2,3)
    shape2 = (1,)

    d1 = Normal(
        mean=np.full(shape=shape1, fill_value=0.2),
        sd=np.full(shape=shape1, fill_value=0.9)
        )
    encoded_d1 = nems_to_json(d1)
    decoded_d1 = nems_from_json(encoded_d1)
    assert d1 == decoded_d1

    p1 = Parameter('p1', shape=shape1, bounds=(-1, 1))
    encoded_p1 = nems_to_json(p1)
    decoded_p1 = nems_from_json(encoded_p1)
    assert p1 == decoded_p1

    p2 = Parameter('p2', shape=shape2)
    p2.freeze()
    phi = Phi(p1, p2)
    encoded_phi = nems_to_json(phi)
    decoded_phi = nems_from_json(encoded_phi)
    assert phi == decoded_phi

    layer = Layer(parameters=phi, input=None, output=['test1', 'test2'])
    layer.unfreeze_parameters()
    encoded_layer = nems_to_json(layer)
    decoded_layer = nems_from_json(encoded_layer)
    assert layer == decoded_layer

    model = Model.from_keywords('wc.18x4', 'fir.4x15')
    encoded_model = nems_to_json(model)
    decoded_model = nems_from_json(encoded_model)
    assert decoded_model == model
