import pytest
import numpy as np

from nems import Model
from nems.models.dataset import DataSet
from nems.layers import Layer


output_tests = [
    None,
    'str',
    ['list1'],
    ['list2', 'list3'],
]

class PassThrough(Layer):
    def evaluate(self, input):
        out = np.array([[f'from {self.name}']])
        if isinstance(input, (list)):
            out = [out]*len(input)
        return out

class Duplicate(Layer):
    def evaluate(self, input):
        return [np.array([[f'from {self.name}']])]*2

class JustOne(Layer):
    def evaluate(self, input_list):
        assert isinstance(input_list, list)
        return np.array([[f'from {self.name}']])


def test_inputs():
    dm = PassThrough(input=None).data_map
    assert dm.args == [None]
    assert dm.kwargs == {}
    assert dm.out == []

    # string gets list-wrapped
    assert PassThrough(input='str').data_map.args == ['str']
    assert PassThrough(input=['list1']).data_map.args == ['list1']
    assert PassThrough(input=[None, 'list2']).data_map.args == [None, 'list2']
    assert PassThrough(input=[None, 'list2']).data_map.kwargs == {}

    # nested lists should be preserved
    assert PassThrough(input=['list1', ['list2', 'list3']]).data_map.args \
        == ['list1', ['list2', 'list3']]

    # dicts go to kwargs, not args
    assert PassThrough(input={'dict1': 'x', 'dict2': 'y'}).data_map.kwargs \
        == {'dict1': 'x', 'dict2': 'y'}
    assert PassThrough(input={'dict1': 'x', 'dict2': 'y'}).data_map.args == []

    # input types can all be combined
    input = ['str', ['list1', 'list2'], {'arg1': None}]
    dm = PassThrough(input=input).data_map
    assert dm.args == ['str', ['list1', 'list2']]
    assert dm.kwargs == {'arg1': None}


def test_outputs():
    layer = PassThrough(output=None)
    layer.data_map.map_outputs('output')
    assert layer.data_map.out == [None]


    layer = PassThrough(output='str')
    layer.data_map.map_outputs('output')
    assert layer.data_map.out == ['str']

    layer = PassThrough(output=['list1', 'list2'])
    layer.data_map.map_outputs(['output1', 'output2'])
    assert layer.data_map.out == ['list1', 'list2']

    with pytest.raises(ValueError):
        # Two outputs specified, but only one given to map
        layer.data_map.map_outputs('output')


def test_evaluate():
    input = {'input': np.array([['input']])}

    model = Model()
    # Will get an error in here somewhere if input/output routing is broken.
    model.add_layers(
        # input should be 'input', the Model default
        PassThrough(name='p1', input=None, output='str'), 
        PassThrough(name='p2', input='str', output=['list1']),
        PassThrough(name='p3', input=None, output=None),
        Duplicate(name='d1', input=None, output=['list2', 'list3']),
        # should map to ['str', 'str.1']. need double [[]] b/c we want a list as a
        # single argument, not two list elements as two arguments
        PassThrough(name='p4', input=[['list2', 'list3']], output='str'),
        # should be same as input=None in this case, and output should map to
        # 'output' (the Model default). In this case, only need a single [] b/c
        # the list is already mapped to a single argument (named 'input_list').
        JustOne(name='i1', input={'input_list': ['str', 'str.1']}, output=None)  
    )

    eval_data = model.evaluate(input)
    # First layer input gets overwritten if None
    assert model.layers[0].data_map.args == ['input']
    # Last layer output overwritten if None
    assert model.layers[-1].data_map.out == [DataSet.default_output]
