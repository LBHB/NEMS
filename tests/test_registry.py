import pytest

from nems import Model
from nems.registry import layer
from nems.layers.base import Layer


class KeywordTester(Layer):
    @layer('tester')
    def from_keyword(keyword):
        options = keyword.split('.')
        kw_head = options[0]
        layer = KeywordTester()
        layer._kw_head = kw_head
        layer._options = options[1:]
        return layer

class MethodNameTester(Layer):
    # `from_keyword` is just a convention
    @layer('different')
    def method_name(keyword):
        return MethodNameTester()

class InitKeywordTester(Layer):

    @layer('init')
    def from_keyword(keyword):
        options = keyword.split('.')
        kw_head = options[0]
        layer = InitKeywordTester()
        layer.kw_head = kw_head
        layer._options = options[1:]
        return layer

class MethodSyntaxTester1(Layer):
    # must return a Layer instance
    @layer('notlayer')
    def from_keyword(keyword):
        return keyword

class MethodSyntaxTester2(Layer):
    # must accept keyword arg
    @layer('noargs')
    def from_keyword():
        return 5


def test_kw():
    model = Model.from_keywords('tester.zero.one.two.three')
    layer = model.layers[0]
    # Should use the correct Layer subclass
    assert isinstance(layer, KeywordTester)
    # Registry should set Layer._name == kw_head
    assert layer.name == 'tester'
    # Full kw string should be passed to `from_keyword`.
    assert layer._options == ['zero', 'one', 'two', 'three']


def test_kw_init_list():
    model = Model(layers=['init.zero.one.two.three.four'])
    layer = model.layers[0]
    # Should use the correct Layer subclass
    assert isinstance(layer, InitKeywordTester)
    # Registry should set Layer._name == kw_head
    assert layer.name == 'init'
    # Full kw string should be passed to `from_keyword`.
    assert layer._options == ['zero', 'one', 'two', 'three', 'four']

def test_kw_init_single():
    model = Model('init.zero.one.two.three.four')
    layer = model.layers[0]
    # Should use the correct Layer subclass
    assert isinstance(layer, InitKeywordTester)
    # Registry should set Layer._name == kw_head
    assert layer.name == 'init'
    # Full kw string should be passed to `from_keyword`.
    assert layer._options == ['zero', 'one', 'two', 'three', 'four']

def test_kw_string():
    # Should result in three layers
    model = Model.from_keywords('tester.one-tester.five-tester.another')
    assert len(model.layers) == 3
    assert model.layers[0].name == 'tester'
    # Duplicate names get incremented
    assert model.layers[1].name != 'tester'


def test_name():
    model = Model.from_keywords('different')
    layer = model.layers[0]
    assert isinstance(layer, MethodNameTester)
    assert layer.name == 'different'


def test_syntax():
    with pytest.raises(TypeError):
        model = Model.from_keywords('notlayer')
    with pytest.raises(TypeError):
        model = Model.from_keywords('noargs')
