import pytest
import numpy as np

from nems.layers.acnet_block import ResAdd


def test_constructors():
    with pytest.raises(TypeError):
        ResAdd()  # no shape
    ra = ResAdd(shape=(8, 10), res_scale=0.1, input=['main', 'skip'])
    assert ra.res_scale == 0.1
    assert ra.input == ['main', 'skip']


def test_from_keyword():
    ra = ResAdd.from_keyword('resadd.80x100.0d1.b1_main.b1_in')
    assert ra.shape == (80, 100)
    assert ra.res_scale == 0.1
    assert ra.input == ['b1_main', 'b1_in']

    ra2 = ResAdd.from_keyword('resadd.8x8.1.m.s')
    assert ra2.res_scale == 1.0


class TestEvaluate:

    def test_shortcut_starts_at_zero(self):
        # At construction, shortcut is all zeros and gamma is 1, so the skip
        # contributes nothing and output == res_scale * main exactly.
        T, cin, cout = 50, 8, 10
        rng = np.random.RandomState(0)
        main = rng.rand(T, cout)
        skip = rng.rand(T, cin)

        ra = ResAdd(shape=(cin, cout), res_scale=0.1, input=['main', 'skip'])
        out = ra.evaluate(main, skip)
        assert np.allclose(out, 0.1 * main)

    def test_skip_projection_applied_after_fitting_shortcut(self):
        T, cin, cout = 50, 4, 4
        rng = np.random.RandomState(1)
        main = rng.rand(T, cout)
        skip = rng.rand(T, cin)
        shortcut = rng.rand(cin, cout)

        ra = ResAdd(shape=(cin, cout), res_scale=1.0, input=['main', 'skip'])
        ra.parameters['shortcut'].update(shortcut)
        out = ra.evaluate(main, skip)
        expected = main + skip @ shortcut
        assert np.allclose(out, expected)

    def test_gamma_scales_main_path(self):
        T, cin, cout = 20, 4, 4
        rng = np.random.RandomState(2)
        main = rng.rand(T, cout)
        skip = rng.rand(T, cin)

        ra = ResAdd(shape=(cin, cout), res_scale=0.5, input=['main', 'skip'])
        ra.parameters['gamma'].update(np.array([2.0]))
        out = ra.evaluate(main, skip)
        # shortcut still zero, so out == gamma * res_scale * main
        assert np.allclose(out, 2.0 * 0.5 * main)

    def test_shape_mismatch_channels(self):
        T, cin, cout = 20, 4, 6
        main = np.random.rand(T, cout)
        skip = np.random.rand(T, cin)
        ra = ResAdd(shape=(cin, cout), input=['main', 'skip'])
        out = ra.evaluate(main, skip)
        assert out.shape == (T, cout)
