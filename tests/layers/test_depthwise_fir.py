import pytest
import numpy as np

from nems.layers.depthwise_fir import DepthwiseFIR


def test_constructors():
    with pytest.raises(TypeError):
        DepthwiseFIR()  # no shape
    dfir = DepthwiseFIR(shape=(7, 4))
    assert dfir.shape == (7, 4)

    with pytest.raises(TypeError):
        # too many dims
        DepthwiseFIR(shape=(7, 4, 2))


def test_from_keyword():
    dfir = DepthwiseFIR.from_keyword('dfir.7x80')
    assert dfir.shape == (7, 80)


class TestEvaluate:

    def test_shape_preserved(self, spectrogram):
        n = spectrogram.shape[-1]
        dfir = DepthwiseFIR(shape=(5, n))
        out = dfir.evaluate(spectrogram)
        assert out.shape == spectrogram.shape

    def test_causal_no_future_leakage(self):
        # An impulse at t=10 should only affect outputs at t>=10 (causal:
        # output[t] depends on input[t-K+1 .. t], never input[t+1:]).
        T, N, K = 30, 3, 5
        x = np.zeros((T, N))
        x[10, :] = 1.0
        dfir = DepthwiseFIR(shape=(K, N))
        dfir.parameters['coefficients'].update(np.ones((K, N)))
        out = dfir.evaluate(x)
        assert np.allclose(out[:10], 0)
        assert not np.allclose(out[10:10 + K], 0)
        assert np.allclose(out[10 + K:], 0)

    def test_no_cross_channel_mixing(self):
        # Unlike FiniteImpulseResponse, output channel c must depend only on
        # input channel c -- zeroing one input channel should not change any
        # other output channel.
        T, N, K = 20, 4, 3
        rng = np.random.RandomState(0)
        x = rng.rand(T, N)
        dfir = DepthwiseFIR(shape=(K, N))
        dfir.parameters['coefficients'].update(rng.rand(K, N))

        out_full = dfir.evaluate(x)

        x_zeroed = x.copy()
        x_zeroed[:, 0] = 0
        out_zeroed = dfir.evaluate(x_zeroed)

        # Channel 0 should differ, all other channels must be untouched.
        assert not np.allclose(out_full[:, 0], out_zeroed[:, 0])
        assert np.allclose(out_full[:, 1:], out_zeroed[:, 1:])

    def test_bias_added_per_channel(self):
        # Matches nn.Conv1d's default bias=True -- Conv1DRowLayer never
        # passes bias=False.
        T, N, K = 10, 3, 4
        rng = np.random.RandomState(3)
        x = rng.rand(T, N)
        dfir = DepthwiseFIR(shape=(K, N))
        dfir.parameters['coefficients'].update(rng.rand(K, N))
        out_no_bias = dfir.evaluate(x)

        bias = np.array([1.0, -2.0, 0.5])
        dfir.parameters['bias'].update(bias)
        out_with_bias = dfir.evaluate(x)
        assert np.allclose(out_with_bias, out_no_bias + bias)

    def test_matches_hand_rolled_reference(self):
        T, N, K = 25, 3, 4
        rng = np.random.RandomState(1)
        x = rng.rand(T, N)
        coef = rng.rand(K, N)

        dfir = DepthwiseFIR(shape=(K, N))
        dfir.parameters['coefficients'].update(coef)
        out = dfir.evaluate(x)

        expected = np.zeros((T, N))
        for c in range(N):
            padded = np.concatenate([np.zeros(K - 1), x[:, c]])
            for t in range(T):
                # output[t] = sum_lag coef[lag, c] * input[t-lag, c]
                window = padded[t:t + K][::-1]  # most-recent-first
                expected[t, c] = np.dot(window, coef[:, c])

        assert np.allclose(out, expected)
