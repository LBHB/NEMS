import pytest

from nems.layers.state import StateGain


def test_constructor():
    with pytest.raises(TypeError):
        # Shape has too few dimensions
        sg = StateGain(shape=(1,))
    sg = StateGain(shape=(2,2))


def test_evaluate(spectrogram, state):
    # Don't think anyone would change the output shape to 1, but just incase.
    assert spectrogram.shape[-1] > 1
    fake_weighted = spectrogram[..., :2]  # pretend 2 weighted channels.
    n_states = state.shape[-1]
    sg = StateGain(shape=(n_states, 2))
    out = sg.evaluate(fake_weighted, state)
    assert out.shape == fake_weighted.shape


def test_broadcast(spectrogram, state):
    assert spectrogram.shape[-1] > 1
    fake_weighted = spectrogram[..., :3]
    n_states = state.shape[-1]
    sg = StateGain(shape=(n_states, 1))
    out = sg.evaluate(fake_weighted, state)
    # Second dim of SG is 1, should broadcast to number of input channels.
    assert out.shape == fake_weighted.shape
