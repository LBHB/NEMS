import pytest

import numpy as np
import tensorflow as tf
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

def test_tf_vs_scipy():
    X=np.random.randn(100,3)
    S2=np.zeros((100,2))
    S2[:,0]=0.5
    S2[50:,1]=1
    S3=np.zeros((100,3))
    S3[:,0]=1
    S3[:,1]=0.5
    S3[50:,2]=1
    sg=StateGain(shape=(3,3))
    sg3=StateGain(shape=(3,3))
    g = np.zeros((3,3))
    g[0,:]=1
    g[1,1]=0.5
    g[2,2]=-0.5
    d = np.zeros((3,3))
    d[0,:]=1
    sg['gain']=g
    sg['offset']=d
    sg3['gain']=g
    sg3['offset']=d
    Y2=sg.evaluate(X, state=S2)
    Y3=sg.evaluate(X, state=S3)

    e2=sg.as_tensorflow_layer(input_shape=[X.shape, S2.shape])
    yt2=e2([X,S2]).numpy()
    e3=sg3.as_tensorflow_layer(input_shape=[X.shape, S3.shape])
    yt3=e3([X,S3]).numpy()
    
    assert np.std(Y2-yt2) < 1e-8
    assert np.std(Y2-Y3) < 1e-8
    assert np.std(Y2-yt3) < 1e-8
