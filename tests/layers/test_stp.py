import pytest
import numpy as np

from nems.layers.stp import ShortTermPlasticity

def test_constructor():
    with pytest.raises(TypeError):
        # No shape
        stp = ShortTermPlasticity()
    # This should raise no errors.
    stp = ShortTermPlasticity(shape=(2,1))
    

"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from nems.layers.stp import ShortTermPlasticity
stp = ShortTermPlasticity(shape=(3,1), fs=100)
input = np.random.randn(100,3)
input[:,1]=0
input[10:20,1]=1
input[25:35,1]=1
input[60:90,1]=1
input[:,2]=0
tf_input = tf.constant(input[np.newaxis,...].astype(np.float32))
tf_stp = stp.as_tensorflow_layer()

output=stp.evaluate(input)
tf_output = tf_stp.call(tf_input).numpy().squeeze(axis=0)

f,ax=plt.subplots(3,1)
ax[0].plot(input[:,0])
ax[0].plot(output[:,0])
ax[0].plot(tf_output[:,0])
ax[1].plot(input[:,1])
ax[1].plot(output[:,1])
ax[1].plot(tf_output[:,1])

ax[2].plot(output-tf_output)


np.allclose(output, tf_output)
"""

