"""Tools for optimizing NEMS Models with TensorFlow."""

import tensorflow as tf

# [AGENT EDIT START | agent: claude | user: svd | reason: NVIDIA Ampere/Ada GPUs run float32 matmul/conv ops (e.g. WeightChannels tensordot, FIR conv1d) through TensorFloat-32 by default once they're large enough to hit tensor cores, truncating the mantissa from 23 bits to ~10 (~1e-3 relative precision). That silently diverged from numpy's true-float32 CPU evaluate() path after every TF fit, showing up as the "Backend (tf) prediction differs from numpy" warning in Model.fit() even though parameters were copied back correctly. Confirmed by isolating a single wc.90x1x100 layer: max diff dropped from 3.65e-4 to 3.28e-7 with TF32 disabled. | date: 2026-09-25]
tf.config.experimental.enable_tensor_float_32_execution(False)
# [AGENT EDIT END]

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


from .backend import TensorFlowBackend
from .layer_tools import NemsKerasLayer, Bounds
