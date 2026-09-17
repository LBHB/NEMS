"""Get ACNet manifold embeddings from a wav file, a waveform, or a gtg.

Demonstrates `nems.models.ACNet` (see project memory `project_acnet_in_nems`
for how this port came to be) loaded with the real released weights, then
`model.get_embeddings(...)` on all three input kinds it accepts. You never
pass a `compress` argument to `get_embeddings` -- the model's own first
layer (built with `compress='log10x'`, the released checkpoint's actual
training config) is the single place compression is specified, and it's
applied internally exactly once, regardless of which input kind you use.
"""
import os

import numpy as np

from nems.models.ACNet import ACNet, load_acnet_v1_weights
from nems.preprocessing.spectrogram import load_wav

# Paths to the released ACNet_v1 package's own artifacts (a separate repo on
# the same shared filesystem; adjust if you're running this elsewhere).
ACNET_V1_DIR = '/auto/users/satya/code/projects_getting_started/ACNet_v1'
WEIGHTS_PATH = os.path.join(ACNET_V1_DIR, 'weights', 'acnet_v1_weights_nems.npz')
EXAMPLE_WAV = os.path.join(ACNET_V1_DIR, 'examples', 'example_esc50_clip.wav')


########################################################
# Build the model and load the real released weights.
#
# A freshly-constructed ACNet() has random weights -- fine for testing
# shapes, useless for actual embeddings. load_acnet_v1_weights populates an
# ACNet() built with matching hyperparameters (its defaults already match
# the release) from a plain-numpy export of the real checkpoint. That npz is
# produced once, outside NEMS (NEMS itself never imports torch):
#   conda activate ptn  # or any env with a working torch -- NOT acnet_v1,
#                        # see project_acnet_in_nems memory for why
#   python ACNet_v1/Claude/claude_debug/export_acnet_v1_weights.py
########################################################
model = ACNet()  # compress='log10x' by default -- matches the checkpoint
model = load_acnet_v1_weights(model, WEIGHTS_PATH)
print(f"Built ACNet: compress={model.layers[0].mode!r}, "
      f"embeddings dim={model.layers[-4].shape[0]}")  # hidden_dim[-1]


########################################################
# 1. Wav file path -> embeddings. Nothing else to specify.
########################################################
embeddings_from_path = model.get_embeddings(EXAMPLE_WAV)
print(f"\n1. From wav path: embeddings shape {embeddings_from_path.shape}")


########################################################
# 2. Already-loaded waveform + fs -> embeddings. Same result as (1) -- the
#    wav-path case just does `load_wav` for you first.
########################################################
wav, fs_stim = load_wav(EXAMPLE_WAV)
embeddings_from_wav = model.get_embeddings(wav, fs=fs_stim)
print(f"2. From waveform + fs: embeddings shape {embeddings_from_wav.shape}, "
      f"matches (1): {np.array_equal(embeddings_from_wav, embeddings_from_path)}")


########################################################
# 3. A gtg you already have (no `fs`) -> embeddings. get_embeddings prints
#    what it's assuming, since it can't verify it from the array alone: raw
#    (uncompressed) gammatone magnitude, compressed internally to match this
#    model's own compress mode.
########################################################
gtg = model._to_gtg(wav, fs_stim)  # exactly what step 2 computed internally
embeddings_from_gtg = model.get_embeddings(gtg)
print(f"3. From precomputed gtg: embeddings shape {embeddings_from_gtg.shape}, "
      f"matches (1): {np.array_equal(embeddings_from_gtg, embeddings_from_path)}")
