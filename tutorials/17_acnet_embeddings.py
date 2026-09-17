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
import matplotlib
matplotlib.use('Agg')  # headless-safe; drop this if you want an interactive window
import matplotlib.pyplot as plt

from nems.models.ACNet import ACNet, load_acnet_v1_weights
from nems.preprocessing.spectrogram import load_wav
from nems.preprocessing.spectrogram.filters import centre_freqs

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


########################################################
# 4. Plot the wav, gtg, and ACNet manifold -- mirroring ACNet_v1's own
#    demo_acnet_embeddings.py (gammatonegram + embeddings heatmaps, shared
#    time axis, 300 dpi), with the raw waveform added as a third panel.
#    Filename-mode input only (`wav`/`gtg`/`embeddings_from_path` above) --
#    no need to replot for the other two input kinds, they're numerically
#    identical (confirmed above).
########################################################
dur_ms = 1e3 * len(wav) / fs_stim

# gtg's channel axis runs low-to-high frequency (nems.preprocessing.
# spectrogram.gammatone.gtgram_xe flips the ERB filterbank to this order) --
# compute matching center frequencies for the y-axis ticks.
cf_khz = centre_freqs(model.fs_gtg, model.num_cfs, model.f_min, model.f_max)[::-1] / 1e3
cf_tick_khz = np.array([0.2, 2, 20])
cf_tick_pos = [int(np.argmin(np.abs(cf_khz - t))) + 0.5 for t in cf_tick_khz]

fig, ax = plt.subplots(3, 1, figsize=(7, 6.6), sharex=True)

t_wav_ms = 1e3 * np.arange(len(wav)) / fs_stim
ax[0].plot(t_wav_ms, wav, linewidth=0.5)
ax[0].set(ylabel='amplitude', title='Waveform')

ax[1].imshow(gtg.T, origin='lower', aspect='auto', extent=(0, dur_ms, 0, gtg.shape[1]))
ax[1].set_yticks(cf_tick_pos)
ax[1].set_yticklabels([f'{t:g}' for t in cf_tick_khz])
ax[1].set(ylabel='CF (kHz)', title='Gammatonegram (input to ACNet, sqrt-domain)')

ax[2].imshow(embeddings_from_path.T, origin='lower', aspect='auto',
             extent=(0, dur_ms, 0, embeddings_from_path.shape[1]))
ax[2].set(ylabel='manifold dimension', xlabel='time (ms)', title='ACNet embeddings')

fig.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), '17_acnet_embeddings.png')
fig.savefig(out_png, dpi=300)
print(f"\n4. Saved wav/gtg/embeddings figure to {out_png}")
