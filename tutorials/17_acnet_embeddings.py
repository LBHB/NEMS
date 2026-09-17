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
# matplotlib.use('Agg')  # headless-safe; drop this if you want an interactive window
import matplotlib.pyplot as plt

from nems.models.ACNet import load_acnet
from nems.preprocessing.spectrogram import load_wav
from nems.preprocessing.spectrogram.filters import centre_freqs

# Local copy of /auto/data/sounds/vocalizations/v2/ferretb1001R.wav.
EXAMPLE_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'ferretb1001R.wav')

# If True (default), the ACNet-embeddings panel's manifold dimensions are
# reordered by descending activity (sum of squared response) so the most
# active dimensions plot together, near the bottom -- purely a display
# convenience; the dimensions themselves, and everything returned by
# get_embeddings, are unaffected.
SORT_ACNET_DIM = True

# If True, the figure in section 4 is written to disk (see out_png below).
# False just builds/shows it in memory -- handy while iterating on the plot.
SAVE_FIGURE = False


########################################################
# Build the model and load the real released weights.
#
# load_acnet(version='v1') builds an ACNet and loads the released, trained
# weights into it -- a freshly-constructed ACNet() alone has random weights,
# fine for testing shapes, useless for actual embeddings. Where that npz
# lives is load_acnet's own concern, not something you need to know; the
# npz itself is produced once, outside NEMS (NEMS itself never imports
# torch), by:
#   conda activate ptn  # or any env with a working torch -- NOT acnet_v1,
#                        # see project_acnet_in_nems memory for why
#   python ACNet_v1/data/export_acnet_v1_weights.py
# version='v2' (sqrt compression) isn't trained/released yet -- raises
# NotImplementedError.
########################################################
model = load_acnet(version='v1')
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
# spectrogram.gammatone.gtgram_xe flips the ERB filterbank to this order --
# confirmed empirically with pure tones, not just by reading the code: a
# 300/1000/5000/15000 Hz tone peaks at channel 2/9/21/30 of 32). Tick
# positions are evenly spaced across the channels actually present, and
# each is labeled with its own true center frequency -- never a rounded
# target value that may not be the nearest channel's actual CF.
cf_khz = centre_freqs(model.fs_gtg, model.num_cfs, model.f_min, model.f_max)[::-1] / 1e3
n_ticks = min(5, len(cf_khz))
tick_idx = np.linspace(0, len(cf_khz) - 1, n_ticks).round().astype(int)
cf_tick_pos = tick_idx + 0.5
cf_tick_labels = [f'{cf_khz[i]:.2g}' for i in tick_idx]

if SORT_ACNET_DIM:
    dim_order = np.argsort(-(embeddings_from_path**2).sum(axis=0))
    embeddings_plotted = embeddings_from_path[:, dim_order]
    embeddings_title = 'ACNet embeddings (sorted by activity)'
else:
    embeddings_plotted = embeddings_from_path
    embeddings_title = 'ACNet embeddings (unsorted)'

plt.rcParams.update({'font.size': plt.rcParams['font.size'] + 2})

fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True,
                       gridspec_kw={'height_ratios': [1, 2.5, 2.5]})

t_wav_ms = 1e3 * np.arange(len(wav)) / fs_stim
ax[0].plot(t_wav_ms, wav, linewidth=0.5, color='k')
ax[0].set(ylabel='amplitude', title='Waveform')
ax[0].set_xmargin(0)
for spine in ax[0].spines.values():
    spine.set_visible(False)
ax[0].tick_params(axis='both', length=0)
ax[0].set_yticks([])

ax[1].imshow(gtg.T, origin='lower', aspect='auto', extent=(0, dur_ms, 0, gtg.shape[1]),
             cmap='gray_r')
ax[1].set_yticks(cf_tick_pos)
ax[1].set_yticklabels(cf_tick_labels)
ax[1].set(ylabel='CF (kHz)', title='Gammatonegram (input to ACNet, sqrt-domain)')

ax[2].imshow(embeddings_plotted.T, origin='lower', aspect='auto',
             extent=(0, dur_ms, 0, embeddings_plotted.shape[1]), cmap='gray_r')
ax[2].set(ylabel='manifold dimension', xlabel='time (ms)', title=embeddings_title)

fig.tight_layout()
if SAVE_FIGURE:
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), '17_acnet_embeddings.png')
    fig.savefig(out_png, dpi=300)
    print(f"\n4. Saved wav/gtg/embeddings figure to {out_png}")
else:
    print("\n4. Built wav/gtg/embeddings figure (SAVE_FIGURE=False, not written to disk)")
