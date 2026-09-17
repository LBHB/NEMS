"""Get ACNet manifold embeddings from a wav file, or from a precomputed gtg.

Demonstrates `nems.models.ACNet` (see tutorial README / project memory
`project_acnet_in_nems` for how this port came to be) loaded with the real
released weights, used two ways:

  1. Raw wav file -> level-norm -> gammatonegram -> embeddings.
  2. A gammatonegram you already have -> embeddings directly (no wav, no
     level-normalization) -- e.g. if some other pipeline already produced it.

The one rule that makes both paths correct, and that this script deliberately
demonstrates getting *wrong* as well as right: **ACNet's internal
compression (its first layer, `PowerCompress`) runs exactly once.** The
released checkpoint was trained with `compress='log10x'`
(`ACNet_v1.acnet_model.DEFAULT_CONFIG`), which is `ACNet()`'s default here
too -- so whatever gtg you feed the model must be *raw, uncompressed
(sqrt-domain) magnitude*, the same convention `nems.preprocessing.
spectrogram.gammagram`/`gtgram` already return. If your gtg came from
somewhere that already applied its own compression (log10x or otherwise),
you must undo that before handing it to the model, or you'll silently
double-compress and get wrong embeddings -- Section 4 below shows exactly
how wrong.
"""
import os

import numpy as np

from nems.models.ACNet import ACNet, load_acnet_v1_weights
from nems.preprocessing.spectrogram import load_wav, nems_audio_preprocess, acnet_gtgram
from nems.preprocessing.spectrogram.gammatone import gammagram

# Paths to the released ACNet_v1 package's own artifacts (a separate repo on
# the same shared filesystem; adjust if you're running this elsewhere).
ACNET_V1_DIR = '/auto/users/satya/code/projects_getting_started/ACNet_v1'
WEIGHTS_PATH = os.path.join(ACNET_V1_DIR, 'weights', 'acnet_v1_weights_nems.npz')
EXAMPLE_WAV = os.path.join(ACNET_V1_DIR, 'examples', 'example_esc50_clip.wav')


########################################################
# 1. Build the model and load the real released weights.
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
print(f"Built ACNet: {len(model.layers)} layers, "
      f"compress={model.layers[0].mode!r}, "
      f"embeddings dim={model.layers[-4].shape[0]}")  # hidden_dim[-1]


########################################################
# 2. Wav file -> embeddings.
#
# acnet_gtgram(..., compress='sqrt') runs level-normalization + the
# gammatone filterbank but leaves compression untouched (PowerCompress's
# 'sqrt' mode is the identity) -- i.e. it hands back exactly the raw
# sqrt-domain magnitude the model's own first layer expects. Do NOT pass
# compress='log10x' here; that would compress twice (see Section 4).
########################################################
wav, fs_stim = load_wav(EXAMPLE_WAV)
gtg_from_wav = acnet_gtgram(
    wav, fs_stim,
    num_cfs=32, f_min=200.0, f_max=20e3, fs_gtg=100.0,  # matches the release
    compress='sqrt',           # <-- uncompressed; the model compresses once
    lbhb_mode=False, overall_db=65, level_mode='exact',  # general-audio defaults
    )
embeddings_from_wav = model.get_embeddings(gtg_from_wav)
print(f"\nSection 2 (wav): gtg shape {gtg_from_wav.shape}, "
      f"embeddings shape {embeddings_from_wav.shape}")


########################################################
# 3. Precomputed gtg -> embeddings, no wav involved.
#
# Same rule: whatever produced your gtg, it must be raw sqrt-domain
# magnitude for ACNet's own compression to be applied correctly exactly
# once. Reproducing Section 2's gtg by calling the level-norm + filterbank
# pieces directly (instead of the acnet_gtgram convenience wrapper) shows
# they're the same array -- acnet_gtgram(compress='sqrt') does nothing more
# than this.
########################################################
sig_normalized, fs0 = nems_audio_preprocess(
    wav, fs_stim, fs_gtg=100.0, f_max=20e3,
    lbhb_mode=False, overall_db=65, level_mode='exact',
    )
gtg_precomputed = gammagram(
    sig_normalized, fs=fs0, window_time=1/100.0, hop_time=1/100.0,
    channels=32, f_min=200.0, f_max=20e3,
    )
embeddings_from_gtg = model.get_embeddings(gtg_precomputed)

assert np.array_equal(gtg_precomputed, gtg_from_wav)
assert np.array_equal(embeddings_from_gtg, embeddings_from_wav)
print(f"\nSection 3 (precomputed gtg): matches Section 2 exactly "
      f"(max|diff|={np.max(np.abs(embeddings_from_gtg - embeddings_from_wav)):.3e})")


########################################################
# 4. What goes wrong if the gtg's compression doesn't match the model's.
#
# Here we deliberately pass an ALREADY log10x-compressed gtg (compress=
# 'log10x' instead of 'sqrt') into a model whose own first layer ALSO
# applies log10x -- i.e. double compression. The resulting "embeddings" are
# silently wrong: no error, no NaN, just a different (incorrect) manifold.
########################################################
gtg_wrong_domain = acnet_gtgram(
    wav, fs_stim, num_cfs=32, f_min=200.0, f_max=20e3, fs_gtg=100.0,
    compress='log10x',  # <-- mismatch: model will compress this AGAIN
    lbhb_mode=False, overall_db=65, level_mode='exact',
    )
embeddings_wrong = model.get_embeddings(gtg_wrong_domain)
diff = np.max(np.abs(embeddings_wrong - embeddings_from_wav))
print(f"\nSection 4 (mismatched compression -- WRONG on purpose): "
      f"max|diff| from correct embeddings = {diff:.3f} "
      f"(correct embeddings range [{embeddings_from_wav.min():.3f}, "
      f"{embeddings_from_wav.max():.3f}])")
print("This is the failure mode 'gtg mode must match acnet mode' refers to: "
      "always generate gtg input with compress='sqrt' (or no compression at "
      "all) and let the model's own PowerCompress layer -- configured to "
      "match the checkpoint it was trained with -- do the compression.")
