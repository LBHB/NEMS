"""Get ACNet manifold embeddings from a wav file, a waveform, or a gtg -- and check
that a freshly-computed gtg agrees with the real, BAPHY-cached `stim` signal for the
same site/stimulus.

Demonstrates `nems.models.ACNet` (see project memory `project_acnet_in_nems`
for how this port came to be) loaded with the real released weights, then
`model.get_embeddings(...)` on all three input kinds it accepts. You never
pass a `compress` argument to `get_embeddings` -- the model's own first
layer (built with `compress='log10x'`, the released checkpoint's actual
training config) is the single place compression is specified, and it's
applied internally exactly once, regardless of which input kind you use.

Units note (section 4): a real BAPHY recording's `rec['stim']` (any `gtgram.fs*.ch*`
loadkey) is NOT the same "raw sqrt-domain magnitude" that `model._to_gtg`/`gammagram`
produce fresh from a wav. `nems_lbhb/runclass.py`'s `NAT_stim` -- what actually builds
that cached signal -- computes the standard gtgram (one sqrt: power -> RMS amplitude)
and then applies a SECOND, unconditional `**0.5` on top. So `rec['stim']` is
double-square-rooted (~power^0.25) relative to a fresh single-sqrt `_to_gtg` output;
comparing the two directly looks like a large, real mismatch that is actually just a
units mismatch. Fix: apply one extra `np.sqrt()` to the freshly-computed gtg before
comparing against `stim0` (done below, section 4 only -- section 3's own demonstration
deliberately stays in the model's normal single-sqrt input convention). See
project memory `project_acnet_in_nems` / `reference_acnet_frontend_numerics` for the
full trace (also: `overall_db`/`fixed_amp_scale` here are ACNet's generic defaults
-- 65 dB / 250 -- not LMD052a's real per-site calibration, a second, separate
approximation on top of this one).
"""
import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')  # headless-safe; drop this if you want an interactive window
import matplotlib.pyplot as plt

from nems.models.ACNet import load_acnet
from nems.preprocessing.spectrogram import load_wav
from nems.preprocessing.spectrogram.filters import centre_freqs
from nems_lbhb import xform_helper, db, baphy_experiment


soundpath = '/auto/data/sounds/BigNat/v3/'


batch=343
siteids, cellids = db.get_batch_sites(batch)

modelnames = [
    'gtgram.fs100.ch32-ld-norm.l1-sev_wc.Nx1x90.g-fir.15x1x90-relu.90.s-wc.90x1x100.l2:4-fir.15x1x100-relu.100.s-wc.100x120.l2:4-relu.120.s-wc.120xR.l2:4-dexp.R_lite.tf.init.lr1e3.t3.es20.jk8.rb4-lite.tf.lr1e4.t5e4-dstrf.d30.t47.p25.ss95.nl',
    'gtgram.fs100.ch32-ld-norm.l1-sev_wc.Nx1x120.g-fir.25x1x120-wc.120xR.l2:4-dexp.R_lite.tf.init.lr1e3.t3.es20.jk8.rb4-lite.tf.lr1e4.t5e4',
    'gtgram.fs100.ch32-ld-norm.l1-sev_wc.Nx1x90.g-fir.15x1x90-relu.90.s-wc.90x1x100.l2:4-fir.15x1x100-relu.100.s-wc.100x120.l2:4-relu.120.s-wc.120xR.l2:4-dexp.R_dfit.v95.u15'
]

shortnames = ['CNN32-bsg', 'LN32', 'sspredxc']

modelname = modelnames[0]
shortname = shortnames[0]

cellid='LMD052a-A'

loadkey='gtgram.fs100.ch32'
ex=baphy_experiment.BAPHYExperiment(cellid=cellid, batch=batch)
rec=ex.get_recording(loadkey=loadkey)


#xf, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname)
#stim=ctx['rec']['stim']
stim=rec['stim'].rasterize()

stim_epochs = stim.epoch_names_matching("STIM_00")
epoch = stim_epochs[0]
wavbase = epoch.replace('STIM_','')
EXAMPLE_WAV = os.path.join(soundpath, wavbase)


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
# model = load_acnet(version='v1')
model = load_acnet(version='v1')  # default = v1
print(f"Built ACNet: compress={model.layers[0].mode!r}, "
      f"embeddings dim={model.layers[-4].shape[0]}")  # hidden_dim[-1]
model.lbhb_mode=True
model.level_mode = 'approx'

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
gtg = model._to_gtg(wav, fs_stim)  # exactly what step 2 computed internally -- single-sqrt
embeddings_from_gtg = model.get_embeddings(gtg)
print(f"3. From precomputed gtg: embeddings shape {embeddings_from_gtg.shape}, "
      f"matches (1): {np.array_equal(embeddings_from_gtg, embeddings_from_path)}")


stim0 = stim.extract_epoch(epoch)[0]

# trim silenct from stim0
stim0=stim0[:,100:1879].T

# [AGENT EDIT START | agent: claude | user: svd | reason: complete TODO -- project stim0 (the batch's own gtgram loader output) through ACNet and compare against the wav-derived gtg/embeddings above | date: 2026-09-18]
########################################################
# 4. Project stim0 (the actual gtgram used to fit the CNN model, straight
#    from `ctx['rec']['stim']`) through ACNet directly. stim0 is BAPHY's real
#    cached recording -- double-square-rooted relative to `gtg` (see module
#    docstring: `NAT_stim` applies its own extra sqrt on top of the standard
#    gtgram). Square it back down to the model's normal single-sqrt input
#    convention before calling get_embeddings, so embeddings_from_stim0 is
#    actually comparable to embeddings_from_gtg (both then see the domain
#    PowerCompress's log10x was calibrated on) -- feeding the raw, still
#    double-sqrt stim0 in directly would silently double-compress it.
########################################################
embeddings_from_stim0 = model.get_embeddings(stim0 ** 2)
print(f"\n4. From rec['stim'] epoch (stim0): embeddings shape {embeddings_from_stim0.shape}")

# For the RAW gtg-vs-stim0 comparison (does ACNet's own front end reproduce the
# real recording?), go the other way: sqrt `gtg` up into stim0's native
# double-sqrt domain, since stim0 is what actually got played/recorded and gtg
# is the thing being checked against it. Align to the shorter of the two before
# comparing (whatever padding/trimming differs between the two pipelines).
gtg_natstim = np.sqrt(gtg)
n_t = min(gtg_natstim.shape[0], stim0.shape[0])
gtg_trim, stim0_trim = gtg_natstim[:n_t], stim0[:n_t]
gtg_mse = np.mean((gtg_trim - stim0_trim) ** 2)
print(f"gtg vs stim0 MSE over first {n_t} bins: {gtg_mse:.6g}")

n_e = min(embeddings_from_gtg.shape[0], embeddings_from_stim0.shape[0])
emb_gtg_trim = embeddings_from_gtg[:n_e]
emb_stim0_trim = embeddings_from_stim0[:n_e]
emb_mse = np.mean((emb_gtg_trim - emb_stim0_trim) ** 2)
print(f"embeddings_from_gtg vs acnet(stim0) MSE over first {n_e} bins: {emb_mse:.6g}")

# Same CF axis convention as nems/tutorials/17_acnet_embeddings.py -- `fs`
# here doesn't actually affect the result since f_max is always given
# explicitly (erb_space only falls back to fs/2 when f_max is None), so
# model.fs_gtg is as good a value to pass as any.
cf_khz = centre_freqs(model.fs_gtg, model.num_cfs, model.f_min, model.f_max)[::-1] / 1e3
n_ticks = min(5, len(cf_khz))
tick_idx = np.linspace(0, len(cf_khz) - 1, n_ticks).round().astype(int)
cf_tick_pos = tick_idx + 0.5
cf_tick_labels = [f'{cf_khz[i]:.2g}' for i in tick_idx]

if SORT_ACNET_DIM:
    # Order by embeddings_from_gtg's own activity, then apply the same
    # order to both embeddings panels -- a fair side-by-side comparison,
    # not two independently-sorted panels.
    dim_order = np.argsort(-(emb_gtg_trim ** 2).sum(axis=0))
    emb_gtg_plotted = emb_gtg_trim[:, dim_order]
    emb_stim0_plotted = emb_stim0_trim[:, dim_order]
    embeddings_title_suffix = ' (sorted by gtg activity)'
else:
    emb_gtg_plotted = emb_gtg_trim
    emb_stim0_plotted = emb_stim0_trim
    embeddings_title_suffix = ' (unsorted)'

dur_ms = 1e3 * n_t / model.fs_gtg

plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey='row')

ax[0, 0].imshow(gtg_trim.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, gtg_trim.shape[1]), cmap='gray_r')
ax[0, 0].set_yticks(cf_tick_pos)
ax[0, 0].set_yticklabels(cf_tick_labels)
ax[0, 0].set(ylabel='CF (kHz)', title='gtg (ACNet front end, from wav)')

ax[0, 1].imshow(stim0_trim.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, stim0_trim.shape[1]), cmap='gray_r')
ax[0, 1].set(title=f"rec['stim'] epoch (MSE={gtg_mse:.3g})")

ax[1, 0].imshow(emb_gtg_plotted.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, emb_gtg_plotted.shape[1]), cmap='gray_r')
ax[1, 0].set(ylabel='manifold dimension', xlabel='time (ms)',
             title='embeddings_from_gtg' + embeddings_title_suffix)

ax[1, 1].imshow(emb_stim0_plotted.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, emb_stim0_plotted.shape[1]), cmap='gray_r')
ax[1, 1].set(xlabel='time (ms)',
             title=f'acnet(stim0){embeddings_title_suffix} (MSE={emb_mse:.3g})')

fig.tight_layout()
if SAVE_FIGURE:
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'acnet_embedding_scratch_stim0_compare.png')
    fig.savefig(out_png, dpi=300)
    print(f"\nSaved gtg/stim0/embeddings comparison figure to {out_png}")
else:
    print("\nBuilt gtg/stim0/embeddings comparison figure (SAVE_FIGURE=False, not written to disk)")
# [AGENT EDIT END]
