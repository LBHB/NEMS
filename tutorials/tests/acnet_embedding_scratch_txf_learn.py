"""Get ACNet manifold embeddings from a wav file, a waveform, or a gtg.

Demonstrates `nems.models.ACNet` (see project memory `project_acnet_in_nems`
for how this port came to be) loaded with the real released weights, then
`model.get_embeddings(...)` on all three input kinds it accepts. You never
pass a `compress` argument to `get_embeddings` -- the model's own first
layer (built with `compress='log10x'`, the released checkpoint's actual
training config) is the single place compression is specified, and it's
applied internally exactly once, regardless of which input kind you use.

2026-09-28: `nems.layers.compression.PowerCompress` had a double-compression
bug (fixed same day, see its own docstring) that affected every `acgram`-
based fit, including `modelname_ac` below, through `NAT_stim`'s "don't apply
square root" acgram path (`nems_db` `runclass.py`, on `origin/dev` -- see
that repo's new `acnet_migration` branch, forked from `dev` for this work).
`modelname_ac` is loaded via `fit_model_xform` (a genuine refit), not
`load_model_xform` (a reload) -- its cached fit/readout predates the
PowerCompress fix and reusing it would replay weights trained on
double-compressed input.

Every affected site's cached front-end *recording* also predates the fix
(confirmed on two: CLT027c's and PRN007a's `acgram.fs100` recordings, both
dated 2026-09-25) -- `_ensure_fresh_acgram_recording` below force-recaches
whenever the resolved recording is older than POWERCOMPRESS_FIX_DATE, for
whichever site `cellid` names, not just those two. recache=True isn't
reachable through the acgram/gtgram loadkey string itself (confirmed by
reading nems_lbhb.plugins.lbhb_loaders._parse_baphy_loadkey and
xform_wrappers.generate_recording_uri -- neither forwards **options that
far), so this monkeypatches BAPHYExperiment.get_recording_uri for the
`acgram`-labeled call only (gtgram/modelname_gt's own recording is
unaffected by this bug and is left alone). Once every site's cache has
actually been regenerated past the cutoff, this becomes a fast no-op.
Confirmed the refit actually changes once recached: CLT027c mean r_test
0.2935 (stale) -> 0.3640 (fresh); PRN007a not yet independently reconfirmed
after being interrupted mid-run on the wrong host (see conversation).
"""
import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')  # headless-safe; drop this if you want an interactive window
import matplotlib.pyplot as plt

from nems.models.ACNet import load_acnet
from nems.preprocessing.spectrogram import load_wav
from nems.preprocessing.spectrogram.filters import centre_freqs
from nems.tools.recording import load_recording
from nems_lbhb import xform_helper, db, baphy_experiment
from nems_lbhb.utils import adjust_uri_prefix

soundpath = '/auto/data/sounds/BigNat/v3/'


batch=390
siteids, cellids = db.get_batch_sites(batch)

modelname_gt = 'gtgram.fs100.ch32-ld-norm.l1-sev_wc.Nx1x90.g-fir.15x1x90-relu.90.s-wc.90x1x100.l2:4-fir.15x1x100-relu.100.s-wc.100x120.l2:4-relu.120.s-wc.120xR.l2:4-dexp.R_lite.tf.init.lr1e3.t3.es20.rb4-lite.tf.lr1e4.t5e4'
modelname_ac = 'acgram.fs100-ld-norm-sev_wc.NxR.l2:4-dexp.R_lite.tf.init.lr1e3.t3.es20.rb2-lite.tf.lr1e4.t5e4'

# cellid='CLT027c'
# cellid='CLT028c'
cellid='PRN018a'

xf, ctx_gt = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname_gt, eval_model=True)

# See the module docstring's 2026-09-28 note: force a recache whenever this
# site's acgram recording predates the PowerCompress fix, for whichever
# `cellid` is set above -- not just the two sites checked by hand so far.
import datetime
POWERCOMPRESS_FIX_DATE = datetime.datetime(2026, 9, 28)


def _ensure_fresh_acgram_recording(cellid, batch, loadkey='acgram.fs100'):
    orig = baphy_experiment.BAPHYExperiment.get_recording_uri

    def _patched(self, generate_if_missing=True, cellid=None, loadkey=None,
                 extra_label=None, recache=False, **kwargs):
        label = extra_label or loadkey or ''
        if (not recache) and ('acgram' in label):
            uri = orig(self, generate_if_missing=False, cellid=cellid, loadkey=loadkey,
                      extra_label=extra_label, recache=False, **kwargs)
            if os.path.exists(uri) and \
                    datetime.datetime.fromtimestamp(os.path.getmtime(uri)) < POWERCOMPRESS_FIX_DATE:
                print(f"{cellid}'s cached {label} recording predates the "
                      f"2026-09-28 PowerCompress fix -- forcing recache")
                recache = True
        return orig(self, generate_if_missing=generate_if_missing, cellid=cellid,
                   loadkey=loadkey, extra_label=extra_label, recache=recache, **kwargs)

    baphy_experiment.BAPHYExperiment.get_recording_uri = _patched
    try:
        baphy_experiment.BAPHYExperiment(batch=batch, cellid=cellid).get_recording_uri(
            extra_label=loadkey, loadkey=loadkey)
    finally:
        baphy_experiment.BAPHYExperiment.get_recording_uri = orig


_ensure_fresh_acgram_recording(cellid, batch)

# Genuine refit, not a reload -- see the module docstring's 2026-09-28 note.
# Once verified (the r_test scatter in section 5 below looks right), rerun
# with SAVE_AC_FIT=True to persist the corrected fit to celldb; until then
# this never touches the shared Results table.
SAVE_AC_FIT = False
xf, ctx_ac = xform_helper.fit_model_xform(cellid=cellid, batch=batch, modelname=modelname_ac,
                                          saveInDB=SAVE_AC_FIT, returnModel=True, autoPlot=False)

rec=load_recording(adjust_uri_prefix(ctx_gt['recording_uri_list'][0]))
rec2=load_recording(adjust_uri_prefix(ctx_ac['recording_uri_list'][0]))

stim=rec['stim'].rasterize()
stim2=rec2['stim'].rasterize()

stim_epochs = stim.epoch_names_matching("STIM_00")
epoch = stim_epochs[0]
wavbase = epoch.replace('STIM_','')
EXAMPLE_WAV = os.path.join(soundpath, wavbase)


# If True (default), the ACNet-embeddings panel's manifold dimensions are
# reordered by descending activity (sum of squared response) so the most
# active dimensions plot together, near the bottom -- purely a display
# convenience; the dimensions themselves, and everything returned by
# get_embeddings, are unaffected.
SORT_ACNET_DIM = False

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
model = load_acnet()  # default = v1
print(f"Built ACNet: compress={model.layers[0].mode!r}, "
      f"embeddings dim={model.layers[-4].shape[0]}")  # hidden_dim[-1]
model.lbhb_mode=True
model.level_mode='approx'

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

print('computing gtg with zero-padding (reduce std slightly)')
wav2 = np.concatenate([np.zeros(fs_stim), wav])
gtg2 = model._to_gtg(wav2, fs_stim)  # exactly what step 2 computed internally
gtg2=gtg2[100:]

embeddings_from_gtg = model.get_embeddings(gtg)
embeddings_from_gtg2 = model.get_embeddings(gtg2)
print(f"3. From precomputed gtg: embeddings shape {embeddings_from_gtg.shape}, "
      f"matches (1): {np.array_equal(embeddings_from_gtg, embeddings_from_path)}")


stim0 = stim.extract_epoch(epoch)[0]**2


ac0 = rec2['stim'].rasterize().extract_epoch(epoch)[0]

# trim silenct from stim0
stim0=stim0[:,100:1879].T
ac0=ac0[:,100:1879].T

# [AGENT EDIT START | agent: claude | user: svd | reason: complete TODO -- project stim0 (the batch's own gtgram loader output) through ACNet and compare against the wav-derived gtg/embeddings above | date: 2026-09-18]
########################################################
# 4. Project stim0 (the actual gtgram used to fit the CNN model, straight
#    from `ctx['rec']['stim']`) through ACNet directly -- it's already a
#    (T, num_cfs) sqrt-domain gammatone spectrogram, the same convention
#    `_to_gtg` expects for a precomputed gtg, so no wav/waveform detour is
#    needed.
########################################################
embeddings_from_stim0 = model.get_embeddings(stim0)
print(f"\n4. From rec['stim'] epoch (stim0): embeddings shape {embeddings_from_stim0.shape}")

# gtg (ACNet's own front end, from EXAMPLE_WAV) and stim0 (the batch's
# gtgram loader, from the same underlying sound) should agree, modulo
# whatever padding/trimming differs between the two pipelines -- align to
# the shorter of the two before comparing.
n_t = min(gtg.shape[0], stim0.shape[0])
n_t = 200
gtg_trim, stim0_trim = gtg[:n_t], stim0[:n_t]
gtg2_trim = gtg2[:n_t]
gtg_mse = np.mean((gtg_trim - stim0_trim) ** 2)
gtg2_mse = np.mean((gtg_trim - gtg2_trim) ** 2)
print(f"gtg vs stim0 MSE over first {n_t} bins: {gtg_mse:.6g}")

n_e = min(embeddings_from_gtg.shape[0], embeddings_from_stim0.shape[0])
n_e=200
emb_gtg_trim = embeddings_from_gtg[:n_e]
emb_gtg2_trim = embeddings_from_gtg2[:n_e]
emb_stim0_trim = embeddings_from_stim0[:n_e]
ac0_trim = ac0[:n_e]
SKIP=0
emb2_mse = np.mean((emb_gtg_trim[SKIP:] - emb_gtg2_trim[SKIP:]) ** 2)
emb_mse = np.mean((emb_gtg_trim[SKIP:] - emb_stim0_trim[SKIP:]) ** 2)
ac0_mse = np.mean((emb_gtg_trim[SKIP:] - ac0_trim[SKIP:]) ** 2)
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
    emb_gtg2_plotted = emb_gtg2_trim[:, dim_order]
    emb_stim0_plotted = emb_stim0_trim[:, dim_order]
    ac0_trim = ac0_trim[:, dim_order]
    embeddings_title_suffix = ' (sorted by gtg activity)'
else:
    emb_gtg_plotted = emb_gtg_trim
    emb_gtg2_plotted = emb_gtg2_trim
    emb_stim0_plotted = emb_stim0_trim
    embeddings_title_suffix = ' (unsorted)'

dur_ms = 1e3 * n_t / model.fs_gtg
plt.rcParams.update({'font.size': 11})

# plt.close('all')
fig, ax = plt.subplots(2, 4, figsize=(12, 6), sharex=True)

ax[0, 0].imshow(gtg_trim.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, gtg_trim.shape[1]), cmap='gray_r')
ax[0, 0].set_yticks(cf_tick_pos)
ax[0, 0].set_yticklabels(cf_tick_labels)
ax[0, 0].set(ylabel='CF (kHz)', title='gtg (ACNet front end, from wav)', xlabel='')

ax[0, 1].imshow(gtg2_trim.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, gtg2_trim.shape[1]), cmap='gray_r')
ax[0, 1].set(title=f"gtg2 (MSE={emb2_mse:.3g})", xlabel='')

ax[0, 2].imshow(stim0_trim.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, stim0_trim.shape[1]), cmap='gray_r')
ax[0, 2].set(title=f"rec['stim'] epoch (MSE={gtg_mse:.3g})", xlabel='')

t = np.arange(gtg_trim.shape[0])/gtg_trim.shape[0]*dur_ms
ax[0, 3].plot(t,gtg_trim.mean(axis=1), label='gtg')
ax[0, 3].plot(t,gtg2_trim.mean(axis=1), label='gtg0')
ax[0, 3].plot(t,stim0_trim.mean(axis=1), label='stim0')
ax[0, 3].set(title='mean spect per time', xlabel='')
ax[0, 3].legend()

ax[1, 0].imshow(emb_gtg_plotted.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, emb_gtg_plotted.shape[1]), cmap='gray_r')
ax[1, 0].set(ylabel='manifold dimension', xlabel='time (ms)',
             title='embeddings_from_gtg' + embeddings_title_suffix)

ax[1, 1].imshow(emb_gtg2_plotted.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, emb_gtg2_plotted.shape[1]), cmap='gray_r')
ax[1, 1].set(xlabel='time (ms)',
             title=f'gtg2{embeddings_title_suffix} (MSE={emb_mse:.3g})')

ax[1, 2].imshow(emb_stim0_plotted.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, emb_stim0_plotted.shape[1]), cmap='gray_r')
ax[1, 2].set(xlabel='time (ms)',
             title=f'acnet(stim0) --NEMS scaling - BAD-- (MSE={emb2_mse:.3g})')

ax[1, 3].imshow(ac0_trim.T, origin='lower', aspect='auto',
                extent=(0, dur_ms, 0, ac0_trim.shape[1]), cmap='gray_r')
ax[1, 3].set(xlabel='time (ms)',
             title=f'acnet0 (MSE={ac0_mse:.3g})')
#ax[1, 2].set(xlabel='time (ms)',
#             title=f'acnet(stim0){embeddings_title_suffix} (MSE={emb_mse:.3g})')

fig.tight_layout()
if SAVE_FIGURE:
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'acnet_embedding_scratch_stim0_compare.png')
    fig.savefig(out_png, dpi=300)
    print(f"\nSaved gtg/stim0/embeddings comparison figure to {out_png}")
else:
    print("\nBuilt gtg/stim0/embeddings comparison figure (SAVE_FIGURE=False, not written to disk)")
# [AGENT EDIT END]


########################################################
# 5. GT (end-to-end deep CNN, gtgram front end) vs. AC (frozen ACNet trunk +
#    freshly-trained per-site readout, acgram front end) -- per-cell r_test.
#    This is the actual task-F question: does ACNet's frozen-trunk transfer
#    learning do worse than training a CNN end-to-end? `modelspec.meta
#    ['r_test']` holds one row per cell in the whole site's recording (not
#    just `cellid` above, which only picked which site to load), aligned
#    with `meta['cellids']` -- align explicitly by name rather than
#    assuming the two fits list cells in the same order.
########################################################
cellids_gt = list(ctx_gt['modelspec'].meta['cellids'])
cellids_ac = list(ctx_ac['modelspec'].meta['cellids'])
r_test_gt = ctx_gt['modelspec'].meta['r_test'][:, 0]
r_test_ac = ctx_ac['modelspec'].meta['r_test'][:, 0]

common_cells = [c for c in cellids_gt if c in cellids_ac]
r_gt = np.array([r_test_gt[cellids_gt.index(c)] for c in common_cells])
r_ac = np.array([r_test_ac[cellids_ac.index(c)] for c in common_cells])
print(f"\n5. {len(common_cells)} cells common to both fits "
      f"(gt has {len(cellids_gt)}, ac has {len(cellids_ac)})")
print(f"   mean r_test: gt={np.nanmean(r_gt):.4f}  ac={np.nanmean(r_ac):.4f}"
      f"  diff(ac-gt)={np.nanmean(r_ac - r_gt):+.4f}")

fig2, ax2 = plt.subplots(figsize=(5, 5))
lims = (min(np.nanmin(r_gt), np.nanmin(r_ac)) - 0.02,
        max(np.nanmax(r_gt), np.nanmax(r_ac)) + 0.02)
ax2.plot(lims, lims, 'k--', lw=1, zorder=1)
ax2.scatter(r_gt, r_ac, s=16, alpha=0.75, zorder=2)
ax2.set(xlim=lims, ylim=lims, xlabel='r_test, end-to-end CNN (gt)',
        ylabel='r_test, frozen ACNet trunk + readout (ac)',
        title=f'{cellid[:7]} (n={len(common_cells)} cells)')
fig2.tight_layout()
if SAVE_FIGURE:
    out_png2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'acnet_embedding_scratch_rtest_scatter.png')
    fig2.savefig(out_png2, dpi=300)
    print(f"Saved gt-vs-ac r_test scatter to {out_png2}")
else:
    print("Built gt-vs-ac r_test scatter (SAVE_FIGURE=False, not written to disk)")
