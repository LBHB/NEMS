# [AGENT EDIT START | agent: claude-opus-5-5 | user: svd | reason: shared cache for per-cell analysis tables stored in model.meta (gabor_tuning, dpc_gabor), so they persist in modelspec.json and only missing rows are refit | date: 2026-10-26]
"""Cache per-cell analysis results (DataFrames) in `model.meta`.

A cached table lives in `meta[key]`, and the options that produced it in
`meta[key + '_opts']`. Both are saved with the model (see nems.tools.json),
so a reloaded model carries its cache.

`cached_table` fits only the requested rows missing from the cache. If the
fit options differ from the cached ones (including a fingerprint of the
fitted filters, see `array_fingerprint`), the cache is discarded and the
requested rows are refit.

A table assigned to meta[key] by hand, with no `_opts` entry, is trusted
as-is (back-compatible with older scripts that did
`model.meta['gabor_tuning'] = model.get_gabor_tuning()`).
"""
import hashlib
import json
import logging

import numpy as np

log = logging.getLogger(__name__)


def array_fingerprint(x):
    """Short hash of an array's values, stable across float32/float64 storage."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return hashlib.md5(x.tobytes() + str(x.shape).encode()).hexdigest()[:16]


def _normalize(opts):
    # round-trip through the model JSON encoder so values compare the same way
    # before and after a save/reload (tuples -> lists, numpy scalars -> python)
    from nems.tools.json import nems_to_json
    return json.loads(nems_to_json(opts))


def cached_table(meta, key, opts, requested, fit_fn, force=False):
    """Return rows `requested` of the table cached in meta[key], fitting any
    that are missing.

    Parameters
    ----------
    meta : dict
        Usually model.meta. Updated in place.
    key : str
        Table name, eg 'gabor_tuning'. Options are stored in meta[key+'_opts'].
    opts : dict
        Everything that changes the fit result. Must be JSON-encodable by
        nems.tools.json.
    requested : list
        Index labels of the rows wanted (cellids, or (cellid, pc_idx) tuples).
    fit_fn : callable
        fit_fn(missing) -> DataFrame with one row per label in `missing`,
        indexed the same way as the cached table.
    force : bool
        If True, discard the cache and refit the requested rows.

    Returns
    -------
    df : DataFrame
        Copy of the requested rows, in requested order.
    n_new : int
        Number of rows fit by this call (0 = everything came from the cache).
    """
    import pandas as pd

    opts = _normalize(opts)
    okey = key + '_opts'
    cache = meta.get(key, None)
    if cache is not None:
        cached_opts = meta.get(okey, None)
        if force:
            log.info(f"{key}: force=True, discarding cached table")
            cache = None
        elif (cached_opts is not None) and (cached_opts != opts):
            changed = sorted(k for k in set(opts) | set(cached_opts)
                             if opts.get(k) != cached_opts.get(k))
            log.info(f"{key}: fit options changed ({changed}), discarding cached table")
            cache = None

    if cache is None:
        missing = list(requested)
    else:
        missing = [r for r in requested if r not in cache.index]

    if len(missing) > 0:
        log.info(f"{key}: fitting {len(missing)}/{len(requested)} requested rows")
        new = fit_fn(missing)
        if cache is None:
            cache = new
        else:
            cache = pd.concat([cache, new])
            cache = cache[~cache.index.duplicated(keep='last')]
        meta[key] = cache
        meta[okey] = opts
    else:
        log.info(f"{key}: all {len(requested)} requested rows found in cache")

    return cache.loc[list(requested)].copy(), len(missing)
# [AGENT EDIT END]
