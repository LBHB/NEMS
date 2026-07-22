# [AGENT EDIT START | agent: claude | user: svd | reason: new module - shared get_joblib_memory helper, factored out of nems.models.LN so it can also back caching in nems.preprocessing.spectrogram | date: 2026-07-16]
"""Miscellaneous utilities shared across the nems package."""
import functools
import logging
import os
import tempfile
from pathlib import Path

import joblib

log = logging.getLogger(__name__)

# [AGENT EDIT START | agent: claude | user: svd | reason: expose a directly-settable override so downstream packages (e.g. nems_db) can point caching at a shared location without needing an env var set before nems is imported | date: 2026-07-16]
# Highest-priority override for get_joblib_memory()'s cache location. Leave as
# None to fall back to NEMS_CACHE_DIR / the package defaults. Downstream code
# can set this directly, e.g.:
#     import nems.tools.utils as nems_utils
#     nems_utils.cache_path = '/auto/data/tmp/tstim'
# Checked fresh on every get_joblib_memory() call, so it takes effect for any
# subsequent cached call even if set after nems has already been imported.
cache_path = None
# [AGENT EDIT END]


def get_joblib_memory():
    """Get a joblib.Memory cache for on-disk memoization.

    Cache location is chosen in order of preference:
    1. `nems.tools.utils.cache_path`, if set (see module docstring/comment).
    2. `NEMS_CACHE_DIR` environment variable, if set.
    3. `~/.cache/nems`.
    4. The system temp directory (`nems-cache` subdirectory).

    If none of these directories are writable, caching is disabled (calls to
    `@memory.cache`-decorated functions run normally, without memoization).

    Re-evaluated on every call (rather than cached in a module-level
    singleton) so that setting `cache_path` at runtime takes effect on the
    next call, regardless of when it's set relative to import order.

    Returns
    -------
    joblib.Memory

    """
    candidates = []

    if cache_path is not None:
        candidates.append(Path(cache_path).expanduser())
    else:
        env_cache = os.environ.get('NEMS_CACHE_DIR')
        if env_cache:
            candidates.append(Path(env_cache).expanduser())

    candidates.extend([
        Path.home() / '.cache' / 'nems',
        Path(tempfile.gettempdir()) / 'nems-cache',
    ])

    for cache_dir in candidates:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return joblib.Memory(cache_dir, verbose=False)
        except OSError:
            log.warning("Unable to use cache directory %s", cache_dir)

    log.warning("No writable cache directory found; disabling joblib cache.")
    return joblib.Memory(location=None, verbose=False)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: svd | reason: decorator-factory form of get_joblib_memory, so callers can write @joblib_memory(...) directly on the function to cache instead of manually wrapping calls | date: 2026-07-16]
def joblib_memory(location=None):
    """Decorator factory: memoize a function's results to disk via joblib.

    Parameters
    ----------
    location : str or Path; optional.
        Fixed cache directory, resolved once at decoration time -- equivalent
        to `joblib.Memory(location).cache`. Use this when the cache location
        is a known constant (e.g. a project-specific shared directory).
        If omitted, the cache directory is instead re-resolved via
        `get_joblib_memory()` on *every call* to the decorated function, so a
        later change to `nems.tools.utils.cache_path` (or `NEMS_CACHE_DIR`)
        takes effect on the next call, regardless of when it's set relative
        to import order. Use this for library-level functions that should
        respect the shared, overridable default.

    Examples
    --------
    Fixed, project-specific cache location:
    >>> @joblib_memory('/auto/data/tmp/mouse_io')
    ... def my_cached_fn(x):
    ...     ...

    Shared, runtime-overridable default:
    >>> @joblib_memory()
    ... def my_cached_fn(x):
    ...     ...

    """
    if location is not None:
        fixed_memory = joblib.Memory(Path(location).expanduser(), verbose=False)
        return fixed_memory.cache

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return get_joblib_memory().cache(func)(*args, **kwargs)
        return wrapper
    return decorator
# [AGENT EDIT END]