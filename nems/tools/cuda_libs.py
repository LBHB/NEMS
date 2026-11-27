# [AGENT EDIT START | agent: claude-opus-5-5 | user: svd | reason: TF 2.21 pip wheel RUNPATH omits nvidia/cusolver/lib; preload it so the GPU registers without conda activation (queue/ARC/IDE launches) | date: 2026-11-27]
"""Make the pip CUDA wheels findable by TensorFlow without `conda activate`.

The TF 2.21 wheel's libtensorflow_framework.so.2 RUNPATH lists the nvidia-*-cu12
wheel lib dirs except nvidia/cusolver/lib. When libcusolver.so.11 can't be
found, TF silently registers no GPU (log: "Cannot dlopen some GPU libraries").
The conda activate hook (etc/conda/activate.d/tf_cusolver.sh) fixes this only
for activated shells. Jobs that run envs/<env>/bin/python directly (local
queue, ARC sbatch, IDEs) skip it.

Loading the library by full path with RTLD_GLOBAL before TF is imported means
TF's later dlopen("libcusolver.so.11") reuses it. This also takes precedence
over any older libcusolver.so.11 on LD_LIBRARY_PATH (eg the CUDA 11.8 copy in
ARC's old envs/nems/lib), so TF gets the version it was built against.
No-op if the nvidia wheels aren't installed or the library is already loaded.
Verified 2026-11-27: local RTX 4070 Ti (bare env, no LD_LIBRARY_PATH) and an
ARC A40 (sbatch LD_LIBRARY_PATH; cholesky/svd/solve on GPU).
"""
import ctypes
import glob
import importlib.util
import os

_LIBS = ('cusolver',)   # wheel subdirs missing from TF's RUNPATH


def preload_cuda_libs(libs=_LIBS):
    """dlopen nvidia/<lib>/lib/lib<lib>*.so* from the pip wheels; returns the paths loaded."""
    try:
        spec = importlib.util.find_spec('nvidia')
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    loaded = []
    for name in libs:
        for d in spec.submodule_search_locations:
            for f in sorted(glob.glob(os.path.join(d, name, 'lib', f'lib{name}*.so*'))):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(f)
                except OSError:
                    pass   # leave it to TF's own search
    return loaded
# [AGENT EDIT END]
