# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NEMS (Neural Encoding Model System) is a Python library for fitting mathematical models to time series data, primarily used to develop and test computational models of auditory encoding in the brains of behaving mammals. Published as `PyNEMS` on PyPI.

This is a refactored version of the archived [NEMS0](https://github.com/LBHB/NEMS0).

## Build & Test Commands

```bash
# Install (editable, with dev tools)
pip install -e .[dev]

# Install with TensorFlow/GPU support
pip install -e .[dev,tf]

# Run all tests
pytest

# Run a single test file
pytest tests/test_registry.py

# Run a specific test
pytest tests/test_registry.py::test_function_name -v
```

## Architecture

### Core Pipeline: Model → Layers → Backend → Fit

The central workflow is: build a `Model` from `Layer`s, then call `model.fit(input, target, backend=...)` which delegates to a `Backend` for optimization.

**`nems.models.base.Model`** — The primary user-facing class (`from nems import Model`). Holds an ordered collection of Layers, manages evaluation (forward pass), fitting, and prediction. Pre-built model subclasses exist in `nems/models/` (e.g., `LN_STRF`, `CNN_pop`).

**`nems.layers.base.Layer`** — Each Layer encapsulates one data-transformation step. Layers define:
- `evaluate()` — the forward computation (NumPy-based)
- `initial_parameters()` — returns a `Phi` containing `Parameter` objects
- `from_keyword()` — constructs a Layer from a keyword string (registered via `@layer` decorator)

All Layer subclasses auto-register in `Layer.subclasses` dict via `__init_subclass__`.

**`nems.layers.base.Phi` / `Parameter`** — `Phi` is a container for a Layer's fittable parameters. Each `Parameter` has values, bounds, and optional priors (from `nems.distributions`).

**`nems.backends`** — Backends translate the NumPy-based Model into an optimization framework:
- `scipy` (default) — SciPy minimize-based optimization
- `tf` / `tensorflow` — TensorFlow-based optimization (optional dependency, only imported if used)

Each backend implements `_build()`, `_fit()`, and `predict()`. Selected via `model.fit(..., backend='scipy')` or `model.fit(..., backend='tf')`.

**`nems.models.dataset.DataSet`** — Container that tracks input/state/target/output arrays flowing through a Model during evaluation and fitting.

### Keyword Registry System

`nems.registry.keyword_lib` is a global `KeywordRegistry` that maps short string keywords to Layer constructors. Layers register via the `@layer('keyword_name')` decorator on their `from_keyword` staticmethod. Models can be constructed from keyword strings: e.g., `Model.from_keywords('wc.18x1-fir.15x1-dexp.1')`.

### Data I/O (`nems.tools`)

- `signal.py` — `SignalBase`, `RasterizedSignal`, `PointProcess` classes for time series data with epoch/channel metadata
- `recording.py` — `Recording` class that holds a dictionary of Signal objects
- `epoch.py` — Epoch manipulation utilities (merge, intersect, filter by name)
- `demo_data/` — Download and load demo datasets

### Other Modules

- `nems.preprocessing` — Spectrogram generation, normalization, raster conversion, data splitting (jackknife)
- `nems.metrics` — Performance metrics (correlation, noise-corrected r)
- `nems.visualization` — matplotlib-based plotting for models, layers, fits, and dSTRF analysis
- `nems.distributions` — Prior distributions for Bayesian parameter specification (Normal, HalfNormal, Uniform, Beta, Gamma, Exponential)

## Key Conventions

- Data arrays are shaped `(T, N)` where T = time bins, N = channels. When samples are present: `(S, T, N)`.
- `numpy >= 1.24.4, < 2.0` is required — NumPy 2.0 is not yet supported.
- TensorFlow backend is optional and lazily imported — code should work without it installed.
- The `@layer` decorator in `nems.registry` is used to register Layer keyword parsers; new layers should follow this pattern.
- Tutorials are in `tutorials/` (numbered Python scripts) and `tutorials/notebooks/` (Jupyter).
