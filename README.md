# NEMS
WIP refactor of lbhb/NEMS

# (temporary) installation instructions
1. Download source code.
```
clone <nems-lite> # using url, ssh, or however you normally clone
```
2a. (pip)
```
pip install -r nems-lite/requirements.txt
pip install -e nems-lite
```
2b. (conda)
```
conda env create -f nems-lite/environment.yml
pip install -e nems-lite
```

Note: `mkl` library for `numpy` does not play well with `tensorflow`.
If using `conda` to install dependencies manually, use `conda-forge`
for `numpy` (which uses `openblas` instead of `mkl`):
`conda install -c conda-forge numpy`
(https://github.com/conda-forge/numpy-feedstock/issues/84)

Coming soon, roughly in order of priority:
* Add more Layers from nems0.
* Add core pre-processing and scoring from nems0.
* Set up readthedocs.
* Convert scripts and dev_notebooks to tutorials where appropriate.
* Try Numba for Layer.evaluate and cost functions.
* Other core features (like jackknifed fits, cross-validation, etc.).
* Migrate to LBHB/NEMS.
* Enable Travis build (.travis.yml is already there, but not yet tested).
* Publish through conda install and pip install (and update readme accordingly).
* Backwards-compatibility tools for loading nems0 models.
* Implement Jax back-end.
... (other things on the massive issues list)
