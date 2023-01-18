=========================
Installation instructions
=========================

Recommended: Install from source
================================
NEMS is still under rapid development, so this is the best way to ensure you're
using the most up-to-date version.

1. Download source code
::

    git clone https://github.com/lbhb/nems

2a. Create and activate a new virtual environment using your preferred
environment manager (example for :code:`venv` below).
::

    python -m venv ./nems-env
    ./nems-env/scripts/activate

2b. Install frozen dependencies. This will install the exact versions used
during development.
::

    pip install -r .\NEMS\requirements.txt


2c. Alternatively, use :code:`conda` to replace both step 2a and step 2b.
::

    conda env create -f NEMS/environment.yml
    conda activate nems-env


3. Install NEMS in editable mode along with optional development tools.
::

    pip install -e NEMS[dev]


4. Run tests to ensure proper installation. We recommend repeating this step
after making changes to the source code.
::

    pytest NEMS


Alternative: PyPI (pip)
=======================

Create a new environment using your preferred environment manager, then use
:code:`pip install`.
::

    conda create -n nems-env python=3.9 pip  # note that python=3.9 is currently required
    pip install PyNEMS                       # note the leading Py


Alternative: `conda install`
============================
Coming soon.


Note: the :code:`mkl` library for :code:`numpy` does not play well with
:code:`tensorflow`. If using :code:`conda` to install dependencies manually,
and you want to use the :code:`tensorflow` backend, use :code:`conda-forge` for
:code:`numpy` (which uses :code:`openblas` instead of :code:`mkl`):
::

    conda install -c conda-forge numpy

(See: `<https://github.com/conda-forge/numpy-feedstock/issues/84>`)
