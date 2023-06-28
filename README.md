# NEMS &middot; ![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg) ![Python 3.10](https://img.shields.io/badge/Python-3.10-green) ![Python 3.10](https://img.shields.io/badge/Python-3.9-green)

The Neural Encoding Model System (NEMS) is helpful for fitting a mathematical model to time series data, plotting the model predictions, and comparing the predictive accuracy of multiple models. We use it to [develop and test computational models of how sound is encoded in the brains of behaving mammals](https://hearingbrain.org/), but it will work with many different types of timeseries data.

* **Note:** this is a refactor of a former version of NEMS that is now archived at [https://github.com/LBHB/NEMS0](https://github.com/LBHB/NEMS0). *This refactor is still very much a work in progress, and is likely to contain bugs, missing documentation, and a lot of TODOs.*.

## Table of Contents & Relevant Links
### [Install](#installation) &middot; [Examples](#examples) &middot; [Upcoming](#upcoming) &middot; [Documentation](#documentation)   

#### - **Laboratory of Brain, Hearing, and Behavior:**  [https://hearingbrain.org/](https://hearingbrain.org/)    
#### - **Archived former NEMS version:**   [https://github.com/LBHB/NEMS0](https://github.com/LBHB/NEMS0)   

<br />
<br />
<br />
<br />

# Installation
## Recommended: `Install from source`
### Prerequisites: Python 3.9/3.10 | Anaconda or Venv
NEMS is still under rapid development, so this is the best way to ensure you're using the most up-to-date version.

1. **Download source code**
```console
git clone https://github.com/LBHB/NEMS.git
```
2. **Set up Virtual Environment Anaconda | Venv**
    - 2a. Create and activate a new virtual environment using your preferred environment manager (example for `venv` below).   
    ```console
    python -m venv ./nems-env
    source nems-env/bin/activate
    ```   
    - 2b. Install frozen dependencies. This will install the exact versions used during development.   
    ```console
    pip install -r ./NEMS/requirements.txt
    ```   
    - 2c. Alternatively, use `conda` to replace both step 2a and step 2b.  
    ```console
    conda env create -f NEMS/environment.yml
    conda activate nems-env
    ```    
3. **Install NEMS in editable mode along with optional development tools.**
```console
pip install -e NEMS[dev]
```
4. **Run tests to ensure proper installation. We recommend repeating this step after making changes to the source code.**
```console
pytest NEMS
```

## Alternative: `PyPI (pip)`
**Create a new environment using your preferred environment manager, then use `pip install`.**
```console
conda create -n nems-env python=3.9 pip  # note that python=3.9 is currently required
pip install PyNEMS                       # note the leading Py
```

##  Alternative: `VSCode(Windows)`    
### Prerequisites: Python 3.9/3.10 | Anaconda

1.	Download the source-code   
```console
git clone https://github.com/LBHB/NEMS.git
```  
2. **Make sure you have Python and Anaconda installed onto your machine**
	- 2a. Go to windows execution alias's and disable the default python.exe & python3.exe
3. **Set up Visual Studio Code**
    - 3a. Run VSCode as admin and open up your NEMS folder
    - 3b. Install Python extension( Recommend installing environment manager extension too )
    - 3c. ctrl-shift-p inside VSCode, search for "Python: Create Environment"
    - 3d. Choose Anaconda 3.10
4. **Set up virtual environment and install NEMS**
    - 4a. Virtual Env
    ```console
    conda env create -f ./environment.yml
    conda activate nems-env
    conda init
    ```
    - 4b. Restart VSCode and install nems module
    ```console
    pip install -e NEMS[dev]
    ```
5. **Test that your installation was successful**
```console
py -m pytest
```

## Alternative: `conda install`.
**Coming soon.**


Note: the `mkl` library for `numpy` does not play well with `tensorflow`.
If using `conda` to install dependencies manually, and you want to use the `tensorflow` backend, use `conda-forge` for `numpy` (which uses `openblas` instead of `mkl`):
```console
conda install -c conda-forge numpy
```
(See: https://github.com/conda-forge/numpy-feedstock/issues/84)

<br />
<br />
<br />
<br />

# Examples
Build a standard linear-nonlinear spectrotemporal receptive field (LN-STRF) model.
```python
from nems import Model
from nems.layers import FiniteImpulseResponse, DoubleExponential

model = Model()
model.add_layers(
    FiniteImpulseResponse(shape=(15, 18)),  # 15 taps, 18 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)
```
Or use the customizable keyword system for faster scripting and prototyping.
```python
from nems import Model

same_model = Model.from_keywords('fir.15x18-dexp.1')
```
Fit the model to (fake) evoked neural activity (in this case, in response to a sound represented by a spectrogram).
```python
import numpy as np

spectrogram = np.random.rand(1000, 18)  # 1000 time bins, 18 channels
response = np.random.rand(1000, 1)      # 1 neural response

fitted_model = model.fit(spectrogram, response)
```
Predict the response to a different stimulus.
```python
test_spectrogram = np.random.rand(1000, 18)
prediction = fitted_model.predict(test_spectrogram)
```
Score the prediction
```python
from nems.metrics import correlation
print(correlation(prediction, response))
# OR
print(model.score(test_spectrogram, response, metric='correlation'))
```
Try the above examples with real data:
```python
import nems

nems.download_demo()
training_dict, test_dict = nems.load_demo()

# Each dictionary contains a 100 hz natural sound spectrogram and
# the PSTH / firing rate of the recorded spiking response.
spectrogram = training_dict['spectrogram']
response = training_dict['response']
```

# Documentation
**Note: Documentation is currently a work-in-progress**   
### - [Main Page](https://nems.readthedocs.io/en/latest/index.html)   
### - [Introduction](https://nems.readthedocs.io/en/latest/tutorials/intro.html)   
### - [Modules](https://nems.readthedocs.io/en/latest/api/modules.html)   
### - [Quickstart](https://nems.readthedocs.io/en/latest/quickstart.html)   


# Upcoming
Roughly in order of priority:
* Add core pre-processing and scoring from nems0.
* Set up readthedocs.
* Other core features (like jackknifed fits, cross-validation, etc.).
* Enable Travis build (.travis.yml is already there, but not yet tested).
* Publish through conda.
* Backwards-compatibility tools for loading nems0 models.
* Implement Jax back-end.
* ... (see issues tracker).
