# NEMS &middot; ![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg) ![Python 3.10](https://img.shields.io/badge/Python-3.10-green) ![Python 3.10](https://img.shields.io/badge/Python-3.9-green)

The Neural Encoding Model System (NEMS) is helpful for fitting a mathematical model to time series data, plotting the model predictions, and comparing the predictive accuracy of multiple models. We use it to [develop and test computational models of how sound is encoded in the brains of behaving mammals](https://hearingbrain.org/), but it will work with many different types of timeseries data.

* **Note:** this is a refactor of a former version of NEMS that is now archived at [https://github.com/LBHB/NEMS0](https://github.com/LBHB/NEMS0). *This refactor is still very much a work in progress, and is likely to contain bugs, missing documentation, and a lot of TODOs.*.

## Table of Contents & Relevant Links
### [Install](#installation) &middot; [Visualization](#visualization) &middot; [Examples](#examples) &middot; [Documentation](#documentation) &middot; [Upcoming](#upcoming)   

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
    - 3a. **Optional** Install Tensorflow
    ```console
    pip install -e NEMS[tf]
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
    pip install -e ./[dev]
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

# Visualization
A variety of our scripts and tutorials focus on providing plots and graphs to visualize data at various points in time. Viewing and interacting with this data is important to understanding our models and for utilizing all of the tools provided.

**There are many ways to visualize this data. We recommend using one of our current guides below.**

<hr />

## IPython
IPython is an interative python shell, typically know for its use in jupyter notebooks. Exact installation can vary based on your environment

### Install
* Install ipython inside your nems-env
```console
    sudo pip3 install ipython
```
#### *VSCode*
* Additionally, set up launch args for ipython so it's used properly running code from terminal
```JSON
"python.terminal.launchArgs": [
    "-m",
    "IPython",
    "--no-autoindent",
],
```
### Usage
1. Start ipython inside your shell
    ```console
    ipython
    ```
2. Should see the following:
    ```console
    In [0]:|
    ```
3. Run script or file with interactive mode active:
    ```console
    %matplotlib
    %run ./path/to/file
    ```
* "exit" to leave ipython
    ```console
    In [n]: exit
    ```

#### *VSCode*
1. Highlight code
2. Click terminal at the top
3. "Run selected code"
4. Press enter in terminal
* If your launch arg's are correct, you can also right-click -> run python -> run select code in terminal

<br />
<br />
<br />
<br />

# Examples
Build a standard linear-nonlinear spectrotemporal receptive field (LN-STRF) model.
```python
import numpy as np
import nems
from nems import Model
from nems.layers import FiniteImpulseResponse, DoubleExponential
from nems.metrics import correlation


# Model layers can be added in sequential order as a list, individually, or via keywords
model = Model()
model.add_layers(
    FiniteImpulseResponse(shape=(15, 18)),  # 15 taps, 18 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)
```
Or use the customizable keyword system for faster scripting and prototyping.
```python

same_model = Model.from_keywords('fir.15x18-dexp.1')
another_model = Model('fir.15x18-dexp.1')
```
Fit the model to (fake) evoked neural activity (in this case, in response to a sound represented by a spectrogram).
```python

spectrogram = np.random.rand(1000, 18)  # 1000 time bins, 18 channels
response = np.stack(spectrogram[:, :5]) # 1 neural response, relative to the spectrogram

fitted_model = model.fit(spectrogram, response)
```
Predict the response to a different stimulus.
```python
test_spectrogram = np.random.rand(1000, 18)
prediction = fitted_model.predict(test_spectrogram)
```
Score the prediction
```python
print(correlation(prediction, response))
# OR
print(model.score(test_spectrogram, response, metric='correlation'))
```
Try the above examples with real data:
```python
# Download our dataset and pull the data as a tuple of training/testing
nems.download_demo()
training_dict, test_dict = nems.load_demo()

# Each dictionary contains a 100 hz natural sound spectrogram and
# the PSTH / firing rate of the recorded spiking response.
spectrogram = training_dict['spectrogram']
response = training_dict['response']
```

**Looking for more?**   
If you're looking for more information, examples, or details you can take a look at our provided tutorials at:   
 * *NEMS/tutorials*   
 * *NEMS/tutorials/notebooks*    
 * or You can also stop by sources and documentation below

<br />
<br />
<br />
<br />

# Documentation
**Note: Documentation is currently a work-in-progress**   
### - [Main Page](https://nems.readthedocs.io/en/latest/index.html)   
### - [Introduction](https://nems.readthedocs.io/en/latest/tutorials/intro.html)   
### - [Modules](https://nems.readthedocs.io/en/latest/api/modules.html)   
### - [Quickstart](https://nems.readthedocs.io/en/latest/quickstart.html)   

<br />
<br />
<br />
<br />

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
