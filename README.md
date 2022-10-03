# NEMS

The Neural Encoding Model System (NEMS) is helpful for fitting a mathematical model to time series data, plotting the model predictions, and comparing the predictive accuracy of multiple models. We use it to [develop and test computational models of how sound is encoded in the brains of behaving mammals](https://hearingbrain.org/), but it will work with many different types of timeseries data.

Note: this is a refactor of a former version of NEMS that is now archived at [LBHB/NEMS0](github.com/LBHB/NEMS0). *This refactor is still very much a work in progress, and is likely to contain bugs, missing documentation, and a lot of TODOs.*.

# Examples
Build a standard linear-nonlinear spectrotemporal receptive field (LN-STRF) model.
```
from nems import Model
from nems.layers import FiniteImpulseResponse, DoubleExponential

model = Model()
model.add_layers(
    FiniteImpulseResponse(shape=(15, 18)),  # 15 taps, 18 spectral channels
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)
```
Or use the customizable keyword system for faster scripting and prototyping.
```
from nems import Model

same_model = Model.from_keywords('fir.15x18-dexp.1')
```
Fit the model to (fake) evoked neural activity (in this case, in response to a sound represented by a spectrogram).
```
import numpy as np

spectrogram = np.random.rand(1000, 18)  # 1000 time bins, 18 channels
response = np.random.rand(1000, 1)      # 1 neural response

fitted_model = model.fit(spectrogram, response)
```
Predict the response to a different stimulus.
```
test_spectrogram = np.random.rand(1000, 18)
prediction = fitted_model.predict(test_spectrogram)
```
Score the prediction
```
from nems.metrics import correlation
print(correlation(prediction, response))
# OR
print(model.score(test_spectrogram, response, metric='correlation'))
```
Try the above examples with real data:
```
import nems

nems.download_demo()
training_dict, test_dict = nems.load_demo()

# Each dictionary contains a 100 hz natural sound spectrogram and
# the PSTH / firing rate of the recorded spiking response.
spectrogram = training_dict['spectrogram']
response = training_dict['response']
```



# Installation instructions
## Recommended: Intall from source.
NEMS is still under rapid development, so this is the best way to ensure you're using the most up-to-date version.
1. Download source code
```
git clone https://github.com/lbhb/nems
```
2. Create a new virtual environment using your preferred environment manager (examples for `conda` below).
```
conda create -n nems-env python=3.9 pip        # manually
# OR (don't do both)
conda env create -f NEMS/environment.yml  # or with .yml
```

3. Install in editable mode with optional development tools.
```
pip install -e NEMS[dev]
```
4. Run tests to ensure proper installation. We recommend repeating this step after making changes to the source code.
```
pytest NEMS
```

## Alternative: PyPI (pip)
Create a new environment using your preferred environment manager, then use pip install.
```
conda create -n nems-env python=3.9 pip
pip install PyNEMS
```

## Alternative: Conda
Coming soon.


Note: the `mkl` library for `numpy` does not play well with `tensorflow`.
If using `conda` to install dependencies manually, and you want to use the `tensorflow` backend, use `conda-forge` for `numpy` (which uses `openblas` instead of `mkl`):
```
conda install -c conda-forge numpy
```
(See: https://github.com/conda-forge/numpy-feedstock/issues/84)


# Upcoming features/updates
Roughly in order of priority:
* Add more Layers from nems0.
* Add core pre-processing and scoring from nems0.
* Set up readthedocs.
* Demo data.
* Other core features (like jackknifed fits, cross-validation, etc.).
* Enable Travis build (.travis.yml is already there, but not yet tested).
* Publish through conda.
* Backwards-compatibility tools for loading nems0 models.
* Try Numba for Layer.evaluate and cost functions.
* Implement Jax back-end.
* ... (see issues tracker).
