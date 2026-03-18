import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, RectifiedLinear, STRF
from nems import visualization
from nems.metrics import correlation
from nems.models.LN import LN_plot_strf


## Synthesize some data 
#
# Model has a linear term and 2nd-order polynomial term
# r = s(8) + s(3)*s(5) + noise

# T x N (time x input channels)
cellid="Dummy"
input_fit = np.random.randn(10000,18)
input_test = np.random.randn(1000,18)

def gen_resp(s):
    r1 = s[:,[8]]
    r1[r1<0]=0
    r2 = s[:,[3]]*s[:,[5]]*3
    r2[r2<0]=0
    r = r1+r2+np.random.randn(s.shape[0],1)
    return r, r1, r2
    
response_fit, _, _ = gen_resp(input_fit)
response_test, r1, r2 = gen_resp(input_test)


## Intialize LN and simple NN models

ln_model = Model(name=f"{cellid}-GLM")
ln_model.add_layers(
    WeightChannels(shape=(18, 1)),  # 18 spectral channels->1 output
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False)  # static nonlinearity, 1 output
)

# Minimal-complexity two-layer network
n_banks=3
nn_model = Model(name=f"{cellid}-NN")
nn_model.add_layers(
    WeightChannels(shape=(18, n_banks)),  # 18 spectral channels->n_banks outputs
    RectifiedLinear(shape=(n_banks,)),  # static nonlinearity, n_bank output
    WeightChannels(shape=(n_banks, 1)),  # n_banks->1 output
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False)  # static nonlinearity, 1 output
)

# We can also see the additional dimension added to our input weight layer
# compared to how our simpler model is set up
print(f'''  LN input layer coefficient shape: {ln_model.layers[0].coefficients.shape}
  NN input layer coefficient shape: {nn_model.layers[0].coefficients.shape}''')


#   sample_from_priors(): Randomizes model parameters
ln_model = ln_model.sample_from_priors()
nn_model = nn_model.sample_from_priors()


# Basic options to quickly fit our models for tf backend
options = {'cost_function': 'squared_error', 
           'early_stopping_delay': 50, 'early_stopping_patience': 100,
           'early_stopping_tolerance': 1e-3, 'validation_split': 0,
           'learning_rate': 5e-3, 'epochs': 2000}

# Fit our models to sythesized data
# We use 'tf' backend to improve training speed.
fitted_ln = ln_model.fit(input_fit, response_fit, fitter_options=options, backend='tf')
fitted_nn = nn_model.fit(input_fit, response_fit, fitter_options=options, backend='tf')

## Plot the model weights

ln_parms0 = ln_model.get_parameter_values()
ln_parms = fitted_ln.get_parameter_values()
nn_parms0 = nn_model.get_parameter_values()
nn_parms = fitted_nn.get_parameter_values()

f,ax=plt.subplots(figsize=(2,2))
ax.plot(ln_parms0['WeightChannels']['coefficients'], '--', label='Init')
ax.plot(ln_parms['WeightChannels']['coefficients'], '-', label='Fit weights')
ax.set_title(f"{fitted_ln.name}")
ax.legend()

f,ax=plt.subplots(1,n_banks+1,figsize=(8,2))
for ic in range(n_banks):
    ax[ic].plot(nn_parms0['WeightChannels']['coefficients'][:,ic], '--', label='Init')
    ax[ic].plot(nn_parms['WeightChannels']['coefficients'][:,ic], '-', label='Fit weights')
    ax[ic].set_title(f"Bank {ic} weights")
ax[0].legend()
ax[-1].plot(nn_parms0['WeightChannels0']['coefficients'], '--', label='Init')
ax[-1].plot(nn_parms['WeightChannels0']['coefficients'], '-', label='Fit weights')
f.suptitle(f"{fitted_nn.name}")
f.tight_layout()


## Generate predictions

pred_ln = fitted_ln.predict(input_test)
pred_nn = fitted_nn.predict(input_test)

# Compare the model predictions to the linear (r1) and 2nd-order (r2) components of the response
# in the test data
f,ax=plt.subplots(3,1, sharex=True)

T=100
ax[0].plot(r1[:T],color='gray')
ax[0].plot(pred_ln[:T],color='orange')
ax[0].plot(pred_nn[:T],color='purple')
ax[0].set_ylabel('Linear component (r1)')

ax[1].plot(r2[:T],color='gray')
ax[1].plot(pred_ln[:T],color='orange')
ax[1].plot(pred_nn[:T],color='purple')
ax[1].set_ylabel('2nd-order comp. (r2)')

ax[2].plot(r1[:T]+r2[:T],color='gray', label='actual')
ax[2].plot(pred_ln[:T],color='orange', label='LN')
ax[2].plot(pred_nn[:T],color='purple', label='NN')
ax[2].set_ylabel('Full (r1+r2)')
ax[2].set_xlabel('Time (bins)')
ax[2].legend()
plt.tight_layout();

## Some built-in generic plots

visualization.plot_predictions({'ln prediction':pred_ln, 'nn prediction':pred_nn},
                               input_test, response_test, correlation=True);


# Plotting our Models after fits
fitted_ln.plot(input_test, target=response_test, figure_kwargs={'facecolor': 'papayawhip'})
fitted_nn.plot(input_test, target=response_test, figure_kwargs={'facecolor': 'lightblue'});
