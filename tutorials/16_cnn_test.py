import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, RectifiedLinear, STRF
from nems import visualization
from nems.metrics import correlation
from nems.models.LN import LN_plot_strf
import tensorflow as tf

# Check if eager execution is enabled
if tf.executing_eagerly():
    print("Eager execution is enabled.")
else:
    print("Eager execution is disabled.")

# Function to get STRF from a LN model
def get_strf(ln_mdl, channels=None):
    wc = ln_mdl.layers[0].coefficients
    fir = ln_mdl.layers[1].coefficients
    strf = wc @ fir.T

    return strf

cellid="DUMMY"
spectrogram_fit = np.random.randn(10,1000,18)
spectrogram_test = np.random.randn(10,1000,18)

def gen_resp(s, R=1):
    r1= s[:,:,[8]]
    r1[r1<0]=0
    r2= np.roll(s[:,:,[3]],1)
    r2[r2<0]=0

    if R==1:
        r = r1+r2+np.random.randn(s.shape[0],s.shape[1],1)
    else:
        r = np.concatenate([
            r1+np.random.randn(s.shape[0],s.shape[1],1),
            r2+np.random.randn(s.shape[0],s.shape[1],1)],
        axis=2)

    return r
R=2
response_fit = gen_resp(spectrogram_fit, R=R)
response_test = gen_resp(spectrogram_test, R=R)

############GETTING STARTED###############
###########################
# Example small convolutional neural network (CNN)
# Creating a CNN is just the same as building any other model
# so far, but with a few more layers.
#
# Here we have an example CNN, which we can later fit to our
# data and predict as we have before
###########################
cnn = Model(name=f"{cellid}-Rank3LNSTRF-5Layer")
cnn.add_layers(
    WeightChannels(shape=(18, 1, 3),regularizer='l2'),  # 18 spectral channels->2 composite channels->3rd dimension channel
    FiniteImpulseResponse(shape=(15, 1, 3)),  # 15 taps, 1 spectral channels, 3 filters
    RectifiedLinear(shape=(3,)),  # Takes FIR 3 output filter and applies ReLU function
    WeightChannels(shape=(3, R)),  # Another set of weights to apply
    RectifiedLinear(shape=(R,), no_shift=False, no_offset=False) # A final ReLU applied to our last input
)

###########################
# Initializing our model parameters
#   sample_from_priors(): Randomizes model parameters & fitter options
###########################
cnn = cnn.sample_from_priors()
cnn_can = cnn.copy()

# Basic options to quickly fit our models for tf backend
# NOTE: This will be explored in the next tutorial
options = {'cost_function': 'nmse', 'early_stopping_delay': 50, 'early_stopping_patience': 100,
           'early_stopping_tolerance': 1e-3, 'validation_split': 0,
           'learning_rate': 5e-3, 'epochs': 2000}
options_can = options.copy()
options_can['fit_algorithm']='can'

# We can also see the additional dimension added to our FIR layer,
# compared to how our simpler model is set up
print(f"FIR coefficient shape: {cnn.layers[1].coefficients.shape}")

# Fit our models to some real data provided by Demo
# We use 'tf' backend to improve training speed.
# See the next tutorial for more info
batch_size = None
fitted_cnn = cnn.fit(spectrogram_fit, response_fit, batch_size=batch_size, fitter_options=options, backend='tf')
fitted_cnn_can = cnn_can.fit(spectrogram_fit, response_fit, batch_size=None, fitter_options=options_can, backend='tf')

# Plotting our Models after fits
# fitted_cnn.plot(spectrogram_test, target=response_test)
# fitted_cnn_can.plot(spectrogram_test, target=response_test)

# Now we can predict some new data
pred_cnn = fitted_cnn.predict(spectrogram_test, batch_size=batch_size)
pred_cnn_can = fitted_cnn_can.predict(spectrogram_test, batch_size=batch_size)

print('custom r:', correlation(pred_cnn, response_test))
print('canned r:', correlation(pred_cnn_can, response_test))


# A quick plot of our models pre and post fitting
#visualization.plot_predictions({'ln prediction':pred_ln, 'cnn prediction':pred_cnn}, spectrogram_test, response_test, correlation=True)

# new self-contained LN model
# from nems.models import LN
# ln = LN.LN_STRF(15,18,3)
# ln = ln.fit_LBHB(X=spectrogram_fit[np.newaxis], Y=response_fit[np.newaxis])
# ln.plot_strf()
