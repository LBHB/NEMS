import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential, RectifiedLinear
from nems import visualization

# More specific options for our model here. You will 
# need TensorFlow installed, see readme install instructions. 
# We also recommend having a GPU set up for these fits
options = {'cost_function': 'squared_error', 'early_stopping_delay': 50, 'early_stopping_patience': 100,
                  'early_stopping_tolerance': 1e-3, 'validation_split': 0,
                  'learning_rate': 5e-3, 'epochs': 2000}

# Setting up Demo Data
training_dict, test_dict = nems.load_demo("TAR010c_data.npz")

spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response']
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response']

########################################################
# Auditory Population Fitting
# A more recent focus has been the fitting of larger populations of inputs.
# Functionally, we are fitting and plotting our data as normal, but the 
# plotted data is impossible to really understand
#
# This type of data will take much longer to process and the resulting data
# will be much more dense than our previous tutorials.
#
# In this example, the given download data 'spectrogram' 
# provides 55 neurons to fit.
########################################################

# Here we can see the dimensions of our input
print(f'The shape of our input data is: {spectrogram_fit.shape}')

# Creating a typical CNN model
cnn = Model(name='Population-CNN')
cnn.add_layers(
    WeightChannels(shape=(18, 1, 30)),
    FiniteImpulseResponse(shape=(15, 1, 30)),
    RectifiedLinear(shape=(30,)),
    WeightChannels(shape=(30, 55)),
    RectifiedLinear(shape=(55,), no_shift=False, no_offset=False),
)
cnn = cnn.sample_from_priors()

# As you can see here, we have many overlapping inputs attempting to fit to our target data
fitted_cnn = cnn.fit(spectrogram_fit, response_fit, fitter_options=options, backend='tf')
visualization.plot_model(fitted_cnn, spectrogram_test, response_test)

# A quick print out of our prediction, stimulus, and response data
# Again all the information here is basically unreadable at the moment
pred_cnn = fitted_cnn.predict(spectrogram_test)
visualization.plot_predictions(pred_cnn, spectrogram_test, response_test)
