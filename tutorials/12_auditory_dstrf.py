import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, RectifiedLinear
from nems import visualization
from nems.tools.dstrf import compute_dpcs

# nems_db import module
from nems.backends import get_backend
from nems.models.dataset import DataSet

# Importing Demo Data
nems.download_demo()
training_dict, test_dict = nems.load_demo("TAR010c_data.npz")

cid=29 # Picking our cell to pull data from
cellid = training_dict['cellid'][cid]
spectrogram_fit = training_dict['spectrogram']
response_fit = training_dict['response'][:,[cid]]
spectrogram_test = test_dict['spectrogram']
response_test = test_dict['response'][:,[cid]]

#  dstrfs from LN and CNN

# Basic Linear Model
ln = Model()
ln.add_layers(
    WeightChannels(shape=(18, 3)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(10, 3)),  # 15 taps, 1 spectral channels
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False)           # static nonlinearity, 1 output
)
ln.name = f"LN_Model"

# A None linear CNN model
cnn = Model()
cnn.add_layers(
    WeightChannels(shape=(18, 1, 3)),  # 18 spectral channels->2 composite channels->3rd dimension channel
    FiniteImpulseResponse(shape=(15, 1, 3)),  # 15 taps, 1 spectral channels, 3 filters
    RectifiedLinear(shape=(3,)), # Takes FIR 3 output filter and applies ReLU function
    WeightChannels(shape=(3, 1)), # Another set of weights to apply
    RectifiedLinear(shape=(1,), no_shift=False, no_offset=False) # A final ReLU applied to our last input
)
cnn.name = f"CNN_Model"

############GETTING STARTED###############
########################################################
# Auditory DSTRF
# Creating locally linear data to evaluate an strf model
# 
#   - Create a model and fit it for strf data
#   - Save and provide a dataset containing layer data
#   - Get the jacobian output of our datasets
#   - Visualize our individual layers and data
########################################################
###########################
# DSTRF
# DSTRF is the process of creating locally
# linear representations of a model. By mapping 
# out sections of our model we can try and view how 
# layers are changing during predictions
# 
# dstrf
#   Stim: Stimulus to predict our model onto
#   D: The memory of DSTRF (In time bins)
#   out_channels: Output channels to use when generating DSTF
#   t_indexes: Time samples
#   backend: Set the backend used, currently support 'tf'
#   reset_backend: Forces new initialization of backend
#   
###########################

# When creating a DSTRF, we are taking a non-linear set of data
# and finding linear representation for parts of that data. If a
# Linear model is used, you will see that our DSTRF values will not
# change

# Fitting both our models to given fit data from our demo data import
fitted_ln = ln.fit(spectrogram_fit, response_fit, backend='tf')
fitted_cnn = cnn.fit(spectrogram_fit, response_fit, backend='tf')

ln_dstrf = fitted_ln.dstrf(spectrogram_fit, D=5, reset_backend=True)

# Note that this linear data does not show any changes between time steps
visualization.plot_dstrf(ln_dstrf['input'])

# In this example we can see how different options may provide changes
# and how we can interact with that data
# .shape[0]: The number of output channels from layers we're keeping track of
# .shape[1]: Time indexes are to decide when we look into the model and save that information
# .shape[2:4]: Output data is the actual output of a given layer at a time interval

print(f'''
      DSTRF Shape: {ln_dstrf['input'].shape} \n 
      Output Channels: {ln_dstrf['input'].shape[0]} \n 
      Time Indexes: {ln_dstrf['input'].shape[1]} \n 
      Output Data: {ln_dstrf['input'].shape[2:4]}''')
cnn_dstrf = fitted_cnn.dstrf(spectrogram_fit, D=15, reset_backend=True)



############ADVANCED###############
###########################
# Principle Components Analysis
# Creating a DSTRF can help us create locally
# linear sets of data at points in time,
# but the resulting data can often be of high
# dimensionality and hard to interpret.
#
# Using Principle Components we can reduce dimensionality,
# allowing us to better interpret our DSTRF's
#
# nems.tools.dstf.compute_dpcs(dstrf, ...)
#   dsrf: Given DSTRF to gather PC's from
#   pc_count: The number of principle components to find
#   norm_mag: If true, returns normalize variances
#   snr_threshold: A given Signal-Noise_Ratio to filter noisy data
#
#   Returns: (pca, pc_mag, projections)
###########################
pca = compute_dpcs(cnn_dstrf)

# If given multiple inputs; pca['input_key']['pcs'].shape would be equivalent
print(f'''
      Shape of our PCAs: {pca["pcs"].shape} \n
      PC Mag: {pca["pc_mag"]} \n 
      PCA projection: {pca["projection"]}''')


###########################
# Visualizing DSTRF's
# By viewing a heatmap of our DSTRF, we can view our data
# at each time step, allowing us to start to intepret new
# information.
###########################

###########################
# plot_dstrf
# Provide a dstrf and a list of heatmaps will be plotted out
###########################

# Traditional heatmap of our multidimensional set of data.
# As you can see, this plot contains a lot of unneeded information
visualization.plot_dstrf(cnn_dstrf['input'])

# Using our Principle Components, we have a much easier time
# interpreting the same data
visualization.plot_dstrf(pca['pcs'], title="PC DSTRF's")

###########################
# plot_dpca
# Directly creates a set of dstrf/pca graphs from the model 
# and spectrogram data itself. 
#   t_skip: How many indexes to skip at each point when creating a full dstrf
#           from our entire input
#   t_len: Max length of our input to reduce overall size of our input
#          when processing entire dstrf on input
#   title: Title of figure
#   xunits: units to be shown on x_label
#   **dstrf_kwargs: Keyword arguments to be passed to our dstrf calls
###########################

# This plot computes a DSTRF based on every single existing 
# timestep on the length of your input. This can take time...
visualization.plot_dpca(fitted_cnn, spectrogram_fit, t_skip=20, t_len=6000, D=15, reset_backened=True)
