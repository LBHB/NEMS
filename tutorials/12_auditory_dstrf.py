# NOTE: Unfinished, provided comments are at best only partially correct and missing context  
# TODO: Finish filling out data, confirm its correct, and make sure imports are working properly
import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential
from nems import visualization

# nems_db import module
from nems_lbhb.projects.bignat.bnt_tools import do_bnt_fit, data_subset
from nems.backends import get_backend
from nems.models.dataset import DataSet

# This indicates that our code is interactive, allowing a
# matplotlib backend to show graphs
plt.ion()

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
# DSTRF Data Setup
# To create a DSTRF we need to save our data from individual
# layers. 
# 
###########################

###########################
# do_bnt_fit()
# Creates a model based on given parameters and returns the fitted model and
# datasets from individual layers
#   - sitecount: 
#   - keywordstrub: The layers to be run on our model
#   - fitkw: Model fit keyword arguments
###########################
sitecount = 1
keywordstub = "wc.19x1x3-fir.10x1x3-relu.3.s-wc.3"
fitkw = 'lite.tf.mi1000.lr1e3.t3.lfse'
modelspec, datasets = do_bnt_fit(sitecount, keywordstub, fitkw=fitkw, pc_count=3, save_results=False)

###########################
# data_subset()
# Once we've gotten our model and base dataset for our DSTRF, we need
# to seperate this into relevant information, by creating a subset
#   - data: Our dataset given by earlier model fit
#   - site-set: ???
#   - output_name: ???
###########################
D=10
self=modelspec
backend='tf'
verbose=1
backend_options = {}

stim, resp = data_subset(datasets, list(datasets.keys()), output_name='pca')

###########################
# Get first 10 ___ of our stimulus dataset, and nest this data
# into another set of arrays
###########################
input = stim[0, :D, :]
if False:
    eval_kwargs = {'batch_size': 0}
else:
    input = input[np.newaxis,:,:]
    eval_kwargs = {'batch_size': None}

###########################
# Wrap our new input into a DataSet class, allowing us to
# broadcast as samples, and use as an input for our backend
#   - as_broadcasted_samples(): 
###########################
data = DataSet(
    input, target=None, target_name=None,
    prediction_name=None, **eval_kwargs)

if eval_kwargs.get('batch_size', 0) != 0:
    # Broadcast prior to passing to Backend so that those details
    # only have to be tracked once.
    data = data.as_broadcasted_samples()

_ = self.evaluate(input, use_existing_maps=False, **eval_kwargs)

###########################
# Tensorflow NEMS model
# Create a model with our given backend, in this case 'tf'.
# We use our modified data as the input, and provide needed
# arguments as well.
###########################
backend_class = get_backend(name=backend)
backend_obj = backend_class(
    self, data, verbose=verbose, eval_kwargs=eval_kwargs,
    **backend_options
)

###########################
# Organizing DSTRF for plotting & analysis
# Creating a DSTRF from our input, model,
# stimulus, and channels
###########################
plt.close('all')
t_indexes = [190, 210, 220, 230, 240, 250, 260]
dstrf = np.zeros((len(t_indexes), input.shape[2], D))

out_channel = 2
for i,t in enumerate(t_indexes):
    w = backend_obj.get_jacobian(stim[:1, (t-D):t, :], out_channel)
    dstrf[i, :, :] = w[0, :, :].numpy().T

# Plot and show
f,ax = plt.subplots(2, len(t_indexes))
vmax = np.max(np.abs(dstrf))
for i,t in enumerate(t_indexes):
    ax[0,i].imshow(stim[0, (t-D):t, :].T)
    ax[1,i].imshow(dstrf[i, :, :], vmin=-vmax, vmax=vmax)

## Uncomment if you don't have an interactive backend installed
#plt.show()
