import matplotlib.pyplot as plt
import numpy as np

import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential
from nems import visualization

from nems_lbhb.projects.bignat.bnt_tools import do_bnt_fit, data_subset
from nems.backends import get_backend
from nems.models.dataset import DataSet

sitecount = 1
keywordstub = "wc.19x1x3-fir.10x1x3-relu.3.s-wc.3"
fitkw = 'lite.tf.mi1000.lr1e3.t3.lfse'
modelspec, datasets = do_bnt_fit(sitecount, keywordstub, fitkw=fitkw, pc_count=3, save_results=False)

D=10
self=modelspec
backend='tf'
verbose=1
backend_options = {}

stim, resp = data_subset(datasets, list(datasets.keys()), output_name='pca')

input = stim[0, :D, :]
if False:
    eval_kwargs = {'batch_size': 0}
else:
    input = input[np.newaxis,:,:]
    eval_kwargs = {'batch_size': None}


data = DataSet(
    input, target=None, target_name=None,
    prediction_name=None, **eval_kwargs)

if eval_kwargs.get('batch_size', 0) != 0:
    # Broadcast prior to passing to Backend so that those details
    # only have to be tracked once.
    data = data.as_broadcasted_samples()

_ = self.evaluate(input, use_existing_maps=False, **eval_kwargs)


backend_class = get_backend(name=backend)
backend_obj = backend_class(
    self, data, verbose=verbose, eval_kwargs=eval_kwargs,
    **backend_options
)

#import tensorflow as tf

plt.close('all')
t_indexes = [190, 210, 220, 230, 240, 250, 260]
dstrf = np.zeros((len(t_indexes), input.shape[2], D))

out_channel=2
for i,t in enumerate(t_indexes):
    w = backend_obj.get_jacobian(stim[:1, (t-D):t, :], out_channel)
    dstrf[i, :, :] = w[0, :, :].numpy().T

f,ax = plt.subplots(2, len(t_indexes))
vmax = np.max(np.abs(dstrf))
for i,t in enumerate(t_indexes):
    ax[0,i].imshow(stim[0, (t-D):t, :].T)
    ax[1,i].imshow(dstrf[i, :, :], vmin=-vmax, vmax=vmax)


