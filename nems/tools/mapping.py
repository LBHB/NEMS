"""
Tools for projecting stimuli into AC model space
"""

import os
import re
import json
import inspect
from functools import partialmethod

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from nems.distributions.base import Distribution
from nems.layers.base import Layer, Phi, Parameter
from nems.models.base import Model
from nems.backends.base import FitResults
from nems.tools import json
from nems.preprocessing.spectrogram import gammagram
from nems0.modules.nonlinearity import _dlog

dir_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(dir_path,'demo_data','saved_models')

def dlog(x, offset):
    """
    Log compression helper function
    :param x: input, needs to be >0, works best if x values range approximately (0, 1)
    :param offset: threshold (d = 10**offset). offset compressed for |offset|>2
    :return: y = np.log((x + d) / d)
    """

    # soften effects of more extreme offsets
    inflect = 2

    if isinstance(offset, int):
        offset = np.array([[offset]])

    adjoffset = offset.copy()
    adjoffset[offset > inflect] = inflect + (offset[offset > inflect]-inflect) / 50
    adjoffset[offset < -inflect] = -inflect + (offset[offset < -inflect]+inflect) / 50

    d = 10.0**adjoffset

    return np.log((x + d) / d)


def load_mapping_model(modelspec=None, version=1):

    if modelspec is None:
        modelspec = 'gtgram.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x1x70.g-fir.15x1x70-relu.70-wc.70x1x80-fir.10x1x80-relu.80-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4'

    modelfilepath = os.path.join(model_path,modelspec+'.json')
    modelspec = json.load_model(modelfilepath)
    layers = modelspec.layers._values[:-2]
    new_modelspec = Model(layers=layers, name=modelspec.name, meta=modelspec.meta.copy(),
                          output_name=modelspec.output_name, fs=modelspec.fs)
    new_modelspec.meta['smin'] = np.zeros((18,1))
    new_modelspec.meta['smax'] = np.array([[2.07361315, 1.9316044 , 1.94671749, 2.08861981, 2.0431995 ,
        1.98397283, 2.00941414, 2.03533967, 2.13307407, 2.05509621,
        1.86860145, 2.05164048, 1.91209762, 2.00511489, 2.08149414,
        1.82319729, 1.82759065, 1.86091482]]).T
    new_modelspec.meta['log_compress']=1
    new_modelspec.meta['rasterfs']=100

    return new_modelspec


def project(modelspec, wav=None, w=None, fs=None,
            raw_scale=250, OveralldB=65, verbose=True):

    if (w is not None) & (fs is not None):
        pass
    elif (wavfile is not None):
        fs, w = wavfile.read(wav)

        w = w / np.iinfo(w.dtype).max
        w *= raw_scale
        sf = 10 ** ((80 - OveralldB) / 20)
        w /= sf
    else:
        raise ValueError('required parameters: wavfile or (w,fs)')

    channels = modelspec.layers[0].shape[0]
    rasterfs = modelspec.meta['rasterfs']
    f_min = 200
    f_max = 20000
    window_time = 1 / rasterfs
    hop_time = 1 / rasterfs
    padbins = int(np.ceil((window_time - hop_time) / 2 * fs))
    log_compress = modelspec.meta['log_compress']
    smin = modelspec.meta['smin']
    smax = modelspec.meta['smax']

    s = gammagram(np.pad(w,[padbins, padbins]), fs, window_time, hop_time, channels, f_min, f_max)
    s = _dlog(s, -log_compress)
    s -= smin.T
    s /= (smax-smin).T

    projection = modelspec.predict(s)

    if verbose:
        f = plt.figure()
        ax = [f.add_subplot(4, 1, 1), f.add_subplot(4, 1, 2), f.add_subplot(2, 1, 2)]
        t = np.arange(len(w)) / fs
        ax[0].plot(t, w)
        ax[0].set_xlim([t[0], t[-1]])
        ax[0].set_xticklabels([])
        ts = s.shape[0] / rasterfs
        im = ax[1].imshow(s.T, origin='lower', extent=[0, ts, -0.5, s.shape[1] + 0.5])
        ax[1].set_xticklabels([])
        ax[2].imshow(projection.T, origin='lower', interpolation='none',
                     extent=[0, ts, -0.5, projection.shape[1] + 0.5])
        if wav is not None:
            ax[0].set_title(os.path.basename(wav))

    return projection

