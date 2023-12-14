"""
Tools for projecting stimuli into AC model space
"""

import os
import re
import json
import inspect
from functools import partialmethod

import numpy as np
import scipy.signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

from nems.distributions.base import Distribution
from nems.layers.base import Layer, Phi, Parameter
from nems.models.base import Model
from nems.backends.base import FitResults
from nems.tools import json
from nems.preprocessing.spectrogram import gammagram
#from nems0.modules.nonlinearity import _dlog
from nems.tools.demo_data.file_management import download_models, model_files

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


def load_mapping_model(name=None, modelname=None, version=1):
    """
    :param name: short name for model. Currently select from ['CNN-18', 'CNN-32']
                 CNN-xx where xx is the number of spectrogram channels on front end
    :param modelname: long-form name of model == filename of json-encoded modelspec
                      that can be loaded with nems.tools.json.load_model()
    :param version: not used (forward compatibility)
    :return: NEMS modelspec with some extra meta-parameters attached that will
             allow projection of arbitrary waveforms into the semi-final (bottleneck)
             layer of the model
    """
    if (name is None) and (modelname is None):
        name='CNN-32'
        # maps to 'gtgram.fs100.ch32.pop-loadpop-norm.l1-popev_wc.32x1x70.g-fir.15x1x70-relu.70.o.s-wc.70x1x90-fir.10x1x90-relu.90.o.s-wc.90x120-relu.120.o.s-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4.json'

    if modelname is None:
        modelname = model_files[name]

    modelfilepath = os.path.join(model_path, modelname)
    
    if os.path.exists(modelfilepath) == False:
        # download from S3
        print(f"{modelfilepath} not found. downloading")
        download_models(modelname=modelname)
    print(f"Loading {modelfilepath}")
    if os.path.exists(modelfilepath) == False:
        raise ValueError(f"Pre-computed model {modelname} not in saved_model path.")
        
    modelspec = json.load_model(modelfilepath)
    layers = modelspec.layers._values[:-2]
    new_modelspec = Model(layers=layers, name=modelspec.name, meta=modelspec.meta.copy(),
                          output_name=modelspec.output_name, fs=modelspec.fs)

    channels = new_modelspec.layers[0].shape[0]
    new_modelspec.meta['smin'] = np.zeros((channels, 1))
    if channels == 18:
        new_modelspec.meta['smax'] = np.array([[2.07361315, 1.9316044 , 1.94671749, 2.08861981, 2.0431995 ,
            1.98397283, 2.00941414, 2.03533967, 2.13307407, 2.05509621,
            1.86860145, 2.05164048, 1.91209762, 2.00511489, 2.08149414,
            1.82319729, 1.82759065, 1.86091482]]).T
    elif channels == 32:
        new_modelspec.meta['smax'] = np.array([
            [0.17319407, 0.17128999, 0.18907563, 0.1996132, 0.19443185,
             0.18742472, 0.17706056, 0.18663591, 0.17925323, 0.19462644,
             0.17475645, 0.17967997, 0.17574365, 0.19425933, 0.17833606,
             0.17270589, 0.17245491, 0.18347096, 0.17326807, 0.15476016,
             0.16378335, 0.15130896, 0.15690385, 0.14773204, 0.13089093,
             0.11983025, 0.10506714, 0.09612146, 0.07826547, 0.07168109,
             0.0430525, 0.06475942]]).T
    if 'norm.l1' in modelname:
        new_modelspec.meta['log_compress']=1
    else:
        new_modelspec.meta['log_compress']=0

    if 'fs200' in modelname:
        new_modelspec.meta['rasterfs'] = 200
    elif 'fs100' in modelname:
        new_modelspec.meta['rasterfs'] = 100
    elif 'fs50' in modelname:
        new_modelspec.meta['rasterfs'] = 50
    else:
        raise ValueError('unsupported fs')

    return new_modelspec


def project(modelspec, wavfilename=None, w=None, fs=None,
            raw_scale=250, OveralldB=65, verbose=True):
    """
    Use modelspec to project a wavefrom (filename wav or vector w) into AC model space. Note that modelspec has to be a special
    'decapitated' model. Ie, a NEMS model fit to a big dataset and with the last, neuron-specific, layers chopped off. Typically
    these are the last two layers (linear mapping from AC space to neurons and output NL).
    See mapping.load_mapping model for loading a NEMS model and perform the necessary chopping (also set some model metadata to
    appropriately pre-process the data).
    :param modelspec: NEMS-format modelspec, with some extra meta-parameters added by load_mapping_model()
    :param wavfilename:
    :param w:
    :param fs:
    :param raw_scale:
    :param OveralldB:
    :param verbose:
    :return: projection [, f]
        T x channel projection of input wav. Optional second output is
        figure handle if verbose=True
    """
    f_min = 200
    f_max = 20000
    if (w is not None) & (fs is not None):
        pass
    elif (wavfilename is not None):
        fs, w = wavfile.read(wavfilename)
        if fs<f_max*2:
            stimlen=len(w)/fs
            finalsamples = int(stimlen*fs_out)
            print(f"Upsampling from {fs} to {fs_out} ({len(w)} to {finalsamples} samples)")
            w = scipy.signal.resample(w, finalsamples)
        w = w / np.iinfo(w.dtype).max
        if raw_scale is not None:
            w *= raw_scale
        else:
            # scale to +/- 5 sin wav = 80 dB RMS
            w *= 3.519 / np.nanstd(w)
        # adjust to stimulus RMS
        sf = 10 ** ((80 - OveralldB) / 20)
        w /= sf
    else:
        raise ValueError('required parameters: wavfile or (w,fs)')

    channels = modelspec.layers[0].shape[0]
    rasterfs = modelspec.meta['rasterfs']
    window_time = 1 / rasterfs
    hop_time = 1 / rasterfs
    padbins = int(np.ceil((window_time - hop_time) / 2 * fs))
    log_compress = modelspec.meta['log_compress']
    smin = modelspec.meta['smin']
    smax = modelspec.meta['smax']

    s = gammagram(np.pad(w,[padbins, padbins]), fs, window_time, hop_time, channels, f_min, f_max)
    s = dlog(s, -log_compress)
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
        ax[0].set_ylabel('Waveform')
        ts = s.shape[0] / rasterfs
        im = ax[1].imshow(s.T, origin='lower', extent=[0, ts, -0.5, s.shape[1] + 0.5],
                          cmap='gray_r', aspect='auto')
        ax[1].set_xticklabels([])
        ax[1].set_ylabel('Spectrogram')
        ax[2].imshow(projection.T, origin='lower', interpolation='none',
                     extent=[0, ts, -0.5, projection.shape[1] + 0.5], aspect='auto')
        ax[2].set_ylabel('Model output')
        ax[2].set_xlabel('Time (s)')
        if wavfilename is not None:
            ax[0].set_title(os.path.basename(wavfilename))

        return projection, f
    else:
        return projection

def spectrogram(wav=None, channels=18, rasterfs=100, w=None, fs=None,
            log_compress=0, raw_scale=250, OveralldB=65, verbose=True):

    if (w is not None) & (fs is not None):
        pass
    elif (wav is not None):
        fs, w = wavfile.read(wav)

        w = w / np.iinfo(w.dtype).max
        if raw_scale is not None:
            w *= raw_scale
        else:
            # scale to +/- 5 sin wav = 80 dB RMS
            w *= 3.519 / np.nanstd(w)
        # adjust to stimulus RMS
        sf = 10 ** ((80 - OveralldB) / 20)
        w /= sf
    else:
        raise ValueError('required parameters: wavfile or (w,fs)')

    f_min = 100
    f_max = 10000
    window_time = 1 / rasterfs
    hop_time = 1 / rasterfs
    padbins = int(np.ceil((window_time - hop_time) / 2 * fs))

    s = gammagram(np.pad(w,[padbins, padbins]), fs, window_time, hop_time, channels, f_min, f_max)
    s = dlog(s, -log_compress)

    if verbose:
        f = plt.figure(figsize=(5,2.5))
        ax = [f.add_subplot(2, 1, 1), f.add_subplot(2, 1, 2)]
        t = np.arange(len(w)) / fs
        ax[0].plot(t, w, 'k', lw=0.5)
        ax[0].set_xlim([t[0], t[-1]])
        ax[0].set_xticklabels([])
        ax[0].set_axis_off()

        ts = s.shape[0] / rasterfs
        im = ax[1].imshow(s.T, origin='lower', extent=[0, ts, -0.5, s.shape[1] + 0.5],
                          cmap='gray_r')
        yi = np.arange(0,channels,10)
        sf = np.log2(f_max/f_min)
        ys=[str(int(np.round(2**(y/channels * sf)*f_min/10,0)*10)) for y in yi]
        ax[1].set_yticks(yi,ys)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Frequency (Hz)')

        if wav is not None:
            ax[0].set_title(os.path.basename(wav))
        plt.tight_layout()
        return f
