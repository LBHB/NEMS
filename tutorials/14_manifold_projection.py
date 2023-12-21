"""
NEMS demo 14: Manifold Projection

Project a waveform into the manifold defined as the penultimate layer of a population CNN
encoding model fit to a large A1 dataset. The resulting projection will provide a
cortical representation of the sound.

Script assumes that interactive matplotlib plotting is enabled.
"""
import matplotlib.pyplot as plt

import nems
from nems.tools import mapping

# load the model used for projection. If it doesn't exist, download it from AWS S3 server
modelname='CNN-32'  # could also be CNN-18 for model that uses 18 channel spectrogram
modelspec = mapping.load_mapping_model(name=modelname)

# any wav file can be loaded here. All we need is the waveform and sampling rate (fs)
wavfile = 'woman_speaking.wav'
fs, w = nems.load_demo(wavfile)

# One important consideration is that the waveform amplitude is scaled to a range that
# makes sense to the model. Here we adjust to 65 dB RMS, which is the level of sounds
# that were used to fit the model.
OveralldB = 65
w = w * 3.519 / w.std()  # normalize to 80 dB RMS
# rescale to OveralldB
sf = 10 ** ((80 - OveralldB) / 20)
w /= sf

# if verbose, mapping.project will generate a figure and return the handle in a tuple.
projection, f = mapping.project(modelspec, w=w, fs=fs, verbose=True)

# if plot does not appear, run:
# plt.show()