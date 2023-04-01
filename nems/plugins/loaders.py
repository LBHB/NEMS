import numpy as np
import pickle

import logging
from pathlib import Path
import gzip
import numpy as np
import pandas as pd

import nems0.analysis.api
import nems0.initializers
import nems0.recording as recording
import nems0.preprocessing as preproc
import nems0.uri
from nems0.fitters.api import scipy_minimize
#from nems0.tf.cnnlink_new import fit_tf, fit_tf_init
from nems0.registry import xform, xmodule

from nems0.signal import RasterizedSignal

log = logging.getLogger(__name__)

# CONFIGURATION

# figure out data and results paths:
signals_dir = Path(nems0.NEMS_PATH) / 'recordings'


@xform()
def vsload(loadkey, cellid=None, batch=None, **kwargs):
    """
    keyword for load_sadagopan
    """
    print(cellid)
    log.info('loadkey=%s',loadkey)

    ops = loadkey.split(".")

    # updates some some defaults
    recname = "_".join(cellid.split("_")[:2])
    options = {'fs': 50, 'batch': batch,
               'cellid': cellid, 'recname': recname}

    for op in ops:

        if op.startswith('fs'):
            options['rasterfs'] = int(op[2:])
        elif op.startswith('ch'):
            options['chancount'] = int(op[2:])

        elif op=='pup':
            options.update({'pupil': True, 'rem': 1})

    xfspec = [['nems0.plugins.loaders.load_sadagopan', options]]
    return xfspec


def load_sadagopan(cellid='MS_u0004_f0025', recname='MS_u0004',
                   stimfile=None, respfile=None, epochsfile=None,
                   fs=50, channel_num=0, **context):
    """
    example file from Sadagopan lab
    """

    if stimfile is None:
        stimfile = signals_dir / (cellid+'_stim.csv.gz')
    if respfile is None:
        respfile = signals_dir / (cellid+'_resp.csv.gz')
    if epochsfile is None:
        epochsfile = signals_dir / (cellid+'_epochs.csv')

    X=np.loadtxt(gzip.open(stimfile, mode='rb'), delimiter=",", skiprows=0)
    Y=np.loadtxt(gzip.open(respfile, mode='rb'), delimiter=",", skiprows=0)
    # get list of stimuli with start and stop times (in sec)
    epochs = pd.read_csv(epochsfile)

    # create NEMS-format recording objects from the raw data
    resp = RasterizedSignal(fs, Y, 'resp', recname, chans=[cellid], epochs=epochs.loc[:])
    stim = RasterizedSignal(fs, X, 'stim', recname, epochs=epochs.loc[:])

    # create the recording object from the signals
    signals = {'resp': resp, 'stim': stim}
    rec = recording.Recording(signals)

    return {'rec': rec}
