import logging
import pickle
from pathlib import Path
import gzip
import numpy as np

import nems.analysis.api
import nems.initializers
from nems.recording import load_recording
import nems.preprocessing as preproc
import nems.uri
from nems.fitters.api import scipy_minimize
from nems.signal import RasterizedSignal

log = logging.getLogger(__name__)

# CONFIGURATION

# figure out data and results paths:
signals_dir = Path(nems.NEMS_PATH) / 'recordings'
modelspecs_dir = Path(nems.NEMS_PATH) / 'modelspecs'

recfile = signals_dir / 'TAR010c.NAT.fs100.ch18.tgz'

rec=load_recording('/Users/svd/python/nems/recordings/TAR010c.NAT.fs100.ch18.tgz')
rec['stim']=rec['stim'].rasterize()
rec['resp']=rec['resp'].rasterize()
rec=preproc.average_away_epoch_occurrences(rec, epoch_regex="^STIM_")

X = rec['stim']._data.copy()
Y = rec['resp']._data.copy()
e = rec['resp'].epochs
e = e[e.name.str.startswith("STIM")]

stimfile = signals_dir / 'TAR010c-NAT-stim.csv'
respfile = signals_dir / 'TAR010c-NAT-resp.csv'
epochfile = signals_dir / 'TAR010c-NAT-epochs.csv'
#np.savetxt(stimfile, X, delimiter=",")
#np.savetxt(respfile, Y, delimiter=",")

e.to_csv(epochfile)
