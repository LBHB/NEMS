import numpy as np
import pickle

def dummy_loader(N=1000, **context):
    """"DUMMY LOADER FUNCTION FOR TESTING xforms.load_wrapper"""

    cellid = "DUMMY01"
    epochs = None
    X = np.zeros([1, 1000])
    X[:, 40] = 1
    X[:, 60] = 1
    X[:, 100] = 2
    X[:, 500] = 2
    X[:, 700] = 1

    Y = X * 2 + 1
    Y[:, 20] = 10
    Y[:, 200] = 10
    Y[:, 600] = 10
    Y[:, 950] = 10

    res = {}
    res['fs'] = 100
    res['resp'] = Y
    res['stim'] = X
    res['epochs'] = epochs

    res['resp_labels'] = [cellid]
    res['stim_labels'] = ['STIM']

    return res


def demo_loader(datafile=None, **context):

    # load stim and resp matrices from pkl files
    if datafile is None:
        datafile = nems0.get_setting('NEMS_RECORDINGS_DIR') + '/TAR010c-18-1.pkl'

    with open(datafile, 'rb') as f:  # Python 3: open(..., 'rb')
        cellid, recname, fs, X, Y, epochs = pickle.load(f)

    res = {}
    res['fs'] = fs
    res['resp'] = Y
    res['stim'] = X
    # res['state'] = State_matrix
    res['epochs'] = epochs

    res['resp_labels'] = [cellid]
    stim_labels = np.round(np.exp(np.linspace(np.log(2), np.log(20), res['stim'].shape[0])), 1)
    res['stim_labels'] = [str(s) for s in stim_labels]

    return res


def load_polley_data(respfile=None, stimfile=None, exptid="RECORDING", channel_num=0, **context):

    # 2p data from Polley Lab at EPL
    if respfile is None:
        respfile = '/Users/svd/data/data_nems_2p/neurons.csv'
    if stimfile is None:
        stimfile = '/Users/svd/data/data_nems_2p/stim_spectrogram.csv'

    res = {}
    res['fs'] = 30
    res['resp'] = np.genfromtxt(respfile, delimiter=',')
    res['stim'] = np.genfromtxt(stimfile, delimiter=',')

    neuron_count, T = res['resp'].shape
    stim_labels = np.round(np.exp(np.linspace(np.log(4), np.log(64), res['stim'].shape[0])), 1)
    res['stim_labels'] = [str(s) for s in stim_labels]
    res['resp_labels'] = ["{}-{:03d}".format(exptid, i) for i in range(neuron_count)]

    return res
