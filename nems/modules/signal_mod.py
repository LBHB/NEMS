"""
modules/signal_mod.py

functions for modifying signals (merging, averaging, subsampling)
"""

import numpy as np
#import nems0.signal as signal
#import nems0.preprocessing as preproc


#def make_state_signal(rec, signals_in=['pupil'], signals_permute=[], o='state'):
#    """
#    DEPRECATED? SHOULD BE IN PREPROCSSING LIBRARIES
#    """
#    x = np.ones([1,rec[signals_in[0]]._matrix.shape[1]])  # Much faster; TODO: Test if throws warnings
#    ones_sig = rec[signals_in[0]]._modified_copy(x)
#    ones_sig.name="baseline"
#
#    state=signal.Signal.concatenate_channels([ones_sig]+[rec[x] for x in signals_in])
#
#    # TODO support for signals_permute
#    if len(signals_permute):
#        raise ValueError("singals_permute not yet supported")
#
#    state.name = o
#
#    return [state]


#def average_sig(rec, i='resp', o='resp'):
#
#    return [preproc.generate_average_sig(rec[i], new_signalname=o,
#            epoch_regex='^STIM_')]


def replicate_channels(rec, i='pred', o='pred', repcount=2):
    """
    replicate stim/pred channels for subsequent state-dependent modules
    """
    fn = lambda x: np.tile(x,(repcount,1))

    return [rec[i].transform(fn, o)]


def _merge_states(x, state):
    """
    inputs
       x - N x T matrix,
       s - 1 X T matrix with integer values 0 ... N-1
    """
    res = np.zeros_like(x[:1, :])
    res.fill(np.nan)
    # print(state.shape)
    for i in range(x.shape[0]):
        res[state[-1:, :] == i] = x[i, state[-1, :] == i]
    return res


def merge_channels(rec, i='pred', o='pred', s='state'):
    """
    complements replicate_channels. merge separate channels based on
    which state they represent
    """
    fn = lambda x: _merge_states(x, rec[s].as_continuous())
    return [rec[i].transform(fn, o)]
