#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:12:37 2018

@author: svd
"""
import numpy as np

import nems0.signal as signal
import nems0.recording as recording
import nems0.modules.stp as stp

def stp_magnitude(tau, u, u2=None, tau2=None, urat=0.5, fs=100, A=0.5, quick_eval=False):
    """ compute effect of stp (tau,u) on a dummy signal and computer effect magnitude
    """
    c = len(tau)
    seg = np.int(fs * 0.05)
    pred = np.concatenate([np.zeros([c, seg * 2]), np.ones([c, seg * 4]) * A/2,
                           np.zeros([c, seg * 4]), np.ones([c, seg]) * A,
                           np.zeros([c, seg]), np.ones([c, seg]) * A,
                           np.zeros([c, seg]), np.ones([c, seg]) * A,
                           np.zeros([c, seg * 2])], axis=1)

    kwargs = {
        'data': pred,
        'name': 'pred',
        'recording': 'rec',
        'chans': ['chan' + str(n) for n in range(c)],
        'fs': fs,
        'meta': {},
    }
    pred = signal.RasterizedSignal(**kwargs)
    r = recording.Recording({'pred': pred})
    if tau2 is None:
        r = stp.short_term_plasticity(r, 'pred', 'pred_out', u=u, tau=tau, quick_eval=quick_eval)
    else:
        r = stp.short_term_plasticity2(r, 'pred', 'pred_out', u=u, tau=tau, u2=u2, tau2=tau2, urat=urat,
                                       quick_eval=quick_eval)

    pred_out = r[0]

    stp_mag = (np.sum(pred.as_continuous()-pred_out.as_continuous(),axis=1) /
               np.sum(pred.as_continuous()))

    return (stp_mag, pred, pred_out)
