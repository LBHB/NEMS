import numpy as np

from nems0.modules.fir import per_channel

def nonparametric(rec, coefficients, i='pred', o='pred'):
    fn = lambda x: per_channel(x, coefficients)
    return [rec[i].transform(fn, o)]