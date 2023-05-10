"""
Functions for computing and analyzing dSTRFs from NEMS model fits
"""

import numpy as np

def compute_dpcs(dstrf, pc_count=3):
    """
    Perform PCA on 4D dstrf matrix, separately for each output channel
    dstrf matrix is [output X time X input channel X time lag]
    :param dstrf:
    :param pc_count:  number of PCs to return for each output channel
    :return: pcs, pc_mag
    """
    # from sklearn.decomposition import PCA

    channel_count = dstrf.shape[0]
    s = list(dstrf.shape)
    s[1] = pc_count
    pcs = np.zeros(s)
    pc_mag = np.zeros((pc_count, channel_count))

    for c in range(channel_count):
        d = np.reshape(dstrf[c, :, :, :], (dstrf.shape[1], s[2] * s[3]))
        # d -= d.mean(axis=0, keepdims=0)

        _u, _s, _v = np.linalg.svd(d.T @ d)
        # print(d.shape, _v.shape)
        # _s = np.sqrt(_s)
        _s /= np.sum(_s[:pc_count])

        pcs[c, :, :, :] = np.reshape(_v[:pc_count, :], [pc_count, s[2], s[3]])

        # flip sign of first PC so that mean is positive
        mdstrf = dstrf[c, :, :, :].mean(axis=0)
        if np.sum(mdstrf * pcs[c, 0, :, :]) < 0:
            pcs[c, 0, :, :] = -pcs[c, 0, :, :]
            # print(f"{c} adjusted to {np.sum(mdstrf * pcs[0,:,:,c])}")
        pc_mag[:, c] = _s[:pc_count]

    return pcs, pc_mag


def dpc_project(dpcs, X):


