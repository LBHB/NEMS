"""
Functions for computing and analyzing dSTRFs from NEMS model fits
"""

import numpy as np
import logging

from sklearn.decomposition import PCA

from nems.layers import FiniteImpulseResponse

log = logging.getLogger(__name__)

def compute_dpcs(dstrf, pc_count=3, norm_mag=False, snr_threshold=5):
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

        if snr_threshold is not None:
            md = d.mean(axis=0, keepdims=True)
            e = np.std(d - md, axis=1) / np.std(md)

            if (e > snr_threshold).sum() > 0:
                log.info(f"Removed {(e>snr_threshold).sum()}/{len(d)} noisy dSTRFs for PCA calculation")
            d = d[(e <= snr_threshold), :]

        # d -= d.mean(axis=0, keepdims=0)

        pca = PCA(n_components=pc_count)
        _u = pca.fit_transform(d)
        _v = pca.components_
        #_u, _s, _v = np.linalg.svd(d.T @ d)
        # print(d.shape, _v.shape)
        # _s = np.sqrt(_s)
        if norm_mag:
            _s = np.sqrt(pca.explained_variance_ratio_)
        else:
            _s = np.sqrt(pca.explained_variance_)
        #import pdb; pdb.set_trace()
        pcs[c, :, :, :] = np.reshape(_v, [pc_count, s[2], s[3]])

        # flip sign of first PC so that mean is positive
        mdstrf = dstrf[c, :, :, :].mean(axis=0)
        if np.sum(mdstrf * pcs[c, 0, :, :]) < 0:
            pcs[c, 0, :, :] = -pcs[c, 0, :, :]
            # print(f"{c} adjusted to {np.sum(mdstrf * pcs[0,:,:,c])}")
        pc_mag[:, c] = _s[:pc_count]

    return pcs, pc_mag


def dpc_project(dpcs, X):
    pass

