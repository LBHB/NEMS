"""
Functions for computing and analyzing dSTRFs from NEMS model fits
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

def compute_dpcs(dstrf, pc_count=3, norm_mag=False, snr_threshold=5, first_lin=False,
                 as_dict=False):
    """
    Perform PCA on 4D dstrf matrix, separately for each output channel
    dstrf matrix is [output X time X input channel X time lag]
    :param dstrf: dict, np.array
        dict: {input1: dstrf_input1, input2: dstrf_input2, ...} supports multiple inputs
        np.array: [outchan X sample X freq X lag] dstrf matrix for a single input
    :param pc_count:  int
       number of PCs to return for each output channel
    :param norm_mag: bool
       if True, pc_mag returned as fraction variance, otherwise raw varaince
    :param snr_threshold: float
       excloude outlier dstrf samples with high variance: variance(sample)/average_variance > snr_threshold
    :param as_dict: bool
       if True, return results as a dict rather than tuple
    :return: depends on input format:
        as_dict==True: dict with keys pcs, pc_mag, projection
        as_dict==False: tuple: (pcs, pc_mag)
    """

    from sklearn.decomposition import PCA

    if type(dstrf) is not dict:
        # convert to dict format for compatibility
        dstrf = {'input': dstrf}
        
    return_dict = {}
    for input_name, input_dstrf in dstrf.items():
        channel_count = input_dstrf.shape[0]
        s = list(input_dstrf.shape)
        s[1] = pc_count
        dmean = np.zeros([s[0]]+s[2:])
        pcs = np.zeros(s)
        pc_mag = np.zeros((pc_count, channel_count))
        projection = np.zeros((channel_count, input_dstrf.shape[1], pc_count))

        for c in range(channel_count):
            features = np.reshape(input_dstrf[c, :, :, :], (input_dstrf.shape[1], s[2] * s[3]))

            if snr_threshold is not None:
                mean_features = features.mean(axis=0, keepdims=True)
                noise = np.std(features - mean_features, axis=1) / np.std(mean_features)

                if (noise > snr_threshold).sum() > 0:
                    log.info(f"Removed {(noise>snr_threshold).sum()}/{len(features)} noisy dSTRFs for PCA calculation")
                fit_features = features[(noise <= snr_threshold), :]
            else:
                fit_features = features

            if first_lin:
                m = fit_features.mean(axis=0, keepdims=True)
                m = m / np.sqrt((m**2).sum())
                mproj = fit_features @ m.T
                mproj_variance = mproj.var(ddof=1)
                fit_features1 = fit_features - mproj @ m

                mproj_all = features @ m.T
                features1 = features - mproj_all @ m

                #f,ax=plt.subplots(3,5)
                #for i in range(5):
                #    mm=np.max(np.abs(fit_features[i,:]))
                #    ax[0,i].imshow(np.reshape(fit_features[i,:],(s[2],s[3])), vmin=-mm, vmax=mm)
                #    ax[1,i].imshow(np.reshape(m*mproj[i],(s[2],s[3])), vmin=-mm, vmax=mm)
                #    ax[2,i].imshow(np.reshape(fit_features[i,:]-m*mproj[i],(s[2],s[3])), vmin=-mm, vmax=mm)
                pca = PCA(n_components=pc_count-1)
                pca = pca.fit(fit_features1)
                transformed_pca = np.concatenate([mproj_all, pca.transform(features1)], axis=1)
                components = np.concatenate([m,pca.components_], axis=0)

                variance = np.concatenate([[mproj_variance], pca.explained_variance_])
                if norm_mag:
                    variance = variance/variance.sum()
                else:
                    # Do we really want sqrt??
                    variance = variance ** 0.5
            else:
                pca = PCA(n_components=pc_count)
                pca = pca.fit(fit_features)
                transformed_pca = pca.transform(features)
                components = pca.components_
                #_u, _s, _v = np.linalg.svd(d.T @ d)
                # print(d.shape, _v.shape)
                # _s = np.sqrt(_s)

                if norm_mag:
                    variance = np.sqrt(pca.explained_variance_ratio_)
                else:
                    variance = np.sqrt(pca.explained_variance_)
            
            dmean[c] = np.reshape(features.mean(axis=0), [s[2], s[3]])
            pcs[c, :, :, :] = np.reshape(components, [pc_count, s[2], s[3]])
            pc_mag[:, c] = variance[:pc_count]
            #print(projection.shape, transformed_pca.shape)
            projection[c, :, :] = transformed_pca

            # flip sign of first PC so that mean is positive
            #mean_dstrf = input_dstrf[c, :, :, :].mean(axis=0)
            #if np.sum(mean_dstrf * pcs[c, 0, :, :]) < 0:
            #    pcs[c, 0, :, :] = -pcs[c, 0, :, :]
            for oi in range(pc_count):
                if pcs[c, oi].sum()<0:
                    log.info(f"Flipping sign of c={c}, pc={oi}, projection oi=dim2, shape={transformed_pca.shape}")
                    pcs[c, oi] = -pcs[c,oi]
                    projection[c, :, oi] = -projection[c, :, oi]
        return_dict[input_name] = {'pcs': pcs, 'pc_mag': pc_mag, 'projection': projection, 'mean': dmean}

    if as_dict:
        return return_dict
    else:
        return return_dict['input']['pcs'], return_dict['input']['pc_mag']


def dpc_project(dpcs, X):
    pass

