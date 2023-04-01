import numpy as np
import logging
log = logging.getLogger(__name__)

def reverse_correlation(rec, modelspec, input_name, tol=1e-2):
    """
    return a new modelspec with updated phi for fir coefficients and level shift
    """

    # figure out which modules are the fir filter and the level adjustment
    fns = []
    for m in modelspec:
        fns.append(m['fn'])
    fns = np.array(fns)
    ind1 = np.argwhere([True if 'nems0.modules.fir' in f else False for f in fns])[0][0]
    ind2 = np.argwhere([True if 'nems0.modules.levelshift' in f else False for f in fns])[0][0]

    # apply mask, if it exists
    if 'mask' in rec.signals.keys():
        log.info("Data len pre-mask: %d", rec['mask'].shape[1])
        rec = rec.apply_mask()
        log.info("Data len post-mask: %d", rec['mask'].shape[1])

    # determine if doing forward or reverse model
    if input_name == 'stim':
        sig1 = 'stim'
        sig2 = 'resp'
    elif input_name == 'resp':
        sig1 = 'resp'
        sig2 = 'stim'

    # define "R" and "S" based on if doing forward or reverse model
    # mean center S.
    R = rec[sig1].as_continuous()
    S = rec[sig2].as_continuous()
    S_mean = S.mean(axis=-1)[:, np.newaxis]
    S = S - S_mean

    # create delay lines
    ndelays = modelspec[ind1]['phi']['coefficients'].shape[1]
    for i in range(0, ndelays):
        if input_name == 'stim':
            roll = i
        else:
            roll = -i
        if i == 0:
            r_roll = np.roll(R, roll, axis=-1)
            R_delay = r_roll
        else:
            r_roll = np.roll(R, roll, axis=-1)
            R_delay = np.concatenate((R_delay, r_roll), axis=0)

    # perform normalized reverse correlation
    Rs = np.matmul(S, R_delay.T)
    CC = np.matmul(R_delay, R_delay.T)
    h = np.matmul(np.linalg.pinv(CC, tol), Rs.T)

    # predict the new stimulus
    S_hat = np.matmul(h.T, R_delay) + S_mean

    # reshape "phis" into the form that modelspecs expects and save
    d1 = R.shape[0]
    d2 = S.shape[0]

    if input_name == 'resp':
        # if reverse model, need to flip fir coefficients for the fir module
        h_ = h.reshape(ndelays, d1, d2)
        h_ = h_[::-1, :, :]
        h = h_.copy()
    else:
        h = h.reshape(ndelays, d1, d2)

    h = h.transpose(0, 2, 1).reshape(ndelays, d1*d2).T

    modelspec[ind1]['phi']['coefficients'] = h
    modelspec[ind2]['phi']['level'] = S_mean

    # add a field holding the prediction, for testing purposes
    modelspec[0]['pred'] = S_hat

    return modelspec
