import numpy as np
import nems0.utils
import logging

log = logging.getLogger(__name__)


def mse(result, pred_name='pred', resp_name='resp'):
    '''
    Given the evaluated data, return the mean squared error

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    pred_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording

    Returns
    -------
    mse : float
        Mean-squared difference between the prediction and response.

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> error = mse(result, 'pred', 'resp')

    Note
    ----
    This function is written to be compatible with both numeric (i.e., Numpy)
    and symbolic (i.e., Theano, TensorFlow) computation. Please do not edit
    unless you know what you're doing. (@bburan TODO: Is this still true?)
    '''
    pred = result[pred_name].as_continuous()
    resp = result[resp_name].as_continuous()
    squared_errors = (pred-resp)**2
    return np.nanmean(squared_errors)


def nmse(result, pred_name='pred', resp_name='resp'):
    '''
    Same as MSE, but normalized by the std of the resp.
    Because it is more computationally expensive than MSE but is otherwise
    equivalent, we suggest using the MSE for fitting and use this as a
    post-fit performance metric only.
    '''
    X1 = result[pred_name].as_continuous()
    X2 = result[resp_name].as_continuous()

    #keepidx = np.isfinite(X1) & np.isfinite(X2)
    #if np.all(~keepidx):
    #    log.debug("All values were NaN or inf in pred and resp")
    #    return 1
    #X1 = X1[keepidx]
    #X2 = X2[keepidx]
    respstd = np.nanstd(X2)
    squared_errors = (X1-X2)**2
    mse = np.sqrt(np.nanmean(squared_errors))

    if respstd == 0:
        return 1
    else:
        return mse / respstd


def j_nmse(result, pred_name='pred', resp_name='resp', njacks=20):
    '''
    Jackknifed estimate of mean and SE on normalized MSE

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    pred_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording

    Returns
    -------
    mse : float
        Correlation coefficient between the prediction and response.
    se_mse : float
        Standard error on nmse, calculated by jackknife (Efron &
        Tibshirani 1986)

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> mse,se_mse = j_nmse(result, 'pred', 'resp', njacks=10)

    Note
    ----
    This function is written to be compatible with both numeric (i.e., Numpy)
    and symbolic (i.e., Theano, TensorFlow) computation. Please do not edit
    unless you know what you're doing. (@bburan TODO: Is this still true?)
    '''

    predmat = result[pred_name].as_continuous()
    respmat = result[resp_name].as_continuous()

    channel_count = predmat.shape[0]
    mse = np.zeros(channel_count)
    se_mse = np.zeros(channel_count)

    for i in range(channel_count):
        pred = predmat[i, :]
        resp = respmat[i, :]
        ff = np.isfinite(pred) & np.isfinite(resp)

        if (np.sum(ff) == 0) or (np.sum(pred[ff]) == 0) or (np.sum(resp[ff]) == 0):
            mse[i] = 1
            se_mse[i] = 0
        else:
            pred = pred[ff]
            resp = resp[ff]
            chunksize = int(np.ceil(len(pred) / njacks / 10))
            chunkcount = int(np.ceil(len(pred) / chunksize / njacks))
            idx = np.zeros((chunkcount, njacks, chunksize))
            for jj in range(njacks):
                idx[:, jj, :] = jj
            idx = np.reshape(idx, [-1])[:len(pred)]
            jc = np.zeros(njacks)
            for jj in range(njacks):
                ff = (idx != jj)

                X1 = pred[ff]
                X2 = resp[ff]

                respstd = np.nanstd(X2)
                squared_errors = (X1-X2)**2
                E = np.sqrt(np.nanmean(squared_errors))
                if respstd == 0:
                    jc[jj] = 1
                else:
                    jc[jj] = E / respstd

            mse[i] = np.nanmean(jc)
            se_mse[i] = np.nanstd(jc) * np.sqrt(njacks-1)

    return mse, se_mse


def nmse_shrink(result, pred_name='pred', resp_name='resp', shrink=0.1):
    '''
    Same as MSE, but normalized by the std of the resp.
    Because it is more computationally expensive than MSE but is otherwise
    equivalent, we suggest using the MSE for fitting and use this as a
    post-fit performance metric only.
    '''

    X1 = result[pred_name].as_continuous()
    X2 = result[resp_name].as_continuous()

    keepidx = np.isfinite(X1) * np.isfinite(X2)
    if np.all(np.logical_not(keepidx)):
        log.debug("All values were NaN or inf in pred and resp")
        return 1

    X1 = X1[keepidx]
    X2 = X2[keepidx]

    # bounds = np.round(np.linspace(0, len(X1) + 1, 11)).astype(int)

    E = np.zeros([10, 1])
    # P = np.std(X2)
    for ii in range(0, 10):
        jj = np.arange(ii, len(X1), 10)
        # print(jj)
        if len(jj) == 0:
            log.info('No data in range?')

        P = np.std(X2[jj])
        if P > 0:
            E[ii] = np.sqrt(np.mean(np.square(X1[jj] - X2[jj]))) / P
        else:
            E[ii] = 1
    #print(E)
    mE = E.mean()
    sE = E.std() / np.sqrt(len(E))
    # print("me={} se={} shrink={}".format(mE,sE,shrink))
    if mE < 1:
        # apply shrinkage filter to 1-E with factors self.shrink
        mse = 1 - nems0.utils.shrinkage(1 - mE, sE, shrink)
    else:
        mse = mE

    return mse
