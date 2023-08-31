"""Cost functions for TensorFlowBackend.

Cost function code by Alexander Tomlinson (Github: arrrobase),
ported from a previous version of NEMS

TODO: Update documentation, revise functions. Lots of hard-coded magic numbers
      and assumptions about data format.

"""

import numpy as np
import tensorflow as tf
import tensorflow.keras

from nems.tools.lookup import FindCallable


# Keras' built-in MSE with default options
keras_mse = tf.keras.losses.MeanSquaredError()

def poisson(response, prediction):
    """Poisson loss."""
    return tf.math.reduce_mean(prediction - response * tf.math.log(prediction + 1e-5), name='poisson')


def drop_nan(response, prediction):
    mask = tf.math.is_finite(response)
    return tf.boolean_mask(response, mask), tf.boolean_mask(prediction, mask)


# TODO: looks like this is also normalizing by... I guess variance of response?
#       (but assumes mean is 0)
def loss_se(response, prediction):
    """Squared error loss."""
    r = tf.boolean_mask(response, tf.math.is_finite(response))
    p = tf.boolean_mask(prediction, tf.math.is_finite(response))

    return (tf.math.reduce_mean(tf.math.square(r - p))) / (tf.math.reduce_mean(tf.math.square(r)))
    #return (tf.math.reduce_mean(tf.math.square(response - prediction))) / (tf.math.reduce_mean(tf.math.square(response)))


def loss_tf_nmse_shrinkage(response, prediction):
    """Normalized means squared error with shrinkage loss."""
    return tf_nmse_shrinkage(response, prediction)


def loss_tf_nmse(response, prediction, per_cell=False):
    """Normalized means squared error loss."""
    if per_cell:
        mE, sE = tf_nmse(response, prediction, per_cell=per_cell)
        return tf.math.reduce_mean(mE)
        #return tf.reduce_mean(tf.boolean_mask(mE, tf.math.is_finite(mE)))
    else:
        mE, sE = tf_nmse(response, prediction, per_cell=per_cell)

        return mE


def tf_nmse_shrinkage(response, prediction, shrink_factor=0.5, per_cell=True, thresh=False):
    """Calculates the normalized mean squared error, with an adjustment for error.
    Averages across the batches, but optionally can return a per cell error.
    :param response:
    :param prediction:
    :param float shrink_factor:
    :param bool per_cell: Whether to also average over cells or not
    :param bool thresh:
    :return: a "shrunk" normalized mean squared error
    """
    mE, sE = tf_nmse(response, prediction, per_cell)

    def shrink(mE, sE, shrink_factor, thresh):
        def shrink_all(mE, sE, shrink_factor, thresh):
            return tf_shrinkage(mE, sE, shrink_factor, thresh)

        def shrink_some(mE, sE, shrink_factor, thresh):
            mask_gt, mask_lt = mE >= 1, mE < 1
            # make zero where mE was > 1
            shrunk = tf_shrinkage(mE, sE, shrink_factor, thresh) * tf.dtypes.cast(mask_lt, mE.dtype)
            # add back in
            mE = shrunk + mE * tf.dtypes.cast(mask_gt, mE.dtype)
            return mE

        mE = tf.cond(tf.math.reduce_all(mE < 1), lambda: shrink_all(mE, sE, shrink_factor, thresh),
                     lambda: shrink_some(mE, sE, shrink_factor, thresh))
        return mE

    mE = tf.cond(tf.math.reduce_any(mE < 1), lambda: shrink(mE, sE, shrink_factor, thresh), lambda: mE)

    if per_cell:
        mE = tf.math.reduce_mean(mE)

    return mE


def tf_nmse(response, prediction, per_cell=False, allow_nan=True):
    """Calculates the normalized mean squared error across batches.
    Optionally can return an average per cell.
    :param response:
    :param prediction:
    :param per_cell: Whether to average across all cells or not
    :return: 2 tensors, one of the mean error, the other of the std of the error. If not per cell, then
     tensor is of shape (), else tensor if of shape (n_cells,) (i.e. last dimension of the resp/pred tensor)
    """
    # hardcoded to use 10 jackknifes for error estimate
    s = response.get_shape().as_list()
    
    n_drop = s[1] % 10
    n_per = int(s[1]/10)
    if n_drop:
        # use slices to handle varying tensor shapes
        drop_slice = [slice(None) for i in range(len(response.shape))]

        # second dim is time
        drop_slice[1] = slice(None, -n_drop)
        drop_slice = tuple(drop_slice)

        _response = response[drop_slice]
        _prediction = prediction[drop_slice]
    else:
        _response = response
        _prediction = prediction

    #if allow_nan:
    #    print("In tf_nmse:", _response.shape, _prediction.shape, 'n_drop:', n_drop, "(Allowing nan)")
    #else:
    #    print("In tf_nmse:", _response.shape, _prediction.shape, 'n_drop:', n_drop)

    _response = tf.reshape(_response, shape=(-1, 10, n_per, s[2]))
    _prediction = tf.reshape(_prediction, shape=(-1, 10, n_per, s[2]))
    #print("After reshape:", _response.shape, _prediction.shape)

    if per_cell:
        # put dimensions not to compute error over first - axis 0
        #_response = tf.experimental.numpy.moveaxis(_response, [1, 3], [0, 1])
        #_prediction = tf.experimental.numpy.moveaxis(_prediction, [1, 3], [0, 1])
        _response = tf.transpose(_response, perm=[1, 3, 0, 2])
        _prediction = tf.transpose(_prediction, perm=[1, 3, 0, 2])
        #print("After move:", _response.shape, _prediction.shape)

        _response = tf.reshape(_response, shape=(10*s[2], -1))
        _prediction = tf.reshape(_prediction, shape=(10*s[2], -1))
    else:
        # put dimensions not to compute error over first - axis 0
        #_response = tf.experimental.numpy.moveaxis(_response, [1], [0])
        #_prediction = tf.experimental.numpy.moveaxis(_prediction, [1], [0])
        _response = tf.transpose(_response, perm=[1, 0, 2, 3])
        _prediction = tf.transpose(_prediction, perm=[1, 0, 2, 3])
        #print("After move:", _response.shape, _prediction.shape)
        _response = tf.reshape(_response, shape=(10, -1))
        _prediction = tf.reshape(_prediction, shape=(10, -1))

    #print("After reshape:", _response.shape, _prediction.shape)

    if allow_nan:
        C = _response.shape[0]
        N = []
        D = []
        for i in range(C):
            r = tf.boolean_mask(_response[i], tf.math.is_finite(_response[i]))
            p = tf.boolean_mask(_prediction[i], tf.math.is_finite(_response[i]))

            squared_error = ((r - p) ** 2)
            N.append(tf.math.reduce_mean(squared_error, axis=-1))
            D.append(tf.math.reduce_mean(r**2, axis=-1))
        #print("After mask:", r.shape, p.shape)
        numers = tf.stack(N, 0)
        denoms = tf.stack(D, 0)
    else:
        squared_error = ((_response - _prediction) ** 2)
        numers = tf.math.reduce_mean(squared_error, axis=-1)
        denoms = tf.math.reduce_mean(_response**2, axis=-1)

    denoms = tf.where(tf.equal(denoms, 0), tf.ones_like(denoms), denoms)

    nmses = (numers / denoms) ** 0.5
    nmses = tf.reshape(nmses, (10,-1))
    mE = tf.math.reduce_mean(nmses, axis=0)
    sE = tf.math.reduce_std(nmses, axis=0) / 10 ** 0.5

    return mE, sE


def tf_shrinkage(mE, sE, shrink_factor=0.5, thresh=False):
    """Adjusts the mean error based on the standard error"""
    mE = 1 - mE
    smd = tf.math.divide_no_nan(abs(mE), sE) / shrink_factor
    smd = 1 - smd ** -2

    if thresh:
        return 1 - mE * tf.dtypes.cast(smd > 1, mE.dtype)

    smd = smd * tf.dtypes.cast(smd > 0, smd.dtype)

    return 1 - mE * smd


# correlation for monitoring
def pearson(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=-2, keepdims=True)
    my = tf.reduce_mean(y, axis=-2, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.nn.l2_normalize(xm, axis = -2)
    t2_norm = tf.nn.l2_normalize(ym, axis = -2)
    r = tf.reduce_mean(tf.reduce_sum(t1_norm*t2_norm, axis=[-2], keepdims=True))

    return r


cost_nicknames = {'squared_error': loss_se, 'nmse': loss_tf_nmse,
                  'nmse_shrinkage': loss_tf_nmse_shrinkage}
get_cost = FindCallable({**globals(), **cost_nicknames}, header='Cost')
