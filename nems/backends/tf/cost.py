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
    return (tf.math.reduce_mean(tf.math.square(response - prediction))) / (tf.math.reduce_mean(tf.math.square(response)))


def loss_tf_nmse_shrinkage(response, prediction):
    """Normalized means squared error with shrinkage loss."""
    return tf_nmse_shrinkage(response, prediction)


def loss_tf_nmse(response, prediction, per_cell=True):
    """Normalized means squared error loss."""
    mE, sE = tf_nmse(response, prediction, per_cell=per_cell)
    if per_cell:
        return tf.math.reduce_mean(mE)
    else:
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


def tf_nmse(response, prediction, per_cell=True):
    """Calculates the normalized mean squared error across batches.
    Optionally can return an average per cell.
    :param response:
    :param prediction:
    :param per_cell: Whether to average across all cells or not
    :return: 2 tensors, one of the mean error, the other of the std of the error. If not per cell, then
     tensor is of shape (), else tensor if of shape (n_cells,) (i.e. last dimension of the resp/pred tensor)
    """
    # TODO: This won't work with higher D data, assumes placement of time.
    #       Also not clear what the n_drop thing is about? Why does it matter
    #       if the number of time bins is divisble by 10?
    n_drop = response.get_shape().as_list()[-2] % 10
    if n_drop:
        # use slices to handle varying tensor shapes
        drop_slice = [slice(None) for i in range(len(response.shape))]

        # second last dim is time
        drop_slice[-2] = slice(None, -n_drop)
        drop_slice = tuple(drop_slice)

        _response = response[drop_slice]
        _prediction = prediction[drop_slice]
    else:
        _response = response
        _prediction = prediction

    if per_cell:
        # Put last dimension (number of output channels) first.
        _response = tf.transpose(_response, np.roll(np.arange(len(response.shape)), 1))
        _prediction = tf.transpose(_prediction, np.roll(np.arange(len(response.shape)), 1))

        # What are the hardcoded 10s about? Related to n_drop?
        _response = tf.reshape(_response, shape=(_response.shape[0], 10, -1))
        _prediction = tf.reshape(_prediction, shape=(_prediction.shape[0], 10, -1))
    else:
        _response = tf.reshape(_response, shape=(10, -1))
        _prediction = tf.reshape(_prediction, shape=(10, -1))

    squared_error = ((_response - _prediction) ** 2)
    nmses = (tf.math.reduce_mean(squared_error, axis=-1) /
             tf.math.reduce_mean(_response**2, axis=-1)) ** 0.5

    # Hard-coded 10 again? Why?
    mE = tf.math.reduce_mean(nmses, axis=-1)
    sE = tf.math.reduce_std(nmses, axis=-1) / 10 ** 0.5

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
