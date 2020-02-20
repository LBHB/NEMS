"""
Tools for mapping NEMS modelspecs to and from Tensorflow CNNs
Uses Sam Norman-Haignere's CNN library as a front end for TF
"""

import os
import copy
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import nems
import nems.modelspec as ms
import nems.tf.cnn as cnn
import nems.utils
from nems.initializers import init_static_nl
from nems.tf import initializers

log = logging.getLogger(__name__)

modelspecs_dir = nems.get_setting('NEMS_RESULTS_DIR')


def map_layer(layer: dict, fn: str, idx: int, modelspec,
              n_input_feats, net_seed, weight_scale, use_modelspec_init: bool,
              distr: str='norm',) -> dict:
    """Maps a module to a tensorflow layer.

    Available conversions:
        wc -> reweight
        fir -> conv
        firbank
        do
        dexp
        dlog -> dlog
        relu
        lvl -> offset
        wg.g

    :param layer:
    :param fn:
    :param idx:
    :param modelspec:
    :param n_input_feats:
    :param net_seed:
    :param weight_scale:
    :param use_modelspec_init:
    :param distr: The type of distribution to init layers with. Only applicable if use_modelspec_init
                       is False.
    """
    phi = modelspec.get('phi', None)
    fn_kwargs = modelspec['fn_kwargs']

    if fn == 'nems.modules.nonlinearity.relu':
        layer['type'] = 'relu'
        c = -phi['offset'].astype('float32').T
        layer['n_kern'] = c.shape[1]
        n = c.shape[1]
        log.info(f'Modelspec2tf: relu init {n} {c}')

        if use_modelspec_init:
            layer['b'] = tf.Variable(c.reshape((1, c.shape[0], c.shape[1])))
        else:
            layer['b'] = tf.abs(cnn.kern2d(1, c.shape[0], c.shape[1], weight_scale,
                                           seed=net_seed, distr=distr))

        layer['Y'] = tf.nn.relu(layer['X'] + layer['b'])

    elif 'levelshift' in fn:
        layer['type'] = 'offset'
        c = phi['level'].astype('float32').T
        layer['n_kern'] = c.shape[1]

        if use_modelspec_init:
            layer['b'] = tf.Variable(c.reshape((1, c.shape[0], c.shape[1])))
        else:
            layer['b'] = tf.abs(cnn.kern2d(1, c.shape[0], c.shape[1], weight_scale,
                                           seed=net_seed, distr=distr))

        layer['Y'] = tf.identity(layer['X'] + layer['b'])

    elif fn == 'nems.modules.nonlinearity.dlog':
        layer['type'] = 'dlog'
        c = phi['offset'].astype('float32')
        layer['n_kern'] = c.shape[1]

        if use_modelspec_init:
            layer['b'] = tf.Variable(c.reshape((1, c.shape[0], c.shape[1])))
        else:
            # TODO: why does this default to 'tnorm' instead of 'norm'? Change to be var distr?
            layer['b'] = tf.abs(cnn.kern2d(1, c.shape[0], c.shape[1], weight_scale,
                                           seed=cnn.seed_to_randint(net_seed) + idx,
                                           distr='tnorm'))

        # clip b at +/-2 to avoid huge compression/expansion
        layer['eb'] = tf.pow(tf.constant(10, dtype=tf.float32), tf.clip_by_value(layer['b'], -2, 2))
        layer['Y'] = tf.math.log((layer['X'] + layer['eb']) / layer['eb'])

    elif fn == 'nems.modules.nonlinearity.double_exponential':
        layer['type'] = 'dexp'
        layer['n_kern'] = phi['base'].size
        s = (1, 1, layer['n_kern'])

        if use_modelspec_init:
            layer['base'] = tf.Variable(phi['base'].astype('float32').reshape(s))
            layer['amplitude'] = tf.Variable(phi['amplitude'].astype('float32').reshape(s))
            layer['shift'] = tf.Variable(phi['shift'].astype('float32').reshape(s))
            layer['kappa'] = tf.Variable(phi['kappa'].astype('float32').reshape(s))
        else:
            log.info('Modelspec2tf: Using TF rand for double exponential')
            layer['base'] = tf.Variable(tf.random.uniform(
                s, minval=0, maxval=1, seed=cnn.seed_to_randint(net_seed + idx)))
            layer['amplitude'] = tf.Variable(tf.random.uniform(
                s, minval=0.1, maxval=0.5, seed=cnn.seed_to_randint(net_seed + 20 + idx)))
            layer['shift'] = tf.Variable(tf.random.normal(
                s, stddev=weight_scale, mean=0, seed=cnn.seed_to_randint(net_seed + 40 + idx)))
            layer['kappa'] = tf.Variable(tf.random.uniform(
                s, minval=0, maxval=2, seed=cnn.seed_to_randint(net_seed + 60 + idx)))

        # base + amplitude * exp(  -exp(np.array(-exp(kappa)) * (x - shift))  )
        layer['Y'] = layer['base'] + layer['amplitude'] * tf.math.exp(
            -tf.math.exp(-tf.math.exp(layer['kappa']) * (layer['X'] - layer['shift'])))

    elif fn == 'nems.modules.fir.basic':
        layer['type'] = 'conv'
        layer['time_win_smp'] = phi['coefficients'].shape[1]

        c = np.fliplr(phi['coefficients']).astype('float32').T
        c = c.reshape((c.shape[0], c.shape[1], 1))
        if np.all(c == 0):
            c = np.ones_like(c, dtype='float32') / 10

        layer['n_kern'] = 1

        if use_modelspec_init:
            layer['W'] = tf.Variable(c)
        else:
            layer['W'] = cnn.kern2d(layer['time_win_smp'], c.shape[1], 1,
                                    weight_scale, seed=net_seed, distr=distr)

        pad_size = np.int32(np.floor(layer['time_win_smp'] - 1))
        X_pad = tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]])
        layer['Y'] = tf.nn.conv1d(X_pad, layer['W'], stride=1, padding='VALID')

    elif fn == 'nems.modules.fir.damped_oscillator':
        layer['type'] = 'conv'
        layer['time_win_smp'] = fn_kwargs['n_coefs']
        layer['rate'] = fn_kwargs.get('rate', 1)
        cross_channels = fn_kwargs.get('cross_channels', False)
        bank_count = fn_kwargs['bank_count']
        layer['n_kern'] = bank_count

        chan_count = int(phi['f1s'].size / bank_count)
        in_chan_count = int(layer['X'].shape[2])
        pad_size = np.int32(np.floor(layer['time_win_smp'] - 1))
        if cross_channels and (bank_count > 1):
            raise ValueError('cross_channels not supported for bank_count > 1.')

        if cross_channels:
            s = (1, 1, 1, chan_count * bank_count)
        elif bank_count == 1:
            # revert to simple conv1d, traditional FIR filter
            s = (1, chan_count, 1)
        elif in_chan_count == bank_count * chan_count:
            # break up inputs to feed into each bank
            s = (1, 1, chan_count * bank_count, 1)
        else:
            # apply each filter to all inputs
            if chan_count != in_chan_count:
                raise ValueError('Either chan_count*bank_count or chan_count must equal in_chan_count.')
            s = (1, 1, chan_count, bank_count)

        if use_modelspec_init:
            layer['f1'] = tf.Variable(phi['f1s'].astype('float32').reshape(s))
            layer['tau'] = tf.Variable(phi['taus'].astype('float32').reshape(s))
            layer['delay'] = tf.Variable(phi['delays'].astype('float32').reshape(s))
            layer['gain'] = tf.Variable(phi['gains'].astype('float32').reshape(s))
        else:
            log.info('Modelspec2tf: Using TF rand for damped oscillator')
            layer['f1'] = tf.Variable(tf.random.uniform(
                s, minval=0, maxval=1, seed=cnn.seed_to_randint(net_seed + idx)))
            layer['tau'] = tf.Variable(tf.random.uniform(
                s, minval=0.1, maxval=0.5, seed=cnn.seed_to_randint(net_seed + 20 + idx)))
            layer['gain'] = tf.Variable(tf.random.normal(
                s, stddev=weight_scale, mean=0, seed=cnn.seed_to_randint(net_seed + 40 + idx)))
            layer['delay'] = tf.Variable(tf.random.uniform(
                s, minval=0, maxval=2, seed=cnn.seed_to_randint(net_seed + 60 + idx)))

        # time lag reversed
        if len(s) == 3:
            layer['t'] = tf.reshape(tf.range(layer['time_win_smp'] - 1, -1, -1, dtype=tf.float32),
                                    [layer['time_win_smp'], 1, 1]) - layer['delay']
        elif cross_channels and (bank_count == 1):
            layer['t'] = tf.reshape(tf.range(layer['time_win_smp'] - 1, -1, -1, dtype=tf.float32),
                                    [layer['time_win_smp'], 1, 1, 1]) - layer['delay']
        else:
            layer['t'] = tf.reshape(tf.range(layer['time_win_smp'] - 1, -1, -1, dtype=tf.float32),
                                    [1, layer['time_win_smp'], 1, 1]) - layer['delay']
        coefficients = tf.math.sin(layer['f1'] * layer['t']) * tf.math.exp(-layer['tau'] * layer['t']) * layer['gain']
        layer['b'] = tf.math.greater(layer['t'], tf.constant(0, dtype=tf.float32))
        layer['W'] = tf.multiply(coefficients, tf.cast(layer['b'], tf.float32))

        if cross_channels & (bank_count == 1):
            # special case: "outer product" convolve each channel with each filter
            # insert placeholder dim on axis=3
            X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 3)
            layer['tY'] = tf.nn.conv2d(X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID')
            layer['Y'] = tf.reshape(layer['tY'], [-1, layer['tY'].shape[1],
                                                  layer['tY'].shape[2] * layer['tY'].shape[3]])
        elif bank_count == 1:
            # original implementation (no filter bank concept)
            X_pad = tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]])
            layer['Y'] = tf.nn.conv1d(X_pad, layer['W'], stride=1, padding='VALID')
        elif in_chan_count == bank_count * chan_count:
            # each bank applied to a segment of the input channels
            # insert placeholder dim on axis=1
            X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 1)
            layer['tY'] = tf.nn.depthwise_conv2d(
                X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID', rate=[1, layer['rate']])
            s = tf.shape(layer['tY'])
            layer['Y'] = tf.reduce_sum(tf.reshape(
                layer['tY'], [s[0], layer['tY'].shape[2],
                              tf.compat.v1.Dimension(bank_count), tf.compat.v1.Dimension(chan_count)]), axis=3)
        else:
            # apply each fir bank to same input channels
            # insert placeholder dim on axis=1
            X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 1)
            layer['tY'] = tf.nn.depthwise_conv2d(X_pad, layer['W'], strides=[1, 1, 1, 1],
                                                 padding='VALID', rate=[1, layer['rate']])
            s = tf.shape(layer['tY'])
            layer['Y'] = tf.reduce_sum(tf.reshape(layer['tY'],
                                                  [s[0], layer['tY'].shape[2], tf.compat.v1.Dimension(chan_count),
                                                   tf.compat.v1.Dimension(bank_count)]), axis=2)

    elif fn == 'nems.modules.fir.filter_bank':
        layer['type'] = 'conv_bank_1d'
        layer['time_win_smp'] = phi['coefficients'].shape[1]
        layer['rate'] = fn_kwargs.get('rate', 1)
        layer['rank'] = None  # we're handling rank with explicit spectral filters
        bank_count = fn_kwargs['bank_count']
        layer['n_kern'] = bank_count

        c = np.fliplr(phi['coefficients']).astype('float32').T
        chan_count = int(c.shape[1] / bank_count)
        in_chan_count = int(layer['X'].shape[2])

        # split inputs into the different kernels
        if bank_count == 1:
            c = c.reshape((c.shape[0], 1, 1, bank_count * chan_count))
        elif in_chan_count == bank_count * chan_count:
            c = c.reshape((1, c.shape[0], chan_count * bank_count, 1))
        else:
            c = c.reshape((1, c.shape[0], chan_count, bank_count))

        if np.all(c == 0):
            c[:, :, :, :] = 1

        # figure out input padding to ensure causality,
        pad_size = np.int32((layer['time_win_smp'] - 1) * layer['rate'])

        if use_modelspec_init:
            layer['W'] = tf.Variable(c)
        else:
            layer['W'] = initializers.weights_norm(c.shape, sig=weight_scale, seed=cnn.seed_to_randint(net_seed) + idx)

        if bank_count == 1:
            # "outer product" convolve each channel with each filter
            # insert placeholder dim on axis=3
            X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 3)
            layer['tY'] = tf.nn.conv2d(X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID')
            layer['Y'] = tf.reshape(layer['tY'],
                                    [-1, layer['tY'].shape[1], layer['tY'].shape[2] * layer['tY'].shape[3]])
        elif in_chan_count == bank_count * chan_count:
            # each bank applied to a segment of the input channels
            # insert placeholder dim on axis=1
            X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 1)
            layer['tY'] = tf.compat.v1.nn.depthwise_conv2d(
                X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID', rate=[1, layer['rate']])
            s = tf.shape(layer['tY'])
            layer['Y'] = tf.reduce_sum(tf.reshape(layer['tY'], [s[0], layer['tY'].shape[2], tf.compat.v1.Dimension(bank_count),
                                                                tf.compat.v1.Dimension(chan_count)]), axis=3)
        else:
            # apply each fir bank to same input channels
            # insert placeholder dim on axis=1
            X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 1)
            layer['tY'] = tf.compat.v1.nn.depthwise_conv2d(X_pad, layer['W'], strides=[1, 1, 1, 1],
                                                 padding='VALID', rate=[1, layer['rate']])
            s = tf.shape(layer['tY'])
            layer['Y'] = tf.reduce_sum(tf.reshape(layer['tY'],
                                                  [s[0], layer['tY'].shape[2], tf.compat.v1.Dimension(chan_count),
                                                   tf.compat.v1.Dimension(bank_count)]), axis=2)

    elif fn == 'nems.modules.weight_channels.basic':
        layer['type'] = 'reweight'
        c = phi['coefficients'].astype('float32').T
        layer['n_kern'] = c.shape[1]

        if use_modelspec_init and np.any(c != 0):
            layer['W'] = tf.Variable(c.reshape((1, c.shape[0], c.shape[1])))
        else:
            s = [1, c.shape[0], c.shape[1]]
            layer['W'] = tf.Variable(tf.random.normal(s, stddev=weight_scale, mean=0,
                                                      seed=cnn.seed_to_randint(net_seed + idx)))

        if fn_kwargs['i'] == 'resp':
            layer['L'] = tf.nn.conv1d(layer['D'], layer['W'], stride=1, padding='SAME')
        else:
            layer['Y'] = tf.nn.conv1d(layer['X'], layer['W'], stride=1, padding='SAME')

    elif fn == 'nems.modules.weight_channels.gaussian':
        layer['n_kern'] = phi['mean'].shape[0]

        # HACK : scale sd by 10 to play well with TF fitter
        mn = phi['mean'].astype('float32')
        sd = phi['sd'].astype('float32') * 10

        if use_modelspec_init:
            layer['m'] = tf.Variable(np.reshape(mn, (1, 1, mn.shape[0])))
            layer['s'] = tf.Variable(np.reshape(sd, (1, 1, sd.shape[0])))
        else:
            log.info('Modelspec2tf: Using TF rand for wcg')
            layer['m'] = tf.Variable(tf.random.uniform(
                [1, 1, layer['n_kern']], minval=0, maxval=1,
                seed=cnn.seed_to_randint(net_seed + idx)))
            layer['s'] = tf.Variable(tf.random.uniform(
                [1, 1, layer['n_kern']], minval=0.1, maxval=0.5,
                seed=cnn.seed_to_randint(net_seed + 20 + idx)))

        # trying to impose NEMS bounds
        b = modelspec['bounds']
        layer['m0'] = tf.clip_by_value(layer['m'], b['mean'][0][0], b['mean'][1][0])
        layer['s0'] = tf.clip_by_value(layer['s'] / 10, b['sd'][0][0], b['sd'][1][0])
        # sd[sd < 0.00001] = 0.00001
        # layer['s0'] = tf.clip_by_value(layer['s'], 0.00001*10, 100)
        layer['f'] = tf.reshape(tf.range(n_input_feats, dtype=tf.float32),
                                [1, n_input_feats, 1]) / n_input_feats
        layer['Wraw'] = tf.exp(-0.5 * tf.square((layer['f'] - layer['m0']) / layer['s0']))
        layer['W'] = layer['Wraw'] / tf.reduce_sum(layer['Wraw'], axis=1)
        layer['Y'] = tf.nn.conv1d(layer['X'], layer['W'], stride=1, padding='SAME')

    # elif fn == 'nems.modules.state.state_dc_gain':
    #     if phi is not None:
    #         g = phi['g'].astype('float32').T
    #         d = phi['d'].astype('float32').T
    #         g = g.reshape((1, g.shape[0], g.shape[1]))
    #         d = d.reshape((1, d.shape[0], d.shape[1]))
    #
    #         if use_modelspec_init:
    #             layer['g'] = tf.Variable(g)
    #             layer['d'] = tf.Variable(d)
    #         else:
    #             layer['g'] = tf.Variable(tf.random_normal(g.shape, stddev=weight_scale
    #                                                       , seed=cnn.seed_to_randint(net_seed + idx)))
    #             layer['d'] = tf.Variable(tf.random_normal(d.shape, stddev=weight_scale,
    #                                                       seed=cnn.seed_to_randint(net_seed + 20 + idx)))
    #
    #     else:
    #         # dc/gain values are fixed
    #         g = fn_kwargs['g'].astype('float32').T
    #         d = fn_kwargs['d'].astype('float32').T
    #         g =g.reshape((1, g.shape[0], g.shape[1]))
    #         d = d.reshape((1, d.shape[0], d.shape[1]))
    #
    #         layer['g'] = tf.constant(g)
    #         layer['d'] = tf.constant(d)
    #
    #     layer['n_kern'] = g.shape[2]
    #     layer['Sg'] = tf.nn.conv1d(layers[0]['S'], layer['g'], stride=1, padding='SAME')
    #     layer['Sd'] = tf.nn.conv1d(layers[0]['S'], layer['d'], stride=1, padding='SAME')
    #     layer['Y'] = layer['X'] * layer['Sg'] + layer['Sd']

    else:
        raise ValueError(f'Module "{fn}" not supported for mapping to cnn layer.')

    return layer


def tf2modelspec(net, modelspec):
    """
    pass TF cnn fit back into modelspec phi.
    TODO: Generate new modelspec if not provided
    DONE: Make sure that the dimension mappings work reliably for filter banks and such
    """

    net_layer_vals = net.layer_vals()
    for i, m in enumerate(modelspec):
        log.info('tf2modelspec: ' + m['fn'])

        if m['fn'] == 'nems.modules.nonlinearity.relu':
            m['phi']['offset'] = -net_layer_vals[i]['b'][0, :, :].T

        elif 'levelshift' in m['fn']:
            m['phi']['level'] = net_layer_vals[i]['b'][0, :, :].T

        elif m['fn'] in ['nems.modules.nonlinearity.double_exponential']:
            # base + amplitude * exp(  -exp(np.array(-exp(kappa)) * (x - shift))  )
            m['phi']['base'] = net_layer_vals[i]['base'][:, 0, :].T
            m['phi']['amplitude'] = net_layer_vals[i]['amplitude'][:, 0, :].T
            m['phi']['kappa'] = net_layer_vals[i]['kappa'][:, 0, :].T
            m['phi']['shift'] = net_layer_vals[i]['shift'][:, 0, :].T

        elif m['fn'] in ['nems.modules.fir.basic']:
            m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][:, :, 0].T)

        elif m['fn'] in ['nems.modules.fir.damped_oscillator']:
            if (m['fn_kwargs']['bank_count'] == 1) and (not m['fn_kwargs']['cross_channels']):
                m['phi']['f1s'] = net_layer_vals[i]['f1'][:, :, 0].T
                m['phi']['taus'] = net_layer_vals[i]['tau'][:, :, 0].T
                m['phi']['gains'] = net_layer_vals[i]['gain'][:, :, 0].T
                m['phi']['delays'] = net_layer_vals[i]['delay'][:, :, 0].T
            else:
                # new depthwise_conv2d
                m['phi']['f1s'] = np.reshape(net_layer_vals[i]['f1'][0, 0, :, :].T, [-1, 1])
                m['phi']['taus'] = np.reshape(net_layer_vals[i]['tau'][0, 0, :, :].T, [-1, 1])
                m['phi']['gains'] = np.reshape(net_layer_vals[i]['gain'][0, 0, :, :].T, [-1, 1])
                m['phi']['delays'] = np.reshape(net_layer_vals[i]['delay'][0, 0, :, :].T, [-1, 1])

        elif m['fn'] in ['nems.modules.fir.filter_bank']:
            if m['fn_kwargs']['bank_count'] == 1:
                m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][:, 0, 0, :].T)
            else:
                # new depthwise_conv2d
                c = net_layer_vals[i]['W'][0, :, :, :]
                c = np.transpose(c, (2, 1, 0))
                c = np.reshape(c, [-1, c.shape[-1]])
                m['phi']['coefficients'] = np.fliplr(c)
            #else:
            #    # inefficient conv2d
            #    m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][:, 0, 0, :].T)

        elif m['fn'] in ['nems.modules.weight_channels.basic']:
            m['phi']['coefficients'] = net_layer_vals[i]['W'][0, :, :].T

        elif m['fn'] in ['nems.modules.nonlinearity.dlog']:
            m['phi']['offset'] = net_layer_vals[i]['b'][0, :, :].T

        elif m['fn'] in ['nems.modules.weight_channels.gaussian']:
            #m['phi']['mean'] = net_layer_vals[i]['m'][0, 0, :].T
            m['phi']['mean'] = np.clip(net_layer_vals[i]['m'][0, 0, :].T,
                                     m['bounds']['mean'][0], m['bounds']['mean'][1])
            m['phi']['sd'] = np.clip(net_layer_vals[i]['s'][0, 0, :].T / 10,
                                     m['bounds']['sd'][0], m['bounds']['sd'][1])

        elif m['fn'] in ['nems.modules.state.state_dc_gain']:
            # if init.st, not fitting these params, no phi, so skip
            if 'phi' in m.keys():
                m['phi']['g'] = net_layer_vals[i]['g'][0, :, :].T
                m['phi']['d'] = net_layer_vals[i]['d'][0, :, :].T

        else:
            raise ValueError("NEMS module fn=%s not supported", m['fn'])

    return modelspec


def _fit_net(F, D, modelspec, seed, fs, log_dir, optimizer='Adam',
             max_iter=1000, learning_rate=0.01, use_modelspec_init=False, S=None,
             loss_type='squared_error', early_stopping_steps=5, early_stopping_tolerance=5e-4,
             distr='norm'):

    n_feats = F.shape[2]
    data_dims = D.shape
    sr_Hz = fs

    tf.compat.v1.reset_default_graph()
    if S is not None:
        state_dims = S.shape[2]
    else:
        state_dims = 0
    log.info(f'rand seed for intialization: {seed}')
    layers = modelspec.modelspec2tf(tps_per_stim=D.shape[1], feat_dims=n_feats,
                          data_dims=D.shape[2], state_dims=state_dims, fs=fs,
                          use_modelspec_init=use_modelspec_init, distr=distr, net_seed=seed)
    net = cnn.Net(data_dims, n_feats, sr_Hz, layers, seed=seed, log_dir=log_dir, loss_type=loss_type, optimizer=optimizer)

    net.train(F, D, max_iter=max_iter, learning_rate=learning_rate, state=S,
              early_stopping_steps=early_stopping_steps, early_stopping_tolerance=early_stopping_tolerance)
    #for k in net.layers[-1].keys():
    #    try:
    #       log.info('size of final layer {}: {}'.format(k, net.layers[-1][k].shape))
    #    except:
    #       pass
    modelspec = tf2modelspec(net, modelspec)

    # record last iter in extra results
    try:
        max_iter = modelspec.meta['extra_results']
        modelspec.meta['extra_results'] = max(max_iter, net.last_iter)
    except KeyError:
        modelspec.meta['extra_results'] = net.last_iter

    return modelspec, net


def fit_tf_init(modelspec=None, est=None, use_modelspec_init=True,
                optimizer='Adam', max_iter=2000, cost_function='squared_error', **context):
    """
    pre-fit a model with the final output NL stripped. TF equivalent of
    nems.initializers.prefit_to_target()
    :param est: A recording object
    :param modelspec: A modelspec object
    :param use_modelspec_init: [True] use input modelspec phi for initialization. Otherwise use random inits
    :param optimizer:
    :param max_iter: max number of training iterations
    :param cost_function: which loss function to use
    :param context: extra stuff from xforms context
    :return: dictionary with modelspec, compatible with xforms
    """

    # preserve input modelspec
    modelspec = modelspec.copy()

    # TODO : get rid of target_module and just take off final module if it's a static NL.
    # and add a lvl.R if one doesn't already exist in modelspec[-2]

    target_module = ['levelshift', 'relu']
    extra_exclude = ['stp', 'rdt_gain', 'state_dc_gain', 'state_gain']

    # figure out last modelspec module to fit
    target_i = None
    if type(target_module) is not list:
        target_module = [target_module]
    for i, m in enumerate(modelspec.modules):
        tlist = [True for t in target_module if t in m['fn']]

        if len(tlist):
            target_i = i + 1
            # don't break. use last occurrence of target module
            if 'levelshift' in m['fn']:
                # unless it's a levelshift, in which case, you might want to skip the last relu!
                break

    if not target_i:
        log.info("target_module: {} not found in modelspec."
                 .format(target_module))
        return modelspec
    else:
        log.info("target_module: {0} found at modelspec[{1}]."
                 .format(target_module, target_i-1))

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    exclude_idx = []
    tmodelspec = ms.ModelSpec()
    for i in range(len(modelspec)):
        m = copy.deepcopy(modelspec[i])

        for fn in extra_exclude:
            # log.info('exluding '+fn)
            # log.info(m['fn'])
            # log.info(m.get('phi'))
            if (fn in m['fn']):
                #if (m.get('phi') is None):
                #    m = priors.set_mean_phi([m])[0]  # Inits phi
                #    log.info('Mod %d (%s) fixing phi to prior mean', i, fn)
                #else:
                #    log.info('Mod %d (%s) fixing phi', i, fn)

                m['fn_kwargs'].update(m['phi'])
                del m['phi']
                del m['prior']
                exclude_idx.append(i)
                # log.info(m)
        if ('relu' in m['fn']):
            log.info('found relu')

        elif ('levelshift' in m['fn']):
            #m = priors.set_mean_phi([m])[0]
            output_name = modelspec.meta.get('output_name', 'resp')
            try:
                mean_resp = np.nanmean(est[output_name].as_continuous(), axis=1, keepdims=True)
            except NotImplementedError:
                # as_continuous only available for RasterizedSignal
                mean_resp = np.nanmean(est[output_name].rasterize().as_continuous(), axis=1, keepdims=True)
            log.info('Mod %d (%s) fixing level to %s mean %.3f',
                     i, m['fn'], output_name, mean_resp[0])
            log.info('resp has %d channels', len(mean_resp))
            m['phi']['level'][:] = mean_resp

        if (i < target_i) or ('merge_channels' in m['fn']):
            tmodelspec.append(m)
    log.info(tmodelspec)

    # fit the subset of modules - this is instead of calling analysis_function in
    # nems.initializers.prefit_to_target
    seed = 100 + modelspec.fit_index
    new_context = fit_tf(modelspec=tmodelspec, est=est, use_modelspec_init=use_modelspec_init,
                     optimizer=optimizer, max_iter=max_iter, cost_function=cost_function,
                     seed=seed, **context)
    tmodelspec = new_context['modelspec']

    for i in np.setdiff1d(np.arange(target_i), np.array(exclude_idx)).tolist():
        modelspec[int(i)] = tmodelspec[int(i)]

    # pre-fit static NL if it exists
    _d = init_static_nl(est=est, modelspec=modelspec)
    modelspec = _d['modelspec']
    #include_names = _d['include_names']
    # TODO : Initialize relu in some intelligent way?

    return {'modelspec': modelspec}


def fit_tf(modelspec=None, est=None,
           use_modelspec_init=True, init_count=1,
           optimizer='Adam', max_iter=1000, cost_function='squared_error',
           early_stopping_steps=5, early_stopping_tolerance=5e-4, learning_rate=0.01,
           distr='norm', seed=50, **context):
    """
    :param est: A recording object
    :param modelspec: A modelspec object
    :param use_modelspec_init: [True] use input modelspec phi for initialization. Otherwise use random inits
    :param init_count: number of random initializations (if use_modelspec_init==False)
    :param optimizer:
    :param max_iter: max number of training iterations
    :param cost_function: cost_function to use when fitting
    :param metaname:
    :param context:
    :return: dictionary with modelspec, compatible with xforms
    """

    start_time = time.time()

    _this_seed = seed + modelspec.fit_index
    log.info(f'seed for this fit: {_this_seed}')
    sr_Hz = est['resp'].fs
    #time_win_sec = 0.1

    new_est = est.apply_mask()
    n_feats = new_est['stim'].shape[0]
    epoch_name = 'REFERENCE'
    e = est['stim'].get_epoch_indices(epoch_name, mask=est['mask'])
    if 'state' in est.signals.keys():
        n_states = est['state'].shape[0]
    else:
        n_states = 0
        S = None
    # length of each segment is length of a reference
    de = e[:, 1] - e[:, 0]
    n_tps_per_stim = de[0]
    if np.sum(np.abs(de-n_tps_per_stim)) > 0:
        epoch_name = 'TRIAL'

    F = np.transpose(est['stim'].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
    D = np.transpose(est['resp'].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
    if n_states > 0:
        S = np.transpose(est['state'].extract_epoch(epoch=epoch_name, mask=est['mask']),[0, 2, 1])

    feat_dims = F.shape
    data_dims = D.shape
    log.info('feat_dims: %s', feat_dims)
    log.info('data_dims: %s', data_dims)

    # check to see if we should log model weights elsewhere
    log_dir = modelspec.meta['modelpath']

    try:
        job_id = os.environ.get('SLURM_JOBID',None)
        if job_id is not None:
           # keep a record of the job id
           modelspec.meta['slurm_jobid'] = job_id
   
           log_dir_root = Path('/mnt/scratch')
           assert log_dir_root.exists()
           log_dir_sub = Path('SLURM_JOBID' + job_id) / str(modelspec.meta['batch'])\
                         / modelspec.meta.get('cellid', "NOCELL")\
                         / modelspec.meta['modelname']
           log_dir = log_dir_root / log_dir_sub
    except KeyError:
        pass

    modelspec_pre = modelspec.copy()
    modelspec, net = _fit_net(F, D, modelspec, _this_seed, new_est['resp'].fs, log_dir=str(log_dir),
                              optimizer=optimizer, max_iter=np.min([max_iter]), learning_rate=learning_rate,
                              use_modelspec_init=use_modelspec_init, S=S, loss_type=cost_function,
                              early_stopping_steps=early_stopping_steps,
                              early_stopping_tolerance=early_stopping_tolerance, distr=distr)

    new_est = eval_tf(modelspec, new_est, log_dir)

    # clear out tmp dir
    if os.environ.get('SLURM_JOBID'):
        try:
            shutil.rmtree(log_dir)
            try:
                for p in list(log_dir.parents)[:4]:
                    p.rmdir()
            except OSError:
                pass
        except:  # TODO what's the error raised by shutil
            pass

    y = new_est['pred'].as_continuous()
    y2 = new_est['pred_nems'].as_continuous()
    E = np.nanstd(y[:,10:]-y2[:,10:])
    """
    try:
        new_est = modelspec.evaluate(new_est)
    except:
        log.info('evaluation of tf->nems models failed')
        import pdb
        pdb.set_trace()

    # test that TF and NEMS models have same prediction
    y = net.predict(F, S=S)
    p1 = y[0, :, 0]
    if not trial_based_reshape:
        #p2 = new_est['pred'].as_continuous()[0,:n_tps_per_stim]
        y2 = np.reshape(new_est['pred'].as_continuous().copy().T, data_dims)
    else:
        y2 = np.transpose(new_est['pred'].extract_epoch(epoch='TRIAL', mask=est['mask']),[0, 2, 1])
    p2 = y2[0, :, 0]
    E = np.nanstd(p1-p2)
    """
    log.info('Mean difference between NEMS and TF model pred: %e', E)
    #import pdb; pdb.set_trace()

    if np.isnan(E) or (E > 1e-2):
        log.info('E too big? Jumping to debug mode.')
        import matplotlib.pyplot as plt
        plt.figure()
        ax1=plt.subplot(2, 1, 1)
        ax1.plot(y[0, :1000], 'b')
        ax1.plot(y2[0, :1000], 'r')
        ax1.plot(y2[0, :1000]-y[0, :1000], '--')
        ax1.legend(('TF', 'NEMS', 'diff'))

        plt.show()
        log.info(modelspec.phi)
        net_layer_vals = net.layer_vals()
        m = modelspec[2]
        if m['fn'] == 'nems.modules.fir.damped_oscillator':
            from nems.modules.fir import do_coefficients
            args = m['fn_kwargs']
            args.update(m['phi'])
            w_nems = do_coefficients(**args)
            w_tf = net_layer_vals[2]['W']
            ax3=plt.subplot(2, 2, 3)
            ax3.plot(np.flipud(np.squeeze(w_tf)))
            ax4=plt.subplot(2, 2, 4)
            ax4.plot(w_nems.T)
        #from nems.modules.weight_channels import gaussian_coefficients
        #log.info(gaussian_coefficients(modelspec.phi[1]['mean'], modelspec.phi[1]['sd'],
        #                      modelspec[1]['fn_kwargs']['n_chan_in']))
        import pdb
        pdb.set_trace()

    nems.utils.progress_fun()

    elapsed_time = (time.time() - start_time)
    modelspec.meta['fitter'] = 'fit_tf'
    modelspec.meta['fit_time'] = elapsed_time
    modelspec.meta['n_parms'] = len(modelspec.phi_vector)

    #import pdb
    #pdb.set_trace()

    return {'modelspec': modelspec}


def eval_tf(modelspec, est, log_dir):
    """
    TODO : evaluate a NEMS model through TF
    :param modelspec:
    :param est:
    :return:
    """

    new_est = modelspec.evaluate(est)
    new_est['pred_nems'] = new_est['pred'].copy()
    
    # extract stim. does it need to be reshaped to be multiple batches? probably not.
    n_feats = new_est['stim'].shape[0]

    # don't need batches, so can use a single "stim" that contains the whole recording
    n_stim = 1
    n_resp = new_est['resp'].shape[0]
    n_tps_per_stim = new_est['resp'].shape[1]

    feat_dims = [n_stim, n_tps_per_stim, n_feats]
    data_dims = [n_stim, n_tps_per_stim, n_resp]

    # extract stimulus matrix
    F = np.reshape(new_est['stim'].as_continuous().copy().T, feat_dims)
    #D = np.reshape(new_est['resp'].as_continuous().copy().T, data_dims)

    if 'state' in est.signals.keys():
        n_states = est['state'].shape[0]
        state_dims = [n_stim, n_tps_per_stim, n_states]
        S = np.reshape(new_est['state'].as_continuous().copy().T, state_dims)
    else:
        n_states = 0
        S = None

    log.info('feat_dims: %s', feat_dims)
    log.info('data_dims: %s', data_dims)

    fs = est['resp'].fs

    tf.compat.v1.reset_default_graph()

    # initialize tf and evaluate -- VALIDAT THAT NEMS and TF match
    layers = modelspec.modelspec2tf(tps_per_stim=n_tps_per_stim, feat_dims=n_feats,
                          data_dims=n_resp, state_dims=n_states, fs=fs,
                          use_modelspec_init=True)
    net = cnn.Net(data_dims, n_feats, fs, layers, seed=0, log_dir=modelspec.meta['modelpath'])

    y = np.reshape(net.predict(F, state=S).T, [n_resp, n_tps_per_stim])

    # paste back into rec
    new_est['pred'] = new_est['pred']._modified_copy(data=y)

    # test that TF and NEMS models have same prediction
    y2 = new_est['pred_nems'].as_continuous()

    #plt.figure()
    #plt.plot(y[0,:1000,0])
    #plt.plot(y2[0,:1000,0])

    #E = np.nanstd(new_est['pred'].as_continuous()-new_est['pred_nems'].as_continuous())

    return new_est
