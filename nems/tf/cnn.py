"""Code for fitting LN network models based on Sam Norman-Haignere's cnn library"""

import itertools
import logging
import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from nems0.tf import initializers, loss_functions
import nems0.utils

log = logging.getLogger(__name__)


def seed_to_randint(seed):
    np.random.seed(seed)
    return np.random.randint(1e9)


def kern2d(n_x, n_y, n_kern, sig, rank=None, seed=0, distr='tnorm'):
    """
    Creates a 2D kernel with different initializations.

    :param n_x: temporal bins (earliest to most recent)
    :param n_y: spectral bins (input channels)
    :param n_kern: number of filters
    :param sig:
    :param rank:
    :param seed:
    :param distr:
    :return:
    """
    log.info(f'kern2D distribution type: {distr}')

    if type(distr) is np.ndarray:
        weights = initializers.weights_matrix(distr)
        log.info(f'kern2D weights shape: {weights.shape}')
        return weights

    elif distr == 'tnorm':
        fn = initializers.weights_tnorm
    elif distr == 'norm':
        fn = initializers.weights_norm
    elif distr == 'zeros':
        fn = initializers.weights_zeros
    elif distr == 'uniform':
        fn = initializers.weights_uniform
    elif distr == 'glorot_uniform':
        fn = initializers.weights_glorot_uniform
    elif distr == 'he_uniform':
        fn = initializers.weights_he_uniform
    else:
        raise NameError('No matching distributions.')

    if rank is None:
        kern2d_seed = seed_to_randint(seed)
        log.info(f'kern2d seed: {kern2d_seed}')
        weights = fn([n_x, n_y, n_kern], sig=sig, seed=kern2d_seed)
    else:
        weights_list = []
        for i in range(n_kern):
            base_seed = seed_to_randint(seed)
            log.info(f'kern2d seed for kern #{i + 1}: {base_seed}')

            weight = tf.reshape(
                tf.matmul(
                    fn([n_x, rank], sig=sig, seed=base_seed + 1),
                    fn([rank, n_y], sig=sig, seed=base_seed + 1 + n_kern)
                ),
                [int(n_x), int(n_y), 1]
            )
            weights_list.append(weight)

        weights = tf.concat(weights_list, 2)

    log.info(f'kern2D weights shape: {weights.shape}')

    return weights


class Net:
    """Custom implementation of tensorflow neural net."""
    def __init__(self, data_dims, n_feats, sr_Hz, layers, loss_type='squared_error',
                 seed=0, log_dir=None, log_id=None, optimizer='Adam'):
        """
        Instantiates the net.

        :param list data_dims: Number of data dimensions.
        :param int n_feats: Number of features.
        :param sr_Hz: Signal rate in Hertz.
        :param list layers: List of model layers.
        :param str loss_type: Type of loss to use.
        :param seed: Random seed.
        :param log_dir: Directory to save logs to. Must be a file path, not an http path.
        :param log_id: Filename of logs saved to log directory.
        :param optimizer: Type of optimizer to use (i.e. Adam vs Gradient Descent, etc.)
        """
        # dimensionality of features and data
        self.n_stim = data_dims[0]
        self.n_tps_input = data_dims[1]
        self.n_resp = data_dims[2]
        self.n_feats = n_feats

        # placeholders for input features and data
        if 'S' in layers[0].keys():
            self.state = layers[0]['S']

        if 'X' in layers[0].keys():
            self.stim = layers[0]['X']
        else:
            self.stim = tf.compat.v1.placeholder(
                'float32', shape=[None, self.n_tps_input, self.n_feats])

        self.resp = tf.compat.v1.placeholder(
            'float32', shape=[None, self.n_tps_input, self.n_resp])

        # layer parameters
        self.layers = layers

        for layer in self.layers:
            layer['time_win_smp'] = np.int32(np.round(sr_Hz * layer['time_win_sec']))
        assert(self.layers[-1]['n_kern'] == self.n_resp)  # TODO what is this?

        # other misc parameters
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.learning_rate = tf.compat.v1.placeholder('float32')

        # training stats
        self.train_loss = []
        self.iteration = []
        self.best_loss = None
        self.best_loss_index = 0
        self.last_iter = 0

        # directory and file info
        if log_dir is None:
            log_dir = Path(r'/auto/data/tmp/cnn_scratch')

        self.log_dir = Path(log_dir)

        if log_id is None:
            self.log_id = 'seed' + str(seed)
        else:
            self.log_id = log_id

        self.initialize()

    def initialize(self):
        """Initialize the properties of the neural net."""
        log.info('Initializing net: setting output, loss, optimizer, globals, tf session')
        self.pred = self.layers[-1]['Y'][:, 0:self.n_tps_input, :]

        # loss type
        if self.loss_type == 'squared_error':
            self.loss = loss_functions.loss_se(self.resp, self.pred)
        elif self.loss_type == 'poisson':
            self.loss = loss_functions.poisson(self.resp, self.pred)
        elif self.loss_type == 'nmse':
            self.loss = loss_functions.loss_tf_nmse(self.resp, self.pred)
        elif self.loss_type == 'nmse_shrinkage':
            self.loss = loss_functions.loss_tf_nmse_shrinkage(self.resp, self.pred)
        else:
            raise NameError(f'Loss must be "squared_error", "poisson", "nmse",'
                            f' or "nmse_shrinkage", not {self.loss_type}')

        # gradient optimizer
        if self.optimizer == 'Adam':
            self.train_step = tf.compat.v1.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'GradientDescent':
            self.train_step = tf.compat.v1.train.GradientDescentOptimizer(
                self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'RMSProp':
            self.train_step = tf.compat.v1.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)
        else:
            raise NameError('No matching optimizer')

        # initialize global variables
        # TODO: why are these set to 1?
        session_conf = tf.compat.v1.ConfigProto(
             intra_op_parallelism_threads=1,
             inter_op_parallelism_threads=1)
        self.sess = tf.compat.v1.Session(config=session_conf)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1)

    def feed_dict(self, stim, resp=None, state=None, indices=None, learning_rate=None):
        """Generates the feed dict that is passed to the training step.

        :param stim:
        :param resp:
        :param state:
        :param indices:
        :param learning_rate:

        :return: Feed dict.
        """
        if indices is None:
            indices = np.arange(stim.shape[0])

        d = {self.stim: stim[indices, :, :]}

        if not (resp is None):
            d[self.resp] = resp[indices, :, :]
        if not (state is None):
            d[self.state] = state[indices, :, :]

        if not (learning_rate is None):
            d[self.learning_rate] = learning_rate

        return d

    def save(self):
        """Saves a checkpoint of the model. Only keeps most recent, overwriting existing."""
        if not self.log_dir.exists():
            self.log_dir.mkdir(exist_ok=True, parents=True)

        filename = self.log_dir / (self.log_id + '-model.ckpt')
        self.saver.save(self.sess, str(filename))

    def load(self):
        """Loads the saved model checkpoint."""
        filename = self.log_dir / (self.log_id + '-model.ckpt')
        self.saver.restore(self.sess, str(filename))

    def train(self, stim, resp, learning_rate=0.01, max_iter=300, eval_interval=30, batch_size=None,
              early_stopping_steps=5, early_stopping_tolerance=5e-4, print_iter=True, state=None):
        """Custom training loop for the net.

        :param stim:
        :param resp:
        :param learning_rate:
        :param max_iter:
        :param eval_interval:
        :param batch_size:
        :param early_stopping_steps:
        :param early_stopping_tolerance:
        :param print_iter:
        :param state:
        """
        # save a baseline
        self.save()

        # samples used for training
        train_indices = np.arange(resp.shape[0])

        # by default batch size equals the size of the training data
        if batch_size is None:
            batch_size = len(train_indices)

        log.info(f'Training with batch_size={batch_size}, LR={learning_rate}, max_iter={max_iter}, '
                 f'early_stopping_steps={early_stopping_steps}, early_stopping_tolerance={early_stopping_tolerance}, optimizer={self.optimizer}.')

        with self.sess.as_default():

            # indices for this batch
            batch_indices = np.arange(0, batch_size)

            # evaluate loss before any training
            if len(self.train_loss) == 0:
                train_dict = self.feed_dict(stim, resp=resp, state=state, indices=train_indices[batch_indices],
                                            learning_rate=learning_rate)
                initial_loss = self.loss.eval(feed_dict=train_dict)
                self.train_loss.append(initial_loss)
                self.best_loss = initial_loss
                self.iteration = [-1]

                if print_iter:
                    log.info("Initial loss=%8.7f", initial_loss)
            
            for epoch_num in itertools.count():
                if max_iter is not None and epoch_num >= max_iter:
                    break

                # update
                train_dict = self.feed_dict(stim, resp=resp, state=state, indices=train_indices[batch_indices],
                                            learning_rate=learning_rate)
                self.train_step.run(feed_dict=train_dict)

                # evaluate loss
                if np.mod(epoch_num, eval_interval) == 0:
                    # tick the progress indicator
                    nems0.utils.progress_fun()
                    self.iteration.append(epoch_num)
                    train_loss = self.loss.eval(feed_dict=train_dict)
                    self.train_loss.append(train_loss)

                    if epoch_num == 0:
                        self.best_loss = train_loss

                    if print_iter:
                        # vary significant digits based on tolerance
                        n_digits = math.ceil(abs(math.log(early_stopping_tolerance, 10))) + 2
                        log.info(f'{epoch_num:04d} loss={train_loss:.{n_digits}f}, '
                                 f'delta={train_loss - self.best_loss:+.{n_digits}f}')

                    # early stopping / saving
                    if early_stopping_steps > 0:
                        if self.best_loss > train_loss:
                            self.best_loss, self.best_loss_index = train_loss, len(self.train_loss)
                            self.save()

                        # early stopping for > 5 non improving iterations
                        # though partly redundant with tolerance early stopping, catches nans
                        if self.best_loss_index <= len(self.train_loss) - early_stopping_steps:
                            log.info(f'Best epoch > {early_stopping_steps} iterations ago, stopping early!')
                            break

                        # early stopping for not exceeding tolerance
                        elif np.all(abs(np.diff(np.array(self.train_loss[-early_stopping_steps - 1:]))) <
                                    early_stopping_tolerance):
                            log.info(f'{early_stopping_steps} epochs without significant improvement, stopping early!')
                            break

                # update batch
                batch_indices = batch_indices + batch_size
                if batch_indices[-1] > len(train_indices) - 1:
                    np.random.shuffle(train_indices)
                    batch_indices = np.arange(batch_size)

            # load the best loss
            if early_stopping_steps > 0:
                self.load()

            # record the last iter step
            self.last_iter = epoch_num

    def predict(self, stim, state=None, sess=None):
        """Generates a prediction from the net.

        :param stim:
        :param state:
        :param sess: TF session.

        :return: A prediction.
        """
        if sess is None:
            sess = self.sess

        with sess.as_default():
            return self.pred.eval(feed_dict=self.feed_dict(stim, state=state))

    def eval_to_layer(self, stim=None, layer_idx=None, state=None, sess=None):
        """Evaluates the net at a specific layer.

        :param stim:
        :param layer_idx:
        :param state:
        :param sess: TF session.

        :return:
        """
        if layer_idx is None:
            layer_idx = len(self.layers) - 1

        if sess is None:
            sess = self.sess

        with sess.as_default():
            return self.layers[layer_idx]['Y'].eval(feed_dict=self.feed_dict(stim, state=state))

    def layer_vals(self, sess=None):
        """Get matrix values out TF variables.

        :param sess: TF session.

        :return: list of layer dictionaries.
        """
        layer_keys = [
            'W',
            'b',
            'm',
            's',
            'g',
            'd',
            'gain',
            'f1',
            'delay',
            'u',
            'tau',
            'base',
            'amplitude',
            'kappa',
            'shift',
        ]

        if sess is None:
            sess = self.sess

        with sess.as_default():

            layers = [{layer_key: layer[layer_key].eval() for layer_key in layer_keys if layer_key in layer}
                      for layer in self.layers]

            return layers


class MultiNet(Net):

    def __init__(self, nets, log_dir=None, log_id=None):

        self.nets = nets
        self.weights = np.ones(len(nets)) / len(nets)
        self.D = nets[0].D

        # other misc parameters
        self.train_loss = []
        self.iteration = []
        self.learning_rate = tf.placeholder('float32')
        self.best_loss = 1e100
        self.best_loss_index = 0

        # directory and file info
        if log_dir is None:
            self.log_dir = '/Users/svnh2/Desktop/projects/scratch'
        else:
            self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if log_id is None:
            self.log_id = 'seed' + str(self.nets[0].seed)

    def build(self):

        for i in range(len(self.nets)):
            self.nets[i].build(initialize=False)
        #             if i == 0:
        #                 self.Y = self.nets[i].Y * self.weights[i]
        #             else:
        #                 self.Y = self.Y + self.nets[i].Y * self.weights[i]

        self.Y = self.nets[0].Y * self.weights[0] + \
            self.nets[1].Y * self.weights[1]

        # loss
        self.loss = tf.reduce_mean(tf.square(self.D - self.Y))

        # gradient optimizer
        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # initialize global variables
        self.initialize()

    def feed_dict(self, stim, resp=None, indices=None, learning_rate=None):

        assert(len(self.nets) == len(stim))

        if indices is None:
            indices = np.arange(stim[0].shape[0])

        d = {}
        for i in range(len(stim)):
            d[self.nets[i].F] = stim[i][indices, :, :]

        if not (resp is None):
            for i in range(len(self.nets)):
                d[self.nets[i].D] = resp[indices, :, :]

        if not (learning_rate is None):
            d[self.learning_rate] = learning_rate

        return d

    def layer_vals(self):

        nets = []
        for i in range(len(self.nets)):
            nets.append(self.nets[i].layer_vals(sess=self.sess))

        return nets
