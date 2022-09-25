import numpy as np
from numba import njit

from nems.registry import layer
from nems.layers.base import Layer, Phi, Parameter, require_shape
from nems.distributions import HalfNormal
from nems.tools.arrays import one_or_more_negative


class ShortTermPlasticity(Layer):

    def __init__(self, quick_eval=False, crosstalk=0, dep_only=True,
                 chunksize=500, **kwargs):
        """TODO: docs

        TODO: additional context.
        Need @SVD's help filling in all these docs, some parameter renaming
        would likely be helpful as well.
        
        Parameters
        ----------
        shape : tuple.
            TODO: what's the shape syntax? Only one dim, multiple? What do
                  the value(s) represent?
        quick_eval : bool; default=True.
            Determines which implementation of the STP algorithm to use.
            TODO: explain why some one might want to use `quick_eval=False`
            (or if there's no good reason, just get rid of it). I think for now
            it's just b/c faciliatation isn't implemented?
        crosstalk : int; default=0;
            TODO: explain what this does.
            TODO: or remove if this isn't used any more?
        dep_only : bool; default=True.
            TODO: explain what this does.
        chunksize : int; default=5.
            TODO: explain what this does.
            NOTE: used to be in units of seconds, now it should be specified
                  in units of bins.

        """

        require_shape(self, kwargs, minimum_ndim=1)
        self.quick_eval = quick_eval
        self.crosstalk = crosstalk
        self.dep_only = dep_only
        self.chunksize = chunksize
        super().__init__(**kwargs)


    def initial_parameters(self):
        """TODO: docs
        
        Layer parameters
        ----------------
        u : np.ndarray.
            TODO: explain purpose.
            TODO: expected shape?
            Prior: HalfNormal(sd=0.1)
            Bounds: (0.0001, np.inf)   TODO: should this use epsilon instead?
        tau : np.ndarray.
            TODO: explain purpose.
            TODO: expected shape?
            Prior: HalfNormal(sd=0.1)
            Bounds: (1e-6, np.inf)     TODO: should this use epsilon instead?
        
        """
        tau_sd = np.full(shape=self.shape, fill_value=0.1)
        tau_prior = HalfNormal(tau_sd)
        u_sd = np.full(shape=self.shape, fill_value=0.1)
        u_prior = HalfNormal(u_sd)

        tau_bounds = (0.0001, np.inf)
        if self.dep_only or self.quick_eval:
            u_bounds = (1e-6, np.inf)
        else:
            u_bounds = (-np.inf, np.inf)

        u = Parameter('u', shape=self.shape, prior=u_prior, bounds=u_bounds)
        tau = Parameter('tau', shape=self.shape, prior=tau_prior,
                        bounds=tau_bounds)

        return Phi(u, tau)


    def bins_to_seconds(self, fs):
        """Get parameter values in units of seconds based on sampling rate."""
        
        u, tau = self.get_parameter_values()
        u_seconds = (u/100)*fs  # TODO: why is there a 100 here?
        tau_seconds = tau/fs

        return u, tau

        # TODO: Previous converesion, delete after review by SVD
        # convert u & tau units from sec to bins
        taui = tau * self.fs
        ui = u / self.fs * 100


    # Authors: SVD, Menoua K.
    # Revision by: JRP.
    def evaluate(self, input):
        """TODO: docs

        TODO: is there a succint equation we can put here to represent what
              STP is doing?

        TODO: brief explanation of what the Layer is doing to inputs.

        """
        
        if one_or_more_negative(input):
            raise ValueError(
                "STP only supports inputs with all non-negative values. "
                "We recommend normalizing inputs between 0 and 1, and/or "
                "preceeding STP with a layer that guarantees positive outputs."
            )

        try:
            n_times, n_channels = input.shape
        except ValueError as e:
            if "not enough values to unpack" in str(e):
                raise ValueError(
                    "STP only supports 2D inputs with shape (T, N), where T "
                    "is the number of time bins and N is the number of output "
                    "channels."
                )

        u, tau = self.get_parameter_values()
        # TODO: temporary fudge factor to deal with scaling priors, since
        #       half-normal can only specify sd. remove this after implementing
        #       a proper parameter scaling system.
        tau *= 100

        # TODO: Refactor if still using, otherwise remove.
        # if self.crosstalk:
        #     # assumes dim of u is 1 !
        #     tstim = np.mean(input, axis=0, keepdims=True)

        if self.quick_eval:
            a = 1 / tau[np.newaxis, ...]
            x = u[np.newaxis, ...] * input
            nonzero = (x > 0)
            # Values that aren't set get multiplied by 0, so value of empty
            # entries doesn't matter and this is much faster than initializing
            # to ones.
            depression_per_bin = np.empty_like(input) 
            x0, imu0 = 0.0, 0.0

            i = 0
            j = self.chunksize
            while i < (n_times - 1):
                # Approximates analytical solution to differential equation:
                # TODO: fill in diffeq (or maybe put this in docstring for
                #       quick_eval kwarg instead).
                xi = x[i:j, :]
                x1 = xi[:1, :]

                ix = self._cumulative_integral_trapz(a + xi) + a + (x0 + x1)/2
                mu = np.exp(ix)
                mu1 = mu[:1, :]
                imu = self._cumulative_integral_trapz(mu * xi) \
                        + (x0 + mu1*x1)/2 + imu0

                _nonzero = nonzero[i:j, :]
                depression_per_bin[i:j, :][_nonzero] = \
                        1 - imu[_nonzero] / mu[_nonzero]
                x0 = xi[-1:, :]
                imu0 = imu[-1:, :] / mu[-1:, :]

                i += self.chunksize
                j += self.chunksize

            # TODO: Neither of these explanations makes sense to me. Ask for clarification.
            # shift depression forward in time by one to allow STP to kick in after the stimulus changes (??)
            # offset depression by one to allow transients
            out = np.multiply(
                input, np.concatenate(
                    # Oddly enough, zeros + 1 is twice as fast as using ones
                    # for this size of array (1-6ish usually)
                    [np.zeros((1, n_channels)) + 1, depression_per_bin[:-1, :]],
                    axis=0
                    ),
                )

        else:
            out = np.empty_like(input)
            for i in range(n_channels):
                depression_per_bin = self._inner_loop(u, tau, input, i)

                # TODO: refactor if still using, otherwise remove
                # if self.crosstalk:
                #     stim_out *= np.expand_dims(td, 0)
                out[:, i] = np.multiply(
                    input[:, i], depression_per_bin,
                    )

        return out


    @staticmethod
    @njit
    def _inner_loop(u, tau, input, i):
        """TODO: docs.
        
        Internal for `evaluate`.
        
        """

        a = 1 / tau[i]
        ustim = 1.0 / tau[i] + u[i] * input[:, i]
        s = ustim.shape[0]
        td = [1]  # initialize dep state vector

        if u[i] == 0:
            # passthru, no STP, preserve stim_out = tstim
            pass
        else:
            if u[i] > 0:
                depression = True
            else:
                depression = False

            for tt in range(1, s):
                delta = a - td[-1] * ustim[tt - 1]
                next_td = td[-1] + delta

                if depression and next_td < 0:
                    next_td = 0
                elif not depression and next_td > 5:
                    # TODO: why 5?  -- hyperparameter?
                    #       avoids explosions, and it's big enough that it's
                    #       essentially "infinity" in biological terms.
                    td[tt] = 5
                td.append(next_td)

        return np.array(td)


    @staticmethod
    def _cumulative_integral_trapz(y):
        """Cumulative integral of y(x) using the trapezoid method.
        
        Compare to `scipy.integrate.cumtrapz`. This method is slightly faster
        due to supporting fewer options, with fixed parameters:
            `x = None`
            `dx = 1`
            `axis = 0`
            `initial = 0`
        
        Parameters
        ----------
        y : np.ndarray.
            Values to integrate.

        Returns
        -------
        np.ndarray
            Integrated values, with the same shape as `y`.
        
        """
        y = (y[:-1, :] + y[1:, :]) / 2.0
        y = np.concatenate([np.zeros((1, y.shape[1])), y], axis=0)
        y = np.cumsum(y, axis=0)

        return y

    @layer('stp')
    def from_keyword(keyword):
        """TODO: docs"""
        shape = None

        options = keyword.split('.')
        for op in options:
            if ('x' in op) and (op[0].isdigit()):
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])

        return ShortTermPlasticity(shape=shape)

    def as_tensorflow_layer(self, **kwargs):

        """
        ported from nems0.tf.layers
        TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer
    

        parent_x0 = self.x0
        fs = self.fs
        crosstalk = self.crosstalk
        if crosstalk:
            # TODO: Remove this after implementing crosstalk
            raise NotImplemented(
                'STP(..., crosstalk=True) is not yet supported for the '
                'TensorFlow backend.'
                )
        parent_chunksize = self.chunksize
        reset_signal = self.reset_signal

        @tf.function
        def _cumtrapz(x, dx=1., initial=0.):
            x = (x[:, :-1] + x[:, 1:]) / 2.0
            x = tf.pad(
                x, ((0, 0), (1, 0), (0, 0)), constant_values=initial
                )

            return tf.cumsum(x, axis=1) * dx


        class ShortTermPlasticityTF(NemsKerasLayer):
            def call(self, inputs):
                # TODO docs.
                # Implementation developed by Menoua K.

                # TODO: Why are different levels of precision hard-coded in
                #       different places? Ideally should change all dtype=
                #       to inputs.dtype to work with the new consistent-dtype
                #       system.
                _zero = tf.constant(0.0, dtype='float32')
                _nan = tf.constant(0.0, dtype='float32')

                s = inputs.shape
                tstim = tf.where(tf.math.is_nan(inputs), _zero, inputs)
                tstim = tf.nn.relu(tstim)

                if parent_x0 is not None:  # x0 should be tf variable to avoid retraces
                    # TODO: is this expanding along the right dim? tstim dims: (None, time, chans)
                    tstim = tstim - tf.expand_dims(parent_x0, axis=1)

                # convert a & tau units from sec to bins
                ui = tf.math.abs(tf.reshape(self.u, (1, -1))) / fs * 100
                taui = tf.math.abs(tf.reshape(self.tau, (1, -1))) * fs

                # convert chunksize from sec to bins
                chunksize = int(parent_chunksize * fs)

                if crosstalk:
                    # assumes dim of u is 1 !
                    tstim = tf.math.reduce_mean(tstim, axis=0, keepdims=True)

                ui = tf.expand_dims(ui, axis=0)
                taui = tf.expand_dims(taui, axis=0)

                a = tf.cast(1.0 / taui, 'float64')
                x = ui * tstim

                # TODO: move these outside __call__ similar to revision of scipy
                if reset_signal is None:
                    reset_times = tf.range(0, s[1] + chunksize - 1, chunksize)
                else:
                    reset_times = tf.where(reset_signal[0, :])[:, 0]
                    reset_times = tf.pad(
                        reset_times, ((0, 1),), constant_values=s[1]
                        )

                td = []
                x0, imu0 = 0.0, 0.0
                for j in range(reset_times.shape[0] - 1):
                    xi = tf.cast(
                        x[:, reset_times[j]:reset_times[j + 1], :],
                        'float64'
                        )
                    ix = _cumtrapz(a + xi, dx=1, initial=0) \
                         + a + (x0 + xi[:, :1]) / 2.0

                    mu = tf.exp(ix)
                    imu = _cumtrapz(mu * xi, dx=1, initial=0) \
                          + (x0 + mu[:, :1] * xi[:, :1]) / 2.0 + imu0

                    valid = tf.logical_and(mu > 0.0, imu > 0.0)
                    mu = tf.where(valid, mu, 1.0)
                    imu = tf.where(valid, imu, 1.0)
                    _td = 1 - tf.exp(tf.math.log(imu) - tf.math.log(mu))
                    _td = tf.where(valid, _td, 1.0)

                    x0 = xi[:, -1:]
                    imu0 = imu[:, -1:] / mu[:, -1:]
                    td.append(tf.cast(_td, 'float32'))
                td = tf.concat(td, axis=1)

                ret = tstim * tf.pad(
                        td[:, :-1, :], ((0, 0), (1, 0), (0, 0)),
                        constant_values=1.0
                        )
                ret = tf.where(tf.math.is_nan(inputs), _nan, ret)

                return ret

        return ShortTermPlasticityTF(self, **kwargs)


class STP(ShortTermPlasticity):
    pass
