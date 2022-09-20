import numpy as np
from numba import njit

from nems.registry import layer
from nems.layers.base import Layer, Phi, Parameter, require_shape
from nems.distributions import Normal, HalfNormal


class ShortTermPlasticity(Layer):

    def __init__(self, fs=100, quick_eval=True, crosstalk=0, x0=None,
                 dep_only=True, chunksize=5, reset_signal=None,
                 **kwargs):
        """TODO: docs

        TODO: additional context.
        Need @SVD's help filling in all these docs, some parameter renaming
        would likely be helpful as well.
        
        Parameters
        ----------
        shape : tuple.
            TODO: what's the shape syntax? Only one dim, multiple? What do
                  the value(s) represent?
        fs : int; default=100.
            Sampling rate. TODO: explain what this is used for.
        quick_eval : bool; default=True.
            Determines which implementation of the STP algorithm to use.
            TODO: explain why some one might want to use `quick_eval=False`
            (or if there's no good reason, just get rid of it). I think for now
            it's just b/c faciliatation isn't implemented?
        crosstalk : int; default=0;
            TODO: explain what this does.
        x0 : TODO: type?; optional.
            TODO: explain what this does.
        dep_only : bool; default=True.
            TODO: explain what this does.
        chunksize : int; default=5.
            TODO: explain what this does.
        reset_signal : TODO: type?; optional.
            TODO: explain what this does.
        
        """

        require_shape(self, kwargs, minimum_ndim=1)
        self.fs = fs
        self.quick_eval = quick_eval
        self.crosstalk = crosstalk
        self.x0 = x0
        self.dep_only = dep_only
        self.chunksize = chunksize
        self.reset_signal = reset_signal
        self._reset_times = None
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
        # TODO: base this on `self.dep_only`, `self.quick_eval`. Currently hard
        #       codes no facilitation.
        u_sd = np.full(shape=self.shape, fill_value=0.1)
        u_prior = HalfNormal(u_sd)

        # TODO: is setting this based on sampling rate really necessary?
        #       if so, explain why
        tau_min = max(0.0001, 2/self.fs)
        tau_bounds = (tau_min, np.inf)
        u_bounds = (1e-6, np.inf)

        u = Parameter('u', shape=self.shape, prior=u_prior, bounds=u_bounds)
        tau = Parameter('tau', shape=self.shape, prior=tau_prior,
                        bounds=tau_bounds)

        return Phi(u, tau)

    # Authors: SVD, Menoua K.
    # Revision by: JRP.
    def evaluate(self, input):
        """TODO: docs

        TODO: is there a succint equation we can put here to represent what
              STP is doing?

        TODO: brief explanation of what the Layer is doing to inputs.

        """

        u, tau = self.get_parameter_values()

        if (u > 0) and np.all(input > 0):
            # Can skip checking `imu > 0` in inner loop of quick_eval.
            # Saves a lot of time for large arrays / small chunk size.
            skip_sign_check = True
        else:
            skip_sign_check = False

        # TODO: get rid of need for transpose here
        tstim = input.T
        s = tstim.shape

        # Input shape should not change (otherwise other parameters would
        # break anyway), so this only needs to be calculated on the first
        # evaluation. Saves almost a millisecond per eval (formerly per loop).
        # TODO: profile memory usage on this. Only saves 10 microseconds
        #       (for the arange version), so if it's a huge memory increase then
        #       this may not be worth caching.
        if self._reset_times is None:
            if self.reset_signal is None:
                # convert chunksize from sec to bins
                chunksize = int(self.chunksize * self.fs)
                reset_times = np.arange(0, s[1] + chunksize - 1, chunksize)
            else:
                reset_times = np.argwhere(self.reset_signal[0, :])[:, 0]
                reset_times = np.append(reset_times, s[1])
            self._reset_times = reset_times
        else:
            reset_times = self._reset_times

        tstim[np.isnan(tstim)] = 0
        if self.x0 is not None:
            tstim -= np.expand_dims(self.x0, axis=1)

        # TODO: Is this necessary? If it's just to remove negative values, would
        #       shifting (and then unshifting at the end) be sufficient? 
        #       Clipping values like this tends to make optimization harder.
        tstim[tstim < 0] = 0

        # TODO: specify parameters in bin units in the first place so that we
        #       don't need to bother with this? Can still add a separate
        #       `bins_to_seconds` method for convenience. Same with chunksize.
        #       (so that ideally `fs` isn't needed at all, except as an arg to
        #        the convenience method).
        # convert u & tau units from sec to bins
        taui = tau * self.fs
        ui = u / self.fs * 100

        # TODO : allow >1 STP channel per input?

        # TODO: only need this for `quick_eval=False`, remove when that gets removed.
        stim_out = tstim  # allocate scaling term

        if self.crosstalk:
            # assumes dim of u is 1 !
            tstim = np.mean(tstim, axis=0, keepdims=True)
        if len(ui.shape) == 1:
            ui = ui[..., np.newaxis]      # ~20x faster than expand_dims
            taui = taui[..., np.newaxis]

        if self.quick_eval:
            a = 1 / taui
            x = ui * tstim

        for i in range(u.shape[0]):
            if self.quick_eval:
                td = np.empty(shape=s)  # ~40x faster than np.ones
                x0, imu0 = 0.0, 0.0

                for j in range(len(reset_times) - 1):
                    si = slice(reset_times[j], reset_times[j + 1])
                    xi = x[:, si]

                    ix = self.cumulative_integral_trapz(a + xi) + a + (x0 + xi[:, :1]) / 2

                    mu = np.exp(ix)
                    imu = self.cumulative_integral_trapz(mu * xi) + (x0 + mu[:, :1] * xi[:, :1]) / 2 + imu0

                    # TODO: doesn't mu have to be > 0 by definition? Why bother with
                    #       the bitwise_and? repeating this on each inner loop adds a few
                    #       milliseconds to the eval time
                    # ff = np.bitwise_and(mu > 0, imu > 0)
                    # TODO: wondering if the same might be true for imu... looks like it 
                    #       should be true if x is positive everywhere? If so, can check that
                    #       one time and skip the indexing on all the loops. Not that much
                    #       time for one index by it adds up. For a 80000 time bin signal
                    #       with fs100 and chunksize 5, doing imu[ff] and mu[ff] on each
                    #       inner loops adds a total of about half a millisecond to the eval.
                    #       Removing this and the bitwise and would be another big speedup.
                    if not skip_sign_check:
                        ff = (imu > 0)
                        # TODO: why ones? do all values get replaced? if so, use empty (faster)
                        #       if not, why does a default value of 1 make sense?
                        _td = np.ones_like(mu)
                        _td[ff] = 1 - np.exp(np.log(imu[ff]) - np.log(mu[ff]))
                    else:
                        _td = 1 - np.exp(np.log(imu) - np.log(mu))
                    
                    td[:, si] = _td
                    x0 = xi[:, -1:]
                    imu0 = imu[:, -1:] / mu[:, -1:]

                # TODO: Neither of these explanations makes sense to me. Ask for clarification.
                # shift td forward in time by one to allow STP to kick in after the stimulus changes (??)
                # offset depression by one to allow transients
                stim_out = tstim * np.concatenate(
                    # Oddly enough, zeros + 1 is twice as fast as using ones
                    # for this size of array (1-6ish usually)
                    [np.zeros((td.shape[0], 1)) + 1, td[:, :-1]], axis=1
                    )
            else:
                # iterate over channels
                td = self.numba_loop(ui, taui, tstim, i)

                if self.crosstalk:
                    stim_out *= np.expand_dims(td, 0)
                else:
                    stim_out[i, :] *= td

        if np.sum(np.isnan(stim_out)):
            raise ValueError('nan value in STP stim_out')

        stim_out[np.isnan(input.T)] = np.nan

        return stim_out.T

    # TODO: add numba to dependencies if we decide to keep this version
    #       (so far, fastest by a wide margin)
    # TODO: refactor rest of evaluate so that (ideally) the entire thing
    #       can be pre-compiled instead of just the loops. But these are the
    #       most important part.
    @staticmethod
    @njit
    def numba_loop(ui, taui, tstim, i):
        a = 1 / taui[i][0]
        ustim = 1.0 / taui[i] + ui[i] * tstim[i, :]
        s = ustim.shape[0]
        td = [1]  # initialize dep state vector

        if ui[i] == 0:
            # passthru, no STP, preserve stim_out = tstim
            depression = None
        elif ui[i] > 0:
            depression = True
        else:
            depression = False

        if depression is None:
            pass
        else:
            # depression
            for tt in range(1, s):
                delta = a - td[-1] * ustim[tt - 1]
                next_td = td[-1] + delta

                if depression and next_td < 0:
                    # TODO: should this be a hyperparameter instead? ex: if signal
                    #       is normalized s.t. mean = 0, this has a different meaning.
                    next_td = 0
                elif not depression and next_td > 5:
                    # TODO: why 5?
                    td[tt] = 5
                td.append(next_td)

        return np.array(td)


    @staticmethod
    def cumulative_integral_trapz(y, dx=1.0, axis=1):
        """Cumulative integral of y(x) using the trapezoid method.
        
        Compare to `scipy.integrate.cumtrapz`. This method is slightly faster
        due to supporting fewer options.
        
        Parameters
        ----------
        y : np.ndarray.
            Values to integrate.
        dx : int; default=1.0.
            Spacing between elements of `y`.
        axis : int; default=1.
            Specifies the axis over which to compute the cumulative sum.

        Returns
        -------
        np.ndarray
            Integrated values, with the same shape as `y`.
        
        """
        # TODO: profile memory usage.
        # TODO: move this to nems.tools?
        y = (y[:, :-1] + y[:, 1:]) / 2.0
        # TODO: Is the padding ever *not* done on the time axis? If not,
        #       concat is ~10x faster and does the same thing.
        #       With this version, even a little faster than
        #       `scipy.integrate.cumtrapz` and does the same thing.
        #       Also set to only use init=0 for further improvement, since
        #       that's the only value that STP uses.
        y = np.concatenate([np.zeros((y.shape[0], 1)), y], axis=1)
        y = np.multiply(np.cumsum(y, axis=axis), dx)

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
