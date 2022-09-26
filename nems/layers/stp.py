import numpy as np

from nems.registry import layer
from nems.layers.base import Layer, Phi, Parameter
from nems.layers.tools import require_shape, pop_shape
from nems.distributions import Normal, HalfNormal


def _cumtrapz(x, dx=1., initial=0., axis=1):
    x = (x[:, :-1] + x[:, 1:]) / 2.0
    x = np.pad(x, ((0, 0), (1, 0)), 'constant', constant_values=(initial, initial))
    # x = tf.pad(x, ((0, 0), (1, 0), (0, 0)), constant_values=initial)
    return np.cumsum(x, axis=axis) * dx


class ShortTermPlasticity(Layer):

    def __init__(self, fs=100, quick_eval=True, crosstalk=0, x0=None,
                 dep_only=True, chunksize=5, reset_signal=None,
                 **kwargs):
        require_shape(self, kwargs, minimum_ndim=1)

        self.fs = fs
        self.quick_eval = quick_eval
        self.crosstalk = crosstalk
        self.x0 = x0
        self.dep_only = dep_only
        self.chunksize = chunksize
        self.reset_signal = reset_signal

        super().__init__(**kwargs)

    def initial_parameters(self):

        taumean = np.full(shape=self.shape, fill_value=0.1)
        tauprior = HalfNormal(taumean)
        umean = np.full(shape=self.shape, fill_value=0.1)
        uprior = HalfNormal(umean)

        tau_bounds = (0.0001, np.inf)
        u_bounds = (1e-6, np.inf)

        u = Parameter('u', shape=self.shape, prior=uprior, bounds=u_bounds)
        tau = Parameter('tau', shape=self.shape, prior=tauprior, bounds=tau_bounds)

        return Phi(u, tau)


    def evaluate(self, input):
        """
        STP core function
        ported from nems0.modules.stp

        def _stp(X, u, tau, x0=None, crosstalk=0, fs=1, reset_signal=None, quick_eval=False, dep_only=False,
                 chunksize=5):
        """

        u, tau = self.get_parameter_values()

        # TODO: get rid of need for transpose here
        tstim = input.astype('float64').T
        s = tstim.shape
        u = u.astype('float64')
        tau = tau.astype('float64')

        tstim[np.isnan(tstim)] = 0
        if self.x0 is not None:
            tstim -= np.expand_dims(self.x0, axis=1)

        tstim[tstim < 0] = 0

        # TODO: deal with upper and lower bounds on dep and facilitation parms
        #       need to know something about magnitude of inputs???
        #       move bounds to fitter? slow

        # limits, assumes input (X) range is approximately -1 to +1
        # TODO: add support for facilitation???
        if self.dep_only or self.quick_eval:
            ui = np.abs(u.copy())
        else:
            ui = u.copy()

        # force tau to have positive sign (probably better done with bounds on fitter)
        taui = np.absolute(tau.copy())
        taui[taui < 2 / self.fs] = 2 / self.fs

        # ui[ui > 1] = 1
        # ui[ui < -0.4] = -0.4

        # avoid ringing if combination of strong depression and
        # rapid recovery is too large
        # rat = ui**2 / taui

        # MK comment out
        # ui[rat>0.1] = np.sqrt(0.1 * taui[rat>0.1])

        # taui[rat>0.08] = (ui[rat>0.08]**2) / 0.08
        # print("rat: %s" % (ui**2 / taui))

        # convert u & tau units from sec to bins
        taui = taui * self.fs
        ui = ui / self.fs * 100

        # convert chunksize from sec to bins
        chunksize = int(self.chunksize * self.fs)

        # TODO : allow >1 STP channel per input?

        # go through each stimulus channel
        stim_out = tstim  # allocate scaling term

        if self.crosstalk:
            # assumes dim of u is 1 !
            tstim = np.mean(tstim, axis=0, keepdims=True)
        if len(ui.shape) == 1:
            ui = np.expand_dims(ui, axis=1)
            taui = np.expand_dims(taui, axis=1)

        #ui=ui.T
        #taui=taui.T

        for i in range(0, len(u)):
            if self.quick_eval:

                a = 1 / taui
                x = ui * tstim

                if self.reset_signal is None:
                    reset_times = np.arange(0, s[1] + chunksize - 1, chunksize)
                else:
                    reset_times = np.argwhere(self.reset_signal[0, :])[:, 0]
                    reset_times = np.append(reset_times, s[1])

                td = np.ones_like(x)
                x0, imu0 = 0., 0.
                for j in range(len(reset_times) - 1):
                    si = slice(reset_times[j], reset_times[j + 1])
                    xi = x[:, si]

                    ix = _cumtrapz(a + xi, dx=1, initial=0, axis=1) + a + (x0 + xi[:, :1]) / 2

                    mu = np.exp(ix)
                    imu = _cumtrapz(mu * xi, dx=1, initial=0, axis=1) + (x0 + mu[:, :1] * xi[:, :1]) / 2 + imu0

                    ff = np.bitwise_and(mu > 0, imu > 0)
                    _td = np.ones_like(mu)
                    _td[ff] = 1 - np.exp(np.log(imu[ff]) - np.log(mu[ff]))
                    td[:, si] = _td

                    x0 = xi[:, -1:]
                    imu0 = imu[:, -1:] / mu[:, -1:]

                # shift td forward in time by one to allow STP to kick in after the stimulus changes (??)
                # stim_out = tstim * td

                # offset depression by one to allow transients
                stim_out = tstim * np.pad(td[:, :-1], ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))
            else:
                a = 1 / taui[i]
                ustim = 1.0 / taui[i] + ui[i] * tstim[i, :]
                # ustim = ui[i] * tstim[i, :]
                td = np.ones_like(ustim)  # initialize dep state vector

                if ui[i] == 0:
                    # passthru, no STP, preserve stim_out = tstim
                    pass
                elif ui[i] > 0:
                    # depression
                    for tt in range(1, s[1]):
                        # td = di[i, tt - 1]  # previous time bin depression
                        # delta = (1 - td) / taui[i] - ui[i] * td * tstim[i, tt - 1]
                        # delta = 1/taui[i] - td * (1/taui[i] - ui[i] * tstim[i, tt - 1])
                        # then a=1/taui[i] and ustim=1/taui[i] - ui[i] * tstim[i,:]
                        delta = a - td[tt - 1] * ustim[tt - 1]
                        td[tt] = td[tt - 1] + delta
                        if td[tt] < 0:
                            td[tt] = 0
                else:
                    # facilitation
                    for tt in range(1, s[1]):
                        delta = a - td[tt - 1] * ustim[tt - 1]
                        td[tt] = td[tt - 1] + delta
                        if td[tt] > 5:
                            td[tt] = 5

                if self.crosstalk:
                    stim_out *= np.expand_dims(td, 0)
                else:
                    stim_out[i, :] *= td

        if np.sum(np.isnan(stim_out)):
            raise ValueError('nan value in STP stim_out')

        # print("(u,tau)=({0},{1})".format(ui,taui))
        stim_out[np.isnan(input.T)] = np.nan

        return stim_out.T

    @layer('stp')
    def from_keyword(keyword):
        """TODO: docs"""
        options = keyword.split('.')
        shape = pop_shape(options)

        return ShortTermPlasticity(shape=shape)

    def as_tensorflow_layer(self, **kwargs):

        """
        ported from nems0.tf.layers
        TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        #u, tau = self.get_parameter_values()
        #u = u.astype('float32')
        #tau = tau.astype('float32')

        parent_x0 = self.x0
        fs = self.fs
        crosstalk = self.crosstalk
        parent_chunksize = self.chunksize
        reset_signal = self.reset_signal

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
                    raise NotImplemented
                    # assumes dim of u is 1 !
                    tstim = tf.math.reduce_mean(tstim, axis=0, keepdims=True)

                ui = tf.expand_dims(ui, axis=0)
                taui = tf.expand_dims(taui, axis=0)

                @tf.function
                def _cumtrapz(x, dx=1., initial=0.):
                    x = (x[:, :-1] + x[:, 1:]) / 2.0
                    x = tf.pad(x, ((0, 0), (1, 0), (0, 0)), constant_values=initial)
                    return tf.cumsum(x, axis=1) * dx

                a = tf.cast(1.0 / taui, 'float64')
                x = ui * tstim

                if reset_signal is None:
                    reset_times = tf.range(0, s[1] + chunksize - 1, chunksize)
                else:
                    reset_times = tf.where(reset_signal[0, :])[:, 0]
                    reset_times = tf.pad(reset_times, ((0, 1),), constant_values=s[1])

                td = []
                x0, imu0 = 0.0, 0.0
                for j in range(reset_times.shape[0] - 1):
                    xi = tf.cast(x[:, reset_times[j]:reset_times[j + 1], :], 'float64')
                    ix = _cumtrapz(a + xi, dx=1, initial=0) + a + (x0 + xi[:, :1]) / 2.0

                    mu = tf.exp(ix)
                    imu = _cumtrapz(mu * xi, dx=1, initial=0) + (x0 + mu[:, :1] * xi[:, :1]) / 2.0 + imu0

                    valid = tf.logical_and(mu > 0.0, imu > 0.0)
                    mu = tf.where(valid, mu, 1.0)
                    imu = tf.where(valid, imu, 1.0)
                    _td = 1 - tf.exp(tf.math.log(imu) - tf.math.log(mu))
                    _td = tf.where(valid, _td, 1.0)

                    x0 = xi[:, -1:]
                    imu0 = imu[:, -1:] / mu[:, -1:]
                    td.append(tf.cast(_td, 'float32'))
                td = tf.concat(td, axis=1)

                # ret = tstim * td
                # offset depression by one to allow transients
                ret = tstim * tf.pad(td[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=1.0)
                ret = tf.where(tf.math.is_nan(inputs), _nan, ret)

                return ret

            #def weights_to_values(self):
            #    values = self.parameter_values
            #    # TODO?? Remove extra dummy axis if one was added, and undo scaling.
            #    return values

        return ShortTermPlasticityTF(self, **kwargs)



class STP(ShortTermPlasticity):
    pass
