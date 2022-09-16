import numpy as np

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter, require_shape


class StateGain(Layer):
    
    state_arg = 'state'  # see Layer docs for details

    def __init__(self, **kwargs):
        """Docs TODO.
        
        Parameters
        ----------
        shape : 2-tuple of int.
            (size of last dimension of state, size of last dimension of input)

        Examples
        --------
        TODO
        
        """
        require_shape(self, kwargs, minimum_ndim=2)
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Docs TODO
        
        Layer parameters
        ----------------
        gain : TODO
            prior:
            bounds:
        offset : TODO
            prior:
            bounds:
        
        """
        zero = np.zeros(shape=self.shape)
        one = np.ones(shape=self.shape)

        gain_mean = zero.copy()
        gain_mean[0,:] = 1  # TODO: Explain purpose of this?
        gain_sd = one/20
        gain_prior = Normal(gain_mean, gain_sd)
        gain = Parameter('gain', shape=self.shape, prior=gain_prior)

        offset_mean = zero
        offset_sd = one
        offset_prior = Normal(offset_mean, offset_sd)
        offset = Parameter('offset', shape=self.shape, prior=offset_prior)
        
        return Phi(gain, offset)

    def evaluate(self, input, state):
        """Multiply and shift input(s) by weighted sums of state channels.
        
        Parameters
        ----------
        input : ndarray
            Data to be modulated by state, typically the output of a previous
            Layer.
        state : ndarray
            State data to modulate input with.

        """

        gain, offset = self.get_parameter_values()
        return np.matmul(state, gain) * input + np.matmul(state, offset)

    @layer('stategain')
    def from_keyword(keyword):
        """Construct StateGain from keyword.
        
        Keyword options
        ---------------
        {digit}x{digit} : specifies shape, (n state channels, n stim channels)
            n stim channels can also be 1, in which case the same weighted
            channel will be broadcast to all stim channels (if there is more
            than 1).
        
        See also
        --------
        Layer.from_keyword

        """
        # TODO: other options from old NEMS
        options = keyword.split('.')
        shape = None
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])

        return StateGain(shape=shape)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        class StateGainTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors, with state second.
                # TODO: Use tensor names to not require this arbitrary order.
                input = inputs[0]
                state = inputs[1]

                with_gain = tf.multiply(tf.matmul(state, self.gain), input)
                with_offset = with_gain + tf.matmul(state, self.offset)
                
                return with_offset

        return StateGainTF(self, **kwargs)

def _state_dexp(x, s, base_g, amplitude_g, kappa_g, offset_g, base_d, amplitude_d, kappa_d, offset_d, per_channel=False):
    '''
     Apparently, numpy is VERY slow at taking the exponent of a negative number
     https://github.com/numpy/numpy/issues/8233
     The correct way to avoid this problem is to install the Intel Python Packages:
     https://software.intel.com/en-us/distribution-for-python

     "current" version of sdexp. separate kappa/amp/base phis for gain/dc.
     So all parameters (g, d, base_g, etc.) are the same shape.

     parameters are pred_inputs X state_inputs
        _sg = _bg + _ag * tf.exp(-tf.exp(-tf.exp(_kg * (tf.expand_dims(s,3) - _og))))
    '''

    n_states = base_g.shape[0]
    n_inputs = base_g.shape[1]
    _n = np.newaxis
    if per_channel:
        for i in range(n_states):
            _sg = base_g[[0],i,_n] + amplitude_g[[0],i,_n] * \
                         np.exp(-np.exp(-np.exp(kappa_g[[0],i,_n]) * (s[i] - offset_g[[0],i,_n])))
            _sd = base_d[[0],i,_n] + amplitude_d[[0],i,_n] * \
                         np.exp(-np.exp(-np.exp(kappa_d[[0],i,_n]) * (s[i] - offset_d[[0],i,_n])))
            if i == 0:
               sg = _sg
               sd = _sd
            else:
               sg = np.concatenate((sg, _sg), axis=0)
               sd = np.concatenate((sd, _sd), axis=0)

        #import pdb; pdb.set_trace()
    else:
        sg = [np.sum(base_g[_n,:,i] + amplitude_g[_n,:,i] *
                         np.exp(-np.exp(-np.exp(kappa_g[_n,:,i]) * (s[:,:n_states] - offset_g[_n,:,i]))),
                         axis=1, keepdims=True) for i in range(n_inputs)]
        sd = [np.sum(base_d[_n,:,i] + amplitude_d[_n,:,i] *
                         np.exp(-np.exp(-np.exp(kappa_d[_n,:,i]) * (s[:,:n_states] - offset_d[_n,:,i]))),
                         axis=1, keepdims=True) for i in range(n_inputs)]
        sg = np.concatenate(sg, axis=1)
        sd = np.concatenate(sd, axis=1)

    return sg * x + sd, sg, sd


class StateDexp(Layer):
    """
    State-dependent modulation through exponential.
    Hacked by SVD from StateGain above. 2022-09-15
    """

    state_arg = 'state'  # see Layer docs for details

    def __init__(self, **kwargs):
        """Docs TODO.

        Parameters
        ----------
        shape : 2-tuple of int. (state_chans, input_chans), ie,
            (size of last dimension of state, size of last dimension of input)

        Examples
        --------
        TODO

        """
        require_shape(self, kwargs, minimum_ndim=2)
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Docs TODO

        Layer parameters
        ----------------
        base_g, amp_g, kappa_g, offset_g : TODO
            prior:
            bounds:
        base_d, amp_d, kappa_d, offset_d : TODO
            prior:
            bounds:

        """
        zeros = np.zeros(shape=self.shape)
        ones = np.ones(shape=self.shape)

        # init gain params
        base_mean_g = zeros.copy()
        base_sd_g = ones.copy()
        amp_mean_g = zeros.copy() + 0
        amp_mean_g[:, 0] = 1  # (1 / np.exp(-np.exp(-np.exp(0)))) # so that gain = 1 for baseline chan
        amp_sd_g = ones.copy() * 0.1
        kappa_mean_g = zeros.copy()
        kappa_sd_g = ones.copy() * 0.1
        offset_mean_g = zeros.copy()
        offset_sd_g = ones.copy() * 0.1

        # init dc params
        base_mean_d = zeros.copy()
        base_sd_d = ones.copy()
        amp_mean_d = zeros.copy() + 0
        amp_sd_d = ones.copy() * 0.1
        kappa_mean_d = zeros.copy()
        kappa_sd_d = ones.copy() * 0.1
        offset_mean_d = zeros.copy()
        offset_sd_d = ones.copy() * 0.1

        return Phi(
            Parameter('base_g', shape=self.shape, prior=Normal(base_mean_g, base_sd_g), bounds=(0,10)),
            Parameter('amp_g', shape=self.shape, prior=Normal(amp_mean_g, amp_sd_g), bounds=(0,10)),
            Parameter('kappa_g', shape=self.shape, prior=Normal(kappa_mean_g, kappa_sd_g), bounds=(-np.inf,np.inf)),
            Parameter('offset_g', shape=self.shape, prior=Normal(offset_mean_g, offset_sd_g), bounds=(-np.inf,np.inf)),
            Parameter('base_d', shape=self.shape, prior=Normal(base_mean_d, base_sd_d), bounds=(-10,10)),
            Parameter('amp_d', shape=self.shape, prior=Normal(amp_mean_d, amp_sd_d), bounds=(-10,10)),
            Parameter('kappa_d', shape=self.shape, prior=Normal(kappa_mean_d, kappa_sd_d), bounds=(-np.inf,np.inf)),
            Parameter('offset_d', shape=self.shape, prior=Normal(offset_mean_d, offset_sd_d), bounds=(-np.inf,np.inf))
        )

    def evaluate(self, input, state):
        """TODO: pull standalone function into class method"""
        """
        Parameters
        ----------
        input : ndarray
            Data to be modulated by state, typically the output of a previous
            Layer.
        state : ndarray
            State data to modulate input with.

        """

        #base_g, amp_g, kappa_g, offset_g, base_d, amp_d, kappa_d, offset_d = \
        #     self.get_parameter_values()

        return _state_dexp(input, state, *self.get_parameter_values())[0]

    @layer('sdexp')
    def from_keyword(keyword):
        """Construct sdexp from keyword.

        Keyword options
        ---------------
        {digit}x{digit} : specifies shape, (n state channels, n stim channels)
            n stim channels can also be 1, in which case the same weighted
            channel will be broadcast to all stim channels (if there is more
            than 1).

        See also
        --------
        Layer.from_keyword

        """
        """TODO: migrate in from NEMS_LBHB.modules"""
        # TODO: other options from old NEMS
        options = keyword.split('.')
        shape = None
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])

        return StateDexp(shape=shape)

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        """TODO: migrate in from NEMS_LBHB.modules"""
        import tensorflow as tf
        from nems.backends.tf.layer_tools import NemsKerasLayer

        """
        class DoubleExponentialTF(NemsKerasLayer):
            def call(self, inputs):
                exp = tf.math.exp(-tf.math.exp(
                    -tf.math.exp(self.kappa) * (inputs - self.shift)
                    ))
                return self.base + self.amplitude * exp

        return DoubleExponentialTF(self, **kwargs)
        """
        class SDexpTF(NemsKerasLayer):

            def call(self, inputs):
                # Assume inputs is a list of two tensors, with state second.
                # TODO: Use tensor names to not require this arbitrary order.
                input = inputs[0]
                state = inputs[1]

                bg = tf.expand_dims(tf.expand_dims(self.base_g, 0) ,0)
                ag = tf.expand_dims(tf.expand_dims(self.amp_g, 0) ,0)
                kg = tf.expand_dims(tf.expand_dims(self.kappa_g, 0) ,0)
                og = tf.expand_dims(tf.expand_dims(self.offset_g, 0) ,0)
                state_g = tf.reduce_sum(bg + ag * tf.math.exp(-tf.math.exp(
                    -tf.math.exp(kg) * (tf.expand_dims(state, -1) - og)
                    )), axis=2)
                bd = tf.expand_dims(tf.expand_dims(self.base_d, 0) ,0)
                ad = tf.expand_dims(tf.expand_dims(self.amp_d, 0) ,0)
                kd = tf.expand_dims(tf.expand_dims(self.kappa_d, 0) ,0)
                od = tf.expand_dims(tf.expand_dims(self.offset_d, 0) ,0)
                state_d = tf.reduce_sum(bd + ad * tf.math.exp(-tf.math.exp(
                    -tf.math.exp(kd) * (tf.expand_dims(state, -1) - od)
                    )), axis=2)

                return state_d + state_g * input

        return SDexpTF(self, **kwargs)

    def keyword(kw):
        '''
        Generate and register modulespec for the state_dexp

        Parameters
        ----------
        kw : str
            Expected format: r'^sdexp\.?(\d{1,})x(\d{1,})$'
            e.g., "sdexp.SxR" or "sdexp.S":
                S : number of state channels (required)
                R : number of channels to modulate (default = 1)
        Options
        -------
        None
        '''
        options = kw.split('.')
        pattern = re.compile(r'^(\d{1,})x(\d{1,})$')
        parsed = re.match(pattern, options[1])
        if parsed is None:
            # backward compatible parsing if R not specified
            pattern = re.compile(r'^(\d{1,})$')
            parsed = re.match(pattern, options[1])
        try:
            n_vars = int(parsed.group(1))
            if len(parsed.groups()) > 1:
                n_chans = int(parsed.group(2))
            else:
                n_chans = 1

        except TypeError:
            raise ValueError("Got TypeError when parsing sdexp2 keyword.\n"
                             "Make sure keyword is of the form: \n"
                             "sdexp2.{n_state_variables} \n"
                             "keyword given: %s" % kw)

        state = 'state'
        set_bounds = False
        # nl_state_chans = 1
        nl_state_chans = n_vars
        per_channel = ('per' in options[2:])
        for o in options[2:]:
            if o == 'lv':
                state = 'lv'
            if o == 'bound':
                set_bounds = True
            # if o == 'snl':
            # state-specific non linearities (snl)
            # only reason this is an option is to allow comparison with old models
            # nl_state_chans = n_vars

        # init gain params
        zeros = np.zeros([n_chans, nl_state_chans])
        ones = np.ones([n_chans, nl_state_chans])
        base_mean_g = zeros.copy()
        base_sd_g = ones.copy()
        amp_mean_g = zeros.copy() + 0
        amp_sd_g = ones.copy() * 0.1
        amp_mean_g[:, 0] = 1  # (1 / np.exp(-np.exp(-np.exp(0)))) # so that gain = 1 for baseline chan
        kappa_mean_g = zeros.copy()
        kappa_sd_g = ones.copy() * 0.1
        offset_mean_g = zeros.copy()
        offset_sd_g = ones.copy() * 0.1

        # init dc params
        base_mean_d = zeros.copy()
        base_sd_d = ones.copy()
        amp_mean_d = zeros.copy() + 0
        amp_sd_d = ones.copy() * 0.1
        kappa_mean_d = zeros.copy()
        kappa_sd_d = ones.copy() * 0.1
        offset_mean_d = zeros.copy()
        offset_sd_d = ones.copy() * 0.1

        template = {
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred',
                          's': state,
                          'n_inputs': n_chans,
                          'chans': n_vars,
                          'per_channel': per_channel,
                          'state_type': 'both'},
            'plot_fns': ['nems0.plots.api.mod_output',
                         'nems0.plots.api.before_and_after',
                         'nems0.plots.api.pred_resp',
                         'nems0.plots.api.state_vars_timeseries',
                         'nems0.plots.api.state_vars_psth_all'],
            'plot_fn_idx': 3,
            'prior': {'base_g': ('Normal', {'mean': base_mean_g, 'sd': base_sd_g}),
                      'amplitude_g': ('Normal', {'mean': amp_mean_g, 'sd': amp_sd_g}),
                      'kappa_g': ('Normal', {'mean': kappa_mean_g, 'sd': kappa_sd_g}),
                      'offset_g': ('Normal', {'mean': offset_mean_g, 'sd': offset_sd_g}),
                      'base_d': ('Normal', {'mean': base_mean_d, 'sd': base_sd_d}),
                      'amplitude_d': ('Normal', {'mean': amp_mean_d, 'sd': amp_sd_d}),
                      'kappa_d': ('Normal', {'mean': kappa_mean_d, 'sd': kappa_sd_d}),
                      'offset_d': ('Normal', {'mean': offset_mean_d, 'sd': offset_sd_d})}
        }
        if set_bounds:
            template['bounds'] = {'base_g': (0, 10),
                                  'amplitude_g': (0, 10),
                                  'kappa_g': (None, None),
                                  'offset_g': (None, None),
                                  'base_d': (-10, 10),
                                  'amplitude_d': (-10, 10),
                                  'kappa_d': (None, None),
                                  'offset_d': (None, None)}

        return sdexp_new(**template)

    def eval(self, rec, i, o, s, g=None, d=None, base=None, amplitude=None, kappa=None,
             base_g=None, amplitude_g=None, kappa_g=None, offset_g=None,
             base_d=None, amplitude_d=None, kappa_d=None, offset_d=None, **kw_args):
        '''
        Parameters
        ----------
        i name of input
        o name of output signal
        s name of state signal
        g - gain to scale s by
        d - dc to offset by
        base, amplitude, kappa - parameters for dexp applied to each state channel
        '''

        fn_kwargs = self.get('fn_kwargs')

        if (base_d is None) & (amplitude_d is None) & (kappa_d is None):
            fn = lambda x: _state_dexp_old(x, rec[s]._data, g, d, base, amplitude, kappa)
        else:
            fn = lambda x: _state_dexp(x, rec[s]._data, base_g, amplitude_g, kappa_g, offset_g,
                                       base_d, amplitude_d, kappa_d, offset_d, per_channel=fn_kwargs['per_channel'])

        # kludgy backwards compatibility
        try:
            p, gain, dc = fn(rec[i]._data)
            pred = rec[i]._modified_copy(p)
            pred.name = o
            gain = pred._modified_copy(gain)
            gain.name = 'gain'
            dc = pred._modified_copy(dc)
            dc.name = 'dc'
            # uncomment to save first-order pred for use by LV models.
            pred0 = rec[i]._modified_copy(p)
            pred0.name = 'pred0'
            return [pred, gain, dc, pred0]

            # uncomment to skip saving first-order pred for use by LV models.
            # return [pred, gain, dc]

        except:
            return [rec[i].transform(fn, o)]

    def tflayer(self):
        """
        layer definition for TF spec
        """
        # import tf-relevant code only here, to avoid dependency
        return []
