import logging

import numpy as np
import scipy.signal
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate

from .base import Layer, Phi, Parameter
from .tools import require_shape, pop_shape
from nems.registry import layer
from nems.distributions import Normal, HalfNormal
from nems.tools.arrays import broadcast_axes

log = logging.getLogger(__name__)


class FiniteImpulseResponse(Layer):

    # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: pooling strategy used to downsample by `stride`; kept as a class attribute (not just an __init__ default) so subclasses/callers can inspect valid options | date: 2026-08-04]
    POOL_MODES = ('mean', 'decimate')
    # [AGENT EDIT END]

    def __init__(self, stride=1, include_anticausal=False, fshape=None,
                 pool_mode='mean', **kwargs):
        """Convolve linear filter(s) with input.

        Parameters
        ----------
        shape : N-tuple
            Determines the shape of `FIR.coefficients`. Axes should be:
            (T time bins, C input channels (rank),  ..., N output channels)
            where only the first two dimensions are required. Aside from the
            time and filter axes (index 0 and -1, respectively), the size of
            each dimension must match the size of the input's dimensions.

            If only two dimensions are present, a singleton dimension will be
            appended to represent a single output. For higher-dimensional data,
            users are responsible for adding this singleton dimension if needed.
        stride : int
            If > 1, downsample the output in time by this factor (see
            `pool_mode`). The full-resolution convolution is always computed
            first; downsampling is applied as the last step.
        pool_mode : str
            How to downsample when `stride > 1`:
            'mean' (default) : average each non-overlapping block of
                `stride` samples (the final, possibly-shorter block is
                averaged over just the samples it has).
            'decimate' : keep only every `stride`-th sample, discarding the
                rest (the old default; cheaper, but throws away information).

        See also
        --------
        nems.layers.base.Layer

        Examples
        --------
        >>> fir = FiniteImpulseResponse(shape=(15,4))   # (time, input channels)
        >>> weighted_input = np.random.rand(10000, 4)   # (time, channels)
        >>> out = fir.evaluate(weighted_input)
        >>> out.shape
        (10000, 1)

        # strf alias
        >>> fir = STRF(shape=(25, 18))                   # full-rank STRF
        >>> spectrogram = np.random.rand(10000,18)
        >>> out = fir.evaluate(spectrogram)
        >>> out.shape
        (10000, 1)

        # FIR alias
        >>> fir = FIR(shape=(25, 4, 100))               # rank 4, 100 filters
        >>> spectrogram = np.random.rand(10000,4)
        >>> out = fir.evaluate(spectrogram)
        >>> out.shape
        (10000, 1, 100)

        """
        require_shape(self, kwargs, minimum_ndim=2)
        self.stride = stride
        self.include_anticausal = include_anticausal
        # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: replace crude subsampling with mean pooling as the default striding strategy for FIR/STRF, configurable for future alternatives | date: 2026-08-04]
        if pool_mode not in self.POOL_MODES:
            raise ValueError(
                f"pool_mode={pool_mode!r} not recognized; must be one of {self.POOL_MODES}"
                )
        self.pool_mode = pool_mode
        # [AGENT EDIT END]
        if not hasattr(self, 'fshape') or self.fshape is None:
            self.fshape = fshape if fshape is not None else kwargs['shape']
        #if not hasattr(self, 'wshape') or self.wshape is None:
        #    self.wshape = wshape if wshape is not None else kwargs['shape']

        super().__init__(**kwargs)


    def initial_parameters(self):
        """Get initial values for `FIR.parameters`.
        
        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `FIR.shape`.
            Prior:  Normal(mean=0, sd=1/size)
            Bounds: (-np.inf, np.inf)

        Returns
        -------
        nems.layers.base.Phi

        """
        mean = np.full(shape=self.shape, fill_value=0.0)
        #sd = np.full(shape=self.shape, fill_value=1/np.prod(self.shape))
        sd = np.full(shape=self.shape, fill_value=1/self.shape[0])
        # TODO: May be more appropriate to make this a hard requirement, but
        #       for now this should stop tiny filter sizes from causing errors.
        if mean.shape[0] > 2:
            #mean[1, :] = 2/np.prod(self.shape)
            #mean[2, :] = -1/np.prod(self.shape)
            mean[1, :] = 2 / self.shape[0]
            mean[2, :] = -1 / self.shape[0]
        prior = Normal(mean, sd)

        coefficients = Parameter(name='coefficients', shape=self.shape,
                                 prior=prior)
        return Phi(coefficients)

    @property
    def coefficients(self):
        """Filter that will be convolved with input.
        
        Re-parameterized subclasses should overwrite this so that `evaluate`
        doesn't need to change.

        Returns
        -------
        coefficients : ndarray
            coefficients.shape = WeightChannels.shape
        
        """
        return self.parameters['coefficients'].values

    def evaluate(self, input):
        """Convolve `FIR.coefficients` with input, then pool/downsample by `stride`."""
        output = self._apply_fir(input)
        # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: move striding to the last step of the processing cascade (was applied inside _apply_fir itself); lets STRF add shift/skip/activation at full resolution and pool once at the very end | date: 2026-08-04]
        if self.stride > 1:
            output = self._pool_time(output)
        # [AGENT EDIT END]
        return output

    def _apply_fir(self, input):
        """Core FIR convolution used by evaluate() and STRF.evaluate().

        Uses a sliding window view + einsum to vectorize across all filter
        outputs without a Python loop over filters.

        The causal FIR formula is:
            output[t] = sum_{lag=0}^{T-1} sum_r  input[t-lag, r] * coef[lag, r]
        which, after prepending T-1 zeros and indexing with f = T-1-lag, becomes:
            output[t] = sum_f sum_r  padded[t+f, r] * coef_time_flipped[f, r]

        Always returns full time resolution -- striding/pooling by `self.stride`
        is applied separately, as the last step of `evaluate()`.
        """
        coefficients = self.coefficients
        if coefficients.ndim == 2:
            coefficients = coefficients[..., np.newaxis]

        # Broadcast only along the output axis — no rank flip needed here.
        input, coefficients, _ = self._broadcast(input, coefficients)

        padding = self._get_filter_padding(input, coefficients)
        padded = np.pad(input, padding)

        filter_length = coefficients.shape[0]
        # Flip time axis so coef[0] corresponds to lag=0 (most recent sample).
        coef_t = np.flip(coefficients, axis=0)

        # windowed: (T_out, rank, n_outputs, filter_length) — a zero-copy view.
        windowed = sliding_window_view(padded, filter_length, axis=0)
        # Sum over rank (r) and filter time (f) in one vectorized pass.
        output = np.einsum('trof,fro->to', windowed, coef_t)

        return output

    # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: shared time-axis pooling for FIR/STRF, replacing the old crude output[::stride] subsample with a mean-pooling default (still selectable via pool_mode); also used as STRF's single, final downsampling step so per-skip-connection pooling logic is no longer needed | date: 2026-08-04]
    def _pool_time(self, x):
        """Downsample `x` along the time axis (axis 0) by `self.stride`.

        Dispatches on `self.pool_mode`:
        'mean'     : average each non-overlapping block of `self.stride`
                     samples (final, possibly-shorter block averaged over
                     just the samples it has).
        'decimate' : keep only every `self.stride`-th sample (block i -> its
                     first sample), discarding the rest.

        Both conventions keep the same reference sample (index i*stride) as
        the first element/only element of block i, so output length is
        always ceil(T/stride) and truncation (for 'mean') only ever happens
        at the end -- matching how a VALID-mode convolution/pooling op only
        truncates at the end.
        """
        stride = self.stride
        if stride <= 1:
            return x

        if self.pool_mode == 'decimate':
            return x[::stride]

        # 'mean'
        T = x.shape[0]
        n_full = T // stride
        full_part = x[:n_full * stride].reshape(
            n_full, stride, *x.shape[1:]
            ).mean(axis=1)
        remainder = T - n_full * stride
        if remainder == 0:
            return full_part
        last_part = x[n_full * stride:].mean(axis=0, keepdims=True)
        return np.concatenate([full_part, last_part], axis=0)
    # [AGENT EDIT END]

    def _reshape_coefficients(self):
        """Get `coefficients` in the format needed for `evaluate`."""
        coefficients = self.coefficients
        if coefficients.ndim == 2:
            # Add a dummy filter/output axis
            coefficients = coefficients[..., np.newaxis]

        # Flip all axes between time (0) and outputs (-1): rank and any extras.
        coefficients = np.flip(coefficients, axis=list(range(1, coefficients.ndim - 1)))

        return coefficients

    def _broadcast(self, input, coefficients):
        """Internal for `evaluate`."""
        # Add axis for n output channels to input if one doesn't exist.
        # NOTE: This will only catch a missing output dimension for 2D data.
        #       For higher-dimensional data, the output dimension needs to be
        #       specified by users.
        insert_dim = None

        if input.ndim < 3:
            if input.shape[1] == coefficients.shape[2]:
                input = input[:, np.newaxis, :]
                insert_dim = 1
            else:
                input = input[..., np.newaxis]
                insert_dim = -1

        if input.shape[-1] < coefficients.shape[-1]:
            try:
                input = broadcast_axes(input, coefficients, axis=-1)
            except ValueError:
                raise TypeError(
                    "Last dimension of FIR input must match last dimension of "
                    "coefficients, or one must be broadcastable to the other."
                    )
        elif coefficients.shape[-1] < input.shape[-1]:
            try:
                coefficients = broadcast_axes(coefficients, input, axis=-1)
            except ValueError:
                raise TypeError(
                    "Last dimension of FIR input must match last dimension of "
                    "coefficients, or one must be broadcastable to the other."
                    )
        
        return input, coefficients, insert_dim

    def _get_filter_padding(self, input, coefficients):
        """Get zeros of correct shape to prepend to input on time axis."""
        filter_length = coefficients.shape[0]

        if self.include_anticausal:
            pre_length = int(np.floor(filter_length/2)) - 1
            post_length = filter_length - pre_length - 1
            padding = [[pre_length, post_length]] + [[0, 0]]*(input.ndim-1)
        else:
            # Prepend 0s on time axis, no padding on other axes
            padding = [[filter_length-1, 0]] + [[0, 0]]*(input.ndim-1)

        return padding

    @layer('fir')
    def from_keyword(keyword):
        """Construct FIR (or subclass) from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape
            (time, input channels a.k.a. rank, ..., output channels).
        p{N}z{M}fs{F} : Use PoleZeroFIR with N poles, M zeros, and
            sample rate F (e.g. 'p2z3fs100').
        s{N} : Temporal stride of N bins.
        dec : Use 'decimate' pool_mode (subsample) instead of the default
            'mean' pooling when stride > 1.
        l2{value} : L2 regularizer, e.g. 'l2e-3'.

        See also
        --------
        Layer.from_keyword

        """
        kwargs = {}
        fir_class = FiniteImpulseResponse

        options = keyword.split('.')
        kwargs['shape'] = pop_shape(options)
        for op in options:
            if op.startswith('p') and op[1].isdigit():
                # Pole-zero parameterization
                fir_class = PoleZeroFIR
                fs_idx = op.index('fs')
                zeros_idx = op.index('z')
                kwargs['n_poles'] = int(op[1:zeros_idx])
                kwargs['n_zeros'] = int(op[zeros_idx+1:fs_idx])
                kwargs['fs'] = int(op[fs_idx+2:])
            elif op == 'dec':
                kwargs['pool_mode'] = 'decimate'
            elif op.startswith('s'):
                kwargs['stride'] = int(op[1:])
            elif op.startswith('l2'):
                kwargs['regularizer'] = op
        fir = fir_class(**kwargs)

        return fir
    
    def as_tensorflow_layer(self, input_shape, **kwargs):
        """Convert FiniteImpulseResponse to a TensorFlow Keras Layer.
        
        Parameters
        ----------
        inputs : tf.Tensor or np.ndarray
            Initial input to Layer, supplied by TensorFlowBackend during model
            building.
        
        Returns
        -------
        FiniteImpulseResponseTF
        
        """

        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer
        #from keras.utils import register_keras_serializable

        old_c = self.parameters['coefficients']
        coefficients = self.coefficients
        if coefficients.ndim == 2:
            # Add a dummy filter/output axis
            coefficients = coefficients[..., np.newaxis]
        new_c = np.flip(coefficients, axis=0)
        filter_width, rank, _ = new_c.shape
        if new_c.ndim > 3:
            raise NotImplementedError(
                "FIR TF implementation currently only works for 2D data."
                )
        new_values = {'coefficients': new_c}  # override Parameter.values

        # Define broadcasting behavior for inputs and coefficients based on
        # input_shape and new_c.shape.
        broadcast_inputs, broadcast_coefficients, n_outputs = \
            self._define_tf_broadcasting(
                tf, input_shape, new_c
                )
        # Define convolution operation, depends on whether a GPU is available.
        # Always computes at full time resolution -- striding/pooling by
        # `self.stride` is applied separately, as the last step of `call`.
        convolve = self._define_tf_convolution(
            tf, filter_width, rank, n_outputs
            )
        # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: move striding to the last step of the processing cascade, using the shared mean/decimate pooling helper instead of a strided conv | date: 2026-08-04]
        pool = self._define_tf_pooling(tf)
        # [AGENT EDIT END]

        #@register_keras_serializable(package="Custom")
        class FiniteImpulseResponseTF(NemsKerasLayer):
            def weights_to_values(self):
                c = self.parameter_values['coefficients']
                unflipped = np.flip(c, axis=0)  # Undo flip time
                unshaped = np.reshape(unflipped, old_c.shape)

                return {'coefficients': unshaped}

            def call(self, inputs):
                # This will add an extra dim if there is no output dimension.
                input_width = tf.shape(inputs)[1] # tf.shape(inputs)[1] or inputs.shape[1]
                # Broadcast output shape if needed.
                inputs = broadcast_inputs(inputs)
                coefs_tensor = tf.convert_to_tensor(self.coefficients, dtype=self.dtype)

                coefficients = broadcast_coefficients(coefs_tensor)
                #coefficients = broadcast_coefficients(self.coefficients)
                # Make None shape explicit
                rank_4 = tf.reshape(inputs, [-1, input_width, rank, n_outputs])
                return pool(convolve(rank_4, coefficients))

        return FiniteImpulseResponseTF(self, new_values=new_values, **kwargs)


    def _define_tf_broadcasting(self, tf, input_shape, new_c):
        """Internal for `as_tensorflow_layer`.
        
        Builds `broadcast_inputs` and `broadcast_coefficients` for use inside
        `call` method.

        Parameters
        ----------
        tf : package
            Reference to imported TensorFlow package.
        input_shape : tuple
            Shape of inputs.
        new_c : np.ndarray
            Reshaped coefficients.

        Returns
        -------
        broadcast_inputs : function
        broadcast_coefficients : function
        n_outputs : int
            Number of broadcasted outputs.

        """

        # Fake input to set up correct broadcasting behavior.
        # Only the number of outputs matters, this drops the batch dimension.
        # TODO: This might mess up with multiple batches similar to WC?
        #       Need to check if list?
        fake_inputs = np.empty(shape=input_shape[1:])
        new_inputs, broadcast_c, insert_dim = self._broadcast(fake_inputs, new_c)
        new_coefs_shape = list(new_c.shape[:-1]) + [broadcast_c.shape[-1]]
        new_inputs_shape = list(new_inputs.shape)
        n_outputs = new_coefs_shape[-1]

        if new_inputs_shape[-1] > input_shape[-1]:
            # If output dimension increased, then TF needs to broadcast output
            # dimension of input in call function.
            if new_inputs.ndim > fake_inputs.ndim:
                # A singleton output dimension needs to be appended as well.
                def expand_inputs(inputs):
                    return tf.expand_dims(inputs, axis=-1)
            else:
                def expand_inputs(inputs):
                    return inputs

            def broadcast_inputs(inputs):
                # Convert None batch shape to int, add singleton output dim
                # if needed. Then broadcast outputs.
                batch_size = tf.keras.backend.shape(inputs)[0]
                shape = [batch_size] + new_inputs_shape
                return tf.broadcast_to(expand_inputs(inputs), shape)

        elif new_inputs.ndim > fake_inputs.ndim:
            # print(new_inputs_shape)
            # print(input_shape)

            def broadcast_inputs(inputs):
                # Convert None batch shape to int, add singleton output dim
                # if needed. Then broadcast outputs.
                batch_size = tf.keras.backend.shape(inputs)[0]
                shape = [batch_size] + new_inputs_shape
                # insert_dim is where a dummy dimension was added to the inputs
                # so that the summing occurs (or not) across the appropriate dims
                # SVD 2025-08-13 -- special case where insert dim is -1, don't add 1
                if insert_dim>=0:
                    ii = insert_dim+1
                else:
                    ii = insert_dim
                return tf.broadcast_to(tf.expand_dims(inputs, axis=ii), shape)

        else:
            # Otherwise, don't need to do anything to inputs.
            def broadcast_inputs(inputs):
                # This will still add a singleton output dim if needed.
                #return tf.reshape(inputs, new_inputs_shape)
                return inputs

        if new_coefs_shape[-1] > new_c.shape[-1]:
            # Coefficients outputs increased, need to broadcast coefs in call.
            def broadcast_coefficients(coefficients):
                return tf.broadcast_to(coefficients, new_coefs_shape)
        else:
            # Otherwise, don't need to do anything to coefficients.
            def broadcast_coefficients(coefficients): return coefficients
        
        return broadcast_inputs, broadcast_coefficients, n_outputs

    def _define_tf_convolution(self, tf, filter_width, rank, n_outputs):
        """Internal for `as_tensorflow_layer`.
        
        Builds `convolution` function for use in `call` method.

        Parameters
        ----------
        tf : package
            Reference to imported TensorFlow package.
        filter_width, rank, n_outputs : coefficient shape components 

        Returns
        -------
        convolution : function

        """

        # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: always convolve at stride=1 (full resolution); striding/pooling now happens once, as the last step of the processing cascade, via _define_tf_pooling | date: 2026-08-04]
        stride = 1
        # [AGENT EDIT END]
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        if num_gpus == 0:
            # Use CPU-compatible (but slower) version.
            def convolve(inputs, coefficients):
                # Reorder coefficients to shape (n outputs, time, rank, 1)
                new_coefs = tf.expand_dims(
                    tf.transpose(coefficients, [2, 0, 1]), -1
                    )
                if self.include_anticausal:
                    pre_length = int(np.floor(filter_width / 2)) - 1
                    post_length = filter_width - pre_length - 1
                    padded_input = tf.pad(
                        inputs, [[0, 0], [pre_length, post_length], [0, 0], [0, 0]]
                    )
                else:
                    padded_input = tf.pad(
                        inputs, [[0, 0], [filter_width-1, 0], [0, 0], [0, 0]]
                        )
                # Reorder input to shape (n outputs, batch, time, rank)
                x = tf.transpose(padded_input, [3, 0, 1, 2])
                fn = lambda t: tf.nn.conv1d(
                    t[0], t[1], stride=stride, padding='VALID'
                    )
                # Apply convolution for each output
                y = tf.map_fn(
                    fn=fn,
                    elems=(x, new_coefs),
                    fn_output_signature=inputs.dtype
                    )
                # Reorder output back to (batch, time, n outputs)
                z = tf.transpose(tf.squeeze(y, axis=3), [1, 2, 0])
                return z
        else:
            # Use GPU-only version (grouped convolutions), much faster.
            def convolve(inputs, coefficients):
                input_width = tf.shape(inputs)[1]
                # Reshape will group by output before rank w/o transpose.
                #print("input shape:", inputs.shape.as_list(), "coef shape:", coefficients.shape.as_list())
                transposed = tf.transpose(inputs, [0, 1, 3, 2])
                # Collapse rank and n_outputs to one dimension.
                # -1 for batch size b/c it can be None.
                reshaped = tf.reshape(
                    transposed, [-1, input_width, rank*n_outputs]
                    )
                if self.include_anticausal:
                    pre_length = int(np.floor(filter_width / 2)) - 1
                    post_length = filter_width - pre_length - 1
                    padded_input = tf.pad(
                        reshaped, [[0, 0], [pre_length, post_length], [0, 0]]
                        )
                else:
                    # Prepend 0's on time axis as initial conditions for filter.
                    padded_input = tf.pad(
                        reshaped, [[0, 0], [filter_width-1, 0], [0, 0]]
                        )
                # Convolve filters with input slices in groups of size `rank`.
                y = tf.nn.conv1d(
                    padded_input, coefficients, stride=stride, padding='VALID'
                    )
                return y

        return convolve

    # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: shared TF-side time-axis pooling for FIR/STRF, mirroring _pool_time (numpy). Used as the single, final downsampling step, so per-skip-connection pooling logic in STRFTF is no longer needed | date: 2026-08-04]
    def _define_tf_pooling(self, tf):
        """Internal for `as_tensorflow_layer`.

        Builds a `pool` function that downsamples axis=1 (time) of a
        (batch, time, channels) tensor by `self.stride`, according to
        `self.pool_mode` ('mean' or 'decimate'). Returns the identity
        function if `self.stride <= 1`.

        Returns
        -------
        pool : function
        """
        stride = self.stride
        pool_mode = self.pool_mode

        if stride <= 1:
            return lambda x: x

        if pool_mode == 'decimate':
            return lambda x: x[:, ::stride]

        # 'mean'
        def pool(x):
            """Block-average over non-overlapping windows of length `stride`,
            mirroring `_pool_time` (numpy) exactly (same block boundaries,
            truncated-not-diluted final block). Uses avg_pool1d with
            right-only zero-padding plus a mask-based divisor correction,
            rather than padding='SAME', because TF's SAME padding for
            pooling ops splits padding between both sides
            (pad_before = pad_total // 2), which would insert padding
            *before* index 0 for pad_total >= 2 and shift every block's
            phase. Branch-free (no tf.cond), traces identically regardless
            of whether T % stride == 0.
            """
            T = tf.shape(x)[1]
            pad_amount = (-T) % stride
            padded = tf.pad(x, [[0, 0], [0, pad_amount], [0, 0]])
            mask = tf.ones_like(x[:, :, :1])
            padded_mask = tf.pad(mask, [[0, 0], [0, pad_amount], [0, 0]])
            # avg_pool1d(padded) = sum_valid/stride per block (zero-padded
            # entries don't change the sum, but still dilute the divisor).
            diluted_mean = tf.nn.avg_pool1d(
                padded, ksize=stride, strides=stride, padding='VALID'
                )
            # avg_pool1d(padded_mask) = valid_count/stride per block.
            valid_frac = tf.nn.avg_pool1d(
                padded_mask, ksize=stride, strides=stride, padding='VALID'
                )
            # Dividing cancels the shared /stride, leaving sum_valid/valid_count.
            return diluted_mean / valid_frac

        return pool
    # [AGENT EDIT END]

# Alias
class FIR(FiniteImpulseResponse):
    pass
    

class STRF(FiniteImpulseResponse):
    """
    agglomerated WeightChannels + FIR filter for low-rank STRF layer

    TODO: support for activation function?
    """
    def __init__(self, stride=1, include_anticausal=False, activation=None,
                 skip_alpha=0, skip_layer=None, wshape=None, fshape=None, nout=None,
                 pool_mode='mean', **kwargs):
        """Spectrotemporal receptive field: fused WeightChannels + FIR layer.

        Parameters
        ----------
        shape : 2-, 3-, or 4-tuple
            2-dim (N, T) : full-rank FIR, no channel-weighting stage.
                wshape = None, fshape = (T, N).
            3-dim (C, R, T) : low-rank STRF, 1 output channel.
                wshape = (C, R), fshape = (T, R), nout = (1, 1).
            4-dim (C, R, T, N) : low-rank STRF, N output channels.
                wshape = (C, R, N), fshape = (T, R, N), nout = (1, N).
            C = input channels, R = rank, T = time bins, N = output channels.
        activation : str or None
            Optional activation after FIR convolution: 'relu'.
        skip_alpha : float
            If != 0, a scaled copy of the input is added to the output.
            Negative: added before activation; positive: added after.
        skip_layer : bool or None
            Convenience flag — True sets skip_alpha=1, False sets it to 0.
        pool_mode : str
            How to downsample when `stride > 1`; see
            `FiniteImpulseResponse.__init__`. Shift/skip/activation are all
            computed at full time resolution; pooling is the last step.

        See also
        --------
        nems.layers.base.Layer

        Examples
        --------
        >>> strf = STRF(shape=(18, 15))         # full-rank, 1 output
        >>> strf = STRF(shape=(18, 1, 15))      # rank-1, 1 output
        >>> strf = STRF(shape=(18, 1, 15, 3))   # rank-1, 3 outputs

        """
        require_shape(self, kwargs, minimum_ndim=2)
        shape = list(kwargs['shape'])

        if len(shape) == 2:
            # Full-rank FIR: no channel-weighting stage.
            # shape = (N, T) → fshape = (T, N), wshape = None.
            self.wshape = None
            self.fshape = (shape[1], shape[0])
            self.nout = None
        elif len(shape) == 3:
            # Low-rank STRF, 1 output. shape = (C, R, T).
            self.wshape = (shape[0], shape[1])
            self.fshape = (shape[2], shape[1])
            self.nout = (1, 1)
        else:
            # Low-rank STRF, N outputs. shape = (C, R, T, N).
            self.wshape = (shape[0], shape[1], shape[3])
            self.fshape = (shape[2], shape[1], shape[3])
            self.nout = (1, shape[3])

        self.activation = activation
        if skip_layer is not None:
            skip_alpha = 1 if skip_layer else 0
        self.skip_alpha = skip_alpha

        super().__init__(stride=stride, include_anticausal=include_anticausal,
                         pool_mode=pool_mode, **kwargs)


    def initial_parameters(self):
        """Get initial values for `STRF.parameters`.

        Layer parameters
        ----------------
        2-dim (wshape is None):
            coefficients : ndarray, shape = fshape = (T, N)
                Prior/bounds match FiniteImpulseResponse convention.
            shift : ndarray, shape = (1, N)
                Prior:  Normal(0, 0.01). Same role as the 3-/4-dim `shift`:
                added directly before any activation (e.g. relu).

        3- or 4-dim:
            wcoefficients : ndarray, shape = wshape = (C, R) or (C, R, N)
                Prior:  Normal(mean=0.01, sd=0.05)
            coefficients : ndarray, shape = fshape = (T, R) or (T, R, N)
                Prior:  Normal(mean≈0, sd=1/T)
            shift : ndarray, shape = nout
                Prior:  Normal(0, 0.01)
            alpha : ndarray, shape = (1,)
                Prior:  Normal(|skip_alpha|, 0.1) or Normal(0, 1)

        Returns
        -------
        nems.layers.base.Phi

        """
        fshape = self.fshape

        if self.wshape is None:
            # 2-dim path: pure FIR — mirrors FiniteImpulseResponse.initial_parameters
            mean = np.full(shape=fshape, fill_value=0.0)
            sd   = np.full(shape=fshape, fill_value=1 / fshape[0])
            if fshape[0] > 2:
                mean[1, :] = 2 / fshape[0]
                mean[2, :] = -1 / fshape[0]
            prior = Normal(mean, sd)

            # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: 2-dim STRF previously had no shift/skip/activation support at all (evaluate() returned directly from _apply_fir); add the same fittable shift used by the 3-/4-dim path so activation='relu' etc. has a pre-activation shift to work with | date: 2026-08-04]
            nout_2d = (1, fshape[-1])
            shiftprior = Normal(np.zeros(shape=nout_2d), np.ones(shape=nout_2d) / 100)
            return Phi(
                Parameter(name='coefficients', shape=fshape, prior=prior),
                Parameter(name='shift', shape=nout_2d, prior=shiftprior),
            )
            # [AGENT EDIT END]

        # 3- or 4-dim path: WC + FIR + shift + alpha
        wshape = self.wshape
        nout   = self.nout

        # Mirrors WeightChannels.initial_parameters exactly, so that STRF's
        # wcoefficients match a standalone WeightChannels layer of shape `wshape`.
        w0 = np.zeros(np.prod(wshape))
        wn = wshape[0] + 1
        w0[::(wn + 1)] = 0.01
        wmean = np.reshape(w0, wshape)
        wmean += np.full(shape=wshape, fill_value=0.01)

        fmean = np.full(shape=fshape, fill_value=0.0)
        if fshape[0] >= 10:
            fmean[1] = 2 / fshape[0]
            fmean[2] = -1 / fshape[0]
        elif fshape[0] > 2:
            fmean[0] =  0.5 / fshape[0]
            fmean[1] =  0.5 / fshape[0]
            fmean[2] = -0.5 / fshape[0]

        wsd = np.full(shape=wshape, fill_value=0.1)
        fsd = np.full(shape=fshape, fill_value=1 / fshape[0])

        wprior     = Normal(wmean, wsd)
        fprior     = Normal(fmean, fsd)
        shiftprior = Normal(np.zeros(shape=nout), np.ones(shape=nout) / 100)

        params = [
            Parameter(name='wcoefficients', shape=wshape, prior=wprior),
            Parameter(name='coefficients',  shape=fshape, prior=fprior),
            Parameter(name='shift',         shape=nout,   prior=shiftprior),
        ]
        #if np.abs(self.skip_alpha) > 0:
        #    alphaprior = Normal(np.array([np.abs(self.skip_alpha)]), np.array([0.1]))
        #    params.append(Parameter(name='alpha', shape=(1,), prior=alphaprior))

        return Phi(*params)

    @property
    def wcoefficients(self):
        """Channel-weighting matrix (C×R or C×R×N).

        Returns None for 2-dim (full-rank FIR) STRF where there is no WC stage.
        """
        if self.wshape is None:
            return None
        return self.parameters['wcoefficients'].values

    @property
    def shift(self):
        """Per-output DC shift added after convolution."""
        return self.parameters['shift'].values

    @property
    def alpha(self):
        """Skip-connection scale factor."""
        #return self.parameters['alpha'].values
        return np.abs(self.skip_alpha)

    @property
    def strf(self):
        """Effective full-rank STRF (C×T or C×T×N) via W·F matrix product.

        Only valid for 3- or 4-dim STRF (wshape is not None).
        """
        if self.wshape is None:
            # Full-rank FIR: coefficients already are the STRF.
            return self.coefficients
        w = self.parameters['wcoefficients'].values
        f = self.parameters['coefficients'].values
        w = np.moveaxis(w, 2, 0)
        f = np.moveaxis(f, (2, 0), (0, 2))
        return np.moveaxis(w @ f, 0, 2)


    # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: striding/pooling moved to the last step of evaluate() (see below), so the skip connection now always operates at full time resolution -- no stride-aware pooling needed here anymore | date: 2026-08-04]
    def _apply_skip(self, output, input, alpha):
        """Add scaled input to output, broadcasting across the channel axis."""
        if input.shape[-1] < output.shape[-1]:
            output[:, :input.shape[-1]] += input * alpha
        else:
            output += input[:, :output.shape[-1]] * alpha
        return output

    def evaluate(self, input):
        """Apply STRF to input.

        2-dim (full-rank FIR): FIR convolution → shift → optional skip/activation
            → pool (if stride > 1).
        3- or 4-dim: channel weighting (like WeightChannels) → FIR convolution
            (like FiniteImpulseResponse) → shift → optional skip/activation
            → pool (if stride > 1).

        Pooling is always the last step: shift/skip/activation are computed
        at full time resolution, then downsampled once at the end (see
        `FiniteImpulseResponse._pool_time`). This also means the skip
        connection no longer needs its own stride-aware pooling logic --
        `input` and `output` are always the same length until the final pool.
        """
        if self.wshape is None:
            # 2-dim: pure FIR, but shift/skip/activation still apply, same as
            # the 3-/4-dim path below.
            output = self._apply_fir(input) + self.shift

            if self.skip_alpha < 0:
                output = self._apply_skip(output, input, self.alpha)

            if self.activation == 'relu':
                output[output < 0] = 0

            if self.skip_alpha > 0:
                output = self._apply_skip(output, input, self.alpha)

            if self.stride > 1:
                output = self._pool_time(output)

            return output
        # [AGENT EDIT END]

        # Weight input channels down to rank, mirroring WeightChannels.evaluate.
        weighted = np.tensordot(input, self.wcoefficients, axes=(1, 0))

        # Convolve along time, mirroring FiniteImpulseResponse.evaluate.
        if self.fshape[0] > 1:
            output = self._apply_fir(weighted) + self.shift
        else:
            output = weighted + self.shift

        if self.skip_alpha < 0:
            output = self._apply_skip(output, input, self.alpha)

        if self.activation == 'relu':
            output[output < 0] = 0

        if self.skip_alpha > 0:
            output = self._apply_skip(output, input, self.alpha)

        if self.stride > 1:
            output = self._pool_time(output)

        return output

    """ TODO: Make sure simply of to inherit from FiniteImpulseResponse?? """
    #def _reshape_coefficients(self):

    #def _broadcast(self, input, coefficients):

    #def _get_filter_padding(self, input, coefficients):

    @layer('strf')
    def from_keyword(keyword):
        """Construct STRF bank layer from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape
            (time, input channels a.k.a. rank, ..., output channels).
        lvl / dexp / relu : Activation function applied after filtering.
        sk : Skip connection with alpha=0.5.
        skl : Skip connection with alpha=-0.5.
        sk{N} : Skip connection with alpha=N/100.
        skl{N} : Skip connection with alpha=-N/100.
        s{N} : Temporal stride of N bins.
        dec : Use 'decimate' pool_mode (subsample) instead of the default
            'mean' pooling when stride > 1.
        l2{value} : L2 regularizer, e.g. 'l2e-3'.

        See also
        --------
        Layer.from_keyword

        """
        kwargs = {}

        options = keyword.split('.')
        kwargs['shape'] = pop_shape(options)
        for op in options[1:]:
            if op in ['lvl','dexp','relu']:
                # default is None
                kwargs['activation']=op
            elif op == 'skl':
                kwargs['skip_alpha'] = -0.1
            elif op == 'sk':
                kwargs['skip_alpha'] = 0.1
            elif op.startswith('skl'):
                kwargs['skip_alpha'] = -int(op[2:]) / 100
            elif op.startswith('sk'):
                kwargs['skip_alpha'] = int(op[2:]) / 100
            elif op == 'dec':
                kwargs['pool_mode'] = 'decimate'
            elif op.startswith('s'):
                kwargs['stride'] = int(op[1:])
            elif op.startswith('l2'):
                kwargs['regularizer'] = op

        strf = STRF(**kwargs)

        return strf
    
    def as_tensorflow_layer(self, input_shape, **kwargs):
        """Convert STRF to a TensorFlow Keras Layer.

        2-dim (wshape is None): FIR convolution (inherited broadcasting
        helpers, same reshape FiniteImpulseResponse.as_tensorflow_layer
        uses) + shift + optional skip connection / activation.

        3- or 4-dim: fused WC (einsum) + FIR (inherited convolution helpers)
        + shift + optional skip connection / activation.

        Parameters
        ----------
        input_shape : tuple
            Shape of the layer input, supplied by TensorFlowBackend.

        Returns
        -------
        NemsKerasLayer subclass
        """
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: 2-dim STRF previously delegated entirely to FiniteImpulseResponse.as_tensorflow_layer, which knows nothing about STRF's shift/skip_alpha/activation attributes -- unify with the 3-/4-dim path below (branching only on wshape_is_none) so those apply uniformly, matching the numpy-side fix | date: 2026-08-04]
        wshape_is_none = self.wshape is None

        old_c_shape = self.parameters['coefficients'].shape   # fshape tuple
        coefficients = self.coefficients
        if coefficients.ndim == 2:
            coefficients = coefficients[..., np.newaxis]
        new_c = np.flip(coefficients, axis=0)
        filter_width, rank, _ = new_c.shape
        new_values = {'coefficients': new_c}

        # Set up FIR broadcasting/convolution helpers (inherited from FIR).
        # broadcast_inputs is only used in the 2-dim (wshape_is_none) path --
        # the WC einsum in the 3-/4-dim path already produces the correct
        # intermediate shape without it.
        broadcast_inputs, broadcast_coefficients, n_outputs = self._define_tf_broadcasting(
            tf, input_shape, new_c
        )
        convolve = self._define_tf_convolution(tf, filter_width, rank, n_outputs)
        pool = self._define_tf_pooling(tf)

        activation  = self.activation
        skip_alpha  = self.skip_alpha
        skip_scale  = self.alpha  # abs(skip_alpha) -- sign only selects pre/post-activation timing below, it should not flip the sign of the added term (mirrors numpy's `self.alpha` property, used the same way in `_apply_skip`)
        fir_len     = self.fshape[0]
        wcoef_ndim  = len(self.wshape) if not wshape_is_none else None   # 2 → (C, R),  3 → (C, R, N)

        class STRFTF(NemsKerasLayer):
            def weights_to_values(self):
                c        = self.parameter_values['coefficients']
                unflipped = np.flip(c, axis=0)          # undo time-flip
                unshaped  = np.reshape(unflipped, old_c_shape)
                vals = {
                    'coefficients':  unshaped,
                    'shift':         self.parameter_values['shift'],
                }
                if not wshape_is_none:
                    vals['wcoefficients'] = self.parameter_values['wcoefficients']
                if 'alpha' in self.parameter_values:
                    vals['alpha'] = self.parameter_values['alpha']
                return vals
                # [AGENT EDIT END]

            def call(self, inputs):
                # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: pooling moved to the last step of call() (see below), so skip now always operates at full time resolution -- no stride-aware pooling needed here anymore | date: 2026-08-04]
                def apply_skip(out):
                    if inputs.shape[-1] < out.shape[-1]:
                        #head = out[:, :, :inputs.shape[-1]] + inputs * self.alpha
                        head = out[:, :, :inputs.shape[-1]] + inputs * skip_scale
                        tail = out[:, :, inputs.shape[-1]:]
                        return tf.concat([head, tail], axis=2)
                    else:
                        #return out + inputs[:, :, :out.shape[-1]] * self.alpha
                        return out + inputs[:, :, :out.shape[-1]] * skip_scale
                # [AGENT EDIT END]

                # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: build rank_4/out for the 2-dim (wshape_is_none) case, mirroring FiniteImpulseResponse.as_tensorflow_layer's own broadcast+reshape exactly, so shift/skip/activation below apply the same way they do for the 3-/4-dim case | date: 2026-08-04]
                if wshape_is_none:
                    # No WC stage -- feed inputs directly into the FIR conv,
                    # same reshape FiniteImpulseResponse.as_tensorflow_layer uses.
                    input_width = tf.shape(inputs)[1]
                    broadcast_in = broadcast_inputs(inputs)
                    rank_4 = tf.reshape(broadcast_in, [-1, input_width, rank, n_outputs])
                    coefs_tensor = tf.convert_to_tensor(self.coefficients, dtype=self.dtype)
                    out = convolve(rank_4, broadcast_coefficients(coefs_tensor))
                else:
                    # Channel weighting — mirrors WeightChannels.as_tensorflow_layer.
                    # Use einsum (not tensordot): Keras 3 traces einsum statically.
                    if wcoef_ndim == 3:
                        # wcoefficients: (C, R, N) → (batch, time, R, N)
                        rank_4 = tf.einsum('bti,irn->btrn', inputs, self.wcoefficients)
                    else:
                        # wcoefficients: (C, R) → (batch, time, R, 1)
                        rank_4 = tf.expand_dims(
                            tf.einsum('bti,ir->btr', inputs, self.wcoefficients),
                            axis=-1,
                        )

                    # FIR convolution — mirrors FiniteImpulseResponse.as_tensorflow_layer.
                    if fir_len > 1:
                        coefs_tensor = tf.convert_to_tensor(
                            self.coefficients, dtype=self.dtype
                        )
                        out = convolve(rank_4, broadcast_coefficients(coefs_tensor))
                    else:
                        out = rank_4
                # [AGENT EDIT END]
                out = out + self.shift

                if skip_alpha < 0:
                    out = apply_skip(out)
                if activation == 'relu':
                    out = tf.nn.relu(out)
                if skip_alpha > 0:
                    out = apply_skip(out)

                return pool(out)

        return STRFTF(self, new_values=new_values, **kwargs)



class PoleZeroFIR(FiniteImpulseResponse):

    def __init__(self, n_poles, n_zeros, fs, **kwargs):
        """TODO: docs.
        
        TODO: Possible to remove need for sampling rate?
        
        """
        self.n_poles = n_poles
        self.n_zeros = n_zeros
        self.fs = fs
        require_shape(self, kwargs, minimum_ndim=2, maximum_ndim=3)
        super().__init__(**kwargs)

    def initial_parameters(self):
        """TODO: docs."""

        # TODO: explain choice of priors
        rank = self.shape[1]
        if len(self.shape) == 3:
            n_filters = self.shape[2]
        else:
            n_filters = 1
        pole_set = np.array([[[0.8, -0.4, 0.1, 0.0, 0]]])[..., :self.n_poles]
        zero_set = np.array([[[0.1,  0.1, 0.1, 0.1, 0]]])[..., :self.n_zeros]

        poles_prior = Normal(
            mean = pole_set.repeat(rank, 0).repeat(n_filters, 1),
            sd = np.ones((rank, n_filters, self.n_poles))*0.3,
            )
        zeros_prior = Normal(
            mean = zero_set.repeat(rank, 0).repeat(n_filters, 1),
            sd = np.ones((rank, n_filters, self.n_zeros))*0.2,
            )
        delays_prior = HalfNormal(sd = np.ones((rank, n_filters))*0.02)
        gains_prior = Normal(
            mean = np.zeros((rank, n_filters))+0.1,
            sd = np.ones((rank, n_filters))*0.2
            )

        poles = Parameter('poles', shape=(rank, n_filters, self.n_poles),
                          prior=poles_prior, bounds=(-1, 1))
        zeros = Parameter('zeros', shape=(rank, n_filters, self.n_zeros),
                          prior=zeros_prior, bounds=(-1, 1))
        # TODO: what do the delays do exactly?
        delays = Parameter('delays', shape=(rank, n_filters),
                           prior=delays_prior, bounds=(0, np.inf))
        gains = Parameter('gains', shape=(rank, n_filters),
                          prior=gains_prior)

        return Phi(poles, zeros, delays, gains)

    @property
    def coefficients(self):
        """TODO: docs."""
        poles, zeros, delays, gains = self.get_parameter_values()

        n_taps, rank = self.shape[:2]
        if len(self.shape) == 2:
            n_filters = 1
        else:
            n_filters = self.shape[-1]

        coefficients = np.zeros((n_taps, rank, n_filters))

        # TODO: can we do this without fs?
        # TODO: explain why 5*original
        fs2 = 5*self.fs                      

        for i in range(rank):
            for j in range(n_filters):
                # TODO: rename variables, improve documentation.
                #       still don't really know what this is doing.
                t = np.arange(0, n_taps*5 + 1) / fs2
                h = scipy.signal.ZerosPolesGain(
                    zeros[i,j], poles[i,j], gains[i,j], dt=1/fs2
                    )
                tout, ir = scipy.signal.dimpulse(h, t=t)
                f = interpolate.interp1d(tout, ir[0][:,0], bounds_error=False,
                                         fill_value=0)

                tnew = np.arange(0, n_taps)/self.fs - delays[i,j] + 1/self.fs
                coefficients[:, i, j] = f(tnew)

        return coefficients

    # TODO: as_tensorflow_layer
