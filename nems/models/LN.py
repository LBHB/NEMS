from .base import Model
from nems.layers import (
    WeightChannels, WeightChannelsGaussian, FiniteImpulseResponse,
    RectifiedLinear, DoubleExponential
    )

class LN_STRF(Model):

    def __init__(self, channels, time_bins, rank=None,
                 gaussian=True, nonlinearity='DoubleExponential',
                 nl_kwargs=None, **model_init_kwargs):
        """Linear-nonlinear Spectro-Temporal Receptive Field model.

        Contains the following layers:
        1. WeightChannels, WeightChannelsGaussian, or None (if full rank).
        2. FiniteImpulseResponse
        3. DoubleExponential, RectifiedLinear, or another static nonlinearity.

        Expects a single sound spectrogram as input, with shape (T, N), and a
        single recorded neural response as a target, with shape (T, 1), where
        T and N are the number of time bins and spectral channels, respectively.

        Based on model architectures described in:
        Thorson, Lienard and David (2015)
        doi: 10.1371/journal.pcbi.1004628

        Parameters
        ----------
        channels : int.
            Number of spectral channels in spectrogram.
        time_bins : int.
            Number of "taps" in FIR filter. We have found that a 150-250ms filter
            typically sufficient for modeling A1 responses, or 15-25 bins for
            a 100 Hz sampling rate.
        rank : int; optional.
            Number of spectral weightings used as input to a reduced-rank filter.
            For example, `rank=1` indicates a frequency-time separable STRF.
            If unspecified, a full-rank filter will be used.
        gaussian : bool; default=True.
            If True, use gaussian functions (1 per `rank`) to parameterize
            spectral weightings. Unused if `rank is None`.
        nonlinearity : str; default='DoubleExponential'.
            Specifies which static nonlinearity to apply after the STRF.
            Default is the double exponential nonlinearity used in the paper
            cited above.
        nl_kwargs : dict; optional.
            Additional keyword arguments for the nonlinearity Layer, like
            `no_shift` or `no_offset` for `RectifiedLinear`.
        model_init_kwargs : dict; optional.
            Additional keyword arguments for `Model.__init__`, like `dtype`
            or `meta`.

        """

        super().__init__(**model_init_kwargs)

        # Add STRF
        if rank is None:
            # Full-rank finite impulse response
            fir = FiniteImpulseResponse(shape=(time_bins, channels))
            self.add_layers(fir)
        else:
            wc_class = WeightChannelsGaussian if gaussian else WeightChannels
            wc = wc_class(shape=(channels, rank))
            fir = FiniteImpulseResponse(shape=(time_bins, rank))
            self.add_layers(wc, fir)

        # Add static nonlinearity
        if nonlinearity in ['DoubleExponential', 'dexp', 'DEXP']:
            nl_class = DoubleExponential
        elif nonlinearity in ['RectifiedLinear', 'relu', 'ReLU']:
            nl_class = RectifiedLinear
        else:
            raise ValueError(
                f'Unrecognized nonlinearity for LN model:  {nonlinearity}.')
        
        if nl_kwargs is None: nl_kwargs = {}
        nonlinearity = nl_class(shape=(1,), **nl_kwargs)
        self.add_layers(nonlinearity)


    @classmethod
    def from_data(cls, input, filter_duration, sampling_rate=1000, **kwargs):
        channels = input.shape[-1]
        time_bins = int(filter_duration/1000 * sampling_rate)
        # TODO: modify initial parameters based on stimulus statistics?
        return LN_STRF(channels, time_bins, **kwargs)

    def fit_LBHB():
        # TODO: 3-stage fit with freezing/unfreezing NL
        pass

    # TODO
    # @module('LNSTRF')
    def from_keyword(keyword):
        # Return a list of module instances matching this pre-built Model?
        # That way these models can be used with kw system as well, e.g.
        # model = Model.from_keywords('LNSTRF')
        #
        # But would need the .from_keywords method to check for list vs single
        # module returned.
        pass
