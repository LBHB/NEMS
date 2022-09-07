from .base import Model

class LN_STRF(Model):
    '''
    A ModelSpec with the following modules:
        1) WeightChannels(shape=(4, n_channels), parameterization='gaussian')
        2) FIR(shape=(4, 15), parameterization='P3Z1')
        3) LevelShift(),
        4) DoubleExponential()

    Based on the best-performing model from
    Thorson, Lienard and David (2015)
    doi: 10.1371/journal.pcbi.1004628

    '''

    def __init__(n_channels):
        # Need to know number of spectral channels in the stimulus
        self.n_channels = n_channels

    # TODO: everything else, this is just to illustrate the idea.

    # @module('LNSTRF')
    def from_keyword(keyword):
        # Return a list of module instances matching this pre-built Model?
        # That way these models can be used with kw system as well, e.g.
        # model = Model.from_keywords('LNSTRF')
        #
        # But would need the .from_keywords method to check for list vs single
        # module returned.
        pass
