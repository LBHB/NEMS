import numpy as np
import pytest

from nems.models.LN import LN_pop


class TestLNPopStride:
    """LN_pop used to append a separate frozen-boxcar `agg` FIR layer after
    the nonlinearity to implement `stride`. That's now handled natively by
    FiniteImpulseResponse's own stride/pool_mode, applied *before* the
    nonlinearity instead of after (accepted behavior change)."""

    @pytest.mark.parametrize("rank,share_tuning", [(None, True), (2, True), (2, False)])
    def test_no_agg_layer_and_correct_pooled_shape(self, rank, share_tuning):
        time_bins, channels_in, channels_out, stride = 8, 3, 2, 3
        model = LN_pop(
            time_bins=time_bins, channels_in=channels_in, channels_out=channels_out,
            stride=stride, rank=rank, share_tuning=share_tuning,
            )

        # Nonlinearity should always be the last layer -- no trailing agg.
        assert model.layers[-1].__class__.__name__ == 'DoubleExponential'
        fir_layers = [l for l in model.layers if type(l).__name__ == 'FiniteImpulseResponse']
        assert len(fir_layers) == 1  # exactly one FIR layer, not two (agg removed)
        assert fir_layers[0].stride == stride

        time = 60
        x = np.random.randn(time, channels_in)
        out = model.evaluate(x)
        pred = out['output'] if isinstance(out, dict) else out
        assert pred.shape[0] == int(np.ceil(time / stride))
        assert pred.shape[-1] == channels_out

    def test_nl_layer_index_for_fit_lbhb(self):
        """`fit_LBHB` assumes the nonlinearity is always `layers[-1]`."""
        model = LN_pop(time_bins=8, channels_in=3, channels_out=2, stride=3, rank=None)
        layer = model.layers[-1]
        assert hasattr(layer, 'skip_nonlinearity')
        layer.skip_nonlinearity()
        layer.unskip_nonlinearity()