import pytest
import numpy as np

from nems.layers.weight_channels import WeightChannels
from nems.layers.filter import FiniteImpulseResponse, STRF, PoleZeroFIR
from nems.tools.debug import generate_random_input, compare_layer_eval

def test_constructor():
    with pytest.raises(TypeError):
        # No shape
        fir = FiniteImpulseResponse()
    with pytest.raises(TypeError):
        # Shape has too few dimensions
        fir = FiniteImpulseResponse(shape=(2,))
    # But this should raise no errors.
    fir = FiniteImpulseResponse(shape=(1,2,3))
    

class TestEvaluate:

    def test_full_rank(self):
        spectrogram = generate_random_input((100,4))
        time, spectral = spectrogram.shape
        # 50 ms filter, 18 spectral channels (full-rank STRF)
        fir = FiniteImpulseResponse(shape=(5, spectral))
        out = fir.evaluate(spectrogram)
        assert out.shape == (time, 1)

    def test_full_with_outputs(self):
        # Same but with 3 outputs (i.e. filter-bank)
        spectrogram = generate_random_input((100,4))
        time, spectral = spectrogram.shape
        fir_bank = FiniteImpulseResponse(shape=(5, spectral, 3))
        bank_out = fir_bank.evaluate(spectrogram)
        # Rank dimension should be squeezed out.
        assert bank_out.shape == (time, 3)

    def test_rank_with_outputs(self):
        # Pretend spectrogram has output dimension, rank 1
        spectrogram = generate_random_input((100,4))
        time, spectral = spectrogram.shape
        spectrogram_with_outputs = spectrogram.reshape(time, 1, spectral)
        fir_with_outputs = FiniteImpulseResponse(shape=(5, 1, spectral))
        outputs_out = fir_with_outputs.evaluate(spectrogram_with_outputs)
        # Rank dimension should be squeezed out.
        assert outputs_out.shape == (time, spectral)

    def test_reapply_same_filter(self):
        # Pretend spectrogram has output dimension, rank 1
        spectrogram = generate_random_input((100,4))
        time, spectral = spectrogram.shape
        spectrogram_with_outputs = spectrogram.reshape(time, 1, spectral)
        fir_no_outputs = FiniteImpulseResponse(shape=(5, 1))  # 1 implied output
        no_outputs_out = fir_no_outputs.evaluate(spectrogram_with_outputs)
        # Rank dimension should be squeezed out, but outputs of spectrogram
        # should be preserved
        assert no_outputs_out.shape == (time, spectral)

    def test_pole_zero(self):
        spectrogram = generate_random_input((100,4))
        time, spectral = spectrogram.shape
        fir_no_outputs = PoleZeroFIR(
            shape=(15, spectral), n_poles=3, n_zeros=1, fs=100
            )
        out = fir_no_outputs.evaluate(spectrogram)
        assert out.shape == (time, 1)

        # pretend spectrogram has an output dimension, rank 1
        spectrogram_with_outputs = spectrogram.reshape(time, 1, spectral)
        fir_with_outputs = PoleZeroFIR(
            shape=(15, 1, spectral), n_poles=2, n_zeros=2, fs=10
        )
        out2 = fir_with_outputs.evaluate(spectrogram_with_outputs)
        assert out2.shape == (time, spectral)

    def test_strf(self):
        spectrogram = generate_random_input((100,6))
        time, spectral = spectrogram.shape

        shape = (spectral, 2, 10, 3) # (C, R, T, N)
        wshape = (shape[0], shape[1], shape[3])
        fshape = (shape[2], shape[1], shape[3])

        strf = STRF(shape=shape)
        out1 = strf.evaluate(spectrogram)

        wc = WeightChannels(shape=wshape)
        fir = FiniteImpulseResponse(shape=fshape)
        out2 = fir.evaluate(wc.evaluate(spectrogram))

        assert out1.shape==out2.shape
        assert np.all(out1==out2)

    def test_strf_kw(self):
        spectrogram = generate_random_input((100,6))
        time, spectral = spectrogram.shape

        shape = (spectral, 2, 10, 3) # (C, R, T, N)
        shape = [str(s) for s in shape]
        wshape = "x".join((shape[0], shape[1], shape[3]))
        fshape = "x".join((shape[2], shape[1], shape[3]))
        shape = "x".join(shape)

        strf = STRF.from_keyword(f"strf.{shape}")
        out1 = strf.evaluate(spectrogram)

        wc = WeightChannels.from_keyword(f"wc.{wshape}")
        fir = FiniteImpulseResponse.from_keyword(f"fir.{fshape}")
        out2 = fir.evaluate(wc.evaluate(spectrogram))

        assert out1.shape==out2.shape
        assert np.all(out1==out2)


    def test_backends(self):
        spectrogram = generate_random_input((100,6))
        time, spectral = spectrogram.shape

        shape = (spectral, 2, 10, 3) # (C, R, T, N)
        strf = STRF(shape=shape)
        strf.set_dtype('float32')

        numpy_out, tf_out = compare_layer_eval(strf, spectrogram)

        assert np.mean((numpy_out.flatten() - tf_out.flatten()) ** 2) < 1e-4

    # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: cover STRF stride+skip with both pool_mode options ('mean' windowed-average, 'decimate' subsample), both T divisible and not divisible by stride, and numpy-vs-TF equivalence | date: 2026-08-04]
    @pytest.mark.parametrize("pool_mode", ['mean', 'decimate'])
    @pytest.mark.parametrize("stride", [1, 2, 3, 5, 7])
    @pytest.mark.parametrize("skip_alpha", [0.3, 0.7, -0.3, -0.7])
    @pytest.mark.parametrize("time", [97, 100])
    def test_strf_stride_skip(self, pool_mode, stride, skip_alpha, time):
        spectral = 4
        spectrogram = generate_random_input((time, spectral))

        shape = (spectral, 1, 5, 2)  # (C, R, T, N)
        strf = STRF(shape=shape, stride=stride, skip_alpha=skip_alpha, pool_mode=pool_mode)
        strf.set_dtype('float32')

        out = strf.evaluate(spectrogram)
        expected_time = int(np.ceil(time / stride))
        assert out.shape == (expected_time, shape[-1])

        numpy_out, tf_out = compare_layer_eval(strf, spectrogram)
        assert numpy_out.shape == tf_out.shape[1:]
        assert np.mean((numpy_out.flatten() - tf_out.flatten()) ** 2) < 1e-4

    def test_pool_time_mean(self):
        """`FiniteImpulseResponse._pool_time` (pool_mode='mean', the default)
        should match a naive per-block-mean loop, for T both divisible and
        not divisible by stride. Tested via STRF since it inherits the
        method unchanged from FiniteImpulseResponse."""
        strf = STRF(shape=(4, 1, 5, 2), stride=3)
        assert strf.pool_mode == 'mean'
        for time in [99, 100]:
            x = generate_random_input((time, 4))
            pooled = strf._pool_time(x)
            n_blocks = int(np.ceil(time / 3))
            reference = np.stack([
                x[i * 3: (i + 1) * 3].mean(axis=0) for i in range(n_blocks)
                ])
            assert np.allclose(pooled, reference)

    def test_pool_time_decimate(self):
        """`pool_mode='decimate'` should reproduce plain subsampling."""
        strf = STRF(shape=(4, 1, 5, 2), stride=3, pool_mode='decimate')
        x = generate_random_input((100, 4))
        pooled = strf._pool_time(x)
        assert np.allclose(pooled, x[::3])

    def test_pool_mode_validation(self):
        with pytest.raises(ValueError):
            FiniteImpulseResponse(shape=(5, 4), pool_mode='bogus')
        with pytest.raises(ValueError):
            STRF(shape=(4, 1, 5, 2), pool_mode='bogus')

    def test_fir_stride_pool_mode(self):
        """Plain FiniteImpulseResponse (not just STRF) should also support
        pool_mode, since striding was moved to the end of evaluate() there
        too (previously baked into _apply_fir)."""
        spectral = 3
        for time in [97, 100]:
            spectrogram = generate_random_input((time, spectral))
            for pool_mode in ['mean', 'decimate']:
                for stride in [1, 2, 3, 5]:
                    fir = FiniteImpulseResponse(
                        shape=(5, spectral), stride=stride, pool_mode=pool_mode
                        )
                    fir.set_dtype('float32')
                    out = fir.evaluate(spectrogram)
                    expected_time = int(np.ceil(time / stride))
                    assert out.shape == (expected_time, 1)

                    numpy_out, tf_out = compare_layer_eval(fir, spectrogram)
                    assert numpy_out.shape == tf_out.shape[1:]
                    assert np.mean((numpy_out.flatten() - tf_out.flatten()) ** 2) < 1e-4
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: cover FIR's newly-added skip connection support (previously STRF-only), same keywords/params/logic, added before pooling | date: 2026-08-04]
    @pytest.mark.parametrize("stride", [1, 2, 3, 5])
    @pytest.mark.parametrize("skip_alpha", [0.0, 0.3, 0.7, -0.3, -0.7])
    @pytest.mark.parametrize("time", [97, 100])
    def test_fir_skip(self, stride, skip_alpha, time):
        spectral = 3
        spectrogram = generate_random_input((time, spectral))

        fir = FiniteImpulseResponse(shape=(5, spectral), stride=stride, skip_alpha=skip_alpha)
        fir.set_dtype('float32')

        out = fir.evaluate(spectrogram)
        expected_time = int(np.ceil(time / stride))
        assert out.shape == (expected_time, 1)

        numpy_out, tf_out = compare_layer_eval(fir, spectrogram)
        assert numpy_out.shape == tf_out.shape[1:]
        assert np.mean((numpy_out.flatten() - tf_out.flatten()) ** 2) < 1e-4

    def test_fir_skip_changes_output(self):
        """A nonzero skip_alpha should produce different output than skip_alpha=0,
        and positive/negative skip_alpha should be equivalent (FIR has no
        activation yet, so the "before/after activation" sign distinction is
        currently a no-op)."""
        spectrogram = generate_random_input((30, 4))
        coefficients = np.random.randn(5, 4) * 0.01

        fir_noskip = FiniteImpulseResponse(shape=(5, 4))
        fir_noskip['coefficients'] = coefficients
        out_noskip = fir_noskip.evaluate(spectrogram)

        fir_pos = FiniteImpulseResponse(shape=(5, 4), skip_alpha=0.5)
        fir_pos['coefficients'] = coefficients
        out_pos = fir_pos.evaluate(spectrogram)
        assert not np.allclose(out_pos, out_noskip)

        fir_neg = FiniteImpulseResponse(shape=(5, 4), skip_alpha=-0.5)
        fir_neg['coefficients'] = coefficients
        out_neg = fir_neg.evaluate(spectrogram)
        assert np.allclose(out_pos, out_neg)

    def test_fir_skip_with_upstream_rank_axis(self):
        """Regression test: when a FIR-with-skip layer's raw input comes
        directly from an upstream layer that retains an explicit (e.g.
        singleton) rank axis -- like WeightChannels with shape (C, R, N) --
        `_apply_skip`/`_define_tf_skip` used to crash with a broadcast
        error, since `output` is always 2D/(batch,T,N) (the FIR conv
        collapses rank away) but raw `input` stayed 3D/(batch,T,R,N)."""
        from nems import Model

        N = 6
        model = Model.from_keywords(f'wc.{N}x1x8-fir.5x1x8.s2.sk.l2:4')
        model.set_dtype('float32')
        model = model.sample_from_priors()

        wc_out = model.layers[0].evaluate(generate_random_input((1, N)))
        assert wc_out.ndim == 3  # confirms the retained singleton rank axis

        spectrogram = generate_random_input((50, N))
        out = model.evaluate(spectrogram)
        assert out['output'].shape == (25, 8)

        numpy_out, tf_out = compare_layer_eval(model.layers[1], model.layers[0].evaluate(spectrogram))
        assert numpy_out.shape == tf_out.shape[1:]
        assert np.mean((numpy_out.flatten() - tf_out.flatten()) ** 2) < 1e-4

    def test_fir_skip_multirank_input_sums_across_rank(self):
        """A raw skip input with rank > 1 should be summed across the rank
        axis before adding, mirroring how the FIR convolution itself
        collapses rank via a weighted sum."""
        fir = FiniteImpulseResponse(shape=(3, 2, 4), skip_alpha=0.5)
        fir.parameters.sample(inplace=True)
        multirank_input = generate_random_input((10, 2, 4))  # rank=2
        out = fir.evaluate(multirank_input)

        conv_only = fir._apply_fir(multirank_input)
        expected_skip_term = multirank_input.sum(axis=1) * fir.alpha
        assert np.allclose(out, conv_only + expected_skip_term)

    def test_fir_skip_multirank_upstream_wc(self):
        """End-to-end: wc.Nx2x30-fir.10x2x30.sk (rank=2), numpy-vs-TF."""
        from nems import Model

        N = 6
        model = Model.from_keywords(f'wc.{N}x2x8-fir.5x2x8.sk.l2:4')
        model.set_dtype('float32')
        model = model.sample_from_priors()

        wc_out = model.layers[0].evaluate(generate_random_input((1, N)))
        assert wc_out.shape[1] == 2  # confirms the genuine rank=2 axis

        spectrogram = generate_random_input((50, N))
        out = model.evaluate(spectrogram)
        assert out['output'].shape == (50, 8)

        numpy_out, tf_out = compare_layer_eval(model.layers[1], model.layers[0].evaluate(spectrogram))
        assert numpy_out.shape == tf_out.shape[1:]
        assert np.mean((numpy_out.flatten() - tf_out.flatten()) ** 2) < 1e-4

    def test_fir_skip_keyword(self):
        fir = FiniteImpulseResponse.from_keyword('fir.5x4.sk')
        assert fir.skip_alpha == 0.1
        fir = FiniteImpulseResponse.from_keyword('fir.5x4.skl')
        assert fir.skip_alpha == -0.1
        fir = FiniteImpulseResponse.from_keyword('fir.5x4.sk25')
        assert fir.skip_alpha == 0.25
        fir = FiniteImpulseResponse.from_keyword('fir.5x4.skl25')
        assert fir.skip_alpha == -0.25
        fir = FiniteImpulseResponse.from_keyword('fir.5x4.s3.dec.sk')
        assert fir.stride == 3
        assert fir.pool_mode == 'decimate'
        assert fir.skip_alpha == 0.1

    def test_strf_skip_keyword_skl_n(self):
        """Regression test: STRF.from_keyword's 'skl{N}' parsing used to
        slice at the wrong index ('skl' is 3 chars, not 2), so e.g. 'skl25'
        raised ValueError instead of setting skip_alpha=-0.25."""
        strf = STRF.from_keyword('strf.4x1x5x2.skl25')
        assert strf.skip_alpha == -0.25
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: cover 2-dim STRF's newly-added shift/skip/activation support (previously silently ignored for the wshape=None / pure-FIR case) | date: 2026-08-04]
    @pytest.mark.parametrize("activation", [None, 'relu'])
    @pytest.mark.parametrize("skip_alpha", [0.0, 0.5, -0.5])
    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_strf_2d_shift_skip_activation(self, activation, skip_alpha, stride):
        spectral = 4
        spectrogram = generate_random_input((97, spectral))

        strf = STRF(
            shape=(spectral, 8), activation=activation,
            skip_alpha=skip_alpha, stride=stride,
            )
        strf.set_dtype('float32')

        # shift should be a fittable parameter, not fixed at 0.
        assert 'shift' in strf.parameters._dict
        strf.parameters.sample(inplace=True)
        assert not np.allclose(strf.parameters['shift'].values, 0)

        out = strf.evaluate(spectrogram)
        expected_time = int(np.ceil(97 / stride))
        assert out.shape == (expected_time, spectral)

        numpy_out, tf_out = compare_layer_eval(strf, spectrogram)
        assert numpy_out.shape == tf_out.shape[1:]
        assert np.mean((numpy_out.flatten() - tf_out.flatten()) ** 2) < 1e-4
    # [AGENT EDIT END]
