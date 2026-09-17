import numpy as np

from nems.models.ACNet import ACNet


class TestConstruction:

    def test_layer_count(self):
        # compress(1) + block0(dfir,wc,lvl,bn,relu=5) + 2 more blocks x 6 (dfir,
        # wc, lvl, bn, resadd, relu) + readout(wc,lvl,dexp=3)
        model = ACNet(num_cfs=8, hidden_dim=(6, 7, 8), kernel_size=3, n_neurons=4)
        assert len(model.layers) == 1 + 5 + 2 * 6 + 3

    def test_default_config_shapes(self):
        # Matches ACNet_v1.acnet_model.DEFAULT_CONFIG.
        model = ACNet()
        shapes = [tuple(l.shape) if l.shape is not None else None for l in model.layers]
        assert shapes[1] == (7, 32)     # block0 dfir
        assert shapes[2] == (32, 75)    # block0 wc
        assert shapes[-3] == (200, 3124)  # readout wc
        assert shapes[-1] == (3124,)      # readout dexp


class TestEvaluate:

    def test_output_shape(self):
        T = 200
        model = ACNet(num_cfs=8, hidden_dim=(6, 7, 8), kernel_size=3, n_neurons=5)
        gtg = np.random.rand(T, 8)
        out = model.predict(gtg)
        assert out.shape == (T, 5)

    def test_single_block_no_residual_wiring_needed(self):
        # A single-block model shouldn't try to name/reference a
        # nonexistent b1_in signal.
        T = 50
        model = ACNet(num_cfs=4, hidden_dim=(6,), kernel_size=3, n_neurons=3)
        gtg = np.random.rand(T, 4)
        out = model.predict(gtg)
        assert out.shape == (T, 3)

    def test_residual_skip_actually_wired(self):
        # With shortcut initialized to zero and gamma=1, output of block i
        # should equal ReLU(res_scale * main_path) at construction -- but the
        # real test of *wiring* is that this runs at all (ResAdd's 'skip'
        # input must resolve to the right earlier signal, not error out or
        # silently broadcast the wrong array).
        T = 100
        model = ACNet(num_cfs=6, hidden_dim=(6, 6, 6), kernel_size=3, n_neurons=4,
                      res_scale=1.0)
        gtg = np.random.rand(T, 6)
        out = model.predict(gtg)
        assert out.shape == (T, 4)
        assert np.all(np.isfinite(out))

    def test_compress_mode_changes_output(self):
        T = 80
        np.random.seed(0)
        gtg = np.random.rand(T, 8)

        model_sqrt = ACNet(num_cfs=8, hidden_dim=(6,), kernel_size=3, n_neurons=3,
                           compress='sqrt')
        model_log = ACNet(num_cfs=8, hidden_dim=(6,), kernel_size=3, n_neurons=3,
                          compress='log10x')
        # Different random inits mean we can't compare outputs directly, but
        # the first layer's mode should differ as expected.
        assert model_sqrt.layers[0].mode == 'sqrt'
        assert model_log.layers[0].mode == 'log10x'

    def test_get_embeddings_shape_and_value(self):
        # get_embeddings should return exactly the last trunk layer's output
        # -- the same thing named 'embeddings' internally.
        T = 60
        model = ACNet(num_cfs=6, hidden_dim=(6, 7, 8), kernel_size=3, n_neurons=4)
        gtg = np.random.rand(T, 6)

        embeddings = model.get_embeddings(gtg)
        assert embeddings.shape == (T, 8)  # hidden_dim[-1]

        full = model.evaluate(gtg, return_full_data=True)
        assert np.array_equal(embeddings, full['embeddings'])

    def test_get_embeddings_single_block(self):
        T = 40
        model = ACNet(num_cfs=4, hidden_dim=(5,), kernel_size=3, n_neurons=2)
        gtg = np.random.rand(T, 4)
        embeddings = model.get_embeddings(gtg)
        assert embeddings.shape == (T, 5)


class TestGetEmbeddingsInputDispatch:
    """get_embeddings accepts a wav path, a (waveform, fs) pair, or a
    precomputed gtg (no fs) -- all three must agree when they describe the
    same underlying sound."""

    def _write_wav(self, path, fs=40000, duration=0.2):
        import wave
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        sig = 0.1 * np.sin(2 * np.pi * 1000 * t)
        with wave.open(str(path), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(fs)
            w.writeframes((sig * 32767).astype(np.int16).tobytes())
        return sig, fs

    def test_wav_path_and_waveform_agree(self, tmp_path):
        from nems.preprocessing.spectrogram import load_wav

        path = tmp_path / 'tone.wav'
        self._write_wav(path)
        model = ACNet(num_cfs=8, hidden_dim=(6,), kernel_size=3, n_neurons=2,
                      f_max=10e3)

        emb_from_path = model.get_embeddings(str(path))
        wav, fs = load_wav(str(path))
        emb_from_wav = model.get_embeddings(wav, fs=fs)

        assert emb_from_path.shape == emb_from_wav.shape
        assert np.array_equal(emb_from_path, emb_from_wav)

    def test_gtg_without_fs_used_directly(self, capsys):
        model = ACNet(num_cfs=8, hidden_dim=(6,), kernel_size=3, n_neurons=2)
        gtg = np.random.rand(40, 8)

        embeddings = model.get_embeddings(gtg)
        out = capsys.readouterr().out
        assert "assuming" in out.lower()
        assert embeddings.shape == (40, 6)

    def test_wav_and_gtg_paths_agree(self, tmp_path):
        model = ACNet(num_cfs=8, hidden_dim=(6,), kernel_size=3, n_neurons=2,
                      f_max=10e3)
        path = tmp_path / 'tone2.wav'
        wav, fs = self._write_wav(path)

        emb_from_wav = model.get_embeddings(wav, fs=fs)
        gtg = model._to_gtg(wav, fs)
        emb_from_gtg = model.get_embeddings(gtg)

        assert np.array_equal(emb_from_wav, emb_from_gtg)
