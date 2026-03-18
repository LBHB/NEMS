"""Focused tests for the TensorFlow backend fit loop.

These tests are intentionally narrow. Each one targets one contract of the
current TF fitting path so that regressions are easy to localize:

- accepted input container types
- fitter-option behavior around validation
- inclusion of layer-added regularization losses
- cleanup behavior across repeated fits
- known failures that should flip from xfail to pass once fixed

The goal is not broad model-quality testing. The goal is to isolate the
fit-loop plumbing that `fit_lite` and other real pipelines depend on.
"""

import gc
import logging

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nems import Model
from nems.backends.tf.backend import TensorFlowBackend
from nems.backends.tf.cost import get_cost
from nems.backends.tf.tools import simple_generator
from nems.layers.state import HRTFGainLayerMLPReg
from nems.models.dataset import DataSet


def _make_array_data(n_samples=8, time_bins=20, input_channels=3, output_channels=1):
    """Create deterministic small arrays for fast, repeatable fit-loop tests."""
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, time_bins, input_channels)).astype(np.float32)
    y = rng.normal(size=(n_samples, time_bins, output_channels)).astype(np.float32)
    return x, y


def _make_linear_model():
    """Return a minimal TF-compatible model with a conventional single output."""
    return Model.from_keywords("wc.3x1-fir.3x1-dexp.1")


def _tf_history_from_model(model):
    """Unwrap the current FitResults misc nesting and return the recorded history."""
    return model.results.misc["misc"]["TensorFlow History"]


class CountingSequence(tf.keras.utils.Sequence):
    """Minimal Sequence that counts `on_epoch_end()` calls.

    This is used to verify that the custom fit loop preserves one of the key
    Keras Sequence contracts: the sequence gets an epoch boundary callback once
    per epoch.
    """

    def __init__(self, x, y, batch_size=2):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.on_epoch_end_calls = 0

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = min((index + 1) * self.batch_size, len(self.y))
        return {"input": self.x[start:stop]}, self.y[start:stop]

    def on_epoch_end(self):
        self.on_epoch_end_calls += 1


def _build_tf_backend(model, x, y, batch_size=2):
    """Build a TensorFlowBackend directly for backend-level assertions.

    Some tests need to compare the backend's recorded losses against a manual
    recomputation. Those are easier to write against `TensorFlowBackend._fit`
    than through `Model.fit`.
    """
    _ = model.evaluate(x[:batch_size], batch_size=batch_size, use_existing_maps=False)
    data = DataSet(x, target=y)
    backend = TensorFlowBackend(
        model.copy(), data, verbose=0, eval_kwargs={"batch_size": batch_size}
    )
    return backend, data


@pytest.mark.parametrize("mode", ["array", "simple_generator", "dataset"])
def test_tf_model_fit_accepts_array_generator_and_dataset(mode):
    """Check that the public `Model.fit` entry point accepts all supported input modes.

    Why this test exists:
    `fit_lite` and related code paths may hand TF either raw arrays, the local
    `simple_generator`, or a `tf.data.Dataset`. This test makes sure all three
    routes get through `Model.fit` and the backend records a loss history.

    How it checks:
    It runs one epoch for each input mode and asserts that the fit produced a
    TensorFlow history with exactly one `loss` entry.
    """
    x, y = _make_array_data()
    model = _make_linear_model()
    fitter_options = {
        "cost_function": "squared_error",
        "epochs": 1,
        "early_stopping_tolerance": 0,
    }

    if mode == "array":
        fitted = model.fit(
            x, y, batch_size=2, backend="tf", fitter_options=fitter_options, verbose=0
        )
    elif mode == "simple_generator":
        generator = simple_generator({"input": x}, y, batch_size=2, shuffle=False)
        fitted = model.fit(
            generator,
            None,
            batch_size=2,
            backend="tf",
            fitter_options=fitter_options,
            verbose=0,
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(({"input": x}, y)).batch(2)
        fitted = model.fit(
            dataset,
            None,
            batch_size=2,
            backend="tf",
            fitter_options=fitter_options,
            verbose=0,
        )

    history = _tf_history_from_model(fitted)
    assert "loss" in history
    assert len(history["loss"]) == 1


def test_tf_sequence_calls_on_epoch_end_each_epoch():
    """Verify that the custom fit loop preserves the Sequence epoch callback.

    Why this test exists:
    When the backend stopped using `keras.Model.fit` for Sequence inputs, it
    became responsible for calling `on_epoch_end()` itself. Missing that call
    breaks shuffling and any Sequence state reset logic.

    How it checks:
    The custom Sequence increments a counter in `on_epoch_end()`. The test fits
    for three epochs and checks both the counter and the recorded loss history.
    """
    x, y = _make_array_data()
    sequence = CountingSequence(x, y, batch_size=2)
    model = _make_linear_model()

    fitted = model.fit(
        sequence,
        None,
        batch_size=2,
        backend="tf",
        fitter_options={
            "cost_function": "squared_error",
            "epochs": 3,
            "early_stopping_tolerance": 0,
        },
        verbose=0,
    )

    assert len(_tf_history_from_model(fitted)["loss"]) == 3
    assert sequence.on_epoch_end_calls == 3


def test_tf_array_validation_data_records_val_loss():
    """Verify the working validation path for array input plus tuple validation data.

    Why this test exists:
    This is the explicit validation path the current custom loop does support.
    It should keep working even as the fit loop changes internally.

    How it checks:
    Training data and validation data are passed separately, then the test
    asserts that `val_loss` was recorded for the single epoch.
    """
    x, y = _make_array_data()
    model = _make_linear_model()

    fitted = model.fit(
        x[2:],
        y[2:],
        batch_size=2,
        backend="tf",
        fitter_options={
            "cost_function": "squared_error",
            "epochs": 1,
            "validation_data": (x[:2], y[:2]),
            "early_stopping_tolerance": 0,
        },
        verbose=0,
    )

    history = _tf_history_from_model(fitted)
    assert "val_loss" in history
    assert len(history["val_loss"]) == 1


def test_tf_array_validation_split_records_val_loss():
    """Verify that `validation_split` correctly records val_loss.

    Why this test exists:
    `fit_lite` forwards `validation_split` into the TF backend. If the backend
    ignores it, real fitting pipelines silently lose validation monitoring.

    How it checks:
    The test asks for `validation_split=0.25` and expects `val_loss` to appear.
    Today it does not, so this test is marked `xfail` until fixed.
    """
    x, y = _make_array_data()
    model = _make_linear_model()

    fitted = model.fit(
        x,
        y,
        batch_size=2,
        backend="tf",
        fitter_options={
            "cost_function": "squared_error",
            "epochs": 1,
            "validation_split": 0.25,
            "early_stopping_tolerance": 0,
        },
        verbose=0,
    )

    assert "val_loss" in _tf_history_from_model(fitted)


def test_tf_dataset_validation_data_accepts_batched_dataset():
    """Verify that a batched `tf.data.Dataset` can be passed as `validation_data`.

    Why this test exists:
    The backend claims to handle `tf.data.Dataset` inputs, but validation data
    should work in dataset form too if the train input is a dataset.

    How it checks:
    The test passes batched train and validation datasets through `Model.fit`
    and expects a recorded `val_loss`. The current backend raises before that
    happens, so this is `xfail`.
    """
    x, y = _make_array_data()
    train_dataset = tf.data.Dataset.from_tensor_slices(({"input": x[:6]}, y[:6])).batch(2)
    val_dataset = tf.data.Dataset.from_tensor_slices(({"input": x[6:]}, y[6:])).batch(2)
    model = _make_linear_model()

    fitted = model.fit(
        train_dataset,
        None,
        batch_size=2,
        backend="tf",
        fitter_options={
            "cost_function": "squared_error",
            "epochs": 1,
            "validation_data": val_dataset,
            "early_stopping_tolerance": 0,
        },
        verbose=0,
    )

    assert "val_loss" in _tf_history_from_model(fitted)


def test_tf_custom_layer_losses_are_added_to_fit_loss_and_logged(caplog):
    """Check that layer-added TF losses are included in training loss and logged.

    Why this test exists:
    Several HRTF/state layers call `add_loss()` and `add_metric()` inside their
    TF implementation. The custom fit loop must include those losses in the
    optimized objective and keep their metrics visible in the log.

    How it checks:
    The test manually recomputes per-batch `(raw loss + regularization loss)`
    from the built backend model, averages those totals, and compares that to
    `results.final_error`. It also captures logs and looks for the regularizer
    metric name.
    """
    rng = np.random.default_rng(456)
    x = rng.normal(size=(4, 5, 6)).astype(np.float32)
    y = np.zeros((4, 5, 12), dtype=np.float32)
    layer = HRTFGainLayerMLPReg(
        shape=(3, 2),
        hidden_units=(4, 4, 4),
        hrtf_regularization={"freq_smooth": 1000.0},
    )
    model = Model(layers=[layer])
    backend, data = _build_tf_backend(model, x, y, batch_size=2)
    cost = get_cost("squared_error")

    expected_losses = []
    for start in range(0, len(y), 2):
        batch_x = {"input": tf.constant(x[start : start + 2])}
        batch_y = tf.constant(y[start : start + 2])
        preds = backend.model(batch_x, training=True)
        raw_loss = cost(batch_y, preds)
        reg_loss = tf.add_n(backend.model.losses)
        expected_losses.append(float(raw_loss + reg_loss))

    with caplog.at_level(logging.INFO, logger="nems.backends.tf.backend"):
        results = backend._fit(
            data,
            eval_kwargs={"batch_size": 2},
            cost_function="squared_error",
            learning_rate=0.0,
            epochs=1,
            early_stopping_tolerance=0,
        )

    assert np.mean(expected_losses) == pytest.approx(results.final_error)
    assert "hrtf_freq_smooth" in caplog.text


def test_tf_initial_error_includes_custom_layer_losses():
    """Verify that initial_error accounts for model.losses (regularization) consistently with training loss.

    Why this test exists:
    The custom loop adds `model.losses` during train/validation steps, so the
    pre-fit `initial_error` should be computed on the same basis. Otherwise
    init/final loss comparisons become misleading.

    How it checks:
    It manually computes the expected pre-fit batch losses including
    regularization and compares that average to `results.initial_error[0]`.
    This currently fails, so it stays `xfail`.
    """
    rng = np.random.default_rng(789)
    x = rng.normal(size=(4, 5, 6)).astype(np.float32)
    y = np.zeros((4, 5, 12), dtype=np.float32)
    layer = HRTFGainLayerMLPReg(
        shape=(3, 2),
        hidden_units=(4, 4, 4),
        hrtf_regularization={"freq_smooth": 1000.0},
    )
    model = Model(layers=[layer])
    backend, data = _build_tf_backend(model, x, y, batch_size=2)
    cost = get_cost("squared_error")

    expected_losses = []
    for start in range(0, len(y), 2):
        batch_x = {"input": tf.constant(x[start : start + 2])}
        batch_y = tf.constant(y[start : start + 2])
        preds = backend.model(batch_x, training=False)
        raw_loss = cost(batch_y, preds)
        reg_loss = tf.add_n(backend.model.losses)
        expected_losses.append(float(raw_loss + reg_loss))

    results = backend._fit(
        data,
        eval_kwargs={"batch_size": 2},
        cost_function="squared_error",
        learning_rate=0.0,
        epochs=1,
        early_stopping_tolerance=0,
    )

    assert np.mean(expected_losses) == pytest.approx(results.initial_error[0])


def test_tf_model_fit_clears_session_for_non_in_place(monkeypatch):
    """Verify cleanup after ordinary TF fits.

    Why this test exists:
    A major motivation for the backend refactor was repeated-fit memory growth.
    `Model.fit` now clears the TF session and runs `gc.collect()` after a
    non-`in_place` fit; this behavior should stay explicit and tested.

    How it checks:
    The test monkeypatches `clear_session()` and `gc.collect()` to record calls
    and asserts that both are invoked once after fitting.
    """
    x, y = _make_array_data()
    model = _make_linear_model()
    calls = []

    def fake_clear_session():
        calls.append("clear_session")

    def fake_collect():
        calls.append("gc_collect")
        return 0

    monkeypatch.setattr(tf.keras.backend, "clear_session", fake_clear_session)
    monkeypatch.setattr(gc, "collect", fake_collect)

    model.fit(
        x,
        y,
        batch_size=2,
        backend="tf",
        fitter_options={
            "cost_function": "squared_error",
            "epochs": 1,
            "early_stopping_tolerance": 0,
        },
        verbose=0,
    )

    assert calls == ["clear_session", "gc_collect"]


def test_tf_model_fit_skips_cleanup_when_retain_backend(monkeypatch):
    """Verify that cleanup is skipped when the caller explicitly retains the backend.

    Why this test exists:
    The non-retained cleanup is correct for batch fitting, but some callers
    need access to the backend object after fitting without switching to
    `in_place=True` and mutating the original model.

    How it checks:
    The same monkeypatched cleanup hooks are used as above, but the fit is run
    with `retain_backend=True`. The test asserts that no cleanup calls were
    made and that the returned model still has a backend object.
    """
    x, y = _make_array_data()
    model = _make_linear_model()
    calls = []

    def fake_clear_session():
        calls.append("clear_session")

    def fake_collect():
        calls.append("gc_collect")
        return 0

    monkeypatch.setattr(tf.keras.backend, "clear_session", fake_clear_session)
    monkeypatch.setattr(gc, "collect", fake_collect)

    fitted = model.fit(
        x,
        y,
        batch_size=2,
        backend="tf",
        fitter_options={
            "cost_function": "squared_error",
            "epochs": 1,
            "early_stopping_tolerance": 0,
        },
        retain_backend=True,
        verbose=0,
    )

    assert calls == []
    assert fitted.backend is not None


def test_tf_hrtfmlpreg_rejects_wrong_hidden_layer_count():
    """Verify that HRTFGainLayerMLPReg raises on construction with != 3 hidden layers.

    Why this test exists:
    The TF implementation hard-codes exactly 3 hidden layers (w1→w2→w3→output).
    Rather than silently producing wrong results, the layer now raises at
    construction time with a clear message when the wrong count is supplied.
    """
    from nems.layers.state import HRTFGainLayerMLPReg
    with pytest.raises(ValueError, match="exactly 3 hidden layers"):
        HRTFGainLayerMLPReg(shape=(3, 2), hidden_units=(4, 4))


def test_tf_hrtfmlpreg_three_hidden_layers_fits():
    """Verify that a 3-hidden-layer hrtfmlpreg keyword model fits correctly.

    Why this test exists:
    Confirms the standard 3-hidden-layer path works end-to-end with the TF
    backend, including variable unit sizes per layer.
    """
    rng = np.random.default_rng(321)
    x = rng.normal(size=(4, 5, 6)).astype(np.float32)
    y = np.zeros((4, 5, 12), dtype=np.float32)
    model = Model.from_keywords("hrtfmlpreg.3x2.h8.h4.h4")

    fitted = model.fit(
        x,
        y,
        batch_size=2,
        backend="tf",
        fitter_options={
            "cost_function": "squared_error",
            "epochs": 1,
            "early_stopping_tolerance": 0,
        },
        verbose=0,
    )

    assert "loss" in _tf_history_from_model(fitted)
