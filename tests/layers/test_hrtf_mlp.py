"""Tests for HRTFGainLayerMLP and HRTFGainLayerMLPReg sample_from_priors behavior.

Verifies that sample_from_priors correctly resamples MLP weights/biases
and that JSON save/load does NOT resample (preserving fitted parameters).
"""

import pytest
import numpy as np

from nems import Model
from nems.layers.state import HRTFGainLayerMLP, HRTFGainLayerMLPReg


FREQ_BINS = 18
EARS = 2
HIDDEN_UNITS = (32, 16)
SHAPE = (FREQ_BINS, EARS)


@pytest.fixture
def dlc_input():
    """Fake DLC input: [dist1, az1_sin, az1_cos, dist2, az2_sin, az2_cos]."""
    np.random.seed(42)
    return np.random.randn(100, 6)


@pytest.fixture
def hrtfmlp_layer():
    return HRTFGainLayerMLP(shape=SHAPE, hidden_units=HIDDEN_UNITS)


@pytest.fixture
def hrtfmlpreg_layer():
    return HRTFGainLayerMLPReg(shape=SHAPE, hidden_units=HIDDEN_UNITS)


# --- Construction tests ---

class TestConstruction:

    def test_mlp_has_expected_parameters(self, hrtfmlp_layer):
        """MLP layer should have w1,b1,...,wN,bN parameters."""
        param_names = list(hrtfmlp_layer.parameters._dict.keys())
        n_mlp_layers = len(HIDDEN_UNITS) + 1
        expected = []
        for i in range(1, n_mlp_layers + 1):
            expected.extend([f'w{i}', f'b{i}'])
        assert param_names == expected

    def test_mlpreg_has_expected_parameters(self, hrtfmlpreg_layer):
        """MLPReg layer should have the same parameter structure as MLP."""
        param_names = list(hrtfmlpreg_layer.parameters._dict.keys())
        n_mlp_layers = len(HIDDEN_UNITS) + 1
        expected = []
        for i in range(1, n_mlp_layers + 1):
            expected.extend([f'w{i}', f'b{i}'])
        assert param_names == expected

    def test_mlp_parameters_not_all_prior_mean(self, hrtfmlp_layer):
        """After construction, parameters should be sampled (not all at prior mean)."""
        for name, param in hrtfmlp_layer.parameters._dict.items():
            values = param.values
            mean = param.prior.mean()
            assert not np.allclose(values, mean), \
                f"Parameter {name} is still at prior mean after construction"

    def test_mlpreg_parameters_not_all_prior_mean(self, hrtfmlpreg_layer):
        """After construction, MLPReg parameters should be sampled."""
        for name, param in hrtfmlpreg_layer.parameters._dict.items():
            values = param.values
            mean = param.prior.mean()
            assert not np.allclose(values, mean), \
                f"Parameter {name} is still at prior mean after construction"


# --- sample_from_priors tests ---

class TestSampleFromPriors:

    def test_layer_sample_changes_parameters(self, hrtfmlp_layer):
        """Calling sample_from_priors on the layer should change parameter values."""
        before = hrtfmlp_layer.get_parameter_values(as_dict=True)
        before = {k: v.copy() for k, v in before.items()}

        hrtfmlp_layer.sample_from_priors(inplace=True)

        after = hrtfmlp_layer.get_parameter_values(as_dict=True)
        any_changed = any(
            not np.allclose(before[k], after[k]) for k in before
        )
        assert any_changed, "sample_from_priors did not change any parameters"

    def test_layer_sample_changes_mlpreg_parameters(self, hrtfmlpreg_layer):
        """Calling sample_from_priors on MLPReg layer should change parameter values."""
        before = hrtfmlpreg_layer.get_parameter_values(as_dict=True)
        before = {k: v.copy() for k, v in before.items()}

        hrtfmlpreg_layer.sample_from_priors(inplace=True)

        after = hrtfmlpreg_layer.get_parameter_values(as_dict=True)
        any_changed = any(
            not np.allclose(before[k], after[k]) for k in before
        )
        assert any_changed, "sample_from_priors did not change any MLPReg parameters"

    def test_model_sample_returns_different_copy(self):
        """Model.sample_from_priors should return a new model with different weights."""
        model = Model.from_keywords(f'hrtfmlp.{FREQ_BINS}x{EARS}.h32.h16')
        original_params = model.layers[0].get_parameter_values(as_dict=True)
        original_params = {k: v.copy() for k, v in original_params.items()}

        sampled = model.sample_from_priors()

        sampled_params = sampled.layers[0].get_parameter_values(as_dict=True)
        any_changed = any(
            not np.allclose(original_params[k], sampled_params[k]) for k in original_params
        )
        assert any_changed, "Model.sample_from_priors returned identical parameters"

    def test_model_sample_does_not_mutate_original(self):
        """Model.sample_from_priors should not modify the original model."""
        model = Model.from_keywords(f'hrtfmlp.{FREQ_BINS}x{EARS}.h32.h16')
        original_params = model.layers[0].get_parameter_values(as_dict=True)
        original_params = {k: v.copy() for k, v in original_params.items()}

        _ = model.sample_from_priors()

        current_params = model.layers[0].get_parameter_values(as_dict=True)
        for k in original_params:
            assert np.allclose(original_params[k], current_params[k]), \
                f"sample_from_priors mutated original model parameter {k}"

    def test_multiple_samples_are_different(self):
        """Requesting n>1 samples should produce distinct parameter sets."""
        model = Model.from_keywords(f'hrtfmlp.{FREQ_BINS}x{EARS}.h32.h16')
        copies = model.sample_from_priors(n=3)

        assert len(copies) == 3
        for i in range(len(copies)):
            for j in range(i + 1, len(copies)):
                p_i = copies[i].layers[0].get_parameter_values()
                p_j = copies[j].layers[0].get_parameter_values()
                any_diff = any(
                    not np.allclose(a, b) for a, b in zip(p_i, p_j)
                )
                assert any_diff, f"Samples {i} and {j} have identical parameters"

    def test_model_sample_mlpreg(self):
        """Model.sample_from_priors should work for MLPReg keyword models."""
        model = Model.from_keywords(f'hrtfmlpreg.{FREQ_BINS}x{EARS}.h32.h16')
        original_params = model.layers[0].get_parameter_values(as_dict=True)
        original_params = {k: v.copy() for k, v in original_params.items()}

        sampled = model.sample_from_priors()

        sampled_params = sampled.layers[0].get_parameter_values(as_dict=True)
        any_changed = any(
            not np.allclose(original_params[k], sampled_params[k]) for k in original_params
        )
        assert any_changed, "Model.sample_from_priors returned identical MLPReg parameters"


# --- Evaluation consistency tests ---

class TestEvaluationConsistency:

    def test_sample_changes_output(self, dlc_input):
        """Different parameter samples should produce different MLP outputs."""
        layer = HRTFGainLayerMLP(shape=SHAPE, hidden_units=HIDDEN_UNITS)
        out1 = layer.evaluate(dlc_input)

        layer.sample_from_priors(inplace=True)
        out2 = layer.evaluate(dlc_input)

        assert not np.allclose(out1, out2), \
            "MLP output unchanged after sample_from_priors"

    def test_sample_changes_mlpreg_output(self, dlc_input):
        """Different parameter samples should produce different MLPReg outputs."""
        layer = HRTFGainLayerMLPReg(shape=SHAPE, hidden_units=HIDDEN_UNITS)
        out1 = layer.evaluate(dlc_input)

        layer.sample_from_priors(inplace=True)
        out2 = layer.evaluate(dlc_input)

        assert not np.allclose(out1, out2), \
            "MLPReg output unchanged after sample_from_priors"


# --- JSON save/load tests (the original bug) ---

class TestJsonRoundTrip:

    def test_json_roundtrip_preserves_mlp_parameters(self):
        """Saving and loading MLP layer via JSON should NOT resample parameters."""
        layer = HRTFGainLayerMLP(shape=SHAPE, hidden_units=HIDDEN_UNITS)
        original_params = layer.get_parameter_values(as_dict=True)
        original_params = {k: v.copy() for k, v in original_params.items()}

        json_str = layer.to_json()
        restored = HRTFGainLayerMLP.from_json(json_str)

        restored_params = restored.get_parameter_values(as_dict=True)
        for k in original_params:
            assert np.allclose(original_params[k], restored_params[k], atol=1e-6), \
                f"JSON roundtrip changed MLP parameter {k}"

    def test_json_roundtrip_preserves_mlpreg_parameters(self):
        """Saving and loading MLPReg layer via JSON should NOT resample parameters."""
        layer = HRTFGainLayerMLPReg(shape=SHAPE, hidden_units=HIDDEN_UNITS)
        original_params = layer.get_parameter_values(as_dict=True)
        original_params = {k: v.copy() for k, v in original_params.items()}

        json_str = layer.to_json()
        restored = HRTFGainLayerMLPReg.from_json(json_str)

        restored_params = restored.get_parameter_values(as_dict=True)
        for k in original_params:
            assert np.allclose(original_params[k], restored_params[k], atol=1e-6), \
                f"JSON roundtrip changed MLPReg parameter {k}"

    def test_model_json_roundtrip_preserves_parameters(self):
        """Full model JSON roundtrip should preserve MLP HRTF parameters."""
        model = Model.from_keywords(f'hrtfmlp.{FREQ_BINS}x{EARS}.h32.h16')
        original_params = model.layers[0].get_parameter_values(as_dict=True)
        original_params = {k: v.copy() for k, v in original_params.items()}

        json_str = model.to_json()
        restored = Model.from_json(json_str)

        restored_params = restored.layers[0].get_parameter_values(as_dict=True)
        for k in original_params:
            assert np.allclose(original_params[k], restored_params[k], atol=1e-6), \
                f"Model JSON roundtrip changed parameter {k}"

    def test_model_json_roundtrip_mlpreg(self):
        """Full model JSON roundtrip should preserve MLPReg parameters."""
        model = Model.from_keywords(f'hrtfmlpreg.{FREQ_BINS}x{EARS}.h32.h16')
        original_params = model.layers[0].get_parameter_values(as_dict=True)
        original_params = {k: v.copy() for k, v in original_params.items()}

        json_str = model.to_json()
        restored = Model.from_json(json_str)

        restored_params = restored.layers[0].get_parameter_values(as_dict=True)
        for k in original_params:
            assert np.allclose(original_params[k], restored_params[k], atol=1e-6), \
                f"Model JSON roundtrip changed MLPReg parameter {k}"


# --- Frozen parameter tests ---

class TestFrozenParameters:

    def test_frozen_layer_not_resampled(self):
        """Frozen MLP layers should not have parameters changed by sample_from_priors."""
        layer = HRTFGainLayerMLP(shape=SHAPE, hidden_units=HIDDEN_UNITS)
        original_params = layer.get_parameter_values(as_dict=True)
        original_params = {k: v.copy() for k, v in original_params.items()}

        layer.freeze_parameters()
        layer.sample_from_priors(inplace=True)

        after = layer.get_parameter_values(as_dict=True)
        for k in original_params:
            assert np.allclose(original_params[k], after[k]), \
                f"Frozen parameter {k} was changed by sample_from_priors"
