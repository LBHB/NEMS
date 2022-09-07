import pytest
import numpy as np


# Generic data fixtures for use by test_layers and others.
spectrogram_seed = 12345
response_seed = 678910
state_seed = 111213

# TODO: Replace with either useful synthetic data or a short snippet of
#       real data.
# TODO: These are based on LBHB's needs (i.e. sound spectrogram and state as
#       inputs, spiking responses as targets). Add tests for other data types.
@pytest.fixture
def spectrogram():
    """100 time bins, 18 spectral channels."""
    np.random.seed(spectrogram_seed)
    return np.random.rand(100, 18)

@pytest.fixture
def response():
    """100 time bins, 1 neural response."""
    np.random.seed(response_seed)
    return np.random.rand(100, 1)

@pytest.fixture
def population_response():
    """100 time bins, 10 neural responses."""
    np.random.seed(response_seed)
    return np.random.rand(100, 10)

@pytest.fixture
def state():
    """100 time bins, 2 state channels."""
    np.random.seed(state_seed)
    return np.random.rand(100, 2)

@pytest.fixture
def spectrogram_with_samples():
    """10 samples, 100 time bins, 18 spectral channels."""
    np.random.seed(spectrogram_seed)
    return np.random.rand(10, 100, 18)

@pytest.fixture
def response_with_samples():
    """10 samples, 100 time bins, 1 neural response."""
    np.random.seed(response_seed)
    return np.random.rand(10, 100, 1)

@pytest.fixture
def state_with_samples():
    """10 samples, 100 time bins, 2 state channels."""
    np.random.seed(state_seed)
    return np.random.rand(10, 100, 2)
