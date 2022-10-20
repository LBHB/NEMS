import numpy as np

from nems.models.base import DataSet


def test_array_input(spectrogram, response, state):
    # Array, no samples.
    d = DataSet(input=spectrogram)
    assert d.inputs == {DataSet.default_input: spectrogram}
    assert d.outputs == {}
    assert d.targets == {}

    d2 = DataSet(input=spectrogram, state=state, target=response)
    assert d2.inputs == {DataSet.default_input: spectrogram, 
                         DataSet.default_state: state}
    assert d2.outputs == {}
    assert d2.targets == {DataSet.default_target: response}


def test_dict_input(spectrogram, response):
    input_dict = {'stimulus': spectrogram}
    target_dict = {'response': response}
    d = DataSet(input=input_dict, target=target_dict)
    # Everything is specified as dict, so they should just be copied in
    # with the same keys and references to the same arrays.
    assert d.inputs == input_dict
    assert d.targets == target_dict
    assert np.shares_memory(d.inputs['stimulus'], input_dict['stimulus'])
    assert np.shares_memory(d.targets['response'], target_dict['response'])
    # as_dict should combine all inputs and targets (and outputs, but none of
    # those in this case).
    assert d.as_dict() == {**input_dict, **target_dict}


def test_samples(spectrogram, spectrogram_with_samples):
    d = DataSet(input=spectrogram)
    # DataSet doesn't know if there are samples are not, wasn't specified.
    assert d.n_samples is None
    
    d2 = DataSet(input=spectrogram_with_samples, has_samples=False)
    # Doesn't matter if the array actually has samples or not, DataSet has no
    # way to differentiate what the axes mean.
    assert d.n_samples is None
    
    d3 = d.prepend_samples()
    # This should add a sample axis and set `has_samples=True`.
    assert d3.inputs[DataSet.default_input].shape == (1,) + spectrogram.shape
    assert d3.n_samples == 1

    d4 = d3.squeeze_samples()
    # Remove sample axis, set `has_samples=False`.
    assert d4.inputs[DataSet.default_input].shape == spectrogram.shape
    assert d4.n_samples is None
    # All of these operations should have preserved memory sharing.
    assert np.shares_memory(d4.inputs[DataSet.default_input], spectrogram)

    # This shouldn't actually change anything since there's just one array, but
    # should still set `has_samples=True` since this method assumes samples
    # exist.
    d5 = d2.as_broadcasted_samples()
    assert d5.n_samples == spectrogram_with_samples.shape[0]
    assert d5.inputs[DataSet.default_input].shape == \
        spectrogram_with_samples.shape
    # Broadcasting shouldn't have changed any values or made copies.
    assert np.allclose(d5.inputs[DataSet.default_input],
                       spectrogram_with_samples)
    assert np.shares_memory(d5.inputs[DataSet.default_input],
                            spectrogram_with_samples)


def test_batches(spectrogram, response, state_with_samples):
    # Add singleton sample axes
    spectrogram = spectrogram[np.newaxis, ...]
    response = response[np.newaxis, ...]
    d = DataSet(input=spectrogram, state=state_with_samples, target=response)
    # This should broadcast sample sizes both for spectrogram -> state and
    # target -> inputs, so that all arrays end up with the same number of
    # samples on the first axis.
    d2 = d.as_broadcasted_samples()
    assert d2.n_samples == state_with_samples.shape[0]
    assert d2.inputs[DataSet.default_input].shape[0] == \
        d2.inputs[DataSet.default_state].shape[0]
    assert d2.inputs[DataSet.default_input].shape[0] == \
        d2.targets[DataSet.default_target].shape[0]

    # This should result in a single batch of size `d2.n_samples`.
    batch_generator = d2.as_batches(batch_size=None)
    for batch in batch_generator:
        assert batch.n_samples == d2.n_samples
    # Should generate individual samples, each with no sample dimension.
    sample_generator = batch.as_samples()
    for sample in sample_generator:
        assert not sample.has_samples
        assert sample.n_samples is None
        assert sample.inputs[DataSet.default_input].shape == \
            spectrogram.shape[1:]
        # Since spectrogram was broadcast on samples, each sample should be
        # a view of the original spectrogram.
        assert np.shares_memory(sample.inputs[DataSet.default_input],
                                spectrogram)

    # This should result in multiple batches, all size 1.
    batch_generator = d2.as_batches(batch_size=1)
    for batch in batch_generator:
        assert batch.has_samples
        assert batch.n_samples == 1


def test_1d_target(spectrogram, response):
    response_1d = response[:, 0]
    d = DataSet(input=spectrogram, target=response_1d)
    assert response_1d.ndim == 1
    assert d.targets[d.target_name].ndim == 2
    assert np.allclose(response_1d, d.targets[d.target_name].flatten())
