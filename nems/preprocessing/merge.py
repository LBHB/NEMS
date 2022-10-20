# TODO
# From supplement doc:
# "align and combine datasets for fitting a single, large model"

# So, for example, combining PEG and A1 data to fit with a single pop model.

from nems.tools.arrays import concatenate_dicts


def combine_neurons(*datasets, neurons_keys='response', axes=-1, newaxis=False):
    """Merge multiple datasets by concatenating on neural variables.

    This is essentially a wrapper for `concatenate_dicts` (and, ultimately,
    `np.concatenate`) with some extra options parsing.

    Parameters
    ----------
    datasets : N-tuple of dict.
        Each dictionary should be in the format expected for the `input` arg of
        `nems.Model.evaluate`. That is:
            `{'key1': np.ndarray, 'key2': np.ndarray, ...}`,
        where each array has shape (T, ..., N) or (S, T, ..., N)

    """
