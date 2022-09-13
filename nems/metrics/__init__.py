"""Collection of performance metrics and related utilities.

Score model outputs by prediction correlation and other metrics.
Measure equivalence, sparsity, and other properties of model predictions and/or
recorded neural responses.

# TODO: These contents don't all exist yet, sketching this out based on
#       the strfy proposal document.
Contents
--------
    `correlation.py` : As above, but compute Pearson correlation coefficient.
    `equivalence.py` : Measure functional similarity of different models.
    `sparseness.py`  : Measure lifetime sparseness of responses and predictions.

"""

from .correlation import correlation, noise_corrected_r

from nems.tools.lookup import FindCallable

metric_nicknames = {}
get_metric = FindCallable({**globals(), **metric_nicknames}, header='Metric')
