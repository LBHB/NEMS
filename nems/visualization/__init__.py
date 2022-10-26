"""Collection of plotting utilities for NEMS objects.

Contents
--------
    fitter.py : Plot error vs fit iteration, prediction vs target, etc.
    layers.py : Layer-specific visualizations, like `stp_input_output`.
    model.py  : Whole-model plots, like `plot_model`.
    tools.py  : Helper functions for formatting matplotlib objects.

"""

from .model import plot_model, plot_layer, simple_strf, plot_strf, input_heatmap
from .layers import stp_input_output
from .fitter import (
    prediction_vs_target, iteration_vs_error, evals_per_iteration,
    parameter_space_pca, parameter_pca_table, scipy_gradient
    )
from .tools import standardize_axes
