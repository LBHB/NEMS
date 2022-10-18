"""Docs TODO"""

from .model import plot_model, plot_layer, simple_strf, plot_strf, input_heatmap
from .tools import make_axes_plotter
from .fitter import (
    prediction_vs_target, iteration_vs_error,evals_per_iteration,
    parameter_space_pca, parameter_pca_table, scipy_gradient
    )
