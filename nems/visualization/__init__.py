"""Collection of plotting utilities for NEMS objects.

Contents
--------
    fitter.py : Plot error vs fit iteration, prediction vs target, etc.
    layers.py : Layer-specific visualizations, like `stp_input_output`.
    model.py  : Whole-model plots, like `plot_model`.
    tools.py  : Helper functions for formatting matplotlib objects.

"""
import matplotlib.pyplot as plt
import matplotlib as mpl

font_size = 8
# colorblind palette from seaborn
palette = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']
#'figure.figsize': (8, 6),
params = {'legend.fontsize': font_size-2,
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'axes.spines.right': False,
          'axes.spines.top': False,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'font.size': font_size,
          'pdf.fonttype': 42,
          'ps.fonttype': 42,
          'axes.prop_cycle': mpl.cycler(color=palette)
          }
plt.rcParams.update(params)

from .model import plot_model, plot_model_outputs, plot_layer, simple_strf, plot_strf, input_heatmap, plot_nl, plot_model_list, plot_dstrf, plot_data, plot_predictions
from .layers import stp_input_output
from .fitter import (
    prediction_vs_target, iteration_vs_error, evals_per_iteration,
    parameter_space_pca, parameter_pca_table, scipy_gradient
    )
from .tools import standardize_axes
