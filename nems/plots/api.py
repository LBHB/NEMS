import matplotlib.pyplot as plt
font_size=8
params = {'legend.fontsize': font_size-2,
          'figure.figsize': (8, 6),
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'axes.spines.right': False,
          'axes.spines.top': False,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

from .assemble import (simple_grid, freeze_defaults, get_predictions,
                       plot_layout, combine_signal_channels, pad_to_signals)
from .scatter import plot_scatter, nl_scatter, plot_nl_io
from .summary import plot_summary, perf_per_cell, pareto
from .specgram import (plot_spectrogram, spectrogram_from_signal,
                       spectrogram_from_epoch, spectrogram, pred_spectrogram,
                       resp_spectrogram, spectrogram_output)
from .timeseries import (plot_timeseries, mod_output, mod_output_all,
                         fir_output_all,
                         timeseries_from_signals, timeseries_from_epoch,
                         before_and_after, timeseries_from_vectors,
                         before_and_after_stp, pred_resp)
from .heatmap import (weight_channels_heatmap, fir_heatmap, strf_heatmap,
                      strf_timeseries, strf_local_lin)
from .file import save_figure, load_figure_img, load_figure_bytes, fig2BytesIO
from .histogram import pred_error_hist
from .state import (state_vars_timeseries, state_var_psth,
                    state_var_psth_from_epoch, state_vars_psth_all,
                    state_gain_parameters,
                    state_gain_plot, model_per_time, cc_comp,
                    cc_comp_per_mask)
from .quickplot import quickplot, before_and_after_scatter
from .diagnostic import diagnostic
from .raster import (raster, psth_from_raster, raster_from_epoch)
from .utils import (ax_remove_box)
