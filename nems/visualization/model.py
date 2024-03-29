import collections.abc
import math
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .tools import ax_remove_box, ax_bins_to_seconds
from nems import metrics
from nems import preprocessing
from nems.tools.dstrf import compute_dpcs
from nems.metrics import correlation

_DEFAULT_PLOT_OPTIONS = {
    'skip_plot_options': False,
    'show_x': False, 'xlabel': None, 'xmax_scale': 1, 'xmin_scale': 1,
    'show_y': True, 'ylabel': None, 'ymax_scale': 1, 'ymin_scale': 1,
    'show_seconds': True,
    'legend': False,
    'margins': (0,.1),
    # Right of axis by default, aligned to top
    'legend_kwargs': {
        'frameon': False, 'bbox_to_anchor': (1, 1), 'loc': 'upper left'
        },
    }
_TEXT_BBOX = dict(boxstyle='round, pad=.25, rounding_size=.15', alpha=.7, facecolor='white')

def set_plot_options(ax, layer_options, time_kwargs=None):
    """Adjust matplotlib axes object in-place according to `layer_options`.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    layer_options : dict
        Layer-specific options with the same dictionary structure as
        `_DEFAULT_PLOT_OPTIONS`. Layer-specific options take precedence.
    time_kwargs : dict; optional.
        Keyword arguments for `.tools.ax_bins_to_seconds`.

    Returns
    -------
    ops : dict
        Merged plot options from `_DEFAULT_PLOT_OPTIONS` and `layer_options`.

    See also
    --------
    nems.layers.base.Layer.plot_options
    
    """
    _dict = copy.deepcopy(_DEFAULT_PLOT_OPTIONS)
    ops = _nested_update(_dict, layer_options)
    if ops['skip_plot_options']: return

    # x-axis
    ax.xaxis.set_visible(ops['show_x'])
    ax.set_xlabel(ops['xlabel'])
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin*ops['xmin_scale'], xmax*ops['xmax_scale'])

    # y-axis
    ax.yaxis.set_visible(ops['show_y'])
    ax.set_ylabel(ops['ylabel'])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin*ops['ymin_scale'], ymax*ops['ymax_scale'])

    # Convert bins -> seconds
    if (time_kwargs is not None) and (ops['show_seconds']):
        ax_bins_to_seconds(ax, **time_kwargs)

    # Add legend
    if ops['legend']:
        ax.legend(**ops['legend_kwargs'])
    
    # Set margins of ax
    ax.margins(*ops['margins'])

    # Remove top and right segments of border around axes
    ax_remove_box(ax)

    return ops


def _nested_update(d, u):
    """Merge two dictionaries that may themselves contain nested dictionaries.
    
    Internal for `set_plot_options`. Using this so that `Layer.plot_options`
    can update some keys of nested dicts (like 'legend_kwargs') without losing
    defaults for other keys.
    TODO: maybe move this to generic NEMS tools? Could be useful elsewhere.
    
    References
    ----------
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def plot_model_outputs(model, input, target=None, target_name=None, n=None,
               select_layers=None, n_columns=1, show_titles=True,
               figure_kwargs=None, sampling_rate=None, time_axis='x',
               conversion_factor=1, decimals=2,
               **eval_kwargs):
    """Plot result of `Layer.evaluate` for each Layer in a Model.
    
    Aliased as `Model.plot_output()`.
    TODO: Insert plot of input on first axis, as with old NEMS? But `input` may
          not always be a single spectrogram now, so user would have to specify
          which array to plot, whether to use imshow vs plot vs something else,
          etc... may not be worth it. Alternatively, could add option to insert
          subfigures (possibly at arbitrary positions?) with user-generated
          content, but aligning the x-axis for those could be tricky.

    TODO: SVD's version (plot_model_with_parameters) does this, can just merge
          that logic possibly.

    Parameters
    ----------
    model : Model
        See `nems.models.base.Model`.
    input : ndarray or dict.
        Input data for Model. See `Model.evaluate` for expected format.
    target : ndarray or list of ndarray; optional.
        Target data for Model. See `Model.fit` for expected format. If provided,
        target(s) will be shown on the last axis of the final layer's figure.
    n : int or None; optional.
        Number of Layers to plot. Defaults to all.
    select_layers : str, int, slice, or list; optional.
        Selects layers to plot. Any valid `index` for `Model.layers[index]` is
        acceptable. Ex: `layers = slice(1,3)` to only plot the Layers at indices
        1 and 2, or `layers = ['fir', 'dexp']` to only plot the Layers named
        'fir' and 'dexp'.
        NOTE: This option supercedes `n` if neither is None.
    n_columns : int; default=1.
        Number of columns to arrange plots on (all columns within a row will
        be filled before moving to the next row). Note that values > 1 will
        reduce the width of subplots slightly to make room for the default
        legend placement. However, it may still be necessary to increase
        figure size if the number of columns is large.
    show_titles : bool; default=True.
        Specify whether to show `Layer.name` as a title above each Layer's
        subfigure.
    figure_kwargs : dict or None; optional.
        Keyword arguments for `matplotlib.pyplot.figure`.
        Ex: `figure_kwargs={'figsize': (10,10)}`.
    sampling_rate : float; optional.
        Sampling rate (in Hz) for `input` (and `target` if given), used to
        convert time_bin labels to seconds (if not None).
    time_axis : str; default='x'.
        'x' or 'y', the axis on the resulting Matplotlib axes objects that
        represents time.
    conversion_factor : int; default=1.
        Multiply seconds by this number to get different units. Floating point
        values must correspond to rational numbers. Scientific notation may only
        be used for multiples of 10, and will be converted to an integer.
        Ex: `conversion_factor=1000` to get units of milliseconds.
            `conversion_factor=1/60` to get units of minutes.
    decimals : int; default=2.
        Number of decimal places to show on new tick labels.
    eval_kwargs : dict; optional.
        Additional keyword arguments to supply to `Model.evaluate`.
        Ex: `input_name='stimulus'`.

    Returns
    -------
    fig, axes : (matplotlib.figure.Figure, matplotlib.axes.Axes)
    
    See also
    --------
    nems.models.base.Model
    nems.layers.base.Layer.plot
    nems.visualization.tools.ax_bins_to_seconds

    """

    batch_size = eval_kwargs.get('batch_size', 0)
    if batch_size != 0:
        raise NotImplementedError(
            "Model visualizations are not yet implemented for batched data.\n"
            "To visualize input, collapse sample dimension."
            )

    # Collect options for `ax_bins_to_seconds`
    if sampling_rate is not None:
        time_kwargs = {
            'sampling_rate': sampling_rate, 'time_axis': time_axis,
            'conversion_factor': conversion_factor, 'decimals': decimals
            }
    else:
        time_kwargs = None

    # Determine which figures to plot and set figure layout.
    # One subfigure per Layer, all in a single column by default
    # (otherwise fill columns before rows).
    figure_kwargs = {} if figure_kwargs is None else figure_kwargs
    figure = plt.figure(**figure_kwargs)
    if select_layers is not None:
        layers = model.layers.__getitem__(select_layers)
        if not isinstance(layers, list):
            layers = [layers]
    else:
        layers = model.layers[:n]

    n_rows = math.ceil(len(layers)/n_columns)
    subfigs = figure.subfigures(n_rows, n_columns)

    if isinstance(subfigs, matplotlib.figure.SubFigure):
        # Only one subfigure
        subfigs = np.array([subfigs])

    layer_info = model.generate_layer_data(input, **eval_kwargs)
    iterator = enumerate(zip(layers, subfigs, layer_info))
    for i, (layer, subfig, info) in iterator:
        output = info['out']
        layer.plot(output, fig=subfig, **layer.plot_kwargs)
        if show_titles:
            # Make room for titles
            subfig.subplots_adjust(top=0.8)
            name = layer.name
            subfig.suptitle(f'({model.get_layer_index(name)}) {name}')
        if n_columns > 1:
            # For multiple columns, shrink width of axes a bit to make room for
            # default legend placement.
            subfig.subplots_adjust(right=0.8)
        for ax in subfig.axes:
            set_plot_options(ax, layer.plot_options, time_kwargs=time_kwargs)

    # Add alternating white/gray shading to make Layers easier to distinguish.
    for shaded_fig in subfigs[checkerboard(subfigs)]:
        shaded_fig.patch.set_facecolor((0, 0, 0, 0.075))  # black, low alpha

    # Final x-axis of the final layer in each column is always visible
    # so that time is visually synchronized for all plots above.
    if len(subfigs.shape) == 1:
        bottom_figs = [subfigs[-1]]
    else:
        bottom_figs = subfigs[-1,:]
    for bottom_fig in bottom_figs:
        for ax in bottom_fig.axes:
            ax.xaxis.set_visible(True)

    # Add plot of target if given, on last axis of last subfig.
    last_ax = subfig.axes[-1]
    if target_name is not None:
        target = input[target_name]
    else:
        target_name = 'Target'
    if target is not None:
        if not isinstance(target, list):
            target = [target]
        for i, y in enumerate(target):
            last_ax.plot(y, label=f'{target_name} {i}', alpha=0.3, color='black')
        last_ax.legend(**_DEFAULT_PLOT_OPTIONS['legend_kwargs'])
        last_ax.autoscale()

    return figure

def compact_label(ax, text, va='top', **textargs):
    x_pos = ax.get_xlim()[0]
    y_pos = ax.get_ylim()[1]
    ax.text(x_pos, y_pos, text, bbox=_TEXT_BBOX, va=va, **textargs)


def plot_model(model, input, target=None, target_name=None, n=None,
               select_layers=None, n_columns=1, show_titles=True,
               figure_kwargs=None, sampling_rate=None, time_axis='x',
               conversion_factor=1, decimals=2, plot_input=True, T_max=1000,
               **eval_kwargs):
    """TODO: revise doc.

    TODO: Combine with plot_model and/or break out some of the copy-pasted code
          as separate functions.

    Parameters
    ----------
    model : Model
        See `nems.models.base.Model`.
    input : ndarray or dict.
        Input data for Model. See `Model.evaluate` for expected format.
    target : ndarray or list of ndarray; optional.
        Target data for Model. See `Model.fit` for expected format. If provided,
        target(s) will be shown on the last axis of the final layer's figure.
    n : int or None; optional.
        Number of Layers to plot. Defaults to all.
    select_layers : str, int, slice, or list; optional.
        Selects layers to plot. Any valid `index` for `Model.layers[index]` is
        acceptable. Ex: `layers = slice(1,3)` to only plot the Layers at indices
        1 and 2, or `layers = ['fir', 'dexp']` to only plot the Layers named
        'fir' and 'dexp'.
        NOTE: This option supercedes `n` if neither is None.
    n_columns : int; default=1.
        Number of columns to arrange plots on (all columns within a row will
        be filled before moving to the next row). Note that values > 1 will
        reduce the width of subplots slightly to make room for the default
        legend placement. However, it may still be necessary to increase
        figure size if the number of columns is large.
    show_titles : bool; default=True.
        Specify whether to show `Layer.name` as a title above each Layer's
        subfigure.
    figure_kwargs : dict or None; optional.
        Keyword arguments for `matplotlib.pyplot.figure`.
        Ex: `figure_kwargs={'figsize': (10,10)}`.
    sampling_rate : float; optional.
        Sampling rate (in Hz) for `input` (and `target` if given), used to
        convert time_bin labels to seconds (if not None).
    time_axis : str; default='x'.
        'x' or 'y', the axis on the resulting Matplotlib axes objects that
        represents time.
    conversion_factor : int; default=1.
        Multiply seconds by this number to get different units. Floating point
        values must correspond to rational numbers. Scientific notation may only
        be used for multiples of 10, and will be converted to an integer.
        Ex: `conversion_factor=1000` to get units of milliseconds.
            `conversion_factor=1/60` to get units of minutes.
    decimals : int; default=2.
        Number of decimal places to show on new tick labels.
    eval_kwargs : dict; optional.
        Additional keyword arguments to supply to `Model.evaluate`.
        Ex: `input_name='stimulus'`.

    Returns
    -------
    fig, axes : (matplotlib.figure.Figure, matplotlib.axes.Axes)

    See also
    --------
    nems.models.base.Model
    nems.layers.base.Layer.plot
    nems.visualization.tools.ax_bins_to_seconds

    """

    # Collect options for `ax_bins_to_seconds`
    if sampling_rate is not None:
        time_kwargs = {
            'sampling_rate': sampling_rate, 'time_axis': time_axis,
            'conversion_factor': conversion_factor, 'decimals': decimals
            }
    else:
        time_kwargs = None

    # Determine which figures to plot and set figure layout.
    # One subfigure per Layer, all in a single column by default
    # (otherwise fill columns before rows).
    figure_kwargs = {} if figure_kwargs is None else figure_kwargs
    figure = plt.figure(**figure_kwargs)

    if select_layers is not None:
        layers = model.layers.__getitem__(select_layers)
        if not isinstance(layers, list):
            layers = [layers]
    else:
        layers = model.layers[:n]
    layer_info = model.generate_layer_data(input, **eval_kwargs)

    if plot_input:
        n_rows = len(layers)+1
    else:
        n_rows = len(layers)
    if target is not None:
        n_rows += 1

    # Setting up our layout for plotting layers and parameters
    spec = figure.add_gridspec(n_rows, 3,
                               left=0.05, right=0.95, top=0.95, bottom=0.05,
                               wspace=0.1, hspace=0.15)
    subaxes = [figure.add_subplot(spec[n, 1:]) for n in range(n_rows)]
    parmaxes = [figure.add_subplot(spec[n+1, 0]) for n in range(n_rows-1)]
    last_ax = subaxes[-1]
    last_px = parmaxes[-1]

    iterator = enumerate(zip(layers, parmaxes, subaxes[1:], layer_info))
    previous_output = None

    # Loop all our layers and plots to visualize our data
    for index, (layer, pax, ax, info) in iterator:
        parameters = list(layer.parameters.keys())

        # Check our final outputs, if we have > 3 we will create a heatmap instead and ignore parmaxes to
        # make room for our response
        if ax == last_ax and target[0][0] > 9999:
            #_____________
            if isinstance(input, dict) or len(input.shape)>2:
                pax.clear()
                pax.imshow(input.T, origin='lower', aspect='auto', interpolation='none')
            else:
                if (index>0) & (layers[index-1].name=='wc'):
                    plot_strf(layer, wc_layer=layers[index-1], ax=pax)
                else:
                    plot_strf(layer, ax=pax)

        # Normal layer->layer plotting
        else:
            # Plotting FIR layers
            if layer.name=='fir':
                if (index>0) & (layers[index-1].name=='wc'):
                    plot_strf(layer, wc_layer=layers[index-1], ax=pax)
                else:
                    plot_strf(layer, ax=pax)
                pax.set_xticklabels([])

            # Plot i-o if nonlinearity
            elif 'nonlinearity' in str(type(layer)):
                plot_nl(layer, [previous_output.min(), previous_output.max()],
                        ax=pax, showlabels=False)
                compact_label(pax, f'nl')
                pax.set_xticklabels([])

            # Plotting coefficients of specific layers
            elif 'coefficients' in parameters:
                if len(layer.coefficients.shape)==2:
                    pax.imshow(layer.coefficients, aspect='auto', interpolation='none', origin='lower')
                    #pax.plot(layer.coefficients, lw=0.5)
                else:
                    pax.imshow(layer.coefficients[:,0,:], aspect='auto', interpolation='none', origin='lower')
                compact_label(pax, f'coefficients')
                pax.set_xticklabels([])

            # Plotting coefficients of specific layers
            elif 'gain' in parameters:
                pax.imshow(layer.parameters['gain'].values, aspect='auto', interpolation='none', origin='lower')
                compact_label(pax, f'gain')
                pax.set_xticklabels([])

            else:
                pax.set_visible(False)

        output = info['out']
        plot_args = layer.plot_kwargs
        plot_args['lw'] = '0.5'
        layer.plot_options['legend'] = False
        layer.plot(output[:T_max], ax=ax, **plot_args)

        if show_titles:
            compact_label(ax, f'({model.get_layer_index(layer.name)}) {layer.name}')

        set_plot_options(ax, layer.plot_options, time_kwargs=time_kwargs)
        previous_output = output

    # Plot input info as well
    if plot_input:
        ax = subaxes[0]
        if isinstance(input, dict):
            _input = input['input'][:T_max]
        else:
            _input = input[:T_max]
        if (len(_input.shape)>1) & (_input.shape[1]>1):
            ax.imshow(_input.T, origin='lower', aspect='auto', interpolation='none')
        else:
            ax.plot(input)

        if show_titles:
            compact_label(ax, 'input')
        ax.xaxis.set_visible(False)

    # Final x-axis of the final layer in each column is always visible
    # so that time is visually synchronized for all plots above.
    last_ax.xaxis.set_visible(True)

    # Add plot of target if given, on last axis of last subfig.
    if target is not None:
        if target_name is not None:
            target = input[target_name]
        else:
            target_name = 'Target'
        if not isinstance(target, list):
            second_last_ax = subaxes[-2]
            if target.shape[1]>2:
                last_ax.imshow(target[:T_max].T, aspect='auto', interpolation='none', origin='lower')
                second_last_ax.imshow(output[:T_max].T, aspect='auto', interpolation='none', origin='lower')
            else:
                last_ax.plot(target[:T_max])
                second_last_ax.plot(output[:T_max])

            compact_label(last_ax, 'Target')
            compact_label(second_last_ax, 'Output')
        else:
            for i, y in enumerate(target):
                last_ax.plot(y, label=f'{target_name} {i}', lw=0.5, zorder=-1)

        cc = correlation(output, target)
        last_px.plot(cc)
        compact_label(last_px, f'CC ({cc.mean():.3f})')
        figure.suptitle(f"{model.name}", fontsize=10)

    else:
        cc = model.meta.get('r_test',[0])[0]
        figure.suptitle(f"{model.name} cc={cc:.3f}", fontsize=10)
    #plt.tight_layout()

    return figure

def plot_nl(layer, range=None, channel=None, ax=None, fig=None, showlabels=True):

    if ax is not None:
        fig = ax.figure
    else:
        if fig is None:
            fig = plt.figure()
        ax = fig.subplots(1, 1)
    if len(range) == 2:
        x = np.linspace(range[0],range[1],100)
    else:
        x = range
    outcount=layer.shape[0]
    x=np.broadcast_to(x[:, np.newaxis], [x.shape[0], outcount])
    y = layer.evaluate(x)
    if channel is None:
        ax.plot(x, y, lw=0.5)
    else:
        ax.plot(x, y[:,channel])
    if showlabels:
        ax.set_xlabel('NL input')
        ax.set_ylabel('NL output')


def simple_strf(model, fir_idx=1, wc_idx=0, fs=None, title=None, ax=None, fig=None):
    """Wrapper for `plot_strf`, gets FIR and WeightChannels from Model.
    
    Parameters
    ----------
    model : nems.models.base.Model
        Model containing a FiniteImpulseResponse Layer and optionally a
        WeightChannels Layer.
    fir_idx : int; default=1.
        Integer index of the FiniteImpulseResponse Layer.
    wc_idx : int; defualt=0.
        Integer index of the WeightChannels Layer, if present.
    ax : Matplotlib axes; optional.
        Axis on which to generate the plot.
    fig : Matplotlib Figure; optional.
        Figure on which to generate the plot.

    Returns
    -------
    Matplotlib Figure
    
    """

    if ax is not None:
        pass
    elif 'nonlinearity' in str(type(model.layers[-1])):
        fig,axs = plt.subplots(1,2)
        ax = axs[0]
        plot_nl(model.layers[-1], [-1,3], ax=axs[1])
        axs[1].set_xlabel('NL input')
        axs[1].set_ylabel('NL output')
    else:
        fig, ax = plt.subplots()

    fir_layer, wc_layer = model.layers[fir_idx, wc_idx]
    plot_strf(fir_layer, wc_layer, ax=ax, fs=fs)
    ax.set_xlabel('Time lag')
    ax.set_ylabel('Freuqency channel')
    if title is not None:
        fig.suptitle(title)
    return fig


def plot_strf(fir_layer, wc_layer=None, fs=1, ax=None, fig=None):
    """Generate a heatmap representing a Spectrotemporal Receptive Field (STRF).
    
    Parameters
    ----------
    fir_layer : nems.layers.FiniteImpulseResponse
        `fir_layer.coefficients.T` will be used to specify the STRF.
    wc_layer : nems.layers.WeightChannels; optional.
        If given, `wc_layer.coefficients @ fir_layer.coefficients.T` will be
        used to specify the STRF (low-rank representation).
    ax : Matplotlib axes; optional.
        Axis on which to generate the plot.
    fig : Matplotlib Figure; optional.
        Figure on which to generate the plot.

    Returns
    -------
    Matplotlib Figure
    
    """
    if ax is not None:
        fig = ax.figure
    else:
        if fig is None:
            fig = plt.figure()
        ax = fig.subplots(1, 1)
    
    if wc_layer is None:
        fir = fir_layer.coefficients
        if len(fir.shape)>2:
            fir=fir[:,:,0]
        strf = fir.T
        vlines=0
        vlines_spacing=0
        filter_count=1
    else:
        wc = wc_layer.coefficients
        fir = fir_layer.coefficients
        if len(fir.shape)>2:
            filter_count = fir.shape[2]
            chan_count = wc.shape[0]
            gap = np.full([wc.shape[0], 1], np.nan)
            strfs = []
            for i in range(filter_count):
                s = wc[:, :, i] @ fir[:, :, i].T
                s = s/np.max(np.abs(s))
                if i>0:
                    strfs.extend([gap, s])
                else:
                    strfs.append(s)

            strf = np.concatenate(strfs, axis=1)

            #wc=wc[:,:,0]
            #fir=fir[:,:,0]
        else:
            strf = wc @ fir.T
            filter_count = 1
    lag_count = fir.shape[0]
    mm = np.nanmax(abs(strf))
    extent=[0, strf.shape[1]/fs, 0, strf.shape[0]]
    ax.imshow(strf, aspect='auto', interpolation='none', origin='lower',
              cmap='bwr', vmin=-mm, vmax=mm, extent=extent)
    vlines = filter_count-1
    vline_spacing = (lag_count+1)/fs
    for i in range(vlines):
        ax.axvline((i+1)*vline_spacing-0.5/fs, color='black', lw=0.5)
    #plt.tight_layout()
    
    return fig


def plot_layer(output, max_lines=5, fig=None, ax=None, **plot_kwargs):
    """Default Layer plot, displays all outputs on a single 2D line plot.
    
    Parameters
    ----------
    output : ndarray or list of ndarray
        Return value of `Layer.evaluate`.
    fig : matplotlib.pyplot.figure.Figure; optional.
        Matplotlib Figure to render the plot on. If not provided, a new figure
        will be generated with default options.
    plot_kwargs : dict
        Additional keyword arguments for `matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.figure.Figure

    See also
    --------
    .plot_model

    """
    if ax is not None:
        fig = ax.figure
    else:
        if fig is None:
            fig = plt.figure()
        ax = fig.subplots(1, 1)

    if(output.ndim > 0):
        all_outputs = [output[...,i] for i in range(output.shape[-1])]
        if all_outputs[0].ndim > 2:
            # TODO: Do something more useful here.
            print("Too many dimensions to plot")
        elif len(all_outputs)<=max_lines:
            for output in all_outputs:
                ax.plot(output, **plot_kwargs)
        else:
            all_outputs = np.reshape(output, (output.shape[0],-1))
            ax.imshow(all_outputs.T, origin='lower', interpolation='none', aspect='auto')
    else:
        print("One of the outputs is a single integer and could not be plotted")
    return fig


def input_heatmap(input, ax=None, extent=None, title='Input',
                  xunits='Time(bins)', ylabel='Channel', add_colorbar=True):
    """Plot heatmap of `input` with channels increasing from bottom to top.
    
    Parameters
    ----------
    input : np.ndarray, dict, or DataSet.
        Input to `model`. See `nems.Model.evaluate`.
    ax : Matplotlib.axes.Axes; optional.
        Axes on which to plot.
    title : str; optional.
    xlabel : str; default='Time (bins)'.
    ylabel : str; default='Firing Rate (Hz)'.

    Returns
    -------
    matplotlib.axes.Axes
    
    """

    if ax is None: ax = plt.gca()
    im = ax.imshow(input.T, aspect='auto', cmap='binary', interpolation='none',
                   origin='lower', extent=extent)
    if add_colorbar:
        plt.colorbar(im, ax=ax, label='Intensity')
    ax.set_xlabel(xunits)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return ax


def checkerboard(array):
    """Get checkerboard-spaced indices for a numpy array.

    Using this to keep background-shading of subfigures alternately-spaced.

    Parameters
    ----------
    array : ndarray
    
    Returns
    -------
    indices : boolean ndarray
        Same shape as array, with True and False in a checkerboard pattern.

    References
    ----------
    From user Eelco Hoogendoorn
    https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy

    Examples
    --------
    >>> x = np.ones(shape=(3,3))
    >>> checkerboard(x)
    array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]])
    >>> x[checkerboard(x)] = 2
    array([[1., 2., 1.],
           [2., 1., 2.],
           [1., 2., 1.]])

    """
    shape = array.shape
    indices = (np.indices(shape).sum(axis=0) % 2).astype(bool)
    return indices

def plot_model_list(model_list, input, target, plot_comparitive=True, plot_full=False, 
                    find_best=False, state=None, correlation=False, display_ratio=.5, **figure_kwargs):
    '''Main plot tool for ModelList()'''
    samples = len(model_list)
    fig_list = []

    pred_list = []
    for model in model_list:
        pred_list.append(model.predict(input, state=state))

    if find_best:
        best_fit = None
        samples += 1

    if plot_comparitive:
        fig, ax = plt.subplots(samples+1, 1, sharex='col', sharey='row')
        plot_data(input, label="Input", title='Test Stimulus', ax=ax[0], imshow=True)

        # Loop through our list, compare models, plots data, and save best model
        for fitidx, model in enumerate(model_list):
            if model.name == "UnnamedModel":
                model.name = f"Model_Fit-{fitidx}"
            if find_best and (best_fit is None or best_fit.results.final_error > model.results.final_error):
                best_fit = fitidx
            plot_data(pred_list[fitidx], label='predicted', title=model.name, target=target, ax=ax[fitidx+1], correlation=correlation, display_ratio=display_ratio, legend=False, **figure_kwargs)

        # Plotting some comparisons with our test data and the best models
        if find_best:
            plot_data(pred_list[best_fit], label='best_fit', title='Best vs Target', target=target, ax=ax[samples+1])
            ax[samples+1].legend()
        fig_list.append(fig)
        ax[1].legend(loc='upper right')

    if plot_full:
        for model in model_list:
            model_figure = model.plot(input, target=target)
            fig_list.append(model_figure)
    return fig_list

def plot_predictions(predictions, input=None, target=None, correlation=False, show_titles=True, display_ratio=.5, **figure_kwargs):
    '''
    Plots a single, or list of, prediction(s) to view and compare.

    Parameters
    ----------
    predictions: list, dict, np.ndarray
        A single or set of predictions to compare. If dictionary, keys
        are the titles of predictions
    input: np.ndarray
        Input used for predictions
    target: np.ndarray
        Target response we want from the prediction
    correlation: boolean
        If true, appends correlation coeff onto prediction title
    show_titles: boolean
        If true, shows the titles of each prediction
    display_ratio: float, 0->1
        Reduces amount of data displayed by trimming the end of plotted data.
        Default 50% is 0.5
    
    '''
    is_dict = False
    keys = None
    if isinstance(predictions, dict):
        keys = [key for key in predictions.keys()]
        is_dict = True
    elif not isinstance(predictions, list):
        predictions = [predictions]

    plots = len(predictions)
    if target is not None and target.shape[1] > 1:
        plots += 1

    fig, ax = plt.subplots(plots+1, 1, sharex='col', sharey='row')
    if input is not None:
        plot_data(input, label="Input", title="Input Data", ax=ax[0], imshow=True, display_ratio=display_ratio, ylabel='Frequency', legend=False, **figure_kwargs)

    for predidx, data in enumerate(predictions):
        if is_dict:
            data = predictions[data]
        title = f"Pred {predidx}"
        if keys:
            title = keys[predidx]
        if data.shape[1] > 3:
            plot_data(data, label=f"Pred {predidx}", title=title, ax=ax[predidx+1], 
                      correlation=correlation, show_titles=show_titles, imshow=True, display_ratio=display_ratio, ylabel='Frequency', legend=False, **figure_kwargs)
        else:
            plot_data(data, label=f"Pred {predidx}", title=title, ax=ax[predidx+1], target=target, 
                      correlation=correlation, show_titles=show_titles, display_ratio=display_ratio, ylabel='Frequency', legend=False, **figure_kwargs)

    if target is not None and target.shape[1] > 1:
        plot_data(target, label="Target", title="Target Data", ax=ax[-1], imshow=True, display_ratio=display_ratio)

    # Adds some labeling to bottom of figure and legend at top of predictions
    if not figure_kwargs:
        set_plot_options(ax[-1], {'legend': False, 'xlabel': 'Time(Bins)', 'show_x': True})
        set_plot_options(ax[1], {'legend': True})
        ax[1].legend(loc='upper center', bbox_to_anchor=(0.5,1.0))
    else:
        set_plot_options(ax[-1], figure_kwargs)
        set_plot_options(ax[1], {'legend': True})
        ax[1].legend(loc='upper center', bbox_to_anchor=(0.5,1.0))
        

    return fig

def plot_data(data, title='Data', label=None, target=None, ax=None, 
              correlation=False, imshow=False, ds_imshow=False, show_titles=True, display_ratio=.5,
               xunits="Bins", **figure_kwargs):
    """
    Plotting most basic/important information of given data. Returns plotted ax.
    Figure keyword arguments can be appended and will be used with set_plot_options().
    
    Parameters
    ----------
    title: string
        Title passed to text of ax
    label: string
        Label given to data plot
    target: np.ndarray
        Optional target data to plot with normal data
    ax: Axes object you wish to plot onto
    correlation: boolean
        If true, appends correlation coeff onto prediction title
    show_titles: boolean
        If true, shows the titles of each prediction
    display_ratio: float, 0->1
        Reduces amount of data displayed by trimming the end of plotted data.
        Default 50% is 0.5
    xlabel: string
        X-Axis label, usually representing time in form of Time-Bins, ms, seconds etc...
    
    """
    indicies, remainder = preprocessing.split.indices_by_fraction(data, display_ratio)
    reduced_data, _ = preprocessing.split.split_at_indices(data, indicies, remainder)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.legend(**_DEFAULT_PLOT_OPTIONS['legend_kwargs'])
    data = np.array(data)
    if not figure_kwargs:
        figure_kwargs = {'legend': True, 'show_x': True, 'ylabel': 'Frequency'}
    # Update plot options with user-given options
    figure_kwargs['xlabel'] = f'Time ({xunits})'
    if correlation:
        title += f" | Correlation: {metrics.correlation(data, target):.2f}"
    if imshow:
        ax.imshow(reduced_data.T, aspect='auto', interpolation='none')
    # DSTRF based imshow settings
    elif ds_imshow:
        absmax = np.max(np.abs(data))
        ax.imshow(reduced_data, aspect='auto', interpolation='none',
                cmap='bwr', vmin=-absmax, vmax=absmax, origin='lower')
    else:
        ax.plot(reduced_data, label=label)
    if target is not None:
        indicies, remainder = preprocessing.split.indices_by_fraction(target, display_ratio)
        reduced_target, _ = preprocessing.split.split_at_indices(target, indicies, remainder)
        ax.plot(reduced_target, label='Target', color='orange', lw=1, zorder=-1)
        
    set_plot_options(ax, figure_kwargs)
    ax.autoscale()
    if show_titles:
        x_pos = ax.get_xlim()[0]
        y_pos = ax.get_ylim()[1]
        ax.text(x_pos, y_pos, title, va='top', bbox=_TEXT_BBOX)
    return ax

# TODO: Iterate through dictionaries for larger inputs
def plot_dstrf(dstrf, title='DSTRF Models', xunits='Bins'):
    """Plotting DSTRF information from dstrf of a model"""
    absmax = np.max(np.abs(dstrf))
    dstrf_count = dstrf.shape[1]
    rows=int(np.ceil(dstrf_count/5))
    cols = int(np.ceil(dstrf_count/rows))
    f,ax=plt.subplots(rows,cols)
    f.supxlabel(f'Time({xunits})')
    f.supylabel('Features')
    f.suptitle(title)
    ax=ax.flatten()[:dstrf_count]
    for i,a in enumerate(ax):
        # flip along time axis so that x axis is timelag
        d = np.fliplr(dstrf[0,i,:,:])
        a.imshow(d, aspect='auto', interpolation='none',
                cmap='bwr', vmin=-absmax, vmax=absmax, origin='lower')
        a.text(a.get_xlim()[0], a.get_ylim()[1], f"DSTRF:{i}", va='top', bbox=_TEXT_BBOX)
    
    plt.tight_layout()
    return ax

def plot_dpcs(dpca, ax=None, title="DSTRF PC", xunits='Bins'):
    """
    Returns plotted graph of given DPCA. If no ax is given,
    a new figure will be created. Returns Axes

    Parameters
    ----------
    dpca: Dict
        Computed DPCA Dictionary
    ax: Axes
        Given matplotlib ax to plot graph onto
    title: 
        Suptitle for the plot figure
    xunits: String
        Unit value name to append to x_label

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.legend(**_DEFAULT_PLOT_OPTIONS['legend_kwargs'])
        fig.suptitle(title)
        fig.supxlabel(f'Time({xunits})')
    
    [plot_data(dpca['projection'][0, :, i], label=f'DPCA {i}', title='DSTRF PCs', display_ratio=1.0, ax=ax, show_titles=False) 
                for i in range(dpca['projection'].shape[2])]
    set_plot_options(ax, {'legend':True, 'margins':(0,.1), 'show_x':True,'legend_kwargs':{'loc': 'upper right', 'frameon':True}})
    ax.text(ax.get_xlim()[0], ax.get_ylim()[1], 'DPCAs', va='top', bbox=_TEXT_BBOX)
    return ax

def plot_dpcs_comparison(model, input, t_skip=20, title="DPCA Comparisons", xunits='Bins', **dstrf_kwargs):
    """
    Using PCA's and DSTRF's to plot PC adjustments over fit data
    and see PC of overall DSTRF's. Returns base GridSpec

    pc_len and t_steps allow for optimized and excitatory PCA's,
    modify these values to create faster dpca's or more relevant dpca's
    
    Parameters
    ----------
    model: Model
        Fitted Model object to be used for DSTRF and predictions
    input: np.ndarray
        Input data to be used for prediction and DSTRF
    t_skip: Int
        How many input values to repeatedly skip over when creating a full dstrf
    t_len: Int
        Length of the input you wish to fully process for full dstrf
    title: String
        Suptitle for the plot figure
    xunits: String
        Unit value name to append to x_label
    
    """
    # We need to define D before we can create any time indexes for our DSTRF
    D=25
    if dstrf_kwargs['D']:
        D = dstrf_kwargs['D']

    t_indexes = np.arange(D, len(input), t_skip)
    pred_model = model.predict(input)

    full_dstrf = model.dstrf(input, t_indexes=t_indexes, **dstrf_kwargs)
    short_dstrf = model.dstrf(input, **dstrf_kwargs)

    full_dpca = compute_dpcs(full_dstrf)
    short_dcpa = compute_dpcs(short_dstrf)

    # Base gridspec to subspec our graphs below
    fig = plt.figure()
    base_gs = gridspec.GridSpec(4, 1, figure=fig)
    fig.supxlabel(f'Time({xunits})')
    fig.suptitle(title)

    # Input, PCA's, and DPCA graphs added to our base gridspec
    gs = [base_gs[x].subgridspec(1,1) for x in range(base_gs.get_geometry()[0]-1)]
    gs.append(base_gs[3].subgridspec(1, short_dcpa['pcs'].shape[1]))
    gs_ax = [fig.add_subplot(y[0,0]) for y in gs]
    gs_ax.append(fig.add_subplot(gs[3][:, :]))

    input_plot = plot_data(input, ax=gs_ax[0], title="Input", imshow=True, figure_kwargs={'show_x': True,'legend':False, 'y_label': 'Hz'}, display_ratio=1.0)
    predict_plot = plot_data(pred_model, title='Prediction', ax=gs_ax[1], figure_kwargs={'legend':False, 'margins':(0, 999), 'show_x': True}, display_ratio=1.0)
    dpca_plot = plot_dpcs(full_dpca, ax=gs_ax[2])

    # DSTRF heatmaps
    for idx, ax in enumerate(gs[3].subplots()):
        data = np.fliplr(short_dcpa['pcs'][0,idx,:,:])
        plot_data(data, ax=ax, ds_imshow=True, title=f'DPCA: {idx}', figure_kwargs={'legend':False, 'show_x':False}, display_ratio=1.0)

    fig.tight_layout()
    return base_gs
