import collections.abc
import math
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .tools import ax_remove_box, ax_bins_to_seconds
from nems import metrics
from nems import preprocessing

_DEFAULT_PLOT_OPTIONS = {
    'skip_plot_options': False,
    'show_x': False, 'xlabel': None, 'xmax_scale': 1, 'xmin_scale': 1,
    'show_y': True, 'ylabel': None, 'ymax_scale': 1, 'ymin_scale': 1,
    'show_seconds': True,
    'legend': False,
    # Right of axis by default, aligned to top
    'legend_kwargs': {
        'frameon': False, 'bbox_to_anchor': (1, 1), 'loc': 'upper left'
        },
    }

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


def plot_model(model, input, target=None, target_name=None, n=None,
               select_layers=None, n_columns=1, show_titles=True,
               figure_kwargs=None, sampling_rate=None, time_axis='x',
               conversion_factor=1, decimals=2, plot_input=True,
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

    # Setting up our layout for plotting layers and parameters
    spec = figure.add_gridspec(n_rows, 3)
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
                    plot_strf(layer, wc_layer=layers[i-1], ax=pax)
                else:
                    plot_strf(layer, ax=pax)

        # Normal layer->layer plotting
        else:
            # Plotting FIR layers
            if layer.name=='fir':
                if (index>0) & (layers[index-1].name=='wc'):
                    plot_strf(layer, wc_layer=layers[i-1], ax=pax)
                else:
                    plot_strf(layer, ax=pax)

            # Plotting coefficients of specific layers
            elif 'coefficients' in parameters:
                if len(layer.coefficients.shape)==2:
                    pax.plot(layer.coefficients, lw=0.5)
                else:
                    pax.imshow(layer.coefficients[:,0,:],
                            aspect='auto', interpolation='none', origin='lower')
                x_pos = pax.get_xlim()[0]
                y_pos = pax.get_ylim()[1]
                title = f'coefficients'
                pax.text(x_pos, y_pos, title, va='top')

            # Plot if non-linear
            elif 'nonlinearity' in str(type(layer)):
                plot_nl(layer, [previous_output.min(), previous_output.max()], ax=pax)
            else:
                pax.set_visible(False)

        output = info['out']
        plot_args = layer.plot_kwargs
        plot_args['lw'] = '0.5'
        layer.plot_options['legend'] = False
        layer.plot(output, ax=ax, **plot_args)

        if show_titles:
            x_pos = ax.get_xlim()[0]
            y_pos = ax.get_ylim()[1]
            name = layer.name
            title = f'({model.get_layer_index(name)}) {name}'
            ax.text(x_pos, y_pos, title, va='top')

        set_plot_options(ax, layer.plot_options, time_kwargs=time_kwargs)
        previous_output = output

    # Plot input info as well
    if plot_input:
        ax = subaxes[0]
        if isinstance(input, dict) or len(input.shape)>2:
            pass
        elif (len(input.shape)>1) & (input.shape[1]>1):
            ax.imshow(input.T, origin='lower', aspect='auto', interpolation='none')
        else:
            ax.plot(input)
        if show_titles:
            title = 'input'
            x_pos = ax.get_xlim()[0]
            y_pos = ax.get_ylim()[1]
            ax.text(x_pos, y_pos, title, va='top', bbox=dict(boxstyle='round, pad=.1, rounding_size=.1', alpha=.7, facecolor='white'))
        ax.set_xlim(subaxes[-1].get_xlim())
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
            target = [target]
        if len(target)>3:
            target=np.concatenate(target, axis=1)
            last_ax.imshow(target, aspect='auto', interpolation='none', origin='lower')
        else:
            # If our given targets is greater than 3, we want to replace graphs with heatmaps
            if len(target[0][0]) > 3:
                second_last_ax = subaxes[-2]
                last_ax.clear()
                second_last_ax.clear()

                last_ax.imshow(target[0].T, aspect='auto', interpolation='none', origin='lower')
                second_last_ax.imshow(output.T, aspect='auto', interpolation='none', origin='lower')

                last_ax.set_xlim(subaxes[0].get_xlim())
                x_pos = last_ax.get_xlim()[0]
                y_pos = last_ax.get_ylim()[1]
                last_ax.text(x_pos, y_pos, 'Target', va='top', bbox=dict(boxstyle='round, pad=.1, rounding_size=.1', alpha=.7, facecolor='white'))

                second_last_ax.set_xlim(subaxes[0].get_xlim())
                x_pos = second_last_ax.get_xlim()[0]
                y_pos = second_last_ax.get_ylim()[1]
                second_last_ax.text(x_pos, y_pos, 'Output', va='top', bbox=dict(boxstyle='round, pad=.1, rounding_size=.1', alpha=.7, facecolor='white'))


                return
            else:
                for i, y in enumerate(target):
                    last_ax.plot(y, label=f'{target_name} {i}', lw=0.5, zorder=-1)
            #last_ax.legend(**_DEFAULT_PLOT_OPTIONS['legend_kwargs'])
        last_ax.autoscale()
        cc = np.corrcoef(target[0][:,0], output[:,0])[0,1]
    else:
        cc = model.meta.get('r_test',[0])[0]

    figure.suptitle(f"{model.name} cc={cc:.3f}", fontsize=10)
    plt.tight_layout()

    return figure

def plot_nl(layer, range=None, channel=None, ax=None, fig=None):

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
    if outcount>1:
        x=np.broadcast_to(x[:,np.newaxis],[x.shape[0],outcount])
    y = layer.evaluate(x)
    if channel is None:
        ax.plot(x, y, lw=0.5)
    else:
        ax.plot(x, y[:,channel])
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


def plot_strf(fir_layer, wc_layer=None, fs=None, ax=None, fig=None):
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
    else:
        wc = wc_layer.coefficients
        fir = fir_layer.coefficients
        if len(fir.shape)>2:
            filter_count = fir.shape[2]
            chan_count = wc.shape[0]
            lag_count = fir.shape[0]
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

    mm = np.nanmax(abs(strf))
    if fs is not None:
        extent=[0, strf.shape[0]/fs, 0, strf.shape[1]]
    else:
        extent=[0, strf.shape[0], 0, strf.shape[1]]

    ax.imshow(strf, aspect='auto', interpolation='none', origin='lower',
              cmap='bwr', vmin=-mm, vmax=mm, extent=extent)
    plt.tight_layout()
    
    return fig


def plot_layer(output, fig=None, ax=None, **plot_kwargs):
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
        else:
            for output in all_outputs:
                ax.plot(output, **plot_kwargs)
    else:
        print("One of the outputs is a single integer and could not be plotted")
    return fig


def input_heatmap(input, ax=None, extent=None, title='Input',
                  xlabel='Time (bins)', ylabel='Channel', add_colorbar=True):
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
    ax.set_xlabel(xlabel)
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
                    find_best=False, state=None, correlation=True, display_reduction=.5):
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
        fig, ax = plt.subplots(samples+2, 1, sharex='col')
        plot_data(input.T, label="Input", title='Test Stimulus', ax=ax[0], imshow=True)
        plot_data(target, label="actual response", title='Test Response', ax=ax[1])

        # Loop through our list, compare models, plots data, and save best model
        for fitidx, model in enumerate(model_list):
            if model.name is "UnnamedModel":
                model.name = f"Model_Fit-{fitidx}"
            if find_best and (best_fit is None or best_fit.results.final_error > model.results.final_error):
                best_fit = fitidx
            plot_data(pred_list[fitidx], label='predicted', title=model.name, target=target, ax=ax[fitidx+2], correlation=correlation, display_reduction=.5)

        # Plotting some comparisons with our test data and the best models
        if find_best:
            plot_data(pred_list[best_fit], label='best_fit', title='Best vs Target', target=target, ax=ax[samples+2])
            ax[samples+2].legend()
        fig_list.append(fig)

    if plot_full:
        for model in model_list:
            model_figure = model.plot(input, target=target)
            fig_list.append(model_figure)
    return fig_list

def plot_predictions(predictions, input=None, target=None, correlation=False, show_titles=True, display_reduction=.5):
    '''
    Plots a single, or list of, predictions to view and compare.

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
    display_reduction: float, 0->1
        Reduces amount of data displayed by trimming the end of plotted data
    
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

    fig, ax = plt.subplots(plots+1, 1, sharex='col')
    if input is not None:
        plot_data(input, label="Input", title="Input Data", ax=ax[0], imshow=True, display_reduction=display_reduction)

    for predidx, data in enumerate(predictions):
        if is_dict:
            data = predictions[data]
        title = f"Pred {predidx}"
        if keys:
            title = keys[predidx]
        if data.shape[1] > 3:
            plot_data(data, label=f"Pred {predidx}", title=title, ax=ax[predidx+1], 
                      correlation=correlation, show_titles=show_titles, imshow=True, display_reduction=display_reduction)
        else:
            plot_data(data, label=f"Pred {predidx}", title=title, ax=ax[predidx+1], target=target, 
                      correlation=correlation, show_titles=show_titles, display_reduction=display_reduction)

    if target is not None and target.shape[1] > 1:
        plot_data(target, label="Target", title="Target Data", ax=ax[-1], imshow=True, display_reduction=display_reduction)
    return fig

def plot_data(data, title, label=None, target=None, ax=None, 
              correlation=False, imshow=False, show_titles=True, display_reduction=.5):
    """
    Plotting most basic/important information of given data. Returns plotted ax
    
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
    display_reduction: float, 0->1
        Reduces amount of data displayed by trimming the end of plotted data  
    
    """
    indicies, remainder = preprocessing.split.indices_by_fraction(data, display_reduction)
    reduced_data, _ = preprocessing.split.split_at_indices(data, indicies, remainder)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if correlation:
        title += f" | Correlation: {metrics.correlation(data, target):.2f}"
    if imshow:
        ax.imshow(reduced_data.T, aspect='auto', interpolation='none')
    else:
        ax.plot(reduced_data, label=label)
    if target is not None:
        indicies, remainder = preprocessing.split.indices_by_fraction(target, display_reduction)
        reduced_target, _ = preprocessing.split.split_at_indices(target, indicies, remainder)
        ax.plot(reduced_target, label='Target', color='orange', lw=1, zorder=-1)
    if show_titles:
        x_pos = ax.get_xlim()[0]
        y_pos = ax.get_ylim()[1]
        ax.text(x_pos, y_pos, title, va='top', bbox=dict(boxstyle='round, pad=.1, rounding_size=.1', alpha=.7, facecolor='white'))
    return ax

# TODO: Iterate through dictionaries for larger inputs
def plot_dstrf(dstrf):
    """Plotting DSTRF information from dstrf of a model"""
    dstrf = dstrf['input']
    absmax = np.max(np.abs(dstrf))
    dstrf_count = dstrf.shape[1]
    rows=int(np.ceil(dstrf_count/5))
    cols = int(np.ceil(dstrf_count/rows))
    f,ax=plt.subplots(rows,cols)
    ax=ax.flatten()[:dstrf_count]
    for i,a in enumerate(ax):
        # flip along time axis so that x axis is timelag
        d = np.fliplr(dstrf[0,i,:,:])
        a.imshow(d, aspect='auto', interpolation='none',
                cmap='bwr', vmin=-absmax, vmax=absmax, origin='lower')
        a.text(a.get_xlim()[0], a.get_ylim()[1], f"D={i}", va='top', bbox=dict(boxstyle='round, pad=.1, rounding_size=.1', alpha=.7, facecolor='white'))
        
    plt.tight_layout()
    return ax

def plot_shift_dstrf(dstrf):
    """Plots graph that compares mean value of dstrf step with respect to the previous one"""
    dstrf = dstrf['input']
    absmax = np.max(np.abs(dstrf))
    dstrf_count = dstrf.shape[1]
    rows=int(np.ceil(dstrf_count/5))
    cols = int(np.ceil(dstrf_count/rows))
    f,ax=plt.subplots(rows,cols, sharex='col', sharey='row')
    ax=ax.flatten()[:dstrf_count]
    x = np.arange(dstrf.shape[2])+.5
    for index, a in enumerate(ax):
        dstrf_set = dstrf[0,index,:,:]
        mean_list = np.array([np.mean(j) for j in dstrf_set])
        if index > 0:
            shift = f"{np.mean(mean_list-prev_mean)*10000:.2f}"
            a.stem(x, np.subtract(mean_list, prev_mean), markerfmt='none', basefmt='none')
            a.plot(prev_mean-prev_mean, color='red', lw=.5)
        else:
            shift = 'N/A'
            a.plot(mean_list)
            prev_mean = mean_list
        a.text(a.get_xlim()[0], a.get_ylim()[1]*1.25, f"D:{index} shift: {shift}", 
                va='top', bbox=dict(boxstyle='round, pad=.1, rounding_size=.1', alpha=.7, facecolor='white'))
        prev_mean = mean_list
    plt.tight_layout()
    return ax

def plot_mean_dstrf(dstrf):
    """Plots mean value of dimensions in each step"""
    dstrf = dstrf['input']
    dstrf_count = dstrf.shape[1]
    rows=int(np.ceil(dstrf_count/5))
    cols = int(np.ceil(dstrf_count/rows))
    f,ax=plt.subplots(rows,cols, sharex='col', sharey='row')
    ax=ax.flatten()[:dstrf_count]
    for index, a in enumerate(ax):
        mean_list = [np.mean(j) for j in dstrf[0, index, :, :]]
        a.plot(mean_list)
        a.text(a.get_xlim()[0], a.get_ylim()[1], f"D={index}", va='top', bbox=dict(boxstyle='round, pad=.1, rounding_size=.1', alpha=.7, facecolor='white'))
        if index > 0:
            a.plot(prev_mean, color='red', lw=.5)
        prev_mean = mean_list
    plt.tight_layout()
    return ax

def plot_absmax_dstrf(dstrf):
    """Plots the absolute max of each dimension for every step"""
    dstrf = dstrf['input']
    dstrf_count = dstrf.shape[1]
    color = matplotlib.cm.get_cmap("Reds", dstrf_count+4)
    f,ax=plt.subplots(1,1)
    for index in range(dstrf_count):
        absmax_list = [np.max(np.abs(j))+index*.0040 for j in dstrf[0, index, :, :]]
        ax.plot(absmax_list, color=color(dstrf_count+4 - index), label=f'D {index}')
    plt.legend(loc="upper left")
    plt.tight_layout()
    return ax

