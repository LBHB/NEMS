from fractions import Fraction
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def ax_remove_box(axes=None):
    """Remove right and top lines from plot border on Matplotlib axes."""
    if axes is None:
        axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

def ax_bins_to_seconds(axes=None, sampling_rate=1.0, time_axis='x',
                       conversion_factor=1.0, decimals=2):
    """Change tick labels on time axis from bins to seconds.
    
    Parameters
    ----------
    axes : matplotlib.axes.Axes; optional.
        Axes to modify. If not provided, `plt.gca()` will be used.
    sampling_rate : float; default=1.0.
        Rate (in Hz) at which data on axes were collected.
    time_axis : str; default='x'.
        'x' or 'y', the axis on `axes` that represents time.
    conversion_factor : int or float; default=1.0.
        Multiply seconds by this number to get different units. Floating point
        values must correspond to rational numbers. Scientific notation may only
        be used for multiples of 10, and will be converted to an integer.
        Ex: `conversion_factor=1000` to get units of milliseconds.
            `conversion_factor=1/60` to get units of minutes.
    decimals : int; default=2.
        Number of decimal places to show on new tick labels.
    
    """
    if axes is None:
        axes = plt.gca()
    if (conversion_factor % 10) == 0:
        # re-cast scientific notation
        conversion_factor = int(conversion_factor)  

    # Get axes methods based on time_axis (x or y)
    tick_getter = getattr(axes, f'get_{time_axis}ticks')
    axis = getattr(axes, f'{time_axis}axis')
    set_tick_labels = getattr(axes, f'set_{time_axis}ticklabels')
    set_label = getattr(axes, f'set_{time_axis}label')

    # Fetch list of current tick locations (which have units of bins).
    # Set locations to be fixed, and update the labels at each location
    ticks = tick_getter().tolist()
    axes.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    labels = [f'{(tick/sampling_rate)*conversion_factor:.{decimals}f}'
               for tick in ticks]
    set_tick_labels(labels)

    if conversion_factor == 1:
        units = ''  # still seconds
    else:
        # Format 1/conversion_factor as a fraction to represent units.
        if conversion_factor < 1:
            denom = 1/conversion_factor
        else:
            denom = conversion_factor
        units = f'{Fraction(1/conversion_factor).limit_denominator(denom)} '
    set_label(f'Time ({units}s)')


def make_axes_plotter(plot_fn, sampling_rate=None, time_kwargs=None,
                      x_margin=False):
    """Create a standardized plotting utility for matplotlib Axes objects.
    
    Parameters
    ----------
    plot_fn : callable.
        Determines what will be plotted on the Axes. Must axcept 'ax' as a
        keyword argument.
    sampling_rate : float; optional.
        If specified, time axis will be labeled with units of seconds.
    time_kwargs : dict; optional.
        Additional keyword arguments for `ax_bins_to_seconds`.
    x_margin : bool; default=False.
        If False, remove whitespace margins from xlim
        (i.e. timeseries will fill the full time axis)

    Returns
    -------
    plotter : callable
    
    """

    if time_kwargs is None: time_kwargs = {}

    def _plotter(sampling_rate, time_kwargs, *args, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        plot_fn(*args, ax=ax, **kwargs)
        if sampling_rate is not None:
            # convert bins to seconds
            ax_bins_to_seconds(ax, sampling_rate, **time_kwargs)
        ax_remove_box(ax)    # remove right and top box borders
        if not x_margin:
            ax.margins(x=0)  # remove white space around xlim

        return ax

    plotter = partial(_plotter, sampling_rate, time_kwargs)
    plotter.__doc__ = plot_fn.__doc__

    return plotter
