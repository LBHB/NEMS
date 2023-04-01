import matplotlib.pyplot as plt
import numpy as np
from nems0.plots.utils import ax_remove_box


def plot_histogram(x, bins=None):
    '''
    Wrapper for matplotlib's plt.histogram for consistent formatting.

    Arguments:
    ----------
    x : array or sequence of arrays
        The data to be plotted.
    bins : integer, sequence, or 'auto'
        The number of bins for the data.
    See matplotlib.pyplot.hist for more details.

    Returns:
    --------
    None
    '''
    x = x[~np.isnan(x)]  # drop NaN values, plt.hist doesn't like them.
    plt.hist(x, bins=bins)


def pred_error_hist(resp, pred, ax=None, channel=0, bins=None,
                    bin_mult=5.0, xlabel='|Resp - Pred|', ylabel='Count',
                    title='Prediction Error'):
    '''
    Plots a histogram of the error between response and prediction.

    Arguments:
    ----------
    resp : signal
        response signal from a loaded recording object
    pred : signal
        prediction signal from a loaded recording object that
        has been used to fit a modelspec.
    ax : matplotlib ax object
        Will be used as the current plotting axes if provided.
    channel : int
        The channel for each of the signals that should be used.
        This should generally be 0 since responses normally only
        have one signal.
    bins : int, sequence, or 'auto'
        Number of bins for the data. See matplotlib.pyplot.his for details.
        If not provided, bins will gussed based on the length of resp and pred.
    bin_mult : float
        The value returned by calc_bins is multipled by this factor,
        to allow finer adjustment of bin count.
    xlabel, ylabel, title : str
        String identifiers that will be used to set title and axis labels
        for the plot.

    Returns:
    --------
    None

    '''
    # TODO: Delete this and move to quickplot?
    #       Would want to fully switch over from summary_plot first
    if ax:
        plt.sca(ax)

    resp_data = resp.as_continuous()[channel]
    pred_data = pred.as_continuous()[channel]
    err_data = np.absolute(resp_data - pred_data)

    if not bins:
        # TODO: Might need to adjust this calculation to find
        #       the optimum.
        length = err_data.shape[0]
        bins = int(bin_mult*(np.ceil(np.sqrt(length))))

    plot_histogram(err_data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def isi_histogram(resp, fs=1000, epoch="REFERENCE", ax=None, channel=0,
                  bins=None, xlabel='ISI (sec)', ylabel='Count',
                  title='ISI histogram'):
    """
    assume data is a N x T array of spike events (0 or 1)
      or N x 1 x T array
    """
    if type(resp) is np.ndarray:
        # ignore epoch
        data = resp
    else:
        data = resp.extract_epoch(epoch)
        fs = resp.fs

    if data.ndim == 3:
        data = data[:,channel,:]

    dd = np.array([])
    for i in range(data.shape[0]):
        spike_times,=np.where(data[i,:] > 0)
        dual_spike_times, = np.where(data[i,:]>1)
        dual_spike_count = np.int(np.sum(data[i,dual_spike_times]))
        dd = np.concatenate((dd,np.diff(spike_times),np.zeros(dual_spike_count)))

    dd /= fs

    if bins is None:
        # TODO: Might need to adjust this calculation to find
        #       the optimum.
        bins = np.arange(0, 0.2, 1/fs) - 1/fs/2

    if ax:
        plt.sca(ax)

    plt.hist(dd, bins=bins)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

