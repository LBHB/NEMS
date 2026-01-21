import numpy as np
from scipy.ndimage import convolve1d

# TODO - incorporate other filters?
# From supplement doc: "apply a range of standard filters to data (smoothing, high-pass, etc)"

def smooth(x,window_len=11,window='hanning', axis=-1):
    """Smooth data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal
    along one or more axes, as specified.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
        axis: axis to smooth. if tuple, smooth along all axes listed
    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if type(axis) is tuple:
        for a in axis:
            if x.shape[a] < window_len:
                raise ValueError(f"Input shape[{a}] needs to be bigger than window size.")
    else:
        if x.shape[axis] < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    if window_len & 0x1:
        w1 = int((window_len+1)/2)
        w2 = int((window_len+1)/2)
    else:
        w1 = int(window_len/2)+1
        w2 = int(window_len/2)

    #print(len(s))

    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    w /= w.sum()
    if type(axis) is tuple:
        y = x.copy()
        for a in axis:
            y = convolve1d(y, w, axis=a, mode='reflect')
    else:
        y = convolve1d(x, w, axis=axis, mode='reflect')

    return y