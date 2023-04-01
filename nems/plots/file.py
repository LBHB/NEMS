import io
import os
import logging
import matplotlib.image as mpimg
import nems0.modelspec as ms


def save_figure(fig, filepath=None, modelspecs=None, save_dir='/tmp',
                format='png'):
    '''
    Saves the given matplotlib figure object, using either a specific
    filepath or a path determined from a list of modelspecs and a
    directory.

    Arguments
    --------
    fig : matplotlib figure
        Any matplotlib figure object

    filepath : str or fileobj
        Specifies the location where the figure should be saved.
        If not provided, modelspecs must be given instead. If a string,
        it should be a complete, absolute path to the save location.

    modelspecs : list
        A list of modelspecs that can be used to determine a filepath
        based on nems0.modelspec.get_modelspec_longname() and save_dir.

    save_dir : str
        Specifies the base directory in which to save the figure,
        if modelspecs are used.

    format : str
        Specifies the file format that should be used to save the figure.
        Compatible choices depend on which matplotlib backend is running,
        but 'png', 'pdf', and 'svg' are almost always supported.

    Returns
    -------
    fname : str
        The filepath that was ultimately used for storage.

    '''
    if filepath and not modelspecs:
        fname = filepath
    elif modelspecs and not filepath:
        fname = _get_figure_filepath(save_dir, modelspecs, format)
    else:
        raise ValueError("save_figure() must be provided either"
                         "a filepath or a list of modelspecs.")
    logging.info("Saving figure as: {}".format(fname))
    fig.savefig(fname)
    return fname


def load_figure_img(filepath=None, modelspecs=None, load_dir=None,
                    format='png'):
    '''
    Loads a saved figure image as a numpy array that can be displayed
    inside python using matplotlib.pyplot.imshow().

    Arguments
    --------
    filepath : str or fileobj
        Specifies the location where the image was stored.
        If not provided, modelspecs must be given instead. If a string,
        it should be a complete, absolute path to the save location.

    modelspecs : list
        A list of modelspecs that can be used to determine a filepath
        based on nems0.modelspec.get_modelspec_longname() and load_dir.

    load_dir : str
        Specifies the base directory in which to save the figure,
        if modelspecs are used.

    format : str
        Specifies the file format that was used to save the figure.

    Returns
    -------
    img : numpy ndarray
        Array containing the image data.

    '''
    if filepath and not modelspecs:
        fname = filepath
    elif modelspecs and not filepath:
        fname = _get_figure_filepath(load_dir, modelspecs, format)
    else:
        raise ValueError("load_figure_img() must be provided either"
                         "a filepath or a list of modelspecs.")
    logging.info("Loading figure image from: {}".format(fname))
    img = mpimg.imread(fname)
    return img


def load_figure_bytes(filepath=None, modelspecs=None, load_dir=None,
                      format='png'):
    '''
    Loads a saved figure image as a bytes object that can be used
    by the web UI or other functions.

    Arguments
    --------
    filepath : str or fileobj
        Specifies the location where the image was stored.
        If not provided, modelspecs must be given instead. If a string,
        it should be a complete, absolute path to the save location.

    modelspecs : list
        A list of modelspecs that can be used to determine a filepath
        based on nems0.modelspec.get_modelspec_longname() and load_dir.

    load_dir : str
        Specifies the base directory in which to save the figure,
        if modelspecs are used.

    format : str
        Specifies the file format that was used to save the figure.

    Returns
    -------
    img : bytes object
        Contains the raw data for the loaded image.

    '''
    if filepath and not modelspecs:
        fname = filepath
    elif modelspecs and not filepath:
        fname = _get_figure_filepath(load_dir, modelspecs, format)
    else:
        raise ValueError("load_figure_img() must be provided either"
                         "a filepath or a list of modelspecs.")
    logging.info("Loading figure image from: {}".format(fname))
    with open(fname, 'r+b') as f:
        img = f.read()
    return img


def _get_figure_filepath(directory, modelspecs, format):
    '''
    Returns a filepath based on a directory, a list of modelspecs,
    and a file format.
    '''
    # TODO: Probably need a smarter way to do figure names since figures
    #       can come from an arbitrary number of modelspecs. Could just
    #       concatenate names for each modelspec? But that would mean some
    #       really long filenames.
    #       For now just uses the long name of the first modelspec until
    #       a better solution is decided on.
    mspec = modelspecs[0]
    mname = ms.get_modelspec_longname(mspec)
    fname = os.path.join(directory, mname) + "." + format
    return fname


def fig2BytesIO(figure):
    '''
    Returns a figure as PNG stored in a BytesIO object.
    '''
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    return buf
