"""utils library

This module contains random utility functions called by a number of different NEMS libraries

"""
import time
import hashlib
import json
import os
from collections.abc import Sequence
import logging
import importlib
import re
from configparser import ConfigParser

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

from . import get_setting
#import nems

log = logging.getLogger(__name__)


def iso8601_datestring():
    '''
    Returns a string containing the present date as a string.
    '''
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

"""
Utils imported from uri for more general purpose JSON encoding/decoding
"""
class NumpyEncoder(json.JSONEncoder):
    '''
    For serializing Numpy arrays safely as JSONs. Modified from:
    https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
    saving as byte64 doesn't work, but using lists instead seems ok.
    '''

    def default(self, obj):
        """
        If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data. data is encoded as a list,
        which makes it text-readable.
        """
        from nems0.distributions.distribution import Distribution
        from nems0.modules import NemsModule

        if issubclass(type(obj), Distribution):
            return obj.tolist()

        if issubclass(type(obj), NemsModule):
            return obj.data_dict

        if isinstance(obj, np.ndarray):
            # currently disabling b64 encoding because it doesn't work and
            # it makes JSON files unreadable. However, it may be worth
            # implementing in the future for different parts of the
            # modelspec
            use_b64_encoding = False
            if use_b64_encoding:
                if obj.flags['C_CONTIGUOUS']:
                    obj_data = obj.data
                else:
                    cont_obj = np.ascontiguousarray(obj)
                    assert(cont_obj.flags['C_CONTIGUOUS'])
                    obj_data = cont_obj.data
                data_encoded = base64.b64encode(obj_data)
            else:
                data_encoded = obj.tolist()

            return dict(__ndarray__=data_encoded,
                        dtype=str(obj.dtype),
                        shape=obj.shape,
                        encoding='list')

        to_json_exists = getattr(obj, "to_json", None)
        if callable(to_json_exists):
            return obj.to_json()

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray with proper shape and dtype,
    or an encoded KeywordRegistry.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """

    if isinstance(dct, dict) and '__ndarray__' in dct:
        # data = base64.b64decode(dct['__ndarray__'])
        data = dct['__ndarray__']
        return np.asarray(data, dct['dtype']).reshape(dct['shape'])

    special_keys = ['level', 'coefficients', 'amplitude', 'kappa',
                    'base', 'shift', 'mean', 'sd', 'u', 'tau', 'offset']

    if isinstance(dct, dict) and any(k in special_keys for k in dct):
        # print("json_numpy_obj_hook: {0} type {1}".format(dct,type(dct)))
        for k in dct:
            if type(dct[k]) is list:
                dct[k] = np.asarray(dct[k])

    if '_KWR_ARGS' in dct:
        from nems0.registry import KeywordRegistry
        return KeywordRegistry.from_json(dct)

    return dct


def adjust_uri_prefix(uri, use_nems_defaults=True):
    """
    if get_setting('USE_NEMS_BAPHY_API') is True: translate file system URI to http --or--
    if get_setting('USE_NEMS_BAPHY_API') is False: translate http URI to file system
    
    Warning! May be too hacky, and unclear where this should be evaluated! Currently run in uri.load_uri, which may be too low-level,
    as there may be situations where you want to hard-code a URI that doesn't match the expectations of the current configuration.
    """
    use_API = get_setting('USE_NEMS_BAPHY_API')
    prefix_is_http = uri.startswith("http")
    rec_prefix = get_setting('NEMS_RECORDINGS_DIR')
    res_prefix = get_setting('NEMS_RESULTS_DIR')
    rec_match = uri.find("/recordings")
    res_match = uri.find("/results")

    if use_API and (not prefix_is_http):
        api_prefix = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT'))
        if uri.startswith(rec_prefix):
            new_uri = uri.replace(rec_prefix, api_prefix + "/recordings" )
            log.info(f"Adjusting URI from {uri} to {new_uri}")
        elif uri.startswith(res_prefix):
            new_uri = uri.replace(res_prefix, api_prefix + "/results" )
            log.info(f"Adjusting URI from {uri} to {new_uri}")
        else:
            new_uri = uri
    elif (not use_API) and prefix_is_http:
        if rec_match:
            new_uri = rec_prefix + uri[(rec_match+11):]
            log.info(f"Adjusting URI from {uri} to {new_uri}")
        elif res_match:
            new_uri = res_prefix + uri[(res_match+8):]
            log.info(f"Adjusting URI from {uri} to {new_uri}")
        else:
            new_uri = uri
    else:
        new_uri = uri

    return new_uri


def recording_filename_hash(name, meta, uri_path='', uncompressed=False):
    """
    name : string
    meta : dictionary (string keys only?)
    uri_path : base path
    uncompressed : boolean
        if True, return .tgz file
        if False, return path to folder

    returns
       guessed_filename : string
        <uri_path>/<batch>/<name>_<meta hash>.tgz

    hashing function to generate recording filenames
    JSON encode meta, then append hash to name
    """
    meta_hash = hashlib.sha1(json.dumps(meta, sort_keys=True, cls=NumpyEncoder).encode('utf-8')).hexdigest()

    if uncompressed:
        guessed_filename = name + "_" + meta_hash + os.sep
    else:
        guessed_filename = name + "_" + meta_hash + '.tgz'

    batch = meta.get('batch', None)
    if batch is not None:
        guessed_filename = os.path.join(str(batch), guessed_filename)

    if uri_path is not None and uri_path != '':
        guessed_filename = os.path.join(uri_path, guessed_filename)

    return guessed_filename


def one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


def split_to_api_and_fn(mystring):
    '''
    Returns (api, fn_name) given a string that would be used to import
    a function from a package.
    '''
    matches = mystring.split(sep='.')
    api = '.'.join(matches[:-1])
    fn_name = matches[-1]
    return api, fn_name


def shrinkage(mH, eH, sigrat=1, thresh=0):
    """
    apply shrinkage transformation to estimated mean value (mH),
    based on the relative size of the standard error (eH)
    """

    smd = np.abs(mH) / (eH + np.finfo(float).eps * (eH == 0)) / sigrat

    if thresh:
        hf = mH * (smd > 1)
    else:
        smd = 1 - np.power(smd, -2)
        smd = smd * (smd > 0)
        # smd[np.isnan(smd)]=0
        hf = mH * smd

    return hf


def progress_fun():
    """
    This function can be redirected to a function that tracks progress
    externally, eg, in a queueing system
    """
    pass


def find_module(query: object, modelspec: object, find_all_matches: object = False, key: object = 'fn') -> object:
    """
    name : string
    modelspec : NEMS modelspec (list of dictionaries)

    find_all_matches : boolean
      if True:
          returns first index where modelspec[]['fn'] contains name
          returns None if no match
      if False
          returns list of index(es) where modelspec[]['fn'] contains name
          returns empty list [] if no match
    """
    if find_all_matches:
        target_i = []
    else:
        target_i = None
    for i, m in enumerate(modelspec):
        if query in m[key]:
            if find_all_matches:
                target_i.append(i)
            else:
                target_i = i
                break

    if not target_i:
        log.debug("target_module: %s not found in modelspec.", query)
    else:
        log.debug("target_module: %s found at modelspec[%d]",
                  query, target_i)

    return target_i


def escaped_split(string: str, delimiter: str):
    """
    Splits string along delimiter, ignoring escaped delimiters. Useful
    for some arguments in keyword strings that need to use delimiter
    for other reason.

    :param string: string to be split
    :param delimiter: character to split on
    :return: list of strings
    """
    # strip out leading and trailing delimiters to avoid empty strings in split
    string = string.strip(delimiter)
    # use a negative lookbehind to ignore matches preceded by '\'
    split = re.split(fr'(?<!\\)\{delimiter}', string)

    return split


def escaped_join(split_list, delimiter: str):
    """
    Joins a list of strings along a delimiter.

    :param split_list: list of strings to join
    :param delimiter: delimiter to join on
    :return: joined string
    """
    return delimiter.join(split_list)


def keyword_extract_options(kw):
    if kw == 'basic' or kw == 'iter':
        # empty options (i.e. just use defaults)
        options = []
    else:
        chunks = escaped_split(kw, '.')
        options = chunks[1:]
    return options



def get_channel_number(sig, channel=None):
    """
    find number of channel in signal sig that matches channel name or number
      specified in channel. default return 0
    """
    if channel is None:
        chanidx = 0
    elif sig.chans is None:
        chanidx = 0
    elif type(channel) is str:
        try:
            chanidx = sig.chans.index(channel)
        except ValueError:
            raise ValueError('channel name not in list')

    elif type(channel) is int:
        chanidx = channel
    else:
        raise ValueError('channel not integer or string')

    if chanidx >= sig.shape[0]:
        raise ValueError('channel number not valid')

    return chanidx


def smooth(x,window_len=11,window='hanning', axis=-1):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

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

    if x.shape[axis] < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
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
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y = convolve1d(x, w/w.sum(), axis=axis, mode='reflect')

    return y


def ax_remove_box(ax=None):
    """
    remove right and top lines from plot border
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def depth(s):
    '''
    Count depth of nesting of a sequence. E.g. [[[1]]] has a depth of 3.
    '''
    i = 0
    x = s
    while isinstance(x, Sequence):
        i += 1
        if x:
            x = x[0]
        else:
            x = None
    return i

# pruffix commands (originally developed for nems_web, but useful for parsing xforms modelnames)
def find_prefix(s_list):
    """Given a list of strings, returns the common prefix to nearest _."""
    prefix = ''
    if (not s_list) or (len(s_list) == 1):
        return prefix
    i = 0
    test = True
    while test:
        # log.debug('while loop, i=%s'%i)
        # log.debug('before for loop, prefix = %s'%prefix)
        for j in range(len(s_list) - 1):
            # look at ith item of each string in list, in order
            if i < len(s_list[j]):
                a = s_list[j][i]
            else:
                a = ''
            if i<len(s_list[j+1]):
                b = s_list[j + 1][i]
            else:
                b = ''
            # log.debug('for loop, a = %s and b = %s'%(a, b))
            if a != b:
                test = False
                break
            if j == len(s_list) - 2:
                prefix += b
        i += 1

    #while prefix and (prefix[-1] != '_'):
    #    prefix = prefix[:-1]

    return prefix


def find_suffix(s_list):
    """Given a list of strings, returns the common suffix to nearest _."""
    suffix = ''
    if (not s_list) or (len(s_list) == 1):
        return suffix
    i = 1
    test = True
    while test:
        # log.debug('while loop, i=%s'%i)
        # log.debug('before for loop, suffix = %s'%suffix)
        for j in range(len(s_list) - 1):
            # look at ith item of each string in reverse order
            a = s_list[j][-1 * i]
            b = s_list[j + 1][-1 * i]
            # print('for loop, a = %s and b = %s'%(a, b))
            if a != b:
                test = False
                break
            if j == len(s_list) - 2:
                suffix += b
        i += 1
    # reverse the order so that it comes out as read left to right
    suffix = suffix[::-1]
    #while suffix and (suffix[0] != '_'):
    #    suffix = suffix[1:]

    return suffix


def find_common(s_list, pre=True, suf=True):
    """Given a list of strings, finds the common suffix and prefix, then
    returns a 3-tuple containing:
        index 0, a new list with prefixes and suffixes removed
        index 1, the prefix that was found.
        index 2, the suffix that was found.
    Takes s_list as list of strings (required), and pre and suf as Booleans
    (optional) to indicate whether prefix and suffix should be found. Both are
    set to True by default.
    """

    prefix = ''
    if pre:
        log.debug("Finding prefixes...")
        prefix = find_prefix(s_list)
    suffix = ''
    if suf:
        log.debug("Finding suffixes...")
        suffix = find_suffix(s_list)
    # shortened = [s[len(prefix):-1*(len(suffix))] for s in s_list]
    shortened = []
    for s in s_list:
        # log.debug("s=%s"%s)
        if prefix:
            s = s[len(prefix):]
            # log.debug("s changed to: %s"%s)
        if suffix:
            s = s[:-1 * len(suffix)]
            # log.debug("s changed to: %s"%s)
        shortened.append(s)
        log.debug("final s: %s" % s)

    return (shortened, prefix, suffix)



def get_default_savepath(modelspec):
    if get_setting('USE_NEMS_BAPHY_API'):
        results_dir = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+ \
                      str(get_setting('NEMS_BAPHY_API_PORT')) + '/results'
    else:
        results_dir = get_setting('NEMS_RESULTS_DIR')

    batch = modelspec.meta.get('batch', 0)
    exptid = modelspec.meta.get('exptid', 'DATA')
    siteid = modelspec.meta.get('siteid', exptid)
    cellid = modelspec.meta.get('cellid', siteid)
    cellids = modelspec.meta.get('cellids', siteid)

    if (siteid == 'DATA') and (type(cellids) is list) and len(cellids) > 1:
        if cellid == 'none':
            siteid = 'none'  # special siteid that uses all sites in a single recording
        else:
            siteid = cellids[0].split("-")[0]
        destination = os.path.join(results_dir, str(batch), siteid,
                                   modelspec.get_longname())
    else:
        destination = os.path.join(results_dir, str(batch), cellid,
                                   modelspec.get_longname())
    log.info('model save destination: %s', destination)
    return destination

lookup_table = {}  # TODO: Replace with real memoization/joblib later

def lookup_fn_at(fn_path, ignore_table=False):
    '''
    Private function that returns a function handle found at a
    given module. Basically, a way to import a single function.
    e.g.
        myfn = _lookup_fn_at('nems0.modules.fir.fir_filter')
        myfn(data)
        ...
    '''

    # default is nems0.xforms.<fn_path>
    if not '.' in fn_path:
        fn_path = 'nems0.xforms.' + fn_path

    if (not ignore_table) and (fn_path in lookup_table):
        fn = lookup_table[fn_path]
    else:
        api, fn_name = split_to_api_and_fn(fn_path)
        api = api.replace('nems_db.xform','nems_lbhb.xform')
        api = api.replace('nems.','nems0.')
        api_obj = importlib.import_module(api)
        if ignore_table:
            importlib.reload(api_obj)  # force overwrite old imports
        fn = getattr(api_obj, fn_name)
        if not ignore_table:
            lookup_table[fn_path] = fn
    return fn

def simple_search(query, collection):
    '''
    Filter cellids or modelnames by simple space-separated search strings.

    Parameters:
    ----------
    query : str
        Search string, see syntax below.
    collection : list
        List of items to compare the search string against.
        Ex: A list of cellids.

    Returns:
    -------
    filtered_collection : list
        Collection with all non-matched entries removed.

    Syntax:
    ------
    space : OR
    &     : AND
    !     : NEGATE
    (In that order of precedence)

    Example:
    collection = ['AMT001', BRT000', 'TAR002', 'TAR003']

    search('AMT', collection)
    >>  ['AMT001']

    search('AMT TAR', collection)
    >>  ['AMT001', 'TAR002', 'TAR003']

    search ('AMT TAR&!003', collection)
    >>  ['AMT001', 'TAR002']

    search ('AMT&TAR', collection)
    >> []


    '''
    filtered_collection = []
    for c in collection:
        for s in query.split(' '):
            if ('&' in s) or ('!' in s):
                ands = s.split('&')
                all_there = True
                for a in ands:
                    negate = ('!' in a)
                    b = a.replace('!', '')
                    if (b in c) or (c in b):
                        if negate:
                            all_there = False
                            break
                        else:
                            continue
                    else:
                        if negate:
                            continue
                        else:
                            all_there = False
                            break
                if all_there:
                    filtered_collection.append(c)
            else:
                if (s in c) or (c in s):
                    filtered_collection.append(c)
                    break
                else:
                    pass

    return filtered_collection


default_configfile = os.path.join(get_setting('SAVED_SETTINGS_PATH') + '/gui.ini')
nems_root = os.path.abspath(get_setting('SAVED_SETTINGS_PATH') + '/../../')

def load_settings(config_group="db_browser_last", configfile=None):

    if configfile is None:
        configfile = default_configfile

    config = ConfigParser(delimiters=('='))

    try:
        config.read(configfile)
        settings = dict(config.items(config_group))
        return settings
    except:
        return {}


def save_settings(config_group="db_browser_last", settings={}, configfile=None):

    if configfile is None:
        configfile = default_configfile

    try:
        config.read(configfile)
    except:
        config = ConfigParser()

    try:
        # Create non-existent section
        config.add_section(config_group)
    except:
        pass

    for k, v in settings.items():
        config.set(config_group, k, v)

    with open(configfile, 'w') as f:
        config.write(f)
