
'''
Default shorthands, or 'keywords,' for generating NEMS modelspecs on a
per-module basis.

Each keyword function is indexed by an instance of the KeywordRegistry class
(see nems0.registry) by the name of the function. At runtime, when a full
keyword is used to index into the registry, the leading portion of the keyword
('kw_head' in the registry class) before the first '.' will be used to find
the appropriate function and the full keyword will be given as the first
argument to that function.

For example, the function `wc(kw)` defined in this module would be index as
`'wc'` in `example_registry`.
At runtime, `example_registry['wc.2x15.g']` would return the same result as
calling the function directly: `wc('wc.2x15.g')`.

Most (if not all) keyword functions expect at least one option to be present
after the initial '.'; some also accept additional options that are selected
as a part of the keyword string. For instance, in the previous example,
`2x15` indicates 2 inputs and 15 outputs, while `g` indicates gaussian
coefficients. Each additional option should be separated by a preceeding '.'

See individual keyword functions for a full description of their
parsing options.

'''

import re
import logging

import numpy as np
from nems0.registry import xmodule
from nems0.utils import escaped_split
log = logging.getLogger(__name__)


def _one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


@xmodule('wc')
def wc(kw):
    '''
    Parses the default modulespec for basic and gaussian channel weighting. By default, weights are initialized
    to small positive value (0.01). TODO: Should this be zero instead??

    Parameter
    ---------
    kw : string
        A string of the form: wc.{n_inputs}x{n_outputs}.option1.option2...

    Options
    -------
    c : Used when n_outputs is greater than n_inputs (overwrites g)
    g : For gaussian coefficients (overwrites c)
    z : initialize all coefficients to zero (or mean zero if shuffling)
    n : To apply normalization
    o : include offset paramater, a constant ("bias") added to each output

    Note that the options are parsed in the order that they are passed
    and some overwrite each other, which means the last option takes
    precedence. For example,

    `wc.15x2.c.g` would be equivalent to `wc.15x2.g`,
    whereas `wc.15x2.g.c` would be equivalent to `wc.15x2.c`.
    '''
    options = kw.split('.')
    in_out_pattern = re.compile(r'^(\d{1,})x(\d{1,})$')
    try:
        parsed = re.match(in_out_pattern, options[1])
        n_inputs = int(parsed.group(1))
        n_outputs = int(parsed.group(2))
    except (TypeError, IndexError):
        # n_inputs x n_outputs should always follow wc.
        # TODO: Ideally would like the order to not matter like with other
        #       options but this seemed like a sensible solution for now
        #       since the information is mandatory.
        raise ValueError("Got TypeError or IndexError when attempting to parse "
                         "wc keyword.\nMake sure <in>x<out> is provided "
                         "as the first option after 'wc', e.g.: 'wc.2x15'"
                         "\nkeyword given: %s" % kw)

    if 'c' in options and 'g' in options:
        log.warning("Options 'c' and 'g' both given for weight_channels, but"
                    " are mutually exclusive. Whichever comes last will "
                    "overwrite the previous option. kw given: {}".format(kw))

    # This is the default for wc, but options might overwrite it.
    fn = 'nems0.modules.weight_channels.basic'
    fn_kwargs = {'i': 'pred',
                 'o': 'pred',
                 'normalize_coefs': False,
                 'chans': n_outputs,
                 }
    tf_layer = 'nems0.tf.layers.WeightChannelsBasic'
    p_coefficients = {'mean': np.full((n_outputs, n_inputs), 0.01),
                      'sd': np.full((n_outputs, n_inputs), 0.05)}
    # add some variety across channels to help get the fitter started
    for i in range(n_outputs):
        x0 = int(i/n_outputs*n_inputs)
        x1 = int((i+1)/n_outputs*n_inputs)
        p_coefficients['mean'][i, x0:x1] = 0.02
    prior = {'coefficients': ('Normal', p_coefficients)}
    normalize = False
    coefs = None
    bounds = None

    for op in options[2:]:  # will be empty if only wc and {in}x{out}
        if op == 'z':
            # weighting scheme from https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
            p_coefficients = {'mean': np.zeros((n_outputs, n_inputs)),
                              'sd': np.full((n_outputs, n_inputs), np.sqrt(2/(n_outputs)))}
            prior = {'coefficients': ('Normal', p_coefficients)}

        elif op == 'u':
            # weighting scheme from https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
            p_coefficients = {'mean': np.zeros((n_outputs, n_inputs))+np.sqrt(2/(n_outputs))/50,
                              'sd': np.full((n_outputs, n_inputs), np.sqrt(2/(n_outputs)))}
            prior = {'coefficients': ('Normal', p_coefficients)}

        elif op == 'rnd':
            # weighting scheme from https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
            save_state = np.random.get_state()
            np.random.seed(100)
            
            p_coefficients = {'mean': np.random.randn(n_outputs, n_inputs)*np.sqrt(2/(n_outputs))/50,
                              'sd': np.full((n_outputs, n_inputs), np.sqrt(2/(n_outputs)))}
            prior = {'coefficients': ('Normal', p_coefficients)}
            
            # restore random state
            np.random.set_state(save_state)

        elif op == 'c':

            if n_outputs == 1:
                p_coefficients = {
                    'mean': np.ones((n_outputs, n_inputs))/n_outputs,
                    'sd': np.full((n_outputs, n_inputs), 0.2),
                }
            else:
                p_coefficients = {
                    'mean': np.eye(n_outputs, n_inputs),
                    'sd': np.full((n_outputs, n_inputs), 0.2),
                }
                if n_outputs > n_inputs:
                    p_coefficients['mean'][n_outputs:, :] = 1 / n_inputs
                elif n_inputs > n_outputs:
                    p_coefficients['mean'][:, n_inputs:] = 1 / n_outputs

            prior = {'coefficients': ('Normal', p_coefficients)}

        elif op == 'g':

            # Generate evenly-spaced filter centers for the starting points
            fn = 'nems0.modules.weight_channels.gaussian'
            fn_kwargs = {'i': 'pred',
                         'o': 'pred',
                         'n_chan_in': n_inputs,
                         'normalize_coefs': False,
                         'chans': n_outputs,
                         }
            tf_layer = 'nems0.tf.layers.WeightChannelsGaussian'
            coefs = 'nems0.modules.weight_channels.gaussian_coefficients'
            mean = np.arange(n_outputs+1)/(n_outputs*2+2) + 0.25
            mean = mean[1:]
            sd = np.full_like(mean, 0.4)

            mean_prior_coefficients = {
                'mean': mean,
                'sd': np.full_like(mean, 0.2),
            }
            sd_prior_coefficients = {'sd': sd}
            prior = {'mean': ('Normal', mean_prior_coefficients),
                     'sd': ('HalfNormal', sd_prior_coefficients)}
            bounds = {
                'mean': (np.full_like(mean, -0.01), np.full_like(mean, 1.01)),
                'sd': (np.full_like(mean, 0.05), np.full_like(mean, 0.6))}

        elif op == 'n':
            normalize = True
            
    if 'n' in options:
        fn_kwargs['normalize_coefs'] = True

    if 'o' in options:
        fn = 'nems0.modules.weight_channels.basic_with_offset'
        o_coefficients = {
            'mean': np.zeros((n_outputs, 1)),
            'sd': np.ones((n_outputs, 1))
        }
        prior['offset'] = ('Normal', o_coefficients)

    if 'r' in options:
        fn = 'nems0.modules.weight_channels.basic_with_rect'
        o_coefficients = {
            'mean': np.zeros((n_outputs, 1)),
            'sd': np.ones((n_outputs, 1))
        }
        prior['offset'] = ('Normal', o_coefficients)

    template = {
        'fn': fn,
        'fn_kwargs': fn_kwargs,
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.weight_channels_heatmap'],
        'plot_fn_idx': 2,
        'prior': prior,
        'tf_layer': tf_layer,
    }
    if bounds is not None:
        template['bounds'] = bounds
#    if normalize:
#        template['norm'] = {'type': 'minmax', 'recalc': 0,
#                            'd': np.zeros([n_outputs, 1]),
#                            'g': np.ones([n_outputs, 1])}

    if coefs is not None:
        template['fn_coefficients'] = coefs

    return template

@xmodule('lv')
def lv(kw):
    '''
    weighted sum of r responses (inputs) into n channels (outputs)

    Parameter
    ---------
    kw : string
        A string of the form: lv.{n_inputs}x{n_outputs}.option1.option2...

    Options
    -------

    '''
    options = kw.split('.')
    in_out_pattern = re.compile(r'^(\d{1,})x(\d{1,})$')
    try:
        parsed = re.match(in_out_pattern, options[1])
        n_inputs = int(parsed.group(1))
        n_outputs = int(parsed.group(2))
    except (TypeError, IndexError):
        # n_inputs x n_outputs should always follow wc.
        # TODO: Ideally would like the order to not matter like with other
        #       options but this seemed like a sensible solution for now
        #       since the information is mandatory.
        raise ValueError("Got TypeError or IndexError when attempting to parse "
                         "wc keyword.\nMake sure <in>x<out> is provided "
                         "as the first option after 'wc', e.g.: 'wc.2x15'"
                         "\nkeyword given: %s" % kw)

    do_gain = ('g' in options)

    if do_gain:
       fn = 'nems0.modules.state.lv_gain'
    else:
       fn = 'nems0.modules.state.lv_additive'
    fn_kwargs = {'i': 'pred', 'o': 'pred', 's': 'state'}

    n_state = int(n_inputs/(n_outputs+1))*2

    # This is the default for wc, but options might overwrite it.
    p_coefficients_in = {'mean': np.zeros((1, n_inputs))+0.01,
                      'sd': np.full((1, n_inputs), 0.1)}
    p_coefficients_in['mean'][:,:int(n_state/2)] = 1
    p_coefficients = {'mean': np.zeros((n_outputs, n_state))+0.00,
                      'sd': np.full((n_outputs, n_state), 0.1)}
    if do_gain:
        p_coefficients['mean'][:,0]=1
    prior = {'coefficients_in': ('Normal', p_coefficients_in),
             'coefficients': ('Normal', p_coefficients),
            }

    bounds = None
    max_in = np.full_like(p_coefficients_in['mean'], np.inf)
    min_in = np.full_like(p_coefficients_in['mean'], -np.inf)
    max_in[:,:int(n_state/2)]=1
    min_in[:,:int(n_state/2)]=1
    bounds = {
        'coefficients_in': (min_in, max_in),
        'coefficients': (np.full_like(p_coefficients['mean'], -np.inf), 
                         np.full_like(p_coefficients['mean'], np.inf))
        }

    template = {
        'fn': fn,
        'fn_kwargs': fn_kwargs,
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.weight_channels_heatmap'],
        'plot_fn_idx': 2,
        'prior': prior
    }

    if bounds is not None:
        template['bounds'] = bounds

    return template


@xmodule('fir')
def fir(kw):
    '''
    Generate and register default modulespec for basic channel weighting

    Parameters
    ----------
    kw : str
        A string of the form: fir.{n_inputs}x{n_coefs}x{n_banks}

    Options
    -------
    None, but x{n_banks} is optional.
    '''
    pattern = re.compile(r'^fir\.?(\d{1,})x(\d{1,})x?(\d{1,})?$')
    ops = kw.split(".")
    kw = ".".join(ops[:2])
    parsed = re.match(pattern, kw)
    try:
        n_inputs = int(parsed.group(1))
        n_coefs = int(parsed.group(2))
        n_banks = parsed.group(3)  # None if not given in keyword string
    except TypeError:
        raise ValueError("Got a TypeError when parsing fir keyword. Make sure "
                         "keyword has the form: \n"
                         "fir.{n_inputs}x{n_coefs}x{n_banks} (banks optional)"
                         "\nkeyword given: %s" % kw)
    if n_banks is None:
        n_banks = 1
    else:
        n_banks = int(n_banks)

    rate = 1
    non_causal = 0
    include_offset = False
    cross_channels = False
    for op in ops:
        if op == 'x':
            cross_channels = True
        elif op.startswith('r'):
            rate = int(op[1:])
        elif op.startswith('nc'):
            # noncausal fir implementation (for reverse model or neg motor offsets)
            if len(op) == 2:
                # default is to make all bins negative or 0
                non_causal = n_coefs-1
            else:
                non_causal = int(op[2:])

        elif op == 'off':
            # add variable offset parameter
            include_offset = True

    p_coefficients = {
        'mean': np.zeros((n_inputs * n_banks, n_coefs)),
        'sd': np.ones((n_inputs * n_banks, n_coefs)) * 0.05,
    }

    if 'fl' in ops:
        p_coefficients['mean'][:] = 1 / (n_inputs * n_coefs)
    elif 'z' in ops:
        p_coefficients['mean'][:] = 0
    elif 'p' in ops:
        p_coefficients['mean'][:, 1+non_causal] = 0.1
        p_coefficients['mean'][:, 2+non_causal] = 0.05
    elif ('l' in ops) and (n_coefs > 2):
        p_coefficients['mean'][:, 1+non_causal] = 0.1
        p_coefficients['mean'][:, 2+non_causal] = 0.05
        p_coefficients['mean'][:, 3+non_causal] = -0.05
        p_coefficients['mean'][:, 4+non_causal] = -0.05
    elif n_coefs > 2:
        p_coefficients['mean'][:, 1+non_causal] = 0.1
        p_coefficients['mean'][:, 2+non_causal] = -0.05
    else:
        p_coefficients['mean'][:, 0] = 0.05


    if (n_banks == 1) and (not cross_channels):
        plot_fn_idx = 4 if n_inputs <=3 else 2
        template = {
            'fn': 'nems0.modules.fir.basic',
            'fn_kwargs': {'i': 'pred', 'o': 'pred', 'non_causal': non_causal,
                          'cross_channels': cross_channels, 'chans': n_coefs,
                          'n_banks': n_banks, 'n_inputs': n_inputs},
            'tf_layer': 'nems0.tf.layers.FIR',
            'plot_fns': ['nems0.plots.api.mod_output',
                         'nems0.plots.api.spectrogram_output',
                         'nems0.plots.api.strf_heatmap',
                         'nems0.plots.api.strf_local_lin',
                         'nems0.plots.api.strf_timeseries',
                         'nems0.plots.api.fir_output_all'],
            'plot_fn_idx': plot_fn_idx,
            'prior': {
                'coefficients': ('Normal', p_coefficients),
            }
        }
    else:
        template = {
            'fn': 'nems0.modules.fir.filter_bank',
            'fn_kwargs': {'i': 'pred', 'o': 'pred', 'non_causal': non_causal,
                          'bank_count': n_banks, 'cross_channels': cross_channels,
                          'chans': n_coefs, 'n_inputs': n_inputs},
            'tf_layer': 'nems0.tf.layers.FIR',
            'plot_fns': ['nems0.plots.api.mod_output',
                         'nems0.plots.api.spectrogram_output',
                         'nems0.plots.api.strf_heatmap',
                         'nems0.plots.api.strf_local_lin',
                         'nems0.plots.api.strf_timeseries',
                         'nems0.plots.api.fir_output_all'],
            'plot_fn_idx': 2,
            'prior': {
                'coefficients': ('Normal', p_coefficients),
            }
        }
    if rate > 1:
        template['fn_kwargs']['rate'] = rate
    if include_offset:
        mean_off = np.zeros((n_inputs, 1))
        sd_off = np.full((n_inputs, 1), 1)
        template['prior']['offsets'] = ('Normal', {'mean': mean_off,
                                                   'sd': sd_off})

    return template


@xmodule('strf')
def strf(kw):
    '''
    Generate a stim_channel x time_bin array of coefficients to be used
    as an inseparable STRF.

    Parameters
    ----------
    kw : str
        A string of the form: strf.{stim_channel_count}x{n_coefs}

    Options
    -------
    f : "first module" - change input from pred to stim
    TODO: approximations with different basis functions

    '''
    options = kw.split('.')
    stim_channels, temporal_bins = [int(a) for a in options[1].split('x')]

    input_name = 'pred'
    for op in options[2:]:
        if op == 'f':
            input_name = 'stim'

    prior_coeffs = {
            'mean': np.zeros((stim_channels, temporal_bins)),
            'sd': np.ones((stim_channels, temporal_bins))
            }
    template = {
            'fn': 'nems0.modules.strf.nonparametric',
            'fn_kwargs': {'i': input_name, 'o': 'pred'},
            'plot_fns': ['nems0.plots.heatmap.nonparametric_strf'],
            'plot_fn_idx': 0,
            'prior': {
                    'coefficients': ('Normal', prior_coeffs)
                    }
            }

    return template


@xmodule('pz')
def pz(kw):
    '''
    Generate and register default modulespec for pole-zero filters

    Parameters
    ----------
    kw : str
        A string of the form: fir.{n_inputs}x{n_coefs}x{n_banks}

    Options
    -------
    None, but x{n_banks} is optional.
    '''
    options = kw.split('.')
    pattern = re.compile(r'^(\d{1,})x(\d{1,})x?(\d{1,})?$')
    parsed = re.match(pattern, options[1])
    try:
        n_inputs = int(parsed.group(1))
        n_coefs = int(parsed.group(2))
        n_banks = parsed.group(3)  # None if not given in keyword string
    except TypeError:
        raise ValueError("Got a TypeError when parsing fir keyword. Make sure "
                         "keyword has the form: \n"
                         "pz.{n_inputs}x{n_coefs}x{n_banks} (banks optional)"
                         "\nkeyword given: %s" % kw)
    if n_banks is None:
        n_banks = 1
    else:
        n_banks = int(n_banks)
    if n_banks > 1:
        raise ValueError("nbanks > 1 not yet supported for pz")

    npoles = 3
    nzeros = 1

    for op in options[2:]:
        if op.startswith('p'):
            npoles = int(op[1:])

        elif op.startswith('z'):
            nzeros = int(op[1:])
    pole_set = np.array([[0.8, -0.4, 0.1, 0, 0]])
    zero_set = np.array([[0.1, 0.1, 0.1, 0.1, 0]])
    p_poles = {
        'mean': np.repeat(pole_set[:,:npoles], n_inputs, axis=0),
        'sd': np.ones((n_inputs, npoles))*.3,
    }
    p_zeros = {
        'mean': np.repeat(zero_set[:,:nzeros], n_inputs, axis=0),
        'sd': np.ones((n_inputs, nzeros))*.2,
    }
    p_delays = {
        'sd': np.ones((n_inputs, 1))*.02,
    }
    p_gains = {
        'mean': np.zeros((n_inputs, 1))+.1,
        'sd': np.ones((n_inputs, 1))*.2,
    }
    plot_fn_idx = 4 if n_inputs <= 3 and n_banks == 1 else 2
    template = {
        'fn': 'nems0.modules.fir.pole_zero',
        'fn_kwargs': {'i': 'pred', 'o': 'pred', 'n_coefs': n_coefs},
        'fn_coefficients': 'nems0.modules.fir.pz_coefficients',
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.strf_heatmap',
                     'nems0.plots.api.strf_local_lin',
                     'nems0.plots.api.strf_timeseries',
                     'nems0.plots.api.fir_output_all'],
        'plot_fn_idx': plot_fn_idx,
        'prior': {
            'poles': ('Normal', p_poles),
            'zeros': ('Normal', p_zeros),
            'gains': ('Normal', p_gains),
            'delays': ('HalfNormal', p_delays),
        }
    }

    return template


@xmodule('do')
def do(kw):
    '''
    Generate and register default modulespec for damped oscillator-based filters.
    Several parameters have bounds

    Parameters
    ----------
    kw : str
        A string of the form: do.{n_inputs}x{n_coefs}x{n_banks}

    Options
    -------
    n_banks : default 1
    x : (False) if true cross each filter with each input (requires n_inputs==1?)
    lN : mean delay=N time bins (default 1)

    '''
    options = kw.split('.')
    pattern = re.compile(r'^(\d{1,})x(\d{1,})x?(\d{1,})?$')
    parsed = re.match(pattern, options[1])
    try:
        n_inputs = int(parsed.group(1))
        n_coefs = int(parsed.group(2))
        n_banks = parsed.group(3)  # None if not given in keyword string
    except TypeError:
        raise ValueError("Got a TypeError when parsing do() keyword. Make sure "
                         "keyword has the form: \n"
                         "do.{n_inputs}x{n_coefs}x{n_banks} (n_banks optional)"
                         "\nkeyword given: %s" % kw)

    if n_banks is None:
        n_banks = 1
    else:
        n_banks = int(n_banks)

    n_channels = n_inputs * n_banks
    cross_channels = False
    mean_delay = 1.5

    # additional options
    for op in options[2:]:
        if op == 'x':
            cross_channels = True
        elif op[:1] == 'l':
            mean_delay=int(op[1:])

    p_f1s = {
        'sd': np.full((n_channels, 1), 0.6)
    }
    p_taus = {
        'sd': np.full((n_channels, 1), 0.3)
    }
    g0 = np.array([[1, -0.5, 1, -0.5, 1, -0.5, 1, -0.5]]).T / 5
    g0 = np.tile(g0, (int(np.ceil(n_channels / len(g0))), 1))[:n_channels, :]
    p_gains = {
            'mean': np.tile(g0[:n_inputs, :], (n_banks, 1)),  # TODO: tile
            'sd': np.ones((n_channels, 1))* 0.2,
    }
    p_delays = {
        'sd': np.full((n_channels, 1), mean_delay)
    }
    plot_fn_idx = 4 if n_inputs <= 3 and n_banks == 1 else 2
    template = {
        'fn': 'nems0.modules.fir.damped_oscillator',
        'fn_kwargs': {'i': 'pred', 'o': 'pred', 'n_coefs': n_coefs, 'chans': n_coefs,
                      'bank_count': n_banks, 'cross_channels': cross_channels, 'n_inputs': n_inputs},
        'fn_coefficients': 'nems0.modules.fir.do_coefficients',
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.strf_heatmap',
                     'nems0.plots.api.strf_local_lin',
                     'nems0.plots.api.strf_timeseries',
                     'nems0.plots.api.fir_output_all'],
        'plot_fn_idx': plot_fn_idx,
        'tf_layer': 'nems0.tf.layers.DampedOscillator',
        'prior': {
            'f1s': ('HalfNormal', p_f1s),
            'taus': ('HalfNormal', p_taus),
            'gains': ('Normal', p_gains),
            'delays': ('HalfNormal', p_delays)},
        'bounds': {
            'f1s': (np.full((n_channels, 1), 1e-15), np.full((n_channels, 1), 2*np.pi)),
            'taus': (np.full((n_channels, 1), 0), np.full((n_channels, 1), np.inf)),
            'gains': (np.full((n_channels, 1), -np.inf), np.full((n_channels, 1), np.inf)),
            'delays': (np.full((n_channels, 1), -1), np.full((n_channels, 1), n_coefs))}
    }

    return template


@xmodule('fird')
def fird(kw):
    '''
    Generate and register default modulespec for fir_dexp filters

    Parameters
    ----------
    kw : str
        A string of the form: fird.{n_outputs}x{n_coefs}x{n_banks}

    Options
    -------
    None, but x{n_banks} is optional.
    '''
    options = kw.split('.')
    pattern = re.compile(r'^(\d{1,})x(\d{1,})x?(\d{1,})?$')
    parsed = re.match(pattern, options[1])
    try:
        n_outputs = int(parsed.group(1))
        n_coefs = int(parsed.group(2))
        n_banks = parsed.group(3)  # None if not given in keyword string
    except TypeError:
        raise ValueError("Got a TypeError when parsing fir keyword. Make sure "
                         "keyword has the form: \n"
                         "fird.{n_outputs}x{n_coefs}x{n_banks} (banks optional)"
                         "\nkeyword given: %s" % kw)
    if n_banks is None:
        n_banks = 1
    else:
        n_banks = int(n_banks)
    if n_banks > 1:
        raise ValueError("nbanks > 1 not yet supported for pz")

    #phi_set = np.array([[1, 2, 1, 3, 2, -0.25]])
    phi_set = np.array([[1, 0.3, 1, 3, 0.3, -0.75]])
    p_phi = {
        'mean': np.repeat(phi_set, n_outputs, axis=0),
        'sd': np.ones((n_outputs, 6))*.1,
    }

    template = {
        'fn': 'nems0.modules.fir.fir_dexp',
        'fn_kwargs': {'i': 'pred', 'o': 'pred', 'n_coefs': n_coefs},
        'prior': {
            'phi': ('Normal', p_phi),
        }
    }

    return template


@xmodule('firexp')
def firexp(kw):
    '''
    Generate and register default modulespec for fir_exp filters

    Parameters
    ----------
    kw : str
        A string of the form: firexp.{n_outputs}x{n_coefs}

    '''
    options = kw.split('.')
    pattern = re.compile(r'^(\d{1,})x(\d{1,})?$')
    parsed = re.match(pattern, options[1])
    try:
        n_chans = int(parsed.group(1))
        n_coefs = int(parsed.group(2))
    except TypeError:
        raise ValueError("Got a TypeError when parsing fir keyword. Make sure "
                         "keyword has the form: \n"
                         "firexp.{n_outputs}x{n_coefs}"
                         "\nkeyword given: %s" % kw)

    tau = np.ones((n_chans, 1))
    a = np.ones((n_chans, 1))
    b = np.zeros((n_chans, 1))
    s = np.zeros((n_chans, 1))
    prior = {'tau': ('Normal', {'mean': tau, 'sd': np.ones(n_chans)})}
    fn_kwargs = {'i': 'pred', 'o': 'pred', 'n_coefs': n_coefs}
    prior.update({
            'a': ('Exponential', {'beta': a}),
            'b': ('Normal', {'mean': b, 'sd': np.ones(n_chans, 1)}),
            's': ('Normal', {'mean': s, 'sd': np.ones(n_chans, 1)})
            })

    template = {
        'fn': 'nems0.modules.fir.fir_exp',
        'fn_kwargs': fn_kwargs,
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.strf_heatmap',
                     'nems0.plots.api.strf_timeseries'],
        'plot_fn_idx': 2,
        'prior': prior,
        'bounds': {'tau': (1e-15, None), 'a': (1e-15, None)}
    }

    return template


@xmodule('lvl')
def lvl(kw):
    '''
    Generate and register default modulespec for the levelshift module.

    Parameters
    ----------
    kw : str
        Expected format: r'^lvl\.(\d{1,})$'

    Options
    -------
    None
    '''
    options = kw.split('.')
    required = '.'.join(options[:2])
    pattern = re.compile(r'^lvl\.?(\d{1,})$')
    parsed = re.match(pattern, required)
    try:
        n_shifts = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got a TypeError when parsing lvl keyword, "
                         "make sure keyword has the form: \n"
                         "lvl.{n_shifts}.\n"
                         "keyword given: %s" % kw)

    template = {
        'fn': 'nems0.modules.levelshift.levelshift',
        'fn_kwargs': {'i': 'pred', 'o': 'pred', 'chans': n_shifts},
        'tf_layer': 'nems0.tf.layers.Levelshift',
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.pred_resp',
                     'nems0.plots.api.null'],
        'plot_fn_idx': 3,
        'prior': {'level': ('Normal', {'mean': np.zeros([n_shifts, 1]),
                                       'sd': np.ones([n_shifts, 1])/100})}

        }

    return template


@xmodule('scl')
def scl(kw):
    '''
    Generate and register default modulespec for the scale module.

    Parameters
    ----------
    kw : str
        Expected format: r'^scl\.(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^scl\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_scales = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got a TypeError when parsing lvl keyword, "
                         "make sure keyword has the form: \n"
                         "scl.{n_scales}.\n"
                         "keyword given: %s" % kw)

    template = {
        'fn': 'nems0.modules.scale.scale',
        'fn_kwargs': {'i': 'pred', 'o': 'pred'},
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     ],
        'plot_fn_idx': 0,
        'prior': {'a': ('Normal', {'mean': np.ones([n_scales, 1]),
                                   'sd': np.ones([n_scales, 1])})}
        }

    return template


@xmodule('sum')
def sum(kw):
    """
    sum signal <sig> over channels (N x T) --> (1 x T) signal
    syntax: sum.<sig>  default <sig>="pred"
    """
    op = kw.split(".")[1:]
    sig = 'pred'
    if len(op) > 0:
        sig = op[0]
    template = {
        'fn': 'nems0.modules.sum.sum_channels',
        'tf_layer': 'nems0.tf.layers.Sum',
        'fn_kwargs': {'i': sig,
                      'o': sig,
                      },
        'phi': {}
        }

    return template


@xmodule('stp')
def stp(kw):
    '''
    Generate and register modulespec for short_term_plasticity module.

    Parameters
    ----------
    kw : str
        Expected format: r'^stp\.?(\d{1,})\.?([zbnstxq.]*)$'

    Options
    -------
    z : Change prior conditions (see function body)
    b : Set bounds on 'tau' to be greater than or equal to 0
    n : Apply normalization
    t : Threshold inputs to synapse
    q : quick version of STP, fits differently for some reason? so not default
    '''
    pattern = re.compile(r'^stp\.?(\d{1,})\.?([zbnstxqwv.\.]*)$')
    parsed = re.match(pattern, kw)
    try:
        n_synapse = int(parsed.group(1))
    except (TypeError, IndexError):
        raise ValueError("Got TypeError or IndexError while parsing stp "
                         "keyword,\nmake sure keyword is of the form: \n"
                         "stp.{n_synapse}.option1.option2...\n"
                         "keyword given: %s" % kw)
    options = parsed.group(2).split('.')

    # Default values, may be overwritten by options
    u_mean = [0.01] * n_synapse
    tau_mean = [0.05] * n_synapse
    x0_mean = [0] * n_synapse
    crosstalk = 0

    quick_eval = ('q' in options)
    normalize = ('n' in options)
    threshold = ('t' in options)
    bounds = ('b' in options)

    for op in options:
        if op == 'z':
            tau_mean = [0.01] * n_synapse
        elif op == 'v':
            u_mean = [1.0] * n_synapse
        elif op == 's':
            u_mean = [0.1] * n_synapse
        elif op == 'w':
            u_mean = [0.001] * n_synapse
        elif op == 'x':
            crosstalk = 1

    u_sd = [0.05] * n_synapse
    if n_synapse == 1:
        # TODO:
        # @SVD: stp1 had this as 0.01, all others 0.05. intentional?
        #       if not can just set tau_sd = [0.05]*n_synapse
        tau_sd = u_sd
    else:
        tau_sd = [0.01]*n_synapse

    template = {
        'fn': 'nems0.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred', 'o': 'pred', 'crosstalk': crosstalk,
                      'quick_eval': quick_eval, 'reset_signal': 'epoch_onsets',
                      'chans': n_synapse},
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.before_and_after',
                     'nems0.plots.api.before_and_after_stp'],
        'plot_fn_idx': 3,
        'prior': {'u': ('Normal', {'mean': u_mean, 'sd': u_sd}),
                  'tau': ('HalfNormal', {'sd': tau_mean})},
        'bounds': {'u': (np.full_like(u_mean, -np.inf), np.full_like(u_mean, np.inf)),
                   'tau': (np.full_like(tau_mean, 0.01), np.full_like(tau_mean, np.inf))}
    }
    if normalize:
        d = np.array([0]*n_synapse)
        g = np.array([1]*n_synapse)
        template['norm'] = {'type': 'minmax', 'recalc': 0, 'd': d, 'g': g}

    if threshold:
        template['prior']['x0'] = ('Normal', {'mean': x0_mean, 'sd': u_sd})
        template['bounds']['x0'] = (np.full_like(x0_mean, -np.inf), np.full_like(x0_mean, np.inf))

    if quick_eval:
        template['tf_layer'] = 'nems0.tf.layers.STPQuick'

    return template


@xmodule('dep')
def dep(kw):
    """ same as stp(kw) but sets kw_args->dep_only = True """
    template = stp(kw.replace('dep','stp'))
    #template['kw_args']['dep_only'] = True
    u_mean = template['prior']['u'][1]['mean']
    tau_mean = template['prior']['tau'][1]['mean']
    template['bounds'] = {'u': (np.full_like(u_mean, 0), np.full_like(u_mean, np.inf)),
                          'tau': (np.full_like(tau_mean, 0.01), np.full_like(tau_mean, np.inf))}

    return template


@xmodule('stp2')
def stp2(kw):
    '''
    Generate and register modulespec for short_term_plasticity2 module. Two plasticity timecoursees

    Parameters
    ----------
    kw : str
        Expected format: r'^stp2\.?(\d{1,})\.?([zbnstxq.]*)$'

    Options
    -------
    z : Change prior conditions (see function body)
    b : Set bounds on 'tau' to be greater than or equal to 0
    n : Apply normalization
    t : Threshold inputs to synapse
    q : quick version of STP, fits differently for some reason? so not default
    '''
    pattern = re.compile(r'^stp2\.?(\d{1,})\.?([zbnstxq.\.]*)$')
    parsed = re.match(pattern, kw)
    try:
        n_synapse = int(parsed.group(1))
    except (TypeError, IndexError):
        raise ValueError("Got TypeError or IndexError while parsing stp "
                         "keyword,\nmake sure keyword is of the form: \n"
                         "stp.{n_synapse}.option1.option2...\n"
                         "keyword given: %s" % kw)
    options = parsed.group(2).split('.')

    # Default values, may be overwritten by options
    u_mean = [0.01] * n_synapse
    tau_mean = [0.1] * n_synapse
    x0_mean = [0] * n_synapse
    crosstalk = 0

    quick_eval = ('q' in options)
    normalize = ('n' in options)
    threshold = ('t' in options)
    bounds = ('b' in options)

    for op in options:
        if op == 'z':
            tau_mean = [0.01] * n_synapse
        elif op == 's':
            u_mean = [0.05] * n_synapse
        elif op == 'x':
            crosstalk = 1

    u_sd = [0.05] * n_synapse
    if n_synapse == 1:
        # TODO:
        # @SVD: stp1 had this as 0.01, all others 0.05. intentional?
        #       if not can just set tau_sd = [0.05]*n_synapse
        tau_sd = u_sd
    else:
        tau_sd = [0.01]*n_synapse

    template = {
        'fn': 'nems0.modules.stp.short_term_plasticity2',
        'fn_kwargs': {'i': 'pred', 'o': 'pred', 'crosstalk': crosstalk,
                      'quick_eval': quick_eval, 'reset_signal': 'epoch_onsets'},
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.before_and_after',
                     'nems0.plots.api.before_and_after_stp'],
        'plot_fn_idx': 3,
        'prior': {'u': ('Normal', {'mean': u_mean, 'sd': u_sd}),
                  'tau': ('Normal', {'mean': tau_mean, 'sd': tau_sd}),
                  'u2': ('Normal', {'mean': [uu*5 for uu in u_mean], 'sd': [uu*5 for uu in u_sd]}),
                  'tau2': ('Normal', {'mean': [tt/5 for tt in tau_mean], 'sd': [tt/5 for tt in tau_sd]}),
                  'urat': ('Normal', {'mean': 0.5, 'sd': 0.2})
                  },
        'bounds': {'u': (np.full_like(u_mean, -np.inf), np.full_like(u_mean, np.inf)),
                  'tau': (np.full_like(u_mean, 0.01), np.full_like(u_mean, np.inf)),
                  'u2': (np.full_like(u_mean, -np.inf), np.full_like(u_mean, np.inf)),
                  'tau2': (np.full_like(u_mean, 0.01), np.full_like(u_mean, np.inf)),
                  'urat': (0, 1)
                  }
        }

    if normalize:
        d = np.array([0]*n_synapse)
        g = np.array([1]*n_synapse)
        template['norm'] = {'type': 'minmax', 'recalc': 0, 'd': d, 'g': g}

    if bounds:
        template['bounds'] = {'tau': (0, None)}

    if threshold:
        template['prior']['x0'] = ('Normal', {'mean': x0_mean, 'sd': u_sd})

    return template


@xmodule('dexp')
def dexp(kw):
    '''
    Generate and register modulespec for double_exponential module.

    Parameters
    ----------
    kw : str
        Expected format: r'^dexp\.?(\d{1,})$'

    Options
    -------
       <n> : n dimensions
       s : apply to state rather than pred (pred==default in/out)
    '''
    #pattern = re.compile(r'^dexp\.?(\d{1,})$')
    #parsed = re.match(pattern, kw)
    ops = kw.split(".")
    if len(ops) == 1:
        raise ValueError("required parameter dexp.<n>")

    n_dims = int(ops[1])
    inout_name = 'pred'
    bounded = False
    for op in ops[2:]:
        if op == 's':
            inout_name = 'state'
        elif op == 'b':
            bounded = True
        else:
            raise ValueError('dexp keyword: invalid option %s' % op)

    base_mean = np.zeros([n_dims, 1]) if n_dims > 1 else np.array([0])
    base_sd = np.ones([n_dims, 1])*0.01 if n_dims > 1 else np.array([0.01])
    amp_mean = base_mean + 5
    amp_sd = base_mean + 0.5
    shift_mean = base_mean
    shift_sd = base_sd
    kappa_mean = base_mean + 1
    kappa_sd = base_sd*10

    template = {
        'fn': 'nems0.modules.nonlinearity.double_exponential',
        'fn_kwargs': {'i': inout_name,
                      'o': inout_name,
                      'chans': n_dims},
        'tf_layer': 'nems0.tf.layers.DoubleExponential',
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.pred_resp',
                     'nems0.plots.api.before_and_after',
                     'nems0.plots.api.nl_scatter'],
        'plot_fn_idx': 4,
        'prior': {'base': ('Normal', {'mean': base_mean, 'sd': base_sd}),
                  'amplitude': ('Normal', {'mean': amp_mean, 'sd': amp_sd}),
                  'shift': ('Normal', {'mean': shift_mean, 'sd': shift_sd}),
                  'kappa': ('Normal', {'mean': kappa_mean, 'sd': kappa_sd})}
        }

    if bounded:
        template['bounds'] = {
                'base': (1e-15, None),
                'amplitude': (None, None),
                'shift': (None, None),
                'kappa': (None, None),
                }

    return template


@xmodule('qsig')
def qsig(kw):
    '''
    Generate and register modulespec for quick_sigmoid module.

    Parameters
    ----------
    kw : str
        Expected format: r'^qsig\.?(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^qsig\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    n_dims = int(parsed.group(1))

    zeros = np.zeros([n_dims, 1]) if n_dims > 1 else np.array([0])
    base_mean = zeros + 0.1
    base_sd = base_mean
    amp_mean = zeros + 0.7
    amp_sd = zeros + 0.5
    shift_mean = zeros + 1.5
    shift_sd = zeros + 1.0
    kappa_mean = base_mean
    kappa_sd = base_mean

    template = {
        'fn': 'nems0.modules.nonlinearity.quick_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.pred_resp',
                     'nems0.plots.api.before_and_after',
                     'nems0.plots.api.nl_scatter'],
        'plot_fn_idx': 2,
        'prior': {'base': ('Normal', {'mean': base_mean, 'sd': base_sd}),
                  'amplitude': ('Normal', {'mean': amp_mean, 'sd': amp_sd}),
                  'shift': ('Normal', {'mean': shift_mean, 'sd': shift_sd}),
                  'kappa': ('Normal', {'mean': kappa_mean, 'sd': kappa_sd})}
        }

    return template


@xmodule('logsig')
def logsig(kw):
    '''
    Generate and registry modulespec for the logistic_sigmoid module.

    Parameters
    ----------
    kw : str
        Expected format: logsig

    Options
    -------
    None

    Note
    ----
    The priors set by this keyword are typically overwritten by
    `nems0.initializers.init_logsig`.
    Additionally, this function performs no parsing at this time,
    so any keyword beginning with 'logsig.' will be equivalent.
    '''
    template = {
        'fn': 'nems0.modules.nonlinearity.logistic_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.pred_resp',
                     'nems0.plots.api.before_and_after',
                     'nems0.plots.api.nl_scatter'],
        'plot_fn_idx': 2,
        'prior': {'base': ('Exponential', {'beta': [0.1]}),
                  'amplitude': ('Exponential', {'beta': [2.0]}),
                  'shift': ('Normal', {'mean': [1.0], 'sd': [1.0]}),
                  'kappa': ('Exponential', {'beta': [0.1]})}
        }

    return template


@xmodule('tanh')
def tanh(kw):
    '''
    Generate and register modulespec for tanh module.

    Parameters
    ----------
    kw : str
        Expected format: r'^tanh\.?(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^tanh\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_dims = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got TypeError while parsing tanh keyword,\n"
                         "make sure keyword is of the form: \n"
                         "tanh.{n_dims} \n"
                         "keyword given: %s" % kw)

    zeros = np.zeros([n_dims, 1]) if n_dims > 1 else np.array([0])
    ones = np.ones([n_dims, 1]) if n_dims > 1 else np.array([1])
    base_mean = zeros
    base_sd = ones
    amp_mean = zeros + 0.2
    amp_sd = ones
    shift_mean = zeros
    shift_sd = ones
    kappa_mean = zeros
    kappa_sd = zeros + 0.1

    template = {
        'fn': 'nems0.modules.nonlinearity.tanh',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.pred_resp',
                     'nems0.plots.api.before_and_after',
                     'nems0.plots.api.nl_scatter'],
        'plot_fn_idx': 2,
        'prior': {'base': ('Normal', {'mean': base_mean, 'sd': base_sd}),
                  'amplitude': ('Normal', {'mean': amp_mean, 'sd': amp_sd}),
                  'shift': ('Normal', {'mean': shift_mean, 'sd': shift_sd}),
                  'kappa': ('Normal', {'mean': kappa_mean, 'sd': kappa_sd})}
    }

    return template


@xmodule('dlog')
def dlog(kw):
    '''
    Generate and register modulespec for dlog module.

    Parameters
    ----------
    kw : str
        Expected format: r'^dlog(\.n\d{1,})?$'

    Options
    -------
    nN : Apply normalization for the given number of channels, N.
         E.g. `n18` or `n2`
    f : fixed log, offset=-1
    cN : Apply separate offset to each of N input channels

    Note
    ----
    The normalization option for this function differs from the typical
    standalone 'n' because the number of channels is only needed if
    normalization is used - otherwise, only 'dlog' is required since the
    number of channels would be redundant information.
    '''
    options = kw.split(".")
    chans = 1
    nchans = 0

    offset = ('f' in options)
    for op in options:
        if op.startswith('c'):
            chans = int(op[1:])
        elif op.startswith('n'):
            nchans = int(op[1:])

    template = {
        'fn': 'nems0.modules.nonlinearity.dlog',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'chans': chans,},
        'tf_layer': 'nems0.tf.layers.Dlog',
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.pred_resp',
                     'nems0.plots.api.before_and_after',
                     'nems0.plots.api.spectrogram',
                     'nems0.plots.api.mod_output_all'],
        'plot_fn_idx': 2,
    }

    if nchans:
        d = np.zeros([nchans, 1])
        g = np.ones([nchans, 1])
        template['norm'] = {'type': 'minmax', 'recalc': 0, 'd': d, 'g': g}

    if offset:
        template['fn_kwargs']['var_offset'] = False
        template['fn_kwargs']['offset'] = np.array([[-1]])
        template['prior'] = {}
    else:
        template['prior'] = {'offset': ('Normal', {
                'mean': np.zeros((chans, 1)),
                'sd': np.full((chans, 1), 0.05)})}

    return template


@xmodule('relu')
def relu(kw):
    '''
    Generate and register modulespec for nonlinearity.relu module.

    Parameters
    ----------
    kw : str
        Expected format: r'^relu(\.n\d{1,})?$'

    Options
    -------
    N : Apply threshold for the given number of channels, N.
         E.g. `n18` or `n2`
    f : fixed threshold of zero
    b : add baseline (spont rate) after threshold

    '''
    options = kw.split(".")[1:]
    chans = 1
    var_offset = True
    fname = 'nems0.modules.nonlinearity.relu'
    baseline = False

    for op in options:
        if op == 'f':
            var_offset = False
        elif op == 'b':
            baseline=True
            fname = 'nems0.modules.nonlinearity.relub'
        else:
            chans = int(op)

    template = {
        'fn': fname,
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'tf_layer': 'nems0.tf.layers.Relu',
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.pred_resp',
                     'nems0.plots.api.resp_spectrogram',
                     'nems0.plots.api.pred_spectrogram',
                     'nems0.plots.api.before_and_after',
                     'nems0.plots.api.perf_per_cell'],
        'plot_fn_idx': 2
    }

    if var_offset is False:
        template['fn_kwargs']['offset'] = np.array([[0]])
        template['fn_kwargs']['var_offset'] = False
    else:
        template['prior'] = {'offset': ('Normal', {
                'mean': np.zeros((chans, 1))-0.1,
                'sd': np.ones((chans, 1))/np.sqrt(chans)})}
    if baseline:
        template['prior']['baseline'] = ('Normal', {
                'mean': np.zeros((chans, 1)),
                'sd': np.ones((chans, 1))*2})

    return template


@xmodule('relsat')
def relsat(kw):
    '''
    Saturated rectifier, similar to relu but uses sigmoidal parameters.

    '''

    # Default mean initialization is just relu with a truncation at y=2
    base_prior = ('Exponential', {'beta': np.array([0])})
    amplitude_prior = ('Exponential', {'beta': np.array([2])})
    shift_prior = ('Normal', {'mean': np.array([0]), 'sd': np.array([0.5])})
    kappa_prior = ('Exponential', {'beta': np.array([1])})

    template = {
        'fn': 'nems0.modules.nonlinearity.saturated_rectifier',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'plot_fns': ['nems0.plots.api.mod_output',
                     'nems0.plots.api.spectrogram_output',
                     'nems0.plots.api.pred_resp',
                     'nems0.plots.api.before_and_after'],
        'prior': {'base': base_prior,
                  'amplitude': amplitude_prior,
                  'shift': shift_prior,
                  'kappa': kappa_prior},
        'plot_fn_idx': 2
    }

    return template


@xmodule('stategain')
def stategain(kw):
    '''
    Generate and register modulespec for the state_dc_gain module.

    Parameters
    ----------
    kw : str
        Expected format: r'^stategain\.?(\d{1,})x(\d{1,})$'
        e.g., "stategain.SxR" :
            S : number of state channels (required)
            R : number of channels to modulate (default = 1)
            (S, R can be left as variables and often figured out during modelspec generation in from_keywords)
    Options
    -------
        .g -- gain only (no dc term)
        .d -- dc only
        .s -- separate dc term for spont period
        .lv -- concatenate latent variable "lv" onto 'state'. Will need to specify accurate S for this to work.
        .xN,M -- exclude state channels number N, M etc.
        .oN -- fix gainoffset to N, initialize gain to 0.
        .bN:M -- set bounds on gain from N to M. N can use d for the decimal place, ex .b0d001:5 means set bounds to [0.001, 5]
                 This is only coded up for setting bonds on the gain. Add stuff if you need to set bounds for dc.
    None
    '''
    options = escaped_split(kw, '.')
    in_out_pattern = re.compile(r'^(\d{1,})x(\d{1,})$')

    try:
        parsed = re.match(in_out_pattern, options[1])
        if parsed is None:
            # backward compatible parsing if R not specified
            n_vars = int(options[1])
            n_chans = 1

        else:
            n_vars = int(parsed.group(1))
            if len(parsed.groups())>1:
                n_chans = int(parsed.group(2))
            else:
                n_chans = 1

    except TypeError:
        raise ValueError("Got TypeError when parsing stategain keyword.\n"
                         "Make sure keyword is of the form: \n"
                         "stategain.{n_variables} or stategain.{n_variables}x{n_chans} \n"
                         "keyword given: %s" % kw)

    gain_only=('g' in options[2:])
    include_spont=('s' in options[2:]) # separate offset for spont than during evoked
    dc_only=('d' in options[2:])
    per_channel=('per' in options[2:])
    if gain_only and dc_only:
        raise ValueError('Cannot have both gain only and dc only.')

    fix_across_channels = 0
    if 'c1' in options[2:]:
        fix_across_channels = 1
    elif 'c2' in options[2:]:
        fix_across_channels = 2
    state = 'state'
    if 'lv' in options[2:]:
        state = 'lv'

    zeros = np.zeros([n_chans, n_vars])
    ones = np.ones([n_chans, n_vars])
    g_mean = zeros.copy()
    g_mean[:, 0] = 1
    g_sd = ones.copy() / 20
    d_mean = zeros
    d_sd = ones

    #If .o# is passed, fix gainoffset to #, initialize gain to 0.
    # y = (np.matmul(g, rec[s]._data) + offset) * x so .g1 will by initialize with no state-dependence
    gainoffset = 0
    bounds=None
    exclude_chans = None

    for op in options[2:]:
        if op.startswith('o'):
            num = op[1:].replace('\\', '')
            gainoffset = float(num)
            g_mean[:, 0] = 0
        elif op.startswith('b'):
            bounds_in = op[1:].replace('\\', '').replace('d', '.')
            bounds_in = bounds_in.split(':')
            bounds = tuple(np.full_like(g_mean,float(bound) - gainoffset) for bound in bounds_in)
        elif op.startswith("x"):
            exclude_chans = [int(x) for x in op[1:].split(',')]

    plot_fns = ['nems0.plots.api.mod_output',
                'nems0.plots.api.spectrogram_output',
                'nems0.plots.api.before_and_after',
                'nems0.plots.api.pred_resp',
                'nems0.plots.api.state_vars_timeseries',
                'nems0.plots.api.state_vars_psth_all',
                'nems0.plots.api.state_gain_plot',
                'nems0.plots.api.state_gain_parameters']
    if dc_only:
        template = {
            'fn': 'nems0.modules.state.state_dc_gain',
            'fn_kwargs': {'i': 'pred', 'o': 'pred', 's': state, 'chans': n_vars, 'n_inputs': n_chans, 'g': g_mean,
                          'state_type': 'dc_only',
                          'exclude_chans': exclude_chans},
            'plot_fns': plot_fns,
            'plot_fn_idx': 5,
            'prior': {'d': ('Normal', {'mean': d_mean, 'sd': d_sd})}
        }
    elif gain_only:
        if bounds is None:
            bounds = {'g': (g_mean - 0.5, g_mean + 0.5)}
        else:
            bounds = {'g': bounds}
        bounds['g'][0][1:, :fix_across_channels] = g_mean[1:, :fix_across_channels]
        bounds['g'][1][1:, :fix_across_channels] = g_mean[1:, :fix_across_channels]
        template = {
            'fn': 'nems0.modules.state.state_gain',
            'fn_kwargs': {'i': 'pred', 'o': 'pred', 's': state, 'chans': n_vars, 'n_inputs': n_chans,
                          'state_type':'gain_only', 'gainoffset':gainoffset,
                          'exclude_chans': exclude_chans},
            'plot_fns': plot_fns,
            'plot_fn_idx': 6,
            'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd})},
            'bounds': bounds
        }
    elif include_spont:
        template = {
           'fn': 'nems0.modules.state.state_sp_dc_gain',
            'fn_kwargs': {'i': 'pred', 'o': 'pred', 's': state,
                          'exclude_chans': exclude_chans},
            'plot_fns': plot_fns,
            'plot_fn_idx': 5,
            'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd}),
                      'd': ('Normal', {'mean': d_mean, 'sd': d_sd}),
                      'sp': ('Normal', {'mean': d_mean, 'sd': d_sd})}
            }
    else:
        if g_mean.shape[0]>2:
            plot_fn_idx=6
        else:
            plot_fn_idx=5

        template = {
            'fn': 'nems0.modules.state.state_dc_gain',
            'fn_kwargs': {'i': 'pred', 'o': 'pred', 's': state, 'chans': n_vars, 'n_inputs': n_chans,
                          'state_type': 'both', 'exclude_chans': exclude_chans, 'per_channel': per_channel},
            # chans/vars backwards for compatibility with tf layer
            'plot_fns': plot_fns,
            'plot_fn_idx': plot_fn_idx,
            'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd}),
                      'd': ('Normal', {'mean': d_mean, 'sd': d_sd})}
            }
    if fix_across_channels:
        template['fn_kwargs'].update({'fix_across_channels': fix_across_channels})

    template['tf_layer'] = 'nems0.tf.layers.StateDCGain'
    return template


@xmodule('stateseg')
def stateseg(kw):
    '''
    Generate and register modulespec for the state_segmented module.

    Parameters
    ----------
    kw : str
        Expected format: r'^stateseg\.?(\d{1,})x(\d{1,})$'
        e.g., "stateseg.SxR" :
            S : number of state channels (required)
            R : number of channels to modulate (default = 1)

    Options
    -------
    None

    TODO: set initial conditions for segmented linear model

    '''
    # parse the keyword
    pattern = re.compile(r'^stateseg\.?(\d{1,})x(\d{1,})$')
    parsed = re.match(pattern, kw)
    if parsed is None:
        # backward compatible parsing if R not specified
        pattern = re.compile(r'^stateseg\.?(\d{1,})$')
        parsed = re.match(pattern, kw)
    try:
        n_vars = int(parsed.group(1))
        if len(parsed.groups())>1:
            n_chans = int(parsed.group(2))
        else:
            n_chans = 1
    except TypeError:
        raise ValueError("Got TypeError when parsing stateseg keyword.\n"
                         "Make sure keyword is of the form: \n"
                         "stategain.{n_variables} \n"
                         "keyword given: %s" % kw)

    # specify initial conditions
    zeros = np.zeros([n_chans, n_vars])
    ones = np.ones([n_chans, n_vars])
    g_mean = zeros.copy()
    g_mean[:, 0] = 1
    g_sd = ones.copy()
    d_mean = zeros
    d_sd = ones

    template = {
        'fn': 'nems0.modules.state.state_segmented',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd}),
                  'd': ('Normal', {'mean': d_mean, 'sd': d_sd})}
        }

    return template


@xmodule('sw')
def sw(kw):
    '''
    Generate and register modulespec for the state.state_weight module.

    Parameters
    ----------
    kw : str
        Expected format: r'^sw\.?(\d{1,})x(\d{1,})$'
        e.g., "stategain.SxR" :
            S : number of state channels (required)
            R : number of channels to modulate (default = 2)

    TODO: support for more than one output channel? (filterbank)

    Options
    -------
    None
    '''
    pattern = re.compile(r'^sw\.?(\d{1,})x(\d{1,})$')
    parsed = re.match(pattern, kw)
    if parsed is None:
        # backward compatible parsing if R not specified
        pattern = re.compile(r'^sw\.?(\d{1,})$')
        parsed = re.match(pattern, kw)
    try:
        n_vars = int(parsed.group(1))
        if len(parsed.groups())>1:
            n_chans = int(parsed.group(2))
        else:
            n_chans = 1
    except TypeError:
        raise ValueError("Got TypeError when parsing stategain keyword.\n"
                         "Make sure keyword is of the form: \n"
                         "sw.{n_variables} \n"
                         "keyword given: %s" % kw)

    zeros = np.zeros([n_chans, n_vars])
    ones = np.ones([n_chans, n_vars])
    g_mean = zeros.copy()
    g_mean[:, 0] = 0.5
    g_sd = ones.copy()
    d_mean = np.zeros([1, n_vars])
    d_sd = np.ones([1, n_vars])

    template = {
        'fn': 'nems0.modules.state.state_weight',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd}),
                  'd': ('Normal', {'mean': d_mean, 'sd': d_sd})}
        }

    return template


@xmodule('rep')
def rep(kw):
    '''
    Generate and register modulespec for replicate_channels module.

    Parameters
    ----------
    kw : str
        Expected format: r'^rep\.?(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^rep\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_reps = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got TypeError while parsing rep keyword. \n"
                         "Make sure keyword is of the form: \n"
                         "rep.{n_reps} \n"
                         "keyword given: %s" % kw)

    template = {
        'fn': 'nems0.modules.signal_mod.replicate_channels',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'repcount': n_reps},
        'phi': {}
        }

    return template


@xmodule('mrg')
def mrg(kw):
    '''
    Generate and register modulespec for merge_channels module.

    Parameters
    ----------
    kw : str
        Expected format: mrg

    Options
    -------
    None

    Note
    ----
    This keyword function performs no parsing. It always returns
    the same modulespec, so any keyword beginning with 'mrg.' is equivalent.
    '''
    template = {
        'fn': 'nems0.modules.signal_mod.merge_channels',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'phi': {}
        }
    return template

@xmodule()
def conv2d(kw):
    # TODO: choose how to initialize weights
    ops = kw.split('.')
    filters = int(ops[1])
    kernel_size = [int(dim) for dim in ops[2].split('x')]  # second op hard-coded as kernel shape
    activation = 'relu'
    layer_count = 1
    flatten = False
    dropout_rate = None
    for op in ops[3:]:
        if op.startswith('actX'):
            activation = op[4:]
            if activation == 'none':
                activation = None
        elif op.startswith('rep'):
            layer_count = int(op[3:])
        elif op == 'flat':
            flatten = True
        elif op.startswith('dr'):
            dropout_rate = float(op[2:])/100

    template = {
        'fn': 'nems0.tf_only.Conv2D_NEMS',   # not a real path, flag for ms.evaluate to use evaluate_tf()
        'tf_layer': 'nems0.tf.layers.Conv2D_NEMS',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'activation': activation,
                      'filters': filters,
                      'kernel_size': kernel_size,
                      'padding': 'same'},
        'phi': {}
    }
    templates = [template]*layer_count
    if flatten:
        flatten_template = {
            'fn': 'nems0.tf_only.FlattenChannels',
            'tf_layer': 'nems0.tf.layers.FlattenChannels',
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred'},
            'phi': {}
        }
        templates.append(flatten_template)

    if dropout_rate is not None:
        dropout_template = {
            'fn': 'nems0.tf_only.Dropout_NEMS',    # not a real path
            'tf_layer': 'nems0.tf.layers.Dropout_NEMS',
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred',
                          'rate': dropout_rate},
            'phi': {}
        }
        # Add dropout after every conv layer
        dropout_templates = [dropout_template]*layer_count
        templates = [t for pair in zip(templates, dropout_templates) for t in pair]

    return templates

@xmodule()
def dense(kw):
    # TODO: choose how to initialize weights
    ops = kw.split('.')
    units = ops[1].split('x')  # first option hard-coded as number of units in each layer
    activation = 'relu'
    for op in ops[2:]:
        if op.startswith('actX'):
            activation = op[4:]
            if activation == 'none':
                activation = None

    templates = []
    for u in units:
        template = {
            'fn': 'nems0.tf_only.Dense_NEMS',    # not a real path
            'tf_layer': 'nems0.tf.layers.Dense_NEMS',
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred',
                          'activation': activation,
                          'units': int(u)},
            'phi': {}
            }
        templates.append(template)

    return templates

@xmodule()
def wcn(kw):
    # TODO: choose how to initialize weights
    ops = kw.split('.')
    units = int(ops[1]) # first option hard-coded as number of units
    for op in ops[2:]:  # just a reminder to skip the first two if options are added later
        pass

    template = {
        'fn': 'nems0.tf_only.WeightChannelsNew',    # not a real path
        'tf_layer': 'nems0.tf.layers.WeightChannelsNew',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'units': units,
                      'initializer': 'random_normal'},
        'phi': {}
        }

    return template


@xmodule()
def drop(kw):
    ops = kw.split('.')
    # first option hard-coded to be dropout rate as percentage, e.x. drop.50  for rate = 0.50
    rate = float(ops[1])/100
    template = {
        'fn': 'nems0.tf_only.Dropout_NEMS',    # not a real path
        'tf_layer': 'nems0.tf.layers.Dropout_NEMS',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'rate': rate},
        'phi': {}
        }

    return template


@xmodule()
def dns(kw):
    '''A NEMS version of multiple dense layers, repeated wc-relu  with the same number of units.

    Note: since every layer uses the same size, the output of the previous layer must match the number of units
    used for this dense layer, ex:  kw1-kw2-wc.100x10-dns.10-dns.10-wc.10xR-

    '''
    options = kw.split('.')
    units = int(options[1])
    reps = 1
    for op in options:
        if op.startswith('rep'):
            reps = int(op[3:])

    wc_template = wc(f'wc.{units}x{units}.g')
    relu_template = relu(f'relu.{units}.f')

    return [wc_template, relu_template]*reps
