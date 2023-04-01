import logging
import re

import numpy as np

from nems0.utils import escaped_split, keyword_extract_options
from nems0.registry import xform, xmodule

log = logging.getLogger(__name__)


# TOOD: Maybe these should go in fitters instead?
#       Not really initializers, but really fitters either.
# move to same place as sev? -- SVD
# TODO: Maybe can keep splitep and avgep as one thing?
#       Would they ever be done separately?
@xform()
def timesplit(kw):
    frac = int(kw.split('.')[1][1:])*0.1
    return [['nems0.xforms.split_at_time', {'fraction': frac}]]


@xform()
def splitep(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    xfspec = [['nems0.xforms.split_by_occurrence_counts',
               {'epoch_regex': epoch_regex}]]
    return xfspec


@xform()
def avgep(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    return [['nems0.xforms.average_away_stim_occurrences',
             {'epoch_regex': epoch_regex}]]


#moved to xforms.py
"""
def sev(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    xfspec = [['nems0.xforms.split_by_occurrence_counts',
               {'epoch_regex': epoch_regex}],
        ['nems0.xforms.average_away_stim_occurrences',
         {'epoch_regex': epoch_regex}]]
    return xfspec
"""

@xform()
def aev(kw):
    xfspec= [['nems0.xforms.use_all_data_for_est_and_val',
                {}]]
    return xfspec


@xform()
def sevst(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    xfspec = [['nems0.xforms.split_by_occurrence_counts',
               {'epoch_regex': epoch_regex}]]
    return xfspec

"""moved to xforms.py"""
"""
def tev(kw):
    ops = kw.split('.')[1:]

    valfrac = 0.1
    for op in ops:
        if op.startswith("vv"):
            valfrac=int(op[2:]) / 1000
        elif op.startswith("v"):
            valfrac=int(op[1:]) / 100

    xfspec = [['nems0.xforms.split_at_time', {'valfrac': valfrac}]]

    return xfspec
"""

@xform()
def jk(kw):
    ops = kw.split('.')[1:]
    jk_kwargs = {}
    do_split = False
    keep_only = 0
    log.info("Setting up N-fold fitting...")
    jk_kwargs['allow_partial_epochs'] = False

    for op in ops:
        if op.startswith('nf'):
            jk_kwargs['njacks'] = int(op[2:])
        elif op == 'stim':
            jk_kwargs['epoch_name'] = "^STIM_"
        elif op == 'tr':
            jk_kwargs['epoch_name'] = "TRIAL"
        elif op == 'm':
            do_split = True
        elif op == 'p':
            jk_kwargs['allow_partial_epochs'] = True
        elif op.startswith('ep'):
            pattern = re.compile(r'^ep(\w{1,})$')
            jk_kwargs['epoch_name'] = re.match(pattern, op).group(1)
        elif op.startswith('o'):
            if len(op)>1:
                keep_only = int(op[1:])
            else:
                keep_only = 1
        elif op.startswith('bt'):
            # jackknife by time
            jk_kwargs['by_time'] = True

    if do_split:
        xfspec = [['nems0.xforms.split_for_jackknife', jk_kwargs]]
    else:
        xfspec = [['nems0.xforms.mask_for_jackknife', jk_kwargs]]
    if keep_only == 1:
        xfspec.append(['nems0.xforms.jack_subset', {'keep_only': keep_only}])
    elif keep_only > 1:
        xfspec.append(['nems0.xforms.jack_subset', {'keep_only': keep_only}])
        xfspec.append(['nems0.xforms.jackknifed_fit', {}])
    else:
        xfspec.append(['nems0.xforms.jackknifed_fit', {}])

    return xfspec


@xform()
def rand(kw):
    ops = kw.split('.')[1:]
    nt_kwargs = {}

    for op in ops:
        if op.startswith('nt'):
            nt_kwargs['ntimes'] = int(op[2:])
        elif op.startswith('S'):
            nt_kwargs['subset'] = [int(i) for i in op[1:].split(',')]

    return [['nems0.xforms.random_sample_fit', nt_kwargs]]


@xform()
def norm(kw):
    """
    Normalize stim and response before splitting/fitting to support
    fitters that can't deal with big variations in values

    default is to normalize minmax (norm.mm) to fall in the range 0 to 1 (keep values
    positive to support log compression)
    norm.ms will normalize to (mean=0, std=1)
    """
    ops = kw.split('.')[1:]
    norm_method = 'minmax'
    log_compress = 'None'
    stim_only = ('s' in ops)
    resp_only = ('r' in ops)

    for op in ops:
        if op == 'ms':
            norm_method = 'meanstd'
        elif op == 'mm':
            norm_method = 'minmax'
        elif op == 'sp':
            norm_method = 'spont'
            resp_only = ('r' in ops)  # default !
        elif op == 'sqrt':
            norm_method = 'sqrt'
            resp_only = True
            stim_only = False
        elif op.startswith('ln'):
            log_compress = -int(op[2:])
        elif op.startswith('l'):
            log_compress = int(op[1:])
            if log_compress > 10:
                log_compress = np.array([log_compress/10])

    if stim_only:
        return [['nems0.xforms.normalize_sig', {'sig': 'stim', 'norm_method': norm_method,
                                               'log_compress': log_compress}],
                ]
    elif resp_only:
        return [['nems0.xforms.normalize_sig', {'sig': 'resp', 'norm_method': norm_method}]]
    else:
        return [['nems0.xforms.normalize_sig', {'sig': 'stim', 'norm_method': norm_method,
                'log_compress': log_compress}],
                ['nems0.xforms.normalize_sig', {'sig': 'resp', 'norm_method': norm_method}],
               ]

@xform()
def lvnoise(kw):

    ops = kw.split('.')[1:]
    rep_count = 1
    for op in ops:
        if op.startswith('r'):
            rep_count=int(op[1:])
    
    return [['nems0.preprocessing.add_noise_signal', {'noise_name': "indep", 'rep_count': rep_count}],
            ['nems0.preprocessing.add_noise_signal', {'noise_name': "lv", 'ref_signal': "state"}]]

@xform()
def shuf(load_key):
    options = load_key.split('.')[1:]
    shuf_sigs=[]
    shuf_recs = ['est','val']
    for op in options:
        if op == 'st':
            shuf_sigs.append('state')
        else:
            raise ValueError('Unknown shuf keyword option {}'.format(op))
    return [['nems0.preprocessing.shuffle',{'sigs':shuf_sigs,'recs':shuf_recs}]]