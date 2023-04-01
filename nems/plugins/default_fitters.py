"""
nems0.plugins.default_fitters.py
"""

import logging
import re

from nems0.utils import escaped_split, keyword_extract_options
from nems0.registry import xform, xmodule

log = logging.getLogger(__name__)

# TODO: Create and expand documentation.
# TODO: set up alias'ing function similar to one for loaders.

@xform()
def mt(fitkey):
    '''
    Set fitting metric to nmse_shrinkage for keywords that come after.

    TODO: Any other metrics as added options?

    '''
    ops = fitkey.split('.')[1:]
    if 'shr' in ops:
        metric = 'nmse_shrink'
    else:
        # Redundant for now since shr is the only option, but will likely
        # add more later.
        metric = 'nmse_shrink'
    return [['nems0.xforms.use_metric', {'metric': metric}]]


@xform()
def pred(fitkey):
    '''
    Evaluate model prediction. Added by default in xform_helper.
    '''
    return [['nems0.xforms.predict', {}]]


@xform()
def stats(fitkey):
    '''
    Add summary statistics to modelspec(s). Added by default in xform_helper.
    '''
    options = fitkey.split('.')[1:]
    fn = 'standard_correlation'
    for op in options:
        if op == 'pm':
            fn = 'correlation_per_model'

    return [['nems0.xforms.add_summary_statistics', {'fn': fn}]]


@xform()
def best(fitkey):
    '''
    Collapse modelspecs to singleton list with only the "best" modelspec.
    '''
    options = escaped_split(fitkey, '.')[1:]
    metakey = 'r_test'
    comparison = 'greatest'

    for op in options:
        if op == '<':
            comparison = 'least'
        elif op == '>':
            comparison = 'greatest'
        else:
            # Assume it's the name of a metakey, and remove any escapes
            metakey = op.replace('\\', '')

    return [['nems0.xforms.only_best_modelspec', {'metakey': metakey,
                                                 'comparison': comparison}]]


@xform()
def sort(fitkey):
    '''
    Sorts modelspecs by specified meta entry in either descending or
    ascending order.
    '''
    ops = escaped_split(fitkey, '.')[1:]
    metakey = 'r_test'
    order = 'descending'

    for op in ops:
        if op in ['a', 'asc', 'ascending']:
            order = op
        elif op in ['d', 'desc', 'descending']:
            order = op
        else:
            # Assume it's the name of a metakey, and remove any escapes
            metakey = op.replace('\\', '')

    return [['nems0.xforms.sort_modelspecs', {'metakey': metakey,
                                             'order': order}]]

@xform()
def lite(fitkey):
    '''
    Fit a nems-lite model

    Parameters
    ----------
    fitkey : str
        Expected format: lite.options
    TODO: options! - mimic basic kw

    '''

    options = _parse_options(fitkey)
    #max_iter, tolerance, fitter, choose_best, rand_count = _parse_basic(options)
    rand_count = options.get('rand_count', 1)
    choose_best = options.get('choose_best', False)

    del options['rand_count']
    del options['choose_best']
    del options['fitter']

    xfspec = []
    if rand_count>1:
        xfspec.append(['nems0.initializers.rand_phi', {'rand_count': rand_count}])
    xfspec.append(['nems0.xforms.fit_lite', options])
    if choose_best:
        xfspec.append(['nems0.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

    return xfspec

@xform()
def basic(fitkey):
    '''
    Perform a fit_basic analysis on a model.

    Parameters
    ----------
    fitkey : str
        Expected format: basic.<fitter>.<jackknifing>.<metric>.<misc>
        Example: basic.cd.nf10.shr.mi1000
        Example translation:
            Use fit_basic with coordinate_descent, nfold fitting with 10 folds,
            with nmse_shrinkage as the metric and 1000 maximum iterations

    Options
    -------
    cd : Use coordinate_descent for fitting (default is scipy_minimize)
    miN : Set maximum iterations to N, where N is any positive integer.
    tN : Set tolerance to 10**-N, where N is any positive integer.

    '''

    xfspec = []

    options = _parse_options(fitkey)
    #max_iter, tolerance, fitter, choose_best, rand_count = _parse_basic(options)
    rand_count = options.get('rand_count', 1)
    choose_best = options.get('choose_best', False)

    del options['rand_count']
    del options['choose_best']

    xfspec = []
    if rand_count>1:
        xfspec.append(['nems0.initializers.rand_phi', {'rand_count': rand_count}])
    xfspec.append(['nems0.xforms.fit_basic', options])
    if choose_best:
        xfspec.append(['nems0.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

    return xfspec


@xform()
def nrc(fitkey):
    '''
    Use normalized reverse correlation to fit phi. Right now, pretty dumb.
    Expects two modules (fir and lvl). Will fit both simultaneously. Can stack
    this with fit basic in order to initialize a full rank model with the reverse
    correlation solution.
    '''
    xfspec = []

    #options = _extract_options(fitkey)
    #max_iter, tolerance, fitter = _parse_basic(options)
    xfspec = [['nems0.xforms.reverse_correlation', {}]]

    return xfspec


@xform()
def newtf(fitkey):
    """New tf fitter.

    TODO
    """
    parm_dict = {}

    use_modelspec_init = False
    cost_function = 'squared_error'
    early_stopping_steps = 5
    early_stopping_tolerance = 5e-4
    early_stopping_val_split = 0.0
    learning_rate = 1e-4
    variable_learning_rate = False
    batch_size = None
    seed = 0
    initializer = 'random_normal'
    rand_count = 0
    pick_best = False
    generate_predictions_for_each_initalization_condition = False
    epoch_name = "REFERENCE"
    freeze_layers = None
    use_tensorboard = False
    kernel_regularizer = None
    fit_fn = 'nems0.tf.cnnlink_new.fit_tf'
    options = _extract_options(fitkey)
    iters_per_loop = 100

    for op in options:
        if op.startswith('slicen'):
            parm_dict['fit_slices_start'] = -int(op[6:])
        elif op.startswith('slice'):
            parm_dict['fit_slices_start'] = int(op[5:])
        elif op[:1] == 'i' and op != 'iso':
            if len(op[1:]) == 0:
                parm_dict['max_iter'] = int(1e10)  # just a really large number
            else:
                parm_dict['max_iter'] = int(op[1:])
        elif op == 'nlis':
            parm_dict['nl_init'] = "scipy"
        elif op.startswith('mc'):
            # multi-cell_index
            fit_fn = 'nems0.tf.cnnlink_new.fit_tf_iterate'
            if len(op[2:]) > 0:
                iters_per_loop = int(op[2:])
        elif op == 'mxr':
            parm_dict['extra_tail_fit'] = 'pre_reset'
        elif op == 'mxpr':
            parm_dict['extra_tail_fit'] = 'pre'
        elif op == 'mxpo':
            parm_dict['extra_tail_fit'] = 'post'
        elif op == 'prop':
            parm_dict['proportional_iter'] = True
        elif op == 'unif':
            parm_dict['proportional_iter'] = False
        elif op == 'mxpo':
            parm_dict['extra_tail_fit'] = 'post'
        elif op[:1] == 'n':
            use_modelspec_init = True
        elif op == 'b':
            pick_best = True
        elif op.startswith('rbp'):
            pick_best = True
            generate_predictions_for_each_initalization_condition = True
            if len(op) == 3:
                rand_count = 10
            else:
                rand_count = int(op[3:])
        elif op.startswith('rb'):
            pick_best = True
            if len(op) == 2:
                rand_count = 10
            else:
                rand_count = int(op[2:])
        elif op.startswith('r'):
            if len(op) == 1:
                rand_count = 10
            else:
                rand_count = int(op[1:])
        elif op.startswith('lr'):
            learning_rate = op[2:]
            if 'e' in learning_rate:
                base, exponent = learning_rate.split('e')
                learning_rate = int(base) * 10 ** -int(exponent)
            else:
                learning_rate = int(learning_rate)
        elif op.startswith('lf'):
            loss_type = op[2:]
            if loss_type == 'se':
                cost_function = 'squared_error'
            if loss_type == 'p':
                cost_function = 'poisson'
            if loss_type == 'nmse':
                cost_function = 'nmse'
            if loss_type == 'nmsepc':
                cost_function = 'nmse_pc'
            if loss_type == 'nmses':
                cost_function = 'nmse_shrinkage'
        elif op == 'v':
            variable_learning_rate = True
        elif op.startswith('et'):
            exp = op[2:].replace('d', '.')
            early_stopping_tolerance = 10 ** -float(exp)
        elif op.startswith('bs'):
            batch_size = int(op[2:])
        elif op[:1] == 'd':
            initializer = op[1:]
            if initializer == 'gu':
                initializer = 'glorot_uniform'
            elif initializer == 'heu':
                initializer = 'he_uniform'
            elif initializer == 'tn':
                initializer = 'truncated_normal'
            elif initializer == 'ln':
                initializer = 'lecun_normal'
        elif op == 'cont':
            epoch_name = ""
        elif op.startswith('FL'):
            if ':' in op:
                # ex: FL0:5  would be freeze_layers = [0,1,2,3,4]
                lower, upper = [int(i) for i in op[2:].split(':')]
                freeze_layers = list(range(lower, upper))
            else:
                # ex: FL2x6x9  would be freeze_layers = [2, 6, 9]
                freeze_layers = [int(i) for i in op[2:].split('x')]

        elif op.startswith('es'):
            early_stopping_val_split = float(op[2:])/100

        elif op == 'tb':
            use_tensorboard = True

        elif op.lower().startswith('l2'):
            kernel_regularizer = op.lower()

    xfspec = []
    if rand_count > 0:
        xfspec.append(['nems0.initializers.rand_phi', {'rand_count': rand_count}])

    parm_dict.update({
        'use_modelspec_init': use_modelspec_init,
        'cost_function': cost_function,
        'fit_function': fit_fn,
        'early_stopping_steps': early_stopping_steps,
        'early_stopping_tolerance': early_stopping_tolerance,
        'early_stopping_val_split': early_stopping_val_split,
        'learning_rate': learning_rate,
        'variable_learning_rate': variable_learning_rate,
        'batch_size': batch_size,
        'initializer': initializer,
        'seed': seed,
        'epoch_name': epoch_name,
        'use_tensorboard': use_tensorboard,
        'kernel_regularizer': kernel_regularizer,
        'iters_per_loop': iters_per_loop,
        })
    if freeze_layers is not None:
        parm_dict['freeze_layers'] = freeze_layers

    xfspec.append(['nems0.xforms.fit_wrapper', parm_dict])

    if generate_predictions_for_each_initalization_condition:
        xfspec.append(['nems0.analysis.test_prediction.predict_and_summarize_for_all_modelspec', {}])
    if pick_best:
        xfspec.append(['nems0.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

    return xfspec


@xform()
def tfinit(fitkey):
    """New tf init.

    TODO
    """
    xfspec = newtf(fitkey)
    idx = [xf[0] for xf in xfspec].index('nems0.xforms.fit_wrapper')
    xfspec[idx][1]['fit_function'] = 'nems0.tf.cnnlink_new.fit_tf_init'

    options = _extract_options(fitkey)
    for op in options:
        if op == 'iso':
            xfspec[idx][1]['isolate_NL'] = True
        if op[:2] == 'xx':
            xfspec[idx][1]['up_to_idx'] = int(op[2:])

    return xfspec


@xform()
def tf(fitkey):
    '''
    Perform a Tensorflow fit, using Sam Norman-Haignere's CNN library

    Parameters
    ----------
    fitkey : str
        Expected format: tf.f<fitter>.i<max_iter>.s<start_conditions>.rb<count>.l<loss_function>
        Example: tf.fAdam.i1000.s20.rb10.lse
        Example translation:
            Use Adam fitter, max 1000 iterations, starting from 20 random
            initial condition, picking the best of 10 random fits, with squared error loss function

    Options
    -------
    f<fitter> : string specifying fitter, passed through to TF {'adam': 'Adam', 'gd': 'GradientDescent'}
    i<N> : Set maximum iterations to N, any positive integer.
    s<S> : Initialize with S random seeds, pick the best performer across
           the entire fit set.
    rb<N>: Picks the best of multiple random fits
    n    : Use modelspec initialized by NEMS
    l<S> : Specify the loss function {'se': 'squared_error', 'p': 'poisson', 'nmse': 'nmse', 'nmses': 'nmse_shrinkage'}
    e<N> : Specify the number of early stopping steps, i.e. the consecutive number of failed conditions before
           early stopping.
    et<N>: Specify the tolerance for early stopping. The value should be an integer, and the tolerance will be
           10 to the power of said negative integer.
    lr<N>e<N>: Specify the learning rate. The value should be two integers separated by the letter "e". The first
               integer will be multiplied by 10 raised to the negative second integer. Ex: lr5e2 = 0.05
    d<S> : String specifying the distribution with which to initialize the layers. Only used if .n not passed
    '''

    options = _extract_options(fitkey)

    max_iter = 10000
    fitter = 'GradientDescent'
    use_modelspec_init = False
    pick_best = False
    rand_count = 0
    loss_type = 'squared_error'
    early_stopping_steps = 5
    early_stopping_tolerance = 1e-5
    learning_rate = 0.01
    distr = 'norm'

    for op in options:
        if op[:1] == 'i':
            if len(op[1:]) == 0:
                max_iter = None
            else:
                max_iter = int(op[1:])
        elif op[:1] == 'f':
            fitter = op[1:]
            if fitter in ['adam', 'a']:
                fitter = 'Adam'
            elif fitter == 'gd':
                fitter = 'GradientDescent'
        elif op[:1] == 'n':
            use_modelspec_init = True
        elif op == 'b':
            pick_best = True
        elif op.startswith('rb'):
            if len(op) == 2:
                rand_count = 10
            else:
                rand_count = int(op[2:])
            pick_best = True
        elif op.startswith('lr'):
            learning_rate = op[2:]
            if 'e' in learning_rate:
                base, exponent = learning_rate.split('e')
                learning_rate = int(base) * 10 ** -int(exponent)
            else:
                learning_rate = int(learning_rate)
        elif op.startswith('r'):
            if len(op) == 1:
                rand_count = 10
            else:
                rand_count = int(op[1:])
        elif op[:1] == 'l':
            loss_type = op[1:]
            if loss_type == 'se':
                loss_type = 'squared_error'
            if loss_type == 'p':
                loss_type = 'poisson'
            if loss_type == 'nmse':
                loss_type = 'nmse'
            if loss_type == 'nmsepc':
                loss_type = 'nmse_pc'
            if loss_type == 'nmses':
                loss_type = 'nmse_shrinkage'
        elif op.startswith('et'):
            early_stopping_tolerance = 1 * 10 ** -int(op[2:])
        elif op[:1] == 'e':
            early_stopping_steps = int(op[1:])
        elif op[:1] == 'd':
            distr = op[1:]
            if distr == 'gu':
                distr = 'glorot_uniform'
            elif distr == 'heu':
                distr = 'he_uniform'

    xfspec = []
    if rand_count > 0:
        xfspec.append(['nems0.initializers.rand_phi', {'rand_count': rand_count}])

    xfspec.append(['nems0.xforms.fit_wrapper',
                   {
                       'max_iter': max_iter,
                       'use_modelspec_init': use_modelspec_init,
                       'optimizer': fitter,
                       'cost_function': loss_type,
                       'fit_function': 'nems0.tf.cnnlink.fit_tf',
                       'early_stopping_steps': early_stopping_steps,
                       'early_stopping_tolerance': early_stopping_tolerance,
                       'learning_rate': learning_rate,
                       'distr': distr,
                   }])

    if pick_best:
        xfspec.append(['nems0.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

    return xfspec


@xform()
def init(kw):
    '''
    Initialize modelspecs in an attempt to avoid getting stuck in
    local minima.
    Written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)

    Parameter
    ---------
    kw : string
        A string of the form: init.option1.option2...

    Options
    -------
    tN : Set tolerance to 10**-N, where N is any positive integer.
    st : Remove state replication/merging before initializing.
    psth : Initialize by fitting to 'psth' intead of 'resp' (default)
    nlN : Initialize nonlinearity with version N
        For dexp, options are {1,2} (default is 2),
            pre 11/29/18 models were fit with v1
            1: amp = np.nanstd(resp) * 3
               kappa = np.log(2 / (np.max(pred) - np.min(pred) + 1))
            2:
               amp = resp[pred>np.percentile(pred,90)].mean()
               kappa = np.log(2 / (np.std(pred)*3))
        For other nonlinearities, mode is not specified yet
    L2f : normalize fir (default false)
    .rN : initialize with N random phis drawn from priors (via init.rand_phi),
          default N=10
    .rbN : initialize with N random phis drawn from priors (via init.rand_phi),
           and pick best mse_fit, default N=10
    .iN : include module N. ie, fit phi for this module. if not specified,
          defaults to 0:len(modelspec). can be repeated for multiple
          modules
    .inegN : include module len(modelspec)-N
    .iiN : include modules 0:(N+1) -- including N! Note that .ii behavior
           differs from .ff and .xx
    .iinegN : include modules len(modelspec)-N:len(modelspec)
    .fN : freeze module N. ie, keep module in model but keep phi fixed
    .fnegN : freeze module len(modelspec)-N
    .ffN : freeze modules N:len(modelspec). Note that .ii behavior
           differs from .ff and .xx
    .ffnegN : freeze modules len(modelspec)-N:len(modelspec)
    .xN : exclude module N. ie, remove from model, assume that it won't break!
    .xnegN : exclude module len(modelspec)-N
    .xxN : exclude modules N:len(modelspec). Note that .ii behavior
           differs from .ff and .xx
    .xxnegN : exclude modules len(modelspec)-N:len(modelspec)
    .n

    TODO: Optimize more, make testbed to check how well future changes apply
    to disparate datasets.

    '''

    ops = escaped_split(kw, '.')[1:]
    st = False
    tf = False

    max_iter = 2000
    tolerance = 10**-5.5
    norm_fir = False
    fit_sig = 'resp'
    nl_kw = {}

    rand_count = 0
    keep_best = False

    sel_options = {'include_idx': [], 'exclude_idx': [], 'freeze_idx': []}
    metric_options = {'metric': 'nmse', 'alpha': 0}

    # TODO add support?
    initialize_nl = False

    # TF- specific parameters
    # TODO: integrate with others, share processing with regular basic/tf
    # eg, early_stopping_tolerance =?= tolerance
    # loss_type =?= metric
    use_modelspec_init = False
    fitter = 'Adam'
    loss_type = 'squared_error'
    early_stopping_steps = 5
    early_stopping_tolerance = 1e-5
    learning_rate = 0.01
    distr = 'norm'
    keep_n = 1
    generate_predictions_for_each_initalization_condition = False

    for op in ops:
        if op == 'st':
            st = True
        elif op.startswith('tf'):
            tf = True
            max_iter = 10000
        elif op.startswith('it'):
            if len(op[2:]) == 0:
                max_iter = None
            else:
                max_iter = int(op[2:])
        elif op=='psth':
            fit_sig = 'psth'
        elif op.startswith('nl'):
            nl_kw = {'nl_mode': int(op[2:])}
        elif op.startswith('t'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            tolerance = 10**tolpower
        elif op.startswith('pLV'):
            # pupil dep. cost function
            metric = 'pup_dep_LV'
            alpha = float(op[3:].replace(',', '.'))
            metric_options.update({'metric': metric, 'alpha': alpha})
        elif op == 'L2f':
            norm_fir = True
        elif op.startswith('rbp'):
            keep_best = True
            generate_predictions_for_each_initalization_condition = True
            if len(op) == 3:
                rand_count = 10
            else:
                rand_count = int(op[3:])
        elif op.startswith('rb'):
            if len(op) == 2:
                rand_count = 10
            else:
                rand_count = int(op[2:])
            keep_best = True
            keep_n = 1
        elif op.startswith('r'):
            if len(op) == 1:
                rand_count = 10
            else:
                rand_count = int(op[1:])
        elif op.startswith('b'):
            keep_best = True
            if len(op) == 1:
                keep_n = 1
            else:
                keep_n = int(op[1:])
        elif op.startswith('iineg'):
            sel_options['include_through'] = -int(op[5:])
        elif op.startswith('ineg'):
            sel_options['include_idx'].append(-int(op[4:]))
        elif op.startswith('ii'):
            sel_options['include_through'] = int(op[2:])
        elif op.startswith('i'):
            if not tf:
                sel_options['include_idx'].append(int(op[1:]))
            else:
                if len(op[1:]) == 0:
                    max_iter = None
                else:
                    max_iter = int(op[1:])

        elif op.startswith('ffneg'):
            sel_options['freeze_after'] = -int(op[5:])
        elif op.startswith('fneg'):
            sel_options['freeze_idx'].append(-int(op[4:]))
        elif op.startswith('ff'):
            sel_options['freeze_after'] = int(op[2:])
        elif op.startswith('f'):
            sel_options['freeze_idx'].append(int(op[1:]))
        elif op.startswith('xxneg'):
            sel_options['exclude_after'] = -int(op[5:])
        elif op.startswith('xneg'):
            sel_options['exclude_idx'].append(-int(op[4:]))
        elif op.startswith('xx'):
            sel_options['exclude_after'] = int(op[2:])
        elif op.startswith('x'):
            sel_options['exclude_idx'].append(int(op[1:]))

            # begin TF-specific options
        elif op == 'n':
            use_modelspec_init = True
        elif op.startswith('lr'):
            learning_rate = op[2:]
            if 'e' in learning_rate:
                base, exponent = learning_rate.split('e')
                learning_rate = int(base) * 10 ** -int(exponent)
            else:
                learning_rate = int(learning_rate)
        elif op[:1] == 'f':
            fitter = op[1:]
            if fitter in ['adam', 'a']:
                fitter = 'Adam'
            elif fitter == 'gd':
                fitter = 'GradientDescent'
        elif op[:1] == 'l':
            loss_type = op[1:]
            if loss_type == 'se':
                loss_type = 'squared_error'
            if loss_type == 'p':
                loss_type = 'poisson'
            if loss_type == 'nmse':
                loss_type = 'nmse'
            if loss_type == 'nmses':
                loss_type = 'nmse_shrinkage'
        elif op.startswith('et'):
            early_stopping_tolerance = 1 * 10 ** -int(op[2:])
        elif op[:1] == 'e':
            early_stopping_steps = int(op[1:])
        elif op[:1] == 'd':
            distr = op[1:]
            if distr == 'gu':
                distr = 'glorot_uniform'
            elif distr == 'heu':
                distr = 'he_uniform'

    bsel = False
    for key in list(sel_options.keys()):
        value = sel_options[key]
        if (type(value) is list) and (len(value)>0):
            bsel=True
        elif (type(value) is int):
            bsel=True
        else:
            del sel_options[key]

    xfspec = []
    if rand_count > 0:
        xfspec.append(['nems0.initializers.rand_phi', {'rand_count': rand_count}])

    sel_options.update({'tolerance': tolerance, 'norm_fir': norm_fir,
                        'nl_kw': nl_kw})
    sel_options.update({'max_iter': max_iter,
                       'use_modelspec_init': use_modelspec_init,
                       'optimizer': fitter,
                       'cost_function': loss_type,
                       'early_stopping_steps': early_stopping_steps,
                       'early_stopping_tolerance': early_stopping_tolerance,
                       'learning_rate': learning_rate,
                       'distr': distr})
    if tf:
        sel_options['fit_function'] = 'nems0.tf.cnnlink.fit_tf_init'
        sel_options['use_modelspec_init'] = use_modelspec_init
    elif st:
        sel_options['fit_function'] = 'nems0.xforms.fit_state_init'
        sel_options['fit_sig'] = fit_sig
    elif bsel:
        sel_options['fit_function'] = 'nems0.xforms.fit_basic_subset'
    else:
        sel_options['fit_function'] = 'nems0.xforms.fit_basic_init'

    # save cost function for use by fitter (default is nmse)
    sel_options.update(metric_options)

    xfspec.append(['nems0.xforms.fit_wrapper', sel_options])

    if generate_predictions_for_each_initalization_condition:
        xfspec.append(['nems0.analysis.test_prediction.predict_and_summarize_for_all_modelspec', {}])
    if keep_best:
        xfspec.append(['nems0.analysis.test_prediction.pick_best_phi', 
            {'criterion': 'mse_fit', 'keep_n': keep_n}])

    return xfspec


@xform()
def iter(fitkey):
    '''
    Perform a fit_iteratively analysis on a model.

    Parameters
    ----------
    fitkey : str
        Expected format: iter.<fitter>.<misc>
        Example: iter.T3,4,7.S0,1.S2,3.S0,1,2,3.ti50.fi20
        Example translation:
            Use fit_iteratively with scipy_minimize
            (since 'cd' option is not present), 50 per-tolerance-level
            iterations, and 20 per-fit iterations.
            Begin with a tolerance level of 10**-3, followed by
            10**-4 and 10**-7. Within each tolerance level,
            first fit modules 0 and 1, then 2 and 3,
            and finally 0, 1, 2, and 3 all together.

    Options
    -------
    cd : Use coordinate_descent for fitting (default is scipy_minimize)
    TN,N,... : Use tolerance levels 10**-N for each N given, where N is
               any positive integer. Default=[10**-4]
    SN,N,... : Fit model indices N, N... for each N given,
               where N is any positive integer or zero. May be provided
               multiple times to iterate over several successive subsets.
    tiN : Perform N per-tolerance-level iterations, where N is any
          positive integer. Default=50
    fiN : Perform N per-fit iterations, where N is any positive integer.
          Default=10
    b : choose best fit_idx based on mse_fit (relevant only when multiple
        initial conditions)
    '''

    # TODO: Support nfold and state fits for fit_iteratively?
    #       And epoch to go with state.
    options = _extract_options(fitkey)
    tolerances, module_sets, fit_iter, tol_iter, fitter, choose_best = \
        _parse_iter(options)

    if 'pop' in options:
        xfspec = [['nems0.analysis.fit_pop_model.fit_population_iteratively',
                   {'module_sets': module_sets, 'fitter': fitter,
                    'tolerances': tolerances, 'tol_iter': tol_iter,
                    'fit_iter': fit_iter}]]

    else:
        xfspec = [['nems0.xforms.fit_iteratively',
                   {'module_sets': module_sets, 'fitter': fitter,
                    'tolerances': tolerances, 'tol_iter': tol_iter,
                    'fit_iter': fit_iter}]]
    if choose_best:
        xfspec.append(['nems0.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

    return xfspec


def _extract_options(fitkey):
    if fitkey == 'basic' or fitkey == 'iter':
        # empty options (i.e. just use defaults)
        options = []
    else:
        chunks = escaped_split(fitkey, '.')
        options = chunks[1:]
    return options

def _parse_options(fitkey, **default_options):
    
    chunks = escaped_split(fitkey, '.')
    ops = chunks[1:]

    # set defaults
    options = {}
    options['max_iter'] = default_options.get('max_iter', 30000)
    options['tolerance'] = default_options.get('tolerance', 1e-5)
    options['fitter'] = default_options.get('fitter', 'scipy_minimize')
    options['choose_best'] = default_options.get('choose_best', False)
    options['rand_count'] = default_options.get('rand_count', 1)

    for op in ops:
        if op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            options['max_iter'] = int(re.match(pattern, op).group(1))

        elif op.startswith('FL'):
            if ':' in op:
                # ex: FL0:5  would be freeze_layers = [0,1,2,3,4]
                lower, upper = [int(i) for i in op[2:].split(':')]
                options['freeze_layers'] = list(range(lower, upper))
            else:
                # ex: FL2x6x9  would be freeze_layers = [2, 6, 9]
                options['freeze_layers'] = [int(i) for i in op[2:].split('x')]
        elif op == 'tf':
            options['backend'] = 'tf'
        elif op == 'sci':
            options['backend'] = 'scipy'
        elif op.startswith('t'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            options['tolerance'] = 10**tolpower
        elif op == 'cd':
            options['fitter'] = 'coordinate_descent'
        elif op == 'b':
            options['choose_best'] = True
        elif op.startswith('rb'):
            if len(op) == 2:
                options['rand_count'] = 10
            else:
                options['rand_count'] = int(op[2:])
            options['choose_best'] = True
        elif op.startswith('lr'):
            learning_rate = op[2:]
            if 'e' in learning_rate:
                base, exponent = learning_rate.split('e')
                options['learning_rate'] = int(base) * 10 ** -int(exponent)
            else:
                options['learning_rate'] = 10 ** -float(learning_rate)
        elif op.startswith('es'):
            options['validation_split'] = float(op[2:])/100
    return options


def _parse_basic(options):
    '''Options specific to basic.'''
    max_iter = 30000
    tolerance = 1e-5
    fitter = 'scipy_minimize'
    choose_best = False
    rand_count = 1
    for op in options:
        if op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            max_iter = int(re.match(pattern, op).group(1))
        elif op.startswith('t'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            tolerance = 10**tolpower
        elif op == 'cd':
            fitter = 'coordinate_descent'
        elif op == 'b':
            choose_best = True
        elif op.startswith('rb'):
            if len(op) == 2:
                rand_count = 10
            else:
                rand_count = int(op[2:])
            choose_best = True

    return max_iter, tolerance, fitter, choose_best, rand_count


def _parse_iter(options):
    '''Options specific to iter.'''
    tolerances = []
    module_sets = []
    fit_iter = 10
    tol_iter = 50
    fitter = 'scipy_minimize'
    choose_best = False
    for op in options:
        if op.startswith('ti'):
            tol_iter = int(op[2:])
        elif op.startswith('fi'):
            fit_iter = int(op[2:])
        elif op.startswith('T'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            nums = op.replace('d', '.').replace('\\', '')
            powers = [float(i) for i in nums[1:].split(',')]
            tolerances.extend([10**(-1*p) for p in powers])
        elif op.startswith('S'):
            indices = [int(i) for i in op[1:].split(',')]
            module_sets.append(indices)
        elif op == 'cd':
            fitter = 'coordinate_descent'
        elif op == 'b':
            choose_best = True

    if not tolerances:
        tolerances = None
    if not module_sets:
        module_sets = None

    return tolerances, module_sets, fit_iter, tol_iter, fitter, choose_best


# need to make sure this is still useful. possibly remove? or integrate into init?
@xform()
def initpop(kw):
    options = keyword_extract_options(kw)

    rnd = False
    flip_pcs = True
    start_count = 1
    usepcs = True
    for op in options:
        if op.startswith("rnd"):
            rnd = True
            if len(op)>3:
                start_count=int(op[3:])
        elif op=='nflip':
            flip_pcs = False

    if rnd:
        xfspec = [['nems0.xforms.fit_wrapper',
                   {'pc_signal': 'rand_resp', 'start_count': start_count,
                    'fit_function': 'nems0.analysis.fit_pop_model.init_pop_rand'}]]
    elif usepcs:
        xfspec = [['nems0.xforms.fit_wrapper',
                   {'flip_pcs': flip_pcs,
                    'fit_function': 'nems0.analysis.fit_pop_model.init_pop_pca'}]]
    return xfspec


@xform()
def ccnorm(fitkey):
    '''
    Perform a fit_ccnorm fit on a model.

    Parameters
    ----------

    Options
    -------

    '''

    options = _extract_options(fitkey)
    max_iter, tolerance, fitter, choose_best, rand_count = _parse_basic(options)

    sel_options = {'max_iter': max_iter, 'tolerance': tolerance}
    sel_options['fit_function'] = 'nems0.analysis.fit_ccnorm.fit_ccnorm'

    for op in options:
        if op[:1] == 'w':
            sel_options['use_metric'] = 'cc_err_w'
            if len(op)>1:
                op = op.replace('n','-')
                op = op.replace('e','.')
                sel_options['alpha']=10**int(op[1:])
        if op == 'md':
            sel_options['use_metric'] = 'cc_err_md'
        if op[:2]=='pc':
            sel_options['noise_pcs'] = int(op[2:])
        if op[:2]=='ss':
            sel_options['shared_pcs'] = int(op[2:])
        if op[:2]=='sh':
            sel_options['shrink_cc'] = int(op[2:])
        if op[:4]=='beta':
            sel_options['beta'] = int(op[4:])
        if op=='r':
            sel_options['also_fit_resp'] = True
        if op=='psth':
            sel_options['force_psth'] = True

        elif op.startswith('ffneg'):
            sel_options['freeze_after'] = -int(op[5:])
        elif op.startswith('fneg'):
            sel_options['freeze_idx'] = sel_options.get('freeze_idx', [])
            sel_options['freeze_idx'].append(-int(op[4:]))
        elif op.startswith('ff'):
            sel_options['freeze_after'] = int(op[2:])
        elif op.startswith('f'):
            sel_options['freeze_idx'] = sel_options.get('freeze_idx', [])
            sel_options['freeze_idx'].append(int(op[1:]))
        elif op.startswith('xxneg'):
            sel_options['exclude_after'] = -int(op[5:])
        elif op.startswith('xneg'):
            sel_options['exclude_idx'] = sel_options.get('exclude_idx', [])
            sel_options['exclude_idx'].append(-int(op[4:]))
        elif op.startswith('xx'):
            sel_options['exclude_after'] = int(op[2:])
        elif op.startswith('x'):
            sel_options['exclude_idx'] = sel_options.get('exclude_idx', [])
            sel_options['exclude_idx'].append(int(op[1:]))

    xfspec = []
    if rand_count > 1:
        freeze_idx=sel_options.get('freeze_idx',None)
        xfspec.append(['nems0.initializers.rand_phi', {'rand_count': rand_count,
                                                      'freeze_idx': freeze_idx}])

    xfspec.append(['nems0.xforms.fit_wrapper', sel_options])

    if choose_best:
        keep_n=1
        xfspec.append(['nems0.analysis.test_prediction.pick_best_phi',
            {'criterion': 'mse_fit', 'keep_n': keep_n}])

    xfspec.append(['nems0.xforms.predict', {}])
    xfspec.append(['nems0.xforms.add_summary_statistics', {}])
    #xfspec.append(['nems_lbhb.stateplots.ddr_pairs', {}])

    return xfspec

@xform()
def pcnorm(fitkey):
    '''
    Perform a fit_pcnorm fit on a model.

    Parameters
    ----------

    Options
    -------

    '''

    options = _extract_options(fitkey)
    max_iter, tolerance, fitter, choose_best, rand_count = _parse_basic(options)

    n_pcs = 2
    for op in options:
        if op[:2]=='pc':
            n_pcs = int(op[2:])

    sel_options = {'max_iter': max_iter, 'tolerance': tolerance, 'n_pcs': n_pcs}
    sel_options['fit_function'] = 'nems0.analysis.fit_ccnorm.fit_pcnorm'

    xfspec = []
    if rand_count > 1:
        xfspec.append(['nems0.initializers.rand_phi', {'rand_count': rand_count}])

    xfspec.append(['nems0.xforms.fit_wrapper', sel_options])

    if choose_best:
        xfspec.append(['nems0.analysis.test_prediction.pick_best_phi',
            {'criterion': 'mse_fit', 'keep_n': keep_n}])

    xfspec.append(['nems0.xforms.predict', {}])
    xfspec.append(['nems0.xforms.add_summary_statistics', {}])
    xfspec.append(['nems0.plots.state.cc_comp', {}])

    return xfspec


