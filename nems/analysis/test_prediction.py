# For testing the predcitive accuracy of a set of modelspecs
import numpy as np
import copy
import logging

import nems0.modelspec as ms
import nems0.metrics.api as nmet
from nems0.analysis.cost_functions import basic_cost
import nems0.priors
import nems0.fitters.mappers
import nems0.segmentors
import nems0.utils

log = logging.getLogger(__name__)


def generate_prediction(est, val, modelspec, use_mask=True, **context):

    # TODO support for multiple recording views/modelspec jackknifes (jack_count>0)
    #  outer loop = fit, inner loop = jackknife ?

    list_val = (type(val) is list)
    list_modelspec = (type(modelspec) is list)
    if list_val | list_modelspec:
        raise ValueError("list-type val recordings or modelspecs no longer supported")

    # Evaluate estimation and validation data

    # three scenarios:
    # 1 fit, 1 est set - standard
    # n fits, n est sets - nfold
    # m fits, 1 est set - multiple initial conditions
    # m * n fits, n est sets - multiple initial conditions + nfold
    fit_count = modelspec.fit_count  # (m)
    n = modelspec.jack_count
    out_est_signals = []
    out_val_signals = []

    for fit_idx in range(fit_count):

        do_inverse_merge = False
        if est.view_count < n:
            # if multiple jackknife fits but only one est/val view. Shouldn't ever happen
            raise Warning('tiling est + val models X views. Should this happen?')
            new_est = est.tile_views(n)
            new_val = val.tile_views(n)
        else:
            # assume est and val have .view_count == len(modelspecs)
            new_est = est.copy()
            new_val = val.copy()
            if n>1:  # ie, jack_count>1
                do_inverse_merge = True

        modelspec.fit_index = fit_idx
        for jack_idx in range(n):
            modelspec.jack_index = jack_idx
            new_est.view_idx = jack_idx
            new_val.view_idx = jack_idx

            # update each view with prediction from corresponding modelspec
            new_est = modelspec.evaluate(new_est)
            new_val = modelspec.evaluate(new_val)

            # this seems kludgy. but where should mask be handled?
            if 'mask' in new_val.signals.keys() and use_mask:
                m = new_val['mask'].as_continuous()
                x = new_val['pred'].as_continuous().copy()
                x[..., m[0, :] == 0] = np.nan
                new_val['pred'] = new_val['pred']._modified_copy(x)

        if do_inverse_merge:
            new_val = new_val.jackknife_inverse_merge()
        out_est_signals.extend(new_est.signal_views)
        out_val_signals.extend(new_val.signal_views)

    new_est.signal_views = out_est_signals
    new_val.signal_views = out_val_signals

    return new_est, new_val


def standard_correlation(est, val, modelspec=None, modelspecs=None, rec=None,
                         use_mask=True, **context):
    # use_mask: mask before computing metrics (if mask exists)
    # Compute scores for validation dat
    r_ceiling = 0

    # deprecated support for modelspecs lists
    if modelspecs is not None:
        raise Warning('Use of modelspecs list is deprecated')

    # by default, assume that model is trying to predict resp signal
    output_name = modelspec.meta.get('output_name', 'resp')

    # TODO: support for multiple views -- if ever desired? usually validation set views
    #       should have been recombined by now, right?
    view_count = val.view_count

    # KLUDGE ALERT!
    # only compute results for first jackknife -- for simplicity, not optimal!
    # only works if view_count==1 or resp_count(# resp channels)==1
    try:
        est_mult = modelspec.jack_count
    except:
        est_mult = 1

    out_chan_count = val[output_name].shape[0]

    r_test = np.zeros((out_chan_count, view_count))
    se_test = np.zeros((out_chan_count, view_count))
    r_fit = np.zeros((out_chan_count, view_count))
    se_fit = np.zeros((out_chan_count, view_count))
    r_floor = np.zeros((out_chan_count, view_count))
    r_ceiling = np.zeros((out_chan_count, view_count))
    mse_test = np.zeros((out_chan_count, view_count))
    se_mse_test = np.zeros((out_chan_count, view_count))
    mse_fit = np.zeros((out_chan_count, view_count))
    se_mse_fit = np.zeros((out_chan_count, view_count))
    ll_test = np.zeros((out_chan_count, view_count))
    ll_fit = np.zeros((out_chan_count, view_count))

    for i in range(view_count):
        if type(use_mask) is str:
            v = val.set_view(i).apply_mask(mask_name=use_mask)
            e = est.set_view(i*est_mult).apply_mask()
            log.info(f"val use_mask={use_mask} shape={v['resp'].shape}")

        elif ('mask' in val.signals.keys()):
            v = val.set_view(i).apply_mask()
            e = est.set_view(i*est_mult).apply_mask()
        else:
            v = val.set_view(i)
            e = est.set_view(i*est_mult)
            use_mask = False
        r_test[:,i], se_test[:,i] = nmet.j_corrcoef(v, 'pred', output_name)
        r_fit[:,i], se_fit[:,i] = nmet.j_corrcoef(e, 'pred', output_name)
        r_floor[:,i] = nmet.r_floor(v, 'pred', output_name)

        mse_test[:,i], se_mse_test[:,i] = nmet.j_nmse(v, 'pred', output_name)
        mse_fit[:,i], se_mse_fit[:,i] = nmet.j_nmse(e, 'pred', output_name)

        ll_test[:,i] = nmet.likelihood_poisson(v, 'pred', output_name)
        ll_fit[:,i] = nmet.likelihood_poisson(e, 'pred', output_name)

        if rec is not None:
            #if 'mask' in rec.signals.keys() and use_mask:
            #    r = rec.apply_mask()
            #else:
            r = rec
            # print('running r_ceiling')
            r_ceiling[:,i] = nmet.r_ceiling(v, r, 'pred', output_name)

    """

        # fix view_index = 0
        i = 0

        if ('mask' in val.signals.keys()) and use_mask:
            v = val.set_view(i).apply_mask()
            e = est.set_view(i*est_mult).apply_mask()
        else:
            v = val.set_view(i)
            e = est.set_view(i*est_mult)
            use_mask = False

        r_test, se_test = nmet.j_corrcoef(v, 'pred', output_name)
        r_fit, se_fit = nmet.j_corrcoef(e, 'pred', output_name)
        r_floor = nmet.r_floor(v, 'pred', output_name)

        mse_test, se_mse_test = nmet.j_nmse(v, 'pred', output_name)
        mse_fit, se_mse_fit = nmet.j_nmse(e, 'pred', output_name)

        ll_test = nmet.likelihood_poisson(v, 'pred', output_name)
        ll_fit = nmet.likelihood_poisson(e, 'pred', output_name)

        if rec is not None:
            if 'mask' in rec.signals.keys() and use_mask:
                r = rec.apply_mask()
            else:
                r = rec
            # print('running r_ceiling')
            r_ceiling = nmet.r_ceiling(v, r, 'pred', output_name)
    """

    modelspec.meta['r_test'] = r_test
    modelspec.meta['se_test'] = se_test
    modelspec.meta['r_floor'] = r_floor
    modelspec.meta['mse_test'] = mse_test
    modelspec.meta['se_mse_test'] = se_mse_test
    modelspec.meta['ll_test'] = ll_test

    modelspec.meta['r_fit'] = r_fit
    modelspec.meta['se_fit'] = se_fit
    modelspec.meta['r_ceiling'] = r_ceiling
    modelspec.meta['mse_fit'] = mse_fit
    modelspec.meta['se_mse_fit'] = se_mse_fit
    modelspec.meta['ll_fit'] = ll_fit

    return modelspec


def correlation_per_model(est, val, modelspecs, rec=None):
    '''
    Expects the lengths of est, val, and modelspecs to match since est[i]
    should have been evaluated on the fitted modelspecs[i], etc.
    Similar to standard_correlation, but saves correlation information
    to every first-module 'meta' entry instead of saving an average
    to only the first modelspec
    '''
    if not len(est) == len(val) == len(modelspecs):
        raise ValueError("est, val, and modelspecs should all be lists"
                         " of equal length. got: %d, %d, %d respectively.",
                         len(est), len(val), len(modelspecs))

    modelspecs = copy.deepcopy(modelspecs)

    r_tests = [nmet.corrcoef(v, 'pred', 'resp') for v in val]
    #se_tests = [np.std(r)/np.sqrt(len(v)) for r, v in zip(r_tests, val)]
    mse_tests = [nmet.nmse(v, 'pred', 'resp') for v in val]
    ll_tests = [nmet.likelihood_poisson(v, 'pred', 'resp') for v in val]

    r_fits = [nmet.corrcoef(e, 'pred', 'resp') for e in est]
    #se_fits = [np.std(r)/np.sqrt(len(v)) for r, v in zip(r_fits, val)]
    mse_fits = [nmet.nmse(e, 'pred', 'resp') for e in est]
    ll_fits = [nmet.likelihood_poisson(e, 'pred', 'resp') for e in est]

    r_floors = [nmet.r_floor(v, 'pred', 'resp') for v in val]
    if rec is None:
        r_ceilings = [None]*len(r_floors)
    else:
        r_ceilings = [nmet.r_ceiling(v, rec, 'pred', 'resp') for v in val]

    for i, m in enumerate(modelspecs):
        m[0]['meta'].update({
                'r_test': r_tests[i], #'se_test': se_tests[i],
                'mse_test': mse_tests[i], 'll_test': ll_tests[i],
                'r_fit': r_fits[i], #'se_fit': se_fits[i],
                'mse_fit': mse_fits[i], 'll_fit': ll_fits[i],
                'r_floor': r_floors[i], 'r_ceiling': r_ceilings[i],
                })

    return modelspecs


def standard_correlation_by_epochs(est,val,modelspec=None,modelspecs=None,epochs_list=None, rec=None, use_mask=True):
    """
    Does the same thing as standard_correlation, excpet with subsets of data
    defined by epochs_list

    To use this, first add epochs to define subsets of data.
    Then, pass epochs_list as a list of subsets to test.
    For example, ['A', 'B', ['A', 'B']] will measure correlations separately
     for all epochs marked 'A', all epochs marked 'B', and all epochs marked
     'A'or 'B'
    """
    # some crazy stuff to maintain backward compatibility
    # eventually we will only support modelspec and deprecate support for
    # modelspecs lists
    if modelspecs is not None:
        raise Warning('Use of modelspecs list is deprecated')
        modelspec = modelspecs[0]
        list_modelspec = True
    else:
        list_modelspec = False

    for epochs in epochs_list:
        # Create a label for this subset. If epochs is a list, join elements with "+"
        epoch_list_str="+".join([str(x) for x in epochs])

        # Make a copy for this subset
        val_copy=copy.deepcopy(val)
        for vc in val_copy:
            vc['resp']=vc['resp'].select_epochs(epochs)

        est_copy=copy.deepcopy(est)
        for ec in est_copy:
            ec['resp']=ec['resp'].select_epochs(epochs)

        modelspec_ = modelspec.copy()

        #CALL standard_correlation to compute metrics on this subset
        modelspec_ = standard_correlation(est_copy, val_copy, modelspec_, rec=rec,use_mask=use_mask)

        names = ['r_test','se_test','r_floor','mse_test','se_mse_test','ll_test',
                 'r_fit','se_fit','r_ceiling','mse_fit','se_mse_fit','ll_fit']
        modelspec.meta[epoch_list_str]={}
        for name in names:
            modelspec.meta[epoch_list_str][name] = modelspec_.meta[name]

    if list_modelspec:
        # backward compatibility
        return [modelspec]
    else:
        return modelspec


def generate_prediction_sets(est, val, modelspecs):
    if type(val) is list:
        # ie, if jackknifing
        new_est = [ms.evaluate(d, m) for m,d in zip(modelspecs,est)]
        new_val = [ms.evaluate(d, m) for m,d in zip(modelspecs,val)]
    else:
        raise ValueError('val and est must be lists')

    return new_est, new_val


def standard_correlation_by_set(est, val, modelspecs):

    # Compute scores for validation data
    r_test = [nmet.corrcoef(p, 'pred', 'resp') for p in val]
    mse_test = [nmet.nmse(p, 'pred', 'resp') for p in val]
    ll_test = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in val]

    # Repeat for est data.
    r_fit = [nmet.corrcoef(p, 'pred', 'resp') for p in est]
    mse_fit = [nmet.nmse(p, 'pred', 'resp') for p in est]
    ll_fit = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in est]
    for i in range(len(modelspecs)):
        modelspecs[i][0]['meta']['r_test'] = r_test[i]
        modelspecs[i][0]['meta']['mse_test'] = mse_test[i]
        modelspecs[i][0]['meta']['ll_test'] = ll_test[i]

        modelspecs[i][0]['meta']['r_fit'] = r_fit[i]
        modelspecs[i][0]['meta']['mse_fit'] = mse_fit[i]
        modelspecs[i][0]['meta']['ll_fit'] = ll_fit[i]

    return modelspecs


def basic_error(data, modelspec, cost_function=None,
                segmentor=nems0.segmentors.use_all_data,
                mapper=nems0.fitters.mappers.simple_vector,
                metric=lambda data: nmet.nmse(data, 'pred', 'resp')):
    '''
    Similar to fit_basic except that it just returns the error for the fitting
    process instead of a modelspec. Intended to be called after a model
    has already been fit.
    '''
    modelspec = copy.deepcopy(modelspec)
    if cost_function is None:
        # Use the cost function defined in this module by default
        cost_function = basic_cost

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    if 'mask' in data.signals.keys():
        log.info("Data len pre-mask: %d", data['mask'].shape[1])
        data = data.apply_mask()
        log.info("Data len post-mask: %d", data['mask'].shape[1])

    packer, unpacker, pack_bounds = mapper(modelspec)
    evaluator = nems0.modelspec.evaluate
    sigma = packer(modelspec)
    error = cost_function(sigma, unpacker, modelspec, data, segmentor,
                          evaluator, metric)

    return error


def predict_and_summarize_for_all_modelspec(modelspec=None, est=None, val=None, est_list=None, val_list=None, criterion=['r_test','se_test','mse_test','ll_test'],
                  metric_fn='nems0.metrics.mse.nmse', jackknifed_fit=False, keep_n=1,
                  IsReload=False, **context):
    """
     For models with multiple fits (eg, based on multiple initial conditions),
      find the best prediction for the recording provided (presumably est data,
      though possibly something held out)

     For jackknifed fits, pick the best fit for each jackknife set. so a F x J modelspec
      is reduced in size to 1 x J. Models are tested with est data for that jackknife.
      This has only been tested with est recording, which is likely should be used.

     :param modelspec: should have fit_count>0
     :param est: view_count should match fit_count, ie,
                 after generate_prediction is called
     :param context: extra context stuff for xforms compatibility.
     :return: modelspec with fit_count==1
     """
    if IsReload:
        return {}
    if modelspec.fit_count <= 1:
        return {}

    if est_list is None:
        est_list = [est]
        val_list = [val]


    if modelspec.jack_count > 1:
        log.info('Not supported yet! jackknife + multifit using tf loss to select')
        import pdb; pdb.set_trace()

    if modelspec.cell_count > 1:
        log.info('Not tested for mega models. I think it will work but confirm. LAS')
        import pdb; pdb.set_trace()
    fit_count = modelspec.fit_count
    modelspec.set_cell(0) #cell is a misnomer here, means site
    Ncells = val_list[0]['resp'].shape[0]
    means = [np.zeros(modelspec.fit_count) for i in range(len(criterion))]
    if modelspec.cell_count == 1:
        indiv_cells = [np.zeros((Ncells,modelspec.fit_count)) for i in range(len(criterion))]
    n=0
    for cell_idx, (this_est, this_val) in enumerate(zip(est_list, val_list)):
        #loop through each "cell_idx" (site index for megamodels)
        for fitidx in range(modelspec.fit_count):
            #loop through each fitidx
            traw = modelspec.raw[cell_idx, fitidx, 0]
            tmodelspec = ms.ModelSpec(traw)
            this_est = tmodelspec.evaluate(this_est)
            this_val = tmodelspec.evaluate(this_val)
            tmodelspec = standard_correlation(est=this_est, val=this_val, modelspec=tmodelspec)
            # average performance across output channels (if more than one output)
            for i,criterion_ in enumerate(criterion):
                means[i][fitidx] += tmodelspec.meta[criterion_].sum(axis=0)
                if modelspec.cell_count == 1:
                    indiv_cells[i][:, fitidx] = tmodelspec.meta[criterion_][:, 0]
        n += tmodelspec.meta[criterion[0]].shape[0]

    for i,criterion_ in enumerate(criterion):
        means[i] = means[i] / n
        modelspec.meta['rand_' + criterion_] = means[i]
        if modelspec.cell_count == 1:
            modelspec.meta['rand_' + criterion_+'_all'] =  indiv_cells[i]

    return {'modelspec': modelspec}

def pick_best_phi(modelspec=None, est=None, val=None, est_list=None, val_list=None, criterion='mse_fit',
                  metric_fn='nems0.metrics.mse.nmse', jackknifed_fit=False, keep_n=1,
                  IsReload=False, **context):

    """
    For models with multiple fits (eg, based on multiple initial conditions),
     find the best prediction for the recording provided (presumably est data,
     though possibly something held out)

    For jackknifed fits, pick the best fit for each jackknife set. so a F x J modelspec
     is reduced in size to 1 x J. Models are tested with est data for that jackknife.
     This has only been tested with est recording, which is likely should be used.

    :param modelspec: should have fit_count>0
    :param est: view_count should match fit_count, ie,
                after generate_prediction is called
    :param context: extra context stuff for xforms compatibility.
    :return: modelspec with fit_count==1
    """
    if IsReload:
        return {}
    if modelspec.fit_count<=1:
        return {}

    if est_list is None:
        est_list=[est]
        val_list=[val]

    #for cellidx,est,val in zip(range(len(est_list)),est_list,val_list):
    #    modelspec.set_cell(cellidx)
    #    est, val = nems0.analysis.api.generate_prediction(est, val, modelspec, jackknifed_fit=jackknifed_fit)
    #    modelspec.recording = val
    #    est_list[cellidx] = est
    #    val_list[cellidx] = val

    #import pdb; pdb.set_trace()

    jack_count = modelspec.jack_count
    fit_count = modelspec.fit_count
    best_idx = np.zeros(jack_count, dtype=int)
    modelspec.set_cell(0)

    # for each jackknife set, figure out best fit. save it to new_raw
    new_raw = np.zeros((modelspec.cell_count, keep_n, jack_count), dtype='O')
    for j in range(jack_count):
        view_range = [i * jack_count + j for i in range(fit_count)]
        x = None
        n = 0

        # support for multi-cell, len(est_list)>1 fits
        #import pdb; pdb.set_trace()

        for cell_idx, this_est in enumerate(est_list):
            #this_modelspec = modelspec.copy(jack_index=j)
            #this_modelspec.cell_index = cell_idx
            modelspec.cell_index = cell_idx
            modelspec.jack_index = j

            if 'loss' in modelspec.meta.keys():
                # quick ranking: use a vector of loss values, one per fitidx stored in meta
                if x is None:
                    x = modelspec.meta['loss']
                else:
                    x = x+modelspec.meta['loss']
                n = n+1
                if j>1:
                    log.info('Not supported yet! jackknife + multifit using tf loss to select')
                    import pdb; pdb.set_trace()
            
            elif (metric_fn == 'nems0.metrics.mse.nmse') & (criterion == 'mse_fit'):
                # for backwards compatibility, run the below code to compute metric specified
                # by criterion.
                # unclear if this works
                x = np.zeros(modelspec.fit_count)
                for fitidx in range(modelspec.fit_count):
                    traw = modelspec.raw[cell_idx, fitidx, j]
                    tmodelspec=ms.ModelSpec(traw)
                    this_est = tmodelspec.evaluate(this_est)
                    this_val = tmodelspec.evaluate(this_val)
                    tmodelspec = standard_correlation(est=this_est, val=this_val, modelspec=tmodelspec)
                    # average performance across output channels (if more than one output)
                    x[fitidx] += tmodelspec.meta[criterion].sum(axis=0)
                n += new_modelspec.meta[criterion].shape[0]

            else:
                # unclear if this works
                fn = nems0.utils.lookup_fn_at(metric_fn)
                tx=[]
                for fitidx in range(modelspec.fit_count):
                    traw = modelspec.raw[cell_idx, fitidx, j]
                    tmodelspec=ms.ModelSpec(traw)
                    this_est = tmodelspec.evaluate(this_est)
                    tx.append(fn(this_est, **context))
                n = n + tx[0].shape[0]
                tx = np.concatenate(tx, axis=1).mean(axis=0)
                if x is None:
                   x = tx
                else:
                   x = x + tx

        x = x / n

        if np.any(x==0):
            raise RuntimeError('Error: At least one elements of modelspec.meta[\'loss\'] is 0. '\
                    'This will cause this fn to pick that phi, but it\'s probably not. Bug in the code?')
        tx = x.copy()
        for n in range(keep_n):
           best_idx[j] = int(np.nanargmin(tx))
           new_raw[:, n, j] = modelspec.raw[:, best_idx[j], j]

           log.info('jack %d: %d/%d best phi (fit_idx=%d) has fit_metric=%.5f',
                    j, n+1, keep_n, best_idx[j], tx[best_idx[j]])
           tx[best_idx[j]] = np.nanmax(tx)

    for cell_index in range(new_raw.shape[0]):
        new_raw[cell_index,0,0][0]['meta'] = modelspec.raw[cell_index,0,0][0]['meta'].copy()
    new_modelspec = ms.ModelSpec(new_raw)
    new_modelspec.set_cell(0)
    new_modelspec.set_jack(0)
    new_modelspec.meta['rand_'+criterion] = x

    return {'modelspec': new_modelspec, 'best_random_idx': best_idx}
