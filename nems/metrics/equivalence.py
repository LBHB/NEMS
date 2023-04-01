import logging
from pathlib import Path
from copy import copy

import numpy as np
import scipy.stats as st
from scipy import linalg
import pandas as pd

import nems0.db as nd
import nems0.xform_helper as xhelp

log = logging.getLogger(__name__)


def equivalence_partial(psth1, psth2, baseline):
    '''Computes partial correlation-based equivalence of two model predictions.

    Parameters:
    ----------
    psth1: np.ndarray
        First model prediction, will be converted to (1, time) array.
    psth2: np.ndarray
        Second model prediction.
    baseline: np.ndarray
        Model prediction to use as baseline - partial correlation will be
        computed between psth1 and psth2 relative to this variable.

    Returns:
    -------
    p: float
        Partial correlation between psth1 and psth2 relative to baseline.

    '''

    # psths may or may not have extra dim already, so flatten and then
    # add dim again to make sure they all match.
    psth1 = psth1.flatten()
    psth2 = psth2.flatten()
    baseline = baseline.flatten()
    C = np.hstack((np.expand_dims(psth1, 0).transpose(),
                   np.expand_dims(psth2, 0).transpose(),
                   np.expand_dims(baseline, 0).transpose()))
    p = partial_corr(C)[0,1]

    return p


# https://gist.github.com/fabianp/9396204419c7b638d38f
def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of
    variables in C, controlling for the remaining variables in C.

    Parameters
    ----------
    C : array-like, shape (n, p)
        Each column of C is taken as a variable

    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j]
        controlling for the remaining variables in C.

    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = st.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def equivalence_partial_batch(batch, model1, model2, baseline, path,
                              load_only=False, force_rerun=False,
                              manual_cellids=None, performance_stat='r_ceiling',
                              test_limit=np.inf, use_sites=True):
    '''
    Computes equivalence and effect size for all cells in a batch.

    Equivalence is computed as the partial correlation between model1 and
    model2 relative to baseline. This process can take several hours, so a save path is required so
    that results can be quickly reloaded in the future. Default behavior for
    subsequent function calls is to:
        1) Load the previously saved dataframe.
        2) Add computations for cells that do not have existing results.
        3) Return the updated dataframe.
    See `load_only` and `force_rerun` parameters to change this behavior.
    Results will also be saved intermittently while computations are carried
    out in case of a crash or uncaught exception.

    Parameters:
    -----------
    batch: int
        CellDB batch number.
    model1, model2, baseline: str
        Model names indicating which models should be compared.
    path: str or pathlib.Path
        Path where results should be saved/loaded from.
    load_only: bool
        Return existing results without updating them. Otherwise, computations
        will be run for cells that do not have existing results. If a dataframe
        does not exist at the specified `path`, raises a ValueError.
    force_rerun:
        Update results for all cells, even if previous results exist.
    manual_cellids: list
        List of cellids to use. By default, asll cellids for the given
        batch will be used.
    test_limit: int or np.inf
        For debugging. Specifies a maximum number of cells to load models for.
    use_sites: bool
        Use siteids as default cell list instead of cellids.

    Returns:
    --------
    df: Pandas dataframe
        Indexed by cellid with column: 'partial_corr' (equivalence).

    '''

    if manual_cellids is None:
        cellids = nd.get_batch_cells(batch=batch, as_list=True)
    else:
        cellids = manual_cellids
    siteids = list(set([c.split('-')[0] for c in cellids]))

    path = Path(path)
    if load_only:
        if not path.is_file():
            raise ValueError("load_only==True, but dataframe does not exist")
        else:
            df = pd.read_pickle(path)
            return df
    else:
        if path.is_file() and (not force_rerun):
            # Only update cells without existing results
            df = pd.read_pickle(path)
            existing_cells = df.index.values.tolist()
            cellids = list(set(cellids) - set(existing_cells))
        else:
            df = pd.DataFrame.from_dict({'cellid': [], 'partial_corr': []})
            df.set_index('cellid', inplace=True)

        partials = []
        used_cells = []
        models = [model1, model2, baseline]
        i = 0
        total_used_cells = 0

        if use_sites:
            site_cache = {site:{} for site in siteids}
        else:
            site_cache = None

        for site in siteids:
            site_cells = [c for c in cellids if site in c]
            for cell in site_cells:
                try:
                    pred1, pred2, baseline_pred = load_models(cell, batch, models, site=site, site_cache=site_cache)
                    # Bare except is intentional - missing result should be value error, but
                    # often there are other errors that pop up after package updates
                    # due to outdated preprocessing functions. Since the equivalence analysis
                    # is often left to run overnight, I would rather skip the exceptions
                    # and continue saving the rest of the batch.  -Jacob
                except Exception as e:
                    log.warning('Missing result or error loading for cell: %s: \n%s',
                                (cell, e))
                    continue

                partial = equivalence_partial(pred1, pred2, baseline_pred)
                partials.append(partial)
                used_cells.append(cell)
                i += 1
                total_used_cells += 1
                if i % 5 == 0:
                    # intermediate save after every 5 cells
                    df = _save_results(df, used_cells, partials, path)
                    partials = []
                    used_cells = []

                if total_used_cells >= test_limit:
                    break

        df = _save_results(df, used_cells, partials, path)

    return df

def _save_results(df, used_cells, partials, path):
    results = {'cellid': used_cells, 'partial_corr': partials}
    df2 = pd.DataFrame.from_dict(results)
    df2.set_index('cellid', inplace=True)
    df = df.append(df2) # combine new results with previous ones
    df.to_pickle(path) # save updated results

    return df


def load_models(cell, batch, models, check_db=True, site=None, site_cache=None):
    '''Load standardized psth from each model, error if not all fits exist.'''

    if check_db:
        # Before trying to load, check database to see if a result exists.
        # Should set False if you know model results are not stored in DB,
        # but exist in file storage.
        df = nd.batch_comp(batch=batch, modelnames=models, cellids=[cell])
        if np.sum(df.isna().values) > 0:
            # at least one cell wasn't fit (or at least not stored in db)
            # so skip trying to load any of them since all are required.
            raise ValueError('Not all results exist for: %s, %d' % (cell, batch))

    # Load all models
    ctxs = []
    for model in models:
        if site_cache is None:
            xf, ctx = xhelp.load_model_xform(cell, batch, model)
        elif model in site_cache[site]:
            log.info("Site %s is cached, skipping load...", site)
            ctx = site_cache[site][model]
        else:
            xf, ctx = xhelp.load_model_xform(cell, batch, model)
            site_cache[site][model] = ctx
        ctxs.append(ctx)

    for ctx in ctxs:
        if ctx['val']['pred'].chans is None:
            ctx['val']['pred'].chans = copy(ctx['val']['resp'].chans)

    # Pull out model predictions and remove times with nan for at least 1 model
    preds = [ctx['val'].apply_mask()['pred'].extract_channels([cell]).as_continuous() for ctx in ctxs]
    ff = np.isfinite(preds[0])
    for pred in preds[1:]:
        ff &= np.isfinite(pred)
    no_nans = [pred[ff] for pred in preds]

    return no_nans


def within_model_equivalence(batch, model1, model2, baseline, model1_h1,
                             model1_h2, model2_h1, model2_h2, baseline_h1,
                             baseline_h2, between_path,
                             within_path1a, within_path1b,
                             within_path2a, within_path2b,
                             cross_path1a, cross_path1b,
                             cross_path2a, cross_path2b,
                             load_only=False, force_rerun=False,
                             manual_cellids=None,
                             test_limit=np.inf):

    # different models, both full data
    between_df = equivalence_partial_batch(batch, model1, model2, baseline,
                                        manual_cellids=manual_cellids,
                                        load_only=load_only,
                                        force_rerun=force_rerun,
                                        path=between_path, test_limit=test_limit)

    # same model, opposite halves of data, first model
    within_df1a = equivalence_partial_batch(batch, model1_h1, model1_h2,
                                            baseline_h1,
                                            manual_cellids=manual_cellids,
                                            path=within_path1a,
                                            load_only=load_only,
                                            force_rerun=force_rerun,
                                            test_limit=test_limit)
    within_df1b = equivalence_partial_batch(batch, model1_h1, model1_h2,
                                            baseline_h2,
                                            manual_cellids=manual_cellids,
                                            path=within_path1b,
                                            test_limit=test_limit,
                                            load_only=load_only,
                                            force_rerun=force_rerun)

    # same model, opposite halves of data, second model
    within_df2a = equivalence_partial_batch(batch, model2_h1, model2_h2,
                                           baseline_h1,
                                           manual_cellids=manual_cellids,
                                           path=within_path2a,
                                           test_limit=test_limit,
                                           load_only=load_only,
                                           force_rerun=force_rerun)
    within_df2b = equivalence_partial_batch(batch, model2_h1, model2_h2,
                                           baseline_h2,
                                           manual_cellids=manual_cellids,
                                           path=within_path2b,
                                           test_limit=test_limit,
                                           load_only=load_only,
                                           force_rerun=force_rerun)

    # different models, opposite halves of data
    cross_df1a = equivalence_partial_batch(batch, model1_h1, model2_h2,
                                           baseline_h1,
                                           manual_cellids=manual_cellids,
                                           path=cross_path1a,
                                           test_limit=test_limit,
                                           load_only=load_only,
                                           force_rerun=force_rerun)
    cross_df1b = equivalence_partial_batch(batch, model1_h1, model2_h2,
                                           baseline_h2,
                                           manual_cellids=manual_cellids,
                                           path=cross_path1b,
                                           test_limit=test_limit,
                                           load_only=load_only,
                                           force_rerun=force_rerun)

    # different models, opposite halves of data
    cross_df2a = equivalence_partial_batch(batch, model1_h2, model2_h1,
                                           baseline_h1,
                                           manual_cellids=manual_cellids,
                                           path=cross_path2a,
                                           test_limit=test_limit,
                                           load_only=load_only,
                                           force_rerun=force_rerun)
    cross_df2b = equivalence_partial_batch(batch, model1_h2, model2_h1,
                                           baseline_h2,
                                           manual_cellids=manual_cellids,
                                           path=cross_path2b,
                                           test_limit=test_limit,
                                           load_only=load_only,
                                           force_rerun=force_rerun)

    # Filter dataframe indices so that each cell is present in every dataframe
    dfs = [between_df, within_df1a, within_df1b, within_df2a, within_df2b,
           cross_df1a, cross_df1b, cross_df2a, cross_df2b]
    df = dfs[0]
    for i, d in enumerate(dfs[1:]):
        df = df.join(d, how='inner', rsuffix='_df%d'%(i+1)) # use intersection of indices
    # Convert columns to meaningful names
    new_keys = ['between_partial',
                'within_partial_model1a', 'within_partial_model1b',
                'within_partial_model2a', 'within_partial_model2b',
                'cross_partial_1a', 'cross_partial_1b',
                'cross_partial_2a', 'cross_partial_2b']
    new_columns = {k: v for k, v in zip(df.columns.values, new_keys)}
    df = df.rename(columns=new_columns)

    cols = df.columns  # in same order as list of dfs
    between_full = df[cols[0]]
    within_half_model1 = df[cols[2:4]].mean(axis=1)
    within_half_model2 = df[cols[4:6]].mean(axis=1)
    between_half = df[cols[6:]].mean(axis=1)

    within_full_model1 = within_half_model1 * (between_full/between_half)
    within_full_model2 = within_half_model2 * (between_full/between_half)

    return within_full_model1, within_full_model2, df


def equivalence_effect_size(batch, models, performance_stat='r_ceiling',
                            manual_cellids=None):

    df = nd.batch_comp(batch=batch, modelnames=models, stat=performance_stat,
                       cellids=manual_cellids)
    stats = [df[model].values for model in models]
    rel_1 = stats[0] - stats[2]
    rel_2 = stats[1] - stats[2]
    effect_sizes = 0.5*(rel_1 + rel_2)
    results = {'performance_effect': effect_sizes, 'cellid': df.index.values}
    df = pd.DataFrame.from_dict(results)
    df.set_index('cellid', inplace=True)

    return df
