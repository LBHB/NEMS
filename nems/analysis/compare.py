import numpy as np
from scipy import stats

#from nems_lbhb.gcmodel.figures.equivalence import partial_corr
from nems0.xform_helper import load_model_xform

# https://gist.github.com/fabianp/9396204419c7b638d38f
def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
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
            beta_i = np.linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = np.linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def compare_predictions(modelspec1, modelspec2, modelspec_ref, rec1, rec2, rec_ref=None, cellid=None, show_plots=False):
   """
   generate predictions and compare performance between modelspec1 and modelspec2. 
   if modelspec_ref provided, measure equivalence
   if rec2==None, use rec1 for both model predictions
   """
   
   if 'mask' in rec1.signals.keys():
       rec1 = rec1.apply_mask()
   
   if rec2 is None:
       rec2 = rec1
   elif 'mask' in rec2.signals.keys():
       rec2 = rec2.apply_mask()

   # assume model has already been evaluated to produce pred
   #rec2 = modelspec2.evaluate(rec2)
   if rec_ref is None:
      rec_ref = modelspec_ref.evaluate(rec2)
   #rec1 = modelspec1.evaluate(rec1)

   if rec1['pred'].shape[0] > 1:
       # pop model, need to extract the relevant channel:
       if cellid is None:
           raise ValueError("must provide cellid for pop model")
       print(f'rec1: selecting cell {cellid}')
       matching_cellid = [i for i,c in enumerate(modelspec1.meta['cellids']) if c==cellid]
       pred1 = rec1['pred'].as_continuous()[matching_cellid,:]
   else:
       pred1 = rec1['pred'].as_continuous()

   if rec2['pred'].shape[0] > 1:
       # pop model, need to extract the relevant channel:
       if cellid is None:
           raise ValueError("must provide cellid for pop model")
       matching_cellid = [i for i,c in enumerate(modelspec2.meta['cellids']) if c==cellid]
       pred2 = rec2['pred'].as_continuous()[matching_cellid,:]
   else:
       pred2 = rec2['pred'].as_continuous()

   if rec_ref['pred'].shape[0] > 1:
       # pop model, need to extract the relevant channel:
       if cellid is None:
           raise ValueError("must provide cellid for pop model")
       matching_cellid = [i for i,c in enumerate(modelspec_ref.meta['cellids']) if c==cellid]
       pred_ref = rec_ref['pred'].as_continuous()[matching_cellid,:]
   else:
       pred_ref = rec_ref['pred'].as_continuous()


   # [0, 1] is for partial correlation between pred0 and pred1,
   # accounting for pred 2. Returns a matrix similar to a correlation matrix.
   C = np.hstack((pred1.transpose(), pred2.transpose(), pred_ref.transpose()))
   pc = partial_corr(C)[0, 1]

   return pc


def test_comparison(cellid="TAR010c-15-5", batch=289, 
        modelname1="ozgf.fs100.ch18.pop-loadpop.cc20.bth-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x30.z-relu.30-wc.30xR.z-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-popspc", 
        modelname2="ozgf.fs100.ch18.pop-loadpop.cc20.bth-norm.l1-popev_wc.18x40.g-stp.40.q.s-fir.1x12x40-relu.40-wc.40x30.z-relu.30-wc.30xR.z-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-popspc",
        modelname_ref="ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-fir.3x15-lvl.1-dexp.1_init-basic",
        recname="val"):

    xf1,ctx1=load_model_xform(cellid, batch, modelname1)
    xf2,ctx2=load_model_xform(cellid, batch, modelname2)
    xfr,ctxr=load_model_xform(cellid, batch, modelname_ref)

    modelspec1 = ctx1['modelspec']
    modelspec2 = ctx2['modelspec']
    modelspec_ref = ctxr['modelspec']
    rec1 = ctx1[recname]
    rec2 = ctx2[recname]
    rec_ref = ctxr[recname]

    return modelspec1, modelspec2, modelspec_ref, rec1, rec2, rec_ref
    
