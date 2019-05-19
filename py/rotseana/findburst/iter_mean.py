'''
Created on Jul 21, 2017

@author: arnon
'''

import numpy as np


def iter_mean(mags, merr, niter=4):
    '''
    ; NAME: ITER_MEAN
    ;
    ; INPUTS: mags - array of magnitudes from light curve. Values of -1
    ;                will be ignored.
    ;         merr - array of magnitude errors.
    ;
    ; OPTIONAL INPUTS:
    ;         mn_iter - number of iterations for iterative mean.
    ;                   The iterative mean reduces the effect of
    ;                   single outliers on the mean.
    ;                   {def=0 no robust, def=4 robust}
    '''

    nomo = np.where(mags > 0,)
    nomo = nomo[0]
    pts = len(nomo)
    if pts != 0:
        chg = np.zeros(pts, float)
        w = 1/(merr[nomo]**2)
        it = 0
        while it <= niter:
            wtot = np.sum(w)
            mean1 = np.sum(mags[nomo]*w)/wtot
            chg = (mags[nomo]-mean1)/merr[nomo]
            w = (1.0/(1.0+(chg/2.0)**2))/(merr[nomo]**2)
            it = it+1

        mnerr = np.sqrt(np.sum(w**2*chg**2)/wtot**2)

        return mean1, mnerr, pts
