'''
Created on Jul 21, 2017

@author: Daniel SEla, Arnon Sela
'''

import numpy as np
from rotseana.iter_mean import iter_mean


def odd_even_indexes(ngd):
    # we need an even number of observations:
    if ngd % 2 > 0:
        ngd -= 1
    pair1 = np.arange(ngd/2)*2
    pair2 = pair1 + 1

    pair1 = pair1.astype(dtype=int)
    pair2 = pair2.astype(dtype=int)

    return pair1, pair2


def ivalue(mags, merr, mn_iter=0, robust=False):
    '''
    ;+
    ; NAME: IVALUE
    ;
    ; PURPOSE: Calculate the Welch/Stetson ivalue of a light curve.
    ;
    ; CALLING SEQUENCE: ivalue,mags,merr,ivalue,mn_iter=mn_iter,/robust
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
    ;         /robust - set this to calculate the robust Ivalue
    ;                   of Stetsons 1996 paper rather than the original
    ;                   ivalue from W/S 1993 paper. The robust version
    ;                   reduces the effect of outliers.
    ; OUTPUTS:
    ;         ivalue - the WS ivalue.
    ;
    ; OPTIONAL OUTPUTS:
    ;
    ; NOTES: If Ivalues are too high, check the object errors, they
    ;        may be too low.
    ;
    ; EXAMPLE:  To calculate the ivalue of object 15 in a match structure:
    ;
    ;         IDL> ivalue, match.m(*,15), match.merr(*,15), ival
    ;
    ; PROCEDURES CALLED: ITER_MEAN
    ;
    ; Adopted from IDL procedure
    ;-

    '''

    num = len(mags)

    gd = np.where(mags != -1)
    gd = gd[0]
    ngd = len(gd)
    if ngd < 2:
        ivalue = -99
        return ivalue

    pair1, pair2 = odd_even_indexes(ngd)
    mn, _, _ = iter_mean(mags[gd], merr[gd], niter=mn_iter)
    if not robust:
        con = np.sqrt(1.0/((ngd/2.0)*((ngd/2.0)-1.0)))
        chg = (mags[gd]-mn)/merr
        ivalue = con*np.sum(chg[gd[pair1]]*chg[gd[pair2]])

    else:
        chg = np.sqrt(ngd/(ngd-1))*(mags[gd]-mn)/merr
        kind = (1.0/ngd)*np.sum(np.abs(chg[gd]))/np.sqrt((1.0/ngd)*np.sum(chg[gd]**2))
        chgarr = chg[gd[pair1]]*chg[gd[pair2]]
        chgarr = chgarr[np.nonzero(chgarr)]
        jind = np.sum(chgarr/np.sqrt(np.abs(chgarr)))/ngd
        ivalue = jind*kind/0.798

    return ivalue
