#!/usr/bin/env python3
'''
Created on Jul 23, 2017

@author: Daniel Sela, Arnon Sela
'''

from rotseana.findburst.read_data_file import read_data_file
from rotseana.findburst.set_flags import set_flags
from rotseana.findburst.filter_obs import filter_obs
from rotseana.findburst.calc_var import calc_var
from rotseana.findburst.lcplot2 import lcplot2
from rotseana.findburst.calc_masks import calc_masks
import numpy as np


def findburst_gd(match, mindelta, minsig, pdffile=None, fits_index=1, minchisq=None, objid=None, refra=None, refdec=None, rotse=None, log=None, plot=False, emask=None, rmask=None, recoverable=False, recover=False, recdir=None, verbose=False):
    '''
    ;+
    ; NAME:    findburst_gd
    ;
    ; CALLING SEQUENCE:    var = findburst_gd(match, mindelta, minsig)
    ;
    ; INPUTS:
    ;    match = match structure produced by make_match_struct
    ;    mindelta = threshold for total variation
    ;    minsig = number of stddev (stat.+sys.) for variation
    ;
    ; Keywords:
    ;    minchisq = minimum clipped chis-squared of good observations
    ;    emask = mask for extraction flags
    ;    rmask = mask for ROTSE observation flags
    ;    log = toggles whether to produce output text and postscript
    ;
    ; Return Value:    structure containing summary information on transient candidates
    ;
    ; PROCEDURE:    Searches for brief optical transients in a matched object
    ;        list from ROTSE trigger response data.
    ;
    ; Adopted from idl procedure
    ;******************************************************************************
    '''

    # Initialization

    # TODO: find_burst has the same initialization: consolidate
    match_file = None
    if isinstance(match, str):
        # assume this is a file to restore
        match_file = match
        match, tele = read_data_file(match_file, fits_index)

    match_m = match.field('M')[0]
    nobj = match_m.shape[0]
    totobs = match_m.shape[1]
    if verbose:
        print('Total number of objects found in ', totobs, ' epochs = ', nobj)

    emask, rmask = calc_masks(emask, rmask)

    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]

    if objid is not None:
        goodobj = np.array(objid)  # specified objects
    elif refra is not None and refdec is not None:
        match_ra = match.field('RA')[0]
        match_dec = match.field('DEC')[0]
        cond = np.logical_and.reduce((np.abs(match_ra-refra) < 0.001,
                                      np.abs(match_dec-refdec) < 0.001))
        goodobj = np.where(cond)
        goodobj = goodobj[0]
    elif minchisq is not None:
        goodobj = filter_obs(match, mindelta, minsig, chisq=minchisq, emask=emask, rmask=rmask)
    else:
        goodobj = filter_obs(match, mindelta, minsig, emask=emask, rmask=rmask)

    if verbose:
        print('rotse', rotse)

    nvar = len(goodobj)
    if verbose:
        print('Number of variables found = ', nvar)

    # Calculate lightcurve quantities from good observations.
    var = calc_var(match=match, nvar=nvar, totobs=totobs, goodobj=goodobj, mindelta=mindelta, minsig=minsig, emask=emask, rmask=rmask, verbose=verbose)

    '''
    if log is not None:
        txtfile = log + '.txt'
        if verbose: print('\nomitting print_var')

        pdffile = log + '.pdf'

        #if rotse is not None:
        lcplot2(pdffile=pdffile, struct=match,index=np.arange(totobs),obj=var.ptr,good=True,syserr=True,offset=True,emask=emask,rmask=rmask,nowait=True, grid=(2,3), )
        #else:
        #    lcplot2(pdffile=pdffile, struct=match,index=np.arange(totobs),obj=var.ptr,good=True,syserr=True,offset=True,emask=emask,rmask=rmask,nowait=True, grid=(2,3), )
    '''

    match_merr = match.field('MERR')[0]
    match_jd = match.field('JD')[0]

    result = list()
    for k in range(len(var)):
        h = np.where(match_m[var[k].ptr, :] > -1.0)
        h = h[0]
        objid = var[k].ptr

        for i in h:
            result.append((objid, match_m[objid, i], match_jd[i], match_merr[objid, i]))
    return result
