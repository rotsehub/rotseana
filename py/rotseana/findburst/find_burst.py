#!/usr/bin/env python3
'''
Created on Jul 22, 2017

@author: Daniel Sela, Arnon Sela
'''
import logging
import numpy as np
from rotseana.findburst.read_data_file import read_data_file, get_data_file_rotse
from rotseana.findburst.filter_obs import filter_obs
from rotseana.findburst.lcplot2 import lcplot2
from rotseana.findburst.print_var import print_var
from rotseana.findburst.recovery import Recovery
from rotseana.findburst.calc_masks import calc_masks
from rotseana.findburst.calc_var import calc_var

logger = logging.getLevelName(__name__)

def find_burst(match, mindelta, minsig, fits_index=1, minchisq=None, refra=None, refdec=None, radius=None, objid=None, rotse=None, log=None, emask=None, rmask=None, recoverable=False, recover=False, recdir=None, verbose=False):
    '''
    ; NAME:    find_burst
    ;
    ; CALLING SEQUENCE:    var = find_burst(match, mindelta, minsig)
    ;
    ; INPUTS:
    ;    match = match structure produced by ID:'s make_match_struct and restored by scipy.io.readsav
    ;        If match is a string, assume it is a file with match data to restore.
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
    ; Based on find_burst.pro IDL procedure
    ; Created:  4/23/99-12/07/00  Bob Kehoe
    ; Adopted to Python:  Jul 8, 2017  Daniel Sela, Arnon sela
    ;******************************************************************************

    '''
    match_file = None
    if isinstance(match, str):
        # assume this is a file to restore
        match_file = match
        # match=readsav(match)['match']
        match, tele = read_data_file(match_file, fits_index)
        if not rotse:
            rotse = get_data_file_rotse(match_file)

    print("step 1")
    match_m = match.field('M')[0]
    nobj = match_m.shape[0]
    totobs = match_m.shape[1]
    print(nobj, totobs)
    if verbose:
        print('Total number of objects found in ', totobs, ' epochs = ', nobj)

    if not radius:
        radius = 0.001  # - radius from given ra, dec

    emask, rmask = calc_masks(emask, rmask)
    print(emask, rmask)
    
    # see if can restore filter_obs
    goodobj_rec = None
    goodobj = None
    if recoverable:
        goodobj_rec = Recovery('goodobj', match_file)
        if recover:
            goodobj = goodobj_rec.load()

    if goodobj is not None:
        # recovered from previously computed
        pass
    elif refra is not None and refdec is not None:
        match_ra = match.field('RA')[0]
        match_dec = match.field('DEC')[0]
        cond = np.logical_and.reduce((np.abs(match_ra-refra) < 0.001,
                                      np.abs(match_dec-refdec) < 0.001))
        goodobj = np.where(cond)
        goodobj = goodobj[0]
        if verbose:
            print(goodobj)
    elif objid is not None:
        goodobj = objid  # single specified object
        nobs = len(match_m[objid, :])  # no. of observation for that objid
        if verbose:
            print("Photometry completed for this object ID", objid)
            print("Printing result")
        for ii in range(nobs):
            match_merr = match.field('MERR')[0]
            match_jd = match.field('JD')[0]
            if verbose:
                print(match_jd[ii], match_m[objid, ii], match_merr[objid, ii])  # This is probably enough.
            # Do we want to directly save to a file??
    elif minchisq is not None:
        goodobj = filter_obs(match, mindelta, minsig, chisq=minchisq, emask=emask, rmask=rmask)
    else:
        goodobj = filter_obs(match, mindelta, minsig, emask=emask, rmask=rmask)

    # save goodobj to use in later run istead of recomputing
    if goodobj_rec:
        goodobj_rec.store(goodobj)

    nvar = len(goodobj)
    if verbose:
        print('Number of variables found = ', nvar)
    if nvar == 0:
        return None

    # see if can restore filter_obs
    var_rec = None
    var = None
    if recoverable:
        var_rec = Recovery('var', match_file)
        if recover:
            var = var_rec.load()

    if var is None:
        var = calc_var(match=match, nvar=nvar, totobs=totobs, goodobj=goodobj, mindelta=mindelta, minsig=minsig, emask=emask, rmask=rmask, verbose=verbose)
        if match_file and var_rec and var is not None:
            var_rec.store(var)

    if log is not None:
        txtfile = log + '.txt'
        if verbose:
            print('\nomitting print_var')
        # print_var( match,var,fname=txtfile)

        pdffile = log + '.pdf'

        lcplot2(pdffile=pdffile, struct=match, index=np.arange(totobs), obj=var.ptr, good=True, syserr=True, offset=True, emask=emask, rmask=rmask, nowait=True, grid=(2, 3),)

    return var

