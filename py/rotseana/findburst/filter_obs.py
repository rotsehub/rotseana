from rotseana.findburst.check_flags import check_flags
from rotseutil.make_rotse_name_sixty import make_rotse_name
# import math
# from pprint import pprint


def filter_obs(match, delta, maxsig, objid=None, chisq=None, ival=None, emask=0, rmask=0, verbose=True):
    '''
    ; Purpose:    Filter list of observations to a preliminary set of probable 
    ;    variables.  Good observations are selected for each source, and these are 
    ;    used to obtain general lightcurve characteristics for further cuts.
    ;
    ; Inputs:
    ;    match            list of source observations
    ;    delta            minimum magnitude range
    ;    maxsig            minimum significance of variation
    ;
    ; Outputs:  NONE
    ;
    ; Keywords:
    ;    chisq            minimum chi-squared per degree-of-freedom, clipped 
    ;                of most extreme well-observed value
    ;    ival            minimum modified Welch-Stetson I-value
    ;    emask            mask for extraction flags
    ;    rmask            mask for ROTSE observation flags
         objid: list of object ids
    ;
    ; Return Value:
    ;    goodobj            list of indices of candidate variables
    ;
    ; Adopted from define_flags.pro idl procedure
    ; Created:  Jul 8, 2017  Daniel Sela, Arnon sela
      Added: filter out negative RAs
    ;******************************************************************************
    '''
    import numpy as np

    if verbose:
        print('''Filtering for significant variation in good observations:
    DELTA > %s
    MAXSIG > %s''' % (delta, maxsig))

    min_nobs = 2
    if chisq is not None:
        if verbose:
            print('    CHISQ > %s' % chisq)
        min_nobs = 3
    else:
        chisq = 0.0
    if verbose:
        print('    EMASK = %s' % emask)
        print('    RMASK = %s' % rmask)
    match_m = match.field('M')[0]
    nobj = match_m.shape[0] if objid is None else len(objid)
    nobs = match_m.shape[1]
    max_delta = np.zeros(nobj, dtype=np.float32)
    max_err = np.zeros(nobj, dtype=np.float32)
    chisq_clip = np.zeros(nobj, dtype=np.float32)

    # Find objects with significant variation in good observations...

    match_flags = match.field('FLAGS')[0]
    match_rflags = match.field('RFLAGS')[0]
    match_msys = match.field('MSYS')[0]
    match_merr = match.field('MERR')[0]
    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]

    collect = {'k': list()}

    for k in range(nobj):
        # indicate progress
        if k % 200 == 0 and verbose:
            print('.', end='', flush=True)

        diff = 0.0

        ko = k if objid is None else objid[k]

        # validate good decline and recline
        if match_ra[ko] < 0:
            continue

        check_flags_emask = np.vectorize(lambda e: emask & e == 0)
        check_flags_rmask = np.vectorize(lambda e: rmask & e == 0)
        cond = np.logical_and.reduce((match_flags[ko, :] > -1,
                                      check_flags_emask(match_flags[ko, :]),
                                      check_flags_rmask(match_rflags[ko, :]),
                                      ))
        goodobs = np.where(cond)
        goodobs = goodobs[0]
        ngdobs = len(goodobs)

        if ngdobs >= min_nobs:
            diff = np.max(match_m[ko, goodobs])-np.min(match_m[ko, goodobs])
            if diff > delta:
                collect['k'].append(ko)
                maxdelta = np.zeros(ngdobs, dtype=np.float32)
                maxerr = np.zeros(ngdobs, dtype=np.float32)
                sum_merr_msys = match_merr[ko, goodobs]**2.0 + (match_msys[ko, goodobs]/200.0)**2.0
                err = np.sqrt(sum_merr_msys)
                err2 = err**2.0

                for l in range(ngdobs):
                    # gl = goodobs[l]
                    # match_obj = match_m[k, ll]
                    diffs = np.abs(match_m[ko, goodobs[l]] - match_m[ko, goodobs])
                    sig = diffs / np.sqrt(err2 + err2[l])
                    i = np.where(np.logical_and.reduce((diffs > delta, sig > maxsig, sig > maxerr[l])))
                    i = i[0]
                    if len(i) > 0:
                        iobs = np.argmax(sig[i])
                        maxerr[l] = sig[i][iobs]
                        maxdelta[l] = diffs[i][iobs]
                        # maxdelta[l] = diffs[iobs]

                iobs = np.argmax(maxerr)
                max_err[k] = maxerr[iobs]
                max_delta[k] = maxdelta[iobs]
                avgmag = np.mean(match_m[ko, goodobs])
                iobs = np.argmax(np.abs(match_m[ko, goodobs]-avgmag))
                i = np.where(match_m[ko, goodobs] != match_m[ko, goodobs[iobs]])
                i = i[0]
                dof = float(ngdobs) - 2.0
                if len(i) >= 2:
                    avgmag_clip = np.mean(match_m[ko, goodobs[i]])
                    chisq_clip[k] = np.sum(((match_m[ko, goodobs[i]] - avgmag_clip)/err[i])**2.0)/dof

    # pprint(collect)
    cond = np.logical_and.reduce((max_delta > delta, max_err > maxsig, chisq_clip > chisq))
    goodobj = np.where(cond)
    goodobj = goodobj[0]
    ngdobj = len(goodobj)
    if objid is not None:
        goodobj = [objid[k] for k in goodobj]
    if verbose:
        print('\nfilter_obs completed: %s' % ngdobj)

    return goodobj
