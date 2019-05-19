'''
Created on Jul 28, 2017

@author: Daniel Sela, Arnon Sela
'''

import numpy as np

from rotseana.findburst.make_var_struct import make_var_struct
from rotseutil.make_rotse_name_sixty import make_rotse_name
from rotseutil.conv2deg import conv2deg
from rotseutil.lightcurve import lightcurve
from rotseana.findburst.ivalue import ivalue


def calc_var(match, nvar, totobs, goodobj, mindelta, minsig, emask=None, rmask=None, verbose=False):
    match_m = match.field('M')[0]
    match_flags = match.field('FLAGS')[0]
    match_rflags = match.field('RFLAGS')[0]
    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]
    match_dra = match.field('DRA')[0]
    match_ddec = match.field('DDEC')[0]
    match_merr = match.field('MERR')[0]
    match_jd = match.field('JD')[0]
    match_msys = match.field('MSYS')[0]
    try:
        match_stat = match.field('STAT')[0]
    except Exception:
        match_stat = None  # assume not a ROTSE1 file
    var = make_var_struct(nvar, totobs)
    obs_dt = var[0].obs.dtype

    for k in range(nvar):
        # show progress
        if verbose and k % 50 == 0:
            print('.', end='', flush=True)

        # Get identification for this variable
        ptr = var[k].ptr = goodobj[k]
        var[k].name = make_rotse_name(match_ra[ptr], match_dec[ptr])

        # Find observations where source was not seen but should have been

        count = 0
        if match_stat is not None:
            cond = np.logical_and.reduce((match_m[ptr, :] == -1.0,
                                          var[k].avgmag < (match_stat.m_lim-1.0),))
            misses = np.where(cond)
            misses = misses[0]
            count = len(misses)
            var[k].nmiss = count
            for miss in misses:
                var[k].obs[miss].state = 1

        # Find observations where source seen, and calculate position and duration information

        okobs = np.where(match_flags[ptr, :] > -1)
        okobs = okobs[0]
        count = len(okobs)
        var[k].nobs = count
        if verbose:
            print('count of okobs:', count)

        if count > 0:
            for i in okobs:
                var[k].obs[i].state = 2
            var[k].duration = max(match_jd[okobs]) - min(match_jd[okobs])

        conv2deg_vectorize = np.vectorize(conv2deg)
        dra = conv2deg_vectorize(match_dra[ptr, :])
        ddec = conv2deg_vectorize(match_ddec[ptr, :])

        # TODO: must be a better way to make this vector assignment
        dis = np.sqrt(dra**2.0 + ddec**2.0)
        posangle = np.arctan(ddec/dra)
        for i in range(len(dis)):
            var[k].obs[i].dis = dis[i]
            var[k].obs[i].posangle = posangle[i]
        var[k].pos_sdv = (np.std(dra)+np.std(ddec)) / 2.0
        var[k].posrange = (np.max(dra)-np.min(dra)) > (np.max(ddec)-np.min(ddec))

        # Obtain list of good observations and use to calculate lightcurve information

        check_flags_emask = np.vectorize(lambda e: emask & e == 0)
        check_flags_rmask = np.vectorize(lambda e: rmask & e == 0)
        cond = np.logical_and.reduce((match_flags[ptr, :] > -1,
                                      check_flags_emask(match_flags[ptr, :]),
                                      check_flags_rmask(match_rflags[ptr, :]),
                                      ))
        gdobs = np.where(cond)
        gdobs = gdobs[0]
        count = len(gdobs)
        var[k].ngdobs = count
        if count > 0:
            for i in gdobs:
                var[k].obs[i].state = 3
            sum_merr_msys = match_merr[ptr, gdobs]**2 + (match_msys[ptr, gdobs]/200.0)**2.0
            err = np.sqrt(sum_merr_msys)
            for i in range(len(gdobs)):
                var[k].obs[gdobs[i]].err = err[i]**2.0
            kth_var = var[k]
            obs = np.rec.array(var[k].obs[gdobs], dtype=obs_dt)
            errobs = [i.err for i in obs]
            # errobs=np.ndarray((len(errobs),), buffer=np.array(errobs),)
            errobs = np.array(errobs)
            if match_stat is not None:
                lightcurve(match_m[ptr, gdobs], errobs, match_stat[gdobs].m_lim,
                           mindelta, minsig, kth_var)

            var[k] = kth_var
            ival = ivalue(match_m[ptr, gdobs], errobs, mn_iter=0)
            var[k].ival = ival
            ival = ivalue(match_m[ptr, gdobs], errobs, mn_iter=4, robust=True)
            var[k].ival2 = ival
    if verbose:
        print('')  # ending show progress
    return var
