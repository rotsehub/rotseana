#!/usr/bin/env python3
'''
Created on Jul 18, 2017

@author: Daniel Sela, Arnon Sela
'''

import numpy as np
import re

from rotseana.findburst.calc_masks import calc_masks
from rotseana.findburst.calc_var import calc_var
from rotseutil.findburst_utils.coords_operations import decim_2_sec
from rotseana.findburst.filter_obs import filter_obs
from rotseana.findburst.lcplot2 import lcplot2
from rotseutil.findburst_utils.make_rotse_name_sixty import make_rotse_name
from rotseana.findburst.read_data_file import read_data_file


# matching "J110526.404+501802.085"
finder = re.compile(r".*J(?P<x>(\d+\.\d+))(?P<y>([+-]\d+\.\d+))")

SEC_24H = 24.0*3600.0


def get_coord_from_file(file, match, tele, filter_=True):
    global finder, SEC_24H
    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]

    ra_dec = zip(match_ra, match_dec)
    xy = list()
    xs = list()
    ys = list()
    objids = list()

    objid = - 1
    for ra, dec in ra_dec:
        objid += 1

        try:
            rotse_name = make_rotse_name(ra, dec, tele=tele)
        except Exception:
            if not filter_:
                msg = ('Warning: skipping object %d in %s;'
                       'it has bad coordinates'
                       '(ra, dec): %s, %s')
                print(msg % (objid, file, ra, dec))
            continue

        f = finder.match(rotse_name)
        x = decim_2_sec(float(f.group('x'))) % SEC_24H
        y = decim_2_sec(float(f.group('y')))
        xy.append(rotse_name)
        xs.append(x)
        ys.append(y)

        # objid == index only of all RA, DEC are good, otherwise some may be skipped
        # hence, we need to keep track.
        objids.append(objid)

    xy = np.array(xy)
    xs = np.array(xs)
    ys = np.array(ys)

    return xy, xs, ys, objids


def findcoords(file, coord, mindelta=0.1, minsig=1.0, minchisq=2.0, fits_index=1,
               error=2.0, emask=None, rmask=None, verbose=False, plot=None,
               filter_=True):
    ''' Fetch j2000 coordinates from match file

    Args:
        file: path to match or fit file
        fits_index: in case of FITS, with extension to read
        coord: coordinate to look for

    Algorithm:
        1. For each file, fetch its related coords (from its RA/DEC)
        2. Make a search criteria according to requested coordinates and error
    '''

    global finder, SEC_24H

    emask, rmask = calc_masks(emask, rmask)

    f = finder.match(coord)
    try:
        # convert 24 to 00 (in seconds)
        x = decim_2_sec(float(f.group('x'))) % SEC_24H
        y = decim_2_sec(float(f.group('y')))
    except Exception:
        print('Error: supplied coordinate is malformed %s; ' % (coord,))
        return None

    # this is computed to later allow cyclic range of 000000-240000
    x_minus_e = (x-error + SEC_24H) % SEC_24H
    x_plus_e = (x+error) % SEC_24H
    # shift=(0 if x_min_e > 0 else -x_min_e) + (0 if x_plus_e < sec_24 else sec_24-x_plus_e)
    # x_min_e+=shift
    # x_plus_e+=shift

    if plot is not None:
        pdffile = plot if plot.endswith('.pdf') else plot+'.pdf'

    matches = file
    result = list()
    nfiles = len(matches)
    file_i = 0
    for match_file in matches:
        file_i += 1
        match, tele = read_data_file(match_file, fits_index)
        xy, xs, ys, objids = get_coord_from_file(match_file, match, tele, filter_)

        # query file coords for similar coords as requested
        # look for coords in error range
        if x_minus_e < x_plus_e:
            condition = np.logical_and.reduce((x_minus_e <= xs, xs <= x_plus_e,))
        else:
            condition = np.logical_or.reduce((x_minus_e <= xs, xs <= x_plus_e,))

        condition = np.logical_and.reduce((condition, y-error <= ys, ys <= y+error))
        ids = np.where(condition)
        ids = ids[0]

        # we need to convert ids to objids
        selected = [objids[i] for i in ids]

        name_map = dict(zip(selected, xy[ids]))
        if not filter_:
            goodobj = selected
        elif minchisq is not None:
            goodobj = filter_obs(match, mindelta, minsig, objid=selected,
                                 chisq=minchisq, emask=emask, rmask=rmask,
                                 verbose=verbose)
        else:
            goodobj = filter_obs(match, mindelta, minsig, objid=selected,
                                 emask=emask, rmask=rmask, verbose=verbose)

        # only if there are more than one file, add file to entries.
        if len(matches) > 1:
            objids = [(match_file, id_, name_map[id_]) for id_ in goodobj]
        else:
            objids = [(id_, name_map[id_]) for id_ in goodobj]

        result.extend(objids)

        nvar = len(goodobj)
        # if verbose: print('Number of variables found = ',nvar)
        if plot and nvar > 0:
            # Calculate lightcurve quantities from good observations.
            match_m = match.field('M')[0]
            totobs = match_m.shape[1]
            var = calc_var(match=match, nvar=nvar, totobs=totobs,
                           goodobj=goodobj, mindelta=mindelta, minsig=minsig,
                           emask=emask, rmask=rmask, verbose=verbose)
            lcplot2(pdffile=pdffile, struct=match, struct_file=match_file,
                    index=np.arange(totobs), obj=var.ptr, good=True, syserr=True,
                    offset=True, emask=emask, rmask=rmask, nowait=True,
                    grid=(2, 3), close_pdf=file_i == nfiles)
    return result


if __name__ == '__main__':
    findcoords(['/var/acrisel/sand/vstars/vstars/dat/000409_xtetrans_1a_match.dat',
                '/var/acrisel/sand/vstars/vstars/dat/rphot_match_130801.fit'],
               'J235959.455+321233.661')
