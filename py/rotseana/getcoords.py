#!/usr/bin/env python3
'''
Created on Jul 18, 2017

@author: Daniel Sela, Arnon Sela
'''

from rotseutil.make_rotse_name_sixty import make_rotse_name
from rotseana.read_data_file import read_data_file
from rotseana.filter_obs import filter_obs
from rotseana.calc_masks import calc_masks
import os


def getcoords(file, mindelta, minsig, minchisq, fits_index=1, emask=None, rmask=None, verbose=False, filter_=True):
    ''' Fetch j2000 coordinates from match file

    Args:
        file: path to file or match or fits file
    '''
    emask, rmask = calc_masks(emask, rmask)

    matches = file
    for match_file in matches:
        match, tele = read_data_file(match_file, fits_index)

        goodobj = None
        if filter_:
            if minchisq is not None:
                goodobj = filter_obs(match, mindelta, minsig, chisq=minchisq, emask=emask, rmask=rmask, verbose=verbose)
            else:
                goodobj = filter_obs(match, mindelta, minsig, emask=emask, rmask=rmask, verbose=verbose)

        match_ra = match.field('RA')[0]
        match_dec = match.field('DEC')[0]

        ra_dec = zip(match_ra, match_dec)

        filename_prefix = ''
        # if len(matches) > 1:
        # short_name=os.path.basename(match_file)
        filename_prefix = "%s " % os.path.realpath(match_file)

        objid = -1
        for ra, dec in ra_dec:
            objid += 1  # this must be here due to the continue within try clause
            if not filter_ or objid in goodobj:
                try:
                    rotse_name = make_rotse_name(ra, dec, tele=tele)
                except Exception:
                    if not filter_:
                        print('Warning: skipping object %d in %s; it has bad coordinates (ra, dec): %s, %s' % (objid, match_file, ra, dec))
                    continue

                print("%s%s %s" % (filename_prefix, objid, rotse_name),)
    return 0
