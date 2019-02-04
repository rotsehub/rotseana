#!/usr/bin/env python3
'''
Created on Jul 23, 2017

@author: Daniel Sela, Arnon Sela
'''

from rotseana.findcoords import findcoords
from rotseana.findburst_gd import findburst_gd
from rotseana.read_data_file import get_data_file_rotse

import os
import matplotlib
matplotlib.use('PDF')


def findcoords_gd(coord, file, mindelta, minchisq, minsig, fits_index, error, with_reference=False, plot=False, log=None, quiet=False, verbose=False):
    coord_ref = coord
    result = list()

    for f in file:
        # create pdf title on page
        short_name = os.path.basename(f)
        rotse = get_data_file_rotse(f)
        coords = findcoords(file=[f], coord=coord_ref, mindelta=mindelta, minsig=minsig, minchisq=minchisq, fits_index=fits_index, error=error, verbose=verbose)
        objids = list()
        for coord in coords:
            if len(coord) > 2:
                match_file, id_, name = coord
            else:
                id_, name = coord
            objids.append(id_)

        if len(objids) > 0:
            answer = findburst_gd(match=f, mindelta=mindelta, minsig=minsig, fits_index=fits_index, minchisq=minchisq, objid=objids, rotse=rotse, verbose=verbose)
            result.extend(answer)

    if verbose:
        print('number of good obs found: %d' % (len(result),))
    csv = None
    if log is not None and len(result) > 0:
        csvfile = log + '.txt' if not log.endswith('.txt') else log
        csv = open(csvfile, 'w')
        if verbose:
            print('writing %s' % csvfile)

    for objid, mag, jd, merr in result:
        sout = '%s %s %s' % (jd, mag, merr)
        if with_reference:
            sout = '%s %d %s' % (short_name, objid, sout)
        if csv:
            csv.write('%s\n' % sout)
        if not quiet:
            print(sout)

    if csv:
        csv.close()
