#!/usr/bin/env python3
'''
Created on Jul 23, 2017

@author: Daniel Sela, Arnon Sela
'''

from rotseana.findburst.matchcoords import matchcoords
from rotseana.findburst.findburst_gd import findburst_gd
from rotseana.findburst.read_data_file import get_data_file_rotse

import itertools
import matplotlib
matplotlib.use('PDF')
# import os


def matchcoords_gd(coord, coord_file, mindelta, minchisq, minsig, fits_index, error, with_reference=False, plot=False, log=None, quiet=False, verbose=False):

    # coord_ref=coord
    result = list()

    # for f in file:
    # create pdf title on page
    # short_name=os.path.basename(coord_file)

    # matchcoords(file, coord, error, verbose)
    match_res = matchcoords(file=coord_file, coord=coord, error=error, verbose=verbose)
    objids = list()
    lineno = 0
    for match in match_res:
        lineno += 1
        if len(match) == 3:
            match_file, id_, jname = match
        else:
            raise Exception("missing values (only %d found); lineno: %d" % (len(match), lineno,))

        objids.append((match_file, id_))

    data = sorted(objids)
    for match_file, objids in itertools.groupby(data, lambda x: x[0]):
        rotse = get_data_file_rotse(match_file)
        objid = [int(id_) for _, id_ in objids]
        answer = findburst_gd(match=match_file, mindelta=mindelta, minsig=minsig, fits_index=fits_index, minchisq=minchisq, objid=objid, rotse=rotse, verbose=verbose)
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
            sout = '%s %d %s' % (coord_file, objid, sout)
        if csv:
            csv.write('%s\n' % sout)
        if not quiet:
            print(sout)

    if csv:
        csv.close()
