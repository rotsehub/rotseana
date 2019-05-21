#!/usr/bin/env python3

from collections import namedtuple
import re
import numpy as np
from rotseutil.findburst_utils.coords_operations import decim_2_sec

finder = re.compile(r".*J(?P<x>(\d+\.\d+))(?P<y>([+-]\d+\.\d+))")
Coord = namedtuple("Coord", ["file", "objid", "jname", "x", "y"])

SEC_24H = 24.0*3600.0


def read_records(name):
    global finder, SEC_24H
    records = list()
    with open(name, 'r') as f:
        lines = f.read()
    for line in lines.split('\n'):
        record = line.split(' ')
        '''
        001004_sky0001_1b_match.datc 30548 ROTSE1 J000103.394+225135.314
        '''
        if len(record) < 3:
            continue
        m = finder.match(record[3])
        x = decim_2_sec(float(m.group('x'))) % SEC_24H
        y = decim_2_sec(float(m.group('y')))
        item = Coord(record[0], record[1], record[3], x, y)
        records.append(item)
    return records


def matchcoords(file, coord, error, verbose):
    ''' search file for coordinates within error range.

    Args:
        file: coords file of the following structure:
            001004_sky0001_1b_match.datc 30548 ROTSE1 J000103.394+225135.314
        coords: coordinate to match
        error: +/- range to match (seconds)
        verbose: show what is being done.

    Returns:
        mape of requested coordinate and matching coordinates.
    '''
    global finder, SEC_24H

    records = read_records(file)
    xs = np.array([i.x for i in records])
    ys = np.array([i.y for i in records])
    objids = [i.objid for i in records]
    source = [i.file for i in records]
    jname = [i.jname for i in records]

    # coords=coord
    result = dict()

    # for coord in coords:
    m = finder.match(coord)
    try:
        # convert 24 to 00 (in seconds)
        x = decim_2_sec(float(m.group('x'))) % SEC_24H
        y = decim_2_sec(float(m.group('y')))
    except Exception:
        raise Exception('supplied coordinate is malformed %s; ' % (coord,))
        # continue
    # this is computed to later allow cyclic range of 000000-240000
    x_minus_e = (x-error + SEC_24H) % SEC_24H
    x_plus_e = (x+error) % SEC_24H

    # look for coords in error range
    if x_minus_e < x_plus_e:
        condition = np.logical_and.reduce((x_minus_e <= xs, xs <= x_plus_e,))
    else:
        # in this case x is on the near the edge an either x_minus_e or x_plus_e failed over the fence
        # hence, the or statement.
        condition = np.logical_or.reduce((x_minus_e <= xs, xs <= x_plus_e,))

    condition = np.logical_and.reduce((condition, y-error <= ys, ys <= y+error))
    ids = np.where(condition)
    ids = ids[0]

    # result[coord]=["%s %s %s" %(source[i], objids[i], jname[i]) for i in ids]
    result = [(source[i], objids[i], jname[i]) for i in ids]
    return result
