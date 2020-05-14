import glob
import os
import numpy as np

from scipy.io import readsav
from astropy.io import fits

def read_fits_file(file, fits_index=1):
    try:
        hdus = fits.open(file, memmap=True)
        hdus_ext = hdus[fits_index]
        match = hdus_ext.data
    except Exception as e:
        raise Exception("cannot read fits data from file: %s" % (file,)) from e
    return match, 'ROTSE3'


def read_match_file(file, *args, **kwargs):
    try:
        match = readsav(file)['match']
    except Exception as e:
        raise Exception("cannot read match data from file: %s" % (file,)) from e
    return match, 'ROTSE1'


def get_data_file_rotse(file):
    if not os.path.isfile(file):
        raise Exception("file not found: %s" % (file,))
    
    file_ext = file.rpartition('.')[2]
    if file_ext == 'fit':
        return 3
    else:
        return 1


def read_data_file(file, fits_index=1, tmpdir='/tmp'):

    if not os.path.isfile(file):
        raise Exception("file not found: %s" % (file,))
    
    file_ext = file.rpartition('.')[2]
    if file_ext == 'fit':
        match, rotse = read_fits_file(file, fits_index)
    else:
        match, rotse = read_match_file(file)
    return match, rotse

def getlc(match, refra, refdec):
    
    match_file = None
    if isinstance(match, str):
        match_file = match
        match, tele = read_data_file(match_file)

    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]
    cond = np.logical_and.reduce((np.abs(match_ra-refra) < 0.001,
                                      np.abs(match_dec-refdec) < 0.001))
    goodobj = np.where(cond)
    objid = goodobj[0]

    match_merr = list(match.field('MERR')[0][objid][0])
    match_m = (list(match.field('M')[0][objid][0]))
    match_jd = match.field('JD')[0]

    curve = list()
    for q in range(len(match_jd)):
        epoch = match_jd[q]
        mag = match_m[q]
        magerr = match_merr[q]
        point = (epoch,mag,magerr)
        curve.append(point)

    lc = list()
    for i in curve:
        if i[1] > 0:
            lc.append(i)

    return lc

def getobjids(match, refra, refdec, radius):

    match_file = None
    if isinstance(match, str):
        match_file = match
        match, tele = read_data_file(match_file)

    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]
    cond = np.logical_and.reduce((np.abs(match_ra-refra) < radius,
                                      np.abs(match_dec-refdec) < radius))
    goodobj = list(np.where(cond)[0])

    return goodobj

def getcoords(match, objid):

    match_file = None
    if isinstance(match, str):
        match_file = match
        match, tele = read_data_file(match_file)

    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]

    ras = list()
    for i in match_ra:
        ras.append(i)

    decs = list()
    for i in match_dec:
        decs.append(i)
   
    result = [ras[objid],decs[objid]]

    return result
