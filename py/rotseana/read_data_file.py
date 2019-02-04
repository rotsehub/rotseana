'''
Created on Jul 19, 2017

@author: Daniel Sela, Arnon Sela
'''

from scipy.io import readsav
from astropy.io import fits
import os


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
    ''' Reads fits and match files into record

    Args:
        file: path to match or fits file
    '''
    if not os.path.isfile(file):
        raise Exception("file not found: %s" % (file,))

    file_ext = file.rpartition('.')[2]
    if file_ext == 'fit':
        match, rotse = read_fits_file(file, fits_index)
    else:
        match, rotse = read_match_file(file)
    return match, rotse
