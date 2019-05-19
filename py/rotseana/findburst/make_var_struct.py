'''
Created on Jul 13, 2017

@author: Daniel Sela, Arnon Sela
'''

import numpy as np
from collections import Iterable


def _create_structured_vector(size, fields, copy=False):
    ''' create np.array of structure filled with default values

    Args:
        shape: tuple, shape of the array
        fields: list of tuples of (name, type, value)
        copy_: copy function for value assignment

    Returns:
        np.array

    Example:
        create 10 elements one dimentional array with x,y record defaults to (-1,-1)
        _create_structure( (10,), ('x', int, -1), ('y', int, -1) )
    '''
    dts = list()
    for name, type_, value in fields:
        if hasattr(value, 'shape'):
            shape = value.shape
        elif isinstance(value, Iterable):
            shape = (len(value))
        else:
            shape = None
        if shape is not None:
            dts.append((name, type_, shape))
        else:
            dts.append((name, type_))

    # dt = np.dtype([(name, type_, ) for name, type_, value in fields])
    dt = np.dtype(dts)

    values = [tuple([value if not copy else np.copy(value) for _, _, value in fields]) for _ in range(size)]
    array = np.rec.array(values, dtype=dt)

    return array, dt


def make_var_struct(nobj, nobs):
    '''
        ; Purpose:    To create structure holding summary information on variables.
        ;
        ; Inputs:
        ;    nobj -- number of objects
        ;    nobs -- number of observations
        ;
        ; Return Value:  initialized structure
        ;
        ; adopted from make_var_struct.pro idl procedure
        ;
        ; Created: BDaniel Sela, Arnon Sela
    '''
    # make observation substructure

    obs_map = [
        ('state', int, -1),
        ('dis', np.float32, -1.0),
        ('posangle', np.float32, -1.0),
        ('err', np.float32, -1.0),
        ('phot', np.float32, -1.0),
        ('photerr', np.float32, -1.0),
        ]

    all_obs, obs_dt = _create_structured_vector(nobs, obs_map)

    var_map = [
        ('name', str, " "), ('ptr', np.int64, 0),
        ('maxmag', np.float32, -1.0), ('errmaxmag', np.float32, -1.0),
        ('minmag', np.float32, -1.0), ('avgmag', np.float32, -1.0),
        ('delta', np.float32, -1.0), ('nmiss', int, 0),
        ('nobs', np.int32, 0), ('ngdobs', np.int32, 0),
        ('sdev', np.float32, -1.0), ('sdevcl', np.float32, -1.0),
        ('chisq', np.float32, -1.0), ('chisqcl', np.float32, -1.0),
        ('maxsig', np.float32, -1.0), ('pos_sdv', np.float32, -1.0),
        ('posrange', np.float32, -1.0), ('avgdev', np.float32, -1.0),
        ('skew', np.float32, -1.0,), ('kurt', np.float32, -1.0), ('duration', np.float32, -1.0),
        ('mdnerr', np.float32, -1.0), ('avgdevsig', np.float32,  -1.0),
        ('avgdevsigcl', np.float32, -1.0), ('bestdelta', np.float32, -1.0),
        ('bestsig', np.float32, -1.0), ('ival', np.float32, -1.0),
        ('ival2', np.float32, -1.0), ('obs', obs_dt, all_obs),
        ]
    # for o in var_map:
    #    print(o[0], len(o))
    all_vars, vars_dt = _create_structured_vector(nobj, var_map, copy=True) 

    return all_vars  # , vars_dt, obs_dt


if __name__ == '__main__':
    # a, vars_dt, obs_dt =make_var_struct(10,30)
    a = make_var_struct(10, 30)
    a[0].obs[0].err = 30
    a[0].obs[2].err = 50

    vars_dt = a[0].dtype
    obs_dt = a[0].obs.dtype

    ind = np.ndarray((5,), buffer=np.array([0, 1, 2, 3, 4]), dtype=np. int64)
    rec = np.rec.array(a[0].obs[ind], dtype=obs_dt)
    obs = np.where(rec.err > 0)
    print(obs, rec[obs])

    rec = np.rec.array(a[0].obs[obs], dtype=obs_dt)
    print([r.err for r in rec])
