#!/usr/bin/env python3
'''
Created on Jul 18, 2017

@author: Daniel Sela, Arnon Sela
'''
import numpy as np
from rotseutil.findburst_utils.create_struct import create_struct


def single_phase(datfile, period, max_freq=3.0, lc_dir=None, no_avg=None, per=None):
    type_map = [
        ('datfile', str),
        ('period', np.zeros((15,), dtype=np.float64)),
        ('chisq', np.zeros((15,), dtype=np.float64)),
        ('cycles', np.zeros((15,), dtype=np.int)),
        ('p1', np.zeros((15,), dtype=np.float32)),
        ('p2', np.zeros((15,), dtype=np.float32)),
        ('p3', np.zeros((15,), dtype=np.float32)),
        ]
    period = create_struct(type_map)
    return period
