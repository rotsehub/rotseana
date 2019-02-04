'''
Created on Jul 24, 2017

@author: Daniel Sela, Arnon Sela
'''

from rotseana.set_flags import set_flags


def calc_masks(emask, rmask):
    bad_eflags = ['ATEDGE', 'SATURATED', 'APINCOMPL']
    bad_rflags = ['NOISYPIX', 'HOTPIX', 'STRANGEPIX', 'BADPOS', 'NOTEMPL', 'PHOTSDEV']
    if not emask:
        emask = set_flags(bad_eflags, type_='EFLAGS')
    if not rmask:
        rmask = set_flags(bad_rflags, type_='RFLAGS')
    return emask, rmask
