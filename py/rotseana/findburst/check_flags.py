'''
Created on Jul 12, 2017

@author: Daniel Sela
'''

'''
Adopted from check_flags.pro IDL procedure
'''

from rotseana.findburst.set_flags import set_flags


def check_flags(flagstr, flagword, type_=None):
    '''
    ;+
    ; NAME: check_flags
    ;
    ; CALLING SEQUENCE:     check_flags,flagstr,flagword,type
    ;
    ; INPUTS:       flagstr: list of flags, or flag mask
    ;         flagword: word to check
    ;
    ; Keywords:    type: type of flags to consult ('EFLAGS' or 'RFLAGS')
    ;
    ; Return Value: new word indicating which bits were on.
    ;
    ; Created:  ul 12, 2017  Daniel Sela
    ;******************************************************************************
    '''
    if type_ is None:
        print('Flag type not set.')
        return -1

    val = set_flags(flagstr, type_=type_) & flagword

    return val
