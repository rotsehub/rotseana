def define_flags(name):
    '''
    ;+
    ; NAME: define_flags
    ;
    ; CALLING SEQUENCE:     define_flags,name
    ;
    ; INPUTS:       name: type of flags to output (ie. 'EFLAGS' or 'RFLAGS'
    ;
    ; Return Value: list of flag names
    ;
    ; Adopted from define_flags.pro idl procedure
    ; Created:  Jul 8, 2017  Daniel Sela, Arnon sela
    ;******************************************************************************
    '''

    if name.find('EFLAGS') > -1:
        flags = ['NEIGHBORS', 'BLENDED', 'SATURATED', 'ATEDGE', 'APINCOMPL', 'ISINCOMPL', 'DBMEMOVR', 'EXMEMOVR']
    elif name.find('RFLAGS') > -1:
        flags = ['HOTPIX', 'NOISYPIX', 'STRANGEPIX', 'BADPOS', 'NOTEMPL', 'PHOTSDEV','BADIMAGE']
    else:
        print('Unknown name.')
        return []
    return flags
