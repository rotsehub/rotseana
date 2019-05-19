from .define_flags import define_flags


def set_flags(tmpflags, type_, val=0):
    '''
    ;+
    ; NAME: set_flags
    ;
    ; CALLING SEQUENCE:     set_flags,flaglist,type,oldword
    ;
    ; INPUTS:       tmpflags: list of flags, or flag mask
    ;
    ; Keywords:    type: type of flags to consult ('EFLAGS' or 'RFLAGS')
    ;        old: input word in which to set bits
    ;
    ; Return Value: new word with requested bits set
    ;
    ; Adopted from define_flags.pro idl procedure
    ; Created:  Jul 8, 2017  Daniel Sela, Arnon sela
    ;******************************************************************************

    '''
    refs = define_flags(type_)
    nrefs = len(refs)

    # Determine which mode running in.  If mask is input, calc.
    # new flag word and return.  Otherwise continue.

    if isinstance(tmpflags, str):
        tmp = len(tmpflags)
        if tmp == 7:
            flags = list(tmpflags)
        elif tmp == 2:
            tmp2 = val & tmpflags
            tmpflags -= tmp2
            val += tmpflags
            return val
    else:
        flags = tmpflags

    # loop over input list of flag names and set new bits.

    for k in range(nrefs):
        k_bit = 2**k
        tmp2 = val & k_bit
        if tmp2 == 0:
            for flag in flags:
                if flag.find(refs[k]) > -1:
                    val += k_bit
    return val
