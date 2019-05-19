'''
Created on Jul 30, 2017

@author: Daniel Sela, Arnon Sela
'''

import collections


def ten(dd=0, mm=0, ss=0):
    '''
        ;+
        ; NAME:
        ;    TEN()
        ; PURPOSE:
        ;    Converts a sexagesimal number or string to decimal.
        ; EXPLANATION:
        ;    Inverse of the SIXTY() function.
        ;
        ; CALLING SEQUENCES:
        ;    X = TEN( [ HOUR_OR_DEG, MIN, SEC ] )
        ;    X = TEN( HOUR_OR_DEG, MIN, SEC )
        ;    X = TEN( [ HOUR_OR_DEG, MIN ] )
        ;    X = TEN( HOUR_OR_DEG, MIN )
        ;    X = TEN( [ HOUR_OR_DEG ] )      <--  Trivial cases
        ;    X = TEN( HOUR_OR_DEG )          <--
        ;
        ;        or
        ;       X = TEN(HRMNSC_STRING)
        ;
        ; INPUTS:
        ;    HOUR_OR_DEG,MIN,SEC -- Scalars giving sexagesimal quantity in
        ;        in order from largest to smallest.
        ;                         or
        ;   HRMNSC_STRING - String giving sexagesmal quantity separated by
        ;               spaces, commas or colons e.g. "10 23 34" or "-3:23:45.2"
        ;               Any negative values should begin with a minus sign.
        ; OUTPUTS:
        ;    Function value returned = double real scalar, decimal equivalent of
        ;    input sexigesimal quantity.  For numeric input, a minus sign on any
        ;   nonzero element of the input vector causes all the elements to be taken
        ;   as < 0.
        ;
        ; EXAMPLES:
        ;       IDL> print,ten(0,-23,34)
        ;                 --> -0.39277778
        ;       IDL> print,ten("-0,23,34")
        ;                 --> -0.39277778
        ; PROCEDURE:
        ;    Mostly involves checking arguments and setting the sign.
        ;
        ;    The procedure TENV can be used when dealing with a vector of
        ;    sexigesimal quantities.
        ;
        ; MODIFICATION HISTORY:
        ;    Written by R. S. Hill, STX, 21 April 87
        ;    Modified to allow non-vector arguments.  RSH, STX, 19-OCT-87
        ;   Recognize -0.0   W. Landsman/B. Stecklum   Dec 2005
        ;   Work with string input  W. Landsman Dec 2008
        ;   Accept comma separator in string input W. Landsman May 2017
        ;-
    '''
    def bad_args():
        print('''Argument(s) should be hours/degrees, minutes (optional),
seconds (optional)   in vector or as separate arguments.
If any one number negative, all taken as negative.''')
        return 0.0

    # if (mm is None and ss is None):
    if isinstance(dd, str):
        temp = dd.strip()
        neg = dd.startswith('-')
        temp = temp.replace(':', ' ')
        temp = temp.replace(',', ' ')
        values = [abs(float(i)) for i in temp.split(' ')]
        value = values[0]
        mm = values[1]
        decimal = value + mm/60.0 + value[2]/3600.0
        if neg:
            decimal = -decimal
        return decimal
    elif isinstance(dd, collections.Iterable):
        vector = list(dd[:])
    elif isinstance(dd, float) or isinstance(dd, int):
        vector = [dd, mm, ss]
    else:
        return bad_args()

    facs = [1.0, 60.0, 3600.0]
    sign = +1.00
    cnt = str(vector).find('-')
    if cnt > -1:
        sign = -1.0
    vector = [abs(i) for i in vector]
    decim = 0.0
    for v, fac in zip(vector, facs):
        decim = decim + float(v)/fac

    return decim*sign


if __name__ == '__main__':
    print(ten(34,))
    print(ten(22, 52, 45.67))
    print(ten(22, 52, -45.67))
