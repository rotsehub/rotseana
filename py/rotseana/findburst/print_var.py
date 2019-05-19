'''
Created on Jul 22, 2017

@author: Daniel Sela, Arnon Sela
'''

import numpy as np


def wstr(*args, newline=''):
    s = " ".join([str(arg) for arg in args])
    s += newline
    return s


def print_var(match, var, fname=None):
    num = var.ptr.shape[0]

    if fname:
        fout = open(fname, 'w')
        write = fout.write
        newline = '\n'
    else:
        write = print
        newline = ''

    match_m = match.field('M')[0]
    match_flags = match.field('FLAGS')[0]
    match_rflags = match.field('RFLAGS')[0]
    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]
    match_msys = match.field('MSYS')[0]

    for k in range(num):
        fout.flush()
        h = np.where(match_m[var[k].ptr, :] > -1.0)
        h = h[0]
        write(wstr('K =', var[k].ptr, ':  RA=', match_ra[var[k].ptr], ' DEC=', match_dec[var[k].ptr], newline=newline))
        write(wstr('     avgmag, delta = ', var[k].avgmag, var[k].delta, newline=newline))
        write(wstr('     mag = ', match_m[var[k].ptr, h], newline=newline))
        write(wstr('     dis = ', *[o.dis for o in var[k].obs[h]], newline=newline))
        write(wstr('     eflags = ', match_flags[var[k].ptr, h], newline=newline))
        write(wstr('     msys = ', match_msys[var[k].ptr, h], newline=newline))
        write(wstr('     rflags = ', match_rflags[var[k].ptr, h], newline=newline))

    write(wstr(' ', newline=newline))
    write(wstr(' ', newline=newline))
    write(wstr(' ', newline=newline))
    for k in range(num):
        fout.flush()
        # fout.write(wstr(var[k].name, var[k].maxdelta, var[k].maxerr, var[k].chisqcl))
        # corrected aggogding to lightcurve.pro
        write(wstr(var[k].name, var[k].bestdelta, var[k].bestsig, var[k].chisqcl, newline=newline))
    write(wstr(' ', newline=newline))
    write(wstr(' ', newline=newline))
    write(wstr(' ', newline=newline))
    for k in range(num):
        fout.flush()
        write(wstr(var[k].name, var[k].delta, var[k].avgmag, match_ra[var[k].ptr], match_dec[var[k].ptr], newline=newline))

    if fname:
        fout.close()
