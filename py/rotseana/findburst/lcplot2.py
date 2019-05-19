'''
Created on Jul 22, 2017

@author: Daniel Sela, Arnon Sela
'''

import numpy as np
import os
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rotseutil.make_rotse_name_sixty import make_rotse_name

this_pdf = None
this_pdffile = None
fig = None


def close_page():
    global this_pdf, fig
    # fig.subplots_adjust(top=0.8, ) #wspace=0.8)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    this_pdf.savefig()
    plt.close()


def lcplot2(struct, index, obj, pdffile=None, struct_file='', gdobj=None, good=False, err=False, syserr=False, offset=0.0, emask=None, rmask=None, flagbadobs=False, nowait=False, grid=(2, 3), _extra=None, close_pdf=True):
    '''
    ;
    ; CALLING SEQUENCE:    lcplot2,struct,index,obj
    ;
    ; Args:
            pdffile: path to file or an open PdfPages object
            struct: a new type match structure
            struct_file: file from which structure was polled
    ;        index: the indices of the observations to plot
    ;        obj: a single object or an array of objects
    ;        gdobj: array of good objects, marked by a 'y' keypress.
    ;        good: plot "good" observations (don't plot -1's)
    ;        err: plot the statistical error on the observations
    ;        syserr: plot the statistical + systematic error (in quadrature)
    ;        offset: Set this to the zero-time offset, or use /offset to set
    ;            it to the time of the first observation
    ;        emask: a mask for which SExtractor flags are "bad". Default = 0
    ;        rmask: a mask for which rotse rflags are "bad". Default = 0
    ;        flagbadobs: plot all observations but put x's on bad ones.
    ;               nowait: plot all, without waiting for a keypress.
    ;
    ; PROCEDURE:    Plots the light curves of many objects
    ;
    ; REVISION HISTORY:
    ;    Adopted from IDL procedure
    ;******************************************************************************
    ;-
    '''
    global this_pdf, this_pdffile, fig

    axarr = None
    axcoord = None

    struct_m = struct.field('M')[0]
    struct_merr = struct.field('MERR')[0]
    struct_msys = struct.field('MSYS')[0]
    struct_ra = struct.field('RA')[0]
    struct_dec = struct.field('DEC')[0]
    struct_exptime = struct.field('EXPTIME')[0]
    struct_flags = struct.field('FLAGS')[0]
    struct_rflags = struct.field('RFLAGS')[0]
    struct_jd = struct.field('JD')[0]

    try:
        struct_stat = struct.field('STAT')[0]
    except Exception:
        struct_stat = None  # assume a none ROTSE1 file

    n = obj.shape
    if n[0] == 0:
        n=1
    else:
        n = n[0]
        # print('Examining light curves for ',n,' objects')

    o = np.copy(obj)

    if nowait:
        wait = 1.0

    if offset == 1:
        offset = struct_jd[0]

    if syserr:
        the_err = np.sqrt(struct_merr**2.0 + (struct_msys/200.0)**2.0)
        err = True
    else:
        the_err = struct_merr

    if rmask is None:
        rmask = 0
    if emask is None:
        emask = 0

    nbad = 0

    # Check if we have an EFFTIME defined; if not use EXPTIME to get bar width

    exp_time = 0
    if struct_stat is not None:
        stat_names = struct_stat.dtype.names
        h = np.where(stat_names == 'EFFTIME')
        h = h[0]
        efcount = len(h)
        if efcount == 1:
            exp_time = struct_stat(0).exptime
        else:
            exp_time = struct_exptime[0]
    bar_width = (float(exp_time)/(24.*60.*60.))/2.
    gdobj = 0

    # handle change in pdffile:
    if this_pdffile != pdffile and this_pdf is not None:
        this_pdf.close()
        this_pdf = None

    if this_pdf is None:
        this_pdffile = pdffile if pdffile.endswith('.pdf') else pdffile+'.pdf'
        this_pdf = PdfPages(pdffile)

    plot_per_page = 6
    plotid = 0
    first_time = True
    for k in range(n):
        i = np.copy(index)

        if good:
            check_flags_emask = np.vectorize(lambda e: emask & e == 0)
            check_flags_rmask = np.vectorize(lambda e: rmask & e == 0)
            cond = np.logical_and.reduce((struct_m[o[k], i] > 0,
                                          struct_m[o[k], i] < 30,
                                          check_flags_emask(struct_flags[o[k], :]),
                                          check_flags_rmask(struct_rflags[o[k], :])))
            j = np.where(cond)
            j = j[0]
            ngood = len(j)
            if ngood > 1:
                i = i[j]
            else:
                print("********No good observations (%s)!!!!!" % k)
                print("Plotting all observations...")

        if flagbadobs:
            check_flags_emask = np.vectorize(lambda e: emask & e != 0)
            check_flags_rmask = np.vectorize(lambda e: rmask & e != 0)
            cond = np.logical_or.reduse((check_flags_rmask(struct_rflags),
                                         check_flags_emask(struct_flags),
                                         ))
            badobs = np.where(cond)
            badobs = badobs[0]
            nbad = len(badobs)

        minmag = min(struct_m[o[k], i] - the_err[o[k], i])
        maxmag = max(struct_m[o[k], i] + the_err[o[k], i])

        if flagbadobs:
            intobs = np.where(struct_m[o[k], index] > 0)
            intobs = intobs[0]
            minmag = min(struct_m[o[k], index[intobs]]-the_err[o[k], index[intobs]])
            maxmag = max(struct_m[o[k], index[intobs]]+the_err[o[k], index[intobs]])

        if plotid % plot_per_page == 0:
            # start new page
            if not first_time:
                # close previous page
                close_page()

            fig, axarr = plt.subplots(3, 2, figsize=(11, 8.5))
            axcoord = [(int(i/2), i-int(i/2)*2) for i in range(3*2)]
            # struct_file
            if struct_file:
                # fig=plt.figure(figsize=(11, 8.5))
                # fig=plt.figure()
                fig.suptitle(os.path.basename(struct_file), fontsize=11, fontweight='bold')
            first_time = False

        fig_i = axcoord[k % 6]
        lc_fig = axarr[fig_i[0], fig_i[1]]
        # lc_fig = fig.add_subplot(3,2, plotid % plot_per_page + 1)
        the_name = make_rotse_name(struct_ra[o[k]], struct_dec[o[k]])
        lc_fig.set_title('object=%6d, Designation: %s' % (o[k], the_name), fontsize=11)
        x = struct_jd[i]-offset
        y = struct_m[o[k], i]
        lc_fig.set_ylim([maxmag, minmag])

        if err:
            lc_fig.errorbar(x, y, yerr=the_err[o[k], i], xerr=bar_width, fmt='+')
        else:
            lc_fig.plot(x, y, '+')

        plotid += 1

    # TODO: add pdf info
    '''
    d = pdf.infodict()
    d['Title'] = 'match_file'
    d['Author'] = 'Daniel Sela'
    d['Subject'] = 'Find Burst'
    d['Keywords'] = 'lightcurve variable star'
    d['CreationDate'] = datetime.today()
    d['ModDate'] =
    '''

    close_page()
    if close_pdf:
        this_pdf.close()
