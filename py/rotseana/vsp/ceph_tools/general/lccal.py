def lccal( match_structures, vra, vdec, requested_refstars, unconex, plots ):

    import math
    import glob
    import os
    import sys

    import numpy as np

    from in_match import read_data_file, getcoords, getobjids
    from tools import mag2flux, flux2mag

    def get_data(refra, refdec, match):

        match_file = None
        if isinstance(match, str):
            match_file = match
            match, tele = read_data_file(match_file)

        match_ra = match.field('RA')[0]
        match_dec = match.field('DEC')[0]
        cond = np.logical_and.reduce((np.abs(match_ra-refra) < 0.001,
                                          np.abs(match_dec-refdec) < 0.001))
        goodobj = np.where(cond)
        objid = goodobj[0]

        match_m_lim = match['STAT'][0]['M_LIM']
        match_exptime = match.field('EXPTIME')[0]
        match_merr = match.field('MERR')[0][objid][0]
        match_m = match.field('M')[0][objid][0]
        match_jd = match.field('JD')[0]

        curve = list()
        for q in range(len(match_jd)):
            epoch = match_jd[q]
            mag = match_m[q]
            magerr = match_merr[q]
            exptime = match_exptime[q] / 86400
            m_lim = match_m_lim[q]
            point = (epoch,mag,magerr,exptime,m_lim)
            curve.append(point)

        lc = list()
        for i in curve:
            #if 0 < i[1] < 98:
            if True:
                lc.append(i)

        return lc

    def avmag( data ):

        fluxs = list()
        for row in data:
            mag = row[1]
            if 0 < mag < 98:
                flux = mag2flux(mag)
                fluxs.append(flux)

        avflux = math.fsum(fluxs) / len(fluxs)
        avmag = flux2mag(avflux)

        return avmag

    def order( lightcurve ):

        output = list()
        while len(lightcurve) > 0:

            dates = list()
            for i in lightcurve:
                dates.append(i[0])

            early = min(dates)

            for j in lightcurve:
                if j[0] == early:
                    output.append(j)
                    lightcurve.remove(j)

        return output

    def epochs( lightcurve ):

        epochlist = list()
        for i in lightcurve:
            epoch = i[0]
            epochlist.append(epoch)

        return epochlist

    def good_epochs( curve ):

        grace_time = (40) / 86400

        good = list()
        for i in range(len(curve)-1):
            if abs(curve[i][0] - curve[i+1][0]) <= curve[i][3] + grace_time:
                if abs(curve[i][1] - curve[i+1][1]) <= 2*(((curve[i][2]**2) + (curve[i+1][2]**2))**0.5):
                    if curve[i][0] not in good:
                        good.append(curve[i][0])
                    if curve[i+1][0] not in good:
                        good.append(curve[i+1][0])

        return good

    def mini_unconex(lightcurve):

        # This only compatible with ROTSE1

        grace_time = (40) / 86400

        output = list()
        for i in range(len(lightcurve)-1):
            if abs(lightcurve[i][0] - lightcurve[i+1][0]) <= lightcurve[i][3] + grace_time:
                if abs(lightcurve[i][1] - lightcurve[i+1][1]) <= 2*(((lightcurve[i][2]**2) + (lightcurve[i+1][2]**2))**0.5):
                    epoch = (lightcurve[i][0] + lightcurve[i+1][0] + lightcurve[i+1][3]) / 2
                    mag = flux2mag((mag2flux(lightcurve[i][1]) + mag2flux(lightcurve[i+1][1])) / 2)
                    err = ((lightcurve[i][2]**(-2) + lightcurve[i+1][2]**(-2))**(-1))**0.5
                    output.append([epoch, mag, err])

        return output

    def closest_refs(candidates, amount):

        proxlist = list()
        prox_and_coord = list()
        for i in candidates:
            prox = ((abs(i[1][0]-vra)**2)+(abs(i[1][1]-vdec)**2))**0.5
            proxlist.append(prox)
            prox_and_coord.append([prox, i])

        sorted_stars = list()
        while len(proxlist) > 0:
            closest = min(proxlist)
            for j in prox_and_coord:
                if j[0] == closest:
                    sorted_stars.append(j[1])
                    proxlist.remove(j[0])


        lightcurves = list()
        coordlist = list()
        for k in range(int(amount)):
            lightcurves.append(sorted_stars[k][0])
            coordlist.append(sorted_stars[k][1])

        return [lightcurves, coordlist]

    def get_plot_yrange(lightcurve):

        mags = list()
        for point in lightcurve:
            mags.append(point[1])

        average_mag = avmag(lightcurve)
        spread = False
        if abs(average_mag - min(mags)) > abs(average_mag - max(mags)):
            spread = abs(average_mag - min(mags))
        else:
            spread = abs(average_mag - max(mags))

        plot_yrange = [average_mag - spread, average_mag + spread]

        return plot_yrange

    def find_refstars( matchs, ra, dec, radius ):

        cands = list()
        matchnum = len(matchs)

        for match in matchs:

            try:
                varlc = get_data(ra,dec,match)
                svarlc = order(varlc)
                var_epochs = epochs(svarlc)

                surroundstars = getobjids(match, ra, dec, radius)
                for star in surroundstars:
                    coords = getcoords(match,star)
                    lc = get_data(coords[0],coords[1],match)
                    slc = order(lc)

                    matched_epochs = list()
                    decent_epochs = list()
                    m_lims = list()
                    for i in var_epochs:
                        for j in slc:
                            if i == j[0]:
                                matched_epochs.append(j[0])
                                m_lims.append(j[4])
                                if j[4]-4 < j[1] < j[4]:
                                    decent_epochs.append(j[0])

                    m_lim = math.fsum(m_lims)/len(m_lims)

                    if matched_epochs == var_epochs and m_lim - 4 < avmag(slc) < m_lim and len(decent_epochs)/len(slc) >= 0.90:
                        allowed_diff = 0.001
                        if not ra - allowed_diff <= coords[0] <= ra + allowed_diff:
                            if not dec - allowed_diff <= coords[1] <= dec + allowed_diff:
                                cand = [coords, slc]
                                cands.append(cand)

                print('your object was found in ',match,"(",len(surroundstars)-1,"non-target objects found in search range )")

            except:
                #print('cannot find your object in ',match) this is not necessarily the problem
                matchnum = matchnum - 1
                pass

        print("using data in ",matchnum," match structures.")

        return [cands,matchnum]

    def confirm_refstars( matchs, pack ):

        print("checking potential reference stars for consistency...")

        allowed_diff = 0.001

        cands = pack[0]
        N = pack[1]

        refstars = list()
        lcgroups = list()
        for i in cands:
            ira = i[0][0]
            idec = i[0][1]

            lightcurvegroup = list()
            for j in cands:
                jra = j[0][0]
                jdec = j[0][1]
                if abs(ira-jra) <= allowed_diff and abs(idec-jdec) <= allowed_diff:
                    lightcurvegroup.append(j[1])
                    cands.remove(j)

            lcgroups.append([lightcurvegroup, [ira, idec]])

        print(len(lcgroups),"potential reference stars passed first cuts")

        for star in lcgroups:
            group = star[0]
            coords = star[1]
            print("the candidate reference star at RA:",round(coords[0],6),"DEC:",round(coords[1],6),"appears in",len(group),"out of ",N," match structures")
            if len(group) == N:
                starlist = list()
                for elt in group:
                    for obs in elt:
                        starlist.append(obs)

                refstars.append([starlist, coords])

        available_refstars = len(refstars)

        if available_refstars >= requested_refstars:
            refstars = closest_refs(refstars, requested_refstars)

            if len(refstars[0]) == requested_refstars:
                print('using the closest',len(refstars[0]),' of ',available_refstars,' available reference stars')
                for i in range(len(refstars[1])):
                    print("reference star ",i," RA: ",refstars[1][i][0]," DEC: ",refstars[1][i][1])
                refstars = refstars[0]

        return refstars

    def get_refavmags( refstars ):

        reference = list()
        for star in refstars:
            refavmag = avmag(star)
            pac = [star, refavmag]
            reference.append(pac)

        return reference

    def get_corrections( matchs, ra, dec, reference ):

        var = list()
        for match in matchs:
            try:
                varlc = get_data(ra,dec,match)
                for obs in varlc:
                    var.append(obs)
            except:
                pass

        svar = order(var)
        epochlist = list()
        for i in svar:
            epochlist.append(i[0])

        corrs = list()
        for epoch in epochlist:

            diffs = list()
            for star in reference:
                lc = star[0]
                trumag = star[1]

                for obs in lc:
                    if obs[0] == epoch and 0 < obs[1] < 98:
                        diff = trumag - obs[1]
                        diffs.append(diff)

            if len(diffs) == len(reference):
                correction = math.fsum(diffs) / len(diffs)
                corr = [epoch, correction]
                corrs.append(corr)

        package = [svar, corrs, reference]

        return package

    def calibrate_refstars( package ):

        # CALIBRATION IS CURRENTLY NOT BEING APPLIED TO REFERENCE STARS

        corrs = package[1]
        reference = package[2]

        '''

        caled = list()
        for star in reference:
            lc = star[0]

            newcurve = list()
            for obs in lc:
                for i in corrs:

                    epoch = i[0]
                    correction = i[1]

                    if obs[0] == epoch:
                        newpoint = [obs[0], obs[1] + correction, obs[2], obs[3]]
                        newcurve.append(newpoint)

            caled.append(newcurve)

        '''

        '''

        goodtimes = list()
        # old line: for elt in caled:
        for elt in reference:
            goodstartimes = good_epochs(elt[0])
            for j in goodstartimes:
                goodtimes.append(j)

        passed_epochs = list()
        for t in goodtimes:
            # old line: if goodtimes.count(t) == len(caled):
            if goodtimes.count(t)/len(reference) >= 0.75:
                passed_epochs.append(t)

        good_corrs = list()
        for e in passed_epochs:
            for c in corrs:
                if c[0] == e and c not in good_corrs:
                    good_corrs.append(c)

        '''
        good_corrs = corrs
        newpackage = [package[0], good_corrs]

        #print('using',len(good_corrs),'out of',len(corrs),'epochs. (',round((len(good_corrs)/len(corrs))*100,2),'% )')

        '''

        if plots == True:

            try:

                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib.use('TkAgg')

                # Refstar 0:

                original0 = reference[0][0]
                improper0 = caled[0]
                proper0 = list()

                xo0 = list()
                yo0 = list()
                xp0 = list()
                yp0 = list()
                for n in original0:
                    xo0.append(n[0])
                    yo0.append(n[1])
                    for ni in good_corrs:
                        if n[0] == ni[0]:
                            p0 = [n[0],n[1]+ni[1],n[2]]
                            proper0.append(p0)
                            xp0.append(p0[0])
                            yp0.append(p0[1])

                xi0 = list()
                yi0 = list()
                for m in improper0:
                    xi0.append(m[0])
                    yi0.append(m[1])

                yrange0 = get_plot_yrange(original0)

                plt.subplot(2,3,1)
                plt.scatter(xo0,yo0)
                plt.ylim(yrange0[0], yrange0[1])
                plt.title('Refstar 0 Original')
                plt.xlabel('MJD')
                plt.ylabel('Magnitude')

                plt.subplot(2,3,2)
                plt.scatter(xi0,yi0)
                plt.ylim(yrange0[0], yrange0[1])
                plt.title('Refstar 0 Calibrated (all epochs)')
                plt.xlabel('MJD')
                plt.ylabel('Magnitude')

                plt.subplot(2,3,3)
                plt.scatter(xp0,yp0)
                plt.ylim(yrange0[0], yrange0[1])
                plt.title('Refstar 0 Calibrated (good epochs)')
                plt.xlabel('MJD')
                plt.ylabel('Magnitude')

                # Refstar 1:

                original1 = reference[1][0]
                improper1 = caled[1]
                proper1 = list()

                xo1 = list()
                yo1 = list()
                xp1 = list()
                yp1 = list()
                for n in original1:
                    xo1.append(n[0])
                    yo1.append(n[1])
                    for ni in good_corrs:
                        if n[0] == ni[0]:
                            p1 = [n[0],n[1]+ni[1],n[2]]
                            proper1.append(p1)
                            xp1.append(p1[0])
                            yp1.append(p1[1])

                xi1 = list()
                yi1 = list()
                for m in improper1:
                    xi1.append(m[0])
                    yi1.append(m[1])

                yrange1 = get_plot_yrange(original1)

                plt.subplot(2,3,4)
                plt.scatter(xo1,yo1)
                plt.ylim(yrange1[0], yrange1[1])
                plt.title('Refstar 1 Original')
                plt.xlabel('MJD')
                plt.ylabel('Magnitude')

                plt.subplot(2,3,5)
                plt.scatter(xi1,yi1)
                plt.ylim(yrange1[0], yrange1[1])
                plt.title('Refstar 1 Calibrated (all epochs)')
                plt.xlabel('MJD')
                plt.ylabel('Magnitude')

                plt.subplot(2,3,6)
                plt.scatter(xp1,yp1)
                plt.ylim(yrange1[0], yrange1[1])
                plt.title('Refstar 1 Calibrated (good epochs)')
                plt.xlabel('MJD')
                plt.ylabel('Magnitude')

                plt.show()

            except:
                print("cannot show plots, this could be the result of an incompatability with matplotlib. Try again with plots argument set to False.")
                pass

        '''

        return newpackage

    def calibrate_var( newpackage ):

        print("these corrections have been applied to your target light curve:")
        print("epoch:            correction (magnitude):")

        varlc = newpackage[0]
        corrs = newpackage[1]

        newcurve = list()
        for obs in varlc:

            for i in corrs:
                epoch = i[0]
                correction = i[1]

                if obs[0] == epoch and 0 < obs[1] < 98:
                    newpoint = [obs[0], obs[1] + correction, obs[2], obs[3]]
                    if 0 < newpoint[1] < 98:
                        print(obs[0], correction)
                        newcurve.append(newpoint)

        print('using',len(newcurve),'out of',len(varlc),'epochs. (',round((len(newcurve)/len(varlc))*100,2),'% )')

        if plots == 'True' or 'true':

            try:

                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib.use('TkAgg')

                xm = list()
                ym = list()
                original_curve = list()
                for m in varlc:
                    if 0< m[1] < 98:
                        xm.append(m[0])
                        ym.append(m[1])
                        original_curve.append(m)

                xn = list()
                yn = list()
                for n in newcurve:
                    xn.append(n[0])
                    yn.append(n[1])

                yrange = get_plot_yrange(original_curve)

                plt.subplot(2,1,2)
                plt.scatter(xn,yn)
                plt.ylim(yrange[0], yrange[1])
                plt.title('Corrected Light Curve')
                plt.xlabel('MJD')
                plt.ylabel('Magnitude')

                plt.subplot(2,1,1)
                plt.scatter(xm,ym)
                plt.ylim(yrange[0], yrange[1])
                plt.title('Original Light Curve')
                plt.xlabel('MJD')
                plt.ylabel('Magnitude')

                plt.show()

            except:
                print("cannot show plots, this could be the result of an incompatability with matplotlib. Try again with plots argument set to False.")
                pass

        return newcurve

    if requested_refstars == 0:
        print("you requested 0 reference stars, but you need at least 1 (preferably more).")
        sys.exit()

    if requested_refstars < 0:
        print("you requested ",int(requested_refstars)," reference stars, but you need at least 1 (preferably more). Now go think about the philosophical implications of a negative amount of reference stars.")
        sys.exit()

    os.chdir(match_structures)
    matchs = list()
    fits = glob.glob("*.fit")
    dats = glob.glob("*.dat")
    datcs = glob.glob("*.datc")
    for fit in fits:
        matchs.append(fit)
    for dat in dats:
        matchs.append(dat)
    for datc in datcs:
        matchs.append(datc)

    radius = 0.1
    found_refstars = 0

    while found_refstars < requested_refstars:

        print("searching for reference stars within ",radius," degrees of your target.")

        print("gathering data...")
        one = find_refstars(matchs,vra,vdec,radius)
        two = confirm_refstars(matchs,one)

        found_refstars = len(two)
        if found_refstars < requested_refstars:
            radius = round(radius + 0.05,2)
            print("you requested ",int(requested_refstars)," reference stars, but only ",found_refstars," were found. Trying again with search radius = ",radius," degrees.")

        if radius == 1:
            print("search radius has reached 1 degree. Reference stars become less useful the farther the are from the target. Closing program.")
            sys.exit()

    three = get_refavmags(two)
    print("calculating corrections for each epoch...")
    four = get_corrections(matchs,vra,vdec,three)
    #print("testing corrections with reference star data")
    opt = calibrate_refstars(four)
    print("correcting target light curve")
    five = calibrate_var(opt)

    if unconex == 'True' or 'true':
        print("Now filtering data")
        print("***BE AWARE: THE UNCONEX BUILT IN TO THIS PROGRAM IS ONLY COMPATIBLE WITH ROTSE1 DATA***")
        original_length = len(five)
        five = mini_unconex(five)
        final_length = len(five)
        print("unconex filter reduced and combined the target light curve from ",original_length," observations to ",final_length," observations.")

    filename = 'lightcurve_ra'+str(vra)+'_dec'+str(vdec)+'.dat'

    print("your target light curve is printed below. You can also find a copy named ",filename," in the ",match_structures," directory.")
    finalcurve = list()
    for a in five:
        fi = [a[0], round(a[1],4), round(a[2],6)]
        finalcurve.append(fi)

    for row in finalcurve:
        print(' '.join(str(elt) for elt in row))

    np.savetxt(filename, finalcurve, fmt = '%.11f')

    return finalcurve

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("match_structures")
parser.add_argument("vra")
parser.add_argument("vdec")
parser.add_argument("requested_refstars")
parser.add_argument("unconex")
parser.add_argument("plots")
args = parser.parse_args()

match_structures = str(args.match_structures)
vra = float(args.vra)
vdec = float(args.vdec)
requested_refstars = float(args.requested_refstars)
unconex = args.unconex
plots = args.plots

ans = lccal(match_structures, vra, vdec, requested_refstars, unconex, plots)

'''

CHANGES!!!
1. added unconex argument
2. all arguments are now required (to reduce confusion)
3. user can run unconex on calibrated data with the unconcex argument

'''
