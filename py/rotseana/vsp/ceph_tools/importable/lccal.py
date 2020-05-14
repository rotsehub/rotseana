def lccal( vra, vdec, radius, plots ):

    import math
    import glob
    import os

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    from in_match import getlc, getcoords, getobjids
    from tools import mag2flux, flux2mag

    matplotlib.use('TkAgg')

    def avmag( data ):
    
        fluxs = list()
        for row in data:
            mag = row[1]
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

        allowed_diff = 0.01

        good = list()
        for i in curve:
            for j in curve: 
                if abs(i[0] - j[0]) <= allowed_diff and i != j:
                    if abs(i[1] - j[1]) <= ((i[2]**2) + (j[2]**2))**0.5:
                        if j[0] not in good:
                            good.append(j[0])

        return good

    def find_refstars( matchs, ra, dec, radius ):
        
        cands = list()
        matchnum = len(matchs)

        print('looking for reference stars in file:')

        for match in matchs:
            print(match)

            try:
                varlc = getlc(match,ra,dec)
                svarlc = order(varlc)
                var_epochs = epochs(svarlc)

                surroundstars = getobjids(match, ra, dec, radius)
                for star in surroundstars:
                    coords = getcoords(match,star)
                    lc = getlc(match,coords[0],coords[1])
                    slc = order(lc)
                    star_epochs = epochs(slc)

                    if star_epochs == var_epochs:
                        if 10 < avmag(slc) < 15:
                            if coords[0] != ra and coords[1] != dec:
                                cand = [coords, slc]
                                cands.append(cand)
                
            except:
                print('cannot find object in this match structure')
                matchnum = matchnum - 1
                pass
  
        return [cands,matchnum]

    def confirm_refstars( matchs, pack ):
    
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

            lcgroups.append(lightcurvegroup)

        for group in lcgroups:

            if len(group) == N:
                starlist = list()
                for elt in group:
                    for obs in elt:
                        starlist.append(obs)

                refstars.append(starlist)

        print('using',len(refstars),'reference stars')

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
                varlc = getlc(match,ra,dec)
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
                    if obs[0] == epoch:
                        if obs[1] < 98:
                            diff = trumag - obs[1]
                            diffs.append(diff)

            if len(diffs) > 0:
                correction = math.fsum(diffs) / len(diffs)
                corr = [epoch, correction]
                corrs.append(corr)

        package = [svar, corrs, reference]
    
        return package

    def calibrate_refstars( package ):

        corrs = package[1]
        reference = package[2]

        caled = list()
        for star in reference:
            lc = star[0]

            newcurve = list()
            for obs in lc:
                for i in corrs:

                    epoch = i[0]
                    correction = i[1]

                    if obs[0] == epoch and correction != 0:
                        newpoint = [obs[0], obs[1] + correction, obs[2]]
    
                        if 0 < newpoint[1] < 98:
                            newcurve.append(newpoint)

            caled.append(newcurve)

        goodtimes = list()
        for elt in caled:
            goodstartimes = good_epochs(elt)
            for j in goodstartimes:
                goodtimes.append(j)

        passed_epochs = list()
        for t in goodtimes:
            if goodtimes.count(t) == len(caled):
                passed_epochs.append(t)

        good_corrs = list()
        for e in passed_epochs:
            for c in corrs:
                if c[0] == e and c not in good_corrs:
                    good_corrs.append(c)

        newpackage = [package[0], good_corrs]

        print('using',len(good_corrs),'out of',len(corrs),'epochs. (',round((len(good_corrs)/len(corrs))*100,2),'% )')

        if plots == True:

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

            plt.subplot(2,3,1)
            plt.scatter(xo0,yo0)
            plt.title('Refstar 0 Original')
            plt.xlabel('MJD')
            plt.ylabel('Magnitude')
                    
            plt.subplot(2,3,2)
            plt.scatter(xi0,yi0)
            plt.title('Refstar 0 Calibrated (all epochs)')
            plt.xlabel('MJD')
            plt.ylabel('Magnitude')

            plt.subplot(2,3,3)
            plt.scatter(xp0,yp0)
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

            plt.subplot(2,3,4)
            plt.scatter(xo1,yo1)
            plt.title('Refstar 1 Original')
            plt.xlabel('MJD')
            plt.ylabel('Magnitude')
                    
            plt.subplot(2,3,5)
            plt.scatter(xi1,yi1)
            plt.title('Refstar 1 Calibrated (all epochs)')
            plt.xlabel('MJD')
            plt.ylabel('Magnitude')

            plt.subplot(2,3,6)
            plt.scatter(xp1,yp1)
            plt.title('Refstar 1 Calibrated (good epochs)')
            plt.xlabel('MJD')
            plt.ylabel('Magnitude')

            plt.show()

        return newpackage
    
    def calibrate_var( newpackage ):

        varlc = newpackage[0]
        corrs = newpackage[1]

        newcurve = list()
        for obs in varlc:

            for i in corrs:
                epoch = i[0]
                correction = i[1]

                if obs[0] == epoch and correction != 0:
                    newpoint = [obs[0], obs[1] + correction, obs[2]]
                    if 0 < newpoint[1] < 98:
                        newcurve.append(newpoint)

        if plots == True:
        
            xn = list()
            yn = list()
            for n in newcurve:
                xn.append(n[0])
                yn.append(n[1])

            plt.subplot(2,1,2)
            plt.scatter(xn,yn)
            plt.title('Corrected Light Curve')
            plt.xlabel('MJD')
            plt.ylabel('Magnitude')

            xm = list()
            ym = list()
            for m in varlc:
                if m[1] < 98:
                    xm.append(m[0])
                    ym.append(m[1])

            plt.subplot(2,1,1)
            plt.scatter(xm,ym)
            plt.title('Original Light Curve')
            plt.xlabel('MJD')
            plt.ylabel('Magnitude')

            plt.show()
       
        return newcurve

    # Find match structures, put into a list.
    os.chdir("matchstr")
    matchs = glob.glob("*.fit")

    one = find_refstars(matchs,vra,vdec,radius)
    two = confirm_refstars(matchs,one)
    three = get_refavmags(two)
    four = get_corrections(matchs,vra,vdec,three)
    opt = calibrate_refstars(four)
    five = calibrate_var(opt)

    finalcurve = list()
    for a in five:
        fi = [a[0], round(a[1],4), round(a[2],6)]
        finalcurve.append(fi)
    
    for row in finalcurve:
        print(' '.join(str(elt) for elt in row))
    
    return finalcurve
