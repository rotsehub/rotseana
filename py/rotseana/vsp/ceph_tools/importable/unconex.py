def unconex( data, kind ):

    import math
    from tools import flux2mag, mag2flux, avmag

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

    def orconex( data ):

        def confirm_obs( in_confirm ):

            allowed_diff = 0.002

            floatlist = list()
            for obs in in_confirm:
                t = float(obs[0])
                m = float(obs[1])
                e = float(obs[2])
                newob = [t, m, e]
                if m < 98:
                    floatlist.append(newob)  

            dubbpairslist = list()
            for i in floatlist:
                for j in floatlist:
                    if abs(i[0] - j[0]) <= allowed_diff:
                        if i != j:
                            pair = [i,j]
                            dubbpairslist.append(pair)

            pairslist = list()
            for i in dubbpairslist:
                pairslist.append(i)
            for i in pairslist:
                for j in pairslist:
                    if i[0][0] == j[1][0] and i[1][0] == j[0][0]:
                        pairslist.remove(j)

            return pairslist

        def consistent_obs( in_consistent ):

            consistent = list()
            for pair in in_consistent:
    
                mag1 = pair[0][1]
                mag2 = pair[1][1]
                diff = abs(mag1 - mag2)
    
                err1 = pair[0][2]
                err2 = pair[1][2]
                toterr = ((err1**2) + (err2**2))**0.5

                if diff <= toterr:
                    consistent.append(pair)

            return consistent

        def make_single( in_make_single ):
    
            exposure_time = 60
            dectime = exposure_time / 86400
    
            newlc = list()
            for pair in in_make_single:
                
                t1 = pair[0][0]
                t2 = pair[1][0]
        
                flux1 = mag2flux(pair[0][1])
                flux2 = mag2flux(pair[1][1])
    
                err1 = pair[0][2]
                err2 = pair[1][2]
    
                time = (t1 + (t2 + dectime)) / 2
                mag = flux2mag((flux1 + flux2) / 2)
                error = ((err1**(-2) + err2**(-2))**(-1))**0.5
    
                newpoint = [time, round(mag,4), round(error,6)]
                newlc.append(newpoint)
    
            return newlc

        one = confirm_obs( data )
        two = consistent_obs( one )
        three = make_single( two )

        return three

    def snuconex( data ):

        def confirm_obs( in_confirm ):
    
            allowed_diff = 0.0009

            floatlist = list()
            for obs in in_confirm:
                t = float(obs[0])
                m = float(obs[1])
                e = float(obs[2])
                newob = [t, m, e]
                if m < 98:
                    floatlist.append(newob)  
    
            i = 0
            groupslist = list()
            while i < len(floatlist):
        
                group = list()
                done = False
                while done == False:
    
                    temp = list()
                    epoch = floatlist[i][0]
                    for elt in floatlist:
                        if epoch <= elt[0] <= (epoch + allowed_diff) and elt not in group:
                            temp.append(elt)

                    for j in temp:
                        if j not in group:
                            group.append(j)

                    if len(temp) > 0:
                        last = group[len(group) - 1]
                        for q in range(len(floatlist)):
                            if last == floatlist[q]:
                                i = q

                    else:
                        done = True

                if len(group) > 1:
                    groupslist.append(group)

                i = i + 1 
       
            return groupslist

        def consistent_obs( in_consistent ):
    
            consistent = list()

            for group in in_consistent:
    
                pairs = list()
                for i in range(len(group) - 1):
                    pairs.append([group[i],group[i+1]])

                check = list()

                for pair in pairs:
                    if abs(pair[0][1] - pair[1][1]) <= ((pair[0][2]**2) + (pair[1][2]**2))**0.5:
                        for j in pair:
                            if j not in check:
                                check.append(j)

                    else:
                        check.append('bad')

                check.append('last')

                q = 0
                while q <= (len(check) - 1):

                    soc = list()

                    done = False
                    while done == False:
            
                        if check[q] != 'bad':
                            soc.append(check[q])
                            if q < (len(check) - 1):
                                q = q + 1
                
                        if check[q] == 'bad' or check[q] == 'last':
                            done = True

                    if len(soc) > 1:
                        consistent.append(soc)

                    q = q + 1
    
            return consistent
    
        def make_single( in_make_single ):
        
            exposure_time = 60
            dectime = exposure_time / 86400
    
            newlc = list()
            for group in in_make_single:
    
                epochs = list()
                errors = list()
                for obs in group:
                    
                    epochs.append(obs[0])
                    errors.append(obs[2])
    
                time = (math.fsum(epochs) + dectime) / len(epochs)
                mag = avmag(group)
                 
                esqs = list()   
                for e in errors:
                    esqs.append(e**(-2))
    
                error = (math.fsum(esqs)**(-1))**0.5
    
                newpoint = [time, round(mag,4), round(error,6)]
                newlc.append(newpoint)
    
            return newlc
    
        one = confirm_obs( data )
        two = consistent_obs( one )
        three = make_single( two )
    
        return three

    lc = order(data)

    if kind == 'orphans':
        ans = orconex(lc)
    elif kind == 'sn':
        ans = snuconex(lc)
    else:
        print('unrecognized type of data scheduling')
        ans = None

    return ans
