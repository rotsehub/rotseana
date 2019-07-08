def psigma( data, guess, width, amplitude, step_size, plots):

    import math
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('TkAgg')

    def phaser( lightcurve, period, epoch ):

        phased = list()
        for obs in lightcurve:
            phase = (obs[0] - epoch) / period
            new = [phase % 1, obs[1], obs[2]]
            phased.append(new)

        output = list()
        for i in phased:
            output.append(i)
            tp = [i[0] + 1, i[1], i[2]]
            output.append(tp)

        return output

    def model( lightcurve, T, amplitude, phase_shift ):

        mags = list()
        for obs in lightcurve:
            mag = obs[1]
            mags.append(mag)

        A = amplitude
        b = math.fsum(mags) / len(mags)

        epochs = list()
        for row in lightcurve:
            epoch = row[0]
            epochs.append(epoch)
    
        mod = list()   
        for t in epochs:
            m = (A*math.cos((((2*math.pi) / T)*t) + phase_shift)) + b
            point = [t, m, 0]
            mod.append(point)

        return mod

    def find_X2( data, model ):

        N = len(data) - 1

        xlist = list()
        for obs in data:
            t = obs[0]
            mi = obs[1]
            e = obs[2]
            for row in model:
                if row[0] == t:
                    m = row[1]

                    x = ((mi - m) / e)**(2)
                    xlist.append(x)

        X2 = math.fsum(xlist) / N

        return X2

    def parameters( lightcurve, guess, amplitude ):

        mags = list()
        for i in lightcurve:
            mags.append(i[1])

        b = math.fsum(mags)/len(mags)

        ps = list()

        p = 0
        while p < 1:
            p = p + 0.01 
            ps.append(p)
        
        chis = list()  
        results = list()
        for k in ps:
            mod = model(lightcurve, guess, amplitude, k)
            X2 = find_X2(lightcurve, mod) / 2
            chis.append(X2)
            res = [X2, k]
            results.append(res)

        for r in results:
            if r[0] == min(chis):
                phase_shift = r[1]

        '''
        nepochs = list()
        nmags = list()
        for n in phaser(lightcurve, guess, lightcurve[0][0]):
            nepochs.append(n[0])
            nmags.append(n[1])

        fmod = model(lightcurve, guess, amplitude, phase_shift)
        mepochs = list()
        mmags = list()
        for f in phaser(fmod, guess, fmod[0][0]):
            mepochs.append(f[0])
            mmags.append(f[1])
        
        plt.scatter(nepochs,nmags)
        plt.scatter(mepochs,mmags)
        plt.show()
        '''

        return phase_shift

    def calc_stats( guess, width, newlist, amplitude, phase_shift, step_size ):

        T = (guess - width) - step_size

        results = list()
        print('calculating chi squared statistics...')
        while T < (guess + width):

            T = T + step_size
    
            comp = model(newlist,T,amplitude,phase_shift)
            X2 = find_X2(newlist,comp)
            res = [T, X2]
    
            results.append(res)

            '''
            ### PLOT ###

            plt.clf()

            x = list()
            y = list()
            xm = list()
            ym = list()
            for m in phaser(comp,T,comp[0][0]):
                xm.append(m[0])
                ym.append(m[1])
            for n in phaser(newlist,T,newlist[0][0]):
                x.append(n[0])
                y.append(n[1])

            plt.scatter(x,y)
            plt.scatter(xm,ym)
            plt.title('Lightcurve (Blue) and Model (Orange)')
            plt.xlabel('MJD')
            plt.ylabel('Magnitude')
            plt.pause(0.000000000001)

            plt.show()
            '''

        print('calculated statistics for',len(results),'periods in range...')
        filtered = list()
        checklist = list()
        for res in results:
            t = res[0]
            X = res[1]
            filtered.append([t, X])
            checklist.append(X)

        minchi2 = min(checklist)
    
        for f in filtered:
            if f[1] == minchi2:
                minimum = f

        output = [results, minimum]

        return output

    def sort( unsorted ):

        sortlist = list()
        while len(unsorted) > 0:
        
            pers = list()
            for i in unsorted:
                per = i[0]
                pers.append(per)

            least = min(pers)
        
            for j in unsorted:
                if j[0] == least:
                    sortlist.append(j)
                    unsorted.remove(j)

        return sortlist

    def zoom( out_calc_stats ):

        data = out_calc_stats[0]
        minimum = out_calc_stats[1]

        x = minimum[0]

        temp = list()
        for i in data:
            diff = abs(x - i[0])
            temp.append([diff,i])

        new = sort(temp)
        zoom = list()

        for q in range(50):
            zoom.append(new[q][1])

        zoomed = sort(zoom)

        return zoomed

    '''
    def parabola( data ):

        zonedis = data[0]
        minimum = data[1]

        step_size = 0.00001
        dx = 1
        dy = 0
        sx = minimum[0]
        sy = minimum[1]

        print('finding parabolic points of the chi squared distribution...')
        parapoints = list()
        parapoints.append(minimum)
        while dx > (0 + step_size):

            if dx > dy:
                dy = dy + step_size
    
            elif dx <= dy:
                dx = dx - step_size

            for i in zonedis:
                if i[0] != sx and i[1] != sy:
                    xi = i[0]
                    yi = i[1]

                y = (dy / dx)*(xi - sx) + sy

                if abs(y - yi) <= 0.01:
                    if yi > sy and xi > sx:
                        sx = xi
                        sy = yi
    
                        dx = 1
                        dy = 0

                        parapoints.append(i)

        step_size = 0.00001
        dx = -1
        dy = 0
        sx = minimum[0]
        sy = minimum[1]
        while dx < (0 - step_size):
    
            if abs(dx) > dy:
                dy = dy + step_size
        
            elif abs(dx) <= dy:
                dx = dx + step_size

            for i in zonedis:
                if i[0] != sx and i[1] != sy:
                    xi = i[0]
                    yi = i[1]

                y = (dy / dx)*(xi - sx) + sy

                if abs(y - yi) <= 0.01:
                    if yi > sy and xi < sx:
                        sx = xi
                        sy = yi
            
                        dx = -1
                        dy = 0

                        parapoints.append(i)

        para = sort(parapoints)
        print('found',len(para),'parabolic points.')

        return para
    '''

    def fit( data ):

        x, y = map(np.array, zip(*data))
        m, s = x.mean(), x.std()
        x_normalized = (x - m)/s

        fit_normalized = a, b, c = np.polyfit(x_normalized, y, 2)
        fit = (a / s**2), (-2*a*m/s**2 + b/s), a*m**2/s**2 - b*m/s + c

        f = np.poly1d(fit)
    
        a = fit[0]
        b = fit[1]
        c = fit[2]

        output = [f,a,b,c]

        return output

    def getsigma( sortedlist ):

        pack = fit(sortedlist)

        f = pack[0]
        a = pack[1]
        b = pack[2]
        c = pack[3]

        x_vertex = (-1)*(b/(2*a))
        y_vertex = f(x_vertex)

        up1 = y_vertex + 1

        sigmaT = (((-1*b) + (((b**2) - (4*a*(c-up1)))**(0.5))) / (2*a)) - x_vertex
        out = [x_vertex, sigmaT]

        return out

    dataplotx = list()
    dataploty = list()
    newlist = list()
    for point in data:
        newpoint = [float(point[0]), float(point[1]), float(point[2])]
        newlist.append(newpoint)
        dataplotx.append(newpoint[0])
        dataploty.append(newpoint[1])

    phase_shift = parameters(newlist,guess,amplitude)

    one = calc_stats(guess,width,newlist,amplitude,phase_shift,step_size)
    two = zoom(one)
    three = getsigma(two)

    period = three[0]
    error = three[1]

    output = [period,error]

    print('period =',period,'+/-',error)

    mo = model(newlist,period,amplitude,phase_shift)
    mod = list()
    for elt in mo:
        mod.append([elt[0],elt[1],0])

    if plots == True:
    
        # Plot of region around the minimum.
    
        matplotlib.use('TkAgg')
        plt.subplot(2,2,3)
        zoomed = two
        zx = list()
        zy = list()
        for z in zoomed:
            zx.append(z[0])
            zy.append(z[1])
        plt.scatter(zx,zy)
        plt.title('Chi Square Distribution in Highlighted Region Near Minimum')
        plt.xlabel('Period')
        plt.ylabel('Chi Square')

        # Plot of whole Chi Squared distribution. 
        # The region around the minimum should be a different color.

        plt.subplot(2,2,2)
        wholedist = one[0]
        for i in wholedist:
            if i in zoomed:
                wholedist.remove(i)
        wx = list()
        wy = list()
        for w in wholedist:
            wx.append(w[0])
            wy.append(w[1]) 
        plt.scatter(wx,wy)
        plt.scatter(zx,zy)
        plt.title('Chi Square Distribution in Specified Range')
        plt.xlabel('Period')
        plt.ylabel('Chi Square')

        # Plot of the parabola points with the parabola fit.
    
        plt.subplot(2,2,4)
        px = list()
        py = list()
        for p in two:
            px.append(p[0])
            py.append(p[1])
        curve = fit(two)
        f = curve[0]
        plt.scatter(px,py)
        plt.plot(px,f(px))
        plt.title('Parabolic Points with Fit')
        plt.xlabel('Period')
        plt.ylabel('Chi Square')
    
        plt.subplot(2,2,1)
        mx = list()
        my = list()
        vx = list()
        vy = list()
        for m in phaser(mod,period,mod[0][0]):
            mx.append(m[0])
            my.append(m[1])
        for n in phaser(newlist,period,newlist[0][0]):
            vx.append(n[0])
            vy.append(n[1])
        plt.scatter(vx,vy)
        plt.scatter(mx,my)
        plt.title('Lightcurve and Model')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
    
        plt.show()
    
    return output

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("indata")
parser.add_argument("guess")
parser.add_argument("width")
parser.add_argument("amplitude")
parser.add_argument("step_size", nargs='?', default=0.000000001)
parser.add_argument("plots", nargs='?', default=True)
args = parser.parse_args()

indata = args.indata
guess = float(args.guess)
width = float(args.width)
amplitude = float(args.amplitude)
step_size = float(args.step_size)
plots = args.plots

ilc = (np.loadtxt(indata)).tolist()

ans = psigma(ilc,guess,width,amplitude,step_size,plots)
