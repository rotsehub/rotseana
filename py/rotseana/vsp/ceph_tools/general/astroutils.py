import argparse
from operator import itemgetter
import os
import sys
import glob
from math import sin, cos, radians, trunc
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from astropy.io import fits

def evaluateBooleanArg(argument):
    if argument == True or argument == 'True' or argument == 'true' or argument == 'Yes' or argument == 'yes' or argument == 'Y' or argument == 'y':
        argument = True
    elif argument == False or argument == 'False' or argument == 'false' or argument == 'No' or argument == 'no' or argument == 'N' or argument == 'n':   
        argument = False
    return argument

def main(file, rightAscension, declination, decimalCoords, convertHjd, truncateHjd, period, epoch, plot, save, verbose):
    def openFile(file):
        data = list(open(file).read().splitlines())
        data = [x.split() for x in data]
        data = [[float(n) for n in x] for x in data]
        return data

    def openMatchStructures(file, rightAscension, declination): 
        def readFitsFile(file, fitsIndex=1):
            try:
                hdus = fits.open(file, memmap=True)
                hdusExt = hdus[fitsIndex]
                match = hdusExt.data
            except Exception as e:
                raise Exception("cannot read fits data from file: %s" % (file,)) from e
            return match, 'ROTSE3'
        
        def readMatchFile(file, *args, **kwargs):
            try:
                match = readsav(file)['match']
            except Exception as e:
                raise Exception("cannot read match data from file: %s" % (file,)) from e
            return match, 'ROTSE1'
        
        def getDataFileRotse(file):
            if not os.path.isfile(file):
                raise Exception("file not found: %s" % (file,))  
            fileExt = file.rpartition('.')[2]
            if fileExt == 'fit':
                return 3
            else:
                return 1
        
        def readDataFile(file, fitsIndex=1, tmpDir='/tmp'):
            if not os.path.isfile(file):
                raise Exception("file not found: %s" % (file,))
            fileExt = file.rpartition('.')[2]
            if fileExt == 'fit':
                match, rotse = readFitsFile(file, fitsIndex)
            else:
                match, rotse = readMatchFile(file)
            return match, rotse
        
        def getMatchStructures(directory):
            cwd = os.getcwd()
            os.chdir(directory)
            matchStructures = list()
            fits = glob.glob("*.fit")
            dats = glob.glob("*.dat")
            datcs = glob.glob("*.datc")
            for fit in fits:
                matchStructures.append(fit)
            for dat in dats:
                matchStructures.append(dat)
            for datc in datcs:
                matchStructures.append(datc)
            return matchStructures, cwd
        
        def findTarget(rightAscension, declination, matchStructures):
            filteredMatchStructures = list()
            targetLightCurve = list()
            for match in matchStructures:
                try:
                    lightCurve = getData(rightAscension, declination, match)
                    for x in lightCurve:
                        targetLightCurve.append(x)
                    print(f"Target found in {match}")
                    filteredMatchStructures.append(match)
                except IndexError:
                    print(f"Cannot find target in {match}; this match structure was removed from the list")
            return filteredMatchStructures, targetLightCurve
        
        def getData(rightAscension, declination, match):
            matchFile = None
            if isinstance(match, str):
                matchFile = match
                match, tele = readDataFile(matchFile)
            matchRa = match.field('RA')[0]
            matchDec = match.field('DEC')[0]
            cond = np.logical_and.reduce((np.abs(matchRa - rightAscension) < 0.001, np.abs(matchDec - declination) < 0.001))
            goodObj = np.where(cond)
            objid = goodObj[0]
            matchMagLim = match['STAT'][0]['M_LIM']
            matchExpTime = match.field('EXPTIME')[0]
            matchMagErr = match.field('MERR')[0][objid][0]
            matchMag = match.field('M')[0][objid][0]
            matchJd = match.field('JD')[0]
            data = [(matchJd[x], matchMag[x], matchMagErr[x], matchExpTime[x] / 86400, matchMagLim[x]) for x in range(len(matchJd)) if 0 < matchMag[x] < 99]
            return data
        
        matchStructures, cwd = getMatchStructures(file)
        matchs, lightCurve = findTarget(rightAscension, declination, matchStructures)
        return lightCurve

    def getR1Night(lightcurve, nights, epochindex):
        rounded_lightcurve = []
        night_count = [0]
        for x in range(len(lightcurve) - 1):
            if int(lightcurve[x][epochindex]) != int(lightcurve[x + 1][epochindex]):
                night_count.append(x + 1)
        final_lightcurve = [n for x in nights for n in lightcurve[night_count[x - 1]:night_count[x]]]
        return final_lightcurve
    
    def mjd2hjd(lightCurve, rightAscension, declination, epoch):
        rightAscension = radians(rightAscension)
        declination = radians(declination)
        jd = [x[0] + 2400000.5 for x in lightCurve]
        days = [x - 2451545 for x in jd]
        meanLongitude = [(0.9856474 * x + 280.46646) % 360 for x in days]
        meanAnomaly = [0.9856003 * x + 357.528 for x in days]
        trueLongitude = [meanLongitude[x] + 1.915 * sin(radians(meanAnomaly[x])) + 0.02 * sin(radians(2 * meanAnomaly[x])) for x in range(len(lightCurve))]
        radVector = [1.00014 - 0.01671 * cos(radians(x)) - 0.00014 * cos(radians(2 * x)) for x in meanAnomaly]
        meanObliqueEcliptic = [23.439 - 0.0000004 * x for x in days]
        hjdCorrection = [(8.3167 * radVector[x]) * ((cos(radians(trueLongitude[x])) * cos(rightAscension) * cos(declination) + (sin(radians(trueLongitude[x])) * (sin(radians(meanObliqueEcliptic[x])) * sin(declination) + cos(declination) * cos(radians(meanObliqueEcliptic[x])) * sin(rightAscension))))) * 0.000694444 for x in range(len(lightCurve))]
        if truncateHjd:
            hjd = [jd[x] - hjdCorrection[x] - 2400000 for x in range(len(lightCurve))]
        else:
            hjd = [jd[x] - hjdCorrection[x] for x in range(len(lightCurve))]
        convertedLightCurve = [[hjd[x], lightCurve[x][1], lightCurve[x][2]] for x in range(len(lightCurve))]
        if epoch != 0:
            epochIndex = [x for x in range(len(lightCurve)) if lightCurve[x][0] == epoch]
            epochIndex = epochIndex[0]
            epoch = hjd[epochIndex]
        return convertedLightCurve, epoch
    
    def phaseLightCurve(lightCurve, period, epoch):
        phasedLightCurve = [(x[0] - epoch) / period for x in lightCurve]
        phasedLightCurve = [x - trunc(x) + 1 if x < 0 else x - trunc(x) for x in phasedLightCurve]
        phasedLightCurve = [[phasedLightCurve[x], lightCurve[x][1], lightCurve[x][2]] for x in range(len(lightCurve))]
        for x in range(len(phasedLightCurve)):
            phasedLightCurve.append([phasedLightCurve[x][0] + 1, phasedLightCurve[x][1], phasedLightCurve[x][2]])
        phasedLightCurve.sort(key = itemgetter(0))
        return phasedLightCurve

    def getPlots(lightCurve, rightAscension, declination, convertHjd, truncateHjd, phasedLightCurve, period, epoch):
        if period != None:
           fig, axs = plt.subplots(2, sharey = True)
           ax1, ax2 = axs
           ax1.set_title(f'Target: {rightAscension:.5f} {declination:.5f} \n Observations: {len(lightCurve)} Period: {period} Epoch: {epoch}')
        else:
            fig, ax1 = plt.subplots(1)
            ax1.set_title(f'Target: {rightAscension:.5f} {declination:.5f} \n Observations: {len(lightCurve)}')
        ax1.errorbar([x[0] for x in lightCurve], [x[1] for x in lightCurve], yerr = [x[2] for x in lightCurve], fmt='o', label = 'Light Curve')
        if convertHjd:
            if truncateHjd:
                ax1.set(xlabel = 'Time (HJD - 2400000)')
            else:
                ax1.set(xlabel = 'Time (HJD)')
        else:
            ax1.set(xlabel = 'Time (MJD)')
        ax1.set(ylabel = 'Magnitude')
        ax1.invert_yaxis()
        ax1.grid(axis = 'both', alpha = 0.75)
        ax1.legend()
        if period != None:
            ax2.errorbar([x[0] for x in phasedLightCurve], [x[1] for x in phasedLightCurve], yerr = [x[2] for x in phasedLightCurve], fmt='o', label = 'Phased Light Curve')
            ax2.set(xlabel = 'Phase')
            ax2.set(ylabel = 'Magnitude')
            ax2.grid(axis = 'both', alpha = 0.75)
            ax2.legend()
    
    def printLightCurve(lightCurve):
        for x in lightCurve:
            print(x[0], x[1], x[2])
                
    def saveLightCurve(lightCurve, rightAscension, declination, phased):
        os.chdir(sys.path[0])
        if phased:
            filename = f'phasedlightcurve_ra{rightAscension:.5f}_dec{declination:.5f}.dat'
        else:
            filename = f'lightcurve_ra{rightAscension:.5f}_dec{declination:.5f}.dat'
        print(f"You can find a copy of the light curve named {filename} in the same directory as astroutils.py")
        np.savetxt(filename, lightCurve, fmt = '%.11f')

    if not decimalCoords:
        rightAscension = float(''.join(rightAscension[:2])) * 15 + float(''.join(rightAscension[2:4])) * 0.25 + float(''.join(rightAscension[4:])) / 240
        declination = float(''.join(declination[:2])) + float(''.join(declination[2:4])) / 60 + float(''.join(declination[4:])) / 3600
    else:
        rightAscension = float(rightAscension)
        declination = float(declination)
    if os.path.isdir(file):  
        lightCurve = openMatchStructures(file, rightAscension, declination)
    else:
        lightCurve = openFile(file)
    if convertHjd:
        lightCurve, epoch = mjd2hjd(lightCurve, rightAscension, declination, epoch)
    else:
        lightCurve = [[x[0], x[1], x[2]] for x in lightCurve]
    if period != None:
        phasedLightCurve = phaseLightCurve(lightCurve, period, epoch)
    if plot:
        if period != None:
            getPlots(lightCurve, rightAscension, declination, convertHjd, truncateHjd, phasedLightCurve, period, epoch)
        else:
            getPlots(lightCurve, rightAscension, declination, convertHjd, truncateHjd, period, period, epoch)
    if save:
        saveLightCurve(lightCurve, rightAscension, declination, False)
        if period != None:
            saveLightCurve(phasedLightCurve, rightAscension, declination, True)
    if verbose:
        print('Light curve:')
        printLightCurve(lightCurve)
        if period != None:
            print('Phased light curve:')
            printLightCurve(phasedLightCurve)           

parser = argparse.ArgumentParser()
parser.add_argument('file', help = 'Path to light curve file or match structure directory')
parser.add_argument('rightAscension', help = 'Target object right ascension')
parser.add_argument('declination', help = 'Target object declination')
parser.add_argument('--decimalCoords', '-c', default = False, help = 'RA and Dec are in decimal form (True)')
parser.add_argument('--convertHjd', '-hjd', default = False, help = 'Convert MJD dates to HJD (True)')
parser.add_argument('--truncateHjd', '-t', default = False, help = 'Truncate HJD dates to HJD-2400000 (True)')
parser.add_argument('--period', '-per', type = float, help = 'Period of phased light curve')
parser.add_argument('--epoch', '-e', type = float, default = 0, help = 'Epoch of phased light curve')
parser.add_argument('--plot', '-p', default = True, help = 'Display plot of light curve/phased light curve (True)')
parser.add_argument('--save', '-s', default = True, help = 'Save light curve/phased light curve to .dat files (True/False)')
parser.add_argument('--verbose', '-v', default = False, help = 'Print light curve/phased light curve to terminal (True/False)')
args = parser.parse_args()
file = args.file
rightAscension = args.rightAscension
declination = args.declination
decimalCoords = evaluateBooleanArg(args.decimalCoords)
convertHjd = evaluateBooleanArg(args.convertHjd)
truncateHjd = evaluateBooleanArg(args.truncateHjd)
period = args.period
epoch = args.epoch
plot = evaluateBooleanArg(args.plot)
save = evaluateBooleanArg(args.save)
verbose = evaluateBooleanArg(args.verbose)
main(file, rightAscension, declination, decimalCoords, convertHjd, truncateHjd, period, epoch, plot, save, verbose)
plt.show()
