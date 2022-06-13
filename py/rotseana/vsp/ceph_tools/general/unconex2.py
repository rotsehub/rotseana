#UNCONEX VERSION 2.3
#UPDATED 6/13/2022

import math
import argparse
import glob
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from scipy import stats as st
from astropy.io import fits
from decimal import Decimal
plt.rc('legend', fontsize = 12)

def evaluateBooleanArg(argument):
    if argument == True or argument == 'True' or argument == 'true' or argument == 'Yes' or argument == 'yes' or argument == 'Y' or argument == 'y':
        argument = True
    elif argument == False or argument == 'False' or argument == 'false' or argument == 'No' or argument == 'no' or argument == 'N' or argument == 'n':   
        argument = False
    else:
        argument = float(argument)
    return argument

def openFile(file):
    data = list(open(file).read().splitlines())
    data = [x.split() for x in data]
    data = [[float(n) for n in x] for x in data]
    return data

def read_fits_file(file, fits_index=1):
    try:
        hdus = fits.open(file, memmap=True)
        hdus_ext = hdus[fits_index]
        match = hdus_ext.data
    except Exception as e:
        raise Exception("cannot read fits data from file: %s" % (file,)) from e
    return match, 'ROTSE3'

def read_match_file(file, *args, **kwargs):
    try:
        match = readsav(file)['match']
    except Exception as e:
        raise Exception("cannot read match data from file: %s" % (file,)) from e
    return match, 'ROTSE1'

def get_data_file_rotse(file):
    if not os.path.isfile(file):
        raise Exception("file not found: %s" % (file,))  
    file_ext = file.rpartition('.')[2]
    if file_ext == 'fit':
        return 3
    else:
        return 1

def read_data_file(file, fits_index=1, tmpdir='/tmp'):
    if not os.path.isfile(file):
        raise Exception("file not found: %s" % (file,))
    file_ext = file.rpartition('.')[2]
    if file_ext == 'fit':
        match, rotse = read_fits_file(file, fits_index)
    else:
        match, rotse = read_match_file(file)
    return match, rotse

def get_data(refra, refdec, match):
    match_file = None
    if isinstance(match, str):
        match_file = match
        match, tele = read_data_file(match_file)
    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]
    cond = np.logical_and.reduce((np.abs(match_ra-refra) < 0.001, np.abs(match_dec-refdec) < 0.001))
    goodobj = np.where(cond)
    objid = goodobj[0]
    match_m_lim = match['STAT'][0]['M_LIM']
    match_obstime = match['STAT'][0]['OBSTIME'] ## gets the observation times
    match_exptime = match.field('EXPTIME')[0]
    match_merr = match.field('MERR')[0][objid][0]
    match_m = match.field('M')[0][objid][0]
    match_flags = match.field('FLAGS')[0][objid][0] # get all of the flags for this objid
    match_jd = match.field('JD')[0]
    curve = list()
    for q in range(len(match_jd)):
        flags = match_flags[q]            
        epoch = float((math.floor(match_jd[q])) + (Decimal(str(match_obstime[q] / 86400)) % 1))
        mag = match_m[q]
        magerr = match_merr[q]
        obstime = match_obstime[q]
        exptime = match_exptime[q]
        m_lim = match_m_lim[q]
        point = [epoch, mag, magerr, exptime, m_lim, obstime, flags]
        curve.append(point)
    return curve

def get_matchstructs(match_structures):
    cwd = os.getcwd()
    os.chdir(match_structures)
    temp_matchs = list()
    fits = glob.glob("*match.fit")
    dats = glob.glob("*match.dat")
    datcs = glob.glob("*match.datc")
    for fit in fits:
        temp_matchs.append(fit)
    for dat in dats:
        temp_matchs.append(dat)
    for datc in datcs:
        temp_matchs.append(datc)
    return temp_matchs, cwd

def find_target(vra, vdec, temp_matchs, verbose):
    matchs = []
    target_lc = []
    nightInterval = []
    for match in temp_matchs:
        try:
            lc = get_data(vra, vdec, match)
            nightInterval.append([lc[0][0], lc[-1][0]])
            for i in lc:
                target_lc.append(i)
            print(f"Target found in {match}")
            matchs.append(match)
        except IndexError:
            if verbose:
                print(f"Cannot find target in {match}; this match structure was removed from the list")
            pass
    return matchs, target_lc, nightInterval

def print_lightcurve(lightcurve, xerror, final):
    if final:
        print("***FINAL TARGET LIGHTCURVE***")
        if xerror:
            for i in lightcurve:
                print(i[0], i[1], i[2], i[3])
        else:
            for i in lightcurve:
                print(i[0], i[1], i[2])
    else:
        print("***TARGET LIGHTCURVE***")
        for i in lightcurve:
            print(i[0], i[1], i[2], i[3], i[4], i[5], i[6])

def save_lightcurve(lightcurve, cwd, vra, vdec):
    os.chdir(cwd)
    filename = f'lightcurve_ra{vra:.5f}_dec{vdec:.5f}.dat'
    print(f"You can find a copy of the light curve named {filename} in the same directory as unconex2.py")
    np.savetxt(filename, lightcurve, fmt = '%.11f')

def save_log(log_params, cwd, vra, vdec):
    os.chdir(cwd)
    filename = f'log_ra{vra:.5f}_dec{vdec:.5f}.dat'
    print(f"You can find a copy of the log file named {filename} in the same directory as unconex2.py")
    open(f'{filename}', 'w').writelines('%s\n' % x for x in log_params)

def getBinaryStr(bitSum):
    binaryStr = bin(bitSum).replace('0b','') #transform to binary
    binaryStr = binaryStr[::-1] #this reverses an array
    while len(binaryStr) < 8: 
        binaryStr += '0' ## add bits until we have a binary number on 8 bits (as used by SExtractor)
    binaryStr = binaryStr[::-1]
    return binaryStr

def applyBitmask(lightCurve, bitmask):
    flagged = [[] for x in range(9)]
    for x in lightCurve:
        x[6] = getBinaryStr(x[6])
        for i in range(len(x[6])):
            if x[6][i] == '1' and bitmask[i] == '1':
                flagged[i].append(x)
                flagged[8].append(x)
    return flagged

def unconex(lightCurve, schedule, threshold, xerror):
    confirmation = [[0, 0] for x in lightCurve] # Array for confirmation state of each observation
    consistence = [[0, 0] for x in lightCurve] # Array for consistence state of each observation
    filtLightCurve = [] # Array for filtered light curve
    graceTime = 40 # Grace time definition 
    for x in range(len(confirmation) - 1): # Loop through each entry in confirmation
        if abs(lightCurve[x][0] - lightCurve[x + 1][0]) <= (lightCurve[x][3] + graceTime) / 86400:
            confirmation[x][1] += 1
            confirmation[x + 1][0] += 1
    for x in range(len(consistence) - 1):
        if confirmation[x][1] == 1 and confirmation[x + 1][0] == 1:
            if abs(lightCurve[x][1] - lightCurve[x + 1][1]) <= threshold * (((lightCurve[x][2] ** 2) + (lightCurve[x + 1][2] ** 2)) ** 0.5):
                consistence[x][1] += 1
                consistence[x + 1][0] += 1
        else: 
            pass
    unconfirmed = [lightCurve[x] for x in range(len(lightCurve)) if confirmation[x][0] != 1 and confirmation[x][1] != 1]
    inconsistent = [lightCurve[x] for x in range(len(lightCurve)) if consistence[x][0] != 1 and consistence[x][1] != 1 and lightCurve[x] not in unconfirmed]
    print(f'{len(lightCurve) - len(unconfirmed)} observations were confirmed')
    print(f'{len(lightCurve) - len(unconfirmed) - len(inconsistent)} observations are consistent')
    averaged = []
    i = 0
    while i in range(len(lightCurve)):
        if i == len(lightCurve) - 1:
            if consistence[i][0] == 1 and consistence[i - 1][1] == 1:
                point = [lightCurve[i][0] + lightCurve[i][3] / 86400 / 2, lightCurve[i][1], lightCurve[i][2]]
                if xerror:
                    point.append(lightCurve[i][3] / 86400 / 2)
                filtLightCurve.append(point)
            i += 1
        elif consistence[i][1] == 1 and consistence[i + 1][0] == 1:
            meanEpoch = (lightCurve[i][0] + lightCurve[i + 1][0] + (lightCurve[i + 1][3]) / 86400) / 2
            meanMag = np.average([lightCurve[i][1], lightCurve[i + 1][1]], weights = [1 / lightCurve[i][2] ** 2, 1 / lightCurve[i + 1][2] ** 2])
            if abs(lightCurve[i][1] - lightCurve[i + 1][1]) >= math.sqrt(2) * np.mean([lightCurve[i][2], lightCurve[i + 1][2]]):
                meanErr = abs(lightCurve[i][1] - lightCurve[i + 1][1]) / math.sqrt(2)
            else:
                meanErr = np.mean([lightCurve[i][2], lightCurve[i + 1][2]]) / math.sqrt(2)
            point = [meanEpoch, meanMag, meanErr]
            if xerror:
                epochErr = abs(lightCurve[i][0] - lightCurve[i + 1][0]) + lightCurve[i + 1][3] / 86400
                point.append(epochErr)
            filtLightCurve.append(point)
            averaged.append([point, lightCurve[i], lightCurve[i + 1]])
            i += 2
        elif consistence[i][0] == 1 and consistence[i - 1][1] == 1:
            point = [lightCurve[i][0] + lightCurve[i][3] / 86400 / 2, lightCurve[i][1], lightCurve[i][2]]
            if xerror:
                point.append(lightCurve[i][3] / 86400 / 2)
            filtLightCurve.append(point)
            i += 1
        else:
            i += 1
    print(f'{len(averaged) * 2} observations were averaged')
    print(f'{len(filtLightCurve) - len(averaged)} consistent but unpaired observations were retained')
    return filtLightCurve, averaged, unconfirmed, inconsistent

def getPlots(lightCurve, filtLightCurve, flagged, averaged, unconfirmed, inconsistent, rightAscension, declination, bitmask, minUncertainty, xerror):
    lightCurve = [[x[0], x[1], x[2]] for x in lightCurve]
    unconfirmed = [[x[0], x[1], x[2]] for x in unconfirmed]
    inconsistent = [[x[0], x[1], x[2]] for x in inconsistent]
    meanErr = np.mean([x[2] for x in lightCurve])
    filtMeanErr = np.mean([x[2] for x in filtLightCurve])
    maskedObs = [x for x in lightCurve if x not in unconfirmed and x not in inconsistent]
    averagedObs = [x[0] for x in averaged]
    unpairedObs = [x for x in filtLightCurve if x not in averagedObs]
    #if xerror:
        #unpairedObs = [x for x in filtLightCurve if x in [[i[0], i[1], i[2], 0] for i in lightCurve]]
    #else:
        #unpairedObs = [x for x in filtLightCurve if x in lightCurve]
    fig, axs = plt.subplots(2, sharex = True, sharey = True)
    ax1, ax2 = axs
    ax1.errorbar([x[0] for x in maskedObs], [x[1] for x in maskedObs], [x[2] for x in maskedObs], fmt = 'o', color = 'tab:blue', label = f'Masked Observations ({len(maskedObs)})')
    ax1.errorbar([x[0] for x in flagged[8]], [x[1] for x in flagged[8]], [x[2] for x in flagged[8]], fmt = '+', color = 'tab:orange', label = f'Flagged Observations ({len(flagged[8])})')
    ax1.errorbar([x[0] for x in unconfirmed], [x[1] for x in unconfirmed], [x[2] for x in unconfirmed], fmt = '+', color = 'tab:olive', label = f'Unconfirmed Observations ({len(unconfirmed)})')
    ax1.errorbar([x[0] for x in inconsistent], [x[1] for x in inconsistent], [x[2] for x in inconsistent], fmt = '+', color = 'tab:red', label = f'Inconsistent Observations ({len(inconsistent)})')
    ax1.set_title(f'Target: {rightAscension:.7f} {declination:.7f}\nBitmask = {bitmask}, $\sigma_{{min}} = {minUncertainty}$\nUnfiltered Light Curve\n$\\bar{{\sigma}} = {meanErr:.6f}$')
    ax1.set(xlabel = 'Time (MJD)')
    ax1.set(ylabel = 'Magnitude')
    ax1.grid(axis = 'both', alpha = 0.75)
    ax1.invert_yaxis()
    ax1.legend()
    if xerror:
        ax2.errorbar([x[0] for x in averagedObs], [x[1] for x in averagedObs], [x[2] for x in averagedObs], [x[3] for x in averagedObs], fmt = 'o', color = 'tab:blue', label = f'Averaged Observations ({len(averagedObs)})')
        ax2.errorbar([x[0] for x in unpairedObs], [x[1] for x in unpairedObs], [x[2] for x in unpairedObs], [x[3] for x in unpairedObs], fmt = 'o', color = 'tab:green', label = f'Unaveraged Observations ({len(unpairedObs)})')
    else:
        ax2.errorbar([x[0] for x in averagedObs], [x[1] for x in averagedObs], [x[2] for x in averagedObs], fmt = 'o', color = 'tab:blue', label = f'Averaged Observations ({len(averagedObs)})')
        ax2.errorbar([x[0] for x in unpairedObs], [x[1] for x in unpairedObs], [x[2] for x in unpairedObs], fmt = 'o', color = 'tab:green', label = f'Unaveraged Observations ({len(unpairedObs)})')   
    ax2.set_title(f'Filtered Light Curve\n$\\bar{{\sigma}} = {filtMeanErr:.6f}$')
    ax2.set(xlabel = 'Time (MJD)')
    ax2.set(ylabel = 'Magnitude')
    ax2.grid(axis = 'both', alpha = 0.75)
    ax2.legend()
    return fig

def getTestPlots(lightCurve, filtLightCurve, averaged, flagged, unconfirmed, inconsistent, rightAscension, declination):
    fig, axs = plt.subplots(1)
    lightCurve = [[x[0], x[1], x[2]] for x in lightCurve]
    pairedObs = [[x[1], x[2]] for x in averaged]
    pairedObs = [y for x in pairedObs for y in x]
    pairedObs = [[x[0], x[1], x[2]] for x in pairedObs]
    avgObs = [x[0] for x in averaged]
    unpairedObs = [x for x in filtLightCurve if x not in avgObs]
    discardedObs = [x for x in lightCurve if x not in pairedObs]
    discardedObs = [x for x in discardedObs if x not in unpairedObs]
    axs.set_title(f'Target: {rightAscension:.7f} {declination:.7f}\nBitmask = {bitmask}, $\sigma_{{min}} = {minUncertainty}$\nLight Curve')
    axs.errorbar([x[0] for x in flagged[8]], [x[1] for x in flagged[8]], [x[2] for x in flagged[8]], fmt = '+', color = 'tab:orange', label = f'Flagged Observations ({len(flagged[8])})')
    axs.errorbar([x[0] for x in pairedObs], [x[1] for x in pairedObs], [x[2] for x in pairedObs], fmt = 'o', color = 'tab:blue', label = f'Paired Observations ({len(pairedObs)})')
    axs.errorbar([x[0] for x in unpairedObs], [x[1] for x in unpairedObs], [x[2] for x in unpairedObs], fmt = 'o', color = 'tab:green', label = f'Unpaired Observations ({len(unpairedObs)})')
    axs.errorbar([x[0] for x in avgObs], [x[1] for x in avgObs], [x[2] for x in avgObs], fmt = 'o', color = 'tab:purple', label = f'Averaged Observations ({len(averaged)})')
    axs.errorbar([x[0] for x in unconfirmed], [x[1] for x in unconfirmed], [x[2] for x in unconfirmed], fmt = '+', color = 'tab:olive', label = f'Unconfirmed Observations ({len(unconfirmed)})')
    axs.errorbar([x[0] for x in inconsistent], [x[1] for x in inconsistent], [x[2] for x in inconsistent], fmt = '+', color = 'tab:red', label = f'Inconsistent Observations ({len(inconsistent)})')
    axs.set(xlabel = 'Time (MJD)')
    axs.set(ylabel = 'Magnitude')
    axs.grid(axis = 'both', alpha = 0.75)
    axs.invert_yaxis()
    axs.legend()

## Write # of flags and the end of file
def writeFlags(binaryBitmask, flagged, vra, vdec, cwd, totalObs):
    filename = f'lightcurve_ra{vra:.5f}_dec{vdec:.5f}.dat' ## put the file name in a variable
    os.chdir(cwd) ## get to the right directory 
    openedFile = open(f'{filename}', 'a') ## open the file to edit it
    if binaryBitmask[7] == '1': ## if user wants to use this flag (selcted this bitmask as an arugment)
        print(f'{len(flagged[7])} observations flagged as biased were removed') # print to console # of observations removed for this flag
        text = str(len(flagged[7])) + ' (' + str(round(len(flagged[7])/totalObs*100,2))+ '%' + ' of total)' + ' observations flagged as biased were removed' # put into a variable the text about # of observations removed for this flag
        openedFile.writelines('\n' + text) # append the text at the end of the file on a new line
    if binaryBitmask[6] == '1':
        print(f'{len(flagged[6])} observations flagged as deblended were removed')
        text = str(len(flagged[6]))+ ' (' + str(round(len(flagged[6])/totalObs*100,2))+ '%' + ' of total)' + ' observations flagged as deblended were removed'
        openedFile.writelines('\n' + text)        
    if binaryBitmask[5] == '1':
        print(f'{len(flagged[5])} observations flagged as saturated were removed')
        text = str(len(flagged[5]))+ ' (' + str(round(len(flagged[5])/totalObs*100,2))+ '%' + ' of total)' + ' observations flagged as saturated were removed'
        openedFile.writelines('\n' + text)    
    if binaryBitmask[4] == '1':
        print(f'{len(flagged[4])} observations flagged as too close to an image boundary were removed')
        text = str(len(flagged[4]))+ ' (' + str(round(len(flagged[4])/totalObs*100,2))+ '%' + ' of total)' + ' observations flagged as too close to an image boundary were removed'
        openedFile.writelines('\n' + text)    
    if binaryBitmask[3] == '1':
        print(f'{len(flagged[3])} observations flagged as having an incomplete or corrupt photometric aperture were removed')
        text = str(len(flagged[3]))+ ' (' + str(round(len(flagged[3])/totalObs*100,2))+ '%' + ' of total)' + ' observations flagged as having an incomplete or corrupt photometric aperture were removed'
        openedFile.writelines('\n' + text)
    if binaryBitmask[2] == '1':
        print(f'{len(flagged[2])} observations flagged as having an incomplete or corrupt isophotal footprint were removed')
        text = str(len(flagged[2]))+ ' (' + str(round(len(flagged[2])/totalObs*100,2))+ '%' + ' of total)' + ' observations flagged as having an incomplete or corrupt isophotal footprint were removed'
        openedFile.writelines('\n' + text)    
    if binaryBitmask[1] == '1':
        print(f'{len(flagged[1])} observations flagged as having a memory overflow during deblending were removed')
        text = str(len(flagged[1]))+ ' (' + str(round(len(flagged[1])/totalObs*100,2))+ '%' + ' of total)' + ' observations flagged as having a memory overflow during deblending were removed'
        openedFile.writelines('\n' + text)    
    if binaryBitmask[0] == '1':
        print(f'{len(flagged[0])} observations flagged as having a memory overflow during extraction were removed')
        text = str(len(flagged[0]))+ ' (' + str(round(len(flagged[0])/totalObs*100,2))+ '%' + ' of total)' + ' observations flagged as having a memory overflow during extraction were removed'
        openedFile.writelines('\n' + text)    


def sexa2deci(rightAscension, declination):
    rightAscension = float(''.join(rightAscension[:2])) * 15 + float(''.join(rightAscension[2:4])) * 0.25 + float(''.join(rightAscension[4:])) / 240
    declination = float(''.join(declination[:2])) + float(''.join(declination[2:4])) / 60 + float(''.join(declination[4:])) / 3600
    return rightAscension, declination

def main(fileDir, schedule, rightAscension, declination, threshold, bitmask, minUncertainty, xerror, plot, picklePlot, save, verbose, log, night, dev):
    if rightAscension[6] == '.':
        rightAscension, declination = sexa2deci(rightAscension, declination)
    else:
        rightAscension = float(rightAscension)
        declination = float(declination)
    allMatchStructs, cwd = get_matchstructs(fileDir)
    matchStructs, extractedObs, nightInterval = find_target(rightAscension, declination, allMatchStructs, verbose)
    if len(matchStructs) == 0:
        print("Unable to locate target in the match structure directory, please verify that the input coordinates are correct")
        sys.exit()
    if night != None:
        extractedObs = [x for x in extractedObs if nightInterval[night - 1][0] <= x[0] <= nightInterval[night - 1][1]]
    print(f'{len(extractedObs)} observations of the target were extracted')
    lightCurve = [x for x in extractedObs if 0 < x[1] < 99]
    print(f'{len(lightCurve)} total observations in target\'s light curve after {len(extractedObs) - len(lightCurve)} unphysical observations were removed')
    if verbose:
        print_lightcurve(lightCurve, xerror, False)
    binaryBitmask = getBinaryStr(bitmask)
    flagged = applyBitmask(lightCurve, binaryBitmask)
    if verbose:
        if binaryBitmask[7] == '1':
            print(f'{len(flagged[7])} observations flagged as biased were removed')
        if binaryBitmask[6] == '1':
            print(f'{len(flagged[6])} observations flagged as deblended were removed')
        if binaryBitmask[5] == '1':
            print(f'{len(flagged[5])} observations flagged as saturated were removed')
        if binaryBitmask[4] == '1':
            print(f'{len(flagged[4])} observations flagged as too close to an image boundary were removed')
        if binaryBitmask[3] == '1':
            print(f'{len(flagged[3])} observations flagged as having an incomplete or corrupt photometric aperture were removed')
        if binaryBitmask[2] == '1':
            print(f'{len(flagged[2])} observations flagged as having an incomplete or corrupt isophotal footprint were removed')
        if binaryBitmask[1] == '1':
            print(f'{len(flagged[1])} observations flagged as having a memory overflow during deblending were removed')
        if binaryBitmask[0] == '1':
            print(f'{len(flagged[0])} observations flagged as having a memory overflow during extraction were removed')
    lightCurve = [x for x in lightCurve if  x not in flagged[8]]
    print(f'{len(lightCurve)} observations in target\'s light curve after {len(flagged[8])} flagged observations were removed')
    for x in lightCurve:
        if x[2] < minUncertainty:
            x[2] = minUncertainty
    filtLightCurve, averaged, unconfirmed, inconsistent = unconex(lightCurve, schedule, threshold, xerror)
    if verbose:
        print_lightcurve(filtLightCurve, xerror, True)
    if plot or picklePlot:
        os.chdir(cwd)
        fig = getPlots(lightCurve, filtLightCurve, flagged, averaged, unconfirmed, inconsistent, rightAscension, declination, bitmask, minUncertainty, xerror)
        if picklePlot:
            filename = f'fig_ra{rightAscension:.5f}+dec{declination:.5f}.pkl'
            pickle.dump(fig, open(f'{filename}', 'wb'))
            print(f"A pickled plot of the light curve named {filename} has been saved in the same directory as unconex2.py")
    if save:
        save_lightcurve(filtLightCurve, cwd, rightAscension, declination)
    if log:
        logParams = [f'Extracted observations: {len(extractedObs)}', f'Physical observations: {len(lightCurve) + len(flagged[8])}', f'Masked observations: {len(lightCurve)}', 
        f'Flagged observations: {len(flagged[8])}', f'Confirmed observations: {len(lightCurve) - len(unconfirmed)}', f'Constistent observations: {len(lightCurve) - len(unconfirmed) - len(inconsistent)}', 
        f'Averaged observations: {len(averaged) * 2}', f'Unpaired observations: {len(filtLightCurve) - len(averaged)}', f'Filtered observations: {len(filtLightCurve)}']
        save_log(logParams, cwd, rightAscension, declination)
    for x in averaged:
        x[1] = [x[1][0], x[1][1], x[1][2], x[1][3]]
        x[2] = [x[2][0], x[2][1], x[2][2], x[2][3]]
    if dev:  
        avgObs = [x[0] for x in averaged]
        priorObs = [x[1] for x in averaged]
        subsequentObs = [x[2] for x in averaged]
        os.chdir(cwd)
        np.savetxt('avgobs_ra'+str(rightAscension)+'_dec'+str(declination)+'.dat', avgObs, fmt = '%.11f')
        np.savetxt('priorobs_ra'+str(rightAscension)+'_dec'+str(declination)+'.dat', priorObs, fmt = '%.11f')
        np.savetxt('subobs_ra'+str(rightAscension)+'_dec'+str(declination)+'.dat', subsequentObs, fmt = '%.11f')
        getTestPlots(lightCurve, filtLightCurve, averaged, flagged, unconfirmed, inconsistent, rightAscension, declination)
    ## print numebr of flagged observations at the end of file
    writeFlags(binaryBitmask,flagged,rightAscension,declination,cwd,len(lightCurve))
    if plot:
        plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("fileDir", type = str, help = 'Path to match structure directory for R1/3 or light curve file for NSVS')
parser.add_argument("schedule", choices = ['R1', 'NSVS', 'R3'], help = 'ROTSE experiment that originated data')
parser.add_argument("ra", help = 'Target\'s right ascension')
parser.add_argument("dec", help = 'Target\'s declination')
parser.add_argument("--threshold", "-t", type = float, default = 2, help = 'Threshold to determine consistent observations')
parser.add_argument("--bitmask", "-b", type = int, default = 0, help = 'Remove flagged observations with the specified bit mask')
parser.add_argument("--minUncertainty", "-u", type = float, default = 0.02, help = 'Minimum uncertainty of magnitude measurements')
parser.add_argument("--xerror", "-x", default = False, help = 'Calculate error along x-axis (time) when filtering data (True to enable)')
parser.add_argument("--plot", "-p", default = True, help = 'Display target\'s light curve before and after processing (False to disable)')
parser.add_argument("--picklePlot", "-pk", default = False, help = 'Save interactive plot of target\'s light curve before and after processing (True to enable)')
parser.add_argument("--save", "-s", default = True, help = 'Save the target\'s processed light curve to a .dat file (False to disable)')
parser.add_argument("--verbose", "-v", default = False, help = 'Print target\'s processed light curve and additional information to terminal (True to enable)')
parser.add_argument("--log", "-l", default = False, help = 'Save processing metrics to log file (True to enable)')
parser.add_argument("--night", "-n", type = int, default = None, help = 'Only process the target\'s light curve for a specifc night')
parser.add_argument("--dev", "-d", default = False, help = 'Enable developer mode to output additional data and plots')
args = parser.parse_args()

fileDir = args.fileDir
schedule = args.schedule
if os.path.isdir(fileDir) and schedule == 'NSVS':
    print('ArgumentError: NSVS/Sky Patrol-schedule data can only be extracted from a .dat light curve file but a path to a directory has been given, please try again')
    sys.exit()
ra = args.ra
dec = args.dec
threshold = args.threshold
if threshold <= 0:
    print('ArgumentError: Value of consistency threshold must be greater than zero, please try again')
    sys.exit()
bitmask = args.bitmask
minUncertainty = args.minUncertainty
if minUncertainty < 0:
    print('ArgumentError: Value of minimum magnitude uncertainty must be greater than or equal to zero, please try again')
    sys.exit()
xerror = evaluateBooleanArg(args.xerror)
plot = evaluateBooleanArg(args.plot)
picklePlot = evaluateBooleanArg(args.picklePlot)
save = evaluateBooleanArg(args.save)
verbose = evaluateBooleanArg(args.verbose)
log = evaluateBooleanArg(args.log)
night = args.night
if night != None and schedule =='NSVS':
    print('ArgumentError: Can only plot/save indvidual nights for ROTSE-I and III data')
    night = None
dev = evaluateBooleanArg(args.dev)
main(fileDir, schedule, ra, dec, threshold, bitmask, minUncertainty, xerror, plot, picklePlot, save, verbose, log, night, dev)
