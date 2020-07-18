import math
import argparse
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
import scipy
from scipy import stats as st
from astropy.io import fits
from decimal import Decimal
plt.rc('legend', fontsize = 12)

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

def getlc(match, refra, refdec):
    match_file = None
    if isinstance(match, str):
        match_file = match
        match, tele = read_data_file(match_file)
    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]
    cond = np.logical_and.reduce((np.abs(match_ra-refra) < 0.001, np.abs(match_dec-refdec) < 0.001))
    goodobj = np.where(cond)
    objid = goodobj[0]
    match_merr = list(match.field('MERR')[0][objid][0])
    match_m = (list(match.field('M')[0][objid][0]))
    match_jd = match.field('JD')[0]
    curve = list()
    for q in range(len(match_jd)):
        epoch = match_jd[q]
        mag = match_m[q]
        magerr = match_merr[q]
        point = (epoch,mag,magerr)
        curve.append(point)
    lc = list()
    for i in curve:
        if i[1] > 0:
            lc.append(i)
    return lc

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
    return curve

def get_matchstructs(match_structures):
    cwd = os.getcwd()
    os.chdir(match_structures)
    temp_matchs = list()
    fits = glob.glob("*.fit")
    dats = glob.glob("*.dat")
    datcs = glob.glob("*.datc")
    for fit in fits:
        temp_matchs.append(fit)
    for dat in dats:
        temp_matchs.append(dat)
    for datc in datcs:
        temp_matchs.append(datc)
    return temp_matchs, cwd

def find_target(vra, vdec, temp_matchs):
    matchs = list()
    target_lc = list()
    for match in temp_matchs:
        try:
            lc = get_data(vra, vdec, match)
            for i in lc:
                target_lc.append(i)
            print(f"Target found in {match}")
            matchs.append(match)
        except IndexError:
            print(f"Cannot find target in {match}; this match structure was removed from the list")
            pass
    return matchs, target_lc

def getobjids(inmatch, refra, refdec, radius):
    match_file = None
    if isinstance(inmatch, str):
        match_file = inmatch
        match, tele = read_data_file(match_file)
    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]
    cond = np.logical_and.reduce((np.abs(match_ra-refra) <= radius, np.abs(match_dec-refdec) <= radius))
    objects = list(np.where(cond)[0])
    goodobj = []
    for x in objects:
        coords = getcoords(inmatch, x)
        if math.sqrt((coords[0] - refra) ** 2 + (coords[1] - refdec) ** 2) <= radius:
            goodobj.append(x)
    return goodobj

def getcoords(match, objid):
    match_file = None
    if isinstance(match, str):
        match_file = match
        match, tele = read_data_file(match_file)
    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]
    ras = list()
    for i in match_ra:
        ras.append(i)
    decs = list()
    for i in match_dec:
        decs.append(i)
    result = [ras[objid],decs[objid]]
    return result

def mag2flux(in_mag):
    out_flux = float((3.636)*((10)**(((-1)*float(in_mag) / (2.5)))))
    return out_flux

def flux2mag(in_flux): 
    out_mag = float((-2.5)*math.log10(float(in_flux) / (3.636)))
    return out_mag

def avmag(data):  
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

def closest_refs(candidates,vra,vdec,requested_refstars):
    proxlist = list()
    prox_and_cand = list()
    for i in candidates:
        prox = ((abs(i[0][0]-vra)**2)+(abs(i[0][1]-vdec)**2))**0.5
        proxlist.append(prox)
        prox_and_cand.append([prox, i])
    sorted_stars = list()
    while (len(proxlist)) > 0:
        closest = min(proxlist)
        for j in prox_and_cand:
            if j[0] == closest:
                sorted_stars.append(j[1])
                proxlist.remove(j[0])
    output = list()
    for k in range(requested_refstars):
        output.append(sorted_stars[k])
    return output

def print_lightcurve(lightcurve, xerror):
    print("***FINAL TARGET LIGHTCURVE***")
    if xerror:
        for i in lightcurve:
            print(i[0], i[1], i[2], i[3])
    else:
        for i in lightcurve:
            print(i[0], i[1], i[2])

def save_lightcurve(lightcurve, cwd, vra, vdec):
    os.chdir(cwd)
    filename = 'lightcurve_ra'+str(vra)+'_dec'+str(vdec)+'.dat'
    print(f"You can find a copy of the lightcurve named {filename} in the same directory as lccal.py")
    np.savetxt(filename, lightcurve, fmt = '%.11f')

def save_log(log_params, cwd, vra, vdec):
    os.chdir(cwd)
    filename = 'log_ra'+str(vra)+'_dec'+str(vdec)+'.dat'
    print(f"You can find a copy of the log file named {filename} in the same directory as lccal.py")
    open(f'{filename}', 'w').writelines('%s\n' % x for x in log_params)

def arg2floatorbool(arg):
    pass

def get_chisq(observations, pval_flag): # Computes chi-square statistic and p-value based on a single Gaussian
    meanmag = np.mean(observations)
    magstdev = np.std(observations)
    observed, bins = np.histogram(observations, bins = 'auto')
    def gaussian(x, meanmag, magstdev): # Gaussian function used to compute probabilities for each bin
        return (1 / (magstdev * (2 * math.pi) ** 0.5)) * (math.e ** (-0.5 * ((x - meanmag) / magstdev) ** 2))
    expected = [scipy.integrate.quad(gaussian, bins[i], bins[i + 1], args = (meanmag, magstdev)) for i in range(len(bins) - 1)] # Integrates Gaussian function over each bin to compute each bin's probability
    expected = [len(observations) * x[0] for x in expected] # Computes expected number of observations in each bin
    chisq, pval = st.chisquare(observed, expected, 1) # Computes chi-square statistic and p-value
    chisq = chisq / ((len(bins) - 1) - 2) # Computes reduced chi-square statistic (chi-square per degree of freedom)
    if pval_flag:
        return chisq, pval
    else:
        return chisq

def R1_unconex(lightcurve, m_lim):
    grace_time = 40 / 86400
    output = []
    used = []
    above_m_lim = 0
    below_m_lim = 0
    removedepochs = 0
    for i in range(len(lightcurve) - 1):
        if not lightcurve[i][4] - 4 <= lightcurve[i][1]:
            below_m_lim += 1
        if not lightcurve[i][1] <= lightcurve[i][4]:
            above_m_lim += 1
        if lightcurve[i] not in used and lightcurve[i + 1] not in used:
            if abs(lightcurve[i][0] - lightcurve[i + 1][0]) <= lightcurve[i][3] + grace_time:
                if abs(lightcurve[i][1] - lightcurve[i + 1][1]) <= 2 * (((lightcurve[i][2] ** 2) + (lightcurve[i + 1][2] ** 2)) ** 0.5):
                    epoch = (lightcurve[i][0] + lightcurve[i + 1][0] + lightcurve[i + 1][3]) / 2
                    mag = flux2mag((mag2flux(lightcurve[i][1]) + mag2flux(lightcurve[i + 1][1])) / 2)
                    xerr = abs(lightcurve[i][0] - lightcurve[i + 1][0]) + lightcurve[i + 1][3]
                    yerr = ((lightcurve[i][2] ** (-2) + lightcurve[i + 1][2] ** (-2)) ** (-1)) ** 0.5
                    if m_lim:
                        if lightcurve[i][4] - 4 <= lightcurve[i][1] <= lightcurve[i][4] and lightcurve[i + 1][4] - 4 <= lightcurve[i + 1][1] <= lightcurve[i + 1][4]:
                            output.append([epoch, xerr, mag, yerr])
                            used.append(lightcurve[i])
                            used.append(lightcurve[i + 1])
                        else:
                            removedepochs += 1
                    else:
                        output.append([epoch, xerr, mag, yerr])
                        used.append(lightcurve[i])
                        used.append(lightcurve[i + 1])
                else:
                    removedepochs += 1           
            else:
                removedepochs += 1
    return output, removedepochs, above_m_lim, below_m_lim
    
def R3_unconex():
    return

def get_R1night(lightcurve, nights, epochindex):
    rounded_lightcurve = []
    night_count = [0]
    for x in range(len(lightcurve)-1):
        if int(lightcurve[x][epochindex]) != int(lightcurve[x + 1][epochindex]):
            night_count.append(x + 1)
    final_lightcurve = [n for x in nights for n in lightcurve[night_count[x - 1]:night_count[x]]]
    return final_lightcurve

def get_R3night():
    pass

def unconex(rotse, match_structures, vra, vdec, m_lim, xerrorbars, plots, log, nights):
    temp_matchs, cwd = get_matchstructs(match_structures)
    matchs, targetcurve = find_target(vra, vdec, temp_matchs)
    targetcurve = [x for x in targetcurve if 0 < x[1] < 99]
    #for x in targetcurve:
        #print(x[0], x[1], x[2], x[3], x[4])
    filtered_targetcurve, removedepochs, above_m_lim, below_m_lim = R1_unconex(targetcurve, m_lim)
    if not xerrorbars:
        filtered_targetcurve = [[obs[0], obs[2], obs[3]] for obs in filtered_targetcurve]
    print(f'Filtration removed {removedepochs} discrepant observations out of {len(targetcurve)} total observations ({round((1 - removedepochs / len(targetcurve)) * 100, 2)}% filtration efficiency)')
    print(f'Filtration averaged {len(targetcurve) - removedepochs} non-discrepant observations to {len(filtered_targetcurve)} observations ({round(len(filtered_targetcurve) / (len(targetcurve) - removedepochs) * 100, 2)}% averaging efficiency)')
    print(f'Filtration retained {len(filtered_targetcurve)} observations out of {len(targetcurve)} total original observations ({round(len(filtered_targetcurve) / len(targetcurve) * 100, 2)}% of original observations)')
    print(f'{above_m_lim} observations exceeded the limiting magnitude range ({round(above_m_lim / len(targetcurve) * 100, 2)}% of original observations)')
    print(f'{below_m_lim} observations fell below the limiting magnitude range ({round(below_m_lim / len(targetcurve) * 100, 2)}% of original observations)')
    if not m_lim:
        print('Note: --m_lim was passed as False and observations outside the limiting magnitude range may have been retained')
    print('Note: averaging efficiency should be ~50%')
    if nights != None:
        targetcurve = get_R1night(targetcurve, nights, 0)
        filtered_targetcurve = get_R1night(filtered_targetcurve, nights, 0)
    if plots and not xerrorbars:
        fig, axs = plt.subplots(2, sharex=True)
        plt.suptitle(f'Target Filtration: {vra} {vdec}')
        axs[0].errorbar([x[0] for x in targetcurve], [x[1] for x in targetcurve], yerr = [x[2] for x in targetcurve], fmt='o')
        axs[0].set_title('Unfiltered Light Curve')
        axs[0].invert_yaxis()
        axs[0].grid(axis='both', alpha=0.75)
        axs[0].set(xlabel = 'Time (MJD)')
        axs[0].set(ylabel='Magnitude')
        axs[1].errorbar([x[0] for x in filtered_targetcurve], [x[1] for x in filtered_targetcurve], yerr = [x[2] for x in filtered_targetcurve], fmt='o', color='g')
        axs[1].set_title('Filtered Light Curve')
        axs[1].invert_yaxis()
        axs[1].grid(axis='both', alpha=0.75)
        axs[1].set(xlabel = 'Time (MJD)')
        axs[1].set(ylabel='Magnitude')
        plt.show(block=False)
    elif plots and xerrorbars:
        fig, axs = plt.subplots(2, sharex=True)
        plt.suptitle(f'Target Filtration: {vra} {vdec}')
        axs[0].errorbar([x[0] for x in targetcurve], [x[1] for x in targetcurve], yerr = [x[2] for x in targetcurve], fmt='o')
        axs[0].set_title('Unfiltered Light Curve')
        axs[0].invert_yaxis()
        axs[0].grid(axis='both', alpha=0.75)
        axs[0].set(xlabel = 'Time (MJD)')
        axs[0].set(ylabel='Magnitude')
        axs[1].errorbar([x[0] for x in filtered_targetcurve], [x[2] for x in filtered_targetcurve], xerr = [x[1] for x in filtered_targetcurve], yerr = [x[3] for x in filtered_targetcurve], fmt='o', color='g')
        axs[1].set_title('Filtered Light Curve')
        axs[1].invert_yaxis()
        axs[1].grid(axis='both', alpha=0.75)
        axs[1].set(xlabel = 'Time (MJD)')
        axs[1].set(ylabel='Magnitude')
        plt.show(block=False)
    if verbose:
        print_lightcurve(filtered_targetcurve, xerrorbars)
    #print(f'Unfiltered mean magnitude: {round(float(np.mean([x[1] for x in targetcurve])), 3)}')
    #print(f'Unfiltered mean error: {round(float(np.mean([x[2] for x in targetcurve])), 3)}')
    #print(f'Unfiltered standard deviation: {round(float(np.std([x[1] for x in targetcurve])), 3)}')
    #print(f'Unfiltered reduced chi-square statistic: {round(float(get_chisq([x[1] for x in targetcurve], True)[0]), 3)}')
    #print(f'Unfiltered p-value: {round(float(get_chisq([x[1] for x in targetcurve], True)[1]), 3)}')
    #print(f'Filtered mean magnitude: {round(float(np.mean([x[1] for x in filtered_targetcurve])), 3)}')
    #print(f'Filtered mean error: {round(float(np.mean([x[2] for x in filtered_targetcurve])), 3)}')
    #print(f'Filtered standard deviation: {round(float(np.std([x[1] for x in filtered_targetcurve])), 3)}')
    #print(f'Filtered reduced chi-square statistic: {round(float(get_chisq([x[1] for x in filtered_targetcurve], True)[0]), 3)}')
    #print(f'Filtered p-value: {round(float(get_chisq([x[1] for x in filtered_targetcurve], True)[1]), 3)}')
    save_lightcurve(filtered_targetcurve, cwd, vra, vdec)
    if log:
        log_params = [f'Total observations: {len(targetcurve)}', f'Final observations: {len(filtered_targetcurve)}', f'Filtration efficiency: {round((1 - removedepochs / len(targetcurve)) * 100, 2)}%', f'Averaging efficiency: {round(len(filtered_targetcurve) / (len(targetcurve) - removedepochs) * 100, 2)}%', 
        f'Observations retained: {round(len(filtered_targetcurve) / len(targetcurve) * 100, 2)}%', f'Observations greater than limiting magnitude range: {round(above_m_lim /  len(targetcurve) * 100, 2)}%', 
        f'Observations less than limiting magnitude range: {round(below_m_lim / len(targetcurve) * 100, 2)}%']
        save_log(log_params, cwd, vra, vdec)
    return

def lccal(rotse, operation, match_structures, vra, vdec, requested_refstars, radius, max_mean_error, chisq_input, avmag_input, decent_epochs_input, teststar, m_lim, xerrorbars, plots, log, nights, verbose):

    def find_refstars(matchs, ra, dec, radius, target_lc):
        def cuts(package):
            cand_coords = package[0]
            lightcurve = package[1]
            per_match = package[2]
            good_obs = package[3]
            def is_not_target():
                allowed_diff = 0.001
                if not ra - allowed_diff <= cand_coords[0] <= ra + allowed_diff and not dec - allowed_diff <= cand_coords[1] <= dec + allowed_diff:
                    return True
                else:
                    return False
            def has_all_epochs():
                target_epochs = [obs[0] for obs in target_lc]
                candidate_epochs = [obs[0] for obs in lightcurve]
                if target_epochs == candidate_epochs:
                    return True
                else:
                    return False
            def avmag_within_mag_limits():
                av_m_lim = math.fsum([obs[4] for obs in lightcurve])/len(lightcurve)
                if av_m_lim-4 <= avmag(lightcurve) <= av_m_lim:
                    return True
                else:
                    return False
            def decent_epochs(threshold):
                if len([obs[1] for obs in good_obs if obs[4]-4 <= obs[1] <= obs[4]])/len(lightcurve) >= threshold:
                    return True
                else:
                    return False
            def within_allowed_chisq(allowed_chisq):
                nights_num = len(per_match)
                passed_nights = 0
                for lc in per_match:
                    chisq = get_chisq([obs[1] for obs in lc if 0 < obs[1] < 99], False)
                    if chisq <= allowed_chisq:
                        passed_nights += 1
                if len([obs for obs in lightcurve if 0 < obs[1] < 99])/len(lightcurve) >= decent_epochs_input and passed_nights/nights_num >= 0.5:
                    return True
                else:
                    return False
            def under_mean_error():
                if math.fsum([obs[2] for obs in good_obs])/len(good_obs) <= max_mean_error:
                    return True
                else:
                    return False
            if not is_not_target():
                return False
            if max_mean_error != False:
                if not under_mean_error():
                    return False
            if not has_all_epochs():
                return False
            if not decent_epochs(decent_epochs_input):
                return False
            if chisq_input != False:
                if not within_allowed_chisq(chisq_input):
                    return False
            return True
        refstars = list()
        surroundstars = getobjids(matchs[0], ra, dec, radius)
        test_candidates = list()
        for star in surroundstars:
            try:
                coords = getcoords(matchs[0], star)
                full_lightcurve = list()
                lightcurves_per_match = list()
                for i in range(len(matchs)):
                    match = matchs[i]
                    match_lightcurve = order(get_data(coords[0],coords[1],match))
                    lightcurves_per_match.append(match_lightcurve)
                    for obs in match_lightcurve:
                        full_lightcurve.append(obs)
                good_lightcurve = [obs for obs in full_lightcurve if 0 < obs[1] < 99]
                package = [coords, full_lightcurve, lightcurves_per_match, good_lightcurve]
                if cuts(package):
                    refstars.append([coords, avmag(good_lightcurve), good_lightcurve])
                else:
                    allowed_diff = 0.001
                    if not ra - allowed_diff <= coords[0] <= ra + allowed_diff and not dec - allowed_diff <= coords[1] <= dec + allowed_diff:
                        test_candidates.append([coords, good_lightcurve, lightcurves_per_match])
            except IndexError:
                pass
        return [refstars, test_candidates]

    def get_corrections(refstars_package, target_lc):
        corrections = list()
        for epoch in [i[0] for i in target_lc if 0 < i[1] < 99]:
            diffs = list()
            for star in refstars_package:
                trumag = star[1]
                lightcurve = star[2]
                for obs in lightcurve:
                    if obs[0] == epoch:
                        diffs.append(trumag - obs[1])
                if len(diffs) == len(refstars_package):
                    correction = math.fsum(diffs) / len(diffs)
                    corrections.append([epoch, correction])
        if verbose:
            print("Applying these corrections to the target light curve:")
            for correction in corrections:
                print(f"Epoch: {correction[0]}, Correction: {correction[1]}")
        return corrections

    def calibrate_target(corrections, target_lc):
        calibrated_target_lightcurve = list()
        target_good_lightcurve = [i for i in target_lc if 0 < i[1] < 99]
        print(f"{len(target_lc)-len(target_good_lightcurve)} observations in the target lightcurve were marked as unusable due to unphysical magnitude measurements (<0 or >99) and were removed")
        for obs in target_good_lightcurve:
            for c in corrections:
                if obs[0] == c[0]:
                    calibrated_target_lightcurve.append([obs[0], obs[1]+c[1], obs[2], obs[3], obs[4]])
        return calibrated_target_lightcurve

    def find_test_star(test_candidates):
        test_candidates = closest_refs(test_candidates,vra,vdec,len(test_candidates))
        test5 = list()
        X2s = list()
        i = 0
        while len(test5) < 5:
            lc = test_candidates[i][1]
            if len(lc)/len(target_lc) >= decent_epochs_input:
                cand_X2s = list()
                for k in test_candidates[i][2]:
                    cand_X2s.append(get_chisq([elt[1] for elt in test_candidates[i][2]], False))   
                X2 = math.fsum(cand_X2s)/len(cand_X2s)
                test5.append([X2, lc, test_candidates[i][2]])
                X2s.append(X2)
            i += 1
            if i >= len(test_candidates):
                print("Could not find suitable test star")
                return None
        for j in test5:
            if min(X2s) == j[0]:      
                return [j[1], j[2]]

    def calibrate_test_star(corrections, test_package):
        test_lc = test_package[0]
        test_by_match = test_package[1]
        '''
        calibrated_test_lc = list()
        for obs in test_lc:
            for c in corrections:
                if obs[0] == c[0]:
                    calibrated_test_lc.append([obs[0], obs[1]+c[1], obs[2]])
        original_X2 = get_chisq([i[1] for i in test_lc])
        calibrated_X2 = get_chisq([i[1] for i in calibrated_test_lc])
        print("calibration applied to a test star changed chi-squared from",original_X2,"to",calibrated_X2)
        '''
        uncalibrated_X2s = list()
        calibrated_X2s = list()
        for lc in test_by_match:
            uncalibrated_X2s.append(get_chisq([k[1] for k in lc], False))
            new_lc = list()
            for obs in lc:
                for c in corrections:
                    if obs[0] == c[0]:
                        new_lc.append([obs[0], obs[1]+c[1], obs[2]])
            calibrated_X2s.append(get_chisq([i[1] for i in new_lc], False))
        print("***TEST STAR NIGHTLY CHI-SQUAREDS***")
        for l in range(len(calibrated_X2s)):
            print(f"Uncalibrated: {uncalibrated_X2s[l]}, Calibrated: {calibrated_X2s[l]}")
        average_uncalibrated_X2 = math.fsum(uncalibrated_X2s)/len(uncalibrated_X2s)
        average_calibrated_X2 = math.fsum(calibrated_X2s)/len(calibrated_X2s)
        print(f"Calibration applied to a test star changed the average nightly chi-squared from {average_uncalibrated_X2} to {average_calibrated_X2}")
        return
    
    def get_plots(uncalibrated_curve, calibrated_curve, filtered_curve):
        if not xerrorbars:
            if operation == 'both':
                fig, axs = plt.subplots(3, sharex=True, sharey=True)
                plt.suptitle(f'Target Calibration and Filtration: {vra} {vdec} \n Reference Stars: {requested_refstars}, Radius: {radius} degrees')
            else: 
                fig, axs = plt.subplots(2, sharex=True, sharey=True)
                plt.suptitle(f'Target Calibration: {vra} {vdec} \n Reference Stars: {requested_refstars}, Radius: {radius} degrees')
            axs[0].errorbar([x[0] for x in uncalibrated_curve], [x[1] for x in uncalibrated_curve], yerr = [x[2] for x in uncalibrated_curve], fmt='o')
            axs[0].set_title('Uncalibrated Light Curve')
            axs[0].set(xlabel='Time (MJD)')
            axs[0].set(ylabel='Magnitude')
            axs[0].grid(axis='both', alpha=0.75)
            axs[0].invert_yaxis()
            axs[1].errorbar([x[0] for x in calibrated_curve], [x[1] for x in calibrated_curve], yerr = [x[2] for x in calibrated_curve], fmt='o', color='g')
            axs[1].set_title('Calibrated Light Curve')
            axs[1].set(xlabel='Time (MJD)')
            axs[1].set(ylabel='Magnitude')
            axs[1].grid(axis='both', alpha=0.75)
            if operation == 'both':
                axs[2].errorbar([x[0] for x in filtered_curve], [x[1] for x in filtered_curve], yerr = [x[2] for x in filtered_curve], fmt='o', color='orange')
                axs[2].set_title('Filtered Light Curve')
                axs[2].grid(axis='both', alpha=0.75)
                axs[2].set(xlabel = 'Time (MJD)')
                axs[2].set(ylabel='Magnitude')
            plt.show(block=False)
        else:
            fig, axs = plt.subplots(3, sharex=True, sharey=True)
            plt.suptitle(f'Target Calibration and Filtration: {vra} {vdec} \n Reference Stars: {requested_refstars}, Radius: {radius} degrees')
            axs[0].errorbar([x[0] for x in uncalibrated_curve], [x[1] for x in uncalibrated_curve], yerr = [x[2] for x in uncalibrated_curve], fmt='o')
            axs[0].set_title('Uncalibrated Light Curve')
            axs[0].invert_yaxis()
            axs[0].grid(axis='both', alpha=0.75)
            axs[0].set(xlabel = 'Time (MJD)')
            axs[0].set(ylabel='Magnitude')
            axs[0].legend()
            axs[1].errorbar([x[0] for x in calibrated_curve], [x[1] for x in calibrated_curve], yerr = [x[2] for x in calibrated_curve], fmt='o', color='g')
            axs[1].set_title('Calibrated Light Curve')
            axs[1].set(xlabel='Time (MJD)')
            axs[1].set(ylabel='Magnitude')
            axs[1].grid(axis='both', alpha=0.75)
            axs[1].legend()
            axs[2].errorbar([x[0] for x in filtered_curve], [x[2] for x in filtered_curve], xerr = [x[1] for x in filtered_curve], yerr = [x[3] for x in filtered_curve], fmt='o', color='orange')            
            axs[2].set_title('Filtered Light Curve')
            axs[2].grid(axis='both', alpha=0.75)
            axs[2].set(xlabel = 'Time (MJD)')
            axs[2].set(ylabel='Magnitude')
            axs[2].legend()
            plt.show(block=False)
        return

    temp_matchs, cwd = get_matchstructs(match_structures)
    matchs, target_lc = find_target(vra, vdec, temp_matchs)
    one = find_refstars(matchs, vra, vdec, radius, target_lc)
    if teststar:
        test_star = find_test_star(one[1])
    one = one[0]
    if len(one) < requested_refstars:
        print(f"You requested {requested_refstars} reference stars, but only {len(one)} were found which meet your specifications")
        print("Try again with a larger search radius or looser specifications")
        sys.exit()
    print(f"Using the nearest {requested_refstars} of {len(one)} available reference stars")
    one = closest_refs(one,vra,vdec,requested_refstars)
    print("***REFERENCE STARS***")
    for i in range(len(one)):
        print(f"Reference Star {i}- RA: {one[i][0][0]}, Dec: {one[i][0][1]}, Mean magnitude: {one[i][1]}, Mean error: {math.fsum([obs[2] for obs in one[i][2]])/len(one[i][2])}, Good observations: {len(one[i][2])}")
    two = get_corrections(one,target_lc)
    if (test_star != None):
        calibrate_test_star(two, test_star)
    three = calibrate_target(two,target_lc)
    unfiltered = three
    print(f'Calibration retained {len(three)} observations out of {len(target_lc)} total observations ({100 * round(len(three) / len(target_lc), 2)}% calibration efficiency)')
    if rotse == 'R1' and operation == 'both':
        three, removedepochs, above_m_lim, below_m_lim = R1_unconex(three, m_lim)
    if rotse == 'R3' and operation == 'both':
        pass
    if xerrorbars and operation == 'both':
        three = [[obs[0], obs[2], obs[3], obs[4]] for obs in three]
    elif not xerrorbars and operation == 'both':
        three = [[obs[0], obs[2], obs[3]] for obs in three]
    if operation == 'both':
        print(f'Filtration removed {removedepochs} discrepant observations out of {len(unfiltered)} total calibrated observations ({round((1 - removedepochs / len(unfiltered)) * 100, 2)}% filtration efficiency)')
        print(f'Filtration averaged {len(unfiltered) - removedepochs} non-discrepant calibrated observations to {len(three)} observations ({round(len(three) / (len(unfiltered) - removedepochs) * 100, 2)}% averaging efficiency)')
        print('Note: averaging efficiency should be ~50%')
        print(f'Filtration retained {len(three)} observations out of {len(unfiltered)} total calibrated observations ({round(len(three) / len(unfiltered) * 100, 2)}% of calibrated observations)')
        print(f'{above_m_lim} observations exceeded the limiting magnitude range ({round(above_m_lim / len(unfiltered) * 100, 2)}% of original observations)')
        print(f'{below_m_lim} observations fell below the limiting magnitude range ({round(below_m_lim / len(unfiltered) * 100, 2)}% of original observations)')
        if not m_lim:
            print('Note: --m_lim was passed as False and observations outside the limiting magnitude range may have been retained')
        print(f'Calibration and filtration together retained {len(three)} observations out {len(target_lc)} total original observations ({round(len(three)/ len(target_lc) * 100, 2)}% of original observations')
    raw = [obs for obs in target_lc if 0 < obs[1] < 99]
    if nights != None:
        raw = get_R1night(raw, nights, 0)
        three = get_R1night(three, nights, 0)
        unfiltered = get_R1night(unfiltered, nights, 0)
    save_lightcurve(three, cwd, vra, vdec)
    if log:
        if operation == 'both':
            log_params = [f'Total observations: {len(target_lc)}', f'Final observations: {len(three)}', f'Calibration effciency: {round(len(unfiltered) / len(target_lc) * 100, 2)}%', f'Filtration efficiency: {round((1 - removedepochs / len(unfiltered)) * 100, 2)}%', f'Averaging efficiency: {round(len(three) / (len(unfiltered) - removedepochs) * 100, 2)}%', 
            f'Observations retained: {round(len(three) / len(unfiltered) * 100, 2)}%', f'Observations greater than limiting magnitude range: {round(above_m_lim /  len(unfiltered) * 100, 2)}%', 
            f'Observations less than limiting magnitude range: {round(below_m_lim / len(unfiltered) * 100, 2)}%']
        else:
            log_params = [f'Total observations: {len(target_lc)}', f'Final observations: {len(three)}', f'Calibration effciency: {round(len(three) / len(target_lc) * 100, 2)}%']
        save_log(log_params, cwd, vra, vdec)
    if verbose:
        print_lightcurve(three, xerrorbars)
    if plots:
        get_plots(raw, unfiltered, three)
    #print(f'Uncalibrated mean magnitude: {round(float(np.mean([x[1] for x in raw])), 3)}')
    #print(f'Uncalibrated mean error: {round(float(np.mean([x[2] for x in raw])), 3)}')
    #print(f'Uncalibrated standard deviation: {round(float(np.std([x[1] for x in raw])), 3)}')
    #print(f'Uncalibrated reduced chi-square statistic: {round(float(get_chisq([x[1] for x in raw], True)[0]), 3)}')
    #print(f'Uncalibrated p-value: {round(float(get_chisq([x[1] for x in raw], True)[1]), 3)}')
    #print(f'Calibrated mean magnitude: {round(float(np.mean([x[1] for x in unfiltered])), 3)}')
    #print(f'Calibrated mean error: {round(float(np.mean([x[2] for x in unfiltered])), 3)}')
    #print(f'Calibrated standard deviation: {round(float(np.std([x[1] for x in unfiltered])), 3)}')
    #print(f'Calibrated reduced chi-square statistic: {round(float(get_chisq([x[1] for x in unfiltered], True)[0]), 3)}')
    #print(f'Calibrated p-value: {round(float(get_chisq([x[1] for x in unfiltered], True)[1]), 3)}')
    #if operation == 'both':
        #print(f'Filtered mean magnitude: {round(float(np.mean([x[1] for x in three])), 3)}')
        #print(f'Filtered mean error: {round(float(np.mean([x[2] for x in three])), 3)}')
        #print(f'Filtered standard deviation: {round(float(np.std([x[1] for x in three])), 3)}')
        #print(f'Filtered reduced chi-square statistic: {round(float(get_chisq([x[1] for x in three], True)[0]), 3)}')
        #print(f'Filtered p-value: {round(float(get_chisq([x[1] for x in three], True)[1]), 3)}')
    return three


parser = argparse.ArgumentParser()
parser.add_argument("rotse", choices = ['R1', 'R3'], help = 'ROTSE experiment that originated data')
parser.add_argument("operation", choices = ['calibrate', 'filter', 'both'], help = 'Operation to perform on data: calibrate with lccal, filter with unconex, or both')
parser.add_argument("match_structures", type = str, help = 'Path to match structure directory')
parser.add_argument("vra", type = float, help = 'Target RA (decimal format)')
parser.add_argument("vdec", type = float, help = 'Target Dec (decimal format)')
parser.add_argument("--requested_refstars", "-ref", default = 5, type = int, help = 'Requested number of reference stars')
parser.add_argument("--radius", "-r", default = 0.1, type = float, help = 'Maximum search radius for reference stars')
parser.add_argument("--max_mean_error", "-e", default = 0.06, help = 'Maximum mean photometric error of reference stars (False to disable)')
parser.add_argument("--chisq", "-c", default = 10, help = 'Maximum reduced chi-square statistic of reference stars on a majority of nights (False to disable)')
parser.add_argument("--avmag", "-a", default = True, help = 'Average magnitude of reference stars within limiting magnitude (False to disable)')
parser.add_argument("--m_lim", "-m", default = True, help = 'Filter observations based upon limiting magnitude (False to disable)')
parser.add_argument("--decent_epochs", "-d", default = 0.9, help = 'Minimum fraction of reference star observations needed after unphysical observations have been removed')
parser.add_argument("--teststar", "-t", default = True, help = 'Locate and calibrate a test star to validate corrections (False to disable)')
parser.add_argument("--xerrorbars", "-x", default = False, help = 'Calculate error along x-axis (time) when filtering data (True to enable)')
parser.add_argument("--plots", "-p", default = False, help = 'Display target lightcurve before and after calibration and/or filtration (True to enable)')
parser.add_argument("--log", "-l", default = True, help = 'Save calibration and/or filtration metrics to log file (False to disable)')
parser.add_argument("--nights", "-n", default = None, help = 'Only save and/or print lightcurve of the given night')
parser.add_argument("--verbose", "-v", default = False, help = 'Print target calibrated and/or filtered lightcurve and additional information to terminal (True to enable)')
args = parser.parse_args()

rotse = args.rotse
operation = args.operation
match_structures = args.match_structures
vra = args.vra
vdec = args.vdec
requested_refstars = args.requested_refstars
radius = args.radius
max_mean_error = args.max_mean_error
if max_mean_error == 'False' or max_mean_error == 'false':
    max_mean_error = False
else:
    max_mean_error = float(max_mean_error)
chisq_input = args.chisq
if chisq_input == 'False' or chisq_input == 'false':
    chisq_input = False
else:
    chisq_input = float(chisq_input)
avmag_input = args.avmag
if avmag_input == 'False' or avmag_input == 'false':
    avmag_input = False
m_lim = args.m_lim
if m_lim == 'False' or m_lim == 'false':
    m_lim = False
decent_epochs_input = args.decent_epochs
if decent_epochs_input == 'False' or decent_epochs_input == 'false':
    decent_epochs_input = False
else:
    decent_epochs_input = float(decent_epochs_input)
teststar = args.teststar
if teststar == 'False' or teststar == 'false':
    teststar = False
else:
    teststar = True
xerrorbars = args.xerrorbars
if xerrorbars == 'True' or teststar == 'true':
    xerrorbars = True
else:
    xerrorbars = False
plots = args.plots
if plots == 'True' or plots == 'true':
    plots = True
else:
    plots = False
log = args.log
if log == 'False' or log == 'false':
    log = False
nights = args.nights
if nights != None:
    nights = nights.split(",")
    nights = [int(x) for x in nights]
verbose = args.verbose
if verbose  == 'True' or verbose == 'true':
    verbose = True
else:
    verbose = False

if rotse == 'R3':
    print('ROTSE-III functionality is not currently supported')
    print('Did you mean ROTSE-I? If so, please try again using \'R1\' instead of \'R3\'')
    sys.exit()
if operation == 'calibrate' or operation == 'both':
    lccal(rotse, operation, match_structures, vra, vdec, requested_refstars, radius, max_mean_error, chisq_input, avmag_input, decent_epochs_input, teststar, m_lim, xerrorbars, plots, log, nights, verbose)
if operation == 'filter':
    unconex(rotse, match_structures, vra, vdec, m_lim, xerrorbars, plots, log, nights)
plt.show()

#TODO: Come up with Github setup compatible with co-op coding
#TODO: Determine specificity of try/except loops
#TODO: Figure out how magnitudes work
#TODO: Look into NumPy sorting/arranging
#TODO: Testing chi-squareds is currently done match structure-to-match structure to facilitate
#nightly light curves. It is not necessarily true that 1 match structure = 1 night. For R3 data,
#this will need to be generalized.
#TODO: Revist iterative search radius

#DEBUGGING/FIXES
#TODO: Why is less than 90% of data being retained?

#CUT ORDER:
# Not target
# Remove unphysical measurements
# Mean error within limit
# Remove epochs whose corrections' standard deviation exceeds limit
# Decent epochs >= minimum
