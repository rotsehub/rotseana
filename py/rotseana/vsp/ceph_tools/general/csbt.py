# created by Aidan Kehoe
# editted by Jacob Juvan
# last updated: June 16, 2025
from scipy.optimize import curve_fit
from operator import itemgetter
from scipy.optimize import fsolve
from scipy.signal import find_peaks
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
import matplotlib
import argparse as args
import numpy as np 
import math
import statistics as st
import sys
import pickle
import os


# define methods
'''
def collect(filename):
#### reads in file that is given returns it to use in the code####
    in_file = open(filename,'r')
    for a_line in in_file:
        print(a_line)
    return in_file
'''

def remove_tempfile():
    return

def collect_table(data, command):
###generates the lightcurve ndarray###
    if command == "sp":
        #single_phase - current unusable#
        lightcurve = np.genfromtxt(data, dtype = None, delimiter = None, skip_header = 2, names = ["phase", "mag", "error"])
    elif command == "lcs":
        #lcvis script - run directly from lcvis#
        import ast
        lightcurve = ast.literal_eval(data)
    elif command == "lcf":
        #lcvis file - run from an lcvis file (default)#
        lightcurve = np.loadtxt(data).tolist()
    return lightcurve
    

def get_lowest_mag(lightcurve, lc_mags_sort, num):
###sorts through the array lightcurve and skips over the dupilcate lines and puts each into separate lines###
    low_phase = []
    low_mag = []
    low_error = []
    lc_mags_popped = lc_mags_sort.copy()
    
    for obs in lc_mags_sort:
        if obs[0] >= 1:
            index = lc_mags_popped.index(obs)
            lc_mags_popped.pop(index)
    
    for i in range(num):
        phase = lc_mags_popped[i][0]
        mag = lc_mags_popped[i][1]
        error = lc_mags_popped[i][2]
        if (phase not in low_phase):
            low_phase.append(phase)
            low_mag.append(mag)      
            low_error.append(error) 

    return low_phase, low_mag, low_error

def get_median(phase, mag):
### finds the meidan of Phase and Mag in the lightcurve file to help find the points around the first parabola###
    mag_median = st.median(mag)
    phase_median = st.median(phase)
    return mag_median, phase_median


def difference2(phase, phase_median, num):
### CURRENTLY UNUSED ###
    savei = []
    for i in range(num):
        diff = abs(phase[i] - phase_median)
        if diff < 0.1:
            savei.append(i)
    return savei
    

def objective_parab(x, a, b, c):
#creates the parbola for each Minima#
    return a * (x - b)**2 + c

def objective_sine(x, a, b, c):
#creates squared sine curve for maxima#
    return a * (np.sin(2 * np.pi * (x - b)))**2 + c

def fit_minimum(phase, mag, a, b, c):
#finds parabolic fit parameters#
    p0 = [None] * 3
    p0[0] = a # width
    p0[1] = b # phase shift
    p0[2] = c # mag shift
    params, pcov = curve_fit(objective_parab, phase, mag, p0)
    return params, pcov

def fit_maximums(phase, mag, a, b, c):
#finds sine fit parameters#
    p0 = [None] * 3
    p0[0] = a # amplitude
    p0[1] = b # phase shift
    p0[2] = c # mag shift
    params, pcov = curve_fit(objective_sine, phase, mag, p0)
    return params, pcov

def get_window(lightcurve, fitted_phase, num):
#Identifies all points part of the identified minima
    phase_min = []
    mag_min = []
    error_min = []
    for i in range(len(lightcurve)):
        abs_dif = abs(lightcurve[i][0] - fitted_phase)
        if abs_dif <= num:
            phase_min.append(lightcurve[i][0])
            mag_min.append(lightcurve[i][1])
            error_min.append(lightcurve[i][2])        
    return phase_min, mag_min, error_min

def calc_x2(phase_min, mag_min, error_min, a, b, c):
#calculates the x2 for each Minima which is the result of the package# 
###CURRENTLY UNUSED###
    mu = []
    calc_sum = []
    for i in range(len(phase_min)):
        mu.append(objective_parab(phase_min[i], a, b, c))
        calc_sum.append(( (mag_min[i] - mu[i]) / error_min[i] )**2)
    total_calc = np.sum(calc_sum)    
    x2 = total_calc / (len(phase_min) - 3)
    return x2

def calc_max_x2(phase_max, mag_max, error_max, a, b, c):
#calculates the x2 for the maxima#
###CURRENTLY UNUSED###
    mu = []
    calc_sum = []
    for i in range(len(phase_max)):
        mu.append(objective_sine(phase_max[i], a, b, c))
        calc_sum.append(( (mag_max[i] - mu[i]) / error_max[i] )**2)
    total_calc = np.sum(calc_sum)    
    max_x2 = total_calc / (len(phase_max) - 3)
    return max_x2


def get_x2(obs, expected, obs_err, expected_err, df):
#calculates x2 for each minima, taking into account fit errors#
    x2 = sum((obs - expected)**2 / (obs_err**2 + expected_err**2))  / df
    return x2




############# Start of setup of array#################

def exlightfile(data, command):
#########Reads in lightcurve file and sorts it by Magnitude##########
    lightcurve = collect_table(data, command)
    lightcurve2 = []
    for i in range(len(lightcurve)):
        if lightcurve[i][0] < 1:
            lightcurve2.append(list(lightcurve[i]))
    for i in range(len(lightcurve)):
        if lightcurve[i][0] < 1:
            lightcurve2.append([lightcurve[i][0]+1, lightcurve[i][1], lightcurve[i][2]])
    lc_mags_sort = sorted(lightcurve2, key = itemgetter(1), reverse = True)
    return lightcurve2, lc_mags_sort



def get_first_min(lc_mags_sort, lightcurve): 
####### Takes the sorted lightcurve and finds the Phase Mag Error and the parabola of it  ############
    phase1, mag1, error1 = get_lowest_mag(lightcurve, lc_mags_sort, numpoints)
    mag1_median, phase1_median = get_median(phase1, mag1)
    if phase1_median < window / 2:
        phase1_median += 1
        for i in range(len(phase1)):
            phase1[i] += 1
    elif phase1_median > 2 - window / 2:
        phase1_median -= 1
        for i in range(len(phase1)):
            phase1[i] -= 1
            
    phase1_cleaned = []
    mag1_cleaned = []
    error1_cleaned = []
    
    for obs in lightcurve:
        if abs(obs[0] - phase1_median) < window / 2:
             phase1_cleaned.append(obs[0])
             mag1_cleaned.append(obs[1])
             error1_cleaned.append(obs[2])
    a_min1 = -50.00
    
    #obtain fit parameters for first minima
    params_min1, pcov_min1 = fit_minimum(phase1_cleaned, mag1_cleaned, a_min1, phase1_median, mag1_median)
    a_min1, b_min1, c_min1 = params_min1
    fittedPhase_min1 = b_min1
    fittedMag_min1 = c_min1
    
    #not useful for calculations, just highlights arbitrary window
    phase_min1, mag_min1, error_min1 = get_window(lightcurve, fittedPhase_min1, window / 2)
    return phase_min1, mag_min1, error_min1, a_min1, fittedPhase_min1, fittedMag_min1, pcov_min1



def get_sec_min(lightcurve, fittedPhase_min1, phase_min1, lc_mags_sort, a):
######## Uses the a fit of the first Minima and finds the different parabola and Mag Phase aand Median ###############
    if (fittedPhase_min1 > 1):
        guessPhase = fittedPhase_min1 - 0.5
    else:
        guessPhase = fittedPhase_min1 + 0.5
    
    lightcurve_no_min1 = lightcurve.copy()
    lc_mags_sort_no_min1 = lc_mags_sort.copy()
    
    for obs in lightcurve:
        if (obs[0] in phase_min1):
            index = lightcurve_no_min1.index(obs)
            lightcurve_no_min1.pop(index)
            index_mags = lc_mags_sort_no_min1.index(obs)
            lc_mags_sort_no_min1.pop(index_mags)
        if (obs[0]+1 in phase_min1):
            index = lightcurve_no_min1.index(obs)
            lightcurve_no_min1.pop(index)
            index_mags = lc_mags_sort_no_min1.index(obs)
            lc_mags_sort_no_min1.pop(index_mags)
        if (obs[0]-1 in phase_min1):
            index = lightcurve_no_min1.index(obs)
            lightcurve_no_min1.pop(index)
            index_mags = lc_mags_sort_no_min1.index(obs)
            lc_mags_sort_no_min1.pop(index_mags)

    phase2, mag2, error2 = get_lowest_mag(lightcurve_no_min1, lc_mags_sort_no_min1, numpoints)
    mag2_median = st.median(mag2)
    phase2_median = st.median(phase2)
    phase2_cleaned = []
    mag2_cleaned = []
    error2_cleaned = []
    
    for obs in lightcurve_no_min1:
        if abs(obs[0] - guessPhase) < window / 2:
             phase2_cleaned.append(obs[0])
             mag2_cleaned.append(obs[1])
             error2_cleaned.append(obs[2])
    
    #obtain fit parameters for second minima
    params_min2, pcov_min2 = fit_minimum(phase2_cleaned, mag2_cleaned, a, phase2_median, mag2_median)
    a_min2, b_min2, c_min2 = params_min2
    
    #not useful for calculations, just highlights arbitrary window
    phase_min2, mag_min2, error_min2 = get_window(lightcurve, b_min2, window / 2)
    return phase_min2, mag_min2, error_min2, a_min2, b_min2, c_min2, pcov_min2

def moving_average(x, w):
#calculate moving average with window size w#
    return np.convolve(x, np.ones(w), 'valid') / w

def find_mins_move_avg(lightcurve):
#find minima based on peaks created by the moving average#
    phase = []
    mag = []
    for obs in lightcurve:
        phase.append(obs[0])
        mag.append(obs[1])
    phase = np.array(phase)
    mag = np.array(mag)
    
    phase_f = moving_average(phase, numpoints)
    mag_f = moving_average(mag, numpoints)
    #to show in same plot
    phase_f_ext = np.hstack([np.ones(numpoints//2)*phase_f[0],phase_f])
    mag_f_ext= np.hstack([np.ones(numpoints//2)*mag_f[0],mag_f])
    
    mins, _ = find_peaks(mag_f_ext, distance=int(len(lightcurve)/10), height = np.average(mag))
    
    phase_mins = phase_f_ext[mins].tolist()
    mag_mins = mag_f_ext[mins].tolist()
    
    return phase_mins, mag_mins

def get_mins_move_avg(lightcurve):
#obtain all points within the window around the minima#
    phase_mins, mag_mins = find_mins_move_avg(lightcurve)
    full_min_phases = []
    full_min_mags = []
    for i in range(len(phase_mins)):
        if phase_mins[i] > (window / 2) and phase_mins[i] < (2 - window / 2):
            full_min_phases.append(phase_mins[i])
            full_min_mags.append(mag_mins[i])
    
    min1_phase = full_min_phases[0]
    min1_mag = full_min_mags[0]
    
    min2_phase = full_min_phases[1]
    min2_mag = full_min_mags[1]
    
    phase1_cleaned = []
    mag1_cleaned = []
    error1_cleaned = []
        
    phase2_cleaned = []
    mag2_cleaned = []
    error2_cleaned = []
    
    for obs in lightcurve:
        if abs(obs[0] - min1_phase) < window / 2:
             phase1_cleaned.append(obs[0])
             mag1_cleaned.append(obs[1])
             error1_cleaned.append(obs[2])
        elif abs(obs[0] - min2_phase) < window / 2:
             phase2_cleaned.append(obs[0])
             mag2_cleaned.append(obs[1])
             error2_cleaned.append(obs[2])
    
    a = -50.00
             
    params_min1, pcov_min1 = fit_minimum(phase1_cleaned, mag1_cleaned, a, min1_phase, min1_mag)
    params_min2, pcov_min2 = fit_minimum(phase2_cleaned, mag2_cleaned, a, min2_phase, min2_mag)
    
    return phase1_cleaned, mag1_cleaned, error1_cleaned, params_min1, phase2_cleaned, mag2_cleaned, error2_cleaned, params_min2, pcov_min1, pcov_min2


def mag_to_flux(mags):
#convert magnitude to flux using Vega as zero-point
    fluxes = 10**((mags + 48.60)/(-2.5))
    return fluxes

def mag_to_flux_error(flux, mag_errors):
#convert magnitude error to flux error#
    flux_errors = np.log(10) / 2.5 * flux * mag_errors # mag-to-flux error propagation, derivative of previous equation
    return flux_errors


def error_propagation_parab(x, a, b, c, pcov):
#parabola fit error propagation, using covariance matrix and partial derivatives#
    dfa = (x - b)**2
    dfb = -2 * a * (x - b)
    dfc = 1
    var_f = (
        dfa**2 * pcov[0, 0] +
        dfb**2 * pcov[1, 1] +
        dfc**2 * pcov[2, 2] +
        2 * dfa * dfb * pcov[0, 1] +
        2 * dfa * dfc * pcov[0, 2] +
        2 * dfb * dfc * pcov[1, 2]
    )
    sigma_f = np.sqrt(var_f)
    return sigma_f

def error_propagation_sin(x, a, b, c, pcov):
#squared-sine fit error propagation, using covariance matrix and partial derivatives#
    dfa = (np.sin(2 * np.pi * (x - b)))**2
    dfb = -2 * a * np.pi * np.sin(4 * np.pi * (x-b))
    dfc = 1
    var_f = (
        dfa**2 * pcov[0, 0] +
        dfb**2 * pcov[1, 1] +
        dfc**2 * pcov[2, 2] +
        2 * dfa * dfb * pcov[0, 1] +
        2 * dfa * dfc * pcov[0, 2] +
        2 * dfb * dfc * pcov[1, 2]
    )
    sigma_f = np.sqrt(var_f)
    return sigma_f

def error_propagation_parab_inverse(y, y_err, a, b, c, pcov):
#error propagation for parabolic fit, but solving for phase instead of magnitude, using covariance matrix and partial derivatives#
    dfa = -(1/2) * np.sqrt((y-c) / (a**3))
    dfb = 1
    dfc = -(1/2) / np.sqrt(a*(y-c))
    dfy = (1/2) / np.sqrt(a*(y-c))
    var_f = (
        dfa**2 * pcov[0, 0] +
        dfb**2 * pcov[1, 1] +
        dfc**2 * pcov[2, 2] +
        dfy**2 * y_err**2 +
        2 * dfa * dfb * pcov[0, 1] +
        2 * dfa * dfc * pcov[0, 2] +
        2 * dfb * dfc * pcov[1, 2]
    )
    sigma_f = np.sqrt(var_f)
    return sigma_f

def get_teff(pri_flux, pri_flux_error, sec_flux, sec_flux_error, max_flux, max_flux_error):
#obtain the effective temperature ratio and error#
    teff = ((max_flux - sec_flux) / (max_flux - pri_flux)) ** (1/4)
    numerator = max_flux - sec_flux
    denominator = max_flux - pri_flux
    common_factor = (1/4) * (numerator / denominator)**(-3/4)
    ds_dt = common_factor * (sec_flux - pri_flux) / (denominator**2)
    ds_du = -common_factor / denominator
    ds_dv = -common_factor * numerator / (denominator**2)
    error = np.sqrt(  (ds_dt**2 * max_flux_error**2) + (ds_du**2 * pri_flux_error**2) + (ds_dv**2 * sec_flux_error**2)  )
    return teff, error

def get_ep_sin2i(max_flux, max_flux_error, min_flux, min_flux_error, pri_flux, pri_flux_error):
#obtain combined oblateness times squared-sine inclination and error#
    ep_sin2i = (max_flux - min_flux) / pri_flux
    dep_dmax = 1 / pri_flux
    dep_dmin = -1 / pri_flux
    dep_dpri = -(max_flux - min_flux) / (pri_flux)**2
    error = np.sqrt(  (dep_dmax**2 * max_flux_error**2) + (dep_dmin**2 * min_flux_error**2) + (dep_dpri**2 * pri_flux_error**2)  )
    return ep_sin2i, error


def get_ecc_peri(pri_a, pri_phase, pri_mag, pri_pcov, sec_a, sec_phase, sec_mag, sec_pcov, max_mag, max_mag_error):
#obtain eccentricity and periastron and their respective errors#
    z = symbols('z')
    pri_fit = pri_a * (z - pri_phase)**2 + pri_mag
    sec_fit = sec_a * (z - sec_phase)**2 + sec_mag
    
    pri_eq = Eq(pri_fit, max_mag)
    sec_eq = Eq(sec_fit, max_mag)
    
    #solve for when the first parabolic fit equals the maximum of the squared-sine fit
    pri_window = solve(pri_eq, z)
    pri_start = float(min(pri_window))
    pri_start_error = error_propagation_parab_inverse(max_mag, max_mag_error, pri_a, pri_phase, pri_mag, pri_pcov)
    pri_end = float(max(pri_window))
    pri_end_error = error_propagation_parab_inverse(max_mag, max_mag_error, pri_a, pri_phase, pri_mag, pri_pcov)
    
    #solve for when the second parabolic fit equals the maximum of the squared-sine fit
    sec_window = solve(sec_eq, z)
    sec_start = float(min(sec_window))
    sec_start_error = error_propagation_parab_inverse(max_mag, max_mag_error, sec_a, sec_phase, sec_mag, sec_pcov)
    sec_end = float(max(sec_window))
    sec_end_error = error_propagation_parab_inverse(max_mag, max_mag_error, sec_a, sec_phase, sec_mag, sec_pcov)
    
    #calculate the duration of each eclipse based on previous calculation
    pri_duration = pri_end - pri_start
    pri_duration_error = pri_start_error**2 + pri_end_error**2
    sec_duration = sec_end - sec_start
    sec_duration_error = sec_start_error**2 + sec_end_error**2
    
    #ensure primary eclipse occurs first
    if sec_phase < pri_phase:
        sec_phase += 1
    sec_phase -= pri_phase
    
    #eccentricity and periastron are calculated through esinw and ecosw equations, then uncoupled
    ecosw = (sec_phase - 0.5)
    ecosw_error = np.sqrt(sec_pcov[1,1]**2)
    
    esinw = (sec_duration - pri_duration) / (sec_duration + pri_duration)
    
    dsin_dsec = (2 * pri_duration) / ((sec_duration + pri_duration)**2)
    dsin_dpri = -(2 * sec_duration) / ((sec_duration + pri_duration)**2)
    esinw_error = np.sqrt(  (dsin_dsec**2 * sec_duration_error**2) + (dsin_dpri**2 * pri_duration_error**2)  )
    
    ecc = (esinw**2 + ecosw**2)**(1/2)
    de_dsin = esinw / np.sqrt(esinw**2 + ecosw**2)
    de_dcos = ecosw / np.sqrt(esinw**2 + ecosw**2)
    ecc_error = np.sqrt(  (de_dsin**2 * esinw_error**2) + (de_dcos**2 * ecosw_error**2)  )
    
    peri = math.atan(esinw / ecosw) * 180 / np.pi
    dp_dsin = (180 / np.pi) * (ecosw / (ecosw**2 + esinw**2))
    dp_dcos = -(180 / np.pi) * (esinw / (ecosw**2 + esinw**2))
    peri_error = np.sqrt(  (dp_dsin**2 * esinw_error**2) + (dp_dcos**2 * ecosw_error**2)  )
    
    #ensure periastron is 0-360 degrees
    if (esinw > 0 and ecosw < 0) or (esinw < 0 and ecosw < 0):
        peri += 180
    elif (esinw < 0 and ecosw > 0):
        peri += 360
    
    return ecc, ecc_error, peri, peri_error
    

def main(data):
    lightcurve, lc_mags_sort = exlightfile(data, command)
    
    # obtain parabolic fit parameters using a method. Note that the window is centered around the found minima rather than the fittedPhase number
    if method == "low_mags":
    #use the low_mags method#
        phase_min1, mag_min1, error_min1, a_min1, fittedPhase_min1, fittedMag_min1, pcov_min1 = get_first_min(lc_mags_sort, lightcurve)
        phase_min2, mag_min2, error_min2, a_min2, fittedPhase_min2, fittedMag_min2, pcov_min2 = get_sec_min(lightcurve, fittedPhase_min1, phase_min1, lc_mags_sort, a_min1)
    elif method == "move_avg":
    #use the move_avg method (default)#
        phase_min1, mag_min1, error_min1, params_min1, phase_min2, mag_min2, error_min2, params_min2, pcov_min1, pcov_min2 = get_mins_move_avg(lightcurve)
        a_min1, fittedPhase_min1, fittedMag_min1 = params_min1
        a_min2, fittedPhase_min2, fittedMag_min2 = params_min2    
    a_error1, fittedPhase_error1, fittedMag_error1 = np.sqrt(np.diag(pcov_min1))
    a_error2, fittedPhase_error2, fittedMag_error2 = np.sqrt(np.diag(pcov_min2))
       
    # this if statement is to distinguish the primary from the secondary minima based on fitted magnitude
    if fittedMag_min1 > fittedMag_min2:
        pri_min_phases = np.array(phase_min1)
        pri_min_mags = np.array(mag_min1)
        pri_min_errors = np.array(error_min1)
        pri_mag = fittedMag_min1
        pri_mag_error = fittedMag_error1
        pri_phase = fittedPhase_min1
        pri_phase_error = fittedPhase_error1
        pri_a = a_min1
        pri_a_error = a_error1
        pri_pcov = pcov_min1
        sec_min_phases = np.array(phase_min2)
        sec_min_mags = np.array(mag_min2)
        sec_min_errors = np.array(error_min2)
        sec_mag = fittedMag_min2
        sec_mag_error = fittedMag_error2
        sec_phase = fittedPhase_min2
        sec_phase_error = fittedPhase_error2
        sec_a = a_min2
        sec_a_error = a_error2
        sec_pcov = pcov_min2
    else:
        pri_min_phases = np.array(phase_min2)
        pri_min_mags = np.array(mag_min2)
        pri_min_errors = np.array(error_min2)
        pri_mag = fittedMag_min2
        pri_mag_error = fittedMag_error2
        pri_phase = fittedPhase_min2
        pri_phase_error = fittedPhase_error2
        pri_a = a_min2
        pri_a_error = a_error2
        pri_pcov = pcov_min2
        sec_min_phases = np.array(phase_min1)
        sec_min_mags = np.array(mag_min1)
        sec_min_errors = np.array(error_min1)
        sec_mag = fittedMag_min1
        sec_mag_error = fittedMag_error1
        sec_phase = fittedPhase_min1
        sec_phase_error = fittedPhase_error1
        sec_a = a_min1
        sec_a_error = a_error1
        sec_pcov = pcov_min1

    max_phases = []
    max_mags = []
    max_errors = []
    
    phases = np.array(lightcurve)[:,0]
    mags = np.array(lightcurve)[:,1]
    errors = np.array(lightcurve)[:,2]
        
    first_min = min(fittedPhase_min1,fittedPhase_min2)
    second_min = max(fittedPhase_min1,fittedPhase_min2)
    
    #obtain points in the maxima of the lightcurve
    for i in range(len(phases)):
        if ((phases[i] > first_min + window/2 and phases[i] < second_min - window/2)
            or (phases[i] > second_min + window/2 and phases[i] < first_min + 1 - window/2)):
            max_phases.append(phases[i])
            max_mags.append(mags[i])
            max_errors.append(errors[i])

    #set an initial amplitude for fitting the sine curve
    initial_mag_shift = np.mean([fittedPhase_min1, fittedPhase_min2]) - np.mean(max_mags)
    initial_amp = np.mean(max_mags)
    
    #obtain sine-fit parameters and covariance matrix (thus errors)
    params_sine, sine_pcov = fit_maximums(max_phases, max_mags, fittedPhase_min1, initial_mag_shift, initial_amp)
    sine_amp, sine_phase_shift, sine_mag_shift = params_sine
    sine_amp_error, sine_phase_error, sine_mag_error = np.sqrt(np.diag(sine_pcov))
    
    #for distinguishing pulsators from EW: split minima in half to check respective chi-squares to parabola
    pri_phase_l = np.array(pri_min_phases)[pri_min_phases < pri_phase]
    pri_mag_l = np.array(pri_min_mags)[pri_min_phases < pri_phase]
    pri_error_l = np.array(pri_min_errors)[pri_min_phases < pri_phase]
    
    pri_phase_r = np.array(pri_min_phases)[pri_min_phases >= pri_phase]
    pri_mag_r = np.array(pri_min_mags)[pri_min_phases >= pri_phase]
    pri_error_r = np.array(pri_min_errors)[pri_min_phases >= pri_phase]
    
    sec_phase_l = np.array(sec_min_phases)[sec_min_phases < sec_phase]
    sec_mag_l = np.array(sec_min_mags)[sec_min_phases < sec_phase]
    sec_error_l = np.array(sec_min_errors)[sec_min_phases < sec_phase]
    
    sec_phase_r = np.array(sec_min_phases)[sec_min_phases >= sec_phase]
    sec_mag_r = np.array(sec_min_mags)[sec_min_phases >= sec_phase]
    sec_error_r = np.array(sec_min_errors)[sec_min_phases >= sec_phase]

    '''
    x2_pp = calc_x2(pri_min_phases, pri_min_mags, pri_min_errors, pri_a, pri_phase, pri_mag)
    x2_ppl = calc_x2(pri_phase_l, pri_mag_l, pri_error_l, pri_a, pri_phase, pri_mag)
    x2_ppr = calc_x2(pri_phase_r, pri_mag_r, pri_error_r, pri_a, pri_phase, pri_mag)
    x2_ss = calc_x2(sec_min_phases, sec_min_mags, sec_min_errors, sec_a, sec_phase, sec_mag)
    x2_ssl = calc_x2(sec_phase_l, sec_mag_l, sec_error_l, sec_a, sec_phase, sec_mag)
    x2_ssr = calc_x2(sec_phase_r, sec_mag_r, sec_error_r, sec_a, sec_phase, sec_mag)
    x2_ps = calc_x2(sec_min_phases, sec_min_mags, sec_min_errors, pri_a, sec_phase, pri_mag)
    x2_sp = calc_x2(pri_min_phases, pri_min_mags, pri_min_errors, sec_a, pri_phase, sec_mag)
    x2_max = calc_max_x2(max_phases, max_mags, max_errors, sine_amp, sine_phase_shift, sine_mag_shift)
    '''

    # x2 function convention: observation mags, expected mags at fit, observation errors, expected errors at fit, degrees of freedom
    x2_pp = get_x2(np.array(pri_min_mags), objective_parab(pri_min_phases, pri_a, pri_phase, pri_mag), np.array(pri_min_errors), error_propagation_parab(pri_min_phases, pri_a, pri_phase, pri_mag, pri_pcov), len(pri_min_mags) - 3)
    x2_ppl = get_x2(np.array(pri_mag_l), objective_parab(pri_phase_l, pri_a, pri_phase, pri_mag), np.array(pri_error_l), error_propagation_parab(pri_phase_l, pri_a, pri_phase, pri_mag, pri_pcov), len(pri_mag_l) - 3)
    x2_ppr = get_x2(np.array(pri_mag_r), objective_parab(pri_phase_r, pri_a, pri_phase, pri_mag), np.array(pri_error_r), error_propagation_parab(pri_phase_r, pri_a, pri_phase, pri_mag, pri_pcov), len(pri_mag_r) - 3)
    x2_ss = get_x2(np.array(sec_min_mags), objective_parab(sec_min_phases, sec_a, sec_phase, sec_mag), np.array(sec_min_errors), error_propagation_parab(sec_min_phases, sec_a, sec_phase, sec_mag, sec_pcov), len(sec_min_mags) - 3)
    x2_ssl = get_x2(np.array(sec_mag_l), objective_parab(sec_phase_l, sec_a, sec_phase, sec_mag), np.array(sec_error_l), error_propagation_parab(sec_phase_l, sec_a, sec_phase, sec_mag, sec_pcov), len(sec_mag_l) - 3)
    x2_ssr = get_x2(np.array(sec_mag_r), objective_parab(sec_phase_r, sec_a, sec_phase, sec_mag), np.array(sec_error_r), error_propagation_parab(sec_phase_r, sec_a, sec_phase, sec_mag, sec_pcov), len(sec_mag_r) - 3)
    x2_ps = get_x2(np.array(sec_min_mags), objective_parab(sec_min_phases, pri_a, sec_phase, pri_mag), np.array(sec_min_errors), error_propagation_parab(sec_min_phases, pri_a, sec_phase, pri_mag, pri_pcov), len(sec_min_mags) - 3)
    x2_sp = get_x2(np.array(pri_min_mags), objective_parab(pri_min_phases, sec_a, pri_phase, sec_mag), np.array(pri_min_errors), error_propagation_parab(pri_min_phases, sec_a, pri_phase, sec_mag, sec_pcov), len(pri_min_mags) - 3)
    x2_max = get_x2(np.array(max_mags), objective_sine(max_phases, sine_amp, sine_phase_shift, sine_mag_shift), np.array(max_errors), error_propagation_sin(max_phases, sine_amp, sine_phase_shift, sine_mag_shift, sine_pcov), len(max_mags) - 3)

    
    x = np.linspace(0,2,400)

    #obtain the maximum value of the sine curve and its error
    sine_max_mag = min(objective_sine(x, sine_amp, sine_phase_shift, sine_mag_shift))
    sine_max_mag_fit_error = error_propagation_sin(sine_max_mag, sine_amp, sine_phase_shift, sine_mag_shift, sine_pcov)
    sine_max_flux = mag_to_flux(sine_max_mag)
    sine_max_fit_flux_error = mag_to_flux_error(sine_max_flux, sine_max_mag_fit_error)
    
    #flux conversions for maximum sine value
    sine_min_mag = max(objective_sine(x, sine_amp, sine_phase_shift, sine_mag_shift))
    sine_min_mag_fit_error = error_propagation_sin(sine_min_mag, sine_amp, sine_phase_shift, sine_mag_shift, sine_pcov)
    sine_min_flux = mag_to_flux(sine_min_mag)
    sine_min_fit_flux_error = mag_to_flux_error(sine_min_flux, sine_min_mag_fit_error)
    
    #more flux conversions for each eclipse at the parabola minimum
    pri_fit_mag_error = error_propagation_parab(pri_phase, pri_a, pri_phase, pri_mag, pri_pcov)
    pri_flux = mag_to_flux(pri_mag)
    pri_fit_flux_error = mag_to_flux_error(pri_flux, pri_fit_mag_error)
    sec_fit_mag_error = error_propagation_parab(sec_phase, sec_a, sec_phase, sec_mag, sec_pcov)
    sec_flux = mag_to_flux(sec_mag)
    sec_fit_flux_error = mag_to_flux_error(sec_flux, sec_fit_mag_error)
    
    #acquire physical parameters for binaries
    teff_ratio, teff_error = get_teff(pri_flux, pri_fit_flux_error, sec_flux, sec_fit_flux_error, sine_max_flux, sine_max_fit_flux_error)
    ep_sin2i, ep_sin2i_error = get_ep_sin2i(sine_max_flux, sine_max_fit_flux_error, sine_min_flux, sine_min_fit_flux_error, pri_flux, pri_fit_flux_error)
    ecc, ecc_error, peri, peri_error = get_ecc_peri(pri_a, pri_phase, pri_mag, pri_pcov, sec_a, sec_phase, sec_mag, sec_pcov, sine_max_mag, sine_max_mag_fit_error)

    #if csbt is ran directly through lcvis, return this dictionary with all parameters necessary for value display and plots
    if command == 'lcs':
        return_lcvis = {"Teff" : teff_ratio, "ep_sin2i" : ep_sin2i, "Eccentricity" : ecc, "Periastron" : peri,
                        "Teff_Error" : teff_error, "ep_sin2i_Error" : ep_sin2i_error, "Eccentricity_Error" : ecc_error, "Periastron_Error" : peri_error,
                        "Max_FitPhase" : sine_phase_shift, "Max_FitMag" : sine_mag_shift, "Max_FitAmp" : sine_amp, 
                        "Max_FitPhase_Error" : sine_phase_error, "Max_FitMag_Error" : sine_mag_error, "Max_FitAmp_Error" : sine_amp_error,
                        "Max_X2" : x2_max, "Max_Phase" : max_phases, "Max_Mag" : max_mags,
                        "Pri_Min_Phases" : pri_min_phases, "Pri_Min_Mags" : pri_min_mags, "Pri_Min_Errors" : pri_min_errors,
                        "Sec_Min_Phases" : sec_min_phases, "Sec_Min_Mags" : sec_min_mags, "Sec_Min_Errors" : sec_min_errors,
                        "Pri_A" : pri_a, "Pri_Phase" : pri_phase, "Pri_Mag" : pri_mag,
                        "Pri_A_Error" : pri_a_error, "Pri_Phase_Error" : pri_phase_error, "Pri_Mag_Error" : pri_mag_error,
                        "Sec_A" : sec_a, "Sec_Phase" : sec_phase, "Sec_Mag" : sec_mag,
                        "Sec_A_Error" : sec_a_error, "Sec_Phase_Error" : sec_phase_error, "Sec_Mag_Error" : sec_mag_error,
                        "FitP,MinP" : x2_pp, "FitP,MinPL" : x2_ppl, "FitP,MinPR" : x2_ppr,
                        "FitS,MinS" : x2_ss, "FitS,MinSL" : x2_ssl, "FitS,MinSR" : x2_ssr,
                        "FitP,MinS" : x2_ps, "FitS,MinP" : x2_sp}
        
        sys.stdout.buffer.write(pickle.dumps(return_lcvis))
        
    #show the fits (True by default)#
    if show_plot == 'true':
        matplotlib.rcParams["font.size"] = 8
        fig, ax = plt.subplots()
        x = np.linspace(0,2,400)
        plt.subplots_adjust(left=0.05,right=0.7,bottom=0.1,top=0.92)
        plt.errorbar(phases, mags, yerr = errors, label = 'Phased Data', color = 'k', fmt=".", markersize=12, elinewidth = 2, zorder = 1)
        plt.scatter(pri_min_phases, pri_min_mags, label = 'Primary', color = 'lightcoral', s=6, marker = 'o', zorder = 2)
        plt.scatter(sec_min_phases, sec_min_mags, label = 'Secondary', color = 'dodgerblue', s=6, marker = 'o', zorder = 2)
        plt.scatter(max_phases, max_mags, label = 'Maximum', color = 'mediumorchid', marker='.', zorder = 2)
        plt.plot(x, objective_parab(x, pri_a, pri_phase, pri_mag), label = 'Primary Fit', color = 'r', linewidth = 3, zorder = 4)
        plt.plot(x, objective_parab(x, pri_a, sec_phase, pri_mag), label = 'Primary Fit on Secondary', color = 'r', linestyle = '--', linewidth = 3, zorder = 5)
        plt.plot(x, objective_parab(x, sec_a, sec_phase, sec_mag), label = 'Secondary Fit', color = 'c', linewidth = 3, zorder = 4)
        plt.plot(x, objective_parab(x, sec_a, pri_phase, sec_mag), label = 'Secondary Fit on Primary', color = 'c', linestyle = '--', linewidth = 3, zorder = 5)
        plt.plot(x, objective_sine(x, sine_amp, sine_phase_shift, sine_mag_shift), label = 'Maxima Fit', color = 'plum', linewidth = 3, zorder = 3)
        plt.ylim(max(mags)+0.2,min(mags)-0.2)
        #left_bound = min([fittedPhase_min1, fittedPhase_min2]) - window*2
        #right_bound = min([fittedPhase_min1, fittedPhase_min2]) + 1
        #plt.xlim(left_bound, right_bound)
        
        fig.text(0.72,0.9,f'Primary Minimum Fit: y = {pri_a:.6f} * (x - {pri_phase:.6f})² + {pri_mag:.6f}'
                 + f'\n Width Error: ±{pri_a_error:.6f}'
                 + f'\n Phase Shift Error: ±{pri_phase_error:.6f}'
                 + f'\n Mag Shift Error: ±{pri_mag_error:.6f}', bbox=dict(boxstyle="round,pad=1", facecolor="r", edgecolor="black", linewidth=2))
        fig.text(0.72,0.8,f'Primary Min Data, Primary Min Fit χ² = {x2_pp:.6f} \n Left Side χ² = {x2_ppl:.6f} \n Right Side χ² = {x2_ppr:.6f}', bbox=dict(boxstyle="round,pad=1", facecolor="r", edgecolor="lightcoral", linewidth=2))
        fig.text(0.72,0.75,f'Secondary Min Data, Primary Min Fit χ² = {x2_ps:.6f}', bbox=dict(boxstyle="round,pad=1", facecolor="r", edgecolor="dodgerblue", linewidth=2))
        
        fig.text(0.72,0.65,f'Secondary Minimum Fit: y = {sec_a:.6f} * (x - {sec_phase:.6f})² + {sec_mag:.6f}'
                 + f'\n Width Error: ±{sec_a_error:.6f}'
                 + f'\n Phase Shift Error: ±{sec_phase_error:.6f}'
                 + f'\n Mag Shift Error: ±{sec_mag_error:.6f}', bbox=dict(boxstyle="round,pad=1", facecolor="c", edgecolor="black", linewidth=2))
        fig.text(0.72,0.55,f'Secondary Min Data, Secondary Min Fit χ² = {x2_ss:.6f} \n Left Side χ² = {x2_ssl:.6f} \n Right Side χ² = {x2_ssr:.6f}', bbox=dict(boxstyle="round,pad=1", facecolor="c", edgecolor="dodgerblue", linewidth=2))
        fig.text(0.72,0.5,f'Primary Min Data, Secondary Min Fit χ² = {x2_sp:.6f}', bbox=dict(boxstyle="round,pad=1", facecolor="c", edgecolor="lightcoral", linewidth=2))
        
        fig.text(0.72,0.35,f'Maxima Fit: y = {sine_amp:.6f} * sin(2π * (x - {sine_phase_shift:.6f}) + {sine_mag_shift:.6f}'
                 + f'\n Amp Error: {sine_amp_error:.6f}'
                 + f'\n Phase Shift Error: {sine_phase_error:.6f}'
                 + f'\n Mag Shift Error: {sine_mag_error:.6f}', bbox=dict(boxstyle="round,pad=1", facecolor="plum", edgecolor="black", linewidth=2))
        fig.text(0.72,0.3,f'Maxima Fit χ² = {x2_max:.6f}', bbox=dict(boxstyle="round,pad=1", facecolor="plum", edgecolor="mediumorchid", linewidth=2))
        
        fig.text(0.72,0.15,f'Effective Temperature Ratio = {teff_ratio:.6f} ±{teff_error:.6f}' + '\n'
                         + f'Oblateness * sin²(Inclination) = {ep_sin2i:.6f} ±{ep_sin2i_error:.6f}' + '\n'
                         + f'Eccentricity = {ecc:.6f} ±{ecc_error:.6f}' + '\n'
                         + f'Periastron = {peri:.6f} ±{peri_error:.6f}', bbox=dict(boxstyle="round,pad=1", facecolor="silver", edgecolor="black", linewidth=2))
    
        
        plt.legend(loc='best')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        plt.title(f'{data} Minima Fits, Method = {method}, Window = {window}, Numpoints = {numpoints}')
        plt.get_current_fig_manager().set_window_title('Chi-Square Binary Test Output')
        
        plt.show()
    
    return

parser = args.ArgumentParser()
parser.add_argument("indata", help = "Object's data file")
parser.add_argument("--numpoints", "-n", help = "Number of points to take when finding lowest magnitudes", default = 20)
parser.add_argument("--window", "-w", help = "Width of minima", default = 0.1)
parser.add_argument("--command", "-c", help = "Location where script was called", default = "lcf")
parser.add_argument("--method", "-m", help = "Method to find minima", default = "move_avg")
parser.add_argument("--show_plot", "-p", help = "Show fit plot", default = "true")
args = parser.parse_args()
data = args.indata
command = args.command
window = float(args.window)
numpoints = int(args.numpoints)
method = args.method
show_plot = args.show_plot
object_name = os.path.basename(data)

run = main(data)
