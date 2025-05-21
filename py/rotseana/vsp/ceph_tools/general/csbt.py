# fit a straight line to the economic data
from scipy.optimize import curve_fit
from operator import itemgetter
from scipy.optimize import fsolve
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
        lightcurve = np.genfromtxt(data, dtype = None, delimiter = None, skip_header = 2, names = ["phase", "mag", "error"])
    elif command == "lcs":
        import ast
        lightcurve = ast.literal_eval(data)
    elif command == "lcf":
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

'''
def getwindowMin(Mag,Phase,Error,PhaseMedian,num):
#gets the points around the first parabola for each Phase Mag and Error using the difference between the each Phase point and the Median#              
    k=range(num)
    abs_dif=[]
    for i in k:
        abs_dif.append(abs(Phase[i] - PhaseMedian))
    x=range(num)
    tunedPhase=[]
    tunedMag =[]
    tunedError=[]
    for i in x:
        if abs_dif[i] <= 0.2:
            tunedPhase.append(Phase[i])
            tunedMag.append(Mag[i])
            tunedError.append(Error[i])
    return tunedPhase,tunedMag,tunedError

def difference(Phase,PhaseMedian,num):
#does the same thing as getWindow Min#
    j=range(num)
    removei=[]
    for i in j:
        diff = abs(Phase[i] - PhaseMedian)
        if diff > 0.1:
            removei.append(i)
    return removei
'''


def difference2(phase, phase_median, num):
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
    return a * (np.sin(2 * np.pi * (x - b)))**2 + c

def fit_minimum(phase, mag, a, b, c):
#finds the fitminimas a b c#
    p0 = [None] * 3
    p0[0] = a
    p0[1] = b
    p0[2] = c
    params, _ = curve_fit(objective_parab, phase, mag, p0)
    return params

def fit_maximums(phase, mag, a, b, c):
    p0 = [None] * 3
    p0[0] = a
    p0[1] = b
    p0[2] = c
    params, _ = curve_fit(objective_sine, phase, mag, p0)
    return params

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
    mu = []
    calc_sum = []
    for i in range(len(phase_min)):
        mu.append(objective_parab(phase_min[i], a, b, c))
        calc_sum.append(( (mag_min[i] - mu[i]) / error_min[i] )**2)
    total_calc = np.sum(calc_sum)    
    x2 = total_calc / (len(phase_min) - 3)
    return x2

def calc_max_x2(phase_max, mag_max, error_max, a, b, c):
    mu = []
    calc_sum = []
    for i in range(len(phase_max)):
        mu.append(objective_sine(phase_max[i], a, b, c))
        calc_sum.append(( (mag_max[i] - mu[i]) / error_max[i] )**2)
    total_calc = np.sum(calc_sum)    
    max_x2 = total_calc / (len(phase_max) - 3)
    return max_x2




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
    
    #savei1 = difference2(phase1, phase1_median, len(phase1))
    phase1_cleaned = []
    mag1_cleaned = []
    error1_cleaned = []
    
    for obs in lightcurve:
        if abs(obs[0] - phase1_median) < window / 2:
             phase1_cleaned.append(obs[0])
             mag1_cleaned.append(obs[1])
             error1_cleaned.append(obs[2])
    a_min1 = -50.00
    
    '''
    for i in savei1:
         phase1_cleaned.append(phase1[i])
         mag1_cleaned.append(mag1[i])
         error1_cleaned.append(error1[i])
    a_min1 = -50.00
    '''
    
    #recalculate based on window instead of the ones saved???
    params_min1 = fit_minimum(phase1_cleaned, mag1_cleaned, a_min1, phase1_median, mag1_median)
    a_min1, b_min1, c_min1 = params_min1
    fittedPhase_min1 = b_min1
    fittedMag_min1 = c_min1
    
    #not useful for calculations, just highlights arbitrary window
    phase_min1, mag_min1, error_min1 = get_window(lightcurve, fittedPhase_min1, window / 2)
    return phase_min1, mag_min1, error_min1, a_min1, fittedPhase_min1, fittedMag_min1

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
    #savei = difference2(phase2, guessPhase, len(phase2))
    phase2_cleaned = []
    mag2_cleaned = []
    error2_cleaned = []
    
    for obs in lightcurve_no_min1:
        if abs(obs[0] - guessPhase) < window / 2:
             phase2_cleaned.append(obs[0])
             mag2_cleaned.append(obs[1])
             error2_cleaned.append(obs[2])
    
    '''
    for i in savei:
        phase2_cleaned.append(phase2[i])
        mag2_cleaned.append(mag2[i])
        error2_cleaned.append(error2[i])
    mag2_median = st.median(mag2_cleaned)
    phase2_median = st.median(phase2_cleaned)
    '''
    
    #recalculate based on window instead of the ones saved???
    params_min2 = fit_minimum(phase2_cleaned, mag2_cleaned, a, phase2_median, mag2_median)
    a_min2, b_min2, c_min2 = params_min2
    
    #not useful for calculations, just highlights arbitrary window
    phase_min2, mag_min2, error_min2 = get_window(lightcurve, b_min2, window / 2)
    return phase_min2, mag_min2, error_min2, a_min2, b_min2, c_min2

def moving_average(x, w):
    """calculate moving average with window size w"""
    return np.convolve(x, np.ones(w), 'valid') / w

def find_mins_move_avg(lightcurve):
    from scipy.signal import find_peaks
    phase = []
    mag = []
    for obs in lightcurve:
        phase.append(obs[0])
        mag.append(obs[1])
    phase = np.array(phase)
    mag = np.array(mag)
    
    n=25
    phase_f = moving_average(phase, n)
    mag_f = moving_average(mag, n)
    #to show in same plot
    phase_f_ext = np.hstack([np.ones(n//2)*phase_f[0],phase_f])
    mag_f_ext= np.hstack([np.ones(n//2)*mag_f[0],mag_f])
    
    mins, _ = find_peaks(mag_f_ext, distance=int(len(lightcurve)/10), height = np.average(mag))
    
    phase_mins = phase_f_ext[mins].tolist()
    mag_mins = mag_f_ext[mins].tolist()
    
    return phase_mins, mag_mins

def get_mins_move_avg(lightcurve):
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
             
    params_min1 = fit_minimum(phase1_cleaned, mag1_cleaned, a, min1_phase, min1_mag)
    
    params_min2 = fit_minimum(phase2_cleaned, mag2_cleaned, a, min2_phase, min2_mag)
    
    return phase1_cleaned, mag1_cleaned, error1_cleaned, params_min1, phase2_cleaned, mag2_cleaned, error2_cleaned, params_min2 
    

def main(data):
    lightcurve, lc_mags_sort = exlightfile(data, command)
    if method == "low_mags":
        phase_min1, mag_min1, error_min1, a_min1, fittedPhase_min1, fittedMag_min1 = get_first_min(lc_mags_sort, lightcurve)
        phase_min2, mag_min2, error_min2, a_min2, fittedPhase_min2, fittedMag_min2 = get_sec_min(lightcurve, fittedPhase_min1, phase_min1, lc_mags_sort, a_min1)
    elif method == "move_avg":
        phase_min1, mag_min1, error_min1, params_min1, phase_min2, mag_min2, error_min2, params_min2 = get_mins_move_avg(lightcurve)
        a_min1, fittedPhase_min1, fittedMag_min1 = params_min1
        a_min2, fittedPhase_min2, fittedMag_min2 = params_min2
        
    if fittedMag_min1 > fittedMag_min2:
        pri_min_phases = phase_min1
        pri_min_mags = mag_min1
        pri_min_errors = error_min1
        pri_mag = fittedMag_min1
        pri_phase = fittedPhase_min1
        pri_a = a_min1
        sec_min_phases = phase_min2
        sec_min_mags = mag_min2
        sec_min_errors = error_min2
        sec_mag = fittedMag_min2
        sec_phase = fittedPhase_min2
        sec_a = a_min2
    else:
        pri_min_phases = phase_min2
        pri_min_mags = mag_min2
        pri_min_errors = error_min2
        pri_mag = fittedMag_min2
        pri_phase = fittedPhase_min2
        pri_a = a_min2
        sec_min_phases = phase_min1
        sec_min_mags = mag_min1
        sec_min_errors = error_min1
        sec_mag = fittedMag_min1
        sec_phase = fittedPhase_min1
        sec_a = a_min1
    
    phases = []
    mags = []
    errors = []
    max_phases = []
    max_mags = []
    max_errors = []
    for i in range(len(lightcurve)):
        phases.append(lightcurve[i][0])
        mags.append(lightcurve[i][1])
        errors.append(lightcurve[i][2])
        
    
    first_min = min(fittedPhase_min1,fittedPhase_min2)
    second_min = max(fittedPhase_min1,fittedPhase_min2)
    
    for i in range(len(phases)):
        if ((phases[i] > first_min + window/2 and phases[i] < second_min - window/2)
            or (phases[i] > second_min + window/2 and phases[i] < first_min + 1 - window/2)):
            max_phases.append(phases[i])
            max_mags.append(mags[i])
            max_errors.append(errors[i])

    initial_mag_shift = np.mean([fittedPhase_min1, fittedPhase_min2]) - np.mean(max_mags)
    initial_amp = np.mean(max_mags)
            
    params_max = fit_maximums(max_phases, max_mags, fittedPhase_min1, initial_mag_shift, initial_amp)
    sine_amp, sine_phase_shift, sine_mag_shift = params_max
    
    pri_phase_l = []
    pri_mag_l = []
    pri_error_l = []
    
    pri_phase_r = []
    pri_mag_r = []
    pri_error_r = []
    
    sec_phase_l = []
    sec_mag_l = []
    sec_error_l = []
    
    sec_phase_r = []
    sec_mag_r = []
    sec_error_r = []
    
    for i in range(len(pri_min_phases)):
        if pri_min_phases[i] < pri_phase:
            pri_phase_l.append(pri_min_phases[i])
            pri_mag_l.append(pri_min_mags[i])
            pri_error_l.append(pri_min_errors[i])
        else:
            pri_phase_r.append(pri_min_phases[i])
            pri_mag_r.append(pri_min_mags[i])
            pri_error_r.append(pri_min_errors[i])
    
    for i in range(len(sec_min_phases)):
        if sec_min_phases[i] < sec_phase:
            sec_phase_l.append(sec_min_phases[i])
            sec_mag_l.append(sec_min_mags[i])
            sec_error_l.append(sec_min_errors[i])
        else:
            sec_phase_r.append(sec_min_phases[i])
            sec_mag_r.append(sec_min_mags[i])
            sec_error_r.append(sec_min_errors[i])
    
    
    x2_pp = calc_x2(pri_min_phases, pri_min_mags, pri_min_errors, pri_a, pri_phase, pri_mag)
    x2_ppl = calc_x2(pri_phase_l, pri_mag_l, pri_error_l, pri_a, pri_phase, pri_mag)
    x2_ppr = calc_x2(pri_phase_r, pri_mag_r, pri_error_r, pri_a, pri_phase, pri_mag)
    x2_ss = calc_x2(sec_min_phases, sec_min_mags, sec_min_errors, sec_a, sec_phase, sec_mag)
    x2_ssl = calc_x2(sec_phase_l, sec_mag_l, sec_error_l, sec_a, sec_phase, sec_mag)
    x2_ssr =calc_x2(sec_phase_r, sec_mag_r, sec_error_r, sec_a, sec_phase, sec_mag)
    x2_ps = calc_x2(sec_min_phases, sec_min_mags, sec_min_errors, pri_a, sec_phase, pri_mag)
    x2_sp = calc_x2(pri_min_phases, pri_min_mags, pri_min_errors, sec_a, pri_phase, sec_mag)
    x2_max = calc_max_x2(max_phases, max_mags, max_errors, sine_amp, sine_phase_shift, sine_mag_shift)
    
    x = np.linspace(0,2,400)

    sine_max_mag = min(objective_sine(x, sine_amp, sine_phase_shift, sine_mag_shift))
    sine_max_flux = 10**((sine_max_mag + 48.60)/(-2.5))
    sine_min_mag = max(objective_sine(x, sine_amp, sine_phase_shift, sine_mag_shift))
    sine_min_flux = 10**((sine_min_mag + 48.60)/(-2.5))
    pri_flux = 10**((pri_mag + 48.60)/(-2.5))
    sec_flux = 10**((sec_mag + 48.60)/(-2.5))
    
    teff_ratio = ((sine_max_flux - sec_flux) / (sine_max_flux - pri_flux)) ** (1/4)

    ep_sin2i = (sine_max_flux - sine_min_flux) / pri_flux  
    
    z = symbols('z')
    min1_fit = a_min1 * (z - fittedPhase_min1)**2 + fittedMag_min1
    min2_fit = a_min2 * (z - fittedPhase_min2)**2 + fittedMag_min2
    
    min1_eq = Eq(min1_fit, sine_max_mag)
    min2_eq = Eq(min2_fit, sine_max_mag)
    
    min1_window = solve(min1_eq, z)
    min2_window = solve(min2_eq, z)
    
    min1_duration = max(min1_window) - min(min1_window)
    min2_duration = max(min2_window) - min(min2_window)
    
    esinw = (min2_duration - min1_duration) / (min1_duration + min2_duration)
    
    min1_loc = fittedPhase_min1
    min2_loc = fittedPhase_min2
    if min2_loc < min1_loc:
        min2_loc += 1
    min2_loc -= min1_loc
    
    ecosw = (min2_loc - 0.5)
    
    ecc = (esinw**2 + ecosw**2)**(1/2)
    peri = math.atan(esinw / ecosw) * 180 / np.pi
    
    if (esinw > 0 and ecosw < 0) or (esinw < 0 and ecosw < 0):
        peri += 180
    elif (esinw < 0 and ecosw > 0):
        peri += 360

    if command == 'lcs':
        return_lcvis = {"Teff" : teff_ratio, "ep_sin2i" : ep_sin2i, "Eccentricity" : ecc, "Periastron" : peri,
                        "Max_FitPhase" : sine_phase_shift, "Max_FitMag" : sine_mag_shift, "Max_FitAmp" : sine_amp, 
                        "Max_X2" : x2_max, "Max_Phase" : max_phases, "Max_Mag" : max_mags,
                        "Pri_Min_Phases" : pri_min_phases, "Pri_Min_Mags" : pri_min_mags, "Pri_Min_Errors" : pri_min_errors,
                        "Sec_Min_Phases" : sec_min_phases, "Sec_Min_Mags" : sec_min_mags, "Sec_Min_Errors" : sec_min_errors,
                        "Pri_A" : pri_a, "Pri_Phase" : pri_phase, "Pri_Mag" : pri_mag,
                        "Sec_A" : sec_a, "Sec_Phase" : sec_phase, "Sec_Mag" : sec_mag,
                        "FitP,MinP" : x2_pp, "FitP,MinPL" : x2_ppl, "FitP,MinPR" : x2_ppr,
                        "FitS,MinS" : x2_ss, "FitS,MinSL" : x2_ssl, "FitS,MinSR" : x2_ssr,
                        "FitP,MinS" : x2_ps, "FitS,MinP" : x2_sp}
        
        sys.stdout.buffer.write(pickle.dumps(return_lcvis))
        

    if show_plot == 'true':
        matplotlib.rcParams["font.size"] = 7
        fig, ax = plt.subplots()
        x = np.linspace(0,2,400)
        plt.subplots_adjust(left=0.05,right=0.7,bottom=0.1,top=0.92)
        plt.errorbar(phases, mags, yerr = errors, label = 'Phased Data', color = 'k', fmt='o', elinewidth = 1, zorder = 1)
        plt.errorbar(pri_min_phases, pri_min_mags, yerr = pri_min_errors, label = 'Primary', color = 'red', fmt='o', elinewidth = 1, zorder = 2)
        plt.errorbar(sec_min_phases, sec_min_mags, yerr = sec_min_errors, label = 'Secondary', color = 'blue', fmt='o', elinewidth = 1, zorder = 2)
        plt.plot(x, objective_parab(x, pri_a, pri_phase, pri_mag), label = 'Primary Fit', color = 'y', linewidth = 3, zorder = 4)
        plt.plot(x, objective_parab(x, pri_a, sec_phase, pri_mag), label = 'Primary Fit on Secondary', color = 'y', linestyle = '--', linewidth = 3, zorder = 5)
        plt.plot(x, objective_parab(x, sec_a, sec_phase, sec_mag), label = 'Secondary Fit', color = 'c', linewidth = 3, zorder = 4)
        plt.plot(x, objective_parab(x, sec_a, pri_phase, sec_mag), label = 'Secondary Fit on Primary', color = 'c', linestyle = '--', linewidth = 3, zorder = 5)
        plt.plot(x, objective_sine(x, sine_amp, sine_phase_shift, sine_mag_shift), label = 'Maxima Fit', color = 'tab:orange', linewidth = 3, zorder = 3)
        plt.ylim(max(mags)+0.2,min(mags)-0.2)
        left_bound = min([fittedPhase_min1, fittedPhase_min2]) - window*2
        right_bound = min([fittedPhase_min1, fittedPhase_min2]) + 1
        plt.xlim(left_bound, right_bound)
        
        fig.text(0.72,0.9,f'Primary Minimum Fit: y = {round(pri_a,6)} * (x - {round(pri_phase,6)})² + {round(pri_mag,6)}', backgroundcolor='y')
        fig.text(0.72,0.8,f'Primary Min Data, Primary Min Fit χ² = {round(x2_pp,6)} \n Left Side χ² = {round(x2_ppl,6)} \n Right Side χ² = {round(x2_ppr,6)}', color='r', backgroundcolor='y')
        fig.text(0.72,0.75,f'Secondary Min Data, Primary Min Fit χ² = {round(x2_ps,6)}', color='b', backgroundcolor='y')
        
        fig.text(0.72,0.65,f'Secondary Minimum Fit: y = {round(sec_a,6)} * (x - {round(sec_phase,6)})² + {round(sec_mag,6)}', backgroundcolor='c')
        fig.text(0.72,0.55,f'Secondary Min Data, Secondary Min Fit χ² = {round(x2_ss,6)} \n Left Side χ² = {round(x2_ssl,6)} \n Right Side χ² = {round(x2_ssr,6)}', color='b', backgroundcolor='c')
        fig.text(0.72,0.5,f'Primary Min Data, Secondary Min Fit χ² = {round(x2_sp,6)}', color='r', backgroundcolor='c')
        
        fig.text(0.72,0.35,f'Maxima Fit: y = {round(sine_amp,6)} * sin(2π * (x - {round(sine_phase_shift,6)}) + {round(sine_mag_shift,6)}', backgroundcolor='tab:orange')
        fig.text(0.72,0.3,f'Maxima Fit χ² = {round(x2_max,6)}', backgroundcolor='tab:orange')
        
        fig.text(0.72,0.15,f'Effective Temperature Ratio = {round(teff_ratio,6)}' + '\n'
                         + f'Oblateness * sin²(Inclination) = {round(ep_sin2i,6)}' + '\n'
                         + f'Eccentricity = {round(ecc,6)}' + '\n'
                         + f'Periastron = {round(peri,6)}', backgroundcolor='tab:gray')
        
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

# port main->end to lcvis 
# get data file from Aiden
