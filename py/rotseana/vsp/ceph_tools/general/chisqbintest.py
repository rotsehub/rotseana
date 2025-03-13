# fit a straight line to the economic data
from scipy.optimize import curve_fit
from operator import itemgetter
import numpy as np 
import statistics as st


# define methods
'''
def collect(filename):
#### reads in file that is given returns it to use in the code####
    in_file = open(filename,'r')
    for a_line in in_file:
        print(a_line)
    return in_file
'''

def collect_table(data, command):
###generates the lightcurve ndarray###
    if command == "sp":
        lightcurve = np.genfromtxt(data, dtype = None, delimiter = None, skip_header = 2, names = ["phase", "mag", "error"])
    elif command == "lcs":
        lightcurve = list(data)
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
    

def objective(x, a, b, c):
#creates the parbola for each Minima#
    return a*(x-b)**2 + c

def fit_minimum(phase, mag, a, b, c):
#finds the fitminimas a b c#
    p0 = [None] * 3
    p0[0] = a
    p0[1] = b
    p0[2] = c
    params, _ = curve_fit(objective, phase, mag, p0)
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
        mu.append(objective(phase_min[i], a, b, c))
        calc_sum.append(( (mag_min[i] - mu[i]) / error_min[i] )**2)
    total_calc = np.sum(calc_sum)    
    x2 = total_calc / (len(phase_min) - 3)
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
        
    BOLD_UNDERLINE = '\033[1;4;30;107m'
    BOLD = '\033[1;4m'
    RED = '\033[41;1m'
    BLUE = '\033[44;1m'
    YELLOW = '\033[43;1m'
    CYAN = '\033[46;1m'
    RESET = '\033[0m'
    
    print()
    print(f'{BOLD_UNDERLINE}Method: {method}; Window: {window}; Method Sample: {numpoints} Points{RESET}')
    print()
    print(f'{YELLOW}First Minimum Fit{RESET}: y = {round(a_min1,6)} * (x-{round(fittedPhase_min1,6)})^2 + {round(fittedMag_min1,6)}')
    print(f'{CYAN}Second Minimum Fit{RESET}: y = {round(a_min2,6)} * (x-{round(fittedPhase_min2,6)})^2 + {round(fittedMag_min2,6)}')
    print()
    
    
    phase_min1_left = []
    mag_min1_left = []
    error_min1_left = []
    
    phase_min1_right = []
    mag_min1_right = []
    error_min1_right = []
    
    phase_min2_left = []
    mag_min2_left = []
    error_min2_left = []
    
    phase_min2_right = []
    mag_min2_right = []
    error_min2_right = []
    
    for i in range(len(phase_min1)):
        if phase_min1[i] < fittedPhase_min1:
            phase_min1_left.append(phase_min1[i])
            mag_min1_left.append(mag_min1[i])
            error_min1_left.append(error_min1[i])
        else:
            phase_min1_right.append(phase_min1[i])
            mag_min1_right.append(mag_min1[i])
            error_min1_right.append(error_min1[i])
    
    for i in range(len(phase_min2)):
        if phase_min2[i] < fittedPhase_min2:
            phase_min2_left.append(phase_min2[i])
            mag_min2_left.append(mag_min2[i])
            error_min2_left.append(error_min2[i])
        else:
            phase_min2_right.append(phase_min2[i])
            mag_min2_right.append(mag_min2[i])
            error_min2_right.append(error_min2[i])
    
    
    x2_11 = calc_x2(phase_min1, mag_min1, error_min1, a_min1, fittedPhase_min1, fittedMag_min1)
    x2_11l = calc_x2(phase_min1_left, mag_min1_left, error_min1_left, a_min1, fittedPhase_min1, fittedMag_min1)
    x2_11r = calc_x2(phase_min1_right, mag_min1_right, error_min1_right, a_min1, fittedPhase_min1, fittedMag_min1)
    print(f'{RED}First Min Data{RESET}, {YELLOW}First Min Fit{RESET} x2 = {BOLD}{round(x2_11,6)}{RESET};   Left Side x2 = {BOLD}{round(x2_11l,6)}{RESET};   Right Side x2 = {BOLD}{round(x2_11r,6)}{RESET}')
    x2_22 = calc_x2(phase_min2, mag_min2, error_min2, a_min2, fittedPhase_min2, fittedMag_min2)
    x2_22l = calc_x2(phase_min2_left, mag_min2_left, error_min2_left, a_min2, fittedPhase_min2, fittedMag_min2)
    x2_22r =calc_x2(phase_min2_right, mag_min2_right, error_min2_right, a_min2, fittedPhase_min2, fittedMag_min2)
    print(f'{BLUE}Second Min Data{RESET}, {CYAN}Second Min Fit{RESET} x2 = {BOLD}{round(x2_22,6)}{RESET};   Left Side x2 = {BOLD}{round(x2_22l,6)}{RESET};   Right Side x2 = {BOLD}{round(x2_22r,6)}{RESET}')
    print()
    x2_21 = calc_x2(phase_min1, mag_min1, error_min1, a_min2, fittedPhase_min1, fittedMag_min2)
    print(f'{RED}First Min Data{RESET}, {CYAN}Second Min Fit{RESET} x2 = {BOLD}{round(x2_21,6)}{RESET}')
    x2_12 = calc_x2(phase_min2, mag_min2, error_min2, a_min1, fittedPhase_min2, fittedMag_min1)
    print(f'{BLUE}Second Min Data{RESET}, {YELLOW}First Min Fit{RESET} x2 = {BOLD}{round(x2_12,6)}{RESET}')
    print()

    if show_plot == 'true':
        phases = []
        mags = []
        errors = []
        x = np.linspace(0,2,400)
        for i in range(len(lightcurve)):
            phases.append(lightcurve[i][0])
            mags.append(lightcurve[i][1])
            errors.append(lightcurve[i][2])
        plt.errorbar(phases, mags, yerr = errors, label = 'Phased Data', color = 'k', fmt='o', elinewidth = 1, zorder = 1)
        plt.errorbar(phase_min1, mag_min1, yerr = error_min1, label = 'Min 1', color = 'red', fmt='o', elinewidth = 1, zorder = 2)
        plt.errorbar(phase_min2, mag_min2, yerr = error_min2, label = 'Min 2', color = 'blue', fmt='o', elinewidth = 1, zorder = 2)
        plt.plot(x, objective(x, a_min1, fittedPhase_min1, fittedMag_min1), label = 'Min 1 Fit', color = 'y', linewidth = 3, zorder = 3)
        plt.plot(x, objective(x, a_min1, fittedPhase_min2, fittedMag_min1), label = 'Min 1 Fit on Min 2', color = 'y', linestyle = '--', linewidth = 3, zorder = 4)
        plt.plot(x, objective(x, a_min2, fittedPhase_min2, fittedMag_min2), label = 'Min 2 Fit', color = 'c', linewidth = 3, zorder = 3)
        plt.plot(x, objective(x, a_min2, fittedPhase_min1, fittedMag_min2), label = 'Min 2 Fit on Min 1', color = 'c', linestyle = '--', linewidth = 3, zorder = 4)        
        plt.ylim(max(mags)+0.2,min(mags)-0.2)
        left_bound = min([fittedPhase_min1, fittedPhase_min2]) - window
        right_bound = max([fittedPhase_min1, fittedPhase_min2]) + window
        plt.xlim(left_bound, right_bound)
        plt.legend()
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        plt.title(f'Minima Fits: {method}')
        plt.show()
    
    return

import matplotlib.pyplot as plt
import argparse as args
parser = args.ArgumentParser()
parser.add_argument("indata", help = "Object's data file")
parser.add_argument("--numpoints", "-n", help = "Number of points to take when finding lowest magnitudes", default = 20)
parser.add_argument("--window", "-w", help = "Width of minima", default = 0.1)
parser.add_argument("--command", "-c", help = "Location where script was called", default = "lcf")
parser.add_argument("--method", "-m", help = "Method to find minima", default = "low_mags")
parser.add_argument("--show_plot", "-p", help = "Show fit plot", default = "true")
args = parser.parse_args()
data = args.indata
command = args.command
window = float(args.window)
numpoints = int(args.numpoints)
method = args.method
show_plot = args.show_plot

run = main(data)

# port main->end to lcvis 
# get data file from Aiden
