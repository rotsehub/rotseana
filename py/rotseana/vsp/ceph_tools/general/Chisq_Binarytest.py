# fit a straight line to the economic data
from scipy.optimize import curve_fit
from operator import itemgetter
import numpy as np 
import statistics
# define the methods

def collect(filename):
#### reads in file that is given returns it to use in the code####
    in_file = open(filename,'r')
    for a_line in in_file:
        print(a_line)
    return in_file

def collecttable(filename):
###sorts through the file and separates each line into order through the mag and separates it into pahse mag and Error###
    lightcurve= np.genfromtxt(filename, dtype=None,delimiter=None,skip_header=2,names=["phase","mag","error"])
    return lightcurve

def getlowestMag(sorted_lc,num):
###sorts through the array lightcurve and skips over the dupilcate lines and puts each into separate lines###
    x=range(num)
    Mag=[]
    Phase=[]
    Error=[]
    for i in x: 
       Mag.append(sorted_lc[len(sorted_lc) -2*i-1][1])
       Phase.append(sorted_lc[len(sorted_lc) -2*i-1][0])
       Error.append(sorted_lc[len(sorted_lc) -2*i-1][2])
    return Mag,Phase,Error

def getMedian(Mag,Phase):
### finds the meidan of Phase and Mag in the lightcurve file to help find the points around the first parabola###
    MagMedian = statistics.median(Mag)
    PhaseMedian = statistics.median(Phase)
    return MagMedian, PhaseMedian

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
        if diff > 0.2:
            removei.append(i)
    return removei

def difference2(Phase,Phasemedian,num):
    j=range(num)
    savei=[]
    print("Phasemedian",Phasemedian)
    for i in j:
        diff = abs(Phase[i] - Phasemedian)
        if diff < 0.2:
            savei.append(i)
    return savei
    

def objective(x, a, b, c):
#creates the parbola for each Minima#
    return a*(x-b)**2 + c

def fitminimum(Mag,Phase, a, b, c):
#finds the fitminimas a b c#
    params, _ = curve_fit(objective, Phase, Mag)
    return params


def getwindow(lightcurve,PhaseMedian,num):
#gets the second lightcurves minima for the Phase Mag and Errors#
    Phase_wind=[]
    Mag_wind=[]
    Error_wind=[]
    length_light = len(lightcurve)
    print('L=',length_light,PhaseMedian)
    j =range(length_light)   
    for i in j:
        abs_dif = abs(lightcurve[i][0] - PhaseMedian)
        if abs_dif <= num:
            Phase_wind.append(lightcurve[i][0])
            Mag_wind.append(lightcurve[i][1])
            Error_wind.append(lightcurve[i][2])
    return Phase_wind,Mag_wind,Error_wind

def calcchisq(Phase_wind,Mag_wind,Error_wind,a,b,c):
#calculates the Chisq for each Minima which is the result of the package# 
    mu = []
    calc_sum=[]
    length_wind = len(Phase_wind)
    k= range(length_wind)
    for i in k:
        mu.append(objective(Phase_wind[i],a,b,c))
        calc_sum.append(((Mag_wind[i]-mu[i])/Error_wind[i])**2)
    total_calc = np.sum(calc_sum)    
    Chisq = total_calc/length_wind-3
    return Chisq

def getMean(guessPhase,Mag,Error,Phase):
    n_bin = 6
    print('n_bin',n_bin)   
    g = np.array(guessPhase)
    wind_Min2 = 0.01
    print('Wind_Min2',wind_Min2)    
    wind_Min = g-0.1
    print('wind_Min',wind_Min)
    wind_Max = g + 0.1
    print('wind_Max',wind_Max)
    binsize = (wind_Max - wind_Min)/n_bin
    print('binsize',binsize)
    Mean_Mag = []
    Erroravg = []
    phase_bin =[0]
    x =  range(n_bin)
    for i in x:
        phase_bin.append(wind_Max + 1/2 * binsize + i * binsize)
        length = len(Phase)
        k = range(length)
        avgfirst=[]
        avgsecond=[]
        Error_final=[]
        for j in k:
            mag_final=[]
            minus_cond = phase_bin-binsize/2
            plus_cond = phase_bin+binsize/2
            if Phase[j] > minus_cond  and Phase[j] < plus_cond:
                mag_final.append(Mag[j])
                print('mag_Final',magFinal) 
                Error_final.append(Error[j])
                print('Error_Final',Error_final)
                avgfirst.append( mag_final[j]/Error_final[j])
                print('Avgfirst',avgfirst)
                avgsecond.append(1.0/Error_final[j])
                print('Avgsecond',avgsecond)
        first_sum = np.sum(avgfirst)
        print('First Sum', first_sum)
        second_sum = np.sum(avgsecond)
        print('Second_Sum',second_sum)
        Mean_Mag.append(first_sum/second_sum)        
        print('Mean_Mag',Mean_Mag)
        Erroravg.append(np.sum(Error_final))
        print('Erroravg',Erroravg)
    return Mean_Mag

    ############# Start of setup of array#################

def exlightfile(data):
#########Reads in lightcurve file and sorts it by Magnitude##########
    a=0.6666666
    lightcurve = collecttable(data)
    sorted_lc= sorted(lightcurve, key=itemgetter(1))
    return lightcurve, sorted_lc

def getfirstMin(sorted_lc,lightcurve): 
####### Takes the sorted lightcurve and finds the Phase Mag Error and the parabola of it  ############
    Mag, Phase,Error=getlowestMag(sorted_lc,20)
    MagMedian, PhaseMedian = getMedian(Mag,Phase)
    removei = difference(Phase,PhaseMedian,20)
    for i in removei:
         Mag.pop(removei[i])
         Phase.pop(removei[i])
         Error.pop(removei[i])
    print(Mag)
    print(Phase)
    a=0.6666666
    params_Min1 = fitminimum(Mag, Phase, a, PhaseMedian, MagMedian)
    a,b,c = params_Min1
    FittedPhase1 = b
    FittedMag1 = c
    print('first fitMinimum:')
    print('y= %.5f *(x-%.5f)**2 + %.5f' %(a,FittedPhase1,FittedMag1))
    Phase_wind,Mag_wind,Error_wind = getwindow(lightcurve,FittedPhase1,0.05)
    return FittedPhase1, FittedMag1, Phase_wind, Mag_wind, Error_wind, a

def getsecMin(lightcurve,FittedPhase1,sorted_lc, a):
######## Uses the a fit of the first Minima and finds the different parabola and Mag Phase aand Median ###############
    guessPhase = [] 
    if FittedPhase1 >1.5:
        guessPhase.append(FittedPhase1 -0.6)
    else:
        guessPhase.append(FittedPhase1+0.6)
    print("GuessPhase=",guessPhase)
    Lightcurve_len = len(lightcurve)
    Mag2,Phase2,Error2= getlowestMag(sorted_lc,Lightcurve_len)
    Phase2_len = len(Phase2)
    if Lightcurve_len > 500:
        Lightcurve_len = 500
    savei = difference2(Phase2,guessPhase,Lightcurve_len)
    savei_len = len(savei)
    l=range(savei_len-1)
    Phase3 = []
    Mag3 = []
    Error3 = []
    for i in l:
        Phase3.append(Phase2[savei[i]])
        Mag3.append(Mag2[savei[i]])
        Error3.append(Error2[savei[i]])
    Phase3_len = len(Phase3)
    Mag3_len = len(Mag3)
    MagMedian2 = statistics.median(Mag3)
    PhaseMedian2 = statistics.median(Phase3)
    print(a)
    params_Min2 = fitminimum(Mag3, Phase3, a, PhaseMedian2,MagMedian2)
    a_prime,b_prime,c_prime = params_Min2
    Phase_windprime,Mag_windprime,Error_windprime=getwindow(lightcurve,b_prime,0.05)
    print('Second Minimum:')
    print('y= %.5f *(x-%.5f)**2 + %.5f' %(a_prime,b_prime,c_prime))
    return Phase_windprime, Mag_windprime, Error_windprime, a_prime, b_prime, c_prime 

def difresult(Phase_windprime, Mag_windprime, Error_windprime, a_prime, b_prime, c_prime,FittedMag1,a,FittedPhase1,Mag_wind,Error_wind, Phase_wind):
######### Uses the data found from the first & second Minima to discover the difference and wether the lightcurve file is a binary or pulsating variable##########
    Chisq11=calcchisq(Phase_wind,Mag_wind,Error_wind,a,FittedPhase1,FittedMag1)
    print('First Min Chisq=',Chisq11)
    Chisq22=calcchisq(Phase_windprime,Mag_windprime,Error_windprime,a_prime,b_prime,c_prime)
    print('Second Min Chisq=',Chisq22) 
    Chisq12=calcchisq(Phase_windprime,Mag_windprime,Error_windprime,a,b_prime,FittedMag1)
    print('Second Min data & first fitminimum=',Chisq12)
    Chisq21=calcchisq(Phase_wind,Mag_wind,Error_wind,a_prime,FittedPhase1,c_prime)
    print('First min data & Second fitminimum=',Chisq21)
    return Chisq11, Chisq22, Chisq12, Chisq21

def MainFunc(data):
    lightcurve, sorted_lc = exlightfile(data)
    FittedPhase1, FittedMag1, Phase_wind, Mag_wind, Error_wind, a = getfirstMin(sorted_lc,lightcurve)
    Phase_windprime, Mag_windprime, Error_windprime, a_prime, b_prime, c_prime = getsecMin(lightcurve,FittedPhase1, sorted_lc,a)
    Chisq11, Chisq22, Chisq12, Chisq21 = difresult(Phase_windprime, Mag_windprime, Error_windprime, a_prime, b_prime, c_prime, FittedMag1, a, FittedPhase1, Mag_wind, Error_wind,Phase_wind)
    return

import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("indata",help = "Object's data file")
args = parser.parse_args()
data = args.indata

run = MainFunc(data)
 
