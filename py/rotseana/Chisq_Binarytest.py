# fit a straight line to the economic data
from numpy import arange
from pandas import read_table
from scipy.optimize import curve_fit
from matplotlib import pyplot
from operator import itemgetter
import numpy as np 
import statistics
# define the methods
def collect(filename):
    in_file = open(filename,'r')
    for a_line in in_file:
        print(a_line)
    return in_file

def collecttable(filename):
    lightcurve= np.genfromtxt(filename, dtype=None,delimiter=None,skip_header=2,names=["phase","mag","error"])
    print(lightcurve)
    print(lightcurve["phase"][0])
    print(lightcurve["mag"])
    print(lightcurve["error"])
    return lightcurve

def Min1(sorted_lc,num):
    x=range(num)
    Mag=[]
    Phase=[]
    Error=[]
    for i in x: 
       print(sorted_lc[i][1])
       Mag.append(sorted_lc[len(sorted_lc) -2*i-1][1])
       Phase.append(sorted_lc[len(sorted_lc) -2*i-1][0])
       Error.append(sorted_lc[len(sorted_lc) -2*i-1][2])
    return Mag,Phase,Error

def getMedian(Mag,Phase):
    MagMedian = statistics.median(Mag)
    PhaseMedian = statistics.median(Phase)
    return MagMedian, PhaseMedian

def tunedMin2(Mag,Phase,Error,PhaseMedian,num):
    k=range(num)
    print(PhaseMedian)
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
    j=range(num)
   # dif=[]
    abs_num=[]
    for i in j:
      # dif.append(PhaseMedian - Phase[j])
        abs_num.append(abs(Phase[i] - PhaseMedian))
       # print("a_n", abs_num)
    k=range(num)
    #print("A", abs_num)
    ilist=[]
    for i in k:     
        #print("i_a", abs_num[i])
        if abs_num[i] > 0.2:
            ilist.append(i)
    return ilist
    
def objective(x, a, b, c):
    return a*(x-b)**2 + c

def fitminimum(Mag,Phase, a, b, c):
    params, _ = curve_fit(objective, Phase, Mag)
    return params

def fitplot(x, y, a, b, c):
    return 

def getwindow(lightcurve,PhaseMedian,num):
    Phase_wind=[]
    Mag_wind=[]
    Error_wind=[]
    length_light = len(lightcurve)
    print('L=',length_light,PhaseMedian)
    j =range(length_light)   
    for i in j:
        abs_dif = abs(lightcurve[i][0] - PhaseMedian)
       # print('abs_dif=', abs_dif, lightcurve[i][0])       
        if abs_dif <= num:
            Phase_wind.append(lightcurve[i][0])
            Mag_wind.append(lightcurve[i][1])
            Error_wind.append(lightcurve[i][2])
    return Phase_wind,Mag_wind,Error_wind

def calcchisq(Phase_wind,Mag_wind,Error_wind,a,b,c):
    mu = []
    calc_sum=[]
    length_wind = len(Phase_wind)
   # print('l_w=', length_wind,a,b,c)
    k= range(length_wind)
    for i in k:
      #  mu = objective(lc_wind[i],a,b,c)
        mu.append(objective(Phase_wind[i],a,b,c))
        calc_sum.append(((Mag_wind[i]-mu[i])/Error_wind[i])**2)
       # print('Mu=',mu[i],'Calc Sum=', calc_sum[i])
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
       # print('Phase_bin',phase_bin[i])
        length = len(Phase)
        k = range(length)
        avgfirst=[]
        avgsecond=[]
        Error_final=[]
        #print('phase',Phase)
        #print('length', length)
        #print('K',k)
        for j in k:
            #print('J',j)
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
       # Mean_Mag.append(np.sum(avgfirst)/np.sum(avgsecond))
        Mean_Mag.append(first_sum/second_sum)        
        print('Mean_Mag',Mean_Mag)
        Erroravg.append(np.sum(Error_final))
        print('Erroravg',Erroravg)
    return Mean_Mag
      
a=0.6666666
lightcurve = collecttable('J1125+4234_b2.txt')
print(lightcurve)
print('lightcurve')
sorted_lc= sorted(lightcurve, key=itemgetter(1))
print("sorted lc ",sorted_lc)
Mag, Phase,Error=Min1(sorted_lc,20)
print("y=",Mag)
print("x=",Phase)
MagMedian, PhaseMedian = getMedian(Mag,Phase)
ilist = difference(Phase,PhaseMedian,20)
print(ilist)
print("M=",MagMedian)
print("P=",PhaseMedian)
for i in ilist:
     Mag.pop(ilist[i])
     Phase.pop(ilist[i])
     Error.pop(ilist[i])
print(Mag)
print(Phase)
params_Min1 = fitminimum(Mag, Phase, a, PhaseMedian, MagMedian)
a,b,c = params_Min1
print('y= %.5f *(x-%.5f)**2 + %.5f' %(a,b,c))
Phase_wind,Mag_wind,Error_wind = getwindow(lightcurve,b,0.05)
####### End of First Minimum Finding  ############

#Chisq= calcchisq(Phase_wind,Mag_wind,Error_wind,a,b,c)
#print('MU=', mu)
#total_sum= np.sum(mu)
#print('Total Sum=',total_sum)
#print('Calc sum=',calc_sum)
#total_calc= np.sum(calc_sum)
#finished_calc= total_calc/length_wind-3
#print('Total calc=',total_calc)
#print('Finished calc=',finished_calc)
guessPhase = [] 
if b >1.5:
    guessPhase.append(b -0.5)
else:
    guessPhase.append(b+0.5)
Mag2,Phase2,Error2= Min1(sorted_lc,99)
tunedPhase2,tunedMag2,tunedError= tunedMin2(Mag2,Phase2,Error2,guessPhase,99)
print(tunedPhase2)
print(tunedMag2)
print(guessPhase)
MagMedian2 = statistics.median(tunedMag2)
#Mean_Mag = getMean(guessPhase, tunedMag2,tunedError,tunedPhase2)
params_Min2 = fitminimum(tunedMag2, tunedPhase2, a, guessPhase,MagMedian2)
a_prime,b_prime,c_prime = params_Min2
Phase_windprime,Mag_windprime,Error_windprime=getwindow(lightcurve,b_prime,0.05)
print('y= %.5f *(x-%.5f)**2 + %.5f' %(a_prime,b_prime,c_prime))
######## End of Second Minimum Finding ###############

Chisq11=calcchisq(Phase_wind,Mag_wind,Error_wind,a,b,c)
Chisq22=calcchisq(Phase_windprime,Mag_windprime,Error_windprime,a_prime,b_prime,c_prime)
Chisq12=calcchisq(Phase_windprime,Mag_windprime,Error_windprime,a,b_prime,c)
Chisq21=calcchisq(Phase_wind,Mag_wind,Error_wind,a_prime,b,c_prime)
print(Chisq11)
print(Chisq22)
print(Chisq12)
print(Chisq21)
