import numpy as np
import csv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("datafile")
parser.add_argument("output_name")
parser.add_argument("kind",nargs='?',default='V')
args = parser.parse_args()
datafile = args.datafile
output_name = args.output_name
kind = args.kind
with open( datafile ) as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    data = list(reader)  
data.remove(data[0])
out = list()
for i in data:
    b1 = i[1]
    b2 = i[2]
    b3 = i[3]
    b4 = i[4]
    b7 = i[7]
    b8 = i[8]
    k = i[9]
    i.remove(b1)
    i.remove(b2)
    i.remove(b3)
    i.remove(b4)
    i.remove(b7)
    i.remove(b8)
    i.remove(k)
    if '>' not in list(i[1]) and i[1] != '99.990' and i[2] != '99.990' and k == kind:
        t = float(i[0]) - 2400000
        m = float(i[1])
        e = float(i[2])
        out.append([t,m,e])
np.savetxt(str(output_name)+'.dat',out)
