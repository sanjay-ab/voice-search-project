import os,sys
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy import interpolate

h=[]
dic={}
with open(sys.argv[1]) as buf:
    for line in buf:
        _,_,d,_,_,_,n=line.rstrip().split(' ')
        h.append((n,float(d)))
        if n not in dic:
            dic[n]=0
        dic[n]+=1
other=[]
hapax=[]
alls=[]
for n,d in h:
#    if d>0.98:
#        continue
    alls.append(d)
    if dic[n]>1:
        other.append(d)
    else:
        hapax.append(d)
bins=100
b,a=np.histogram(other,bins=bins)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
for i in range(len(b)):
    print('o',b[i],a[i])
b,a=np.histogram(hapax,bins=bins)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
for i in range(len(b)):
    print('h',b[i],a[i])
b,a=np.histogram(alls,bins=bins)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
b=np.append(b,b[-1]);a=np.append(a,a[-1]+0.01)
for i in range(len(b)):
    print('a',b[i],a[i])
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
b = savgol_filter(b, 5, 2)
for i in range(len(a)-1):
    print('c',b[i],a[i])
#print(len(other),len(hapax))
#print(np.mean(other),np.std(other))
#print(np.mean(hapax),np.std(hapax))

