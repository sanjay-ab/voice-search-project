import os,sys
import numpy as np

dic={}
with open(sys.argv[1]) as buf:
    for line in buf:
        f,sp,s,e,ff,ssp,ss,ee,distance=line.rstrip().split(' ')
        distance=float(distance)
        key='!'.join([f,sp,s,e])
        kkey='!'.join([ff,ssp,ss,ee])
        if key not in dic:
            dic[key]=(kkey,distance)
        else:
            _,d=dic[key]
            if d<distance:
                dic[key]=(kkey,distance)
        if kkey not in dic:
            dic[kkey]=(key,distance)
        else:
            _,d=dic[kkey]
            if d<distance:
                dic[kkey]=(key,distance)
for key in dic:
    n,d=dic[key]
    print(' '.join(key.split('!')),d)
