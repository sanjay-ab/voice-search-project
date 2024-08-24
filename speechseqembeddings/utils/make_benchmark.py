import os
import sys
import numpy as np
import random
m=[]
words={}
with open(sys.argv[1]) as buf:
    for line in buf:
        f,spk,vs,ve,s,e,trans=line.rstrip().split(" ")
        if float(e)-float(s)>=2:
            continue
        m.append((f,spk,vs,ve,s,e,trans))
random.shuffle(m)
newm=[]
c=0
for item in m:
    f,spk,vs,ve,s,e,trans=item   
    if trans not in words:
        words[trans]=[]
    words[trans].append(item)
    c+=1
    if c>=200000:
        break
c=0
for trans in words:
    if len(words[trans])>1:
        for item in words[trans]:
            print(item)
            c+=1
            if c>150000:
                break
