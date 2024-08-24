import os,sys
import numpy as np
def ttr(F):
    N = len(F)
    ttr = 0
    for i in range(N):
        ttr += 1/F[i]
    return ttr/N

def entropy(F):
    N = len(F)
    entropy = 0
    for i in range(N):
        entropy += np.log(F[i]/N)
    return - entropy/N

def estim_model(c_0,c_1,N,alpha = 0.4, gold = False):
    if not gold:
        slope = 0.03608695652173913
        c_1 = slope*N
    model = []
    for i in range(c_0):
        model.append(1)
    for i in range(c_0,N):
        model.append(np.power(c_1,(i**alpha-c_0**alpha)/(N**alpha-c_0**alpha)))
    return model

h=[]
dic={}
thresh=float(sys.argv[2])
with open(sys.argv[1]) as buf:
    for line in buf:
        _,_,d,_,_,_,n=line.rstrip().split(' ')
        h.append((n,float(d)))
        if n not in dic:
            dic[n]=0
        dic[n]+=1
hapax=0
c0=0
for n,d in h:
    if dic[n]==1:
        hapax+=1
    if d<thresh:    
        c0+=1        
types=len(dic)
tokens=len(h)
res=estim_model(c0,None,tokens)
ettr=ttr(res)
ettr=ttr(res)
rttr=types/tokens
print('tokens',tokens,'hapax',hapax,'c0',c0,np.around(100*rttr,2),np.around(100*ettr,2),np.abs(np.around(100*(np.abs(rttr-ettr)/rttr),2)))

