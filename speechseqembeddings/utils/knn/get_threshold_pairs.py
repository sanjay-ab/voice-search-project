import tqdm
import os,sys
import numpy as np

# get pairs so that 50% of segments only are represented

pair_file=sys.argv[1] # pair file in train/
nb_segments=int(sys.argv[2]) # how many segments in corpus?
output_file=sys.argv[3]
if len(sys.argv)==5:
    percent=float(sys.argv[4])
else:
    percent=0.7
nb_segments*=percent # we are looking for half of the segments
print("computing threshold")
print('keeping',percent,'% of segments')
hist={}
count={}
allkeys={}
c=0
with open(pair_file) as buf:
    for line in buf:
        tmp=line.rstrip().split(" ")
        f,_,s,e,_,_,_,_,d=tmp
        d=int(float(d)*100)
        if d not in hist:
            hist[d]={}
            count[d]=0
        count[d]+=1
        if (f,s,e) not in allkeys:
            allkeys[(f,s,e)]=0
            if (f,s,e) not in hist[d]:
                hist[d][(f,s,e)]=0
nb_uniq_segs=0
nb_segs=0
keys=np.sort(list(hist.keys()))[::-1]
print('allkeys',len(allkeys))
d=[]
for key in keys:
    nb_uniq_segs+=len(hist[key])
    nb_segs+=count[key]
    d.append(np.abs(nb_uniq_segs-nb_segments))
    print('distance',key,'nb uniq segments',nb_uniq_segs,'nb segments',nb_segs)
i=np.argsort(d)[0]
threshold=keys[i]
print('threshold is',threshold)
saved=[]
with open(pair_file) as buf:
    for line in buf:
        tmp=line.rstrip()
        sim=tmp.split(" ")[-1]
        sim=int(float(sim)*100)
        if sim>=threshold:
            saved.append(tmp)
with open(output_file,"w") as buf:
    buf.write('\n'.join(saved)+'\n')
    
