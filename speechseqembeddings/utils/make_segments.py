import numpy as np
import sys,os


#from a vad file, output all segments
vad_file=sys.argv[1]
output_file=sys.argv[2]
segments=[]
sample_rate=16
min_word_length=sample_rate
max_word_length=96

with open(vad_file,'r') as vads:
    for line in vads:
        f,spk,vs,ve=line.rstrip().split(' ')
        ve,vs=np.around([float(ve),float(vs)],decimals=2)
        dur=(ve-vs)*100
        if dur>=min_word_length:
            s=int(round(vs*100))
            e=int(round(ve*100))
            e=e-(e-s)%sample_rate
            for start in range(s,e-min_word_length+1,sample_rate):
                for end in range(start+min_word_length,min(start+max_word_length+1,e+1),sample_rate):
                    segments.append(f+" "+spk+" "+str(vs)+" "+str(ve)+" "+str(start/100)+" "+str(end/100)+" NOTRANS")
with open(output_file,'w') as buf:
    buf.write('\n'.join(segments)+'\n')
