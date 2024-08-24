import os,sys
import numpy as np
# from TDE .wrd and .vad, output the vads and words file
# vads file:
# filename.wav spk start end
# word file
# filename.wav spk vad_start vad_end start end

seg_file=sys.argv[1]
corpus=sys.argv[2] # or subset if Librispeech
vad_file=sys.argv[3]
output_dir=sys.argv[4]
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
max_segment_dur=2.0
segs,vads={},{}
with open(seg_file) as buf:
    for line in buf:
        tmp=line.rstrip().split(' ')
        f,s,e=tmp[:3]
        if corpus in ['test-clean','test-other','dev-clean','dev-other','train-clean-360','train-clean-100','train-other-500']:
            ff=f.split('-')
            fid=os.path.join(corpus,ff[0],ff[1],f+'.flac')
            spk=ff[0]
        else:
            fid=f+'.wav'
            spk=f
        if fid not in segs:
            segs[fid]=[]
        if len(tmp)==4:
            trans=tmp[3]
        else:
            trans='NOTRANS'
        s,e=float(int(100*float(s)))/100,float(int(100*float(e)))/100
        if e-s>max_segment_dur:
            continue
        assert s<e
        segs[fid].append((spk,s,e,trans))
with open(vad_file) as buf: 
    for line in buf:
        fid,s,e=line.rstrip().split(' ')    
        s,e=max(0,float(int(100*float(s)-1))/100),float(int(100*float(e)+1))/100
        if fid not in vads:
            vads[fid]={'start':[],'end':[]}
        vads[fid]['start'].append(s)
        vads[fid]['end'].append(e)

for fid in vads:
    vads[fid]['start']=np.array(vads[fid]['start']) 
    vads[fid]['end']=np.array(vads[fid]['end']) 

words,outvads=[],[]
tmp={}
for fid in segs:
    fid_noext=fid.split('/')[-1].split('.')[0]
    for spk,s,e,trans in segs[fid]:
        ind_vs=np.where(vads[fid_noext]['start']<=s)[0]
        ind_ve=np.where(vads[fid_noext]['end']>=e)[0]
        if len(ind_vs)==0 or len(ind_ve)==0:
        #    print('NOT IN VADS',spk,s,e,trans)
            continue
        ind_vs,ind_ve=ind_vs[-1],ind_ve[0]
        vs=vads[fid_noext]['start'][ind_vs]
        ve=vads[fid_noext]['end'][ind_ve]
        if ind_vs!=ind_ve:
            if np.abs(e-ve)<=0.01:
                ind_ve-=1
            elif np.abs(s-vs)<=0.01:
                ind_vs+=1
            else:
                #print('SKIPPINGG',vs,ve,spk,s,e,trans)
                continue
            vs=vads[fid_noext]['start'][ind_vs]
            ve=vads[fid_noext]['end'][ind_ve]
        if not (vs<=s and ve>=e):
            #print('SKIPPING',vs,ve,fid,s,e,trans)
            continue
        vs,ve,s,e=str(vs),str(ve),str(s),str(e)
        words.append(' '.join([fid,spk,vs,ve,s,e,trans]))
        key=' '.join([fid,spk,vs,ve])
        if key not in tmp:
            tmp[key]=0
            outvads.append(key)
with open(os.path.join(output_dir,corpus+'-vads'),'w') as buf:
    buf.write('\n'.join(outvads)+'\n')  
       
with open(os.path.join(output_dir,corpus+'-words'),'w') as buf:
    buf.write('\n'.join(words)+'\n')  
            
