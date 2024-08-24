import numpy as np
import os,sys
import pandas as pd
# from alignment_phone create all possible ngrams respecting
# the vads and min_dur<item<max_dur

def concat_ngram(ngram,item):
    #ngram : fid s1 e1 ph1,ph2
    #item: fid s2 e2 ph
    #out: fid s1 e2 ph1,ph2,ph
    f1,spk,vs,ve,s1,e1,phs=ngram
    f2,s2,e2,ph=item
    s1=float(s1)
    s2=float(s2)
    e1=float(e1)
    e2=float(e2)
    assert(e1<=s2)
    assert(f1==f2)
    assert vs<=s2,(ngram,item)
    assert ve>=e2,(ve,e2,ngram,item)
    return [f1,spk,vs,ve,s1,e2,phs+','+ph]

# from an array
#[[fid,start,end,phone],[...],..,[...]]
# output all possible ngram items between 0.2 and 0.6 ms
def output_all_ngrams(arr,vstart,vend,spk):
    #max_lengths=min(8,len(arr))
    min_len=0.08
    max_len=0.96
    ngrams=[]
    #print("lllllllllllllllllllllllllll")
    
    for i in range(len(arr)):
        fid1,s1,e1,ph1=arr[i]
        if ph1=='SIL':
            continue
        current_ngram=[fid1,spk,vstart,vend,s1,e1,ph1]
        s1=float(s1)
        e1=float(e1)
        ngram_dur=e1-s1
        assert(e1>s1)
        if ngram_dur>max_len:
            continue    
        if ngram_dur>=min_len:
            ngrams.append(current_ngram)
        #try to append more phone to current ngram
        for j in range(i+1,len(arr)):
            fid2,s2,e2,ph2=arr[j]
            if ph2=='SIL':
                continue
            s2=float(s2)
            e2=float(e2)
            dur=e2-s2
            assert s2>=e1, arr[j]
            assert e2>s2, arr[j]
            ngram_dur+=dur
            current_ngram=concat_ngram(current_ngram,arr[j])
            if ngram_dur>max_len:
                break    
            if ngram_dur>=min_len:
                ngrams.append(current_ngram)
    return ngrams      

def main():
    alignment_file=sys.argv[1]
    vad_file=sys.argv[2]
    alignment=pd.read_csv(alignment_file,sep=' ',header=None,names=['file','start','end','phone'])
    vad=pd.read_csv(vad_file,sep=' ',header=None,names=['file','spk','start','end'])
    alignment_groups=alignment.groupby('file')
    vad_groups=vad.groupby('file')
    c=0
    ngrams=[]
    for filename in vad_groups.groups:
        fid=filename.split('.')[0]
        alignment_current_file=alignment_groups.get_group(fid).to_numpy()
        dict_start2index={}
        dict_end2index={}
        for index in range(len(alignment_current_file)):
            _,start,end,_=alignment_current_file[index]
            start=np.around(start,decimals=2)
            end=np.around(end,decimals=2)
            dict_start2index[start]=index
            dict_end2index[end]=index
        for fid,spk,start,end in vad_groups.get_group(filename).to_numpy():
            start=np.around(start,decimals=2)
            end=np.around(end,decimals=2)
            if not start in dict_start2index:
                tmp=np.around(start-0.01,2)
                tmpbis=np.around(start+0.01,2)
                if tmp in dict_start2index:
                    start=tmp
                elif tmpbis in dict_start2index:
                    start=tmpbis
                else:       
                    print('skipping')
                    continue
            start_index=dict_start2index[start]
            if not end in dict_end2index:
                tmp=np.around(end-0.01,2)
                tmpbis=np.around(end+0.01,2)
                if tmp in dict_end2index:
                    end=tmp
                elif tmpbis in dict_end2index:
                    end=tmpbis
                else:
                    print('skipping')
                    continue
            end_index=dict_end2index[end]
                 
            ngrams.extend(output_all_ngrams(alignment_current_file[start_index:end_index+1],start,end,spk))
            
    for fid,spk,vstart,vend,start,end,ngram in ngrams:
        if ngram=='':
            continue
        print(fid+'.wav',spk,vstart,vend,np.round(start,decimals=4),np.round(end,decimals=4),ngram)
main()

