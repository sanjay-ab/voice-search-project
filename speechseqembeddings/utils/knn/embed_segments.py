# !/usr/bin/env python
# Author = The Zero Resource Challenge Team
from utils.data import get_features
import numpy as np
import os,sys
import torch
import random

def read_randomsegments(folderPath,max_duration):

    randomseg = {}
    count_train=0
    segments,utt2spk_list=[],[]
    utt2spk={}
    with open(folderPath) as f:
        for line in f:
            path, spk, vstart, vend, start, end, _ =line.split()
            if float(end)-float(start)>=max_duration:
                continue
            fid=path
            #fid=os.path.basename(path).split('.')[0]
            segments.append((spk,fid, vstart, vend, float(start), float(end)))
            if fid not in utt2spk:
                utt2spk[fid]=spk
                utt2spk_list.append((fid,spk))
    print('total nb of segments',len(segments))
    max_block_size=40000
    nb_max_segments=4000000
    counts={}
    print('please make sure segments must are sorted by vad and increasing start time')
    previous_spk,_,_,_,previous_start,previous_end=segments[0]
    randint=int(100*nb_max_segments/len(segments))
    if randint<100:
        print('not all segments are necessary, taking',randint,'% of segments')
    for segment in segments:
        if randint<100 and np.random.randint(0,100)>randint:
            continue
        spk,fid,vstart,vend,start,end=segment
        assert end>start,segment
        if spk==previous_spk:
            assert previous_start<=start
        if spk not in counts:
            counts[spk]=[0,0]# count and index
        if counts[spk][0]>max_block_size and start>previous_end:
            counts[spk][0]=0
            counts[spk][1]+=1
        counts[spk][0]+=1
        spk+='_'+str(counts[spk][1])
        vad=spk+'-'+vstart+'-'+vend
        if spk not in randomseg:
            randomseg[spk]={}
        if vad not in randomseg[spk]:
            randomseg[spk][vad]=[]
        randomseg[spk][vad].append((fid,start, end))
    
        previous_spk=spk
        previous_start=start
        previous_end=end
        count_train+=1
        if count_train>=nb_max_segments:
            break
    print("nb of segments",count_train)
    print(counts)
    return randomseg,utt2spk_list

def process_file(args):
    '''Segment into word candidates, downsample, and write into h5 format'''
    npz_data, max_frames, framerate, segments, model = args

    # The variable utterances stores, for each utterance,
    # the last embedding's index in that utterance.
    # All embeddings in an utterance are between the previous
    # utterance's index and the index of the current utterance.
    utt_idx,utt_num=0,0
    embeddings,utterances,timestamps,batch_vector,filenames=[],[],[],[],[]
    npz_dict={}
    with torch.no_grad(): 
        for vad in segments:
            utt_num+=1
            for i in range(len(segments[vad])):
                path, embedding_start, embedding_end=segments[vad][i]
                fid=os.path.basename(path).split('.')[0]
                if fid not in npz_dict:
                    npz_dict[fid]=npz_data[fid].ravel()[0]
                utt_idx += 1
                utterances.append([[utt_num, utt_idx,fid]])
                frames=get_features(npz_dict,embedding_start,embedding_end,max_frames,framerate,fid)   
                timestamps.append(np.around([embedding_start*100,embedding_end*100],decimals=0).reshape(1,-1))  
                batch_vector.append(frames)     
                if len(batch_vector)>=800 or i==len(segments[vad])-1:
                    batch_vector=torch.cat([b for b in batch_vector])
                    feats= model(batch_vector.cuda().float())
                    embeddings.append(feats.cpu().detach())
                    batch_vector=[]                
#    utterances.append([[utt_num+1, utt_idx,fid]])
    embeddings=np.float32(torch.cat(embeddings))
    utterances=np.concatenate(utterances,axis=0)
    timestamps=np.concatenate(timestamps,axis=0)
    return embeddings,timestamps,utterances

