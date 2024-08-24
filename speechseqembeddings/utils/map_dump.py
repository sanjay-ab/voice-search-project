from sklearn.decomposition import PCA
import sys,os
import json
import torch
import torchaudio
import numpy as np
import tqdm
import fairseq
from utils.models import SSEmodel
from utils.metrics import compute_map
import torchaudio.transforms as T
import argparse

def apply_task(dict_path_wavs,ssl,sse,ssl_layer,device,max_segment_size,task,output_file):
    feat_sr=ssl.feat_sr
    out={'embeddings':{},'times':{}}
    times={}
    offset=5
    segment_ids=[]
    batch_size=200
    c=0
    for path_wav in tqdm.tqdm(dict_path_wavs):
        batch,batch_times=[],[] 
        wav_data, sr = torchaudio.load(path_wav)
        fname=os.path.basename(path_wav).split('.')[0]
        if sr!=16000:
            resampler=T.Resample(sr,16000,dtype=wav_data.dtype)
            print('resampling to 16000')
            wav_data=resampler(wav_data)
            sr=16000
        signal=wav_data.to(device)
        signal_end=len(signal[0])/sr
        for vad in dict_path_wavs[path_wav]:
            vad_batch=[]
            segment_items=dict_path_wavs[path_wav][vad]
            if len(segment_items)==0:
                continue
            vs,ve=[float(v) for v in vad.split('-')]
            offs=min(offset,vs) 
            offe=min(offset,signal_end-ve)
            nb_frame_offs=int(offs*feat_sr)
            nb_frame_offe=int(offe*feat_sr)
            nb_frames=int((ve-vs)*feat_sr)+1
            frames=ssl(signal[:,int((vs-offs)*sr):int((ve+offe)*sr)],padding_mask=None,mask=False,features_only=True,layer=ssl_layer)['x'].squeeze(0)
            if nb_frame_offe>0:
                frames=frames[nb_frame_offs:-nb_frame_offe]
            elif nb_frame_offs>0:
                frames=frames[nb_frame_offs:]
            assert len(frames)>0
            times=np.linspace(vs+1/(feat_sr*2),ve+1/(feat_sr*2),frames.shape[0],endpoint=False)
            for item in segment_items:
                try:
                    start,end,trans_id,trans=item
                    #end+=0.02
                    t = np.where(np.logical_and(times >= start, times <= end))[0]
                    assert len(t)<=1+np.ceil((end-start)*feat_sr),(len(t),item,(end-start)*feat_sr)
                    features=frames[t,:]
                    pad=torch.zeros(max_segment_size,features.size(1)).to(device)
                    pad[:len(features)]=features
                except:
                    print(item,'failed')
                vad_batch.append(pad.unsqueeze(0))
                segment_ids.append(trans_id)
                batch_times.append([start,end])
            c+=len(segment_items)
            vad_batch=torch.cat(vad_batch,0)
            if vad_batch.size(0)>batch_size:
                nb_batches=int(vad_batch.size(0)/batch_size)+1
                for i in range(nb_batches):
                    tmp=vad_batch[i*batch_size:(i+1)*batch_size]
                    embs=sse(tmp).detach().cpu()
                    batch.append(embs)    
            else:
                vad_batch=sse(vad_batch).detach().cpu()
                batch.append(vad_batch)
        batch=torch.cat(batch,0) 
        batch_times=np.array(batch_times)
        out['embeddings'][fname]=batch.numpy()
        out['times'][fname]=batch_times
        #if c>1000:
        #    break
        c+=1
    if task=='dump':    
        np.savez(output_file,**out) 
        print('saved at',output_file)
    else:
        embeddings=[out['embeddings'][key] for key in out['embeddings']]
        embeddings=np.concatenate(np.array(embeddings,dtype=object),axis=0)
        segment_ids=np.array(segment_ids)
        print('computing MAP on tensors:',embeddings.shape)
        map_value=compute_map(embeddings,segment_ids,faster=False)
        print('MAP:',map_value) 
        
    return

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_wavs",type=str,help='path to main wav directory',default='/gpfsdswork/dataset/LibriSpeech')
    parser.add_argument("--path_segments",type=str,help='segment lists, each line should be: <relative wav file> <spk_id> <vad start> <vad end> <segment start> <segment end> <transcription> (transcription can be None)',default='sse_benchmark/dev-clean-ngrams-subset')
    parser.add_argument('--task',type=str,help='choose map or dump. map return the MAP score and dump returns a npz file with segment embeddings',default='map')
    parser.add_argument('--output_file',type=str,help='output filename in case task is dump',default=None)
    parser.add_argument('--path_ssl',type=str,help='path to wav2vec2.0 small',default='pretrained/wav2vec_small.pt')
    parser.add_argument('--path_sse',type=str,help='path to pretrained SSE model',default='pretrained/librispeech_unsup/')
    return parser.parse_args(argv)

if __name__ == "__main__":
    args=parse_arguments(sys.argv[1:])
    print('arguments',args)
    # path_wavs=args.path_wavs 
    path_wavs="LibriSpeech" 
    # path_segments=args.path_segments 
    path_segments="sse_benchmark/dev-clean-ngrams-subset" 
    task="map"
    output_file=args.output_file
    path_ssl=args.path_ssl
    path_sse="pretrained/librispeech_unsup/"
    if args.task=='dump':
        assert args.output_file is not None,'please provide a name for the outputfile'
    if torch.cuda.is_available():
        device='cuda:0'
        torch.cuda.set_device(device)
    else:
        device='cpu'
    dict_path_wavs={} 
    trans2ind={}
    count=0    
    trans=None
    with open(path_segments) as buf:
        for line in buf:
            path,_,vs,ve,s,e,trans=line.rstrip().split(' ')
            path=os.path.join(path_wavs,path)
            assert os.path.isfile(path),path
            if path not in dict_path_wavs:
                dict_path_wavs[path]={}
            key=vs+'-'+ve
            if key not in dict_path_wavs[path]:
                dict_path_wavs[path][key]=[]
            s,e,vs,ve=[float(ts) for ts in [s,e,vs,ve]]
            if trans not in trans2ind:
                trans2ind[trans]=count
                count+=1
            dict_path_wavs[path][key].append((s,e,trans2ind[trans],trans))
    # LOADING SSL
    ssl_layer=8
    ssl, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([path_ssl])
    ssl=ssl[0].float().eval().to(device)
    ssl.feat_sr=50
    max_segment_size=100 # 2 seconds
    # LOADING SSE
    input_size = 768
    sse_config=os.path.join(path_sse,'params.json')
    with open(sse_config) as buf:
        config=json.load(buf) 
    sse = SSEmodel(
        input_size=input_size,
        n_conv_layers=config["n_conv_layers"],
        transformer_dim=config["transformer_dim"],
        n_heads=config["n_heads"],
        n_transformer_layers=config["n_transformer_layers"],
        device=device,
        )
    sse_model=os.path.join(path_sse,'model_checkpoint.tar')
    sse.load_state_dict(torch.load(sse_model, map_location=torch.device(device))['model_state_dict'])
    sse.eval().to(device)

    apply_task(dict_path_wavs,ssl,sse,ssl_layer,device,max_segment_size,task,output_file)
