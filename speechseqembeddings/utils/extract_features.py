import sys,os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import tqdm
import fairseq
from utils.models import SSEmodel
import argparse

def forward_ssl(signal,ssl):
    frames=ssl(signal,padding_mask=None,mask=False,features_only=True,layer=ssl_layer)['x'].squeeze(0)
    frames=frames.detach().cpu().numpy() 
    return frames

def dump_features(dict_path_wavs,ssl,ssl_layer,device,output_dir):
    skipped=0
    offset=5 # 5 second extra audio on each side when encoding a vad
    feat_sr=ssl.feat_sr
    for spk in tqdm.tqdm(dict_path_wavs):
        output_path_npz=os.path.join(output_dir,spk+'.npz')
        if os.path.isfile(output_path_npz):
            print(output_path_npz,'already exists')
            continue
        npz={}
        for path_wav in dict_path_wavs[spk]:
            batch=[]
            batch_times=[]
            fname=os.path.basename(path_wav).split('.')[0]
            signal, sr = torchaudio.load(path_wav)
            if sr!=16000:
                print('resampling to 16000')
                resampler=T.Resample(sr,16000,dtype=signal.dtype)
                signal=resampler(signal)
                sr=16000               
            signal=signal.to(device)
            end=np.around(len(signal[0])/sr,2)
            assert fname not in npz
            for vs,ve in dict_path_wavs[spk][path_wav]:
                offs=min(offset,vs) 
                offe=min(offset,end-ve)
                nb_frame_offs=int(offs*feat_sr)
                nb_frame_offe=int(offe*feat_sr)
                frames=forward_ssl(signal[:,int((vs-offs)*sr):int((ve+offe)*sr)],ssl)
                if nb_frame_offe>0:
                    frames=frames[nb_frame_offs:-nb_frame_offe]
                else:
                    frames=frames[nb_frame_offs:]
                assert len(frames)>0,(frames,vs,ve,end,offset,offs,offe,path_wav)
                times=np.linspace(vs+1/(feat_sr*2),ve+1/(feat_sr*2),frames.shape[0],endpoint=False)
                assert times.shape[0]==frames.shape[0],(times.shape,frames.shape)
                batch.append(frames)
                batch_times.append(times)
            batch=np.concatenate(batch)
            batch_times=np.concatenate(batch_times)
            npz[fname]={'features':batch,'times':batch_times}
            assert batch.shape[0]>0,(fname,batch)
        np.savez(output_path_npz,**npz)
    return

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_wavs",type=str,help='path to main wav corpus directory',required=True)
    parser.add_argument("--path_vads",type=str,help='path to vads',required=True)
    parser.add_argument('--output_dir',type=str,help='output directory',required=True)
    parser.add_argument('--path_ssl',type=str,help='path to wav2vec2.0 small',default='pretrained/wav2vec_small.pt')
    return parser.parse_args(argv)

if __name__ == "__main__":
    args=parse_arguments(sys.argv[1:])
    print('arguments',args)
    path_wavs=args.path_wavs 
    path_vads=args.path_vads 
    output_dir=args.output_dir
    path_ssl=args.path_ssl
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if torch.cuda.is_available():
        device='cuda:0'
        torch.cuda.set_device(device)
    else:
        device='cpu'
    dict_path_wavs={} 
    seen_paths={}
    max_vad_length=20
    with open(path_vads) as buf:
        for line in buf:
            path,spk,vs,ve=line.rstrip().split(' ')
            path=os.path.join(path_wavs,path)
            assert os.path.isfile(path),path
            if spk not in dict_path_wavs:
                dict_path_wavs[spk]={}
            if path not in dict_path_wavs[spk]:
                dict_path_wavs[spk][path]=[] 
            vs,ve=float(vs),float(ve)
            if ve-vs<0.04:
                continue
            dur=ve-vs
            assert dur>0
            if dur>max_vad_length:
                nb_vads=int(dur/(max_vad_length)+1)
                for i in range(nb_vads):
                    if ve-vs<=0.04:
                        break
                    end=min(vs+max_vad_length,ve)
                    dict_path_wavs[spk][path].append((vs,end))
                    vs+=max_vad_length
            else:
                dict_path_wavs[spk][path].append((vs,ve))
    # LOADING SSL
    ssl_layer=8
    ssl, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([path_ssl])
    ssl=ssl[0].float().eval().to(device)
    ssl.feat_sr=50
    dump_features(dict_path_wavs,ssl,ssl_layer,device,output_dir)
