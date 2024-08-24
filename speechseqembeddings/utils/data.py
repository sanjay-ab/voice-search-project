import sys
import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch import nn
import augment
from torch.utils.data import Dataset
import fairseq
from fairseq.models.wav2vec import Wav2Vec2Model
import time
import tqdm

class Transform(nn.Module):

    """
    Module augmenting an input waveform with a specified transform.
    The augmentation performed is random, governed by a single parameter
    evolving in a preset range.
    """
    def __init__(self, sample_rate,slowest=0.8,fastest=1.5):
        super(Transform, self).__init__()

        """
        Parameters:
            - sample_rate (int): the sample rate of the audio files used.
        """
        self.config_file='Code/librispeech_concat/random_pitch_profile.py' 
        self.sample_rate = sample_rate
        self.slowest=slowest
        self.fastest=fastest
        self.src_info = {"rate": sample_rate}
        self.target_info = {"channels": 1, "length": 0, "rate": sample_rate}

    def forward(self, wave_signal, effect_name,do_tempo=True,pitch_aug=None):

        """
        Apply the transform specified in the effect_name to the
        input wave_signal.

        Parameters:
            - wave_signal (torch.tensor): the input waveform.
            - effect_name (str): name of the effect to apply.
        Returns:
            - wave_transfo (torch.tensor): the transformed waveform.
        """
        
        if len(effect_name)==0:
            return wave_signal,1
        # Initialize the effect
        effect = augment.EffectChain()
        # Add the effect
        if "noise" in effect_name:
            # add a gaussian noise
            if wave_signal is not None:
                param = np.random.randint(20, 30)
                noise_generator = lambda: torch.zeros_like(wave_signal).uniform_()
                effect.additive_noise(noise_generator, snr=param)
        if "tempo" in effect_name and do_tempo:
            # speed up or slow down the signal without changing its pitch
            tempo_param = self.slowest + (self.fastest - self.slowest) * np.random.random()
            effect.tempo("-s", tempo_param)
        else:
            tempo_param=1
        if "pitch" in effect_name:

            if pitch_aug is None:
                pitch_aug = np.random.randint(-300, 300)
            #if np.random.randint(0,2)==1: 
            #    param = np.random.randint(-600, -200)
            #else: 
            #    param = np.random.randint(200, 600)
            
            effect.pitch(pitch_aug).rate(self.sample_rate)
        if "echo" in effect_name:
            # add an echo to the signal
            param = np.random.randint(5, 50)
            effect.echo(0.8, 0.88, param, 0.4)
        if "bass" in effect_name:
            # filter or boost the bass frequencies
            # "bass_boost" or "bass_trim"
            if np.random.randint(0,2)==1: 
                param = np.random.randint(15, 30)
            else:
                param = np.random.randint(-50, -20)
            effect.bass(param)
        if "treble" in effect_name:
            # filter or boost the upper frequencies
            # "treble_boost" or "treble_trim"
            if np.random.randint(0,2)==1: 
                param = np.random.randint(10, 30)
            else:
                param = np.random.randint(-40, -20)
            effect.treble(param)
        if "reverb" in effect_name:
            # add reverb
            param = np.random.randint(0, 100)
            #param = np.random.randint(50, 100)
            effect.reverb(50, 50, param).channels()

        # Apply the effect
        wave_transfo=None
        if wave_signal is not None:
            wave_transfo = effect.apply(
                wave_signal, src_info=self.src_info, target_info=self.target_info)
            # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
            # and the effect chain includes eg `pitch`
            if torch.isnan(wave_transfo).any() or torch.isinf(wave_transfo).any():
                print("nan found")
                wave_transfo=wave_signal
        assert tempo_param>0,tempo_param

        return wave_transfo,tempo_param,pitch_aug


def load_precomputed_features(path_feat_folder,spks_to_load=None):
    data_dict = dict()
    for file in tqdm.tqdm(path_feat_folder):
        spk=file.split('/')[-1].split('.')[0]
        if spks_to_load is not None and spk not in spks_to_load:
            continue
        feat = np.load(file,allow_pickle=True)
        for name in feat:
            tmp=feat[name].ravel()[0]
            data_dict[name]={'times':None,'features':None} 
            data_dict[name]['times']= tmp['times']
            data_dict[name]['features']= tmp['features']
    print('nb of features files',len(data_dict))
    return data_dict

def get_features(npz_dict,start,end,max_nb_frames,framerate,fname=None,check=True):
    if 'times' in npz_dict[fname]:
        features=npz_dict[fname]['features']
        times=npz_dict[fname]['times']
    else:
        feat = np.load(npz_dict[fname],allow_pickle=True)
        tmp=feat[fname].ravel()[0]
        features=tmp['features']
        times=tmp['times']
    #npz_dict[name]['features']= tmp['features']
    t = np.where(np.logical_and(times >= start, times <= end))[0]
    if check:
        assert int(np.abs(len(t)-(end-start)*framerate))<=2,(len(t),end-start,start,end,fname)
    else:
        assert len(t)>0,(len(t),end-start,start,end,fname)
    features=torch.from_numpy(features[t, :])
    assert features.size(0)<=max_nb_frames, (features.size(),max_nb_frames,start,end)
    feat=torch.zeros((max_nb_frames,features.size(1)))
    feat[:features.shape[0],:]=features
    feat=feat.unsqueeze(0)
    return feat

def load_pairs(path_items):
    with open(path_items, "r") as file:
        file_items = file.readlines()
    pairs={}
    count=0
    batch_count=0
    batches={0:[]}
    print('loading pairs and removing same file pairs')
    batch_ind=0
    store_frames=True
    for pair in tqdm.tqdm(file_items):
        tmp=pair.split(' ')
        if len(tmp)==10:
            fid1,spk1,s1,e1,fid2,spk2,s2,e2,sim,batch_ind=tmp
            batch_ind=int(batch_ind)
            store_frames=False
            if float(sim)<0.0085:
                continue
            #if np.random.randint(2)==0:
            #    continue
        elif len(tmp)==9:
            fid1,spk1,s1,e1,fid2,spk2,s2,e2,sim=tmp
        else:
            fid1,s1,e1,fid2,s2,e2,sim=tmp
            spk1,book1,_=fid1.split('-')
            spk2,book2,_=fid2.split('-')
            fid1+='.flac'
            fid2+='.flac'
        s1,e1,s2,e2=float(s1),float(e1),float(s2),float(e2)
        if fid1==fid2: 
            # removing same file pairs
            continue
        #if fid1==fid2:
        #    # if same file, then no overlap is tolerated
        #    maxs=max(s1,s2)
        #    mine=min(e1,e2)
        #    if maxs<=mine:
        #        continue
        fid1=os.path.basename(fid1).split('.')[0]
        fid2=os.path.basename(fid2).split('.')[0]
        if batch_ind not in pairs:
            pairs[batch_ind]={'pairs':[],'spks':set()}
        pairs[batch_ind]['pairs'].append([fid1,spk1,float(s1),float(e1),fid2,spk2,float(s2),float(e2)])
        pairs[batch_ind]['spks'].add(spk1)
        pairs[batch_ind]['spks'].add(spk2)
            
        count+=1
    print('nb of pairs',count)
    print('STORE FRAMES',store_frames) 
    return pairs,store_frames

def create_dict_from_item(path_item,nb_vads=2,min_vad_len=1,path_wavs=None,mode=None):
    with open(path_item, "r") as file:
        file_item = file.readlines()
    # Fill the dictionnary
    dict_labels_to_seg = {}
    count=0
    if mode=='vad_aug':
        ind=0
    else:    
        prev_item=file_item[0].rstrip().split(' ')
        prev_label=prev_item[-1]
        ind=1 # skipping first pair

    for line in file_item[ind:]:
        if mode=='vad_aug':
            path,spk,vs,ve=line.rstrip().split(' ')[:4]
            if 'cp1a' in spk or 'cp1b' in spk:
                spk='pelucchi'
            if 'LangA' in spk or 'LangB' in spk:
                spk='pelucchi'
            fid=os.path.basename(path).split('.')[0]
            path=os.path.join(path_wavs,path)
            assert os.path.isfile(path),(path,path_wavs)  
            label=spk
            t_beg, t_end = float(vs), float(ve)
            if t_end-t_beg<min_vad_len:
                continue
        else:
            item=line.rstrip().split(' ')
            label=item[-1]
            if prev_label==label:
                prev_item=item
                continue
            path,spk,vs,ve,s,e,label=prev_item
            prev_item=item
            prev_label=item[-1]
            #if 'B,IY1' in label:
            #    continue
            #if 'K,AA1' in label:
            #    continue
            #if 'M,EY1' in label:
            #    continue
            #if 'F,UW1' in label:
            #    continue
            
            t_beg, t_end = float(s), float(e)
            fid=os.path.basename(path).split('.')[0]
            path=spk
        count+=1
        if label not in dict_labels_to_seg:
            dict_labels_to_seg[label] = []
        dict_labels_to_seg[label].append([path, fid, t_beg, t_end])  
    print('nb of segments loaded:',count)
    list_labels=[]
    distribution=[]
    nb_items=0
    # computing the distribution of vads per wav
    for key in dict_labels_to_seg:
        count=len(dict_labels_to_seg[key])
        if count>=nb_vads:
            list_labels.append(key)
            distribution.append(count)
            nb_items+=count
    distribution=np.array(distribution, dtype='float')
    dsum=np.sum(distribution) 
    distribution=distribution/dsum
    return dict_labels_to_seg,np.array(list_labels),distribution, nb_items

def create_mapping_labels_to_id(path_items, label_position):
    """
    Create a dict mapping the string labels of type "phn1,phn2,..." to their integer id.

    Parameters:
        - path_items (str): path to the text file
        - label_position (int): the position of the label in the lines of the text file
    Returns:
        - mapp_label_to_id (dict): of type {'label': id}
    """

    # load the list of lines
    list_lines = load_items(path_items)
    # retrieve all the labels in the file

    return mapp_label_to_id


class UnsupDataset(Dataset):

    def __init__(
        self,
        path_item,
        gpu_id=None,
        max_nb_frames=100,
        ssl_path=None,
        ssl_layer=None,
        path_wavs=None):

        # fixed parameters
        self.mode='vad_aug'
        self.path_wavs=path_wavs
        self.gpu_id=gpu_id
        min_vad_len=1.2 #in seconds
        self.max_nb_vads=20 #nb of vads to sample from one speaker
        # should be 20
        print('minimal nb of vads required per speaker',self.max_nb_vads)

        self.speech_per_batch=20 # total speech signal for each batch, in seconds
        self.batch_size=250
        self.sr = 16000
        self.feat_sr=50
        self.slowest_tempo=0.5
        self.fastest_tempo=1.8
        self.offset=int(self.sr*0.16) # around 36 segments per secodns
        self.max_nb_frames=int(0.960*self.feat_sr/self.slowest_tempo)+2
        self.constant_effects=['tempo']
        
        # load the items depending on whether they are pairs or not
        print("Loading training vads...")
        self.items,self.list_labels,self.distribution,self.nb_items = create_dict_from_item(path_item,nb_vads=self.max_nb_vads,min_vad_len=min_vad_len,path_wavs=self.path_wavs,mode=self.mode)
        self.dict_features={}
        for spk in tqdm.tqdm(self.items):
            for item in self.items[spk]:
                file,fname=item[:2]    
                if fname in self.dict_features:
                    continue
                w, s = torchaudio.load(file)
                if s!=16000:
                    resampler=T.Resample(s,16000,dtype=w.dtype)
                    w=resampler(w)
                    s=16000
                self.dict_features[fname] = w
        # loading SSL model
        ssl, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_path])
        self.ssl=ssl[0].float().eval().to(self.gpu_id)
        self.ssl_layer=ssl_layer 
        
        self.transform_fn = Transform(self.sr,slowest=self.slowest_tempo,fastest=self.fastest_tempo)
        print('nb items for training',self.nb_items)

    def __len__(self):
        return self.nb_items 

    def __getitem__(self, idx):
        
        # get wav from a random vad, then augment it 
        # and turn in into speech features
        #id_seg = np.random.randint(0,len(self.items))
        #name, t_beg, t_end = self.items[id_seg]
        all_feat=[]
        all_feat_aug=[]
        data=[]
        spks_and_vads=[]
        #choosing one speaker and some vads from that spk
        label_spkid = np.random.choice(len(self.list_labels), 1,p=self.distribution)
        label=self.list_labels[label_spkid[0]]
        vad_indices = np.random.choice(len(self.items[label]), self.max_nb_vads,replace=False)
        for i in range(self.max_nb_vads):
            spks_and_vads.append([label,vad_indices[i]])
        label_id=None
        wav_augmented=None
        feat_augmented=None
        tempo_aug2=None
        k=0
        total_vad_len=0
        while k<self.max_nb_vads:
            label,vad_index=spks_and_vads[k]
            _, name, t_beg, t_end = self.items[label][vad_index]
            n=(t_end-t_beg)*self.sr/self.offset
            k+=1
            assert t_end*self.sr<=len(self.dict_features[name][0])+160,(t_end*self.sr,self.items[label][vad_index],len(self.dict_features[name][0]))
            wav=self.dict_features[name][0][int(t_beg*self.sr):int(t_end*self.sr)] 
            vad_info=(name,t_beg,t_end)
            len_wav_original=int(wav.size(0))
            wav_augmented,tempo_aug2,pitch_aug=self.transform_fn(wav,self.constant_effects) 
            wav,tempo_aug1,_=self.transform_fn(wav,self.constant_effects) 
            #nb of frames of each shift along the first augmentation
            #of the vad
            wav=torch.reshape(wav,(1,-1)).to(self.gpu_id)
            wav_augmented=torch.reshape(wav_augmented,(1,-1)).to(self.gpu_id)
            feat=self.ssl(wav,padding_mask=None,mask=False,features_only=True,layer=self.ssl_layer)['x'].detach()
            feat_augmented=self.ssl(wav_augmented,padding_mask=None,mask=False,features_only=True,layer=self.ssl_layer)['x'].detach()
            #recomputing some values because w2v2 sampling
            #rate is unpredictable
            feat_sr=feat.size(1)/(wav.size(1)/self.sr)
            hz2fr=feat_sr/self.sr # from Hz to frames
            
            all_feat.append(feat)
            all_feat_aug.append(feat_augmented)
            data.append([len_wav_original,tempo_aug1,tempo_aug2,self.offset,self.max_nb_frames,hz2fr,self.batch_size,self.gpu_id,vad_info])
            total_vad_len+=t_end-t_beg
            if total_vad_len>self.speech_per_batch:
                break

        return all_feat, all_feat_aug,data

class GoldDataset(Dataset):
    """
    The Gold Dataset for the Buckeye.
    """

    def __init__(
        self,
        path_features,
        path_items,
        mode,
        max_nb_frames=None,
        preloaded_frames=None,
    ):
        self.mode=mode
        self.sr = 16000
        self.max_nb_frames = max_nb_frames
        self.framerate=50
        self.use_negatives=False
        if self.use_negatives:
            self.max_nb_frames=150
        self.path_features=path_features 
        if self.mode=='pairs':
            self.items,self.store_frames = load_pairs(path_items)
            self.batch_ind=0
            self.nb_batches=len(self.items)
        else:
            self.store_frames=True
            self.items,self.list_labels,self.distribution ,_= create_dict_from_item(path_items)
       
        if preloaded_frames is not None:
            print("Using preloaded frames")
            self.dict_features = preloaded_frames
        elif self.store_frames:
            print("Loading frames...")
            self.dict_features = load_precomputed_features(path_features)
        else:
            print("Loading frames... batch",self.batch_ind)
            spks_to_load=self.items[self.batch_ind]['spks']
            self.dict_features = load_precomputed_features(self.path_features,spks_to_load)
            print(len(self.items[self.batch_ind]['pairs']),len(self.dict_features)) 
        self.epoch_has_started=False
    
    def __len__(self):
        if self.mode=='pairs':
            if not self.store_frames and self.epoch_has_started:
                self.epoch_has_started=False
                self.batch_ind=(self.batch_ind+1)%self.nb_batches
                spks_to_load=self.items[self.batch_ind]['spks']
                print("Loading frames... batch",self.batch_ind)
                self.dict_features = load_precomputed_features(self.path_features,spks_to_load)
            
            return len(self.items[self.batch_ind]['pairs'])
        else:
            return len(self.items)

    def __getitem__(self, idx):
        self.epoch_has_started=True
        if self.mode=='pairs':
            # read the line from the list of items
            id_label = np.random.randint(0,len(self.items[self.batch_ind]['pairs']))
            name_1, _ ,t_beg_1,t_end_1, name_2, _ , t_beg_2,t_end_2 = self.items[self.batch_ind]['pairs'][id_label]
        else:
            # sample a label in the list according
            # to the distribution of ngrams
            label_id = np.random.choice(len(self.list_labels), 1,p=self.distribution)
            label=self.list_labels[label_id][0]
            # take a random gold pair from that label
            id_seg_1, id_seg_2 = np.random.choice(len(self.items[label]), 2, replace=False)
            _, name_1, t_beg_1, t_end_1 = self.items[label][id_seg_1]
            _, name_2, t_beg_2, t_end_2 = self.items[label][id_seg_2]
        #if self.preload_frames:
        feat_1=get_features(self.dict_features,t_beg_1, t_end_1,self.max_nb_frames,self.framerate,name_1)
        feat_2=get_features(self.dict_features,t_beg_2, t_end_2,self.max_nb_frames,self.framerate,name_2)
 
        if self.use_negatives:
            feat_negs_1=self.get_hard_negatives(t_beg_1,t_end_1,name_1)
            #feat_negs_2=self.get_hard_negatives(t_beg_2,t_end_2,name_2)
            feat_negs_2=torch.zeros(0)
        else:
            feat_negs_1,feat_negs_2=None,None
        assert len(feat_1)!=0,(t_beg_1,t_end_1,name_1)
        assert len(feat_2)!=0,(t_beg_2,t_end_2,name_2)
        return feat_1, feat_2, feat_negs_1, feat_negs_2
    
    def get_hard_negatives(self,fstart,fend,spk):
        hardnegs=[]
        wlen=fend-fstart

        mid=fstart+wlen/2
        stmp=max(0,fstart-wlen/2)
        etmp=fend+wlen/2
        hz=0.02 # 1 frame in s

        lmargin=fstart-stmp
        assert mid>fstart,(start,end,mid)
        assert mid<fend,(start,end,mid) 
        hardnegs.append([fstart,mid])
        hardnegs.append([mid,fend])
        hardnegs.append([stmp,mid])

        if lmargin<=1*hz:
            hardnegs.append([fend+wlen,fend+wlen*2])
            hardnegs.append([mid,fend+wlen])
        else:
            hardnegs.append([stmp,fstart])
            hardnegs.append([stmp,fend])
        
        hardnegs.append([mid,etmp])
        #hardnegs.append([stmp,etmp])
        
        feat_negs=[]
        for s,e in hardnegs:
            feat=get_features(self.dict_features,s,e,self.max_nb_frames,self.framerate,spk,check=False)
            feat_negs.append(feat)
        
        feat_negs=torch.cat(feat_negs,0)
        return feat_negs

class TestDataset(Dataset):

    def __init__(
        self,
        path_item,
        path_features,
        max_nb_frames=None,
        preloaded_frames=None,
        ):
        # load the waveforms in memory
        if preloaded_frames is None:
            print("Loading preextracted test frames...")
            self.dict_features = load_precomputed_features(path_features)
        else:
            print("Using preloaded frames")
            self.dict_features = preloaded_frames
        self.items=[]
        label_ids={}
        with open(path_item) as buf:
            for line in buf:
                path,spk,_,_,s,e,label=line.rstrip().split(' ')
                if label not in label_ids:
                    label_ids[label]=len(label_ids)+1
                fid=os.path.basename(path).split('.')[0]
                self.items.append((fid,spk,float(s),float(e),label_ids[label]))
        print('nb of items for testing',len(self.items))
        self.sr = 16000
        self.max_nb_frames = max_nb_frames 
        assert self.max_nb_frames is not None
        self.framerate=50

    def __len__(self):

        return len(self.items)

    def __getitem__(self, idx):
        # read the line from the list of items
        name,spk,t_beg,t_end,id_label = self.items[idx]
        feat=get_features(self.dict_features,t_beg, t_end,self.max_nb_frames,self.framerate,name)
        assert len(feat)!=0,(name,t_beg,t_end) 
        return feat, id_label
    

def get_collate_fn(mode):
    """
    Define the different custom collate_fn functions and returns
    the appropriate one given dataset_name.

    Parameters:
        - mode (str): in ['train' or 'dev']
    Returns:
        - collate_fn (function): a collate_fn function
    """
    
    
    def slice_and_pad(start_ind,end_ind,features,max_nb_frames,gpu_id,dropout):
        assert start_ind<end_ind
        assert end_ind-features.size(0)<=1,(end_ind,features.size())
        assert end_ind-start_ind<=max_nb_frames,[start_ind,end_ind,max_nb_frames]
        pad=torch.zeros((max_nb_frames,features.size(1))).to(gpu_id)
        frames=features[start_ind:end_ind]
        pad[:frames.size(0),:]=frames
        return pad.unsqueeze(0)

    def collate_fn_unsup(list_elements):
        #in unsupervised mode, get_item is called only once
        list_vads=list_elements[0]
        _,_,_,offset,max_nb_frames,hz2fr,batch_size,gpu_id,_=list_vads[2][0]
        batch_size=int(batch_size)
        offset=int(offset)
        max_nb_frames=int(max_nb_frames)
        first_context=[]
        second_context=[]
        labels=[]
        ind=0
        dropout=False
        nb_vads=len(list_vads[0])
        #looping over all vads in batch
        for i in range(nb_vads):
            feat = list_vads[0][i][0]
            feat_augmented = list_vads[1][i][0]
            init_len,tempo_aug1,tempo_aug2,_,_,_,_,_,vad_info=list_vads[2][i]
            init_len=int(init_len)
            c=0
            for start in range(0,init_len,offset):
                for end in range(start+offset,init_len,offset):
                    if end>=init_len-160:
                        break 
                    if end-start>0.96*16000:
                        continue
                    # first element
                    start_ind=int(np.around((start/tempo_aug1)*hz2fr,0))
                    end_ind=int(np.around((end/tempo_aug1)*hz2fr,0)) 
                    assert len(feat)+1>=end_ind,(feat.size(),start,end,start_ind,end_ind,hz2fr,init_len,tempo_aug1)
                    assert start_ind<end_ind,(feat.size(),start_ind,end_ind,hz2fr,init_len,offset,start,end)
                    frames=slice_and_pad(start_ind,end_ind,feat,max_nb_frames,gpu_id,dropout)
                    start_ind_aug=int(np.around((start/tempo_aug2)*hz2fr,0))
                    end_ind_aug=int(np.around((end/tempo_aug2)*hz2fr,0))
                    frames_aug=slice_and_pad(start_ind_aug,end_ind_aug,feat_augmented,max_nb_frames,gpu_id,dropout)
                    first_context.append(frames)
                    second_context.append(frames_aug)
                    labels.append(ind) 
                    ind+=1
        
        first_context = torch.cat(first_context,dim=0)
        second_context = torch.cat(second_context,dim=0)
        labels = torch.tensor(labels)
        # getting only batch_size number of pairs
        if batch_size<len(labels):
        #assert len(labels)>=batch_size,(len(labels))
            indices=np.random.choice(len(labels),batch_size,replace=False) 
            indices=torch.tensor(indices)
            first_context=first_context[indices]
            second_context=second_context[indices]
            labels=labels[indices]
         
        # concatenate the parallele pairs in a same tensor
        # the output is thus of shape (2 x batch_size, *, *)
        cat_context = torch.cat([first_context, second_context], dim=0)
        cat_labels = torch.cat([labels, labels],dim=0)
        return cat_context, cat_labels, None, None
    
    def collate_fn_gold(list_elements):
        n = len(list_elements)
        # gather the elements
        first_context = torch.cat([list_elements[k][0] for k in range(n)])
        second_context = torch.cat([list_elements[k][1] for k in range(n)])
        if list_elements[0][2] is not None:
            negs = torch.cat([list_elements[k][2] for k in range(n)])
            negs2 = torch.cat([list_elements[k][3] for k in range(n)])
        else:
            negs,negs2 = None,None
        cat_context = torch.cat([first_context, second_context], dim=0)
        tmp_labels=torch.arange(len(first_context))
        cat_labels = torch.cat([tmp_labels, tmp_labels])
        return cat_context, cat_labels, negs, negs2
        
    def collate_fn_test(list_elements):
        n = len(list_elements)
        context = torch.cat([list_elements[k][0] for k in range(n)])
        labels = [list_elements[k][1] for k in range(n)]
        return context, labels

    if mode == "train":
        return collate_fn_gold
    elif mode=='train_unsup':
        return collate_fn_unsup
    elif mode == "test":
        return collate_fn_test
    else:
        print('wrong collate mode',mode)
        sys.exit()
