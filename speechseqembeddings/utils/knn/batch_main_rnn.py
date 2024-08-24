#!/usr/bin/env python
# Author: the CoML Team
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import argparse
import yaml
import sys,os
import torch
from pathlib2 import Path
import json
from tqdm import tqdm
from knn import KNN
from logger import Logger
from embed_segments import read_randomsegments,  process_file
from utils.models import SSEmodel
from multiprocessing import Pool
from sklearn.decomposition import PCA

class Pipeline():
    '''This class is used to store model paramaters and
    the various steps of the pipeline as methods.'''

    def __init__(self, verbose=False):

        # Read CLI arguments
        parser = argparse.ArgumentParser()

        parser.add_argument('exp_dir', metavar='exp_dir',
                help='experience directory')
        parser.add_argument('feature_dir', metavar='feature dir',
                help='feature directory')
        parser.add_argument("extraction_cores" , metavar = "extraction_cores",
                help= "number of cores for feature extraction")
        parser.add_argument('pretrained_model', metavar='pretrained_model',
                help='pretrained model path')
        parser.add_argument('random_segments', metavar='training segments',
                help='...')
        parser.add_argument('knn_k', type=int,default=100,
                help='...')
        # Number of cores should be 1 if the features are already extracted, it would lead to faster computation
        parser.add_argument('-v', '--verbose', action='store_true',
                help='Should print information and debug messages')

        args = parser.parse_args()
        self.apply_pca=True
        self.knn_k=args.knn_k
        if self.apply_pca:
            self.embeddings_dim=128
        else:
            self.embeddings_dim=512
        self.framerate=50 # Wav2vec2.0 frames per seconds
        self.max_nb_frames=100 # longeest segment should be 2 seconds max
        self.max_duration=100*(1/self.framerate)
        self.extraction_cores = int(args.extraction_cores)
        self.verbose = args.verbose
        self.pretrained_model=args.pretrained_model
        self.random_segments=args.random_segments
        
        exp_dir = args.exp_dir
        self.embeddings_dir=exp_dir+"/embeddings/"
        self.faiss_save_dir=exp_dir+"/faiss/"
        self.output_dir=exp_dir
        self.feature_file=args.feature_dir

        # Create output directory if need be
        Path(self.embeddings_dir).mkdir(exist_ok=True,parents=True)
        Path(self.faiss_save_dir).mkdir(exist_ok=True,parents=True)
        Path(self.output_dir).mkdir(exist_ok=True,parents=True)

        # Handdle logging
        self.logger = Logger(
            logDirPath=self.output_dir,
            verbose=verbose,
            exp_name=''
        )
        self.verbose = verbose

        self.logger.log('Saving terms to: {}'.format(self.embeddings_dir))

        
        path_model=os.path.join(self.pretrained_model,'model_checkpoint.tar')
        path_config=os.path.join(self.pretrained_model,'params.json')
        with open(path_config) as buf:
            config=json.load(buf) 
        model = SSEmodel(
            input_size=768,
            n_conv_layers=config["n_conv_layers"],
            transformer_dim=config["transformer_dim"],
            n_heads=config["n_heads"],
            n_transformer_layers=config["n_transformer_layers"],
            )
        self.embedding_size = config['transformer_dim']
        model.cuda()
        state_dict=torch.load(path_model,map_location='cuda')
            
        model.load_state_dict(state_dict['model_state_dict'])
        self.model=model.eval()
        print("TRAINED MODEL LOADED, dimension",self.embedding_size)

    def downsample(self):
        '''Parses raw audio files using a subsampling method
        described in the repo.'''
        # Create output directory if necessary

        self.logger.log('Starting segment candidate generation...')
        randomseg,self.utt2spk = read_randomsegments(self.random_segments,self.max_duration)
        emb_list_len=len(list(randomseg.keys()))
        if emb_list_len==len(os.listdir(self.embeddings_dir)):
            print("downsampling is already done!")
            return
        print("files to embed",emb_list_len)
        items=None

        all_feat_dict={}
        labels_dict={}
        decode=False
        self.logger.log('Creating terms...')
        embeddings_dict={}
        concat_embeddings=[]
        for key in tqdm(randomseg):
            tmp='_'.join(key.split('_')[:-1])
            npz_data=np.load(os.path.join(self.feature_file,tmp+'.npz'),allow_pickle=True)
            element=(npz_data, self.max_nb_frames,self.framerate, randomseg[key], self.model)
            embeddings,timestamps,utterances=process_file(element) 
            if len(concat_embeddings)==0:
                concat_embeddings=embeddings
            else:
                concat_embeddings=np.concatenate((concat_embeddings,embeddings))
            embeddings_dict[key]=(embeddings,timestamps,utterances)
        print('embedded segments',concat_embeddings.shape)
        if self.apply_pca:
            print('computing pca')
            pca=PCA(whiten=False,n_components=pipeline.embeddings_dim)
            max_pca_embs=min(1000000,concat_embeddings.shape[0])
            indices=np.arange(concat_embeddings.shape[0])
            np.random.shuffle(indices)
            indices=indices[:max_pca_embs]
            pca=pca.fit(concat_embeddings[indices])
            concat_embeddings=None # free memory
        
        for key in tqdm(embeddings_dict): 
            embeddings,timestamps,utterances=embeddings_dict[key]
            if self.apply_pca:
                embeddings=pca.transform(embeddings)
                #embeddings=embeddings[:,:]
                #print(embeddings.shape) 
            out_path = os.path.join(self.embeddings_dir, key)    
            np.savez(out_path,features=embeddings,indexes=timestamps,utterances=utterances)      

    def search(self,knn,file_path) : 
        return knn.search(file_path)
    
def run_KNN(pipeline):
    return

if __name__ == "__main__":
    pipeline = Pipeline(verbose=True)
    pipeline.downsample()
    file_list = os.listdir(pipeline.embeddings_dir)
    batch=[]
    print("cpus",pipeline.extraction_cores)
    batch_size=100 # each embedding file contains 100k segment embeddings, so 10M embeddings are used for each kNN search
    def search(path_file):
        knn.search(path_file)
    print(file_list)
    for i in range(len(file_list)):
        element=file_list[i]
        batch.append(element)
        if len(batch)>=batch_size or i==len(file_list)-1:
            knn = KNN(
                    embedding_dimension=pipeline.embeddings_dim,
                    embeddings_dir=pipeline.embeddings_dir,
                    logger=pipeline.logger,
                    faiss_save_dir=pipeline.faiss_save_dir,
                    utt_to_spk=pipeline.utt2spk,
                    k=pipeline.knn_k)
            knn.create_faiss_index(embedding_files=batch)
            print("#######NEW KNN INDEX")
            print(batch)
            batch=[os.path.join(pipeline.embeddings_dir,f) for f in batch]
            #for f in batch:
            #    knn.search(f)
            pool = Pool(processes = pipeline.extraction_cores)
            pool.map(search, batch)
            pool.close()
            batch=[]
    assert len(batch)==0,len(batch)
    pipeline.logger.logDoneFile()
