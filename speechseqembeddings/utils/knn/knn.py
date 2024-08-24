from __future__ import division
from __future__ import print_function
import datetime
import faiss
import numpy as np
import os
import sys
import time
import random

from collections import defaultdict, OrderedDict
from tqdm import tqdm

# Local packages imports
from nms import non_max_suppression_fast

class KNN():
    def __init__(self, embedding_dimension=None,
                 embeddings_dir=None,
                 logger=None,faiss_save_dir=None,utt_to_spk=None,k=10):
        self.embeddings_dir = embeddings_dir
        self.logger = logger
        self.faiss_save_dir = faiss_save_dir
        self.k = k
        self.nms_overlap_threshold = 0.4
        self.self_overlap_threshold = 0.05
        self.embedding_dimension = embedding_dimension

        self.filename_to_file_index = OrderedDict()
        for i in range(len(utt_to_spk)):
            embedding_filename,spk=utt_to_spk[i]
            ind=len(self.filename_to_file_index)
            self.filename_to_file_index[embedding_filename]=spk 

    def create_faiss_index(self, embedding_files, chunk_size=1000):
        print("Initiating Faiss Index")
        self.index_flat = faiss.IndexFlatIP(self.embedding_dimension)
        self.index = faiss.IndexIDMap(self.index_flat)

        # Declare variables
        #To stack the embeddings and select the x best ones.
        self.stacked_embeddings = np.zeros((0, self.embedding_dimension), dtype='float32')
        self.embedding_index_to_interval = np.zeros((0, 2), dtype='object')
        self.embedding_index_to_utterance = np.zeros((0, 1), dtype='int')
        self.embedding_index_to_filename=[]

        self.embedding_offset = OrderedDict()
        self.utterance_offset = OrderedDict()

        embedding_offset = 0
        utterance_offset = 0
        should_index = False

        # Iterate through all embedding files
        for embedding_file in tqdm(embedding_files):
            if 'npz' not in embedding_file:
                continue
            embedding_file_path = os.path.join(self.embeddings_dir, embedding_file)
            embedding_file_name = os.path.splitext(os.path.basename(embedding_file))[0]
            embeddings, intervals, utterances =\
                self.load_h5_embedding_file(embedding_file_path)
            n_embeddings, embedding_dimension = embeddings.shape
            should_index = should_index or not all(os.path.exists(
                os.path.join(self.faiss_save_dir, '{}-{}-{}.npy').format(
                    embedding_file_name, self.k, matrix
                )) for matrix in ["D", "I"])
            # Load file information
            
            self.normalize(embeddings)
            
            self.stacked_embeddings = np.concatenate((
                self.stacked_embeddings,
                embeddings
            ))

            # Fill embedding to interval times dictionary
            self.embedding_index_to_interval = np.concatenate((
                self.embedding_index_to_interval,
                intervals
            ))
            
            # Fill embedding to utterance dictionary
            prev_utt_last_emb_index = 0
            
            for utt_index, last_emb_index,utt_name in utterances:
                #if last_emb_index<prev_utt_last_emb_index:
                #    prev_utt_last_emb_index=-1
                utt_index=int(utt_index)
                last_emb_index=int(last_emb_index)
                self.embedding_index_to_utterance = np.append(
                    self.embedding_index_to_utterance,
                    [utt_index] * (last_emb_index - prev_utt_last_emb_index)
                )
                self.embedding_index_to_filename.append(utt_name)
                prev_utt_last_emb_index = last_emb_index
            # Fill offsets
            self.embedding_offset[embedding_file_name] = embedding_offset
            embedding_offset += n_embeddings


        # Fill the index in chunks
        n_stacked_embeddings = int(self.stacked_embeddings.shape[0])
        print('INDEX SIZE',n_stacked_embeddings)
        n_chunks = int(np.ceil(n_stacked_embeddings / chunk_size))
        self.stacked_embeddings_indices = np.arange(n_stacked_embeddings)

        ## Important: converting to float32 since float64 prevents
        ## Faiss from working properly
        self.stacked_embeddings = self.stacked_embeddings.astype('float32')

        if should_index:
            for i in range(n_chunks):
                max_index = min((i+1) * chunk_size, n_stacked_embeddings)
                self.index.add_with_ids(
                    self.stacked_embeddings[i * chunk_size:max_index],
                    self.stacked_embeddings_indices[i * chunk_size:max_index]
                )
        else:
            self.logger.log('Not creating FAISS index.')

    def run_faiss(self, embedding_file_path,pbar=None):
        """
        This function runs the FAISS program
        And returns the D and I matrices that represent:
        * D : of size (n_embeddings * k) : the distance matrix
        between each embedding and its k closst neighbors in the dataset
        * I: Same size, containing the index of the closest neighbors
        """
        s_embeddings, _ , _ =\
            self.load_h5_embedding_file(embedding_file_path)
        # Perform Faiss search of given file
        # in previously indexed files.
        self.logger.updatePbarDescription('Faiss Search', pbar)

        self.normalize(s_embeddings)
        t0 = time.time()

        embedding_file_name = os.path.splitext(os.path.basename(embedding_file_path))[0]

        D_path = os.path.join(self.faiss_save_dir, '{}-{}-D.npy'.format(
            embedding_file_name, self.k))
        I_path = os.path.join(self.faiss_save_dir, '{}-{}-I.npy'.format(
            embedding_file_name, self.k))

        # check if D doesn't already exist
        if os.path.exists(D_path) and os.path.exists(I_path):
            print('loading existing D and I')
            D = np.load(D_path)
            I = np.load(I_path)
        else:
            D, I = self.index.search(s_embeddings, self.k)
            np.save(D_path, D)
            np.save(I_path, I)
            t1 = time.time()

        return D, I

    def search(self, embedding_file_path, pbar=None):
        '''Given an embedding file, assuming that the faiss index
        was previously computed or loaded, compute the k nearest
        neighbours of all terms of this file.'''

        embedding_file_name = os.path.splitext(os.path.basename(embedding_file_path))[0]
        self.logger.log('Searching {}'.format(embedding_file_path))
        D, I = self.run_faiss(embedding_file_path)
        # extract good pairs from FAISS results
        self.logger.log('   Pairs selection {}'.format(embedding_file_name))
        self.original_extract(D, I,embedding_file_name)
    

    def original_extract(self, D, I, embedding_file_name,pbar=None):
        """
        Perform selection over the pairs of embeddings found.
        """
        #tmp_fname='_'.join(embedding_file_name.split('_')[:-1])
        #s_file_index = self.filename_to_file_index[tmp_fname]
        s_file_index = '_'.join(embedding_file_name.split('_')[:-1])  
        
        ## Flatten distance matrix for sorting
        if D.shape[0]==0 :
            return
        apply_mean=False
        if apply_mean:
            print("mean")
            D_mean=np.mean(D,axis=0)
            D=D-D_mean
        D_flattened = D.flatten()
        D_flattened_index = np.arange(len(D_flattened))
        

        self.logger.log(
            '\tNumber {} pairs; min cos similarity: {})'.format(
                len(D_flattened_index),
                np.around(np.min(D_flattened),decimals=2)
            ),
            pbar=pbar
        )

        t1 = time.time()

        filled_utt_pairs = set()
        utt_pair_occurrences = defaultdict(int)
        utt_pair_embedding_pairs = defaultdict(lambda: [])

        #self.logger.updatePbarDescription('Filling utterance pairs', pbar)

        discovered_pairs = []
        self_overlapping_pairs = 0
        remaining=0 
        for distance_index in D_flattened_index :
            #if D_flattened[distance_index]<0.80:
            #    continue
            nth_source_emb = int(distance_index // self.k)
            nth_target_emb = distance_index % self.k

            s_emb_index = self.embedding_offset[embedding_file_name] + nth_source_emb
            t_emb_index = I[nth_source_emb, nth_target_emb]

            # This condition proves useful when dealing with
            # certain Faiss indices (for instance HNSW).
            if t_emb_index == -1:
                continue
            t_utt_index  =self.embedding_index_to_utterance[t_emb_index]
            t_interval   =self.embedding_index_to_interval[t_emb_index]
            s_interval   =self.embedding_index_to_interval[s_emb_index]
            s_filename = self.embedding_index_to_filename[s_emb_index]
            t_filename = self.embedding_index_to_filename[t_emb_index]
            t_file_index =self.embedding_index_to_file_index(t_emb_index)
            t_file_index = '_'.join(t_file_index.split('_')[:-1])  
            if t_filename == s_filename:
                # check self overlapping threshold
                if self.intervals_overlap(s_interval[:2], t_interval[:2]):
                    self_overlapping_pairs += 1
                    continue
            assert s_file_index in s_filename,(s_file_index,s_filename)
            assert t_file_index in t_filename,(t_file_index,t_filename)
            
            info = [
                s_filename, t_filename,
                s_file_index, t_file_index,
                int(s_interval[0]), int(s_interval[1]),
                int(t_interval[0]), int(t_interval[1]),
                D_flattened[distance_index],
            ]
            
            utt_pair_embedding_pairs[(s_emb_index,t_utt_index)].append(info)
            remaining+=1
        print('remaining pairs',remaining,self_overlapping_pairs)

        # Display timing information and log utt info
        t2 = time.time()

        # Perform NMS and prepare data printing
        t3 = time.time()
        length_utts= len(utt_pair_embedding_pairs)
        for key in utt_pair_embedding_pairs :
            # Perform NMS here
            boxes = np.array(utt_pair_embedding_pairs[key])[:, 4:9]
            boxes= boxes.astype(float)
    
            nms_boxes, pick = non_max_suppression_fast(boxes,
                                                       self.nms_overlap_threshold)
            # Add pairs to best_discovered_redux for printing
            for box in np.array(utt_pair_embedding_pairs[key])[pick]:
                discovered_pairs.append(box)
        
        # Display timing information and log utt info
        t4 = time.time()
        self.logger.log(
            '\tNumber of pairs kept after discarding overlap : {}'.format(
                len(discovered_pairs)
            ),
        )
    
        # Write output file
        self.logger.logPairs(discovered_pairs,embedding_file_name)
    # UTIL FUNCTIONS
    def intervals_overlap(self, interval1, interval2):
        """
        Returns true if interval1 and interval2 are overlapping.
        Both intervals must be tuples in the form (start, end).

        The function assumes they are from the same file

        Condition used :
        - Lengths difference must be less than the same_length_threshold
        - Overlap must be less than the
        """
        start1, end1 = interval1
        start2, end2 = interval2

        if start2 >= end1 or start1 >= end2:
            return False

        #len_interval1 = max(end1 - start1, 1)
        len_interval1 = end1 - start1
        #len_interval2 = max(end2 - start2, 1)
        len_interval2 = end2 - start2

        # Overlap condition
        o_start = max(start1, start2)
        o_end = min(end1, end2)
        overlap_condition1 = (float(o_end - o_start) / len_interval1) >= self.self_overlap_threshold
        overlap_condition2 = (float(o_end - o_start) / len_interval2) >= self.self_overlap_threshold
        #overlap1 = (float(o_end - o_start) / len_interval1)
        #overlap2 = (float(o_end - o_start) / len_interval2) 
        #if overlap1!=1 and overlap2!=1:
        #    print(interval1,interval2,overlap1,overlap2)                
        return overlap_condition1 or overlap_condition2

    def box_overlap(self, box1, box2):
        """
        Returns true if box1 and box2 are overlapping.
        Both boxes must be tuples in the form (interval1, interval2)

        Condition used :
        - Lengths difference must be less than the same_length_threshold
        - Overlap must be less than the
        """
        source1, target1 = box1
        source2, target2 = box2

        return self.intervals_overlap(source1, source2) and \
            self.intervals_overlap(target1, target2)

    def embedding_index_to_file_index(self, embedding_index):
        found_filename = None
        for filename in self.embedding_offset:
            if self.embedding_offset[filename] <= embedding_index:
                found_filename = filename
            else:
                break
        #found_filename='_'.join(found_filename.split('_')[:-1])
        #return self.filename_to_file_index[found_filename]
        return found_filename

    def load_h5_embedding_file(self, feat):
        f=np.load(feat)
        features = f['features'][:]
        intervals = f['indexes'][:]
        utterances = f['utterances'][:]
        return features.astype('float32'), intervals, utterances

    def normalize(self, data):
        '''Normalize numpy array data in-place.'''
        # TODO: does this op double the size of data/features in memory?
        norm = np.sqrt(np.sum(data**2.0, axis=1))
        data /= norm[:, None]

