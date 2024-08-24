import argparse
import logging
import math
import numpy as np
import os
import pandas as pd
import sys
import numpy
from itertools import combinations, count
from scipy.special import comb
from tqdm import tqdm
from transcribe_utils import read_gold_intervals, get_intervals, Stream_stats


def frequency_from_timestamps(class_file_path, gold_file_path,outpath,sil_phones):
    gold, transcriptions, ix2symbols, symbol2ix = read_gold_intervals(gold_file_path)

    with open(class_file_path, 'r') as f:
        out=[]
        for line in tqdm(f):
            line=line.strip().split(' ')
            path,spk,vs,ve,on,off=line[:6]
            fname=os.path.basename(path).split('.')[0]
            fon=float(on)
            foff=float(off)
            _, t = get_intervals(fname, fon, foff, gold, transcriptions)
            t=np.array(t)
            # removing silences
            for sil in sil_phones:
                t=np.delete(t, np.where(t == sil)) 
            ngram=','.join(t)
            if len(ngram)==0:
                ngram='NOTRANS'
            tmp=' '.join([path,spk,vs,ve,on,off,ngram])
            out.append(tmp)
            
    with open(outpath,'w') as buf:
        buf.write('\n'.join(out))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('class_file_path', metavar='class_file_path',
            help='Path to file containing segments')
    parser.add_argument('gold_file_path', metavar='gold_file_path',
            help='Path fo file containing gold phones')
    parser.add_argument('outpath', \
            help='Filename for outputting pairs with ned')

    args = parser.parse_args()
    # segments should be speech without long silences
    # at least no more than 0.2 s
    sil_phones=['SIL','sil','sp'] # silences annotations
    frequency_from_timestamps(
        args.class_file_path, args.gold_file_path,args.outpath,sil_phones)

